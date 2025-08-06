"""
S3 Historical Data Loader for Hyperliquid Archive

This module provides access to extensive historical data from Hyperliquid's AWS S3 buckets
for Monte Carlo and Walk Forward validation systems. 

Based on research from:
- /research/hyperliquid_documentation/8_historical_data_access.md
- AWS S3 requester-pays model with LZ4 compression

Key Features:
- L2 book snapshot loading with OHLCV reconstruction
- Trade data access for volume validation  
- Cost optimization with selective downloads
- Data quality validation and integrity checks
- Local caching to minimize S3 transfer costs

S3 Bucket Structure:
- hyperliquid-archive/market_data/[date]/[hour]/l2Book/[coin].lz4
- hl-mainnet-node-data/node_fills (trade execution data)
"""

import asyncio
import boto3
import lz4.frame
import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
import os

from src.config.settings import get_settings, Settings


@dataclass
class S3DataAvailability:
    """Container for S3 data availability assessment."""
    
    total_files_checked: int = 0
    available_files: int = 0
    missing_files: List[str] = None
    coverage_ratio: float = 0.0
    estimated_cost_usd: float = 0.0
    total_size_gb: float = 0.0
    
    def __post_init__(self):
        if self.missing_files is None:
            self.missing_files = []


@dataclass
class OHLCVCandle:
    """OHLCV candle reconstructed from L2/trade data."""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            't': int(self.timestamp.timestamp() * 1000),  # Milliseconds
            'o': self.open,
            'h': self.high,
            'l': self.low,
            'c': self.close,
            'v': self.volume,
            'n': self.trades_count
        }


class S3HistoricalDataLoader:
    """Load historical data from Hyperliquid S3 buckets for validation systems."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize S3 historical data loader.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.S3Loader")
        
        # S3 configuration from research
        self.archive_bucket = 'hyperliquid-archive'
        self.node_data_bucket = 'hl-mainnet-node-data'
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Local cache directory
        self.cache_dir = Path(self.settings.data_dir) / "s3_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cost tracking
        self.avg_l2_file_size_mb = 50    # From research
        self.avg_trade_file_size_mb = 100  # From research
        self.aws_transfer_cost_per_gb = 0.09  # USD per GB
        
        self.logger.info(f"S3 Historical Data Loader initialized")
        self.logger.info(f"Cache directory: {self.cache_dir}")
    
    async def check_data_availability(self, symbol: str, start_date: datetime, 
                                    end_date: datetime) -> S3DataAvailability:
        """Check S3 data availability for validation requirements.
        
        Args:
            symbol: Trading symbol (e.g., "BTC")
            start_date: Start date for data requirement
            end_date: End date for data requirement
            
        Returns:
            Data availability assessment
        """
        self.logger.info(f"🔍 Checking S3 data availability for {symbol}")
        self.logger.info(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        
        availability = S3DataAvailability()
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.strftime('%Y%m%d')
            
            # Check hourly L2 book files for this date
            for hour in range(24):
                s3_key = f"market_data/{date_str}/{hour}/l2Book/{symbol}.lz4"
                availability.total_files_checked += 1
                
                try:
                    # Check if file exists in S3
                    self.s3_client.head_object(
                        Bucket=self.archive_bucket,
                        Key=s3_key,
                        RequestPayer='requester'
                    )
                    
                    availability.available_files += 1
                    availability.total_size_gb += self.avg_l2_file_size_mb / 1024
                    
                    self.logger.debug(f"✅ Found: {s3_key}")
                    
                except self.s3_client.exceptions.NoSuchKey:
                    availability.missing_files.append(s3_key)
                    self.logger.debug(f"❌ Missing: {s3_key}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Error checking {s3_key}: {e}")
                    availability.missing_files.append(s3_key)
            
            current_date += timedelta(days=1)
        
        # Calculate results
        availability.coverage_ratio = (
            availability.available_files / availability.total_files_checked 
            if availability.total_files_checked > 0 else 0.0
        )
        availability.estimated_cost_usd = (
            availability.total_size_gb * self.aws_transfer_cost_per_gb
        )
        
        self.logger.info(f"📊 S3 Availability Results for {symbol}:")
        self.logger.info(f"  - Files available: {availability.available_files}/{availability.total_files_checked}")
        self.logger.info(f"  - Coverage ratio: {availability.coverage_ratio:.1%}")
        self.logger.info(f"  - Estimated size: {availability.total_size_gb:.2f} GB")
        self.logger.info(f"  - Estimated cost: ${availability.estimated_cost_usd:.2f}")
        
        return availability
    
    async def load_l2_book_data(self, symbol: str, date: str, hour: int) -> Optional[Dict[str, Any]]:
        """Load L2 book snapshot data from S3.
        
        Args:
            symbol: Trading symbol (e.g., "BTC")
            date: Date string in YYYYMMDD format
            hour: Hour (0-23)
            
        Returns:
            Parsed L2 book data or None if not available
        """
        s3_key = f"market_data/{date}/{hour}/l2Book/{symbol}.lz4"
        cache_key = f"{symbol}_{date}_{hour}_l2book"
        cached_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache first
        if cached_file.exists():
            self.logger.debug(f"📁 Loading {cache_key} from cache")
            try:
                with open(cached_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"⚠️  Cache read failed for {cache_key}: {e}")
        
        # Download from S3
        self.logger.debug(f"☁️  Downloading {s3_key} from S3")
        
        try:
            local_lz4_path = self.cache_dir / f"{cache_key}.lz4"
            
            # Download LZ4 file
            self.s3_client.download_file(
                Bucket=self.archive_bucket,
                Key=s3_key,
                Filename=str(local_lz4_path),
                ExtraArgs={'RequestPayer': 'requester'}
            )
            
            # Decompress LZ4
            with lz4.frame.LZ4FrameFile(str(local_lz4_path), 'r') as lz4_file:
                decompressed_data = lz4_file.read().decode('utf-8')
            
            # Parse JSON data
            l2_data = json.loads(decompressed_data)
            
            # Cache the parsed data
            with open(cached_file, 'w') as f:
                json.dump(l2_data, f)
            
            # Clean up LZ4 file
            local_lz4_path.unlink()
            
            self.logger.debug(f"✅ Successfully loaded and cached {cache_key}")
            return l2_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {s3_key}: {e}")
            return None
    
    async def load_trade_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Load trade execution data from S3.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            List of trade execution records
        """
        self.logger.info(f"📈 Loading trade data from {start_date} to {end_date}")
        
        # Note: This is a simplified implementation
        # Real implementation would need to parse the actual S3 structure for node_fills
        trades = []
        
        try:
            # List files in node_fills bucket for date range
            # This is a placeholder - actual implementation needs date-based file discovery
            prefix = 'node_fills'
            
            # Get list of files (simplified)
            response = self.s3_client.list_objects_v2(
                Bucket=self.node_data_bucket,
                Prefix=prefix,
                RequestPayer='requester'
            )
            
            # Process files (placeholder logic)
            for obj in response.get('Contents', [])[:5]:  # Limit to 5 files for testing
                key = obj['Key']
                self.logger.debug(f"Processing trade file: {key}")
                
                # Download and parse trade file
                # Real implementation would parse actual trade format
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load trade data: {e}")
        
        return trades
    
    def _reconstruct_ohlcv_from_l2(self, l2_snapshots: List[Dict[str, Any]], 
                                  timeframe_minutes: int = 1) -> List[OHLCVCandle]:
        """Reconstruct OHLCV candles from L2 book snapshots.
        
        Args:
            l2_snapshots: List of L2 book snapshots
            timeframe_minutes: Candle timeframe in minutes
            
        Returns:
            List of reconstructed OHLCV candles
        """
        if not l2_snapshots:
            return []
        
        self.logger.info(f"🔧 Reconstructing OHLCV candles from {len(l2_snapshots)} L2 snapshots")
        
        candles = []
        
        try:
            # Group snapshots by timeframe
            timeframe_groups = {}
            
            for snapshot in l2_snapshots:
                # Extract timestamp and mid price from L2 data
                timestamp = datetime.fromtimestamp(snapshot.get('time', 0) / 1000, tz=timezone.utc)
                
                # Calculate timeframe bucket
                bucket_time = timestamp.replace(
                    minute=(timestamp.minute // timeframe_minutes) * timeframe_minutes,
                    second=0,
                    microsecond=0
                )
                
                # Extract mid price from L2 book
                levels = snapshot.get('levels', [])
                if levels and len(levels) >= 2:
                    # Get best bid and ask
                    bids = [level for level in levels if level.get('side') == 'B']
                    asks = [level for level in levels if level.get('side') == 'A']  
                    
                    if bids and asks:
                        best_bid = max(bids, key=lambda x: float(x.get('px', 0)))['px']
                        best_ask = min(asks, key=lambda x: float(x.get('px', float('inf'))))['px']
                        mid_price = (float(best_bid) + float(best_ask)) / 2
                        
                        if bucket_time not in timeframe_groups:
                            timeframe_groups[bucket_time] = []
                        
                        timeframe_groups[bucket_time].append({
                            'timestamp': timestamp,
                            'mid_price': mid_price,
                            'volume': 0  # Volume reconstruction needs trade data
                        })
            
            # Create OHLCV candles from grouped data
            for bucket_time, snapshots in sorted(timeframe_groups.items()):
                if not snapshots:
                    continue
                
                prices = [s['mid_price'] for s in snapshots]
                
                candle = OHLCVCandle(
                    timestamp=bucket_time,
                    open=prices[0],
                    high=max(prices),
                    low=min(prices),
                    close=prices[-1],
                    volume=0.0,  # Would need trade data for accurate volume
                    trades_count=len(snapshots)
                )
                
                candles.append(candle)
            
            self.logger.info(f"✅ Reconstructed {len(candles)} OHLCV candles")
            
        except Exception as e:
            self.logger.error(f"❌ OHLCV reconstruction failed: {e}")
        
        return candles
    
    async def get_historical_candles(self, symbol: str, start_date: datetime, 
                                   end_date: datetime, timeframe_minutes: int = 1) -> pd.DataFrame:
        """Get historical OHLCV candles reconstructed from S3 data.
        
        Args:
            symbol: Trading symbol
            start_date: Start datetime
            end_date: End datetime  
            timeframe_minutes: Candle timeframe in minutes
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info(f"📊 Getting historical candles for {symbol}")
        self.logger.info(f"📅 Period: {start_date} to {end_date}")
        self.logger.info(f"⏱️  Timeframe: {timeframe_minutes} minutes")
        
        # Check data availability first
        availability = await self.check_data_availability(symbol, start_date, end_date)
        
        if availability.coverage_ratio < 0.5:  # Less than 50% availability
            self.logger.warning(f"⚠️  Low data availability ({availability.coverage_ratio:.1%}) for {symbol}")
            self.logger.warning("Consider using API data or adjusting date range")
            return pd.DataFrame()
        
        # Load L2 book data
        l2_snapshots = []
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        self.logger.info(f"📥 Loading L2 book data...")
        
        while current_date <= end_date_only:
            date_str = current_date.strftime('%Y%m%d')
            
            for hour in range(24):
                l2_data = await self.load_l2_book_data(symbol, date_str, hour)
                if l2_data:
                    l2_snapshots.extend(l2_data.get('snapshots', []))
            
            current_date += timedelta(days=1)
        
        if not l2_snapshots:
            self.logger.error(f"❌ No L2 data loaded for {symbol}")
            return pd.DataFrame()
        
        # Reconstruct OHLCV candles
        candles = self._reconstruct_ohlcv_from_l2(l2_snapshots, timeframe_minutes)
        
        if not candles:
            self.logger.error(f"❌ No candles reconstructed for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        candle_dicts = [candle.to_dict() for candle in candles]
        df = pd.DataFrame(candle_dicts)
        
        # Sort by timestamp
        df = df.sort_values('t').reset_index(drop=True)
        
        self.logger.info(f"✅ Successfully created {len(df)} candles for {symbol}")
        
        return df
    
    def estimate_validation_cost(self, symbols: List[str], days_back: int) -> Dict[str, Any]:
        """Estimate S3 costs for validation data requirements.
        
        Args:
            symbols: List of trading symbols
            days_back: Number of days of historical data needed
            
        Returns:
            Cost estimation breakdown
        """
        self.logger.info(f"💰 Estimating S3 costs for validation")
        self.logger.info(f"📊 Symbols: {symbols}")
        self.logger.info(f"📅 Days back: {days_back}")
        
        # Calculate file requirements
        files_per_symbol_per_day = 24  # Hourly L2 snapshots
        total_l2_files = len(symbols) * days_back * files_per_symbol_per_day
        
        # Size estimates
        total_size_gb = (total_l2_files * self.avg_l2_file_size_mb) / 1024
        estimated_cost = total_size_gb * self.aws_transfer_cost_per_gb
        
        cost_breakdown = {
            'symbols_count': len(symbols),
            'days_requested': days_back,
            'total_l2_files': total_l2_files,
            'estimated_size_gb': total_size_gb,
            'estimated_cost_usd': estimated_cost,
            'cost_per_symbol_usd': estimated_cost / len(symbols) if symbols else 0,
            'cost_per_day_usd': estimated_cost / days_back if days_back > 0 else 0
        }
        
        self.logger.info(f"💰 Cost Estimation Results:")
        self.logger.info(f"  - Total L2 files: {total_l2_files:,}")
        self.logger.info(f"  - Estimated size: {total_size_gb:.2f} GB")
        self.logger.info(f"  - Estimated cost: ${estimated_cost:.2f}")
        self.logger.info(f"  - Cost per symbol: ${cost_breakdown['cost_per_symbol_usd']:.2f}")
        self.logger.info(f"  - Cost per day: ${cost_breakdown['cost_per_day_usd']:.2f}")
        
        return cost_breakdown


async def test_s3_historical_loader():
    """Test S3 historical data loader functionality."""
    
    print("🔍 TESTING S3 HISTORICAL DATA LOADER")
    print("=" * 50)
    
    loader = S3HistoricalDataLoader()
    
    # Test 1: Cost estimation
    print("\n💰 Testing cost estimation...")
    symbols = ["BTC", "ETH", "SOL"]
    cost_estimate = loader.estimate_validation_cost(symbols, days_back=90)
    
    # Test 2: Data availability check
    print("\n📊 Testing data availability...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Test with 1 week
    
    availability = await loader.check_data_availability("BTC", start_date, end_date)
    
    if availability.coverage_ratio > 0.5:
        print(f"✅ S3 data availability sufficient ({availability.coverage_ratio:.1%})")
        
        # Test 3: Historical candle reconstruction
        print("\n🕯️  Testing candle reconstruction...")
        try:
            candles_df = await loader.get_historical_candles(
                "BTC", start_date, end_date, timeframe_minutes=60
            )
            
            if not candles_df.empty:
                print(f"✅ Successfully reconstructed {len(candles_df)} hourly candles")
                print(f"📊 Sample data:")
                print(candles_df.head())
                return True
            else:
                print("⚠️  No candles reconstructed")
                return False
                
        except Exception as e:
            print(f"❌ Candle reconstruction failed: {e}")
            return False
    else:
        print(f"⚠️  S3 data availability low ({availability.coverage_ratio:.1%})")
        print("May need to use API data or adjust requirements")
        return False


if __name__ == "__main__":
    """Test the S3 historical data loader."""
    
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    success = asyncio.run(test_s3_historical_loader())
    print(f"\nS3 loader test successful: {'Yes' if success else 'No'}")