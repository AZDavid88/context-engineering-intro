"""
Dynamic Asset Data Collector - Production-Ready Multi-Timeframe Data Pipeline

This module implements the Dynamic Asset Data Collection system that integrates seamlessly
with the enhanced asset filter discovery system. Key features:

- API-Only Data Collection: Pure Hyperliquid Mainnet API strategy (no S3 dependencies)
- Tradeable Assets Validation: Filters for assets with proper trading constraints
- Multi-Timeframe Data Maximization: Separate 5000-bar downloads (1h: 208 days, 15m: 52 days)
- Discovery System Integration: Connects with enhanced_asset_filter.py output
- Research-Backed Patterns: Full compliance with /research/ anti-hallucination protocol

Integration Pipeline:
Discovery → Tradeable Filter → Multi-Timeframe Collection → Genetic Evolution
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..discovery.enhanced_asset_filter import EnhancedAssetFilter, RequestPriority
from ..data.hyperliquid_client import HyperliquidClient
from ..config.settings import Settings


@dataclass
class DataCollectionMetrics:
    """Comprehensive metrics for data collection performance tracking."""
    
    # Collection performance
    total_assets_processed: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    total_collection_time: float = 0.0
    
    # Data quality metrics
    total_bars_collected: Dict[str, int] = None
    data_completeness_rate: float = 0.0
    timestamp_alignment_errors: int = 0
    
    # API performance metrics
    api_calls_made: int = 0
    api_rate_limit_hits: int = 0
    average_api_latency: float = 0.0
    
    # Memory usage tracking
    peak_memory_usage_gb: float = 0.0
    data_storage_size_mb: float = 0.0
    
    def __post_init__(self):
        if self.total_bars_collected is None:
            self.total_bars_collected = {'1h': 0, '15m': 0}
    
    @property
    def collection_success_rate(self) -> float:
        """Calculate collection success rate percentage."""
        if self.total_assets_processed == 0:
            return 0.0
        return (self.successful_collections / self.total_assets_processed) * 100
    
    @property
    def total_bars_collected_all(self) -> int:
        """Total bars collected across all timeframes."""
        return sum(self.total_bars_collected.values())


@dataclass 
class AssetDataSet:
    """Complete dataset for a single asset across multiple timeframes."""
    
    asset_symbol: str
    timeframe_data: Dict[str, pd.DataFrame]  # '1h' -> DataFrame, '15m' -> DataFrame
    collection_timestamp: datetime
    data_quality_score: float
    bars_collected: Dict[str, int]
    
    # Additional data for genetic seeds requiring enhanced datasets
    funding_rate_data: Optional[pd.DataFrame] = None
    volume_profile_data: Optional[Dict[str, Any]] = None
    microstructure_data: Optional[Dict[str, Any]] = None
    
    def validate_data_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate data integrity across timeframes.
        
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        validation_errors = []
        
        # Check timeframe data exists
        required_timeframes = ['1h', '15m']
        for tf in required_timeframes:
            if tf not in self.timeframe_data or self.timeframe_data[tf].empty:
                validation_errors.append(f"Missing or empty {tf} data")
        
        # Check minimum bar counts
        min_bars = {'1h': 1000, '15m': 1000}  # Minimum viable for genetic evolution
        for tf, min_count in min_bars.items():
            if tf in self.bars_collected and self.bars_collected[tf] < min_count:
                validation_errors.append(f"{tf} insufficient bars: {self.bars_collected[tf]} < {min_count}")
        
        # Check timestamp alignment
        if '1h' in self.timeframe_data and '15m' in self.timeframe_data:
            if not self._check_timestamp_alignment():
                validation_errors.append("Timestamp alignment issues between timeframes")
        
        return len(validation_errors) == 0, validation_errors
    
    def _check_timestamp_alignment(self) -> bool:
        """Check if timeframe data has proper timestamp alignment."""
        try:
            # Get latest timestamps from each timeframe
            h1_latest = self.timeframe_data['1h'].index[-1] if not self.timeframe_data['1h'].empty else None
            m15_latest = self.timeframe_data['15m'].index[-1] if not self.timeframe_data['15m'].empty else None
            
            if h1_latest is None or m15_latest is None:
                return False
            
            # Allow up to 1 hour difference between latest timestamps
            time_diff = abs((h1_latest - m15_latest).total_seconds())
            return time_diff <= 3600  # 1 hour tolerance
            
        except Exception:
            return False


class DynamicAssetDataCollector:
    """
    Production-ready dynamic asset data collector with multi-timeframe capability.
    
    This class implements the core data collection pipeline for the genetic trading system,
    connecting seamlessly with the enhanced asset filter and providing comprehensive
    datasets for genetic algorithm evolution.
    
    Key Features:
    - API-Only Strategy: Pure Hyperliquid Mainnet API (no S3 dependencies)
    - Tradeable Assets Validation: Ensures only tradeable assets are collected
    - Multi-Timeframe Data Maximization: 5000 bars per timeframe
    - Rate Limiting Compliance: Advanced rate limiting integration
    - Memory Optimization: Efficient data handling for large datasets
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize dynamic asset data collector.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.client = HyperliquidClient(settings)
        self.logger = logging.getLogger(__name__)
        
        # Data collection configuration
        self.max_bars_per_timeframe = 5000
        self.supported_timeframes = ['1h', '15m']
        self.timeframe_limits = {
            '1h': {'bars': 5000, 'days': 208},    # 5000 bars = ~208 days
            '15m': {'bars': 5000, 'days': 52}     # 5000 bars = ~52 days
        }
        
        # Performance tracking
        self.collection_metrics = DataCollectionMetrics()
        self.collected_datasets: Dict[str, AssetDataSet] = {}
        
        # Rate limiting integration (will be connected to enhanced_asset_filter)
        self.rate_limiter = None
        
        self.logger.info("🚀 Dynamic Asset Data Collector initialized with API-only strategy")
    
    async def connect_with_discovery_system(self, enhanced_filter: EnhancedAssetFilter):
        """
        Connect with the enhanced asset filter discovery system.
        
        Args:
            enhanced_filter: Enhanced asset filter instance with rate limiting
        """
        self.rate_limiter = enhanced_filter.rate_limiter
        self.logger.info("✅ Connected with enhanced asset discovery system")
    
    async def collect_assets_data_pipeline(
        self, 
        discovered_assets: List[str],
        include_enhanced_data: bool = True
    ) -> Dict[str, AssetDataSet]:
        """
        Main pipeline for collecting multi-timeframe data for discovered assets.
        
        This is the primary entry point that integrates with the enhanced asset filter
        output and produces comprehensive datasets ready for genetic evolution.
        
        Args:
            discovered_assets: Assets identified by enhanced asset filter
            include_enhanced_data: Whether to collect additional data for advanced genetic seeds
            
        Returns:
            Dictionary of asset_symbol -> AssetDataSet with comprehensive multi-timeframe data
        """
        self.logger.info(f"🔍 Starting data collection pipeline for {len(discovered_assets)} assets")
        
        pipeline_start_time = time.time()
        
        # Reset metrics for this collection run
        self.collection_metrics = DataCollectionMetrics()
        self.collection_metrics.total_assets_processed = len(discovered_assets)
        
        try:
            # Step 1: Filter for tradeable assets only (CRITICAL)
            tradeable_assets = await self._filter_tradeable_assets_only(discovered_assets)
            
            # Step 2: Collect multi-timeframe data with rate limiting
            collected_datasets = await self._collect_multi_timeframe_data_batch(
                tradeable_assets, include_enhanced_data
            )
            
            # Step 3: Validate data integrity across all collected datasets
            validated_datasets = await self._validate_collected_datasets(collected_datasets)
            
            # Step 4: Generate collection summary
            self.collection_metrics.total_collection_time = time.time() - pipeline_start_time
            self._generate_collection_summary(validated_datasets)
            
            # Store datasets for future access
            self.collected_datasets.update(validated_datasets)
            
            self.logger.info(f"✅ Data collection pipeline completed: {len(validated_datasets)} assets ready")
            
            return validated_datasets
            
        except Exception as e:
            self.logger.error(f"❌ Data collection pipeline failed: {e}")
            raise
    
    async def _filter_tradeable_assets_only(self, all_assets: List[str]) -> List[str]:
        """
        Filter for tradeable assets only using asset contexts from Hyperliquid API.
        
        This ensures we only collect data for assets that can actually be traded,
        preventing wasted resources on non-tradeable assets.
        
        Args:
            all_assets: All discovered assets from enhanced filter
            
        Returns:
            List of assets that are confirmed tradeable
        """
        self.logger.info("🔍 Filtering for tradeable assets using asset contexts...")
        
        try:
            await self.client.connect()
            
            # Get asset contexts with rate limiting
            if self.rate_limiter:
                asset_contexts = await self.rate_limiter.execute_rate_limited_request(
                    self.client.get_asset_contexts,
                    cache_key="tradeable_asset_contexts",
                    cache_category="asset_metadata",
                    priority=RequestPriority.CRITICAL
                )
            else:
                asset_contexts = await self.client.get_asset_contexts()
            
            if not asset_contexts:
                self.logger.warning("⚠️ No asset contexts received - using all assets as fallback")
                return all_assets
            
            # Extract tradeable asset names with proper constraints
            tradeable_assets = []
            tradeable_constraints = {}
            
            for context in asset_contexts:
                asset_name = context.get('name')
                
                # Check if asset is tradeable (has proper trading constraints)
                if (asset_name and 
                    'maxLeverage' in context and 
                    'szDecimals' in context and
                    context.get('maxLeverage', 0) > 0 and
                    not context.get('onlyIsolated', False)):  # Avoid isolated-only assets
                    
                    tradeable_assets.append(asset_name)
                    tradeable_constraints[asset_name] = {
                        'max_leverage': context.get('maxLeverage'),
                        'size_decimals': context.get('szDecimals'),
                        'only_isolated': context.get('onlyIsolated', False)
                    }
            
            # Filter provided assets to only include confirmed tradeable ones
            filtered_assets = [asset for asset in all_assets if asset in tradeable_assets]
            
            # Log detailed filtering results
            self.logger.info(f"📊 Asset tradeability analysis:")
            self.logger.info(f"   🔍 Total discovered assets: {len(all_assets)}")
            self.logger.info(f"   ✅ Confirmed tradeable assets: {len(tradeable_assets)}")
            self.logger.info(f"   🎯 Final filtered assets: {len(filtered_assets)}")
            
            if len(filtered_assets) < len(all_assets):
                non_tradeable = set(all_assets) - set(filtered_assets)
                self.logger.info(f"   ❌ Non-tradeable assets excluded: {list(non_tradeable)}")
            
            return filtered_assets
            
        except Exception as e:
            self.logger.error(f"❌ Failed to filter tradeable assets: {e}")
            # Fallback to all assets if filtering fails (graceful degradation)
            self.logger.warning("⚠️ Using all discovered assets as fallback due to filtering error")
            return all_assets
        finally:
            await self.client.disconnect()
    
    async def _collect_multi_timeframe_data_batch(
        self, 
        tradeable_assets: List[str],
        include_enhanced_data: bool = True
    ) -> Dict[str, AssetDataSet]:
        """
        Collect multi-timeframe data for all tradeable assets with batch optimization.
        
        Args:
            tradeable_assets: Confirmed tradeable assets
            include_enhanced_data: Whether to collect enhanced data for advanced genetic seeds
            
        Returns:
            Dictionary of collected asset datasets
        """
        self.logger.info(f"📊 Collecting multi-timeframe data for {len(tradeable_assets)} assets")
        
        collected_datasets = {}
        
        try:
            await self.client.connect()
            
            # Process assets in batches for memory efficiency
            batch_size = min(10, len(tradeable_assets))  # Process up to 10 assets simultaneously
            
            for batch_start in range(0, len(tradeable_assets), batch_size):
                batch_end = min(batch_start + batch_size, len(tradeable_assets))
                batch_assets = tradeable_assets[batch_start:batch_end]
                
                self.logger.info(f"📦 Processing batch {batch_start//batch_size + 1}: {len(batch_assets)} assets")
                
                # Collect data for batch assets concurrently
                batch_tasks = [
                    self._collect_single_asset_data(asset, include_enhanced_data)
                    for asset in batch_assets
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process batch results
                for asset, result in zip(batch_assets, batch_results):
                    if isinstance(result, AssetDataSet):
                        collected_datasets[asset] = result
                        self.collection_metrics.successful_collections += 1
                    else:
                        self.logger.error(f"❌ Failed to collect data for {asset}: {result}")
                        self.collection_metrics.failed_collections += 1
                
                # Brief pause between batches to respect rate limits
                if batch_end < len(tradeable_assets):  # Not the last batch
                    await asyncio.sleep(1)
            
            return collected_datasets
            
        except Exception as e:
            self.logger.error(f"❌ Batch data collection failed: {e}")
            raise
        finally:
            await self.client.disconnect()
    
    async def _collect_single_asset_data(
        self, 
        asset: str, 
        include_enhanced_data: bool = True
    ) -> AssetDataSet:
        """
        Collect complete dataset for a single asset across all timeframes.
        
        Args:
            asset: Asset symbol to collect data for
            include_enhanced_data: Whether to collect enhanced data for advanced genetic seeds
            
        Returns:
            Complete AssetDataSet with multi-timeframe data
        """
        asset_start_time = time.time()
        
        try:
            self.logger.info(f"📈 Collecting data for {asset}")
            
            # Collect primary timeframe data
            timeframe_data = {}
            bars_collected = {}
            
            for timeframe in self.supported_timeframes:
                self.logger.info(f"   ⏱️  Collecting {timeframe} data for {asset}")
                
                df, bar_count = await self._collect_timeframe_data(asset, timeframe)
                timeframe_data[timeframe] = df
                bars_collected[timeframe] = bar_count
                
                self.collection_metrics.total_bars_collected[timeframe] += bar_count
                
                # Brief pause between timeframe collections
                await asyncio.sleep(0.5)
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(timeframe_data, bars_collected)
            
            # Create asset dataset
            asset_dataset = AssetDataSet(
                asset_symbol=asset,
                timeframe_data=timeframe_data,
                collection_timestamp=datetime.now(),
                data_quality_score=data_quality_score,
                bars_collected=bars_collected
            )
            
            # Collect enhanced data if requested (for advanced genetic seeds)
            if include_enhanced_data:
                await self._collect_enhanced_data_for_asset(asset_dataset)
            
            collection_time = time.time() - asset_start_time
            self.logger.info(f"✅ {asset} collection completed in {collection_time:.2f}s (quality: {data_quality_score:.2f})")
            
            return asset_dataset
            
        except Exception as e:
            self.logger.error(f"❌ Failed to collect data for {asset}: {e}")
            raise
    
    async def _collect_timeframe_data(self, asset: str, timeframe: str) -> Tuple[pd.DataFrame, int]:
        """
        Collect data for a specific asset and timeframe using API-only strategy.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe ('1h' or '15m')
            
        Returns:
            Tuple of (DataFrame with OHLCV data, number of bars collected)
        """
        try:
            # Calculate time range for maximum bars
            end_time = int(datetime.now().timestamp() * 1000)
            
            # Timeframe to milliseconds mapping
            timeframe_ms = {
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000
            }
            
            max_bars = self.timeframe_limits[timeframe]['bars']
            start_time = end_time - (max_bars * timeframe_ms[timeframe])
            
            # API call with rate limiting
            if self.rate_limiter:
                candles = await self.rate_limiter.execute_rate_limited_request(
                    self.client.get_candles,
                    cache_key=f"candles_{asset}_{timeframe}_{start_time}",
                    cache_category="historical_data",
                    priority=RequestPriority.HIGH,
                    symbol=asset,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
            else:
                candles = await self.client.get_candles(
                    symbol=asset,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
            
            self.collection_metrics.api_calls_made += 1
            
            if not candles:
                self.logger.warning(f"⚠️ No candles received for {asset} {timeframe}")
                return pd.DataFrame(), 0
            
            # Convert to DataFrame
            df = self._convert_candles_to_dataframe(candles)
            bar_count = len(df)
            
            self.logger.info(f"   📊 Collected {bar_count} bars for {asset} {timeframe}")
            
            return df, bar_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to collect {timeframe} data for {asset}: {e}")
            return pd.DataFrame(), 0
    
    def _convert_candles_to_dataframe(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert Hyperliquid candle data to pandas DataFrame.
        
        Args:
            candles: List of candle dictionaries from Hyperliquid API
            
        Returns:
            DataFrame with OHLCV data and proper datetime index
        """
        if not candles:
            return pd.DataFrame()
        
        try:
            # Convert candles to DataFrame
            df_data = []
            for candle in candles:
                df_data.append({
                    'timestamp': candle.get('t', 0),
                    'open': float(candle.get('o', 0)),
                    'high': float(candle.get('h', 0)),
                    'low': float(candle.get('l', 0)),
                    'close': float(candle.get('c', 0)),
                    'volume': float(candle.get('v', 0))
                })
            
            df = pd.DataFrame(df_data)
            
            if df.empty:
                return df
            
            # Convert timestamp to datetime and set as index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            # Sort by datetime to ensure proper order
            df.sort_index(inplace=True)
            
            # Ensure all columns are numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert candles to DataFrame: {e}")
            return pd.DataFrame()
    
    async def _collect_enhanced_data_for_asset(self, asset_dataset: AssetDataSet):
        """
        Collect enhanced data for genetic seeds requiring additional information.
        
        This method collects funding rates, volume profile, and microstructure data
        as identified in the genetic seed data requirements analysis.
        
        Args:
            asset_dataset: Asset dataset to enhance with additional data
        """
        try:
            asset = asset_dataset.asset_symbol
            
            # Collect funding rate data (for funding_rate_carry_seed)
            # Note: This would require additional API endpoints or calculations
            # Placeholder for now - to be implemented based on specific genetic seed requirements
            
            self.logger.info(f"📊 Enhanced data collection for {asset} - placeholder for future implementation")
            
            # Future implementations:
            # - Funding rate history collection
            # - Volume profile analysis 
            # - Microstructure data (bid-ask spread, order book depth)
            # - Institutional flow indicators
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced data collection failed for {asset_dataset.asset_symbol}: {e}")
    
    def _calculate_data_quality_score(
        self, 
        timeframe_data: Dict[str, pd.DataFrame], 
        bars_collected: Dict[str, int]
    ) -> float:
        """
        Calculate data quality score based on completeness and consistency.
        
        Args:
            timeframe_data: Dictionary of timeframe DataFrames
            bars_collected: Dictionary of bar counts per timeframe
            
        Returns:
            Data quality score (0.0 to 1.0)
        """
        try:
            quality_factors = []
            
            # Factor 1: Data completeness (bars collected vs target)
            for timeframe in self.supported_timeframes:
                target_bars = self.timeframe_limits[timeframe]['bars']
                actual_bars = bars_collected.get(timeframe, 0)
                completeness = min(1.0, actual_bars / target_bars)
                quality_factors.append(completeness)
            
            # Factor 2: Data consistency (no missing values, proper OHLCV structure)
            for timeframe, df in timeframe_data.items():
                if df.empty:
                    quality_factors.append(0.0)
                    continue
                
                # Check for required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                has_all_cols = all(col in df.columns for col in required_cols)
                
                # Check for reasonable price relationships (high >= low, etc.)
                price_consistency = (
                    (df['high'] >= df['low']).all() and
                    (df['high'] >= df['open']).all() and
                    (df['high'] >= df['close']).all() and
                    (df['low'] <= df['open']).all() and
                    (df['low'] <= df['close']).all()
                )
                
                consistency_score = 1.0 if (has_all_cols and price_consistency) else 0.5
                quality_factors.append(consistency_score)
            
            # Calculate overall quality score
            if not quality_factors:
                return 0.0
            
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate data quality score: {e}")
            return 0.0
    
    async def _validate_collected_datasets(
        self, 
        collected_datasets: Dict[str, AssetDataSet]
    ) -> Dict[str, AssetDataSet]:
        """
        Validate all collected datasets for integrity and completeness.
        
        Args:
            collected_datasets: Dictionary of collected asset datasets
            
        Returns:
            Dictionary of validated datasets (invalid datasets removed)
        """
        self.logger.info(f"🔍 Validating {len(collected_datasets)} collected datasets")
        
        validated_datasets = {}
        validation_errors = []
        
        for asset, dataset in collected_datasets.items():
            is_valid, errors = dataset.validate_data_integrity()
            
            if is_valid:
                validated_datasets[asset] = dataset
            else:
                validation_errors.extend([f"{asset}: {error}" for error in errors])
                self.logger.warning(f"⚠️ Dataset validation failed for {asset}: {errors}")
        
        # Update metrics
        self.collection_metrics.data_completeness_rate = (
            len(validated_datasets) / len(collected_datasets) * 100 
            if collected_datasets else 0
        )
        
        if validation_errors:
            self.logger.warning(f"⚠️ Dataset validation issues:")
            for error in validation_errors[:10]:  # Log first 10 errors
                self.logger.warning(f"   - {error}")
        
        self.logger.info(f"✅ Dataset validation completed: {len(validated_datasets)} valid datasets")
        
        return validated_datasets
    
    def _generate_collection_summary(self, validated_datasets: Dict[str, AssetDataSet]):
        """
        Generate comprehensive collection summary with performance metrics.
        
        Args:
            validated_datasets: Final validated datasets
        """
        try:
            # Calculate comprehensive metrics
            total_success_rate = self.collection_metrics.collection_success_rate
            total_bars = self.collection_metrics.total_bars_collected_all
            
            # Average data quality
            if validated_datasets:
                avg_quality = sum(dataset.data_quality_score for dataset in validated_datasets.values()) / len(validated_datasets)
            else:
                avg_quality = 0.0
            
            # Generation collection report
            self.logger.info("📊 Data Collection Pipeline Summary:")
            self.logger.info("=" * 50)
            self.logger.info(f"   📈 Assets Processed: {self.collection_metrics.total_assets_processed}")
            self.logger.info(f"   ✅ Successful Collections: {self.collection_metrics.successful_collections}")
            self.logger.info(f"   ❌ Failed Collections: {self.collection_metrics.failed_collections}")
            self.logger.info(f"   🎯 Success Rate: {total_success_rate:.1f}%")
            self.logger.info(f"   ⏱️  Total Collection Time: {self.collection_metrics.total_collection_time:.2f}s")
            self.logger.info(f"   📊 Total Bars Collected: {total_bars:,}")
            self.logger.info(f"   📋 1h Bars: {self.collection_metrics.total_bars_collected['1h']:,}")
            self.logger.info(f"   📋 15m Bars: {self.collection_metrics.total_bars_collected['15m']:,}")
            self.logger.info(f"   🏆 Average Data Quality: {avg_quality:.2f}")
            self.logger.info(f"   🌐 API Calls Made: {self.collection_metrics.api_calls_made}")
            self.logger.info(f"   💾 Data Completeness: {self.collection_metrics.data_completeness_rate:.1f}%")
            
            # Ready for genetic evolution indicator
            if len(validated_datasets) > 0 and avg_quality > 0.7:
                self.logger.info("🧬 READY FOR GENETIC EVOLUTION: High-quality datasets available")
            else:
                self.logger.warning("⚠️  Data quality concerns - review before genetic evolution")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate collection summary: {e}")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive collection summary for external systems.
        
        Returns:
            Dictionary with collection metrics and status
        """
        return {
            'collection_metrics': {
                'total_assets_processed': self.collection_metrics.total_assets_processed,
                'successful_collections': self.collection_metrics.successful_collections,
                'failed_collections': self.collection_metrics.failed_collections,
                'success_rate': self.collection_metrics.collection_success_rate,
                'total_collection_time': self.collection_metrics.total_collection_time,
                'total_bars_collected': self.collection_metrics.total_bars_collected_all,
                'api_calls_made': self.collection_metrics.api_calls_made,
                'data_completeness_rate': self.collection_metrics.data_completeness_rate
            },
            'dataset_status': {
                'total_datasets': len(self.collected_datasets),
                'average_quality_score': (
                    sum(ds.data_quality_score for ds in self.collected_datasets.values()) / 
                    len(self.collected_datasets) if self.collected_datasets else 0.0
                ),
                'ready_for_evolution': len(self.collected_datasets) > 0
            },
            'collection_timestamp': datetime.now().isoformat()
        }
    
    async def get_asset_data(self, asset: str) -> Optional[AssetDataSet]:
        """
        Get collected data for a specific asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            AssetDataSet if available, None otherwise
        """
        return self.collected_datasets.get(asset)
    
    async def get_multi_asset_data(
        self, 
        timeframe: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get multi-asset data for a specific timeframe (for genetic evolution input).
        
        Args:
            timeframe: Timeframe to extract ('1h' or '15m')
            
        Returns:
            Dictionary of asset -> DataFrame for the specified timeframe
        """
        multi_asset_data = {}
        
        for asset, dataset in self.collected_datasets.items():
            if timeframe in dataset.timeframe_data:
                multi_asset_data[asset] = dataset.timeframe_data[timeframe]
        
        return multi_asset_data


# Integration Pipeline Orchestrator
class IntegratedPipelineOrchestrator:
    """
    Orchestrates the complete pipeline: Discovery → Data Collection → Genetic Evolution.
    
    This class serves as the main integration point between the enhanced asset filter,
    dynamic asset data collector, and genetic evolution engine.
    """
    
    def __init__(self, settings: Settings):
        """Initialize pipeline orchestrator."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.asset_filter = EnhancedAssetFilter(settings)
        self.data_collector = DynamicAssetDataCollector(settings)
        
        # Pipeline state
        self.pipeline_results = {}
        
    async def execute_full_pipeline(self, enable_optimizations: bool = True) -> Dict[str, Any]:
        """
        Execute the complete integrated pipeline.
        
        Args:
            enable_optimizations: Whether to enable advanced optimizations
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.logger.info("🚀 Starting integrated pipeline execution")
        pipeline_start_time = time.time()
        
        try:
            # Stage 1: Enhanced asset discovery
            self.logger.info("📍 Stage 1: Enhanced Asset Discovery")
            discovered_assets, asset_metrics = await self.asset_filter.filter_universe(
                enable_optimizations=enable_optimizations
            )
            
            # Stage 2: Connect data collector with discovery system
            self.logger.info("📍 Stage 2: Connecting Data Collection Pipeline") 
            await self.data_collector.connect_with_discovery_system(self.asset_filter)
            
            # Stage 3: Dynamic multi-timeframe data collection
            self.logger.info("📍 Stage 3: Dynamic Multi-Timeframe Data Collection")
            collected_datasets = await self.data_collector.collect_assets_data_pipeline(
                discovered_assets, include_enhanced_data=True
            )
            
            # Stage 4: Prepare data for genetic evolution
            self.logger.info("📍 Stage 4: Preparing Data for Genetic Evolution")
            evolution_ready_data = {
                '1h': await self.data_collector.get_multi_asset_data('1h'),
                '15m': await self.data_collector.get_multi_asset_data('15m')
            }
            
            # Compile pipeline results
            pipeline_time = time.time() - pipeline_start_time
            
            self.pipeline_results = {
                'discovery_results': {
                    'discovered_assets': discovered_assets,
                    'asset_metrics': asset_metrics,
                    'filter_summary': self.asset_filter.get_enhanced_filter_summary(
                        discovered_assets, asset_metrics
                    )
                },
                'data_collection_results': {
                    'collected_datasets': collected_datasets,
                    'collection_summary': self.data_collector.get_collection_summary()
                },
                'evolution_ready_data': evolution_ready_data,
                'pipeline_metrics': {
                    'total_pipeline_time': pipeline_time,
                    'assets_discovered': len(discovered_assets),
                    'datasets_collected': len(collected_datasets),
                    'ready_for_evolution': len(evolution_ready_data['1h']) > 0,
                    'pipeline_success': True
                }
            }
            
            self.logger.info(f"✅ Integrated pipeline completed successfully in {pipeline_time:.2f}s")
            self.logger.info(f"🧬 Ready for genetic evolution with {len(evolution_ready_data['1h'])} assets")
            
            return self.pipeline_results
            
        except Exception as e:
            self.logger.error(f"❌ Integrated pipeline failed: {e}")
            
            # Return partial results for debugging
            self.pipeline_results['pipeline_metrics'] = {
                'pipeline_success': False,
                'error_message': str(e),
                'total_pipeline_time': time.time() - pipeline_start_time
            }
            
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and results."""
        return {
            'pipeline_executed': bool(self.pipeline_results),
            'last_execution_results': self.pipeline_results,
            'components_status': {
                'asset_filter_ready': self.asset_filter is not None,
                'data_collector_ready': self.data_collector is not None,
                'integration_established': (
                    self.data_collector.rate_limiter is not None 
                    if hasattr(self.data_collector, 'rate_limiter') else False
                )
            }
        }