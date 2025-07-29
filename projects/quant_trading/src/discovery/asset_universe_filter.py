"""
Asset Universe Filter - Research-Backed Implementation

Intelligently reduces 180 Hyperliquid assets to optimal subset for genetic algorithm focus.
Based on validated research data sources and API capabilities.

Architecture: Hierarchical Genetic Discovery - Phase 1
Implementation: Using L2 book depth, historical volatility, and correlation analysis
Data Sources: Hyperliquid candleSnapshot, l2Book, and allMids endpoints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..data.hyperliquid_client import HyperliquidClient
from ..config.settings import Settings


class FilterCriteria(Enum):
    """Asset filtering criteria with thresholds."""
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"  
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    STABILITY = "stability"


@dataclass
class AssetMetrics:
    """Comprehensive asset evaluation metrics."""
    symbol: str
    
    # Liquidity metrics (from L2 book)
    avg_bid_depth: float = 0.0
    avg_ask_depth: float = 0.0
    bid_ask_spread: float = 0.0
    depth_imbalance: float = 0.0
    
    # Volatility metrics (from historical candles)  
    daily_volatility: float = 0.0
    intraday_volatility: float = 0.0
    volatility_stability: float = 0.0
    
    # Trading metrics (from asset meta)
    max_leverage: int = 1
    size_decimals: int = 0
    
    # Derived metrics
    liquidity_score: float = 0.0
    volatility_score: float = 0.0
    trading_score: float = 0.0
    composite_score: float = 0.0
    
    # Correlation data
    correlations: Dict[str, float] = field(default_factory=dict)


class ResearchBackedAssetFilter:
    """
    Intelligent asset universe filtering using validated Hyperliquid data sources.
    
    Reduces 180 assets to 20-30 optimal candidates for genetic algorithm focus.
    Uses multi-criteria evaluation: liquidity, volatility, correlation, leverage.
    """
    
    def __init__(self, config: Settings):
        """Initialize with validated configuration and thresholds."""
        self.config = config
        self.client = HyperliquidClient(config)
        self.logger = logging.getLogger(__name__)
        
        # Research-based filtering thresholds
        self.min_liquidity_depth = 50000.0  # $50k minimum depth
        self.max_bid_ask_spread = 0.002     # 0.2% maximum spread
        self.min_daily_volatility = 0.015   # 1.5% minimum daily vol
        self.max_daily_volatility = 0.12    # 12% maximum daily vol
        self.max_correlation = 0.75         # Maximum correlation with others
        self.min_leverage = 3               # Minimum leverage for interesting trades
        
        # Target universe size
        self.target_universe_size = 25
        
        # Cache for performance
        self._asset_metrics_cache: Dict[str, AssetMetrics] = {}
        self._last_cache_update: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)
    
    async def filter_universe(
        self, 
        assets: Optional[List[str]] = None,
        refresh_cache: bool = False
    ) -> Tuple[List[str], Dict[str, AssetMetrics]]:
        """
        Main filtering method - reduces 180 assets to optimal subset.
        
        Args:
            assets: Asset list to filter (defaults to all discovered assets)
            refresh_cache: Force cache refresh for real-time filtering
            
        Returns:
            Tuple of (filtered_assets, asset_metrics)
        """
        self.logger.info("🔍 Starting intelligent asset universe filtering...")
        
        # Get asset universe if not provided
        if assets is None:
            assets = await self._discover_all_assets()
        
        self.logger.info(f"   📊 Evaluating {len(assets)} assets for filtering")
        
        # Calculate comprehensive metrics for all assets
        asset_metrics = await self._calculate_asset_metrics(assets, refresh_cache)
        
        # Apply multi-stage filtering
        filtered_assets = await self._apply_filtering_stages(asset_metrics)
        
        self.logger.info(f"   ✅ Filtered to {len(filtered_assets)} optimal assets")
        
        # Return filtered universe with metrics
        filtered_metrics = {
            asset: metrics for asset, metrics in asset_metrics.items() 
            if asset in filtered_assets
        }
        
        return filtered_assets, filtered_metrics
    
    async def _discover_all_assets(self) -> List[str]:
        """Discover all available assets using validated meta endpoint."""
        self.logger.info("   🌐 Discovering all available assets...")
        
        try:
            await self.client.connect()
            asset_contexts = await self.client.get_asset_contexts()
            assets = [ctx['name'] for ctx in asset_contexts]
            
            self.logger.info(f"   ✅ Discovered {len(assets)} tradeable assets")
            return assets
            
        except Exception as e:
            self.logger.error(f"   ❌ Asset discovery failed: {e}")
            raise
        finally:
            await self.client.disconnect()
    
    async def _calculate_asset_metrics(
        self, 
        assets: List[str], 
        refresh_cache: bool = False
    ) -> Dict[str, AssetMetrics]:
        """
        RATE LIMIT OPTIMIZED: Calculate comprehensive metrics using batch endpoints.
        
        Uses batch-optimized Hyperliquid strategy from research:
        - allMids for instant price data (1 API call for all assets)
        - Selective L2 book sampling (reduced API calls)
        - Smart caching to minimize repeated calls
        """
        # Check cache validity
        if (not refresh_cache and self._asset_metrics_cache and 
            self._last_cache_update and 
            datetime.now() - self._last_cache_update < self._cache_ttl):
            
            self.logger.info("   📋 Using cached asset metrics (cache valid)")
            return self._asset_metrics_cache
        
        self.logger.info("   📊 RATE LIMIT OPTIMIZED: Calculating asset metrics...")
        
        try:
            await self.client.connect()
            
            # OPTIMIZATION 1: Get all mid prices in ONE API call (research-backed)
            self.logger.info("   🚀 Batch fetching all mid prices (1 API call vs 180)")
            all_mids = await self.client.get_all_mids()
            
            # OPTIMIZATION 2: Pre-filter assets with price data
            active_assets = [asset for asset in assets if asset in all_mids]
            inactive_assets = set(assets) - set(active_assets)
            
            if inactive_assets:
                self.logger.info(f"   📉 Filtered out {len(inactive_assets)} inactive assets")
            
            self.logger.info(f"   ✅ Processing {len(active_assets)} active assets")
            
            # OPTIMIZATION 3: Smart batching with rate limit compliance
            # Research shows: 1200 requests/minute = 20 requests/second max
            batch_size = 10  # Conservative batching for rate limit compliance
            request_delay = 0.6  # 600ms between batches (research-backed)
            
            asset_batches = [
                active_assets[i:i + batch_size] 
                for i in range(0, len(active_assets), batch_size)
            ]
            
            all_metrics = {}
            
            for batch_num, batch in enumerate(asset_batches, 1):
                self.logger.info(f"   ⚡ Processing batch {batch_num}/{len(asset_batches)} (Rate limit compliant)")
                
                # OPTIMIZATION 4: Selective data collection
                batch_tasks = [
                    self._calculate_optimized_asset_metrics(asset, all_mids.get(asset, 0))
                    for asset in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect successful results
                for asset, result in zip(batch, batch_results):
                    if isinstance(result, AssetMetrics):
                        all_metrics[asset] = result
                    else:
                        self.logger.warning(f"   ⚠️ Failed to process {asset}: {result}")
                
                # OPTIMIZATION 5: Research-backed rate limit compliance
                if batch_num < len(asset_batches):
                    await asyncio.sleep(request_delay)
                    self.logger.debug(f"   ⏱️ Rate limit delay: {request_delay}s")
            
            # Update cache with longer TTL for rate limit efficiency
            self._asset_metrics_cache = all_metrics
            self._last_cache_update = datetime.now()
            
            self.logger.info(f"   ✅ OPTIMIZED: Calculated metrics for {len(all_metrics)} assets")
            self.logger.info(f"   📊 API call reduction: ~{len(assets) * 4} → ~{len(asset_batches) * batch_size * 2}")
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"   ❌ Optimized metrics calculation failed: {e}")
            raise
        finally:
            await self.client.disconnect()
    
    async def _calculate_optimized_asset_metrics(self, asset: str, mid_price: float) -> AssetMetrics:
        """RATE LIMIT OPTIMIZED: Calculate metrics with minimal API calls."""
        try:
            metrics = AssetMetrics(symbol=asset)
            
            # OPTIMIZATION: Use mid_price for basic filtering before expensive calls
            try:
                price_float = float(mid_price) if mid_price else 0.0
            except (ValueError, TypeError):
                price_float = 0.0
            
            if price_float <= 0:
                self.logger.debug(f"   📉 Skipping {asset} - no price data")
                return metrics
            
            # OPTIMIZATION: Prioritize L2 data (1 API call) over multiple candle calls
            try:
                liquidity_data = await self._get_liquidity_metrics(asset)
                
                metrics.avg_bid_depth = liquidity_data['bid_depth']
                metrics.avg_ask_depth = liquidity_data['ask_depth']
                metrics.bid_ask_spread = liquidity_data['spread']
                metrics.depth_imbalance = liquidity_data['imbalance']
                metrics.liquidity_score = self._calculate_liquidity_score(liquidity_data)
                
            except Exception as e:
                self.logger.debug(f"   ⚠️ L2 data failed for {asset}: {e}")
                # Set defaults for failed liquidity metrics
                metrics.liquidity_score = 0.1
            
            # OPTIMIZATION: Simplified volatility calculation (1 API call vs 2-3)
            try:
                volatility_data = await self._get_simplified_volatility_metrics(asset)
                
                metrics.daily_volatility = volatility_data['daily_vol']
                metrics.volatility_stability = volatility_data['vol_stability']
                metrics.volatility_score = self._calculate_volatility_score(volatility_data)
                
            except Exception as e:
                self.logger.debug(f"   ⚠️ Volatility data failed for {asset}: {e}")
                # Estimate volatility from mid price (no API call)
                estimated_vol = self._estimate_volatility_from_price(price_float)
                metrics.daily_volatility = estimated_vol
                metrics.volatility_score = min(0.8, estimated_vol / 0.1)  # Normalize to 0-1
            
            # OPTIMIZATION: Skip expensive correlation analysis for initial filtering
            # We'll do correlation analysis later only on top candidates
            
            # Calculate composite score with available data
            metrics.composite_score = (
                0.6 * metrics.liquidity_score +    # Weight liquidity higher 
                0.4 * metrics.volatility_score     # Simplify weighting
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"   ❌ Failed to calculate optimized metrics for {asset}: {e}")
            # Return minimal metrics instead of failing completely
            return AssetMetrics(symbol=asset)
    
    async def _get_liquidity_metrics(self, asset: str) -> Dict[str, float]:
        """Get liquidity metrics using L2 book data."""
        try:
            # Get L2 order book snapshot
            l2_data = await self.client.get_l2_book(asset)
            
            if not l2_data or 'levels' not in l2_data:
                raise ValueError(f"Invalid L2 data for {asset}")
            
            levels = l2_data['levels']
            bids = levels[0] if levels and len(levels) > 0 else []
            asks = levels[1] if levels and len(levels) > 1 else []
            
            # Calculate depth metrics
            bid_depth = sum(float(bid['sz']) * float(bid['px']) for bid in bids[:5])
            ask_depth = sum(float(ask['sz']) * float(ask['px']) for ask in asks[:5])
            
            # Calculate spread
            if bids and asks:
                best_bid = float(bids[0]['px'])
                best_ask = float(asks[0]['px'])
                spread = (best_ask - best_bid) / best_bid
            else:
                spread = 1.0  # Very wide spread for illiquid assets
            
            # Calculate depth imbalance
            total_depth = bid_depth + ask_depth
            imbalance = abs(bid_depth - ask_depth) / total_depth if total_depth > 0 else 1.0
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'spread': spread,
                'imbalance': imbalance
            }
            
        except Exception as e:
            self.logger.error(f"   ❌ L2 data error for {asset}: {e}")
            return {'bid_depth': 0, 'ask_depth': 0, 'spread': 1.0, 'imbalance': 1.0}
    
    async def _get_volatility_metrics(self, asset: str) -> Dict[str, float]:
        """Get volatility metrics using historical candle data."""
        try:
            # Get 30 days of daily data for volatility analysis
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            daily_candles = await self.client.get_candles(
                asset, 
                interval='1d',
                start_time=start_time,
                end_time=end_time
            )
            
            if not daily_candles:
                raise ValueError(f"No daily candles for {asset}")
            
            # Calculate daily returns
            closes = [float(candle['c']) for candle in daily_candles]
            if len(closes) < 10:
                raise ValueError(f"Insufficient data for {asset}")
            
            daily_returns = np.diff(closes) / closes[:-1]
            daily_vol = np.std(daily_returns)
            
            # Get hourly data for intraday volatility
            hourly_candles = await self.client.get_candles(
                asset,
                interval='1h', 
                start_time=int((datetime.now() - timedelta(days=7)).timestamp() * 1000),
                end_time=end_time
            )
            
            intraday_vol = 0.0
            if hourly_candles and len(hourly_candles) > 24:
                hourly_closes = [float(candle['c']) for candle in hourly_candles]
                hourly_returns = np.diff(hourly_closes) / hourly_closes[:-1]
                intraday_vol = np.std(hourly_returns) * np.sqrt(24)  # Annualized
            
            # Calculate volatility stability (consistency)
            if len(daily_returns) >= 14:
                vol_windows = [
                    np.std(daily_returns[i:i+7]) 
                    for i in range(len(daily_returns) - 6)
                ]
                vol_stability = 1.0 / (1.0 + np.std(vol_windows)) if vol_windows else 0.0
            else:
                vol_stability = 0.0
            
            return {
                'daily_vol': daily_vol,
                'intraday_vol': intraday_vol,
                'vol_stability': vol_stability
            }
            
        except Exception as e:
            self.logger.error(f"   ❌ Volatility data error for {asset}: {e}")
            return {'daily_vol': 0.0, 'intraday_vol': 0.0, 'vol_stability': 0.0}
    
    async def _get_simplified_volatility_metrics(self, asset: str) -> Dict[str, float]:
        """RATE LIMIT OPTIMIZED: Get volatility with minimal API calls."""
        try:
            # Get only 7 days of daily data (reduced from 30 days)
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            
            daily_candles = await self.client.get_candles(
                asset, 
                interval='1d',
                start_time=start_time,
                end_time=end_time
            )
            
            if not daily_candles or len(daily_candles) < 3:
                # Not enough data - use estimation
                raise ValueError(f"Insufficient candle data for {asset}")
            
            # Calculate daily returns from available data
            closes = [float(candle['c']) for candle in daily_candles]
            daily_returns = np.diff(closes) / closes[:-1]
            daily_vol = np.std(daily_returns)
            
            # Simplified volatility stability
            vol_stability = 1.0 / (1.0 + np.std(daily_returns)) if len(daily_returns) > 2 else 0.5
            
            return {
                'daily_vol': daily_vol,
                'vol_stability': vol_stability
            }
            
        except Exception as e:
            self.logger.debug(f"   ⚠️ Simplified volatility failed for {asset}: {e}")
            return {'daily_vol': 0.05, 'vol_stability': 0.5}  # Conservative defaults
    
    def _estimate_volatility_from_price(self, mid_price: float) -> float:
        """Estimate volatility from price level (no API calls)."""
        # Simple heuristic: higher priced assets tend to be less volatile
        # This is a rough approximation when API calls fail
        if mid_price > 1000:
            return 0.02  # Large cap, lower volatility
        elif mid_price > 100:
            return 0.04  # Mid cap
        elif mid_price > 10:
            return 0.06  # Smaller cap
        else:
            return 0.08  # Very small cap, higher volatility
    
    def _calculate_liquidity_score(self, liquidity_data: Dict[str, float]) -> float:
        """Calculate normalized liquidity score (0-1)."""
        depth_score = min(1.0, (liquidity_data['bid_depth'] + liquidity_data['ask_depth']) / (2 * self.min_liquidity_depth))
        spread_score = max(0.0, 1.0 - liquidity_data['spread'] / self.max_bid_ask_spread)
        balance_score = max(0.0, 1.0 - liquidity_data['imbalance'])
        
        return 0.5 * depth_score + 0.3 * spread_score + 0.2 * balance_score
    
    def _calculate_volatility_score(self, volatility_data: Dict[str, float]) -> float:
        """Calculate normalized volatility score (0-1) - optimal range preferred."""
        daily_vol = volatility_data['daily_vol']
        
        # Optimal volatility range for intraday strategies
        if self.min_daily_volatility <= daily_vol <= self.max_daily_volatility:
            vol_score = 1.0 - abs(daily_vol - 0.04) / 0.04  # Peak at 4% daily vol
        elif daily_vol < self.min_daily_volatility:
            vol_score = daily_vol / self.min_daily_volatility
        else:
            vol_score = max(0.0, 1.0 - (daily_vol - self.max_daily_volatility) / 0.1)
        
        stability_score = volatility_data['vol_stability']
        
        return 0.7 * vol_score + 0.3 * stability_score
    
    async def _apply_filtering_stages(
        self, 
        asset_metrics: Dict[str, AssetMetrics]
    ) -> List[str]:
        """Apply multi-stage filtering to select optimal assets."""
        self.logger.info("   🎯 Applying multi-stage filtering...")
        
        # Stage 1: Basic viability filters
        viable_assets = []
        for asset, metrics in asset_metrics.items():
            if (metrics.liquidity_score > 0.3 and 
                metrics.volatility_score > 0.2 and
                metrics.avg_bid_depth > 1000.0):  # Basic liquidity threshold
                viable_assets.append(asset)
        
        self.logger.info(f"   ✅ Stage 1: {len(viable_assets)} viable assets")
        
        # Stage 2: Correlation diversity filter
        diverse_assets = await self._apply_correlation_filter(viable_assets, asset_metrics)
        self.logger.info(f"   ✅ Stage 2: {len(diverse_assets)} diverse assets")
        
        # Stage 3: Composite score ranking
        scored_assets = [
            (asset, asset_metrics[asset].composite_score)
            for asset in diverse_assets
        ]
        scored_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Select top assets up to target size
        selected_assets = [
            asset for asset, score in scored_assets[:self.target_universe_size]
        ]
        
        self.logger.info(f"   ✅ Stage 3: Selected top {len(selected_assets)} assets")
        
        return selected_assets
    
    async def _apply_correlation_filter(
        self, 
        assets: List[str],
        asset_metrics: Dict[str, AssetMetrics]
    ) -> List[str]:
        """Filter assets to ensure correlation diversity."""
        if len(assets) <= self.target_universe_size:
            return assets
        
        self.logger.info("   📊 Calculating correlation matrix for diversity...")
        
        try:
            # Get correlation data for assets
            correlation_matrix = await self._calculate_correlation_matrix(assets)
            
            # Greedy selection for diverse portfolio
            selected = []
            remaining = assets.copy()
            
            # Start with highest scoring asset
            best_asset = max(remaining, key=lambda a: asset_metrics[a].composite_score)
            selected.append(best_asset)
            remaining.remove(best_asset)
            
            # Add assets with low correlation to selected ones
            while len(selected) < self.target_universe_size and remaining:
                best_candidate = None
                best_score = -1
                
                for candidate in remaining:
                    # Calculate max correlation with selected assets
                    max_corr = max(
                        abs(correlation_matrix.get((candidate, selected_asset), 0.0))
                        for selected_asset in selected
                    )
                    
                    # Score combines composite score with diversity bonus
                    diversity_bonus = max(0.0, 1.0 - max_corr / self.max_correlation)
                    candidate_score = (
                        0.6 * asset_metrics[candidate].composite_score +
                        0.4 * diversity_bonus
                    )
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            return selected
            
        except Exception as e:
            self.logger.error(f"   ❌ Correlation filtering failed: {e}")
            # Fall back to simple ranking if correlation analysis fails
            return sorted(assets, key=lambda a: asset_metrics[a].composite_score, reverse=True)[:self.target_universe_size]
    
    async def _calculate_correlation_matrix(self, assets: List[str]) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise correlations using historical price data."""
        correlation_matrix = {}
        
        try:
            await self.client.connect()
            
            # Get price data for all assets
            price_data = {}
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            for asset in assets:
                try:
                    candles = await self.client.get_candles(
                        asset,
                        interval='1d',
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if candles and len(candles) >= 10:
                        closes = [float(candle['c']) for candle in candles]
                        returns = np.diff(closes) / closes[:-1]
                        price_data[asset] = returns
                        
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Price data error for {asset}: {e}")
            
            # Calculate pairwise correlations
            assets_with_data = list(price_data.keys())
            for i, asset1 in enumerate(assets_with_data):
                for j, asset2 in enumerate(assets_with_data):
                    if i < j:  # Only calculate upper triangle
                        try:
                            returns1 = price_data[asset1]
                            returns2 = price_data[asset2]
                            
                            # Align data lengths
                            min_len = min(len(returns1), len(returns2))
                            if min_len >= 10:
                                corr = np.corrcoef(
                                    returns1[:min_len], 
                                    returns2[:min_len]
                                )[0, 1]
                                
                                if not np.isnan(corr):
                                    correlation_matrix[(asset1, asset2)] = corr
                                    correlation_matrix[(asset2, asset1)] = corr
                                    
                        except Exception as e:
                            self.logger.warning(f"   ⚠️ Correlation error {asset1}-{asset2}: {e}")
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"   ❌ Correlation matrix calculation failed: {e}")
            return {}
        finally:
            await self.client.disconnect()
    
    def get_filter_summary(self, filtered_assets: List[str], metrics: Dict[str, AssetMetrics]) -> Dict:
        """Generate comprehensive filtering summary for analysis."""
        if not filtered_assets:
            return {
                "error": "No assets selected",
                "selected_assets": 0,
                "target_size": self.target_universe_size,
                "assets": [],
                "average_liquidity_score": 0.0,
                "average_volatility_score": 0.0,
                "average_composite_score": 0.0,
                "score_range": {"min": 0.0, "max": 0.0},
                "top_performers": []
            }
        
        # Calculate summary statistics
        liquidity_scores = [metrics[asset].liquidity_score for asset in filtered_assets]
        volatility_scores = [metrics[asset].volatility_score for asset in filtered_assets]
        composite_scores = [metrics[asset].composite_score for asset in filtered_assets]
        
        return {
            "selected_assets": len(filtered_assets),
            "target_size": self.target_universe_size,
            "assets": filtered_assets,
            "average_liquidity_score": np.mean(liquidity_scores),
            "average_volatility_score": np.mean(volatility_scores),
            "average_composite_score": np.mean(composite_scores),
            "score_range": {
                "min": min(composite_scores),
                "max": max(composite_scores)
            },
            "top_performers": sorted(
                [(asset, metrics[asset].composite_score) for asset in filtered_assets],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }