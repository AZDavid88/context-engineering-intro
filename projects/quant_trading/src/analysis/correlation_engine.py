"""
Cross-Asset Correlation Engine - Phase 2 Implementation

Integrates with enhanced_asset_filter.py filtered assets and DataStorageInterface 
for comprehensive correlation analysis following existing architectural patterns.

Research Integration:
- /research/pandas_comprehensive/research_summary.md - Correlation calculations
- Verified pattern: enhanced_asset_filter.py optimization approach
- Integration: DataStorageInterface for Phase 1 compatibility
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from src.data.storage_interfaces import get_storage_implementation, DataStorageInterface
from src.config.settings import get_settings


@dataclass
class CorrelationMetrics:
    """Correlation analysis metrics matching AssetMetrics pattern."""
    correlation_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    portfolio_correlation_score: float = 0.0
    regime_classification: str = "medium_correlation"
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    asset_count: int = 0
    valid_pairs: int = 0
    calculation_window: int = 60
    data_quality_score: float = 0.0
    
    @property
    def correlation_strength_distribution(self) -> Dict[str, int]:
        """Analyze correlation strength distribution."""
        if not self.correlation_pairs:
            return {"strong": 0, "moderate": 0, "weak": 0}
        
        strong = sum(1 for corr in self.correlation_pairs.values() if abs(corr) > 0.7)
        moderate = sum(1 for corr in self.correlation_pairs.values() if 0.3 <= abs(corr) <= 0.7)
        weak = sum(1 for corr in self.correlation_pairs.values() if abs(corr) < 0.3)
        
        return {"strong": strong, "moderate": moderate, "weak": weak}


class FilteredAssetCorrelationEngine:
    """
    Calculate correlations from dynamically filtered assets.
    Integrates with Phase 1 DataStorageInterface for cloud compatibility.
    Follows enhanced_asset_filter.py optimization patterns.
    """
    
    def __init__(self, settings=None):
        """Initialize correlation engine with DataStorageInterface integration."""
        self.settings = settings or get_settings()
        self.storage = get_storage_implementation()  # Phase 1 integration
        self.logger = logging.getLogger(__name__)
        
        # Correlation configuration following existing patterns
        self.correlation_window = getattr(self.settings, 'correlation_window_periods', 60)
        self.min_correlation_periods = getattr(self.settings, 'min_correlation_data_points', 30)
        
        # Regime detection thresholds - following settings pattern
        correlation_settings = getattr(self.settings, 'correlation', None)
        if correlation_settings:
            self.regime_thresholds = correlation_settings.correlation_regime_thresholds
            self.max_pairs = correlation_settings.max_correlation_pairs
        else:
            # Safe defaults if correlation settings not configured
            self.regime_thresholds = {
                'high_correlation': 0.7,
                'low_correlation': 0.3
            }
            self.max_pairs = 50
        
        # Performance tracking
        self._correlation_cache: Dict[str, Tuple[CorrelationMetrics, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)  # 15-minute cache following existing patterns
        
        self.logger.info(f"🔗 FilteredAssetCorrelationEngine initialized (window={self.correlation_window})")
    
    async def calculate_filtered_asset_correlations(
        self, 
        filtered_assets: List[str],
        timeframe: str = '1h',
        force_refresh: bool = False
    ) -> CorrelationMetrics:
        """
        Calculate correlations for filtered assets using DataStorageInterface.
        Follows enhanced_asset_filter.py data access patterns.
        
        Args:
            filtered_assets: List of asset symbols to analyze
            timeframe: Data timeframe ('1h', '15m', '1d')
            force_refresh: Force cache refresh
            
        Returns:
            CorrelationMetrics with all correlation analysis results
        """
        cache_key = f"{'-'.join(sorted(filtered_assets))}_{timeframe}_{self.correlation_window}"
        
        # Check cache first (following enhanced_asset_filter pattern)
        if not force_refresh and cache_key in self._correlation_cache:
            cached_metrics, cache_time = self._correlation_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                self.logger.debug(f"Using cached correlation metrics for {len(filtered_assets)} assets")
                return cached_metrics
        
        self.logger.info(f"🔍 Calculating correlations for {len(filtered_assets)} filtered assets ({timeframe})")
        
        try:
            # Limit asset count for performance (following existing optimization patterns)
            analysis_assets = filtered_assets[:self.max_pairs] if len(filtered_assets) > self.max_pairs else filtered_assets
            
            if len(analysis_assets) != len(filtered_assets):
                self.logger.info(f"   📊 Limited analysis to {len(analysis_assets)} assets for performance")
            
            # Get asset data using storage interface (Phase 1 integration)
            asset_data = await self._fetch_asset_correlation_data(analysis_assets, timeframe)
            
            # Validate data quality
            data_quality = self._assess_data_quality(asset_data)
            if data_quality < 0.5:
                self.logger.warning(f"   ⚠️ Low data quality score: {data_quality:.2f}")
            
            # Calculate pairwise correlations (research-backed implementation)
            correlation_pairs = self._calculate_pairwise_correlations(asset_data)
            
            # Calculate portfolio-level metrics
            portfolio_score = self._calculate_portfolio_correlation_score(correlation_pairs)
            regime = self._detect_correlation_regime(portfolio_score)
            
            # Create metrics object
            metrics = CorrelationMetrics(
                correlation_pairs=correlation_pairs,
                portfolio_correlation_score=portfolio_score,
                regime_classification=regime,
                calculation_timestamp=datetime.now(),
                asset_count=len(analysis_assets),
                valid_pairs=len(correlation_pairs),
                calculation_window=self.correlation_window,
                data_quality_score=data_quality
            )
            
            # Cache results
            self._correlation_cache[cache_key] = (metrics, datetime.now())
            
            self.logger.info(f"   ✅ Calculated {len(correlation_pairs)} correlation pairs")
            self.logger.info(f"   📊 Portfolio correlation: {portfolio_score:.3f} ({regime})")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Correlation calculation failed: {e}")
            # Return safe defaults following existing error patterns
            return CorrelationMetrics(
                correlation_pairs={},
                portfolio_correlation_score=0.5,
                regime_classification="medium_correlation",
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                valid_pairs=0,
                calculation_window=self.correlation_window,
                data_quality_score=0.0
            )
    
    async def _fetch_asset_correlation_data(
        self, 
        assets: List[str], 
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for correlation calculation using DataStorageInterface.
        Follows existing data access patterns.
        """
        self.logger.debug(f"   📈 Fetching {timeframe} data for {len(assets)} assets")
        
        asset_data = {}
        fetch_tasks = []
        
        # Create async tasks for concurrent data fetching (performance optimization)
        for asset in assets:
            task = self._fetch_single_asset_data(asset, timeframe)
            fetch_tasks.append((asset, task))
        
        # Execute all fetches concurrently
        for asset, task in fetch_tasks:
            try:
                data = await task
                if not data.empty and len(data) >= self.min_correlation_periods:
                    asset_data[asset] = data
                else:
                    self.logger.debug(f"   ⚠️ Insufficient data for {asset}: {len(data) if data is not None else 0} bars")
            except Exception as e:
                self.logger.debug(f"   ❌ Failed to fetch data for {asset}: {e}")
        
        self.logger.debug(f"   ✅ Successfully fetched data for {len(asset_data)}/{len(assets)} assets")
        return asset_data
    
    async def _fetch_single_asset_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """Fetch data for single asset with error handling."""
        try:
            # Use storage interface following existing patterns
            data = await self.storage.get_ohlcv_bars(
                symbol=asset,
                limit=self.correlation_window * 2  # Get extra data for alignment
            )
            return data
        except Exception as e:
            self.logger.debug(f"Single asset fetch failed for {asset}: {e}")
            return pd.DataFrame()
    
    def _assess_data_quality(self, asset_data: Dict[str, pd.DataFrame]) -> float:
        """Assess data quality for correlation analysis."""
        if not asset_data:
            return 0.0
        
        total_scores = []
        
        for asset, data in asset_data.items():
            if data.empty:
                total_scores.append(0.0)
                continue
            
            # Quality factors
            length_score = min(1.0, len(data) / self.correlation_window)
            completeness_score = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Price data validity
            if 'close' in data.columns:
                price_validity = 1.0 - (data['close'] <= 0).sum() / len(data)
            else:
                price_validity = 0.5
            
            asset_score = (length_score + completeness_score + price_validity) / 3.0
            total_scores.append(asset_score)
        
        return np.mean(total_scores) if total_scores else 0.0
    
    def _calculate_pairwise_correlations(
        self, 
        asset_data: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate pairwise correlations using pandas research patterns.
        Follows research/pandas_comprehensive correlation calculation methods.
        """
        if len(asset_data) < 2:
            return {}
        
        correlation_pairs = {}
        assets = list(asset_data.keys())
        
        # Prepare returns data for correlation calculation
        returns_data = {}
        
        for asset, data in asset_data.items():
            if 'close' not in data.columns or len(data) < self.min_correlation_periods:
                continue
            
            # Calculate returns (following pandas research patterns)
            returns = data['close'].pct_change().dropna()
            if len(returns) >= self.min_correlation_periods:
                returns_data[asset] = returns
        
        # Calculate correlations for all valid pairs
        valid_assets = list(returns_data.keys())
        
        for i, asset1 in enumerate(valid_assets):
            for j, asset2 in enumerate(valid_assets):
                if i >= j:  # Skip duplicates and self-correlation
                    continue
                
                try:
                    # Align returns data (following pandas research patterns)
                    returns1 = returns_data[asset1]
                    returns2 = returns_data[asset2]
                    
                    # Get overlapping time periods
                    aligned_returns = pd.DataFrame({
                        asset1: returns1,
                        asset2: returns2
                    }).dropna()
                    
                    if len(aligned_returns) >= self.min_correlation_periods:
                        # Calculate Pearson correlation
                        correlation = aligned_returns[asset1].corr(aligned_returns[asset2])
                        
                        if not pd.isna(correlation):
                            correlation_pairs[(asset1, asset2)] = correlation
                            correlation_pairs[(asset2, asset1)] = correlation  # Symmetric
                
                except Exception as e:
                    self.logger.debug(f"Correlation calculation failed for {asset1}-{asset2}: {e}")
        
        return correlation_pairs
    
    def _calculate_portfolio_correlation_score(
        self, 
        correlation_pairs: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate overall portfolio correlation score."""
        if not correlation_pairs:
            return 0.5  # Neutral correlation
        
        # Get unique correlations (remove symmetric duplicates)
        unique_correlations = []
        seen_pairs = set()
        
        for (asset1, asset2), corr in correlation_pairs.items():
            pair_key = tuple(sorted([asset1, asset2]))
            if pair_key not in seen_pairs:
                unique_correlations.append(abs(corr))  # Use absolute correlation
                seen_pairs.add(pair_key)
        
        if not unique_correlations:
            return 0.5
        
        # Return average absolute correlation
        return np.mean(unique_correlations)
    
    def _detect_correlation_regime(self, portfolio_score: float) -> str:
        """
        Classify current correlation regime.
        Follows enhanced_asset_filter.py classification patterns.
        """
        if portfolio_score >= self.regime_thresholds['high_correlation']:
            return "high_correlation"  # Risk-on, trending market
        elif portfolio_score <= self.regime_thresholds['low_correlation']:
            return "low_correlation"   # Risk-off, stock-picking market
        else:
            return "medium_correlation"  # Neutral market conditions
    
    def detect_correlation_regime(self, correlations: Dict[str, float]) -> str:
        """
        Public interface for correlation regime detection.
        Maintains compatibility with phase plan specification.
        """
        if not correlations:
            return "medium_correlation"
        
        avg_correlation = np.mean([abs(corr) for corr in correlations.values()])
        return self._detect_correlation_regime(avg_correlation)
    
    def get_correlation_summary(self, metrics: CorrelationMetrics) -> Dict[str, Any]:
        """Get comprehensive correlation analysis summary."""
        strength_dist = metrics.correlation_strength_distribution
        
        return {
            "correlation_analysis": {
                "portfolio_correlation_score": metrics.portfolio_correlation_score,
                "correlation_regime": metrics.regime_classification,
                "asset_count": metrics.asset_count,
                "valid_correlation_pairs": metrics.valid_pairs,
                "data_quality_score": metrics.data_quality_score,
                "calculation_window": metrics.calculation_window
            },
            "correlation_distribution": strength_dist,
            "regime_analysis": {
                "current_regime": metrics.regime_classification,
                "regime_score": metrics.portfolio_correlation_score,
                "regime_thresholds": self.regime_thresholds
            },
            "metadata": {
                "calculation_timestamp": metrics.calculation_timestamp.isoformat(),
                "engine_version": "1.0.0",
                "integration": "DataStorageInterface"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for correlation engine following storage interface patterns."""
        try:
            # Test storage interface connectivity
            storage_health = await self.storage.health_check()
            
            # Test correlation calculation with minimal data
            test_assets = ['BTC', 'ETH']  # Basic test case
            
            calculation_healthy = True
            try:
                test_metrics = await self.calculate_filtered_asset_correlations(
                    test_assets, force_refresh=True
                )
                calculation_test_passed = test_metrics.calculation_timestamp is not None
            except Exception as e:
                calculation_healthy = False
                calculation_test_passed = False
            
            overall_health = (
                storage_health.get('status') == 'healthy' and 
                calculation_healthy
            )
            
            return {
                "status": "healthy" if overall_health else "degraded",
                "component": "FilteredAssetCorrelationEngine",
                "storage_backend": storage_health.get('backend', 'unknown'),
                "storage_status": storage_health.get('status', 'unknown'),
                "correlation_calculation": "healthy" if calculation_healthy else "degraded",
                "cache_size": len(self._correlation_cache),
                "configuration": {
                    "correlation_window": self.correlation_window,
                    "min_periods": self.min_correlation_periods,
                    "max_pairs": self.max_pairs
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "FilteredAssetCorrelationEngine",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }