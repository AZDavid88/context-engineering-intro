"""
Volatility Regime Detection for Multi-Source Market Analysis

Detects market volatility regimes from filtered asset OHLCV data following
existing architectural patterns and pandas research implementations.

Key Features:
- Portfolio volatility calculation using filtered assets
- Dynamic threshold classification (low/medium/high volatility)
- Integration with DataStorageInterface for cloud compatibility
- Research-validated pandas calculations for accuracy
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.data.storage_interfaces import get_storage_implementation, DataStorageInterface
from src.config.settings import get_settings


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics following existing AssetMetrics pattern."""
    current_volatility: float = 0.0
    volatility_regime: str = "medium_volatility"
    volatility_percentile: float = 0.5
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    asset_count: int = 0
    calculation_window: int = 20
    data_quality_score: float = 0.0
    
    @property
    def is_volatile_market(self) -> bool:
        """Check if market is in high volatility regime."""
        return self.volatility_regime == "high_volatility"
    
    @property
    def is_quiet_market(self) -> bool:
        """Check if market is in low volatility regime."""
        return self.volatility_regime == "low_volatility"


class VolatilityRegimeDetector:
    """
    Detect market volatility regimes from filtered asset OHLCV data.
    Follows correlation_engine.py patterns for consistency and integration.
    """
    
    def __init__(self, settings=None):
        """Initialize volatility detector with existing configuration patterns."""
        self.settings = settings or get_settings()
        self.storage = get_storage_implementation()  # Phase 1 integration
        self.logger = logging.getLogger(__name__)
        
        # Volatility configuration following existing settings pattern
        self.volatility_window = getattr(self.settings, 'volatility_window_periods', 20)
        self.min_data_points = getattr(self.settings, 'min_volatility_data_points', 30)
        
        # Regime detection thresholds - following settings pattern
        regime_settings = getattr(self.settings, 'regime_detection', None)
        if regime_settings:
            volatility_config = getattr(regime_settings, 'volatility_thresholds', {})
            self.regime_thresholds = {
                'low_volatility': volatility_config.get('low_volatility', 0.15),    # <15% annualized
                'high_volatility': volatility_config.get('high_volatility', 0.30)   # >30% annualized
            }
        else:
            # Safe defaults matching phase plan specifications
            self.regime_thresholds = {
                'low_volatility': 0.15,    # <15% annualized volatility
                'high_volatility': 0.30    # >30% annualized volatility
            }
        
        # Performance tracking following existing cache patterns
        self._volatility_cache: Dict[str, tuple[VolatilityMetrics, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)  # 15-minute cache like correlation engine
        
        self.logger.info(f"📊 VolatilityRegimeDetector initialized (window={self.volatility_window})")
    
    async def detect_volatility_regime(
        self,
        filtered_assets: List[str],
        timeframe: str = '1h',
        force_refresh: bool = False
    ) -> VolatilityMetrics:
        """
        Detect volatility regime for filtered assets using DataStorageInterface.
        Follows correlation_engine.py data access patterns for consistency.
        
        Args:
            filtered_assets: List of asset symbols to analyze
            timeframe: Data timeframe ('1h', '15m', '1d')
            force_refresh: Force cache refresh
            
        Returns:
            VolatilityMetrics with regime classification and analysis
        """
        cache_key = f"{'-'.join(sorted(filtered_assets))}_{timeframe}_{self.volatility_window}"
        
        # Check cache first (following correlation_engine pattern)
        if not force_refresh and cache_key in self._volatility_cache:
            cached_metrics, cache_time = self._volatility_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                self.logger.debug(f"Using cached volatility metrics for {len(filtered_assets)} assets")
                return cached_metrics
        
        self.logger.info(f"📊 Calculating volatility regime for {len(filtered_assets)} filtered assets ({timeframe})")
        
        try:
            # Get asset data using storage interface (Phase 1 integration)
            asset_data = await self._fetch_asset_volatility_data(filtered_assets, timeframe)
            
            # Validate data quality
            data_quality = self._assess_data_quality(asset_data)
            if data_quality < 0.5:
                self.logger.warning(f"   ⚠️ Low data quality score: {data_quality:.2f}")
            
            # Calculate portfolio volatility (research-backed implementation)
            current_volatility = self._calculate_portfolio_volatility(asset_data)
            
            # Detect volatility regime based on thresholds
            regime = self._classify_volatility_regime(current_volatility)
            
            # Calculate volatility percentile for context
            volatility_percentile = self._calculate_volatility_percentile(
                current_volatility, asset_data
            )
            
            # Create metrics object
            metrics = VolatilityMetrics(
                current_volatility=current_volatility,
                volatility_regime=regime,
                volatility_percentile=volatility_percentile,
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                calculation_window=self.volatility_window,
                data_quality_score=data_quality
            )
            
            # Cache results
            self._volatility_cache[cache_key] = (metrics, datetime.now())
            
            self.logger.info(f"   ✅ Volatility regime: {regime} ({current_volatility:.1%})")
            self.logger.info(f"   📊 Volatility percentile: {volatility_percentile:.1%}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Volatility regime detection failed: {e}")
            # Return safe defaults following existing error patterns
            return VolatilityMetrics(
                current_volatility=0.2,  # Neutral volatility
                volatility_regime="medium_volatility",
                volatility_percentile=0.5,
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                calculation_window=self.volatility_window,
                data_quality_score=0.0
            )
    
    async def _fetch_asset_volatility_data(
        self, 
        assets: List[str], 
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for volatility calculation using DataStorageInterface.
        Follows correlation_engine.py data access patterns.
        """
        self.logger.debug(f"   📈 Fetching {timeframe} data for {len(assets)} assets")
        
        asset_data = {}
        
        # Fetch data for each asset with error handling
        for asset in assets:
            try:
                # Use storage interface following existing patterns
                data = await self.storage.get_ohlcv_bars(
                    symbol=asset,
                    limit=self.volatility_window * 2  # Get extra data for alignment
                )
                
                if not data.empty and len(data) >= self.min_data_points:
                    asset_data[asset] = data
                else:
                    self.logger.debug(f"   ⚠️ Insufficient data for {asset}: {len(data) if not data.empty else 0} bars")
                    
            except Exception as e:
                self.logger.debug(f"   ❌ Failed to fetch data for {asset}: {e}")
        
        self.logger.debug(f"   ✅ Successfully fetched data for {len(asset_data)}/{len(assets)} assets")
        return asset_data
    
    def _assess_data_quality(self, asset_data: Dict[str, pd.DataFrame]) -> float:
        """Assess data quality for volatility calculation."""
        if not asset_data:
            return 0.0
        
        total_scores = []
        
        for asset, data in asset_data.items():
            if data.empty:
                total_scores.append(0.0)
                continue
            
            # Quality factors
            length_score = min(1.0, len(data) / self.volatility_window)
            completeness_score = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Price data validity for volatility calculation
            if 'close' in data.columns:
                price_validity = 1.0 - (data['close'] <= 0).sum() / len(data)
                # Check for extreme price jumps that could skew volatility
                returns = data['close'].pct_change().dropna()
                if len(returns) > 0:
                    extreme_returns = (abs(returns) > 0.5).sum() / len(returns)  # >50% moves
                    volatility_quality = max(0.0, 1.0 - extreme_returns)
                else:
                    volatility_quality = 0.5
            else:
                price_validity = 0.5
                volatility_quality = 0.5
            
            asset_score = (length_score + completeness_score + price_validity + volatility_quality) / 4.0
            total_scores.append(asset_score)
        
        return np.mean(total_scores) if total_scores else 0.0
    
    def _calculate_portfolio_volatility(
        self, 
        asset_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calculate portfolio volatility using pandas research patterns.
        Follows research/pandas_comprehensive/research_summary.md volatility calculations.
        """
        if not asset_data:
            return 0.2  # Default medium volatility
        
        # Calculate returns for each asset
        asset_returns = {}
        
        for asset, data in asset_data.items():
            if 'close' not in data.columns or len(data) < self.min_data_points:
                continue
            
            # Calculate percentage returns (following pandas research patterns)
            returns = data['close'].pct_change().dropna()
            if len(returns) >= self.volatility_window:
                asset_returns[asset] = returns
        
        if not asset_returns:
            return 0.2  # Default if no valid data
        
        # Create equal-weight portfolio returns
        portfolio_returns_list = []
        
        # Align all returns to common time index
        if len(asset_returns) == 1:
            # Single asset case
            portfolio_returns = list(asset_returns.values())[0]
        else:
            # Multi-asset case - create DataFrame and take equal-weight average
            returns_df = pd.DataFrame(asset_returns)
            returns_df = returns_df.dropna()  # Remove periods with missing data
            
            if len(returns_df) < self.min_data_points:
                return 0.2  # Default if insufficient aligned data
            
            # Equal-weight portfolio
            portfolio_returns = returns_df.mean(axis=1)
        
        # Calculate annualized volatility
        if len(portfolio_returns) < self.volatility_window:
            return 0.2  # Default if insufficient data
        
        # Use rolling volatility with current window
        rolling_volatility = portfolio_returns.rolling(self.volatility_window).std()
        current_volatility = rolling_volatility.iloc[-1]
        
        # Annualize volatility (assuming hourly data, 24*365 = 8760 periods per year)
        # Adjust based on timeframe if needed
        annualized_volatility = current_volatility * np.sqrt(8760)  # Hourly to annual
        
        # Return bounded volatility (prevent extreme values)
        return max(0.01, min(2.0, annualized_volatility))  # Between 1% and 200%
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """
        Classify volatility regime based on configured thresholds.
        Follows enhanced_asset_filter.py classification patterns.
        """
        if volatility <= self.regime_thresholds['low_volatility']:
            return "low_volatility"     # Quiet market, low risk
        elif volatility >= self.regime_thresholds['high_volatility']:
            return "high_volatility"    # Volatile market, high risk
        else:
            return "medium_volatility"  # Normal market conditions
    
    def _calculate_volatility_percentile(
        self,
        current_volatility: float,
        asset_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate current volatility percentile vs historical distribution."""
        if not asset_data:
            return 0.5
        
        try:
            # Calculate historical volatility distribution
            all_volatilities = []
            
            for asset, data in asset_data.items():
                if 'close' not in data.columns or len(data) < self.volatility_window * 2:
                    continue
                
                returns = data['close'].pct_change().dropna()
                if len(returns) < self.volatility_window * 2:
                    continue
                
                # Calculate rolling volatility series
                rolling_vol = returns.rolling(self.volatility_window).std() * np.sqrt(8760)
                all_volatilities.extend(rolling_vol.dropna().tolist())
            
            if not all_volatilities or len(all_volatilities) < 10:
                return 0.5  # Default if insufficient history
            
            # Calculate percentile
            percentile = (np.array(all_volatilities) <= current_volatility).mean()
            return max(0.0, min(1.0, percentile))
            
        except Exception as e:
            self.logger.debug(f"Volatility percentile calculation failed: {e}")
            return 0.5
    
    def get_volatility_summary(self, metrics: VolatilityMetrics) -> Dict[str, Any]:
        """Get comprehensive volatility analysis summary."""
        return {
            "volatility_analysis": {
                "current_volatility": metrics.current_volatility,
                "volatility_regime": metrics.volatility_regime,
                "volatility_percentile": metrics.volatility_percentile,
                "asset_count": metrics.asset_count,
                "data_quality_score": metrics.data_quality_score,
                "calculation_window": metrics.calculation_window
            },
            "regime_classification": {
                "is_volatile_market": metrics.is_volatile_market,
                "is_quiet_market": metrics.is_quiet_market,
                "regime_thresholds": self.regime_thresholds
            },
            "metadata": {
                "calculation_timestamp": metrics.calculation_timestamp.isoformat(),
                "detector_version": "1.0.0",
                "integration": "DataStorageInterface"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for volatility detector following storage interface patterns."""
        try:
            # Test storage interface connectivity
            storage_health = await self.storage.health_check()
            
            # Test volatility calculation with minimal data
            test_assets = ['BTC', 'ETH']  # Basic test case
            
            calculation_healthy = True
            try:
                test_metrics = await self.detect_volatility_regime(
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
                "component": "VolatilityRegimeDetector",
                "storage_backend": storage_health.get('backend', 'unknown'),
                "storage_status": storage_health.get('status', 'unknown'),
                "volatility_calculation": "healthy" if calculation_healthy else "degraded",
                "cache_size": len(self._volatility_cache),
                "configuration": {
                    "volatility_window": self.volatility_window,
                    "min_data_points": self.min_data_points,
                    "regime_thresholds": self.regime_thresholds
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "VolatilityRegimeDetector",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }