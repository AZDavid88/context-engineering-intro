"""
Volume Regime Detection for Multi-Source Market Analysis

Detects volume regimes from existing OHLCV volume data following DataStorageInterface patterns.
Analyzes market participation patterns without requiring additional data collection.

Key Features:
- Portfolio volume pattern analysis from filtered assets
- Market participation regime classification (low/normal/high volume)
- Integration with DataStorageInterface for consistency
- Volume trend and momentum analysis
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
class VolumeRegimeMetrics:
    """Volume regime analysis metrics following existing patterns."""
    volume_regime: str = "normal_volume"
    current_volume_ratio: float = 1.0
    volume_trend: str = "stable"
    volume_momentum: float = 0.0
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    asset_count: int = 0
    calculation_window: int = 20
    data_quality_score: float = 0.0
    
    @property
    def is_high_participation(self) -> bool:
        """Check if market has high participation (high volume)."""
        return self.volume_regime == "high_volume"
    
    @property
    def is_low_participation(self) -> bool:
        """Check if market has low participation (low volume)."""
        return self.volume_regime == "low_volume"
    
    @property
    def is_volume_increasing(self) -> bool:
        """Check if volume is trending upward."""
        return self.volume_trend == "increasing"


class VolumeRegimeDetector:
    """
    Detect volume regimes from existing OHLCV volume data.
    Follows volatility_regime_detector.py patterns for architectural consistency.
    """
    
    def __init__(self, settings=None):
        """Initialize volume detector with existing configuration patterns."""
        self.settings = settings or get_settings()
        self.storage = get_storage_implementation()  # Phase 1 integration
        self.logger = logging.getLogger(__name__)
        
        # Volume configuration following existing settings pattern
        self.volume_window = getattr(self.settings, 'volume_window_periods', 20)
        self.min_data_points = getattr(self.settings, 'min_volume_data_points', 30)
        
        # Regime detection thresholds following settings pattern
        regime_settings = getattr(self.settings, 'regime_detection', None)
        if regime_settings:
            volume_config = getattr(regime_settings, 'volume_thresholds', {})
            self.regime_thresholds = {
                'low_volume': volume_config.get('low_volume', 0.7),    # <70% of average
                'high_volume': volume_config.get('high_volume', 1.3)   # >130% of average
            }
        else:
            # Safe defaults matching phase plan specifications
            self.regime_thresholds = {
                'low_volume': 0.7,     # <70% of average volume - quiet market
                'high_volume': 1.3     # >130% of average volume - active market
            }
        
        # Performance tracking following existing cache patterns
        self._volume_cache: Dict[str, tuple[VolumeRegimeMetrics, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)  # Match other detectors
        
        self.logger.info(f"📊 VolumeRegimeDetector initialized (window={self.volume_window})")
    
    async def detect_volume_regime(
        self,
        filtered_assets: List[str],
        timeframe: str = '1h',
        force_refresh: bool = False
    ) -> VolumeRegimeMetrics:
        """
        Detect volume regime for filtered assets using DataStorageInterface.
        Follows volatility_regime_detector.py patterns for consistency.
        
        Args:
            filtered_assets: List of asset symbols to analyze
            timeframe: Data timeframe ('1h', '15m', '1d')
            force_refresh: Force cache refresh
            
        Returns:
            VolumeRegimeMetrics with regime classification and analysis
        """
        cache_key = f"{'-'.join(sorted(filtered_assets))}_{timeframe}_{self.volume_window}"
        
        # Check cache first
        if not force_refresh and cache_key in self._volume_cache:
            cached_metrics, cache_time = self._volume_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                self.logger.debug(f"Using cached volume metrics for {len(filtered_assets)} assets")
                return cached_metrics
        
        self.logger.info(f"📊 Calculating volume regime for {len(filtered_assets)} filtered assets ({timeframe})")
        
        try:
            # Get asset data using storage interface
            asset_data = await self._fetch_asset_volume_data(filtered_assets, timeframe)
            
            # Validate data quality
            data_quality = self._assess_data_quality(asset_data)
            if data_quality < 0.5:
                self.logger.warning(f"   ⚠️ Low data quality score: {data_quality:.2f}")
            
            # Calculate portfolio volume indicators
            volume_ratio = self._calculate_portfolio_volume_ratio(asset_data)
            volume_trend = self._analyze_volume_trend(asset_data)
            volume_momentum = self._calculate_volume_momentum(asset_data)
            
            # Classify volume regime
            regime = self._classify_volume_regime(volume_ratio)
            
            # Create metrics object
            metrics = VolumeRegimeMetrics(
                volume_regime=regime,
                current_volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                volume_momentum=volume_momentum,
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                calculation_window=self.volume_window,
                data_quality_score=data_quality
            )
            
            # Cache results
            self._volume_cache[cache_key] = (metrics, datetime.now())
            
            self.logger.info(f"   ✅ Volume regime: {regime} (ratio: {volume_ratio:.2f})")
            self.logger.info(f"   📈 Volume trend: {volume_trend} (momentum: {volume_momentum:+.2f})")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Volume regime detection failed: {e}")
            # Return safe defaults
            return VolumeRegimeMetrics(
                volume_regime="normal_volume",
                current_volume_ratio=1.0,
                volume_trend="stable",
                volume_momentum=0.0,
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                calculation_window=self.volume_window,
                data_quality_score=0.0
            )
    
    async def _fetch_asset_volume_data(
        self, 
        assets: List[str], 
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for volume analysis using DataStorageInterface.
        Follows existing data access patterns.
        """
        self.logger.debug(f"   📈 Fetching {timeframe} data for {len(assets)} assets")
        
        asset_data = {}
        
        # Fetch data for each asset with error handling
        for asset in assets:
            try:
                # Use storage interface following existing patterns
                data = await self.storage.get_ohlcv_bars(
                    symbol=asset,
                    limit=self.volume_window * 2  # Get extra data for calculations
                )
                
                if not data.empty and len(data) >= self.min_data_points and 'volume' in data.columns:
                    asset_data[asset] = data
                else:
                    self.logger.debug(f"   ⚠️ Insufficient volume data for {asset}: {len(data) if not data.empty else 0} bars")
                    
            except Exception as e:
                self.logger.debug(f"   ❌ Failed to fetch data for {asset}: {e}")
        
        self.logger.debug(f"   ✅ Successfully fetched volume data for {len(asset_data)}/{len(assets)} assets")
        return asset_data
    
    def _assess_data_quality(self, asset_data: Dict[str, pd.DataFrame]) -> float:
        """Assess data quality for volume analysis."""
        if not asset_data:
            return 0.0
        
        total_scores = []
        
        for asset, data in asset_data.items():
            if data.empty:
                total_scores.append(0.0)
                continue
            
            # Quality factors
            length_score = min(1.0, len(data) / self.volume_window)
            completeness_score = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Volume data validity
            if 'volume' in data.columns:
                # Check for non-negative volumes
                volume_validity = 1.0 - (data['volume'] < 0).sum() / len(data)
                # Check for non-zero volumes (some periods may have zero volume)
                non_zero_volumes = (data['volume'] > 0).sum()
                volume_availability = non_zero_volumes / len(data) if len(data) > 0 else 0.0
            else:
                volume_validity = 0.0
                volume_availability = 0.0
            
            asset_score = (length_score + completeness_score + volume_validity + volume_availability) / 4.0
            total_scores.append(asset_score)
        
        return np.mean(total_scores) if total_scores else 0.0
    
    def _calculate_portfolio_volume_ratio(
        self, 
        asset_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calculate portfolio volume ratio vs historical average.
        Follows pandas research patterns for volume analysis.
        """
        if not asset_data:
            return 1.0  # Default neutral ratio
        
        volume_ratios = []
        
        for asset, data in asset_data.items():
            if 'volume' not in data.columns or len(data) < self.volume_window:
                continue
            
            volume_data = data['volume']
            
            # Skip if insufficient data or all zeros
            if len(volume_data) < self.volume_window or volume_data.sum() == 0:
                continue
            
            # Calculate rolling average and current ratio
            volume_ma = volume_data.rolling(self.volume_window).mean()
            if len(volume_ma) > 0 and volume_ma.iloc[-1] > 0:
                current_volume = volume_data.iloc[-1]
                volume_ratio = current_volume / volume_ma.iloc[-1]
                volume_ratios.append(volume_ratio)
        
        if not volume_ratios:
            return 1.0  # Default if no valid data
        
        # Return average volume ratio across portfolio
        portfolio_ratio = np.mean(volume_ratios)
        
        # Bound the ratio to prevent extreme values
        return max(0.1, min(5.0, portfolio_ratio))
    
    def _analyze_volume_trend(
        self, 
        asset_data: Dict[str, pd.DataFrame]
    ) -> str:
        """Analyze volume trend direction across portfolio."""
        if not asset_data:
            return "stable"
        
        trend_signals = []
        
        for asset, data in asset_data.items():
            if 'volume' not in data.columns or len(data) < self.volume_window:
                continue
            
            volume_data = data['volume']
            
            if len(volume_data) < self.volume_window:
                continue
            
            # Calculate short and long volume averages
            short_window = max(5, self.volume_window // 4)
            long_window = self.volume_window
            
            short_avg = volume_data.rolling(short_window).mean().iloc[-1]
            long_avg = volume_data.rolling(long_window).mean().iloc[-1]
            
            if short_avg > 0 and long_avg > 0:
                ratio = short_avg / long_avg
                if ratio > 1.1:  # 10% higher
                    trend_signals.append(1)  # Increasing
                elif ratio < 0.9:  # 10% lower
                    trend_signals.append(-1)  # Decreasing
                else:
                    trend_signals.append(0)  # Stable
        
        if not trend_signals:
            return "stable"
        
        # Determine overall trend
        avg_trend = np.mean(trend_signals)
        if avg_trend > 0.2:
            return "increasing"
        elif avg_trend < -0.2:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volume_momentum(
        self, 
        asset_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate volume momentum indicator."""
        if not asset_data:
            return 0.0
        
        momentum_values = []
        
        for asset, data in asset_data.items():
            if 'volume' not in data.columns or len(data) < self.volume_window:
                continue
            
            volume_data = data['volume']
            
            if len(volume_data) < 10:  # Need minimum data for momentum
                continue
            
            # Calculate volume momentum as rate of change
            recent_avg = volume_data.rolling(5).mean().iloc[-1]
            previous_avg = volume_data.rolling(5).mean().iloc[-6] if len(volume_data) >= 10 else recent_avg
            
            if previous_avg > 0:
                momentum = (recent_avg - previous_avg) / previous_avg
                momentum_values.append(momentum)
        
        if not momentum_values:
            return 0.0
        
        # Return average momentum, bounded to reasonable range
        avg_momentum = np.mean(momentum_values)
        return max(-1.0, min(1.0, avg_momentum))
    
    def _classify_volume_regime(self, volume_ratio: float) -> str:
        """
        Classify volume regime based on volume ratio.
        Follows existing classification patterns.
        """
        if volume_ratio <= self.regime_thresholds['low_volume']:
            return "low_volume"      # Quiet market, low participation
        elif volume_ratio >= self.regime_thresholds['high_volume']:
            return "high_volume"     # Active market, high participation
        else:
            return "normal_volume"   # Average market participation
    
    def get_volume_regime_summary(self, metrics: VolumeRegimeMetrics) -> Dict[str, Any]:
        """Get comprehensive volume regime analysis summary."""
        return {
            "volume_regime_analysis": {
                "volume_regime": metrics.volume_regime,
                "current_volume_ratio": metrics.current_volume_ratio,
                "volume_trend": metrics.volume_trend,
                "volume_momentum": metrics.volume_momentum,
                "asset_count": metrics.asset_count,
                "data_quality_score": metrics.data_quality_score,
                "calculation_window": metrics.calculation_window
            },
            "market_participation": {
                "is_high_participation": metrics.is_high_participation,
                "is_low_participation": metrics.is_low_participation,
                "is_volume_increasing": metrics.is_volume_increasing,
                "participation_description": self._get_participation_description(metrics.volume_regime)
            },
            "regime_thresholds": self.regime_thresholds,
            "metadata": {
                "calculation_timestamp": metrics.calculation_timestamp.isoformat(),
                "detector_version": "1.0.0",
                "integration": "DataStorageInterface"
            }
        }
    
    def _get_participation_description(self, regime: str) -> str:
        """Get human-readable market participation description."""
        descriptions = {
            "low_volume": "Low market participation - institutional or thin market conditions",
            "normal_volume": "Normal market participation - balanced trading activity",
            "high_volume": "High market participation - active trading, possible news/events"
        }
        return descriptions.get(regime, "Unknown volume environment")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for volume detector following storage interface patterns."""
        try:
            # Test storage interface connectivity
            storage_health = await self.storage.health_check()
            
            # Test volume calculation with minimal data
            test_assets = ['BTC', 'ETH']
            
            calculation_healthy = True
            try:
                test_metrics = await self.detect_volume_regime(
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
                "component": "VolumeRegimeDetector",
                "storage_backend": storage_health.get('backend', 'unknown'),
                "storage_status": storage_health.get('status', 'unknown'),
                "volume_calculation": "healthy" if calculation_healthy else "degraded",
                "cache_size": len(self._volume_cache),
                "configuration": {
                    "volume_window": self.volume_window,
                    "min_data_points": self.min_data_points,
                    "regime_thresholds": self.regime_thresholds
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "VolumeRegimeDetector",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }