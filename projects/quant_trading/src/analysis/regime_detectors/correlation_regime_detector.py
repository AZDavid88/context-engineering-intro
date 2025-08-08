"""
Correlation Regime Detection for Multi-Source Market Analysis

Detects correlation regimes using Phase 2 correlation analysis integration.
Directly integrates with correlation_engine.py following existing patterns.

Key Features:
- Direct integration with FilteredAssetCorrelationEngine
- Market correlation state classification (breakdown/normal/high)
- Portfolio-level correlation regime analysis
- Seamless Phase 2 correlation engine integration
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.analysis.correlation_engine import FilteredAssetCorrelationEngine, CorrelationMetrics
from src.config.settings import get_settings


@dataclass
class CorrelationRegimeMetrics:
    """Correlation regime analysis metrics."""
    correlation_regime: str = "normal_correlation"
    average_correlation: float = 0.5
    correlation_strength_distribution: Dict[str, int] = field(default_factory=dict)
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    asset_count: int = 0
    valid_correlation_pairs: int = 0
    data_quality_score: float = 0.0
    
    @property
    def is_correlation_breakdown(self) -> bool:
        """Check if market is in correlation breakdown (stock-picking environment)."""
        return self.correlation_regime == "correlation_breakdown"
    
    @property
    def is_high_correlation(self) -> bool:
        """Check if market is in high correlation (risk-on/risk-off environment)."""
        return self.correlation_regime == "high_correlation"
    
    @property
    def is_diversification_effective(self) -> bool:
        """Check if diversification is likely effective (correlation_breakdown/normal)."""
        return self.correlation_regime in ["correlation_breakdown", "normal_correlation"]


class CorrelationRegimeDetector:
    """
    Detect correlation regimes using Phase 2 correlation analysis.
    Integrates directly with FilteredAssetCorrelationEngine for seamless operation.
    """
    
    def __init__(self, correlation_engine: Optional[FilteredAssetCorrelationEngine] = None, settings=None):
        """Initialize correlation regime detector with existing correlation engine."""
        self.settings = settings or get_settings()
        self.correlation_engine = correlation_engine or FilteredAssetCorrelationEngine(self.settings)
        self.logger = logging.getLogger(__name__)
        
        # Use correlation engine thresholds directly (upstream standard)
        self.regime_thresholds = self.correlation_engine.regime_thresholds
        
        # Performance tracking
        self._regime_cache: Dict[str, tuple[CorrelationRegimeMetrics, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)  # Match correlation engine cache
        
        self.logger.info(f"🔗 CorrelationRegimeDetector initialized with correlation engine")
    
    async def detect_correlation_regime(
        self,
        filtered_assets: List[str],
        timeframe: str = '1h',
        force_refresh: bool = False
    ) -> CorrelationRegimeMetrics:
        """
        Detect correlation regime using existing correlation engine.
        Leverages Phase 2 correlation analysis for regime classification.
        
        Args:
            filtered_assets: List of asset symbols to analyze
            timeframe: Data timeframe ('1h', '15m', '1d')
            force_refresh: Force cache refresh
            
        Returns:
            CorrelationRegimeMetrics with regime classification
        """
        cache_key = f"{'-'.join(sorted(filtered_assets))}_{timeframe}"
        
        # Check cache first
        if not force_refresh and cache_key in self._regime_cache:
            cached_metrics, cache_time = self._regime_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                self.logger.debug(f"Using cached correlation regime for {len(filtered_assets)} assets")
                return cached_metrics
        
        self.logger.info(f"🔗 Detecting correlation regime for {len(filtered_assets)} assets ({timeframe})")
        
        try:
            # Get correlations from Phase 2 correlation engine
            correlation_metrics = await self.correlation_engine.calculate_filtered_asset_correlations(
                filtered_assets, timeframe, force_refresh
            )
            
            # Extract correlation data
            correlation_pairs = correlation_metrics.correlation_pairs
            portfolio_correlation_score = correlation_metrics.portfolio_correlation_score
            
            # Map correlation engine classification to Phase 3 plan specifications
            engine_regime = correlation_metrics.regime_classification
            regime = self._map_correlation_engine_to_regime_detector(engine_regime)
            
            # Calculate average absolute correlation for analysis
            avg_correlation = self._calculate_average_correlation(correlation_pairs)
            
            # Get strength distribution from correlation metrics
            strength_distribution = correlation_metrics.correlation_strength_distribution
            
            # Create regime metrics
            regime_metrics = CorrelationRegimeMetrics(
                correlation_regime=regime,
                average_correlation=avg_correlation,
                correlation_strength_distribution=strength_distribution,
                calculation_timestamp=datetime.now(),
                asset_count=correlation_metrics.asset_count,
                valid_correlation_pairs=correlation_metrics.valid_pairs,
                data_quality_score=correlation_metrics.data_quality_score
            )
            
            # Cache results
            self._regime_cache[cache_key] = (regime_metrics, datetime.now())
            
            self.logger.info(f"   ✅ Correlation regime: {regime}")
            self.logger.info(f"   📊 Average correlation: {avg_correlation:.3f}")
            self.logger.info(f"   🔗 Valid pairs: {correlation_metrics.valid_pairs}")
            
            return regime_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Correlation regime detection failed: {e}")
            # Return safe defaults using Phase 3 plan naming
            return CorrelationRegimeMetrics(
                correlation_regime="normal_correlation",
                average_correlation=0.5,
                correlation_strength_distribution={"strong": 0, "moderate": 0, "weak": 0},
                calculation_timestamp=datetime.now(),
                asset_count=len(filtered_assets),
                valid_correlation_pairs=0,
                data_quality_score=0.0
            )
    
    def _map_correlation_engine_to_regime_detector(self, engine_regime: str) -> str:
        """
        Map Phase 2 correlation engine classifications to Phase 3 plan specifications.
        
        Phase 2 engine returns: low_correlation, medium_correlation, high_correlation
        Phase 3 plan specifies: correlation_breakdown, normal_correlation, high_correlation
        """
        mapping = {
            "low_correlation": "correlation_breakdown",    # <20% - Stock-picking market
            "medium_correlation": "normal_correlation",   # 20-60% - Balanced market
            "high_correlation": "high_correlation"        # >60% - Risk-on/off market
        }
        return mapping.get(engine_regime, "normal_correlation")

    def _calculate_average_correlation(
        self, 
        correlation_pairs: Dict[tuple[str, str], float]
    ) -> float:
        """Calculate average absolute correlation from correlation pairs."""
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
        
        return np.mean(unique_correlations)
    
    def _classify_correlation_regime(self, portfolio_score: float) -> str:
        """
        Classify correlation regime based on portfolio correlation score.
        Follows enhanced_asset_filter.py classification patterns.
        """
        if portfolio_score <= self.regime_thresholds['correlation_breakdown']:
            return "correlation_breakdown"  # Stock-picking market, low correlation
        elif portfolio_score >= self.regime_thresholds['high_correlation']:
            return "high_correlation"       # Risk-on/risk-off market, high correlation
        else:
            return "normal_correlation"     # Balanced market conditions
    
    def get_correlation_regime_summary(self, metrics: CorrelationRegimeMetrics) -> Dict[str, Any]:
        """Get comprehensive correlation regime analysis summary."""
        return {
            "correlation_regime_analysis": {
                "correlation_regime": metrics.correlation_regime,
                "average_correlation": metrics.average_correlation,
                "asset_count": metrics.asset_count,
                "valid_correlation_pairs": metrics.valid_correlation_pairs,
                "data_quality_score": metrics.data_quality_score
            },
            "correlation_distribution": metrics.correlation_strength_distribution,
            "market_implications": {
                "is_correlation_breakdown": metrics.is_correlation_breakdown,
                "is_high_correlation": metrics.is_high_correlation,
                "diversification_effective": metrics.is_diversification_effective,
                "trading_environment": self._get_trading_environment_description(metrics.correlation_regime)
            },
            "regime_thresholds": self.regime_thresholds,
            "metadata": {
                "calculation_timestamp": metrics.calculation_timestamp.isoformat(),
                "detector_version": "1.0.0",
                "integration": "Phase2_CorrelationEngine"
            }
        }
    
    def _get_trading_environment_description(self, regime: str) -> str:
        """Get human-readable trading environment description."""
        descriptions = {
            "correlation_breakdown": "Stock-picking environment - individual asset analysis favored",
            "normal_correlation": "Balanced environment - mixed strategies effective",
            "high_correlation": "Risk-on/risk-off environment - macro themes dominate"
        }
        return descriptions.get(regime, "Unknown correlation environment")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for correlation regime detector."""
        try:
            # Test correlation engine health
            correlation_health = await self.correlation_engine.health_check()
            
            # Test regime detection with minimal data
            test_assets = ['BTC', 'ETH']
            
            regime_detection_healthy = True
            try:
                test_metrics = await self.detect_correlation_regime(
                    test_assets, force_refresh=True
                )
                detection_test_passed = test_metrics.calculation_timestamp is not None
            except Exception as e:
                regime_detection_healthy = False
                detection_test_passed = False
            
            overall_health = (
                correlation_health.get('status') == 'healthy' and 
                regime_detection_healthy
            )
            
            return {
                "status": "healthy" if overall_health else "degraded",
                "component": "CorrelationRegimeDetector",
                "correlation_engine_status": correlation_health.get('status', 'unknown'),
                "storage_backend": correlation_health.get('storage_backend', 'unknown'),
                "regime_detection": "healthy" if regime_detection_healthy else "degraded",
                "cache_size": len(self._regime_cache),
                "configuration": {
                    "regime_thresholds": self.regime_thresholds
                },
                "integration": {
                    "correlation_engine": "Phase2_FilteredAssetCorrelationEngine",
                    "cache_ttl_minutes": int(self._cache_ttl.total_seconds() / 60)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "CorrelationRegimeDetector",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }