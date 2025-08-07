"""
Correlation Signal Generator - Trading Signals from Cross-Asset Analysis

Generates trading signals from correlation analysis using existing
DataStorageInterface and following fear_greed_client.py signal patterns.

Integration Points:
- FilteredAssetCorrelationEngine for correlation metrics
- DataStorageInterface for consistent data access
- Following fear_greed_client.py signal generation patterns
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from src.analysis.correlation_engine import FilteredAssetCorrelationEngine, CorrelationMetrics
from src.data.storage_interfaces import get_storage_implementation
from src.config.settings import get_settings


class CorrelationSignalGenerator:
    """
    Generate trading signals from correlation analysis.
    Uses DataStorageInterface for data access following existing patterns.
    Follows fear_greed_client.py signal generation structure.
    """
    
    def __init__(self, correlation_engine: Optional[FilteredAssetCorrelationEngine] = None):
        """Initialize signal generator with correlation engine integration."""
        self.correlation_engine = correlation_engine or FilteredAssetCorrelationEngine()
        self.storage = get_storage_implementation()
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # Signal generation parameters (following fear_greed pattern)
        self.signal_smoothing_window = getattr(self.settings, 'correlation_signal_smoothing', 5)
        self.signal_threshold = getattr(self.settings, 'correlation_signal_threshold', 0.1)
        
        # Signal cache (following existing caching patterns)
        self._signal_cache: Dict[str, Tuple[pd.Series, datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)
        
        self.logger.info("🔗 CorrelationSignalGenerator initialized")
    
    async def generate_correlation_signals(
        self,
        asset_symbol: str,
        filtered_assets: List[str],
        timeframe: str = '1h',
        lookback_periods: int = 100
    ) -> pd.Series:
        """
        Generate correlation-based signals for specific asset.
        Returns Series matching existing signal generation patterns.
        
        Args:
            asset_symbol: Target asset for signal generation
            filtered_assets: List of assets for correlation analysis
            timeframe: Data timeframe for analysis
            lookback_periods: Number of periods for signal history
            
        Returns:
            pd.Series with correlation-based trading signals [-1.0 to 1.0]
        """
        cache_key = f"{asset_symbol}_{'-'.join(sorted(filtered_assets))}_{timeframe}_{lookback_periods}"
        
        # Check cache first (following existing caching patterns)
        if cache_key in self._signal_cache:
            cached_signals, cache_time = self._signal_cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                self.logger.debug(f"Using cached correlation signals for {asset_symbol}")
                return cached_signals
        
        self.logger.info(f"🔍 Generating correlation signals for {asset_symbol} vs {len(filtered_assets)} assets")
        
        try:
            # Get correlation metrics
            correlation_metrics = await self.correlation_engine.calculate_filtered_asset_correlations(
                filtered_assets, timeframe
            )
            
            # Get asset price data for signal timing (following storage interface pattern)
            asset_data = await self.storage.get_ohlcv_bars(
                symbol=asset_symbol,
                limit=lookback_periods
            )
            
            if asset_data.empty:
                self.logger.warning(f"No data available for {asset_symbol}")
                return pd.Series(dtype=float, name='correlation_signals')
            
            # Create correlation signal series (following fear_greed pattern)
            signals = pd.Series(
                index=asset_data.index, 
                dtype=float, 
                name='correlation_signals'
            )
            
            # Calculate asset-specific correlation metrics
            asset_correlation_metrics = await self._calculate_asset_correlation_metrics(
                asset_symbol, filtered_assets, correlation_metrics, timeframe
            )
            
            # Generate base signals from correlation analysis
            signals = self._generate_base_correlation_signals(
                signals, asset_correlation_metrics, correlation_metrics
            )
            
            # Apply signal processing and smoothing
            signals = self._process_correlation_signals(signals, asset_data)
            
            # Cache results
            self._signal_cache[cache_key] = (signals, datetime.now())
            
            self.logger.info(f"   ✅ Generated {len(signals.dropna())} correlation signals for {asset_symbol}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Signal generation failed for {asset_symbol}: {e}")
            # Return empty series with proper structure
            return pd.Series(dtype=float, name='correlation_signals')
    
    async def _calculate_asset_correlation_metrics(
        self,
        asset_symbol: str,
        filtered_assets: List[str],
        correlation_metrics: CorrelationMetrics,
        timeframe: str
    ) -> Dict[str, float]:
        """Calculate asset-specific correlation metrics."""
        asset_metrics = {
            'correlation_strength': 0.0,
            'diversification_score': 0.0,
            'regime_alignment': 0.0,
            'relative_correlation': 0.0
        }
        
        try:
            # Get asset-specific correlations
            asset_correlations = self._get_asset_specific_correlations(
                asset_symbol, correlation_metrics.correlation_pairs
            )
            
            if not asset_correlations:
                return asset_metrics
            
            # Calculate correlation strength (average absolute correlation)
            asset_metrics['correlation_strength'] = np.mean([
                abs(corr) for corr in asset_correlations.values()
            ])
            
            # Calculate diversification score (inverse of correlation strength)
            asset_metrics['diversification_score'] = 1.0 - asset_metrics['correlation_strength']
            
            # Calculate regime alignment (how well asset fits current regime)
            portfolio_correlation = correlation_metrics.portfolio_correlation_score
            asset_vs_portfolio = abs(asset_metrics['correlation_strength'] - portfolio_correlation)
            asset_metrics['regime_alignment'] = 1.0 - min(1.0, asset_vs_portfolio * 2)
            
            # Relative correlation vs portfolio average
            if portfolio_correlation > 0:
                asset_metrics['relative_correlation'] = (
                    asset_metrics['correlation_strength'] / portfolio_correlation - 1.0
                )
            
        except Exception as e:
            self.logger.debug(f"Asset correlation metrics calculation failed: {e}")
        
        return asset_metrics
    
    def _get_asset_specific_correlations(
        self,
        asset_symbol: str,
        correlation_pairs: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """
        Extract correlations specific to target asset.
        Returns dict of {other_asset: correlation_value}.
        """
        asset_correlations = {}
        
        for (asset1, asset2), correlation in correlation_pairs.items():
            if asset1 == asset_symbol:
                asset_correlations[asset2] = correlation
            elif asset2 == asset_symbol:
                asset_correlations[asset1] = correlation
        
        return asset_correlations
    
    def _generate_base_correlation_signals(
        self,
        signals: pd.Series,
        asset_correlation_metrics: Dict[str, float],
        correlation_metrics: CorrelationMetrics
    ) -> pd.Series:
        """
        Generate base correlation signals following fear_greed_client.py patterns.
        """
        # Base signal strength from correlation metrics
        correlation_strength = asset_correlation_metrics['correlation_strength']
        diversification_score = asset_correlation_metrics['diversification_score']
        regime_alignment = asset_correlation_metrics['regime_alignment']
        
        # Generate signal based on correlation regime and asset characteristics
        regime = correlation_metrics.regime_classification
        
        if regime == "high_correlation":
            # High correlation regime: favor momentum, penalize diversified assets
            base_signal = correlation_strength * 0.5 - diversification_score * 0.3
        elif regime == "low_correlation":
            # Low correlation regime: favor diversified assets, stock-picking environment
            base_signal = diversification_score * 0.6 - correlation_strength * 0.2
        else:
            # Medium correlation: balanced approach
            base_signal = regime_alignment * 0.4
        
        # Apply regime-specific adjustments
        regime_multiplier = self._get_regime_multiplier(regime)
        base_signal *= regime_multiplier
        
        # Clamp signal to [-1.0, 1.0] range (following existing signal patterns)
        base_signal = np.clip(base_signal, -1.0, 1.0)
        
        # Broadcast signal to all time periods (static correlation-based signal)
        signals[:] = base_signal
        
        return signals
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get regime-specific signal multiplier."""
        regime_multipliers = {
            "high_correlation": 0.8,    # Reduce signal strength in high correlation
            "medium_correlation": 1.0,  # Neutral multiplier
            "low_correlation": 1.2      # Amplify signals in low correlation
        }
        return regime_multipliers.get(regime, 1.0)
    
    def _process_correlation_signals(
        self,
        signals: pd.Series,
        asset_data: pd.DataFrame
    ) -> pd.Series:
        """
        Process and smooth correlation signals following existing patterns.
        """
        try:
            # Apply smoothing if enabled (following fear_greed smoothing pattern)
            if self.signal_smoothing_window > 1:
                signals = signals.rolling(
                    window=self.signal_smoothing_window, 
                    center=True,
                    min_periods=1
                ).mean()
            
            # Apply threshold filtering (remove weak signals)
            signals = signals.where(
                abs(signals) >= self.signal_threshold, 
                0.0
            )
            
            # Add market structure signals if available
            if 'volume' in asset_data.columns:
                signals = self._enhance_with_volume_confirmation(signals, asset_data)
            
            # Final signal validation and clamping
            signals = signals.clip(-1.0, 1.0)
            signals = signals.fillna(0.0)  # Replace NaN with neutral signal
            
        except Exception as e:
            self.logger.debug(f"Signal processing failed: {e}")
            # Return safe signals on processing failure
            signals = signals.fillna(0.0)
        
        return signals
    
    def _enhance_with_volume_confirmation(
        self,
        signals: pd.Series,
        asset_data: pd.DataFrame
    ) -> pd.Series:
        """Enhance correlation signals with volume confirmation."""
        try:
            # Calculate volume moving average for confirmation
            volume_ma = asset_data['volume'].rolling(20, min_periods=5).mean()
            volume_ratio = asset_data['volume'] / volume_ma
            
            # Align indices
            aligned_signals = signals.reindex(volume_ratio.index, fill_value=0.0)
            
            # Apply volume confirmation (reduce signals during low volume)
            volume_confirmation = np.clip(volume_ratio * 0.5 + 0.5, 0.5, 1.5)
            enhanced_signals = aligned_signals * volume_confirmation
            
            return enhanced_signals.clip(-1.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Volume enhancement failed: {e}")
            return signals
    
    def get_signal_strength_analysis(
        self,
        signals: pd.Series,
        asset_symbol: str
    ) -> Dict[str, Any]:
        """
        Analyze signal strength and characteristics following fear_greed pattern.
        """
        if signals.empty:
            return {
                "signal_analysis": "no_signals",
                "asset": asset_symbol
            }
        
        signal_stats = {
            "signal_count": len(signals.dropna()),
            "positive_signals": (signals > 0).sum(),
            "negative_signals": (signals < 0).sum(),
            "neutral_signals": (signals == 0).sum(),
            "average_signal_strength": signals.abs().mean(),
            "max_signal_strength": signals.abs().max(),
            "signal_volatility": signals.std()
        }
        
        # Signal regime classification
        if signal_stats["average_signal_strength"] > 0.5:
            signal_regime = "strong_signals"
        elif signal_stats["average_signal_strength"] > 0.2:
            signal_regime = "moderate_signals"
        else:
            signal_regime = "weak_signals"
        
        return {
            "asset": asset_symbol,
            "signal_regime": signal_regime,
            "signal_statistics": signal_stats,
            "signal_direction_bias": (
                "bullish" if signals.mean() > 0.1 else
                "bearish" if signals.mean() < -0.1 else
                "neutral"
            ),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def batch_generate_correlation_signals(
        self,
        asset_symbols: List[str],
        filtered_assets: List[str],
        timeframe: str = '1h',
        lookback_periods: int = 100
    ) -> Dict[str, pd.Series]:
        """
        Generate correlation signals for multiple assets efficiently.
        """
        self.logger.info(f"🔍 Batch generating correlation signals for {len(asset_symbols)} assets")
        
        # Create tasks for concurrent signal generation
        signal_tasks = {}
        for asset in asset_symbols:
            task = self.generate_correlation_signals(
                asset, filtered_assets, timeframe, lookback_periods
            )
            signal_tasks[asset] = task
        
        # Execute all signal generation tasks concurrently
        results = {}
        for asset, task in signal_tasks.items():
            try:
                signals = await task
                results[asset] = signals
            except Exception as e:
                self.logger.error(f"Batch signal generation failed for {asset}: {e}")
                results[asset] = pd.Series(dtype=float, name='correlation_signals')
        
        self.logger.info(f"   ✅ Generated signals for {len(results)} assets")
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for correlation signal generator."""
        try:
            # Test correlation engine health
            engine_health = await self.correlation_engine.health_check()
            
            # Test storage interface
            storage_health = await self.storage.health_check()
            
            # Test signal generation with minimal data
            test_asset = 'BTC'
            test_filtered_assets = ['BTC', 'ETH']
            
            signal_generation_healthy = True
            try:
                test_signals = await self.generate_correlation_signals(
                    test_asset, test_filtered_assets
                )
                signal_test_passed = isinstance(test_signals, pd.Series)
            except Exception as e:
                signal_generation_healthy = False
                signal_test_passed = False
            
            overall_health = (
                engine_health.get('status') == 'healthy' and
                storage_health.get('status') == 'healthy' and
                signal_generation_healthy
            )
            
            return {
                "status": "healthy" if overall_health else "degraded",
                "component": "CorrelationSignalGenerator",
                "correlation_engine_status": engine_health.get('status', 'unknown'),
                "storage_status": storage_health.get('status', 'unknown'),
                "signal_generation": "healthy" if signal_generation_healthy else "degraded",
                "cache_size": len(self._signal_cache),
                "configuration": {
                    "signal_smoothing_window": self.signal_smoothing_window,
                    "signal_threshold": self.signal_threshold,
                    "cache_ttl_minutes": self._cache_ttl.total_seconds() / 60
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "CorrelationSignalGenerator",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }