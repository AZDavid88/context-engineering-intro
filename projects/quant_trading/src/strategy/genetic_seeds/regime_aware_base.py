"""
Regime-Aware Genetic Seed Base Class - Phase 3 Enhancement

Base class for regime-aware genetic seeds that adapt their behavior based on
multi-source market regime analysis. Extends existing BaseSeed architecture
with regime-specific parameter adjustment and signal modification.

Key Features:
- Optional regime awareness (backward compatibility maintained)
- Regime-specific genetic parameter evolution
- Dynamic signal adjustment based on market conditions
- Integration with CompositeRegimeDetectionEngine
- Genetic algorithm environmental pressure application
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes, SeedType
from src.analysis.regime_detection_engine import CompositeRegimeDetectionEngine, RegimeAnalysis, CompositeRegime
from src.config.settings import get_settings


class RegimeAwareSeed(BaseSeed):
    """
    Base class for regime-aware genetic seeds.
    Extends BaseSeed with market regime adaptation capabilities.
    
    This class maintains full backward compatibility - seeds work identically
    when regime_engine is None, but gain adaptive capabilities when provided.
    """
    
    def __init__(self, 
                 genes: SeedGenes, 
                 regime_engine: Optional[CompositeRegimeDetectionEngine] = None):
        """
        Initialize regime-aware seed with optional regime detection.
        
        Args:
            genes: Seed genetic parameters
            regime_engine: Optional composite regime detection engine
        """
        super().__init__(genes)
        self.regime_engine = regime_engine
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Regime-specific parameters added to genetic evolution
        if self.regime_engine is not None:
            self._add_regime_genetic_parameters()
        
        # Regime awareness state
        self._last_regime_analysis: Optional[RegimeAnalysis] = None
        self._regime_analysis_cache_ttl = 600  # 10 minutes cache
        
        self.logger.debug(f"RegimeAwareSeed initialized (regime_aware: {self.is_regime_aware})")
    
    def _add_regime_genetic_parameters(self) -> None:
        """Add regime-specific parameters to genetic evolution."""
        # These parameters control how signals are adjusted based on regime
        regime_parameters = {
            # Signal multipliers for different regimes
            'risk_on_signal_multiplier': 1.2,      # Amplify signals in favorable conditions
            'risk_off_signal_multiplier': 0.8,     # Dampen signals in defensive conditions
            'transitional_signal_dampening': 0.5,  # Reduce signals during regime transitions
            
            # Confidence thresholds
            'regime_confidence_threshold': 0.7,    # Minimum confidence to apply adjustments
            'stability_threshold': 0.8,            # Minimum stability to trust regime
            
            # Regime-specific strategy parameters
            'risk_on_lookback_multiplier': 1.1,    # Longer lookbacks in risk-on
            'risk_off_lookback_multiplier': 0.9,   # Shorter lookbacks in risk-off
            
            # Position sizing adjustments
            'regime_position_sizing_factor': 0.2,  # How much regime affects position size
        }
        
        # Add to existing genetic parameters
        self.genes.parameters.update(regime_parameters)
    
    @property
    def is_regime_aware(self) -> bool:
        """Check if this seed has regime awareness capabilities."""
        return self.regime_engine is not None
    
    async def generate_regime_aware_signals(self, 
                                          data: pd.DataFrame,
                                          filtered_assets: Optional[List[str]] = None,
                                          all_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
                                          timeframe: str = '1h') -> pd.Series:
        """
        Generate signals with regime awareness.
        
        This is the main entry point for regime-aware signal generation.
        Falls back to standard signal generation if regime engine not available.
        
        Args:
            data: OHLCV data for signal generation
            filtered_assets: List of assets for regime analysis
            all_asset_data: Complete asset data for regime detection
            timeframe: Data timeframe
            
        Returns:
            Regime-adjusted trading signals
        """
        # Generate base trading signals using existing logic
        base_signals = self.generate_signals(data)
        
        # If no regime awareness or insufficient data, return base signals
        if not self.is_regime_aware or filtered_assets is None:
            return base_signals
        
        try:
            # Get current market regime analysis
            regime_analysis = await self._get_regime_analysis(
                filtered_assets, all_asset_data, timeframe
            )
            
            # Apply regime-based signal adjustments
            regime_adjusted_signals = self._apply_regime_adjustments(
                base_signals, regime_analysis, data
            )
            
            return regime_adjusted_signals
            
        except Exception as e:
            self.logger.warning(f"Regime analysis failed, using base signals: {e}")
            return base_signals
    
    async def _get_regime_analysis(self,
                                 filtered_assets: List[str],
                                 all_asset_data: Optional[Dict[str, pd.DataFrame]],
                                 timeframe: str) -> RegimeAnalysis:
        """Get regime analysis with caching for performance."""
        
        # Check cache first
        if (self._last_regime_analysis is not None and 
            self._is_regime_analysis_fresh()):
            return self._last_regime_analysis
        
        # Get fresh regime analysis
        regime_analysis = await self.regime_engine.detect_composite_regime(
            filtered_assets, timeframe, force_refresh=False
        )
        
        # Cache for performance
        self._last_regime_analysis = regime_analysis
        
        return regime_analysis
    
    def _is_regime_analysis_fresh(self) -> bool:
        """Check if cached regime analysis is still fresh."""
        if self._last_regime_analysis is None:
            return False
        
        age = (datetime.now() - self._last_regime_analysis.calculation_timestamp).total_seconds()
        return age < self._regime_analysis_cache_ttl
    
    def _apply_regime_adjustments(self, 
                                base_signals: pd.Series,
                                regime_analysis: RegimeAnalysis,
                                data: pd.DataFrame) -> pd.Series:
        """
        Apply regime-based signal adjustments.
        Core regime adaptation logic that can be overridden by subclasses.
        
        Args:
            base_signals: Original trading signals
            regime_analysis: Current market regime analysis
            data: OHLCV data for additional context
            
        Returns:
            Regime-adjusted signals
        """
        adjusted_signals = base_signals.copy()
        
        # Only apply adjustments if regime confidence is high enough
        if regime_analysis.regime_confidence < self.genes.parameters['regime_confidence_threshold']:
            self.logger.debug(f"Low regime confidence ({regime_analysis.regime_confidence:.2f}), "
                            f"using base signals")
            return adjusted_signals
        
        # Get regime-specific multiplier
        multiplier = self._get_regime_signal_multiplier(regime_analysis)
        
        # Apply signal adjustment
        adjusted_signals = adjusted_signals * multiplier
        
        # Apply regime-specific parameter adjustments
        adjusted_signals = self._apply_regime_parameter_adjustments(
            adjusted_signals, regime_analysis, data
        )
        
        # Log adjustment details
        self.logger.debug(f"Applied regime adjustments: "
                         f"regime={regime_analysis.composite_regime.value}, "
                         f"multiplier={multiplier:.2f}, "
                         f"confidence={regime_analysis.regime_confidence:.2f}")
        
        return adjusted_signals
    
    def _get_regime_signal_multiplier(self, regime_analysis: RegimeAnalysis) -> float:
        """Get signal multiplier based on current regime."""
        
        composite_regime = regime_analysis.composite_regime
        
        if composite_regime == CompositeRegime.RISK_ON:
            # Amplify signals in favorable market conditions
            return self.genes.parameters['risk_on_signal_multiplier']
            
        elif composite_regime == CompositeRegime.RISK_OFF:
            # Dampen signals in defensive market conditions
            return self.genes.parameters['risk_off_signal_multiplier']
            
        elif composite_regime == CompositeRegime.TRANSITIONAL:
            # Reduce signal strength during regime transitions
            return self.genes.parameters['transitional_signal_dampening']
            
        else:  # NEUTRAL
            # No adjustment for neutral conditions
            return 1.0
    
    def _apply_regime_parameter_adjustments(self, 
                                          signals: pd.Series,
                                          regime_analysis: RegimeAnalysis,
                                          data: pd.DataFrame) -> pd.Series:
        """
        Apply regime-specific parameter adjustments.
        Can be overridden by subclasses for strategy-specific logic.
        
        Args:
            signals: Current trading signals
            regime_analysis: Market regime analysis
            data: OHLCV data
            
        Returns:
            Further adjusted signals
        """
        # Base implementation: no additional adjustments
        # Subclasses can override this for strategy-specific regime adaptations
        return signals
    
    def get_regime_genetic_pressure(self, regime_analysis: RegimeAnalysis) -> Dict[str, float]:
        """
        Get genetic algorithm environmental pressure from regime analysis.
        Integrates with existing genetic algorithm evolution system.
        
        Args:
            regime_analysis: Current market regime analysis
            
        Returns:
            Dictionary of environmental pressure parameters
        """
        if not self.is_regime_aware:
            return {}
        
        return self.regime_engine.generate_genetic_algorithm_pressure(regime_analysis)
    
    def get_regime_aware_fitness_modifier(self, 
                                        regime_analysis: RegimeAnalysis,
                                        base_fitness: float) -> float:
        """
        Get fitness modifier based on regime awareness.
        
        Rewards strategies that adapt well to current market regime.
        
        Args:
            regime_analysis: Current market regime analysis
            base_fitness: Base fitness score
            
        Returns:
            Modified fitness score
        """
        if not self.is_regime_aware or not regime_analysis.is_high_confidence:
            return base_fitness
        
        # Reward strategies that show regime awareness
        regime_awareness_bonus = 0.05  # 5% bonus for regime-aware strategies
        
        # Additional bonus for stable regimes (more predictable)
        stability_bonus = regime_analysis.regime_stability * 0.03  # Up to 3% bonus
        
        return base_fitness * (1.0 + regime_awareness_bonus + stability_bonus)
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime awareness state."""
        if not self.is_regime_aware:
            return {"regime_aware": False}
        
        summary = {
            "regime_aware": True,
            "regime_parameters": {
                key: value for key, value in self.genes.parameters.items()
                if 'regime' in key or 'risk_' in key or 'transitional' in key
            },
            "last_regime_analysis": None,
            "analysis_cache_fresh": False
        }
        
        if self._last_regime_analysis is not None:
            summary["last_regime_analysis"] = {
                "composite_regime": self._last_regime_analysis.composite_regime.value,
                "confidence": self._last_regime_analysis.regime_confidence,
                "stability": self._last_regime_analysis.regime_stability,
                "timestamp": self._last_regime_analysis.calculation_timestamp.isoformat()
            }
            summary["analysis_cache_fresh"] = self._is_regime_analysis_fresh()
        
        return summary
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate base trading signals (must be implemented by subclasses).
        This maintains the existing BaseSeed interface.
        """
        pass
    
    # Backward compatibility methods
    def generate_signals_with_regime(self, 
                                   data: pd.DataFrame,
                                   filtered_assets: Optional[List[str]] = None,
                                   all_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
                                   timeframe: str = '1h') -> pd.Series:
        """
        Convenience method for async signal generation.
        Wraps the async generate_regime_aware_signals method.
        """
        import asyncio
        
        # Run async method in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate_regime_aware_signals(data, filtered_assets, all_asset_data, timeframe)
        )


class RegimeAwareEnhancedSeed(RegimeAwareSeed):
    """
    Enhanced regime-aware seed with additional regime-specific features.
    
    This class provides more advanced regime adaptations for strategies
    that can benefit from deeper regime analysis.
    """
    
    def _apply_regime_parameter_adjustments(self, 
                                          signals: pd.Series,
                                          regime_analysis: RegimeAnalysis,
                                          data: pd.DataFrame) -> pd.Series:
        """
        Enhanced regime parameter adjustments.
        Implements more sophisticated regime-based adaptations.
        """
        adjusted_signals = signals.copy()
        composite_regime = regime_analysis.composite_regime
        
        # Regime-specific lookback adjustments
        if hasattr(self, '_apply_regime_lookback_adjustment'):
            adjusted_signals = self._apply_regime_lookback_adjustment(
                adjusted_signals, regime_analysis, data
            )
        
        # Volatility regime specific adjustments
        if 'high_volatility' in [regime_analysis.volatility_regime]:
            # Reduce signal frequency in high volatility
            adjusted_signals = self._smooth_signals_for_volatility(adjusted_signals)
        
        # Correlation regime specific adjustments
        if 'high_correlation' in [regime_analysis.correlation_regime]:
            # Adjust for high correlation environments
            adjusted_signals = self._adjust_for_correlation_regime(adjusted_signals)
        
        return adjusted_signals
    
    def _smooth_signals_for_volatility(self, signals: pd.Series) -> pd.Series:
        """Smooth signals during high volatility periods."""
        # Simple smoothing using rolling average
        window = 3
        return signals.rolling(window=window, center=True).mean().fillna(signals)
    
    def _adjust_for_correlation_regime(self, signals: pd.Series) -> pd.Series:
        """Adjust signals for high correlation environments."""
        # In high correlation environments, reduce signal strength slightly
        # as individual asset signals may be less effective
        return signals * 0.9
    
    def get_enhanced_regime_metrics(self, regime_analysis: RegimeAnalysis) -> Dict[str, Any]:
        """Get enhanced metrics specific to this strategy's regime adaptation."""
        base_metrics = self.get_regime_summary()
        
        if regime_analysis:
            base_metrics.update({
                "regime_specific_adjustments": {
                    "volatility_adjustment": "high_volatility" in [regime_analysis.volatility_regime],
                    "correlation_adjustment": "high_correlation" in [regime_analysis.correlation_regime],
                    "signal_smoothing_active": regime_analysis.composite_regime == CompositeRegime.TRANSITIONAL
                },
                "confidence_metrics": {
                    "meets_confidence_threshold": regime_analysis.is_high_confidence,
                    "regime_stability": regime_analysis.regime_stability
                }
            })
        
        return base_metrics