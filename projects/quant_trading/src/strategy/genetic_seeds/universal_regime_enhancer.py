"""
Universal Regime Enhancement Wrapper - Phase 3 Implementation

Universal wrapper that adds regime awareness to ANY existing genetic seed
without modifying the original seed code. Follows Phase 2 correlation
enhancement patterns for architectural consistency.

Key Features:
- Zero-modification enhancement of existing seeds
- DRY principle compliance - one wrapper for all seeds
- Backward compatibility maintained
- Optional regime awareness activation
- Integration with CompositeRegimeDetectionEngine
- Follows universal_correlation_enhancer.py patterns

Usage:
    # Enhance any existing seed with regime awareness
    original_seed = MomentumMACDSeed(genes)
    regime_aware_seed = UniversalRegimeEnhancer(original_seed, regime_engine)
    
    # Or wrap multiple seeds
    enhanced_seeds = enhance_seeds_with_regime_awareness(seed_list, regime_engine)
"""

import logging
from typing import Dict, List, Optional, Any, Union, Type
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes, SeedType
from src.analysis.regime_detection_engine import CompositeRegimeDetectionEngine, RegimeAnalysis, CompositeRegime
from src.config.settings import get_settings


class UniversalRegimeEnhancer:
    """
    Universal wrapper that adds regime awareness to any existing genetic seed.
    
    This wrapper follows the same pattern as universal_correlation_enhancer.py,
    providing regime awareness to all seeds without code duplication.
    
    The wrapper maintains complete compatibility with the original seed
    while adding regime-based signal adjustments and parameter adaptations.
    """
    
    def __init__(self, 
                 base_seed: BaseSeed,
                 regime_engine: Optional[CompositeRegimeDetectionEngine] = None,
                 regime_enhancement_config: Optional[Dict[str, Any]] = None):
        """
        Initialize universal regime enhancer wrapper.
        
        Args:
            base_seed: Original seed to enhance
            regime_engine: Composite regime detection engine
            regime_enhancement_config: Optional configuration overrides
        """
        self.base_seed = base_seed
        self.regime_engine = regime_engine
        self.logger = logging.getLogger(f"{__name__}.{base_seed.__class__.__name__}")
        
        # Configuration with sensible defaults
        default_config = {
            # Signal adjustment multipliers
            'risk_on_signal_multiplier': 1.2,      # 20% boost in favorable conditions
            'risk_off_signal_multiplier': 0.8,     # 20% reduction in defensive conditions
            'transitional_signal_dampening': 0.6,  # 40% reduction during transitions
            
            # Confidence and stability thresholds
            'regime_confidence_threshold': 0.7,    # Minimum confidence for adjustments
            'stability_threshold': 0.8,            # Minimum stability threshold
            
            # Performance enhancement factors
            'regime_fitness_bonus': 0.05,          # 5% fitness bonus for regime awareness
            'stability_fitness_bonus': 0.03,      # Up to 3% bonus for stable regimes
            
            # Strategy-specific adjustments
            'volatility_regime_sensitivity': 1.0,  # How much volatility affects signals
            'correlation_regime_sensitivity': 1.0, # How much correlation affects signals
            'volume_regime_sensitivity': 0.8,      # How much volume affects signals
        }
        
        # Merge with provided config
        self.config = default_config.copy()
        if regime_enhancement_config:
            self.config.update(regime_enhancement_config)
        
        # Add regime parameters to genetic evolution
        if self.is_regime_aware:
            self._enhance_genetic_parameters()
        
        # Regime analysis caching
        self._last_regime_analysis: Optional[RegimeAnalysis] = None
        self._regime_cache_ttl = 600  # 10 minutes
        
        self.logger.info(f"UniversalRegimeEnhancer initialized for {base_seed.__class__.__name__} "
                        f"(regime_aware: {self.is_regime_aware})")
    
    @property
    def is_regime_aware(self) -> bool:
        """Check if regime enhancement is active."""
        return self.regime_engine is not None
    
    def _enhance_genetic_parameters(self) -> None:
        """Add regime-specific parameters to genetic evolution."""
        # Add regime enhancement parameters to the base seed's genetic parameters
        regime_genetic_params = {
            'regime_risk_on_multiplier': self.config['risk_on_signal_multiplier'],
            'regime_risk_off_multiplier': self.config['risk_off_signal_multiplier'],
            'regime_transition_dampening': self.config['transitional_signal_dampening'],
            'regime_confidence_threshold': self.config['regime_confidence_threshold'],
            'regime_volatility_sensitivity': self.config['volatility_regime_sensitivity'],
            'regime_correlation_sensitivity': self.config['correlation_regime_sensitivity'],
        }
        
        # Safely add to existing parameters
        if hasattr(self.base_seed, 'genes') and hasattr(self.base_seed.genes, 'parameters'):
            self.base_seed.genes.parameters.update(regime_genetic_params)
    
    async def generate_enhanced_signals(self, 
                                      data: pd.DataFrame,
                                      filtered_assets: Optional[List[str]] = None,
                                      all_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
                                      timeframe: str = '1h') -> pd.Series:
        """
        Generate regime-enhanced signals from the base seed.
        
        This is the main enhancement interface that wraps the original
        seed's signal generation with regime awareness.
        
        Args:
            data: OHLCV data for signal generation
            filtered_assets: Assets for regime analysis
            all_asset_data: Complete asset data for regime detection
            timeframe: Data timeframe
            
        Returns:
            Regime-enhanced trading signals
        """
        # Generate base signals from original seed (unchanged)
        base_signals = self.base_seed.generate_signals(data)
        
        # If no regime awareness, return original signals
        if not self.is_regime_aware or filtered_assets is None:
            return base_signals
        
        try:
            # Get regime analysis
            regime_analysis = await self._get_regime_analysis(filtered_assets, timeframe)
            
            # Apply regime enhancements
            enhanced_signals = self._apply_regime_enhancements(
                base_signals, regime_analysis, data
            )
            
            self.logger.debug(f"Applied regime enhancements: "
                            f"regime={regime_analysis.composite_regime.value}, "
                            f"confidence={regime_analysis.regime_confidence:.2f}")
            
            return enhanced_signals
            
        except Exception as e:
            self.logger.warning(f"Regime enhancement failed, using base signals: {e}")
            return base_signals
    
    async def _get_regime_analysis(self, 
                                 filtered_assets: List[str],
                                 timeframe: str) -> RegimeAnalysis:
        """Get regime analysis with intelligent caching."""
        
        # Check if cached analysis is still fresh
        if (self._last_regime_analysis is not None and 
            self._is_cache_fresh()):
            return self._last_regime_analysis
        
        # Get fresh regime analysis
        regime_analysis = await self.regime_engine.detect_composite_regime(
            filtered_assets, timeframe, force_refresh=False
        )
        
        # Cache for performance
        self._last_regime_analysis = regime_analysis
        
        return regime_analysis
    
    def _is_cache_fresh(self) -> bool:
        """Check if regime analysis cache is still fresh."""
        if self._last_regime_analysis is None:
            return False
        
        age = (datetime.now() - self._last_regime_analysis.calculation_timestamp).total_seconds()
        return age < self._regime_cache_ttl
    
    def _apply_regime_enhancements(self, 
                                 base_signals: pd.Series,
                                 regime_analysis: RegimeAnalysis,
                                 data: pd.DataFrame) -> pd.Series:
        """
        Apply regime enhancements to base signals.
        
        This method implements the core regime enhancement logic
        that works universally across all seed types.
        """
        enhanced_signals = base_signals.copy()
        
        # Only apply if confidence threshold is met
        confidence_threshold = self.config['regime_confidence_threshold']
        if regime_analysis.regime_confidence < confidence_threshold:
            self.logger.debug(f"Low regime confidence ({regime_analysis.regime_confidence:.2f} < {confidence_threshold:.2f})")
            return enhanced_signals
        
        # Apply composite regime adjustments
        regime_multiplier = self._get_regime_multiplier(regime_analysis.composite_regime)
        enhanced_signals = enhanced_signals * regime_multiplier
        
        # Apply individual regime factor adjustments
        enhanced_signals = self._apply_individual_regime_factors(
            enhanced_signals, regime_analysis
        )
        
        # Apply strategy-specific enhancements based on seed type
        enhanced_signals = self._apply_seed_type_specific_enhancements(
            enhanced_signals, regime_analysis, data
        )
        
        return enhanced_signals
    
    def _get_regime_multiplier(self, composite_regime: CompositeRegime) -> float:
        """Get signal multiplier based on composite regime."""
        
        if composite_regime == CompositeRegime.RISK_ON:
            return self.config['risk_on_signal_multiplier']
        elif composite_regime == CompositeRegime.RISK_OFF:
            return self.config['risk_off_signal_multiplier']
        elif composite_regime == CompositeRegime.TRANSITIONAL:
            return self.config['transitional_signal_dampening']
        else:  # NEUTRAL
            return 1.0
    
    def _apply_individual_regime_factors(self, 
                                       signals: pd.Series,
                                       regime_analysis: RegimeAnalysis) -> pd.Series:
        """Apply adjustments based on individual regime factors."""
        adjusted_signals = signals.copy()
        
        # Volatility regime adjustments
        if regime_analysis.volatility_regime == 'high_volatility':
            # Reduce signal strength in high volatility
            volatility_factor = 1.0 - (0.2 * self.config['volatility_regime_sensitivity'])
            adjusted_signals *= volatility_factor
        
        # Correlation regime adjustments
        if regime_analysis.correlation_regime == 'high_correlation':
            # Reduce signals in high correlation (macro dominance)
            correlation_factor = 1.0 - (0.15 * self.config['correlation_regime_sensitivity'])
            adjusted_signals *= correlation_factor
        elif regime_analysis.correlation_regime == 'correlation_breakdown':
            # Boost signals in correlation breakdown (stock picking)
            correlation_factor = 1.0 + (0.1 * self.config['correlation_regime_sensitivity'])
            adjusted_signals *= correlation_factor
        
        # Volume regime adjustments
        if regime_analysis.volume_regime == 'low_volume':
            # Reduce signals in low volume (thin market)
            volume_factor = 1.0 - (0.1 * self.config['volume_regime_sensitivity'])
            adjusted_signals *= volume_factor
        
        return adjusted_signals
    
    def _apply_seed_type_specific_enhancements(self, 
                                             signals: pd.Series,
                                             regime_analysis: RegimeAnalysis,
                                             data: pd.DataFrame) -> pd.Series:
        """
        Apply seed-type-specific regime enhancements.
        
        Different strategy types benefit from different regime adaptations.
        """
        seed_type = getattr(self.base_seed, 'seed_type', None)
        
        if seed_type == SeedType.MOMENTUM:
            return self._enhance_momentum_strategy(signals, regime_analysis)
        elif seed_type == SeedType.MEAN_REVERSION:
            return self._enhance_mean_reversion_strategy(signals, regime_analysis)
        elif seed_type == SeedType.VOLATILITY:
            return self._enhance_volatility_strategy(signals, regime_analysis)
        elif seed_type == SeedType.VOLUME:
            return self._enhance_volume_strategy(signals, regime_analysis)
        else:
            # Generic enhancement for unknown types
            return self._enhance_generic_strategy(signals, regime_analysis)
    
    def _enhance_momentum_strategy(self, 
                                 signals: pd.Series,
                                 regime_analysis: RegimeAnalysis) -> pd.Series:
        """Enhance momentum strategies based on regime."""
        enhanced = signals.copy()
        
        # Momentum works better in trending markets (risk-on)
        if regime_analysis.composite_regime == CompositeRegime.RISK_ON:
            enhanced *= 1.1  # 10% boost for momentum in risk-on
        elif regime_analysis.composite_regime == CompositeRegime.TRANSITIONAL:
            enhanced *= 0.8  # 20% reduction during transitions
        
        return enhanced
    
    def _enhance_mean_reversion_strategy(self, 
                                       signals: pd.Series,
                                       regime_analysis: RegimeAnalysis) -> pd.Series:
        """Enhance mean reversion strategies based on regime."""
        enhanced = signals.copy()
        
        # Mean reversion works better in range-bound markets
        if regime_analysis.volatility_regime == 'low_volatility':
            enhanced *= 1.15  # 15% boost in low volatility
        elif regime_analysis.volatility_regime == 'high_volatility':
            enhanced *= 0.75  # 25% reduction in high volatility
        
        return enhanced
    
    def _enhance_volatility_strategy(self, 
                                   signals: pd.Series,
                                   regime_analysis: RegimeAnalysis) -> pd.Series:
        """Enhance volatility strategies based on regime."""
        enhanced = signals.copy()
        
        # Volatility strategies benefit from regime transitions
        if regime_analysis.composite_regime == CompositeRegime.TRANSITIONAL:
            enhanced *= 1.2  # 20% boost during transitions
        elif regime_analysis.regime_stability < 0.5:
            enhanced *= 1.1  # 10% boost in unstable regimes
        
        return enhanced
    
    def _enhance_volume_strategy(self, 
                               signals: pd.Series,
                               regime_analysis: RegimeAnalysis) -> pd.Series:
        """Enhance volume strategies based on regime."""
        enhanced = signals.copy()
        
        # Volume strategies work better with high participation
        if regime_analysis.volume_regime == 'high_volume':
            enhanced *= 1.1  # 10% boost in high volume
        elif regime_analysis.volume_regime == 'low_volume':
            enhanced *= 0.9  # 10% reduction in low volume
        
        return enhanced
    
    def _enhance_generic_strategy(self, 
                                signals: pd.Series,
                                regime_analysis: RegimeAnalysis) -> pd.Series:
        """Generic enhancement for unknown strategy types."""
        # Conservative generic enhancements
        enhanced = signals.copy()
        
        # Apply stability-based adjustment
        if regime_analysis.regime_stability < 0.5:
            enhanced *= 0.95  # Slight reduction in unstable conditions
        
        return enhanced
    
    def calculate_enhanced_fitness(self, 
                                 base_fitness: float,
                                 regime_analysis: Optional[RegimeAnalysis] = None) -> float:
        """
        Calculate regime-enhanced fitness score.
        
        Rewards strategies that demonstrate regime awareness.
        """
        if not self.is_regime_aware or regime_analysis is None:
            return base_fitness
        
        enhanced_fitness = base_fitness
        
        # Regime awareness bonus
        if regime_analysis.is_high_confidence:
            regime_bonus = self.config['regime_fitness_bonus']
            enhanced_fitness *= (1.0 + regime_bonus)
        
        # Stability bonus
        stability_bonus = (regime_analysis.regime_stability * 
                          self.config['stability_fitness_bonus'])
        enhanced_fitness *= (1.0 + stability_bonus)
        
        return enhanced_fitness
    
    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get summary of regime enhancement capabilities."""
        summary = {
            "base_seed_type": self.base_seed.__class__.__name__,
            "seed_type": getattr(self.base_seed, 'seed_type', 'unknown'),
            "regime_aware": self.is_regime_aware,
            "enhancement_config": self.config.copy(),
            "last_regime_analysis": None,
            "cache_status": "no_cache"
        }
        
        if self._last_regime_analysis:
            summary["last_regime_analysis"] = {
                "composite_regime": self._last_regime_analysis.composite_regime.value,
                "confidence": self._last_regime_analysis.regime_confidence,
                "stability": self._last_regime_analysis.regime_stability,
                "timestamp": self._last_regime_analysis.calculation_timestamp.isoformat()
            }
            summary["cache_status"] = "fresh" if self._is_cache_fresh() else "stale"
        
        return summary
    
    # Delegate all other methods to the base seed for full compatibility
    def __getattr__(self, name):
        """Delegate unknown attributes to the base seed."""
        return getattr(self.base_seed, name)


def enhance_seeds_with_regime_awareness(
    seeds: List[BaseSeed],
    regime_engine: Optional[CompositeRegimeDetectionEngine] = None,
    enhancement_config: Optional[Dict[str, Any]] = None
) -> List[UniversalRegimeEnhancer]:
    """
    Enhance multiple seeds with regime awareness.
    
    Convenience function for bulk enhancement of seed collections.
    
    Args:
        seeds: List of seeds to enhance
        regime_engine: Regime detection engine
        enhancement_config: Optional configuration
        
    Returns:
        List of regime-enhanced seeds
    """
    enhanced_seeds = []
    
    for seed in seeds:
        enhanced_seed = UniversalRegimeEnhancer(
            base_seed=seed,
            regime_engine=regime_engine,
            regime_enhancement_config=enhancement_config
        )
        enhanced_seeds.append(enhanced_seed)
    
    return enhanced_seeds


def enhance_seed_registry_with_regime_awareness(
    seed_registry_class: Type,
    regime_engine: Optional[CompositeRegimeDetectionEngine] = None
) -> Type:
    """
    Create a regime-aware version of a seed registry class.
    
    This function dynamically creates a new registry class that wraps
    all seeds with regime awareness capabilities.
    
    Args:
        seed_registry_class: Original seed registry class
        regime_engine: Regime detection engine
        
    Returns:
        Enhanced seed registry class
    """
    
    class RegimeAwareSeedRegistry(seed_registry_class):
        """Regime-aware seed registry that wraps all seeds with regime enhancement."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.regime_engine = regime_engine
            self._enhance_all_seeds()
        
        def _enhance_all_seeds(self):
            """Enhance all seeds in the registry with regime awareness."""
            if not self.regime_engine:
                return
            
            # Enhance all registered seeds
            for seed_type, seed_classes in self.seeds.items():
                enhanced_classes = []
                for seed_class in seed_classes:
                    # Create wrapper class that enhances instances
                    class RegimeEnhancedSeedClass(seed_class):
                        def __new__(cls, *args, **kwargs):
                            # Create original instance
                            instance = super().__new__(cls)
                            # Wrap with regime enhancement
                            return UniversalRegimeEnhancer(instance, regime_engine)
                    
                    enhanced_classes.append(RegimeEnhancedSeedClass)
                
                self.seeds[seed_type] = enhanced_classes
    
    return RegimeAwareSeedRegistry


# Convenience functions for common use cases
async def generate_regime_aware_signals_for_seed(
    seed: BaseSeed,
    data: pd.DataFrame,
    regime_engine: CompositeRegimeDetectionEngine,
    filtered_assets: List[str],
    timeframe: str = '1h'
) -> pd.Series:
    """
    Convenience function to generate regime-aware signals for any seed.
    
    Quick way to add regime awareness to a single seed without wrapping.
    """
    enhancer = UniversalRegimeEnhancer(seed, regime_engine)
    return await enhancer.generate_enhanced_signals(
        data, filtered_assets, timeframe=timeframe
    )


def create_regime_aware_seed_factory(
    base_factory_function,
    regime_engine: Optional[CompositeRegimeDetectionEngine] = None
):
    """
    Create a regime-aware version of a seed factory function.
    
    Wraps any seed factory function to automatically enhance seeds
    with regime awareness.
    """
    
    def regime_aware_factory(*args, **kwargs):
        # Create base seed using original factory
        base_seed = base_factory_function(*args, **kwargs)
        
        # Enhance with regime awareness
        if regime_engine:
            return UniversalRegimeEnhancer(base_seed, regime_engine)
        else:
            return base_seed
    
    return regime_aware_factory