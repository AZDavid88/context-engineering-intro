"""
Correlation-Enhanced Base Seed - Foundation for Correlation-Aware Genetic Seeds

Extends BaseSeed with correlation signal integration following existing
genetic seed patterns and framework. Provides foundation for all
correlation-enhanced genetic seed variants.

Integration Points:
- BaseSeed framework for genetic parameter evolution
- CorrelationSignalGenerator for cross-asset analysis
- DataStorageInterface for consistent data access
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import logging

from .base_seed import BaseSeed, SeedGenes, SeedType
from src.signals.correlation_signals import CorrelationSignalGenerator
from src.config.settings import get_settings


class CorrelationEnhancedSeed(BaseSeed):
    """
    Base class for genetic seeds with correlation signal integration.
    Extends existing BaseSeed with correlation awareness following framework patterns.
    
    This class provides correlation enhancement capabilities to any genetic seed
    while maintaining full compatibility with the existing genetic algorithm framework.
    """
    
    def __init__(self, genes: SeedGenes):
        """Initialize correlation-enhanced seed with genetic parameters."""
        super().__init__(genes)
        
        # Initialize correlation signal generator
        self.correlation_signal_generator = CorrelationSignalGenerator()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Add correlation-specific parameters to genetic evolution
        # Following base_seed.py parameter pattern
        correlation_params = {
            'correlation_confirmation_threshold': 0.5,  # Threshold for correlation confirmation
            'correlation_regime_adjustment': 0.2,       # Adjustment factor based on correlation regime
            'diversification_bonus': 0.1,               # Bonus for diversified signals
            'correlation_weight': 0.3,                  # Weight of correlation signals in enhancement
            'regime_sensitivity': 0.5                   # Sensitivity to correlation regime changes
        }
        
        # Only add parameters that don't already exist (allow genetic exploration)
        for param_name, default_value in correlation_params.items():
            if param_name not in self.genes.parameters:
                self.genes.parameters[param_name] = default_value
        
        # Cache for correlation data (performance optimization)
        self._correlation_cache: Dict[str, Any] = {}
        self._last_correlation_update: Optional[datetime] = None
        
        self.logger.debug(f"Initialized {self.__class__.__name__} with correlation enhancement")
    
    @property
    def correlation_parameter_bounds(self) -> Dict[str, tuple]:
        """
        Return bounds for correlation-specific genetic parameters.
        Follows base_seed.py parameter_bounds pattern.
        """
        return {
            'correlation_confirmation_threshold': (0.1, 0.9),
            'correlation_regime_adjustment': (0.0, 0.5),
            'diversification_bonus': (0.0, 0.3),
            'correlation_weight': (0.1, 0.8),
            'regime_sensitivity': (0.1, 1.0)
        }
    
    async def generate_enhanced_signals(
        self, 
        data: pd.DataFrame,
        filtered_assets: List[str],
        current_asset: str,
        timeframe: str = '1h'
    ) -> pd.Series:
        """
        Generate signals enhanced with correlation analysis.
        Follows existing generate_signals() pattern with correlation enhancement.
        
        Args:
            data: OHLCV DataFrame for the asset
            filtered_assets: List of all assets for correlation analysis
            current_asset: Current asset symbol
            timeframe: Data timeframe for correlation analysis
            
        Returns:
            pd.Series: Enhanced trading signals
        """
        try:
            # Generate base trading signals using existing implementation
            base_signals = self.generate_signals(data)
            
            # Skip correlation enhancement if insufficient assets
            if len(filtered_assets) < 2:
                self.logger.debug(f"Insufficient assets for correlation enhancement: {len(filtered_assets)}")
                return base_signals
            
            # Get correlation signals for this asset
            correlation_signals = await self.correlation_signal_generator.generate_correlation_signals(
                current_asset, filtered_assets, timeframe, lookback_periods=len(data)
            )
            
            # Apply correlation enhancement following genetic parameter evolution
            enhanced_signals = self._apply_correlation_enhancement(
                base_signals, correlation_signals, current_asset, filtered_assets
            )
            
            self.logger.debug(f"Enhanced {len(enhanced_signals.dropna())} signals for {current_asset}")
            return enhanced_signals
            
        except Exception as e:
            self.logger.error(f"Enhanced signal generation failed: {e}")
            # Fallback to base signals on error
            return self.generate_signals(data)
    
    def _apply_correlation_enhancement(
        self,
        base_signals: pd.Series,
        correlation_signals: pd.Series,
        current_asset: str,
        filtered_assets: List[str]
    ) -> pd.Series:
        """
        Apply correlation-based signal enhancement using genetic parameters.
        Follows existing parameter usage patterns from base_seed.py.
        
        Args:
            base_signals: Base trading signals from the genetic seed
            correlation_signals: Correlation-based signals
            current_asset: Asset symbol being analyzed
            filtered_assets: All assets in correlation analysis
            
        Returns:
            pd.Series: Enhanced signals combining base and correlation analysis
        """
        if correlation_signals.empty or base_signals.empty:
            return base_signals
        
        try:
            # Align series indices (handle different lengths)
            aligned_base, aligned_corr = base_signals.align(
                correlation_signals, 
                join='left',  # Keep all base signal timestamps
                fill_value=0.0
            )
            
            # Get genetic parameters for enhancement
            correlation_weight = self.get_parameter('correlation_weight', 0.3)
            confirmation_threshold = self.get_parameter('correlation_confirmation_threshold', 0.5)
            regime_adjustment = self.get_parameter('correlation_regime_adjustment', 0.2)
            diversification_bonus = self.get_parameter('diversification_bonus', 0.1)
            
            # Calculate enhancement components
            enhanced = aligned_base.copy()
            
            # 1. Correlation Confirmation Enhancement
            # Strengthen signals when correlation supports the direction
            correlation_confirmation = self._calculate_correlation_confirmation(
                aligned_base, aligned_corr, confirmation_threshold
            )
            
            # 2. Regime-Based Adjustment
            # Adjust signal strength based on correlation regime
            regime_adjustment_factor = self._calculate_regime_adjustment(
                aligned_corr, regime_adjustment
            )
            
            # 3. Diversification Bonus
            # Bonus for signals in low-correlation environments
            diversification_factor = self._calculate_diversification_factor(
                aligned_corr, diversification_bonus
            )
            
            # Apply correlation enhancement (weighted combination)
            base_weight = 1.0 - correlation_weight
            
            # Combine base signals with correlation enhancement
            enhanced = (
                aligned_base * base_weight + 
                aligned_corr * correlation_weight
            )
            
            # Apply confirmation and regime adjustments
            enhanced = enhanced * correlation_confirmation * regime_adjustment_factor
            
            # Add diversification bonus
            enhanced = enhanced + (diversification_factor * abs(enhanced))
            
            # Final signal processing
            enhanced = self._finalize_enhanced_signals(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.debug(f"Correlation enhancement calculation failed: {e}")
            return base_signals
    
    def _calculate_correlation_confirmation(
        self,
        base_signals: pd.Series,
        correlation_signals: pd.Series,
        threshold: float
    ) -> pd.Series:
        """Calculate correlation confirmation factor."""
        try:
            # Signals are confirmed when they align with correlation analysis
            signal_alignment = np.sign(base_signals) * np.sign(correlation_signals)
            correlation_strength = abs(correlation_signals)
            
            # Confirmation factor: stronger when signals align and correlation is strong
            confirmation = 1.0 + (
                (signal_alignment > 0) * 
                (correlation_strength > threshold) * 
                0.3  # 30% boost for confirmed signals
            )
            
            return confirmation.fillna(1.0)
            
        except Exception:
            return pd.Series(1.0, index=base_signals.index)
    
    def _calculate_regime_adjustment(
        self,
        correlation_signals: pd.Series,
        regime_adjustment_factor: float
    ) -> pd.Series:
        """Calculate regime-based signal adjustment."""
        try:
            # Estimate current correlation regime from signal strength
            correlation_strength = abs(correlation_signals).rolling(10, min_periods=3).mean()
            
            # Adjust signals based on correlation regime
            # High correlation: reduce signal strength (crowded trades)
            # Low correlation: maintain or increase signal strength (diversification opportunities)
            regime_factor = 1.0 - (correlation_strength * regime_adjustment_factor)
            
            # Clamp adjustment factor to reasonable bounds
            regime_factor = regime_factor.clip(0.5, 1.5)
            
            return regime_factor.fillna(1.0)
            
        except Exception:
            return pd.Series(1.0, index=correlation_signals.index)
    
    def _calculate_diversification_factor(
        self,
        correlation_signals: pd.Series,
        diversification_bonus: float
    ) -> pd.Series:
        """Calculate diversification bonus factor."""
        try:
            # Higher bonus when correlation is low (more diversification potential)
            correlation_strength = abs(correlation_signals).rolling(5, min_periods=2).mean()
            
            # Inverse relationship: low correlation = high diversification bonus
            diversification = diversification_bonus * (1.0 - correlation_strength)
            
            return diversification.clip(0.0, diversification_bonus).fillna(0.0)
            
        except Exception:
            return pd.Series(0.0, index=correlation_signals.index)
    
    def _finalize_enhanced_signals(self, enhanced_signals: pd.Series) -> pd.Series:
        """Final processing of enhanced signals."""
        try:
            # Apply final constraints and smoothing
            
            # 1. Clamp signals to valid range
            enhanced_signals = enhanced_signals.clip(-1.0, 1.0)
            
            # 2. Handle NaN values
            enhanced_signals = enhanced_signals.fillna(0.0)
            
            # 3. Apply minimal smoothing to reduce noise (optional genetic parameter)
            regime_sensitivity = self.get_parameter('regime_sensitivity', 0.5)
            if regime_sensitivity < 0.8:  # Apply smoothing for low sensitivity
                smoothing_window = max(2, int(5 * (1 - regime_sensitivity)))
                enhanced_signals = enhanced_signals.rolling(
                    window=smoothing_window,
                    center=True,
                    min_periods=1
                ).mean()
            
            # 4. Final range validation
            enhanced_signals = enhanced_signals.clip(-1.0, 1.0)
            
            return enhanced_signals
            
        except Exception as e:
            self.logger.debug(f"Signal finalization failed: {e}")
            return enhanced_signals.fillna(0.0).clip(-1.0, 1.0)
    
    def get_correlation_enhancement_summary(
        self,
        base_signals: pd.Series,
        enhanced_signals: pd.Series
    ) -> Dict[str, Any]:
        """
        Get summary of correlation enhancement effects.
        Useful for genetic algorithm fitness evaluation.
        """
        try:
            if base_signals.empty or enhanced_signals.empty:
                return {"enhancement": "no_data"}
            
            # Calculate enhancement statistics
            base_strength = abs(base_signals).mean()
            enhanced_strength = abs(enhanced_signals).mean()
            
            signal_correlation = base_signals.corr(enhanced_signals)
            enhancement_factor = enhanced_strength / base_strength if base_strength > 0 else 1.0
            
            # Signal direction analysis
            base_direction_changes = (base_signals.diff().fillna(0) != 0).sum()
            enhanced_direction_changes = (enhanced_signals.diff().fillna(0) != 0).sum()
            
            return {
                "base_signal_strength": float(base_strength),
                "enhanced_signal_strength": float(enhanced_strength),
                "enhancement_factor": float(enhancement_factor),
                "signal_correlation": float(signal_correlation) if not pd.isna(signal_correlation) else 0.0,
                "base_direction_changes": int(base_direction_changes),
                "enhanced_direction_changes": int(enhanced_direction_changes),
                "enhancement_parameters": {
                    param: self.get_parameter(param, 0.0) 
                    for param in self.correlation_parameter_bounds.keys()
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.debug(f"Enhancement summary calculation failed: {e}")
            return {"enhancement": "calculation_failed", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for correlation-enhanced seed."""
        try:
            # Check base seed health
            base_health = {
                "base_seed_type": self.genes.seed_type.value,
                "genetic_parameters": len(self.genes.parameters),
                "correlation_parameters": len(self.correlation_parameter_bounds)
            }
            
            # Check correlation signal generator health
            signal_generator_health = await self.correlation_signal_generator.health_check()
            
            overall_health = signal_generator_health.get('status') == 'healthy'
            
            return {
                "status": "healthy" if overall_health else "degraded",
                "component": f"CorrelationEnhanced{self.__class__.__name__}",
                "base_seed": base_health,
                "signal_generator_status": signal_generator_health.get('status', 'unknown'),
                "correlation_enhancement": "enabled",
                "cache_size": len(self._correlation_cache),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": f"CorrelationEnhanced{self.__class__.__name__}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }