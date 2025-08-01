from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Bollinger Bands Genetic Seed - Seed #13
This seed implements Bollinger Bands volatility-based trading signals with genetic
algorithm parameter evolution. Optimized for crypto market volatility regimes and
position sizing adaptation.

Key Features:
- Genetic parameter evolution for lookback period and multiplier
- Volatility regime detection and adaptive position scaling
- Squeeze detection for breakout trading
- Multi-timeframe Bollinger band analysis
- Fat-tail distribution handling for crypto markets

Based on research from:
- /research/vectorbt_comprehensive/02_genetic_algorithm_integration_guide.md
- VectorBT BBANDS indicator implementation patterns
"""

@genetic_seed
class BollingerBandsSeed(BaseSeed):
    """Bollinger Bands trading seed with genetic volatility adaptation."""
    
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Bollinger_Bands"
    
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Bollinger Bands volatility-based trading strategy with genetic adaptation. "
                "Generates signals based on price position relative to dynamic volatility bands. "
                "Includes squeeze detection, volatility regime adaptation, and position scaling "
                "optimized for crypto market fat-tail distributions.")
    
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'lookback_period',
            'volatility_multiplier',
            'squeeze_threshold',
            'breakout_strength',
            'position_scaling_factor'
        ]
    
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max)."""
        return {
            'lookback_period': (10.0, 50.0),         # BB calculation period (crypto optimized)
            'volatility_multiplier': (1.5, 3.0),     # Standard deviation multiplier (fat-tail handling)
            'squeeze_threshold': (0.05, 0.25),       # Volatility squeeze detection (normalized)
            'breakout_strength': (0.01, 0.05),       # Minimum breakout strength (1-5%)
            'position_scaling_factor': (0.5, 1.5)    # Position size scaling based on regime
        }
    
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize Bollinger Bands seed.
        
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.VOLATILITY
        
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'lookback_period': 20.0,
                'volatility_multiplier': 2.0,
                'squeeze_threshold': 0.15,
                'breakout_strength': 0.02,
                'position_scaling_factor': 1.0
            }
        
        super().__init__(genes, settings)
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands and related volatility indicators.
        
        Args:
            data: OHLCV market data
            
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        lookback = int(self.genes.parameters['lookback_period'])
        multiplier = self.genes.parameters['volatility_multiplier']
        squeeze_thresh = self.genes.parameters['squeeze_threshold']
        
        # Primary Bollinger Bands calculation
        close = data['close']
        rolling_mean = close.rolling(window=lookback).mean()
        rolling_std = close.rolling(window=lookback).std()
        
        # Bollinger Band lines
        bb_upper = rolling_mean + (multiplier * rolling_std)
        bb_lower = rolling_mean - (multiplier * rolling_std)
        bb_middle = rolling_mean
        
        # Bollinger Band indicators
        bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized band width
        bb_percent = (close - bb_lower) / (bb_upper - bb_lower)  # %B indicator
        
        # Volatility squeeze detection
        bb_squeeze = bb_width < bb_width.rolling(window=20).quantile(squeeze_thresh)
        
        # Bollinger Band position
        above_upper = close > bb_upper
        below_lower = close < bb_lower
        inside_bands = (close >= bb_lower) & (close <= bb_upper)
        
        # Multi-timeframe analysis (if sufficient data)
        if len(data) > lookback * 2:
            # Longer-term Bollinger Bands for regime detection
            long_lookback = min(lookback * 2, len(data) // 3)
            long_rolling_mean = close.rolling(window=long_lookback).mean()
            long_rolling_std = close.rolling(window=long_lookback).std()
            
            bb_upper_long = long_rolling_mean + (multiplier * long_rolling_std)
            bb_lower_long = long_rolling_mean - (multiplier * long_rolling_std)
            bb_width_long = (bb_upper_long - bb_lower_long) / long_rolling_mean
        else:
            bb_width_long = bb_width
        
        # Volatility regime classification
        vol_regime_high = bb_width > bb_width.rolling(window=lookback).quantile(0.8)
        vol_regime_low = bb_width < bb_width.rolling(window=lookback).quantile(0.2)
        
        # Price momentum for breakout confirmation
        price_momentum = close.pct_change(periods=3)
        
        # Volatility acceleration
        vol_acceleration = bb_width.diff()
        expanding_bands = vol_acceleration > 0
        contracting_bands = vol_acceleration < 0
        
        return {
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'bb_width': bb_width,
            'bb_percent': bb_percent,
            'bb_squeeze': bb_squeeze,
            'above_upper': above_upper,
            'below_lower': below_lower,
            'inside_bands': inside_bands,
            'bb_width_long': bb_width_long,
            'vol_regime_high': vol_regime_high,
            'vol_regime_low': vol_regime_low,
            'price_momentum': price_momentum,
            'expanding_bands': expanding_bands,
            'contracting_bands': contracting_bands
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Bollinger Bands trading signals.
        
        Args:
            data: OHLCV market data
            
        Returns:
            Series of trading signals (-1.0 to 1.0)
        """
        # Calculate technical indicators internally (following existing pattern)
        indicators = self.calculate_technical_indicators(data)
        
        # Get genetic parameters
        breakout_strength = self.genes.parameters['breakout_strength']
        
        # Extract indicators
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_percent = indicators['bb_percent']
        bb_squeeze = indicators['bb_squeeze']
        vol_regime_high = indicators['vol_regime_high']
        vol_regime_low = indicators['vol_regime_low']
        price_momentum = indicators['price_momentum']
        expanding_bands = indicators['expanding_bands']
        
        close = data['close']
        
        # Ultra-simplified signal logic (matching RSI pattern)
        
        # Basic Bollinger Band signals (following research patterns)
        # Entry long when price breaks above upper band OR oversold
        entry_long = (
            (close > bb_upper) |            # Price breaks above upper band
            (bb_percent < 0.2)              # Oversold condition
        )
        
        # Entry short when price breaks below lower band OR overbought  
        entry_short = (
            (close < bb_lower) |            # Price breaks below lower band
            (bb_percent > 0.8)              # Overbought condition
        )
        
        # Simple exits (opposite conditions)
        exit_long = (bb_percent > 0.6)     # Exit when reaching upper area
        exit_short = (bb_percent < 0.4)    # Exit when reaching lower area
        
        # Convert to single signal series (following RSI pattern)
        signals = pd.Series(0.0, index=data.index)
        
        # Apply entry signals first
        signals[entry_long] = 1.0
        signals[entry_short] = -1.0
        
        # Only apply exits where we don't have strong entry signals
        # This prevents immediate cancellation of valid entries
        strong_entries = entry_long | entry_short
        signals[exit_long & ~strong_entries] = 0.0
        signals[exit_short & ~strong_entries] = 0.0
        
        # Apply safety filters
        signals = safe_fillna_zero(signals)
        
        return signals
    
    def calculate_position_size_multiplier(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Calculate position size multiplier based on volatility regime.
        
        Args:
            data: OHLCV market data
            indicators: Technical indicators
            
        Returns:
            Position size multiplier series
        """
        scaling_factor = self.genes.parameters['position_scaling_factor']
        
        bb_width = indicators['bb_width']
        vol_regime_high = indicators['vol_regime_high']
        vol_regime_low = indicators['vol_regime_low']
        
        # Base position size
        position_multiplier = pd.Series(1.0, index=data.index)
        
        # Scale down in high volatility regimes
        position_multiplier = position_multiplier.where(
            ~vol_regime_high, 
            position_multiplier * (scaling_factor * 0.7)
        )
        
        # Scale up in low volatility regimes (if configured)
        if scaling_factor > 1.0:
            position_multiplier = position_multiplier.where(
                ~vol_regime_low,
                position_multiplier * (scaling_factor * 1.2)
            )
        
        # Normalize to prevent extreme scaling
        position_multiplier = position_multiplier.clip(0.1, 2.0)
        
        return safe_fillna(position_multiplier, 1.0)