
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Stochastic Oscillator Genetic Seed - Seed #4
This seed implements the Stochastic Oscillator momentum indicator with genetic
parameter evolution for optimal %K and %D periods, overbought/oversold levels,
and divergence detection.
Key Features:
- Adaptive %K and %D periods through genetic evolution
- Dynamic overbought/oversold thresholds
- Stochastic divergence detection
- Multi-timeframe confirmation
"""
@genetic_seed
class StochasticOscillatorSeed(BaseSeed):
    """Stochastic Oscillator trading seed with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Stochastic_Oscillator"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Stochastic Oscillator momentum strategy. Generates signals based on "
                "%K and %D line crossovers, overbought/oversold conditions, and "
                "momentum divergences. Includes multi-timeframe confirmation.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'k_period',
            'd_period',
            'overbought_level',
            'oversold_level',
            'divergence_sensitivity'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'k_period': (10.0, 20.0),       # Tighter bounds avoid noise; K tuned for speed
            'd_period': (2.0, 5.0),         # D tuned for speed; avoid excessive smoothing
            'overbought_level': (70.0, 85.0), # Highs (85) avoid noise; standard crypto range
            'oversold_level': (15.0, 30.0),   # Lows (15) avoid noise; tighter crypto bounds
            'divergence_sensitivity': (0.0, 1.0) # Divergence detection weight
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize Stochastic Oscillator seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.MOMENTUM
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'k_period': 14.0,
                'd_period': 3.0,
                'overbought_level': 80.0,
                'oversold_level': 20.0,
                'divergence_sensitivity': 0.5
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator and related indicators.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        k_period = int(self.genes.parameters['k_period'])
        d_period = int(self.genes.parameters['d_period'])
        # Calculate %K (fast stochastic)
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        # Handle division by zero
        price_range = highest_high - lowest_low
        price_range = price_range.where(price_range != 0, 0.01)  # Avoid division by zero
        k_percent = 100 * (data['close'] - lowest_low) / price_range
        # Calculate %D (slow stochastic - moving average of %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        # Calculate momentum and velocity
        k_momentum = k_percent.diff()
        d_momentum = d_percent.diff()
        k_velocity = k_momentum.rolling(window=3).mean()
        # Overbought/oversold zones
        overbought_level = self.genes.parameters['overbought_level']
        oversold_level = self.genes.parameters['oversold_level']
        overbought_zone = k_percent >= overbought_level
        oversold_zone = k_percent <= oversold_level
        neutral_zone = (k_percent > oversold_level) & (k_percent < overbought_level)
        # Crossover signals
        k_above_d = k_percent > d_percent
        k_above_d_prev = safe_fillna_false(k_above_d.shift(1))
        k_crosses_above_d = k_above_d & (~k_above_d_prev)
        k_crosses_below_d = (~k_above_d) & k_above_d_prev
        # Divergence detection
        price_highs = data['close'].rolling(window=20).max()
        price_lows = data['close'].rolling(window=20).min()
        stoch_highs = k_percent.rolling(window=20).max()
        stoch_lows = k_percent.rolling(window=20).min()
        # Bullish divergence: price makes lower low, stochastic makes higher low
        bullish_divergence = (
            (data['close'] <= price_lows.shift(1)) & 
            (k_percent > stoch_lows.shift(1)) &
            oversold_zone
        )
        # Bearish divergence: price makes higher high, stochastic makes lower high
        bearish_divergence = (
            (data['close'] >= price_highs.shift(1)) & 
            (k_percent < stoch_highs.shift(1)) &
            overbought_zone
        )
        # Multi-timeframe stochastic (for confirmation)
        k_short = self._calculate_stochastic_k(data, max(3, k_period // 2))
        k_long = self._calculate_stochastic_k(data, min(50, k_period * 2))
        # Volume confirmation (if available)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_ma
        else:
            volume_ratio = pd.Series(1.0, index=data.index)
        # Price momentum for confirmation
        price_momentum = data['close'].pct_change(periods=3)
        return {
            'k_percent': k_percent,
            'd_percent': d_percent,
            'k_momentum': k_momentum,
            'd_momentum': d_momentum,
            'k_velocity': k_velocity,
            'overbought_zone': overbought_zone,
            'oversold_zone': oversold_zone,
            'neutral_zone': neutral_zone,
            'k_crosses_above_d': k_crosses_above_d,
            'k_crosses_below_d': k_crosses_below_d,
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'k_short': k_short,
            'k_long': k_long,
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum
        }
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate %K stochastic for given period.
        Args:
            data: OHLCV data
            period: Lookback period
        Returns:
            %K stochastic values
        """
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        price_range = highest_high - lowest_low
        price_range = price_range.where(price_range != 0, 0.01)
        return 100 * (data['close'] - lowest_low) / price_range
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Stochastic Oscillator trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        divergence_sensitivity = self.genes.parameters['divergence_sensitivity']
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        # Signal 1: Crossover signals in oversold/overbought zones
        crossover_signals = self._generate_crossover_signals(indicators)
        # Signal 2: Divergence signals
        divergence_signals = self._generate_divergence_signals(indicators)
        # Signal 3: Zone-based signals (buy oversold, sell overbought)
        zone_signals = self._generate_zone_signals(indicators)
        # Combine signals with weights
        signals = (
            crossover_signals * 0.5 +
            divergence_signals * divergence_sensitivity +
            zone_signals * (1.0 - divergence_sensitivity) * 0.5
        )
        # Apply multi-timeframe confirmation
        signals = self._apply_timeframe_confirmation(signals, indicators)
        # Ensure signals are within valid range
        signals = signals.clip(-1.0, 1.0)
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def _generate_crossover_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on %K and %D crossovers.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Crossover-based signals
        """
        signals = pd.Series(0.0, index=indicators['k_percent'].index)
        # Simplified crossover logic (following research pattern)
        # Remove restrictive zone requirements - crossovers are valuable anywhere
        bullish_crossover = (
            indicators['k_crosses_above_d'] |  # Basic crossover signal
            (indicators['k_crosses_above_d'] & indicators['oversold_zone'])  # OR enhanced in oversold
        )
        bearish_crossover = (
            indicators['k_crosses_below_d'] |  # Basic crossover signal  
            (indicators['k_crosses_below_d'] & indicators['overbought_zone'])  # OR enhanced in overbought
        )
        # Volume and momentum as signal enhancers, not requirements
        volume_confirmed = indicators['volume_ratio'] >= 1.0
        momentum_up = indicators['price_momentum'] > 0
        momentum_down = indicators['price_momentum'] < 0
        # Simplified signal generation - use OR logic for flexibility
        buy_conditions = (
            bullish_crossover |  # Basic crossover
            (bullish_crossover & volume_confirmed)  # OR volume-enhanced crossover
        )
        sell_conditions = (
            bearish_crossover |  # Basic crossover
            (bearish_crossover & volume_confirmed)  # OR volume-enhanced crossover
        )
        # Calculate signal strength based on distance from extreme levels
        buy_strength = np.where(
            buy_conditions,
            np.minimum(1.0, (self.genes.parameters['oversold_level'] - indicators['k_percent']) / 20.0 + 0.3),
            0.0
        )
        sell_strength = np.where(
            sell_conditions,
            -np.minimum(1.0, (indicators['k_percent'] - self.genes.parameters['overbought_level']) / 20.0 + 0.3),
            0.0
        )
        signals = pd.Series(buy_strength + sell_strength, index=indicators['k_percent'].index)
        return signals
    def _generate_divergence_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on stochastic divergences.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Divergence-based signals
        """
        signals = pd.Series(0.0, index=indicators['k_percent'].index)
        # Bullish divergence signals
        bullish_div_strength = np.where(
            indicators['bullish_divergence'],
            0.8,  # Strong signal
            0.0
        )
        # Bearish divergence signals
        bearish_div_strength = np.where(
            indicators['bearish_divergence'],
            -0.8,  # Strong signal
            0.0
        )
        signals = pd.Series(bullish_div_strength + bearish_div_strength, 
                          index=indicators['k_percent'].index)
        return signals
    def _generate_zone_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on overbought/oversold zones.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Zone-based signals
        """
        signals = pd.Series(0.0, index=indicators['k_percent'].index)
        # Simplified zone logic (following research pattern)
        # Basic zone signals with optional momentum confirmation
        oversold_buy = (
            indicators['oversold_zone'] |  # Basic oversold signal
            (indicators['oversold_zone'] & (indicators['k_velocity'] > 0))  # OR momentum-confirmed
        )
        overbought_sell = (
            indicators['overbought_zone'] |  # Basic overbought signal
            (indicators['overbought_zone'] & (indicators['k_velocity'] < 0))  # OR momentum-confirmed
        )
        # Signal strength based on how extreme the levels are
        buy_strength = np.where(
            oversold_buy,
            np.minimum(0.6, (self.genes.parameters['oversold_level'] - indicators['k_percent']) / 30.0 + 0.2),
            0.0
        )
        sell_strength = np.where(
            overbought_sell,
            -np.minimum(0.6, (indicators['k_percent'] - self.genes.parameters['overbought_level']) / 30.0 + 0.2),
            0.0
        )
        signals = pd.Series(buy_strength + sell_strength, index=indicators['k_percent'].index)
        return signals
    def _apply_timeframe_confirmation(self, signals: pd.Series, 
                                   indicators: Dict[str, pd.Series]) -> pd.Series:
        """Apply multi-timeframe confirmation to signals.
        Args:
            signals: Base signals
            indicators: Dictionary of indicators including multi-timeframe stochastics
        Returns:
            Confirmed signals
        """
        confirmed_signals = signals.copy()
        # Multi-timeframe alignment factor
        for i in range(len(signals)):
            if abs(signals.iloc[i]) > 0.1:  # Significant signal
                # Check alignment with other timeframes
                current_k = indicators['k_percent'].iloc[i]
                short_k = indicators['k_short'].iloc[i]
                long_k = indicators['k_long'].iloc[i]
                # Calculate alignment score
                alignment_score = 1.0
                # Reduce signal if timeframes disagree
                if signals.iloc[i] > 0:  # Buy signal
                    if short_k < current_k * 0.8 or long_k < current_k * 0.8:
                        alignment_score *= 0.7
                else:  # Sell signal
                    if short_k > current_k * 1.2 or long_k > current_k * 1.2:
                        alignment_score *= 0.7
                confirmed_signals.iloc[i] *= alignment_score
        return confirmed_signals
    def get_stochastic_regime(self, data: pd.DataFrame) -> str:
        """Identify current stochastic regime.
        Args:
            data: OHLCV market data
        Returns:
            String describing current stochastic regime
        """
        indicators = self.calculate_technical_indicators(data)
        current_k = indicators['k_percent'].iloc[-1] if len(data) > 0 else 50
        if current_k <= self.genes.parameters['oversold_level']:
            return "oversold"
        elif current_k >= self.genes.parameters['overbought_level']:
            return "overbought"
        elif current_k < 40:
            return "bearish"
        elif current_k > 60:
            return "bullish"
        else:
            return "neutral"
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on stochastic conditions.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get stochastic conditions
        indicators = self.calculate_technical_indicators(data)
        current_k = indicators['k_percent'].iloc[-1] if len(data) > 0 else 50
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust based on stochastic extreme level
        if signal > 0:  # Buy signal
            # Larger size when more oversold
            stoch_adjustment = max(0.5, (self.genes.parameters['oversold_level'] - current_k + 30) / 50)
        else:  # Sell signal
            # Larger size when more overbought
            stoch_adjustment = max(0.5, (current_k - self.genes.parameters['overbought_level'] + 30) / 50)
        # Adjust for signal strength
        signal_adjustment = abs(signal)
        # Calculate final position size
        position_size = base_size * signal_adjustment * stoch_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        k_period = self.genes.parameters.get('k_period', 14)
        d_period = self.genes.parameters.get('d_period', 3)
        oversold = self.genes.parameters.get('oversold_level', 20)
        overbought = self.genes.parameters.get('overbought_level', 80)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"Stoch({k_period:.0f},{d_period:.0f})[{oversold:.0f}/{overbought:.0f}]{fitness_str}"
