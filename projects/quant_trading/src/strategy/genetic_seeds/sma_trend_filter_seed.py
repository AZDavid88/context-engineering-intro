
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
SMA Trend Filter Genetic Seed - Seed #5
This seed implements Simple Moving Average-based trend filtering with genetic
parameter evolution for optimal trend detection, direction bias, and momentum confirmation.
Key Features:
- Multi-timeframe SMA trend analysis through genetic evolution
- Adaptive trend strength measurement and direction bias
- Trend reversal detection with momentum confirmation
- Dynamic filtering strength based on market conditions
"""
@genetic_seed
class SMATrendFilterSeed(BaseSeed):
    """SMA Trend Filter trading seed with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "SMA_Trend_Filter"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("SMA trend filtering strategy. Identifies and filters trades based on "
                "multiple timeframe SMA trends, trend strength, and momentum confirmation. "
                "Adapts filtering intensity based on trend clarity and market conditions.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'fast_sma_period',
            'slow_sma_period',
            'trend_strength_period',
            'filter_sensitivity',
            'momentum_confirmation'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'fast_sma_period': (30.0, 80.0),       # Smaller SMAs for fast assets; avoid noise
            'slow_sma_period': (150.0, 300.0),     # Larger SMAs for big-cap trend clarity
            'trend_strength_period': (10.0, 100.0), # Trend strength lookback (10-100 bars)
            'filter_sensitivity': (0.0, 1.0),       # Filter sensitivity (0=weak, 1=strong)
            'momentum_confirmation': (0.0, 1.0)     # Momentum confirmation weight
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize SMA Trend Filter seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.TREND_FOLLOWING
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'fast_sma_period': 50.0,    # Updated to fit new bounds [30.0, 80.0]
                'slow_sma_period': 200.0,   # Updated to fit new bounds [150.0, 300.0]
                'trend_strength_period': 30.0,
                'filter_sensitivity': 0.7,
                'momentum_confirmation': 0.5
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate SMA trend indicators and related metrics.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        fast_period = int(self.genes.parameters['fast_sma_period'])
        slow_period = int(self.genes.parameters['slow_sma_period'])
        strength_period = int(self.genes.parameters['trend_strength_period'])
        # Calculate SMAs
        fast_sma = data['close'].rolling(window=fast_period).mean()
        slow_sma = data['close'].rolling(window=slow_period).mean()
        # Price position relative to SMAs
        price_above_fast = data['close'] > fast_sma
        price_above_slow = data['close'] > slow_sma
        fast_above_slow = fast_sma > slow_sma
        # Trend direction signals
        bullish_trend = price_above_fast & price_above_slow & fast_above_slow
        bearish_trend = ~price_above_fast & ~price_above_slow & ~fast_above_slow
        mixed_trend = ~(bullish_trend | bearish_trend)
        # Trend strength calculation
        trend_strength = self._calculate_trend_strength(data, fast_sma, slow_sma, strength_period)
        # SMA slope analysis
        fast_slope = fast_sma.diff(periods=3)
        slow_slope = slow_sma.diff(periods=3)
        # Trend momentum
        trend_momentum = (fast_sma / slow_sma - 1) * 100  # Percentage difference
        trend_acceleration = trend_momentum.diff()
        # Distance metrics
        price_distance_fast = (data['close'] / fast_sma - 1) * 100
        price_distance_slow = (data['close'] / slow_sma - 1) * 100
        sma_distance = (fast_sma / slow_sma - 1) * 100
        # Trend reversal signals
        trend_reversals = self._detect_trend_reversals(fast_sma, slow_sma, trend_momentum)
        # Multi-timeframe trend confirmation
        long_sma = data['close'].rolling(window=min(500, slow_period * 3)).mean()
        long_term_trend = data['close'] > long_sma
        # Volume confirmation (if available)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_trend = data['volume'] > volume_ma
        else:
            volume_trend = pd.Series(True, index=data.index)
        # Price momentum for confirmation
        price_momentum = data['close'].pct_change(periods=5).rolling(window=3).mean()
        # Volatility-based trend clarity
        volatility = data['close'].pct_change().rolling(window=20).std()
        trend_clarity = trend_strength / (volatility + 0.001)  # Avoid division by zero
        return {
            'fast_sma': fast_sma,
            'slow_sma': slow_sma,
            'long_sma': long_sma,
            'price_above_fast': price_above_fast,
            'price_above_slow': price_above_slow,
            'fast_above_slow': fast_above_slow,
            'bullish_trend': bullish_trend,
            'bearish_trend': bearish_trend,
            'mixed_trend': mixed_trend,
            'trend_strength': trend_strength,
            'fast_slope': fast_slope,
            'slow_slope': slow_slope,
            'trend_momentum': trend_momentum,
            'trend_acceleration': trend_acceleration,
            'price_distance_fast': price_distance_fast,
            'price_distance_slow': price_distance_slow,
            'sma_distance': sma_distance,
            'trend_reversals': trend_reversals,
            'long_term_trend': long_term_trend,
            'volume_trend': volume_trend,
            'price_momentum': price_momentum,
            'trend_clarity': trend_clarity,
            'volatility': volatility
        }
    def _calculate_trend_strength(self, data: pd.DataFrame, fast_sma: pd.Series, 
                                slow_sma: pd.Series, period: int) -> pd.Series:
        """Calculate trend strength based on SMA relationships and price action.
        Args:
            data: OHLCV data
            fast_sma: Fast SMA values
            slow_sma: Slow SMA values
            period: Lookback period for strength calculation
        Returns:
            Trend strength values (0-1 scale)
        """
        # SMA separation as trend strength indicator
        sma_separation = abs(fast_sma - slow_sma) / slow_sma
        # Price momentum alignment with trend
        price_momentum = data['close'].pct_change(periods=3)
        sma_momentum = fast_sma.pct_change(periods=3)
        momentum_alignment = (price_momentum * sma_momentum).rolling(window=5).mean()
        # Trend consistency (how often price stays on correct side of SMA)
        price_above_fast = data['close'] > fast_sma
        trend_consistency = price_above_fast.rolling(window=period).mean()
        # Normalize to 0-1 where 0.5 is neutral
        trend_consistency = abs(trend_consistency - 0.5) * 2
        # Combine strength components
        strength_components = [
            np.clip(sma_separation * 50, 0, 1),  # SMA separation
            np.clip(abs(momentum_alignment) * 10, 0, 1),  # Momentum alignment
            trend_consistency  # Trend consistency
        ]
        trend_strength = pd.concat(strength_components, axis=1).mean(axis=1)
        return safe_fillna_zero(trend_strength)
    def _detect_trend_reversals(self, fast_sma: pd.Series, slow_sma: pd.Series, 
                              trend_momentum: pd.Series) -> Dict[str, pd.Series]:
        """Detect potential trend reversal signals.
        Args:
            fast_sma: Fast SMA values
            slow_sma: Slow SMA values
            trend_momentum: Trend momentum values
        Returns:
            Dictionary of reversal signals
        """
        # SMA crossovers
        fast_crosses_above_slow = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        fast_crosses_below_slow = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        # Momentum divergence (momentum weakening while trend continues)
        momentum_declining = trend_momentum < trend_momentum.shift(3)
        momentum_accelerating = trend_momentum > trend_momentum.shift(3)
        # Trend exhaustion (extreme momentum readings)
        momentum_extreme_high = trend_momentum > trend_momentum.rolling(window=50).quantile(0.9)
        momentum_extreme_low = trend_momentum < trend_momentum.rolling(window=50).quantile(0.1)
        return {
            'bullish_crossover': fast_crosses_above_slow,
            'bearish_crossover': fast_crosses_below_slow,
            'momentum_weakening': momentum_declining,
            'momentum_strengthening': momentum_accelerating,
            'trend_exhaustion_bull': momentum_extreme_high & momentum_declining,
            'trend_exhaustion_bear': momentum_extreme_low & momentum_accelerating
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate SMA trend filter signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        filter_sensitivity = self.genes.parameters['filter_sensitivity']
        momentum_confirmation = self.genes.parameters['momentum_confirmation']
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        # Signal 1: Trend direction signals
        trend_signals = self._generate_trend_direction_signals(indicators)
        # Signal 2: Trend strength filters
        strength_filters = self._generate_strength_filters(indicators)
        # Signal 3: Momentum confirmation signals
        momentum_signals = self._generate_momentum_confirmation_signals(indicators)
        # Signal 4: Reversal signals
        reversal_signals = self._generate_reversal_signals(indicators)
        # Combine signals with weights
        signals = (
            trend_signals * (1.0 - filter_sensitivity) * 0.4 +
            strength_filters * filter_sensitivity * 0.3 +
            momentum_signals * momentum_confirmation * 0.2 +
            reversal_signals * 0.1
        )
        # Apply multi-timeframe confirmation
        signals = self._apply_trend_confirmation(signals, indicators)
        # Ensure signals are within valid range
        signals = signals.clip(-1.0, 1.0)
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def _generate_trend_direction_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on trend direction.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Trend direction signals
        """
        signals = pd.Series(0.0, index=indicators['fast_sma'].index)
        # Strong bullish trend
        strong_bullish = (
            indicators['bullish_trend'] &
            (indicators['trend_strength'] > 0.6) &
            (indicators['fast_slope'] > 0) &
            (indicators['slow_slope'] > 0)
        )
        # Strong bearish trend
        strong_bearish = (
            indicators['bearish_trend'] &
            (indicators['trend_strength'] > 0.6) &
            (indicators['fast_slope'] < 0) &
            (indicators['slow_slope'] < 0)
        )
        # Moderate bullish trend
        moderate_bullish = (
            indicators['bullish_trend'] &
            (indicators['trend_strength'] > 0.3) &
            (indicators['price_distance_fast'] > 0)
        )
        # Moderate bearish trend
        moderate_bearish = (
            indicators['bearish_trend'] &
            (indicators['trend_strength'] > 0.3) &
            (indicators['price_distance_fast'] < 0)
        )
        # Calculate signal strength based on trend characteristics
        bullish_strength = np.where(
            strong_bullish,
            np.minimum(1.0, indicators['trend_strength'] * 1.5),
            np.where(
                moderate_bullish,
                np.minimum(0.7, indicators['trend_strength'] * 1.2),
                0.0
            )
        )
        bearish_strength = np.where(
            strong_bearish,
            -np.minimum(1.0, indicators['trend_strength'] * 1.5),
            np.where(
                moderate_bearish,
                -np.minimum(0.7, indicators['trend_strength'] * 1.2),
                0.0
            )
        )
        signals = pd.Series(bullish_strength + bearish_strength, index=indicators['fast_sma'].index)
        return signals
    def _generate_strength_filters(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on trend strength filtering.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Strength-filtered signals
        """
        signals = pd.Series(0.0, index=indicators['fast_sma'].index)
        # High strength trend signals
        high_strength_bull = (
            (indicators['trend_strength'] > 0.7) &
            (indicators['trend_clarity'] > indicators['trend_clarity'].rolling(window=20).median()) &
            indicators['bullish_trend']
        )
        high_strength_bear = (
            (indicators['trend_strength'] > 0.7) &
            (indicators['trend_clarity'] > indicators['trend_clarity'].rolling(window=20).median()) &
            indicators['bearish_trend']
        )
        # Medium strength with confirmation
        medium_strength_bull = (
            (indicators['trend_strength'] > 0.4) &
            (indicators['trend_momentum'] > 0) &
            (indicators['sma_distance'] > 0.5) &
            indicators['bullish_trend']
        )
        medium_strength_bear = (
            (indicators['trend_strength'] > 0.4) &
            (indicators['trend_momentum'] < 0) &
            (indicators['sma_distance'] < -0.5) &
            indicators['bearish_trend']
        )
        # Calculate filtered signal strength
        bull_filter_strength = np.where(
            high_strength_bull,
            0.9,
            np.where(medium_strength_bull, 0.6, 0.0)
        )
        bear_filter_strength = np.where(
            high_strength_bear,
            -0.9,
            np.where(medium_strength_bear, -0.6, 0.0)
        )
        signals = pd.Series(bull_filter_strength + bear_filter_strength, 
                          index=indicators['fast_sma'].index)
        return signals
    def _generate_momentum_confirmation_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on momentum confirmation.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Momentum-confirmed signals
        """
        signals = pd.Series(0.0, index=indicators['fast_sma'].index)
        # Accelerating bullish momentum
        accelerating_bull = (
            (indicators['trend_acceleration'] > 0) &
            (indicators['price_momentum'] > 0) &
            (indicators['volume_trend']) &
            indicators['bullish_trend']
        )
        # Accelerating bearish momentum
        accelerating_bear = (
            (indicators['trend_acceleration'] < 0) &
            (indicators['price_momentum'] < 0) &
            (indicators['volume_trend']) &
            indicators['bearish_trend']
        )
        # Sustained momentum
        sustained_bull = (
            (indicators['trend_momentum'] > indicators['trend_momentum'].rolling(window=10).mean()) &
            (indicators['price_distance_fast'] > 1.0) &
            indicators['bullish_trend']
        )
        sustained_bear = (
            (indicators['trend_momentum'] < indicators['trend_momentum'].rolling(window=10).mean()) &
            (indicators['price_distance_fast'] < -1.0) &
            indicators['bearish_trend']
        )
        # Calculate momentum signal strength
        momentum_bull_strength = np.where(
            accelerating_bull,
            0.8,
            np.where(sustained_bull, 0.5, 0.0)
        )
        momentum_bear_strength = np.where(
            accelerating_bear,
            -0.8,
            np.where(sustained_bear, -0.5, 0.0)
        )
        signals = pd.Series(momentum_bull_strength + momentum_bear_strength,
                          index=indicators['fast_sma'].index)
        return signals
    def _generate_reversal_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate trend reversal signals.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Reversal signals
        """
        signals = pd.Series(0.0, index=indicators['fast_sma'].index)
        reversals = indicators['trend_reversals']
        # Bullish reversal signals
        bullish_reversal = (
            reversals['bullish_crossover'] |
            reversals['trend_exhaustion_bear']
        )
        # Bearish reversal signals
        bearish_reversal = (
            reversals['bearish_crossover'] |
            reversals['trend_exhaustion_bull']
        )
        # Reversal signal strength based on context
        reversal_bull_strength = np.where(
            bullish_reversal & (indicators['trend_strength'] > 0.3),
            0.7,
            0.0
        )
        reversal_bear_strength = np.where(
            bearish_reversal & (indicators['trend_strength'] > 0.3),
            -0.7,
            0.0
        )
        signals = pd.Series(reversal_bull_strength + reversal_bear_strength,
                          index=indicators['fast_sma'].index)
        return signals
    def _apply_trend_confirmation(self, signals: pd.Series, 
                                indicators: Dict[str, pd.Series]) -> pd.Series:
        """Apply multi-timeframe trend confirmation.
        Args:
            signals: Base signals
            indicators: Dictionary of indicators
        Returns:
            Confirmed signals
        """
        confirmed_signals = signals.copy()
        # Long-term trend confirmation
        for i in range(len(signals)):
            if abs(signals.iloc[i]) > 0.1:  # Significant signal
                # Check long-term trend alignment
                long_term_bullish = indicators['long_term_trend'].iloc[i]
                current_signal = signals.iloc[i]
                # Reduce signal if opposing long-term trend
                if current_signal > 0 and not long_term_bullish:
                    confirmed_signals.iloc[i] *= 0.6  # Reduce bullish signal in bear market
                elif current_signal < 0 and long_term_bullish:
                    confirmed_signals.iloc[i] *= 0.6  # Reduce bearish signal in bull market
                # Enhance signal if aligned with long-term trend
                elif current_signal > 0 and long_term_bullish:
                    confirmed_signals.iloc[i] *= 1.2  # Enhance bullish signal in bull market
                elif current_signal < 0 and not long_term_bullish:
                    confirmed_signals.iloc[i] *= 1.2  # Enhance bearish signal in bear market
                # Ensure we don't exceed bounds
                confirmed_signals.iloc[i] = np.clip(confirmed_signals.iloc[i], -1.0, 1.0)
        return confirmed_signals
    def get_trend_regime(self, data: pd.DataFrame) -> str:
        """Identify current trend regime.
        Args:
            data: OHLCV market data
        Returns:
            String describing current trend regime
        """
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return "unknown"
        current_strength = indicators['trend_strength'].iloc[-1]
        current_bullish = indicators['bullish_trend'].iloc[-1]
        current_bearish = indicators['bearish_trend'].iloc[-1]
        if current_strength > 0.7:
            if current_bullish:
                return "strong_bullish"
            elif current_bearish:
                return "strong_bearish"
        elif current_strength > 0.4:
            if current_bullish:
                return "moderate_bullish"
            elif current_bearish:
                return "moderate_bearish"
        elif current_strength > 0.2:
            return "weak_trend"
        else:
            return "sideways"
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on trend conditions.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get trend conditions
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return 0.0
        current_strength = indicators['trend_strength'].iloc[-1]
        current_clarity = indicators['trend_clarity'].iloc[-1]
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust based on trend strength and clarity
        strength_adjustment = max(0.3, min(1.5, current_strength * 2))
        clarity_adjustment = max(0.5, min(1.3, current_clarity / indicators['trend_clarity'].median()))
        # Adjust for signal strength
        signal_adjustment = abs(signal)
        # Calculate final position size
        position_size = base_size * signal_adjustment * strength_adjustment * clarity_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        fast_period = self.genes.parameters.get('fast_sma_period', 20)
        slow_period = self.genes.parameters.get('slow_sma_period', 50)
        sensitivity = self.genes.parameters.get('filter_sensitivity', 0.7)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"SMATrend({fast_period:.0f},{slow_period:.0f})[{sensitivity:.2f}]{fitness_str}"
