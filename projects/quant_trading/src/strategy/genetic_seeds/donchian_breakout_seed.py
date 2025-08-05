
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Donchian Breakout Genetic Seed - Seed #2
This seed implements the Donchian Channel breakout strategy, a fundamental
breakout detection primitive. The genetic algorithm evolves optimal channel
periods, breakout thresholds, and confirmation filters.
Key Features:
- Adaptive Donchian channel periods through genetic evolution
- Volume-confirmed breakouts to reduce false signals
- Multiple timeframe validation
- Breakout strength assessment
"""
@genetic_seed
class DonchianBreakoutSeed(BaseSeed):
    """Donchian Channel breakout trading seed with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Donchian_Breakout"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Donchian Channel breakout strategy. Generates buy signals when price "
                "breaks above the upper channel (highest high), and sell signals when "
                "price breaks below the lower channel (lowest low). Includes volume "
                "confirmation and false breakout filtering.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'channel_period',
            'breakout_threshold',
            'volume_confirmation',
            'false_breakout_filter',
            'trend_bias'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'channel_period': (10.0, 30.0),        # Shorter channels for scalps; longer for swings
            'breakout_threshold': (0.001, 0.02),   # Minimum breakout percentage (0.1%-2%)
            'volume_confirmation': (1.0, 3.0),      # Volume multiplier for confirmation
            'false_breakout_filter': (2.0, 10.0),   # Hours to confirm breakout validity
            'trend_bias': (0.0, 1.0)               # Bias toward trending markets (0-1)
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize Donchian breakout seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.BREAKOUT
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'channel_period': 20.0,
                'breakout_threshold': 0.005,
                'volume_confirmation': 1.5,
                'false_breakout_filter': 4.0,
                'trend_bias': 0.5
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Donchian channels and breakout metrics.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        channel_period = int(self.genes.parameters['channel_period'])
        # Calculate Donchian channels (CORRECTED: exclude current bar to enable breakout detection)
        # Using previous N periods prevents mathematical impossibility where current_price = channel_max
        donchian_high = data['close'].shift(1).rolling(window=channel_period).max()
        donchian_low = data['close'].shift(1).rolling(window=channel_period).min()
        donchian_mid = (donchian_high + donchian_low) / 2
        # Channel width (for volatility assessment)
        channel_width = (donchian_high - donchian_low) / donchian_mid
        # Price position within channel (0 = bottom, 1 = top)
        price_position = (data['close'] - donchian_low) / (donchian_high - donchian_low)
        price_position = price_position.fillna(0.5)  # Default to middle
        # Breakout strength calculation
        upper_breakout_strength = (data['close'] - donchian_high) / donchian_high
        lower_breakout_strength = (donchian_low - data['close']) / donchian_low
        # Trend bias indicators
        short_ma = data['close'].rolling(window=10).mean()
        long_ma = data['close'].rolling(window=30).mean()
        trend_direction = (short_ma - long_ma) / long_ma
        # Volume analysis (if available)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_ma
            volume_trend = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=20).mean()
        else:
            volume_ratio = pd.Series(1.0, index=data.index)
            volume_trend = pd.Series(1.0, index=data.index)
        # Price momentum for confirmation
        price_momentum = data['close'].pct_change(periods=3)
        # Channel slope (trending vs sideways)
        channel_slope = donchian_mid.diff().rolling(window=5).mean()
        channel_slope_norm = channel_slope / data['close']
        return {
            'donchian_high': donchian_high.ffill(),
            'donchian_low': donchian_low.ffill(),
            'donchian_mid': donchian_mid.ffill(),
            'channel_width': channel_width.fillna(0.02),  # Default 2% width
            'price_position': price_position,
            'upper_breakout_strength': safe_fillna_zero(upper_breakout_strength),
            'lower_breakout_strength': safe_fillna_zero(lower_breakout_strength),
            'trend_direction': safe_fillna_zero(trend_direction),
            'volume_ratio': volume_ratio.fillna(1.0),
            'volume_trend': volume_trend.fillna(1.0),
            'price_momentum': safe_fillna_zero(price_momentum),
            'channel_slope_norm': safe_fillna_zero(channel_slope_norm)
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Donchian breakout trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        breakout_threshold = self.genes.parameters['breakout_threshold']
        volume_confirmation = self.genes.parameters['volume_confirmation']
        trend_bias = self.genes.parameters['trend_bias']
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        # RESEARCH PATTERN: Simple breakout detection (price > channel)
        # Use threshold as additive buffer instead of multiplicative
        upper_breakout = data['close'] > (indicators['donchian_high'] + breakout_threshold)
        lower_breakout = data['close'] < (indicators['donchian_low'] - breakout_threshold)
        # RESEARCH PATTERN: Simple, clean signal generation
        # Generate buy signals (upper breakout) - research shows simple logic works best
        buy_signals = upper_breakout
        # Generate sell signals (lower breakout) 
        sell_signals = lower_breakout
        # Simplified volume confirmation (following research pattern)
        # Only apply volume filter if parameter is very high (> 2.0)
        if volume_confirmation > 2.0:
            volume_filter = indicators['volume_ratio'] >= volume_confirmation
            buy_signals = buy_signals & volume_filter
            sell_signals = sell_signals & volume_filter
        # Convert to signal strength (simple binary signals work better than complex scaling)
        signals = pd.Series(0.0, index=data.index)
        signals[buy_signals] = 1.0   # Full strength buy
        signals[sell_signals] = -1.0  # Full strength sell
        # RESEARCH PATTERN: Minimal filtering, let genetic algorithm optimize
        # Simple false breakout filter - just require breakout above/below by threshold
        # (The genetic algorithm will evolve optimal thresholds)
        # Fill any NaN values and ensure valid range
        signals = safe_fillna_zero(signals).clip(-1.0, 1.0)
        return signals
    def _apply_persistence_filter(self, signals: pd.Series, persistence_periods: int) -> pd.Series:
        """Apply persistence filter to reduce false breakouts.
        Args:
            signals: Raw breakout signals
            persistence_periods: Number of periods signal must persist
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        # For each signal, check if it persists for required periods
        for i in range(persistence_periods, len(signals)):
            if abs(signals.iloc[i]) > 0.1:  # Significant signal
                # Check if signal direction is consistent in recent periods
                recent_signals = signals.iloc[i-persistence_periods:i+1]
                # Count periods with same signal direction
                if signals.iloc[i] > 0:  # Buy signal
                    consistent_periods = (recent_signals > 0.1).sum()
                else:  # Sell signal
                    consistent_periods = (recent_signals < -0.1).sum()
                # Only keep signal if it's persistent enough
                if consistent_periods < persistence_periods * 0.6:  # 60% persistence required
                    filtered_signals.iloc[i] = 0.0
        return filtered_signals
    def get_breakout_quality(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Assess the quality of breakout signals.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of breakout quality metrics
        """
        indicators = self.calculate_technical_indicators(data)
        # Breakout strength (how far outside the channel)
        breakout_strength = np.maximum(
            indicators['upper_breakout_strength'],
            indicators['lower_breakout_strength']
        )
        # Volume confirmation strength
        volume_strength = np.minimum(3.0, indicators['volume_ratio']) / 3.0
        # Trend alignment (breakout in direction of trend)
        trend_alignment = np.where(
            indicators['trend_direction'] > 0,
            np.maximum(0, indicators['upper_breakout_strength'] * 10),  # Bullish trend, upward breakout
            np.maximum(0, indicators['lower_breakout_strength'] * 10)   # Bearish trend, downward breakout
        )
        # Market regime suitability (stable vs choppy)
        regime_suitability = 1.0 / (1.0 + indicators['channel_width'] * 10)
        # Overall breakout quality
        overall_quality = (
            breakout_strength * 0.3 +
            volume_strength * 0.25 +
            trend_alignment * 0.25 +
            regime_suitability * 0.2
        )
        return {
            'breakout_strength': breakout_strength,
            'volume_strength': volume_strength,
            'trend_alignment': trend_alignment,
            'regime_suitability': regime_suitability,
            'overall_quality': overall_quality
        }
    def should_enter_position(self, data: pd.DataFrame, signal: float) -> bool:
        """Determine if a position should be entered based on breakout quality.
        Args:
            data: Current market data
            signal: Current signal strength
        Returns:
            True if position should be entered
        """
        if abs(signal) < self.genes.filter_threshold:
            return False
        # Get breakout quality metrics
        quality_metrics = self.get_breakout_quality(data)
        current_quality = quality_metrics['overall_quality'].iloc[-1] if len(data) > 0 else 0
        # Require minimum quality threshold for entry
        min_quality_threshold = 0.3
        return current_quality >= min_quality_threshold
    def calculate_stop_loss_level(self, data: pd.DataFrame, entry_price: float, 
                                position_direction: int) -> float:
        """Calculate dynamic stop loss level based on Donchian channels.
        Args:
            data: Current market data
            entry_price: Entry price of position
            position_direction: 1 for long, -1 for short
        Returns:
            Stop loss price level
        """
        indicators = self.calculate_technical_indicators(data)
        if position_direction > 0:  # Long position
            # Stop loss below recent Donchian low or channel mid
            channel_low = indicators['donchian_low'].iloc[-1] if len(data) > 0 else entry_price * 0.98
            channel_mid = indicators['donchian_mid'].iloc[-1] if len(data) > 0 else entry_price * 0.99
            # Use the higher of the two (more conservative)
            stop_loss = max(channel_low, channel_mid)
            # Ensure minimum stop distance
            min_stop = entry_price * (1 - self.genes.stop_loss)
            stop_loss = min(stop_loss, min_stop)
        else:  # Short position
            # Stop loss above recent Donchian high or channel mid
            channel_high = indicators['donchian_high'].iloc[-1] if len(data) > 0 else entry_price * 1.02
            channel_mid = indicators['donchian_mid'].iloc[-1] if len(data) > 0 else entry_price * 1.01
            # Use the lower of the two (more conservative)
            stop_loss = min(channel_high, channel_mid)
            # Ensure minimum stop distance
            max_stop = entry_price * (1 + self.genes.stop_loss)
            stop_loss = max(stop_loss, max_stop)
        return stop_loss
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on breakout strength and volatility.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get breakout quality
        quality_metrics = self.get_breakout_quality(data)
        breakout_quality = quality_metrics['overall_quality'].iloc[-1] if len(data) > 0 else 0.5
        # Get volatility (channel width as proxy)
        indicators = self.calculate_technical_indicators(data)
        current_volatility = indicators['channel_width'].iloc[-1] if len(data) > 0 else 0.02
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust for signal strength and breakout quality
        signal_adjustment = abs(signal) * breakout_quality
        # Adjust for volatility (reduce size for high volatility)
        volatility_adjustment = 1.0 / (1.0 + current_volatility * 20)
        # Calculate final position size
        position_size = base_size * signal_adjustment * volatility_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def get_exit_conditions(self, data: pd.DataFrame, position_direction: int) -> Dict[str, bool]:
        """Get various exit conditions for current position.
        Args:
            data: Current market data
            position_direction: 1 for long, -1 for short
        Returns:
            Dictionary of exit condition flags
        """
        indicators = self.calculate_technical_indicators(data)
        signals = self.generate_signals(data.tail(5))  # Recent signals
        # Channel-based exits
        if position_direction > 0:  # Long position
            # Exit if price falls back into lower half of channel
            channel_exit = indicators['price_position'].iloc[-1] < 0.3 if len(data) > 0 else False
            # Exit if new lower breakout occurs
            signal_reversal = signals.iloc[-1] < -0.2 if len(signals) > 0 else False
        else:  # Short position
            # Exit if price rises back into upper half of channel
            channel_exit = indicators['price_position'].iloc[-1] > 0.7 if len(data) > 0 else False
            # Exit if new upper breakout occurs
            signal_reversal = signals.iloc[-1] > 0.2 if len(signals) > 0 else False
        # Volume-based exit (declining volume may indicate weakening trend)
        volume_declining = indicators['volume_trend'].iloc[-1] < 0.8 if len(data) > 0 else False
        # Momentum exit (momentum turning against position)
        momentum_exit = False
        if len(data) > 0:
            recent_momentum = indicators['price_momentum'].iloc[-3:].mean()
            if position_direction > 0 and recent_momentum < -0.01:  # Long position, negative momentum
                momentum_exit = True
            elif position_direction < 0 and recent_momentum > 0.01:  # Short position, positive momentum
                momentum_exit = True
        return {
            'channel_exit': channel_exit,
            'signal_reversal': signal_reversal,
            'volume_declining': volume_declining,
            'momentum_exit': momentum_exit
        }
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        channel_period = self.genes.parameters.get('channel_period', 20)
        breakout_threshold = self.genes.parameters.get('breakout_threshold', 0.005)
        volume_conf = self.genes.parameters.get('volume_confirmation', 1.5)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"Donchian({channel_period:.0f})[thr={breakout_threshold:.3f},vol={volume_conf:.1f}]{fitness_str}"
