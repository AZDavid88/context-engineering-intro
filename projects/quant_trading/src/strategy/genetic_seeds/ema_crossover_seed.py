
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
EMA Crossover Genetic Seed - Seed #1
This seed implements the Exponential Moving Average crossover strategy,
one of the most fundamental momentum-based trading primitives. The genetic
algorithm evolves optimal EMA periods and thresholds.
Key Features:
- Dual EMA crossover with genetic parameter evolution
- Configurable fast/slow periods and confirmation thresholds
- Momentum filter to reduce false signals
- Risk management integration
"""
@genetic_seed
class EMACrossoverSeed(BaseSeed):
    """EMA Crossover trading seed with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "EMA_Crossover"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Exponential Moving Average crossover strategy. Generates buy signals when "
                "fast EMA crosses above slow EMA, and sell signals on the reverse crossover. "
                "Includes momentum filter and risk management.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'fast_ema_period',
            'slow_ema_period', 
            'momentum_threshold',
            'signal_strength',
            'trend_filter'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'fast_ema_period': (5.0, 15.0),     # Fast EMA period (crypto-optimized: 3x faster response)
            'slow_ema_period': (18.0, 34.0),    # Slow EMA period (crypto-appropriate timing)
            'momentum_threshold': (0.001, 0.05), # Minimum momentum for signal (0.1%-5%)
            'signal_strength': (0.1, 1.0),      # Signal strength multiplier
            'trend_filter': (0.0, 0.02)         # Trend filter threshold (0-2%)
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize EMA crossover seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.MOMENTUM
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'fast_ema_period': 12.0,
                'slow_ema_period': 26.0,
                'momentum_threshold': 0.01,
                'signal_strength': 0.5,
                'trend_filter': 0.005
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate EMA indicators and momentum metrics.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        fast_period = int(self.genes.parameters['fast_ema_period'])
        slow_period = int(self.genes.parameters['slow_ema_period'])
        # Ensure fast period is less than slow period
        if fast_period >= slow_period:
            fast_period = max(5, slow_period - 5)
        # Calculate EMAs using pandas (primary method from PRP)
        fast_ema = data['close'].ewm(span=fast_period, adjust=True).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=True).mean()
        # Calculate momentum and trend metrics
        price_momentum = data['close'].pct_change(periods=5)  # 5-period momentum
        ema_spread = (fast_ema - slow_ema) / slow_ema  # Normalized EMA spread
        ema_slope_fast = fast_ema.diff()  # Fast EMA slope
        ema_slope_slow = slow_ema.diff()  # Slow EMA slope
        # Volume-weighted momentum (if volume available)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_ma
        else:
            volume_ratio = pd.Series(1.0, index=data.index)
        return {
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'ema_spread': ema_spread,
            'price_momentum': price_momentum,
            'ema_slope_fast': ema_slope_fast,
            'ema_slope_slow': ema_slope_slow,
            'volume_ratio': volume_ratio
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate EMA crossover trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        fast_ema = indicators['fast_ema']
        slow_ema = indicators['slow_ema']
        ema_spread = indicators['ema_spread']
        price_momentum = indicators['price_momentum']
        volume_ratio = indicators['volume_ratio']
        # Get genetic parameters
        momentum_threshold = self.genes.parameters['momentum_threshold']
        signal_strength = self.genes.parameters['signal_strength']
        trend_filter = self.genes.parameters['trend_filter']
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        # Ultra-simplified crossover signals (matching working seeds)
        crossover_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        crossover_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        
        # Simple trend following (no complex filters)
        buy_conditions = (
            crossover_up |                          # EMA crossover up
            (fast_ema > slow_ema * 1.005)          # Fast EMA significantly above slow
        )
        
        sell_conditions = (
            crossover_down |                        # EMA crossover down  
            (fast_ema < slow_ema * 0.995)          # Fast EMA significantly below slow
        )
        # Apply signal strength (fix: use actual signal_strength value)
        signals[buy_conditions] = 1.0  # Full strength buy signal
        signals[sell_conditions] = -1.0  # Full strength sell signal
        # Apply genetic signal strength scaling
        signals = signals * signal_strength
        # Ensure signals are within valid range
        signals = signals.clip(-1.0, 1.0)
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on market conditions.
        Args:
            data: OHLCV market data
        Returns:
            Series of signal strength values (0-1)
        """
        indicators = self.calculate_technical_indicators(data)
        # Factors that increase signal strength:
        # 1. Large EMA spread (strong trend)
        # 2. High momentum
        # 3. Volume confirmation
        # 4. Consistent EMA slope direction
        ema_spread_strength = np.abs(indicators['ema_spread']).clip(0, 0.05) / 0.05
        momentum_strength = np.abs(indicators['price_momentum']).clip(0, 0.05) / 0.05
        volume_strength = indicators['volume_ratio'].clip(1.0, 3.0) / 3.0
        # Slope consistency (both EMAs moving in same direction)
        slope_consistency = (
            (indicators['ema_slope_fast'] * indicators['ema_slope_slow'] > 0).astype(float)
        )
        # Combine strength factors
        signal_strength = (
            ema_spread_strength * 0.4 +
            momentum_strength * 0.3 +
            volume_strength * 0.2 +
            slope_consistency * 0.1
        )
        return signal_strength.clip(0.0, 1.0)
    def get_risk_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate risk metrics for position sizing and risk management.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of risk metrics
        """
        indicators = self.calculate_technical_indicators(data)
        # Volatility-based risk (ATR proxy)
        price_volatility = data['close'].rolling(window=14).std() / data['close']
        # Trend strength (higher = lower risk)
        trend_strength = np.abs(indicators['ema_spread'])
        # Signal confidence (based on multiple confirmations)
        signal_strength = self.get_signal_strength(data)
        return {
            'volatility': price_volatility,
            'trend_strength': trend_strength,
            'signal_confidence': signal_strength
        }
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on signal strength and risk metrics.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get current risk metrics
        risk_metrics = self.get_risk_metrics(data)
        current_volatility = risk_metrics['volatility'].iloc[-1] if len(risk_metrics['volatility']) > 0 else 0.02
        signal_confidence = risk_metrics['signal_confidence'].iloc[-1] if len(risk_metrics['signal_confidence']) > 0 else 0.5
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust for signal strength and confidence
        signal_adjustment = abs(signal) * signal_confidence
        # Adjust for volatility (lower position size for higher volatility)
        volatility_adjustment = 1.0 / (1.0 + current_volatility * 50)  # Scale volatility impact
        # Calculate final position size
        position_size = base_size * signal_adjustment * volatility_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def should_exit_position(self, data: pd.DataFrame, current_position: float, 
                           entry_price: float, current_price: float) -> bool:
        """Determine if current position should be exited.
        Args:
            data: Current market data
            current_position: Current position size (positive for long, negative for short)
            entry_price: Entry price of the position
            current_price: Current market price
        Returns:
            True if position should be exited
        """
        # Calculate current return
        if current_position > 0:  # Long position
            return_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            return_pct = (entry_price - current_price) / entry_price
        # Exit conditions:
        # 1. Stop loss hit
        if return_pct <= -self.genes.stop_loss:
            return True
        # 2. Take profit hit
        if return_pct >= self.genes.take_profit:
            return True
        # 3. Signal reversal
        current_signals = self.generate_signals(data.tail(10))  # Last 10 bars
        latest_signal = current_signals.iloc[-1]
        # Exit if signal reverses direction
        if current_position > 0 and latest_signal < -0.1:  # Long position, sell signal
            return True
        elif current_position < 0 and latest_signal > 0.1:  # Short position, buy signal
            return True
        return False
    def get_entry_quality(self, data: pd.DataFrame) -> float:
        """Assess the quality of the current entry opportunity.
        Args:
            data: OHLCV market data
        Returns:
            Entry quality score (0-1, higher is better)
        """
        indicators = self.calculate_technical_indicators(data)
        risk_metrics = self.get_risk_metrics(data)
        # Quality factors:
        # 1. Signal strength
        signal_strength = self.get_signal_strength(data).iloc[-1] if len(data) > 0 else 0
        # 2. Trend consistency (EMAs aligned)
        fast_ema = indicators['fast_ema'].iloc[-1] if len(indicators['fast_ema']) > 0 else 0
        slow_ema = indicators['slow_ema'].iloc[-1] if len(indicators['slow_ema']) > 0 else 0
        trend_alignment = 1.0 if abs(fast_ema - slow_ema) > 0 else 0.5
        # 3. Volume confirmation
        volume_conf = min(1.0, indicators['volume_ratio'].iloc[-1] if len(indicators['volume_ratio']) > 0 else 1.0)
        # 4. Low volatility (more predictable)
        volatility = risk_metrics['volatility'].iloc[-1] if len(risk_metrics['volatility']) > 0 else 0.02
        volatility_score = max(0.0, 1.0 - volatility * 20)  # Lower volatility = higher score
        # Combine quality factors
        entry_quality = (
            signal_strength * 0.4 +
            trend_alignment * 0.3 +
            volume_conf * 0.2 +
            volatility_score * 0.1
        )
        return min(1.0, max(0.0, entry_quality))
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        fast_period = self.genes.parameters.get('fast_ema_period', 12)
        slow_period = self.genes.parameters.get('slow_ema_period', 26)
        momentum = self.genes.parameters.get('momentum_threshold', 0.01)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"EMA({fast_period:.0f},{slow_period:.0f})[mom={momentum:.3f}]{fitness_str}"
