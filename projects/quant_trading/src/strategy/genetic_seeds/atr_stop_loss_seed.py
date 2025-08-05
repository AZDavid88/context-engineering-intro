
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
ATR Stop Loss Genetic Seed - Seed #6
This seed implements Average True Range (ATR) based stop loss and position sizing
with genetic parameter evolution for optimal risk management and volatility adaptation.
Key Features:
- Dynamic ATR-based stop loss calculation through genetic evolution
- Volatility-adaptive position sizing and risk management
- Trailing stop loss with ATR expansion/contraction
- Multiple exit strategies based on risk-reward ratios
"""
@genetic_seed
class ATRStopLossSeed(BaseSeed):
    """ATR Stop Loss trading seed with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "ATR_Stop_Loss"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("ATR-based risk management strategy. Uses Average True Range for "
                "dynamic stop loss placement, position sizing, and volatility-adaptive "
                "exits. Combines multiple ATR timeframes for robust risk control.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'atr_period',
            'stop_loss_multiplier',
            'trailing_stop_multiplier',
            'position_size_atr_factor',
            'volatility_adjustment'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'atr_period': (10.0, 20.0),             # ATR calculation period (crypto noise reduction)
            'stop_loss_multiplier': (1.2, 2.5),     # ATR multiplier for stop loss (crypto volatility survival)
            'trailing_stop_multiplier': (0.3, 3.0), # ATR multiplier for trailing stops
            'position_size_atr_factor': (0.1, 2.0), # ATR factor for position sizing
            'volatility_adjustment': (0.0, 1.0)     # Volatility regime adjustment weight
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize ATR Stop Loss seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.RISK_MANAGEMENT
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'atr_period': 14.0,
                'stop_loss_multiplier': 2.0,
                'trailing_stop_multiplier': 1.5,
                'position_size_atr_factor': 0.5,
                'volatility_adjustment': 0.7
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate ATR and related risk management indicators.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        atr_period = int(self.genes.parameters['atr_period'])
        stop_multiplier = self.genes.parameters['stop_loss_multiplier']
        trailing_multiplier = self.genes.parameters['trailing_stop_multiplier']
        # Calculate True Range
        true_range = self._calculate_true_range(data)
        # Calculate ATR (Average True Range)
        atr = true_range.rolling(window=atr_period).mean()
        # Calculate ATR percentages relative to price
        atr_percentage = (atr / data['close']) * 100
        # Multi-timeframe ATR for context
        atr_short = true_range.rolling(window=max(3, atr_period // 2)).mean()
        atr_long = true_range.rolling(window=min(100, atr_period * 2)).mean()
        # ATR-based levels
        atr_stop_loss_long = data['close'] - (atr * stop_multiplier)
        atr_stop_loss_short = data['close'] + (atr * stop_multiplier)
        # Trailing stop levels
        atr_trailing_long = data['close'] - (atr * trailing_multiplier)
        atr_trailing_short = data['close'] + (atr * trailing_multiplier)
        # Volatility regime analysis
        volatility_regime = self._analyze_volatility_regime(atr, atr_percentage)
        # ATR trend analysis
        atr_trend = atr.pct_change(periods=5).rolling(window=3).mean()
        atr_expanding = atr_trend > 0.02  # ATR expanding
        atr_contracting = atr_trend < -0.02  # ATR contracting
        # Risk-reward calculations
        risk_reward_ratios = self._calculate_risk_reward_levels(data, atr)
        # Position size adjustments based on ATR
        position_size_factor = self._calculate_atr_position_factor(atr, atr_percentage)
        # Breakout probability based on ATR squeeze
        atr_squeeze = self._detect_atr_squeeze(atr, atr_short, atr_long)
        # Volatility-adjusted signals
        volatility_signals = self._generate_volatility_signals(data, atr, atr_percentage)
        # Support/Resistance levels based on ATR
        atr_support = data['low'].rolling(window=20).min() - atr
        atr_resistance = data['high'].rolling(window=20).max() + atr
        return {
            'true_range': true_range,
            'atr': atr,
            'atr_percentage': atr_percentage,
            'atr_short': atr_short,
            'atr_long': atr_long,
            'atr_stop_loss_long': atr_stop_loss_long,
            'atr_stop_loss_short': atr_stop_loss_short,
            'atr_trailing_long': atr_trailing_long,
            'atr_trailing_short': atr_trailing_short,
            'volatility_regime': volatility_regime,
            'atr_trend': atr_trend,
            'atr_expanding': atr_expanding,
            'atr_contracting': atr_contracting,
            'risk_reward_1_to_1': risk_reward_ratios['rr_1_to_1'],
            'risk_reward_1_to_2': risk_reward_ratios['rr_1_to_2'],
            'risk_reward_1_to_3': risk_reward_ratios['rr_1_to_3'],
            'position_size_factor': position_size_factor,
            'atr_squeeze': atr_squeeze,
            'volatility_signals': volatility_signals,
            'atr_support': atr_support,
            'atr_resistance': atr_resistance
        }
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range for ATR calculation.
        Args:
            data: OHLCV data
        Returns:
            True Range values
        """
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = data['close'].shift(1)
        range1 = data['high'] - data['low']
        range2 = abs(data['high'] - prev_close)
        range3 = abs(data['low'] - prev_close)
        true_range = pd.concat([range1, range2, range3], axis=1).max(axis=1)
        return true_range.fillna(range1)  # Use high-low for first value
    def _analyze_volatility_regime(self, atr: pd.Series, atr_percentage: pd.Series) -> Dict[str, pd.Series]:
        """Analyze current volatility regime.
        Args:
            atr: ATR values
            atr_percentage: ATR as percentage of price
        Returns:
            Dictionary of volatility regime indicators
        """
        # Historical percentiles for regime classification
        atr_percentile_75 = atr.rolling(window=100).quantile(0.75)
        atr_percentile_25 = atr.rolling(window=100).quantile(0.25)
        # Volatility regimes
        high_volatility = atr > atr_percentile_75
        low_volatility = atr < atr_percentile_25
        normal_volatility = ~(high_volatility | low_volatility)
        # Volatility trend
        atr_ma_short = atr.rolling(window=5).mean()
        atr_ma_long = atr.rolling(window=20).mean()
        volatility_rising = atr_ma_short > atr_ma_long
        volatility_falling = atr_ma_short < atr_ma_long
        return {
            'high_volatility': high_volatility,
            'low_volatility': low_volatility,
            'normal_volatility': normal_volatility,
            'volatility_rising': volatility_rising,
            'volatility_falling': volatility_falling,
            'atr_percentile_rank': atr.rolling(window=100).rank(pct=True)
        }
    def _calculate_risk_reward_levels(self, data: pd.DataFrame, atr: pd.Series) -> Dict[str, pd.Series]:
        """Calculate risk-reward ratio levels based on ATR.
        Args:
            data: OHLCV data
            atr: ATR values
        Returns:
            Dictionary of risk-reward levels
        """
        stop_multiplier = self.genes.parameters['stop_loss_multiplier']
        # Calculate target levels for different risk-reward ratios
        atr_risk = atr * stop_multiplier
        # For long positions
        rr_1_to_1_long = data['close'] + atr_risk  # 1:1 risk-reward
        rr_1_to_2_long = data['close'] + (atr_risk * 2)  # 1:2 risk-reward
        rr_1_to_3_long = data['close'] + (atr_risk * 3)  # 1:3 risk-reward
        # For short positions
        rr_1_to_1_short = data['close'] - atr_risk
        rr_1_to_2_short = data['close'] - (atr_risk * 2)
        rr_1_to_3_short = data['close'] - (atr_risk * 3)
        return {
            'rr_1_to_1': {'long': rr_1_to_1_long, 'short': rr_1_to_1_short},
            'rr_1_to_2': {'long': rr_1_to_2_long, 'short': rr_1_to_2_short},
            'rr_1_to_3': {'long': rr_1_to_3_long, 'short': rr_1_to_3_short}
        }
    def _calculate_atr_position_factor(self, atr: pd.Series, atr_percentage: pd.Series) -> pd.Series:
        """Calculate position sizing factor based on ATR.
        Args:
            atr: ATR values
            atr_percentage: ATR as percentage of price
        Returns:
            Position sizing adjustment factor
        """
        atr_factor = self.genes.parameters['position_size_atr_factor']
        # Inverse relationship: higher volatility = smaller position
        # Normalize ATR percentage to 0-1 scale
        atr_normalized = np.clip(atr_percentage / 5.0, 0.1, 2.0)  # 5% ATR as reference
        position_factor = atr_factor / atr_normalized
        return position_factor.clip(0.1, 2.0)  # Reasonable bounds
    def _detect_atr_squeeze(self, atr: pd.Series, atr_short: pd.Series, atr_long: pd.Series) -> pd.Series:
        """Detect ATR squeeze conditions (low volatility before breakouts).
        Args:
            atr: Main ATR values
            atr_short: Short-term ATR
            atr_long: Long-term ATR
        Returns:
            ATR squeeze signals
        """
        # ATR squeeze: when short-term ATR is compressed relative to long-term
        atr_ratio = atr_short / atr_long
        squeeze_threshold = 0.8
        # Historical ATR compression
        atr_percentile = atr.rolling(window=50).rank(pct=True)
        low_atr_percentile = atr_percentile < 0.2  # Bottom 20% of ATR readings
        # Squeeze conditions
        atr_squeeze = (atr_ratio < squeeze_threshold) & low_atr_percentile
        return atr_squeeze
    def _generate_volatility_signals(self, data: pd.DataFrame, atr: pd.Series, 
                                   atr_percentage: pd.Series) -> Dict[str, pd.Series]:
        """Generate signals based on volatility analysis.
        Args:
            data: OHLCV data
            atr: ATR values
            atr_percentage: ATR percentage
        Returns:
            Dictionary of volatility-based signals
        """
        # Volatility breakout signals
        atr_expansion = atr > atr.rolling(window=20).mean() * 1.5
        price_breakout = (
            (data['close'] > data['high'].rolling(window=20).max().shift(1)) |
            (data['close'] < data['low'].rolling(window=20).min().shift(1))
        )
        volatility_breakout = atr_expansion & price_breakout
        # Mean reversion in high volatility
        high_vol_reversion = (
            (atr_percentage > atr_percentage.rolling(window=50).quantile(0.8)) &
            (data['close'].pct_change().abs() > atr_percentage / 100 * 2)
        )
        # Low volatility accumulation
        low_vol_accumulation = (
            (atr_percentage < atr_percentage.rolling(window=50).quantile(0.3)) &
            (data['volume'] > data['volume'].rolling(window=20).mean() if 'volume' in data.columns else True)
        )
        return {
            'volatility_breakout': volatility_breakout,
            'high_vol_reversion': high_vol_reversion,
            'low_vol_accumulation': low_vol_accumulation
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ATR-based risk management signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        volatility_adjustment = self.genes.parameters['volatility_adjustment']
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        # Signal 1: Volatility breakout signals
        breakout_signals = self._generate_breakout_signals(indicators, data)
        # Signal 2: Mean reversion signals in high volatility
        reversion_signals = self._generate_reversion_signals(indicators, data)
        # Signal 3: ATR squeeze breakout signals
        squeeze_signals = self._generate_squeeze_signals(indicators, data)
        # Signal 4: Risk-reward optimization signals
        risk_reward_signals = self._generate_risk_reward_signals(indicators, data)
        # Combine signals with volatility adjustment
        signals = (
            breakout_signals * 0.3 +
            reversion_signals * 0.2 +
            squeeze_signals * 0.3 +
            risk_reward_signals * 0.2
        )
        # Apply volatility regime adjustments
        signals = self._apply_volatility_adjustments(signals, indicators, volatility_adjustment)
        # Ensure signals are within valid range
        signals = signals.clip(-1.0, 1.0)
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def _generate_breakout_signals(self, indicators: Dict[str, pd.Series], 
                                 data: pd.DataFrame) -> pd.Series:
        """Generate volatility breakout signals.
        Args:
            indicators: Dictionary of calculated indicators
            data: OHLCV data
        Returns:
            Breakout signals
        """
        signals = pd.Series(0.0, index=data.index)
        # Volatility expansion with price breakout
        breakout_conditions = indicators['volatility_signals']['volatility_breakout']
        # Direction determination
        price_direction = data['close'].pct_change()
        # Bullish breakout (price up + volatility expansion)
        bullish_breakout = breakout_conditions & (price_direction > 0)
        # Bearish breakout (price down + volatility expansion)
        bearish_breakout = breakout_conditions & (price_direction < 0)
        # Signal strength based on ATR expansion magnitude
        atr_expansion_factor = indicators['atr'] / indicators['atr'].rolling(window=20).mean()
        bullish_strength = np.where(
            bullish_breakout,
            np.minimum(1.0, atr_expansion_factor * 0.5),
            0.0
        )
        bearish_strength = np.where(
            bearish_breakout,
            -np.minimum(1.0, atr_expansion_factor * 0.5),
            0.0
        )
        signals = pd.Series(bullish_strength + bearish_strength, index=data.index)
        return signals
    def _generate_reversion_signals(self, indicators: Dict[str, pd.Series], 
                                  data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals in high volatility.
        Args:
            indicators: Dictionary of calculated indicators
            data: OHLCV data
        Returns:
            Mean reversion signals
        """
        signals = pd.Series(0.0, index=data.index)
        # High volatility mean reversion conditions
        reversion_conditions = indicators['volatility_signals']['high_vol_reversion']
        # Price extremes relative to ATR levels
        price_above_resistance = data['close'] > indicators['atr_resistance']
        price_below_support = data['close'] < indicators['atr_support']
        # Reversion signals
        bearish_reversion = reversion_conditions & price_above_resistance
        bullish_reversion = reversion_conditions & price_below_support
        # Signal strength based on distance from ATR levels
        resistance_distance = (data['close'] - indicators['atr_resistance']) / indicators['atr']
        support_distance = (indicators['atr_support'] - data['close']) / indicators['atr']
        bullish_reversion_strength = np.where(
            bullish_reversion,
            np.minimum(0.8, support_distance),
            0.0
        )
        bearish_reversion_strength = np.where(
            bearish_reversion,
            -np.minimum(0.8, resistance_distance),
            0.0
        )
        signals = pd.Series(bullish_reversion_strength + bearish_reversion_strength, 
                          index=data.index)
        return signals
    def _generate_squeeze_signals(self, indicators: Dict[str, pd.Series], 
                                data: pd.DataFrame) -> pd.Series:
        """Generate ATR squeeze breakout signals.
        Args:
            indicators: Dictionary of calculated indicators
            data: OHLCV data
        Returns:
            Squeeze breakout signals
        """
        signals = pd.Series(0.0, index=data.index)
        # ATR squeeze breakout
        squeeze_active = indicators['atr_squeeze']
        squeeze_ending = squeeze_active.shift(1) & ~squeeze_active  # Squeeze just ended
        # Price breakout direction
        price_change = data['close'].pct_change()
        # Breakout signals
        bullish_squeeze_breakout = squeeze_ending & (price_change > 0)
        bearish_squeeze_breakout = squeeze_ending & (price_change < 0)
        # Signal strength based on squeeze duration and breakout magnitude
        squeeze_duration = squeeze_active.rolling(window=50).sum()
        breakout_magnitude = abs(price_change) / indicators['atr_percentage'] * 100
        bullish_strength = np.where(
            bullish_squeeze_breakout,
            np.minimum(1.0, (squeeze_duration / 10 + breakout_magnitude) * 0.3),
            0.0
        )
        bearish_strength = np.where(
            bearish_squeeze_breakout,
            -np.minimum(1.0, (squeeze_duration / 10 + breakout_magnitude) * 0.3),
            0.0
        )
        signals = pd.Series(bullish_strength + bearish_strength, index=data.index)
        return signals
    def _generate_risk_reward_signals(self, indicators: Dict[str, pd.Series], 
                                    data: pd.DataFrame) -> pd.Series:
        """Generate signals based on risk-reward optimization.
        Args:
            indicators: Dictionary of calculated indicators
            data: OHLCV data
        Returns:
            Risk-reward optimized signals
        """
        signals = pd.Series(0.0, index=data.index)
        # Favorable risk-reward setups
        current_price = data['close']
        # Long setup: price near support with good upside potential
        long_risk = current_price - indicators['atr_stop_loss_long']
        long_reward_1_to_2 = indicators['risk_reward_1_to_2']['long'] - current_price
        long_rr_ratio = long_reward_1_to_2 / long_risk
        # Short setup: price near resistance with good downside potential  
        short_risk = indicators['atr_stop_loss_short'] - current_price
        short_reward_1_to_2 = current_price - indicators['risk_reward_1_to_2']['short']
        short_rr_ratio = short_reward_1_to_2 / short_risk
        # Good risk-reward opportunities
        good_long_rr = long_rr_ratio > 1.5
        good_short_rr = short_rr_ratio > 1.5
        # Signal strength based on risk-reward quality
        long_rr_strength = np.where(
            good_long_rr,
            np.minimum(0.7, long_rr_ratio / 3.0),
            0.0
        )
        short_rr_strength = np.where(
            good_short_rr,
            -np.minimum(0.7, short_rr_ratio / 3.0),
            0.0
        )
        signals = pd.Series(long_rr_strength + short_rr_strength, index=data.index)
        return signals
    def _apply_volatility_adjustments(self, signals: pd.Series, indicators: Dict[str, pd.Series],
                                    volatility_adjustment: float) -> pd.Series:
        """Apply volatility regime adjustments to signals.
        Args:
            signals: Base signals
            indicators: Dictionary of indicators
            volatility_adjustment: Adjustment weight
        Returns:
            Volatility-adjusted signals
        """
        adjusted_signals = signals.copy()
        # Get volatility regime
        vol_regime = indicators['volatility_regime']
        # Adjustments based on volatility regime
        for i in range(len(signals)):
            if abs(signals.iloc[i]) > 0.1:  # Significant signal
                current_signal = signals.iloc[i]
                # High volatility: reduce position sizes
                if vol_regime['high_volatility'].iloc[i]:
                    adjusted_signals.iloc[i] = current_signal * (1 - volatility_adjustment * 0.3)
                # Low volatility: can increase position sizes
                elif vol_regime['low_volatility'].iloc[i]:
                    adjusted_signals.iloc[i] = current_signal * (1 + volatility_adjustment * 0.2)
                # Volatility rising: be more cautious
                if vol_regime['volatility_rising'].iloc[i]:
                    adjusted_signals.iloc[i] *= (1 - volatility_adjustment * 0.1)
        return adjusted_signals.clip(-1.0, 1.0)
    def get_volatility_regime(self, data: pd.DataFrame) -> str:
        """Identify current volatility regime.
        Args:
            data: OHLCV market data
        Returns:
            String describing current volatility regime
        """
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return "unknown"
        vol_regime = indicators['volatility_regime']
        current_atr_percentile = vol_regime['atr_percentile_rank'].iloc[-1]
        if current_atr_percentile > 0.8:
            return "high_volatility"
        elif current_atr_percentile < 0.2:
            return "low_volatility"
        elif vol_regime['volatility_rising'].iloc[-1]:
            return "volatility_rising"
        elif vol_regime['volatility_falling'].iloc[-1]:
            return "volatility_falling"
        else:
            return "normal_volatility"
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate ATR-adjusted position size.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get ATR conditions
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return 0.0
        current_position_factor = indicators['position_size_factor'].iloc[-1]
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust for signal strength
        signal_adjustment = abs(signal)
        # ATR-based position adjustment (inverse relationship with volatility)
        atr_adjustment = current_position_factor
        # Calculate final position size
        position_size = base_size * signal_adjustment * atr_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def calculate_stop_loss_price(self, data: pd.DataFrame, entry_price: float, 
                                direction: str) -> float:
        """Calculate ATR-based stop loss price.
        Args:
            data: Current market data
            entry_price: Entry price
            direction: 'long' or 'short'
        Returns:
            Stop loss price
        """
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return entry_price
        current_atr = indicators['atr'].iloc[-1]
        stop_multiplier = self.genes.parameters['stop_loss_multiplier']
        if direction.lower() == 'long':
            return entry_price - (current_atr * stop_multiplier)
        else:  # short
            return entry_price + (current_atr * stop_multiplier)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        atr_period = self.genes.parameters.get('atr_period', 14)
        stop_multiplier = self.genes.parameters.get('stop_loss_multiplier', 2.0)
        position_factor = self.genes.parameters.get('position_size_atr_factor', 0.5)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"ATRStop({atr_period:.0f},{stop_multiplier:.1f})[{position_factor:.2f}]{fitness_str}"
