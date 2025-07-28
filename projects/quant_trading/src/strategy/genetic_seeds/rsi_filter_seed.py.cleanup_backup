


from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional



from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

    
    
    
    
    
        
        
        
    
        
            
        
        
        
        
    
        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
        
            
        
        
        
        
        
        
        
    
        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
        
            
        
        
        
        
    
        
            
        
    
        
            
        
        
        
        
        
        
    
        
            
        
        
        
        
    
        
            
        
        
        
        
        
        
    
        
"""
RSI Filter Genetic Seed - Seed #3
This seed implements RSI-based overbought/oversold filtering and mean reversion
signals. The genetic algorithm evolves optimal RSI periods, thresholds, and
confirmation parameters for both trend-following and contrarian strategies.
Key Features:
- Adaptive RSI periods and thresholds through genetic evolution
- Dual-mode operation: trend filter or mean reversion signals
- Divergence detection for enhanced signal quality
- Multi-timeframe RSI confirmation
"""
@genetic_seed
class RSIFilterSeed(BaseSeed):
    """RSI-based filter and signal generator with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "RSI_Filter"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("RSI-based overbought/oversold filter and mean reversion strategy. "
                "Can operate as a trend filter (avoid trades in extreme conditions) "
                "or generate contrarian signals (buy oversold, sell overbought). "
                "Includes divergence detection and multi-timeframe confirmation.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'rsi_period',
            'oversold_threshold',
            'overbought_threshold',
            'operation_mode',
            'divergence_weight'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max)."""
        return {
            'rsi_period': (7.0, 35.0),          # RSI calculation period
            'oversold_threshold': (15.0, 35.0),  # Oversold level (typically 20-30)
            'overbought_threshold': (65.0, 85.0), # Overbought level (typically 70-80)
            'operation_mode': (0.0, 1.0),       # 0=filter mode, 1=signal mode
            'divergence_weight': (0.0, 1.0)     # Weight for divergence signals
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize RSI filter seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.MEAN_REVERSION
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'rsi_period': 14.0,
                'oversold_threshold': 25.0,
                'overbought_threshold': 75.0,
                'operation_mode': 0.7,  # Mostly signal mode
                'divergence_weight': 0.3
            }
        super().__init__(genes, settings)
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pandas (primary method from PRP).
        Args:
            prices: Price series (typically close prices)
            period: RSI calculation period
        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        # Calculate relative strength
        rs = gain / loss
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)  # Default to neutral for NaN values
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate RSI and related indicators.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        rsi_period = int(self.genes.parameters['rsi_period'])
        # Primary RSI calculation
        rsi = self.calculate_rsi(data['close'], rsi_period)
        # Multi-timeframe RSI for confirmation
        rsi_short = self.calculate_rsi(data['close'], max(5, rsi_period // 2))
        rsi_long = self.calculate_rsi(data['close'], min(50, rsi_period * 2))
        # RSI momentum (rate of change)
        rsi_momentum = rsi.diff()
        rsi_velocity = rsi.diff().rolling(window=3).mean()
        # RSI extremes and zones
        oversold_level = self.genes.parameters['oversold_threshold']
        overbought_level = self.genes.parameters['overbought_threshold']
        oversold_zone = rsi <= oversold_level
        overbought_zone = rsi >= overbought_level
        neutral_zone = (rsi > oversold_level) & (rsi < overbought_level)
        # RSI divergence calculation
        price_highs = data['close'].rolling(window=20).max()
        price_lows = data['close'].rolling(window=20).min()
        rsi_highs = rsi.rolling(window=20).max()
        rsi_lows = rsi.rolling(window=20).min()
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_divergence = (
            (data['close'] <= price_lows.shift(1)) & 
            (rsi > rsi_lows.shift(1)) &
            oversold_zone
        )
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_divergence = (
            (data['close'] >= price_highs.shift(1)) & 
            (rsi < rsi_highs.shift(1)) &
            overbought_zone
        )
        # RSI trend analysis
        rsi_trend = rsi.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        # Volume confirmation (if available)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_ma
        else:
            volume_ratio = pd.Series(1.0, index=data.index)
        # Price momentum for confirmation
        price_momentum = data['close'].pct_change(periods=5)
        return {
            'rsi': rsi,
            'rsi_short': rsi_short,
            'rsi_long': rsi_long,
            'rsi_momentum': rsi_momentum,
            'rsi_velocity': rsi_velocity,
            'oversold_zone': oversold_zone,
            'overbought_zone': overbought_zone,
            'neutral_zone': neutral_zone,
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'rsi_trend': rsi_trend,
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI-based trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        operation_mode = self.genes.parameters['operation_mode']
        divergence_weight = self.genes.parameters['divergence_weight']
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        if operation_mode > 0.5:  # Signal generation mode
            signals = self._generate_mean_reversion_signals(indicators)
        else:  # Filter mode
            signals = self._generate_trend_filter_signals(data, indicators)
        # Apply divergence signals if enabled
        if divergence_weight > 0.1:
            divergence_signals = self._generate_divergence_signals(indicators)
            signals = signals * (1 - divergence_weight) + divergence_signals * divergence_weight
        # Ensure signals are within valid range
        signals = signals.clip(-1.0, 1.0)
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def _generate_mean_reversion_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate mean reversion signals based on RSI extremes.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Mean reversion signals
        """
        signals = pd.Series(0.0, index=indicators['rsi'].index)
        # Basic mean reversion: buy oversold, sell overbought
        oversold_signals = indicators['oversold_zone'] & (indicators['rsi_momentum'] > 0)
        overbought_signals = indicators['overbought_zone'] & (indicators['rsi_momentum'] < 0)
        # Multi-timeframe confirmation
        mtf_bullish = (
            (indicators['rsi_short'] <= indicators['rsi'] * 1.1) &  # Short-term aligned
            (indicators['rsi_long'] <= indicators['rsi'] * 1.2)       # Long-term not too extreme
        )
        mtf_bearish = (
            (indicators['rsi_short'] >= indicators['rsi'] * 0.9) &  # Short-term aligned
            (indicators['rsi_long'] >= indicators['rsi'] * 0.8)       # Long-term not too extreme
        )
        # Volume confirmation
        volume_confirmed = indicators['volume_ratio'] >= 1.0
        # Price momentum confirmation (looking for momentum exhaustion)
        momentum_exhaustion_buy = (
            (indicators['price_momentum'] <= 0) &  # Negative price momentum
            (indicators['rsi_velocity'] > 0)          # But RSI starting to turn up
        )
        momentum_exhaustion_sell = (
            (indicators['price_momentum'] >= 0) &  # Positive price momentum
            (indicators['rsi_velocity'] < 0)          # But RSI starting to turn down
        )
        # Generate buy signals (oversold mean reversion)
        buy_conditions = (
            oversold_signals &
            mtf_bullish &
            volume_confirmed &
            momentum_exhaustion_buy
        )
        # Generate sell signals (overbought mean reversion)
        sell_conditions = (
            overbought_signals &
            mtf_bearish &
            volume_confirmed &
            momentum_exhaustion_sell
        )
        # Calculate signal strength based on RSI extreme level
        oversold_strength = np.where(
            buy_conditions,
            np.minimum(1.0, (self.genes.parameters['oversold_threshold'] - indicators['rsi']) / 20.0),
            0.0
        )
        overbought_strength = np.where(
            sell_conditions,
            -np.minimum(1.0, (indicators['rsi'] - self.genes.parameters['overbought_threshold']) / 20.0),
            0.0
        )
        signals = pd.Series(oversold_strength + overbought_strength, index=indicators['rsi'].index)
        return signals
    def _generate_trend_filter_signals(self, data: pd.DataFrame, 
                                     indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate trend-following signals filtered by RSI conditions.
        Args:
            data: OHLCV market data
            indicators: Dictionary of calculated indicators
        Returns:
            Trend-filtered signals
        """
        signals = pd.Series(0.0, index=data.index)
        # Basic trend detection
        short_ma = data['close'].rolling(window=10).mean()
        long_ma = data['close'].rolling(window=30).mean()
        trend_up = short_ma > long_ma
        trend_down = short_ma < long_ma
        # RSI filters for trend following
        # In uptrend: buy when RSI pulls back but stays above oversold
        # In downtrend: sell when RSI rallies but stays below overbought
        rsi_pullback_buy = (
            (indicators['rsi'] > self.genes.parameters['oversold_threshold'] + 5) &
            (indicators['rsi'] < 50) &
            (indicators['rsi_momentum'] > 0)  # RSI starting to turn up
        )
        rsi_rally_sell = (
            (indicators['rsi'] < self.genes.parameters['overbought_threshold'] - 5) &
            (indicators['rsi'] > 50) &
            (indicators['rsi_momentum'] < 0)  # RSI starting to turn down
        )
        # Volume and momentum confirmation
        volume_confirmed = indicators['volume_ratio'] >= 1.0
        price_momentum_up = indicators['price_momentum'] > 0
        price_momentum_down = indicators['price_momentum'] < 0
        # Generate trend-following signals with RSI filter
        buy_conditions = (
            trend_up &
            rsi_pullback_buy &
            volume_confirmed &
            price_momentum_up
        )
        sell_conditions = (
            trend_down &
            rsi_rally_sell &
            volume_confirmed &
            price_momentum_down
        )
        # Signal strength based on trend strength and RSI position
        trend_strength = abs(short_ma - long_ma) / long_ma
        buy_strength = np.where(
            buy_conditions,
            np.minimum(1.0, trend_strength * 10 * (1 - indicators['rsi'] / 100)),
            0.0
        )
        sell_strength = np.where(
            sell_conditions,
            -np.minimum(1.0, trend_strength * 10 * (indicators['rsi'] / 100)),
            0.0
        )
        signals = pd.Series(buy_strength + sell_strength, index=data.index)
        return signals
    def _generate_divergence_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate signals based on RSI divergences.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Divergence-based signals
        """
        signals = pd.Series(0.0, index=indicators['rsi'].index)
        # Bullish divergence signals (stronger when in oversold zone)
        bullish_div_strength = np.where(
            indicators['bullish_divergence'],
            np.minimum(1.0, (self.genes.parameters['oversold_threshold'] - indicators['rsi']) / 30.0 + 0.3),
            0.0
        )
        # Bearish divergence signals (stronger when in overbought zone)
        bearish_div_strength = np.where(
            indicators['bearish_divergence'],
            -np.minimum(1.0, (indicators['rsi'] - self.genes.parameters['overbought_threshold']) / 30.0 + 0.3),
            0.0
        )
        signals = pd.Series(bullish_div_strength + bearish_div_strength, index=indicators['rsi'].index)
        return signals
    def get_rsi_regime(self, data: pd.DataFrame) -> str:
        """Identify current RSI regime.
        Args:
            data: OHLCV market data
        Returns:
            String describing current RSI regime
        """
        indicators = self.calculate_technical_indicators(data)
        current_rsi = indicators['rsi'].iloc[-1] if len(data) > 0 else 50
        if current_rsi <= self.genes.parameters['oversold_threshold']:
            return "oversold"
        elif current_rsi >= self.genes.parameters['overbought_threshold']:
            return "overbought"
        elif current_rsi < 40:
            return "bearish"
        elif current_rsi > 60:
            return "bullish"
        else:
            return "neutral"
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on RSI conditions and signal strength.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get RSI conditions
        indicators = self.calculate_technical_indicators(data)
        current_rsi = indicators['rsi'].iloc[-1] if len(data) > 0 else 50
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust based on RSI extreme level
        if signal > 0:  # Buy signal
            # Larger size when more oversold
            rsi_adjustment = max(0.5, (self.genes.parameters['oversold_threshold'] - current_rsi + 20) / 40)
        else:  # Sell signal
            # Larger size when more overbought
            rsi_adjustment = max(0.5, (current_rsi - self.genes.parameters['overbought_threshold'] + 20) / 40)
        # Adjust for signal strength
        signal_adjustment = abs(signal)
        # Calculate final position size
        position_size = base_size * signal_adjustment * rsi_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def should_exit_position(self, data: pd.DataFrame, current_position: float, 
                           entry_price: float, current_price: float) -> bool:
        """Determine if position should be exited based on RSI conditions.
        Args:
            data: Current market data
            current_position: Current position size
            entry_price: Entry price
            current_price: Current price
        Returns:
            True if position should be exited
        """
        # Standard stop loss/take profit
        if super().should_exit_position(data, current_position, entry_price, current_price):
            return True
        # RSI-based exits
        indicators = self.calculate_technical_indicators(data)
        current_rsi = indicators['rsi'].iloc[-1] if len(data) > 0 else 50
        if current_position > 0:  # Long position
            # Exit if RSI becomes very overbought or shows bearish divergence
            if (current_rsi >= self.genes.parameters['overbought_threshold'] + 5 or
                indicators['bearish_divergence'].iloc[-1] if len(data) > 0 else False):
                return True
        elif current_position < 0:  # Short position
            # Exit if RSI becomes very oversold or shows bullish divergence
            if (current_rsi <= self.genes.parameters['oversold_threshold'] - 5 or
                indicators['bullish_divergence'].iloc[-1] if len(data) > 0 else False):
                return True
        return False
    def get_signal_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence level for current RSI signals.
        Args:
            data: OHLCV market data
        Returns:
            Confidence level (0-1)
        """
        indicators = self.calculate_technical_indicators(data)
        # Confidence factors:
        # 1. RSI extreme level (more extreme = higher confidence)
        current_rsi = indicators['rsi'].iloc[-1] if len(data) > 0 else 50
        extreme_factor = max(
            (self.genes.parameters['oversold_threshold'] - current_rsi) / 30,
            (current_rsi - self.genes.parameters['overbought_threshold']) / 30
        )
        extreme_confidence = max(0, min(1, extreme_factor))
        # 2. Multi-timeframe alignment
        rsi_alignment = 1.0 - abs(indicators['rsi'].iloc[-1] - indicators['rsi_short'].iloc[-1]) / 50
        # 3. Divergence presence
        divergence_present = (
            indicators['bullish_divergence'].iloc[-1] or 
            indicators['bearish_divergence'].iloc[-1]
        ) if len(data) > 0 else False
        divergence_confidence = 0.3 if divergence_present else 0.0
        # 4. Volume confirmation
        volume_confidence = min(1.0, indicators['volume_ratio'].iloc[-1] / 2.0) if len(data) > 0 else 0.5
        # Combine confidence factors
        total_confidence = (
            extreme_confidence * 0.4 +
            rsi_alignment * 0.3 +
            divergence_confidence * 0.2 +
            volume_confidence * 0.1
        )
        return max(0.0, min(1.0, total_confidence))
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        rsi_period = self.genes.parameters.get('rsi_period', 14)
        oversold = self.genes.parameters.get('oversold_threshold', 25)
        overbought = self.genes.parameters.get('overbought_threshold', 75)
        mode = "signal" if self.genes.parameters.get('operation_mode', 0.5) > 0.5 else "filter"
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"RSI({rsi_period:.0f})[{oversold:.0f}/{overbought:.0f},{mode}]{fitness_str}"