
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Ichimoku Cloud Genetic Seed - Seed #7
This seed implements the Ichimoku Kinko Hyo system with genetic parameter evolution
for optimal trend identification, cloud analysis, and multi-timeframe confirmation.
Key Features:
- Complete Ichimoku system with genetic parameter evolution
- Cloud analysis for trend direction and strength
- Kumo breakouts and Tenkan/Kijun crossovers
- Multi-timeframe cloud confirmation and momentum analysis
"""
@genetic_seed
class IchimokuCloudSeed(BaseSeed):
    """Ichimoku Cloud trading seed with genetic parameter evolution."""
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Ichimoku_Cloud"
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Ichimoku Kinko Hyo cloud-based strategy. Uses Tenkan-sen, Kijun-sen, "
                "Senkou Span A/B, and Chikou Span for trend analysis. Identifies cloud "
                "breakouts, line crossovers, and momentum confirmations with genetic optimization.")
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'tenkan_period',
            'kijun_period',
            'senkou_span_b_period',
            'cloud_strength_weight',
            'momentum_confirmation'
        ]
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'tenkan_period': (7.0, 12.0),           # Tighter for day-trades or default for trend clarity
            'kijun_period': (20.0, 34.0),           # Optimized range for crypto trend detection
            'senkou_span_b_period': (40.0, 80.0),   # Conservative upper bound for crypto volatility
            'cloud_strength_weight': (0.0, 1.0),    # Cloud analysis weight
            'momentum_confirmation': (0.0, 1.0)     # Momentum confirmation weight
        }
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize Ichimoku Cloud seed.
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.TREND_FOLLOWING
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'tenkan_period': 9.0,
                'kijun_period': 26.0,
                'senkou_span_b_period': 52.0,
                'cloud_strength_weight': 0.7,
                'momentum_confirmation': 0.6
            }
        super().__init__(genes, settings)
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud indicators and related metrics.
        Args:
            data: OHLCV market data
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        tenkan_period = int(self.genes.parameters['tenkan_period'])
        kijun_period = int(self.genes.parameters['kijun_period'])
        senkou_b_period = int(self.genes.parameters['senkou_span_b_period'])
        # Calculate Ichimoku lines
        ichimoku_lines = self._calculate_ichimoku_lines(data, tenkan_period, kijun_period, senkou_b_period)
        # Calculate cloud metrics
        cloud_metrics = self._calculate_cloud_metrics(ichimoku_lines)
        # Calculate crossover signals
        crossover_signals = self._calculate_crossover_signals(ichimoku_lines, data)
        # Calculate momentum confirmation signals
        momentum_signals = self._calculate_momentum_signals(ichimoku_lines, data)
        # Calculate cloud strength
        cloud_strength = self._calculate_cloud_strength(ichimoku_lines, data)
        # Price position analysis
        price_position = self._analyze_price_position(ichimoku_lines, data)
        # Multi-timeframe cloud analysis
        multi_tf_analysis = self._multi_timeframe_cloud_analysis(data, ichimoku_lines)
        # Combine all indicators
        all_indicators = {}
        all_indicators.update(ichimoku_lines)
        all_indicators.update(cloud_metrics)
        all_indicators.update(crossover_signals)
        all_indicators.update(momentum_signals)
        all_indicators.update(cloud_strength)
        all_indicators.update(price_position)
        all_indicators.update(multi_tf_analysis)
        return all_indicators
    def _calculate_ichimoku_lines(self, data: pd.DataFrame, tenkan_period: int, 
                                kijun_period: int, senkou_b_period: int) -> Dict[str, pd.Series]:
        """Calculate the five Ichimoku lines.
        Args:
            data: OHLCV data
            tenkan_period: Tenkan-sen period
            kijun_period: Kijun-sen period
            senkou_b_period: Senkou Span B period
        Returns:
            Dictionary of Ichimoku lines
        """
        # Tenkan-sen (Conversion Line) = (High + Low) / 2 over tenkan_period
        tenkan_high = data['high'].rolling(window=tenkan_period).max()
        tenkan_low = data['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        # Kijun-sen (Base Line) = (High + Low) / 2 over kijun_period
        kijun_high = data['high'].rolling(window=kijun_period).max()
        kijun_low = data['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        # Senkou Span A (Leading Span A) = (Tenkan-sen + Kijun-sen) / 2, shifted forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        # Senkou Span B (Leading Span B) = (High + Low) / 2 over senkou_b_period, shifted forward
        senkou_b_high = data['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = data['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        # Chikou Span (Lagging Span) = Close shifted back
        chikou_span = data['close'].shift(-kijun_period)
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    def _calculate_cloud_metrics(self, ichimoku_lines: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate cloud-related metrics.
        Args:
            ichimoku_lines: Dictionary of Ichimoku lines
        Returns:
            Dictionary of cloud metrics
        """
        senkou_a = ichimoku_lines['senkou_span_a']
        senkou_b = ichimoku_lines['senkou_span_b']
        # Cloud direction (bullish when Senkou A > Senkou B)
        cloud_bullish = senkou_a > senkou_b
        cloud_bearish = senkou_a < senkou_b
        # Cloud thickness (strength indicator)
        cloud_thickness = abs(senkou_a - senkou_b)
        cloud_thickness_pct = (cloud_thickness / ((senkou_a + senkou_b) / 2)) * 100
        # Cloud boundaries
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        cloud_middle = (cloud_top + cloud_bottom) / 2
        # Cloud trend (expanding or contracting)
        cloud_expanding = cloud_thickness > cloud_thickness.shift(5)
        cloud_contracting = cloud_thickness < cloud_thickness.shift(5)
        return {
            'cloud_bullish': cloud_bullish,
            'cloud_bearish': cloud_bearish,
            'cloud_thickness': cloud_thickness,
            'cloud_thickness_pct': cloud_thickness_pct,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'cloud_middle': cloud_middle,
            'cloud_expanding': cloud_expanding,
            'cloud_contracting': cloud_contracting
        }
    def _calculate_crossover_signals(self, ichimoku_lines: Dict[str, pd.Series], 
                                   data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate crossover signals.
        Args:
            ichimoku_lines: Dictionary of Ichimoku lines
            data: OHLCV data
        Returns:
            Dictionary of crossover signals
        """
        tenkan = ichimoku_lines['tenkan_sen']
        kijun = ichimoku_lines['kijun_sen']
        senkou_a = ichimoku_lines['senkou_span_a']
        senkou_b = ichimoku_lines['senkou_span_b']
        chikou = ichimoku_lines['chikou_span']
        # TK Cross (Tenkan/Kijun crossover)
        tk_bullish_cross = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tk_bearish_cross = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
        # Price vs Kumo (Cloud) breakouts
        price = data['close']
        price_above_cloud = price > pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        price_below_cloud = price < pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        price_in_cloud = ~(price_above_cloud | price_below_cloud)
        # Kumo breakouts
        price_above_prev = safe_fillna_false(price_above_cloud.shift(1))
        price_below_prev = safe_fillna_false(price_below_cloud.shift(1))
        kumo_bullish_breakout = price_above_cloud & (~price_above_prev)
        kumo_bearish_breakout = price_below_cloud & (~price_below_prev)
        # Chikou Span confirmations
        chikou_above_price = chikou > data['close'].shift(-26)  # Chikou above price 26 periods ago
        chikou_below_price = chikou < data['close'].shift(-26)  # Chikou below price 26 periods ago
        # Kijun-sen bounces
        kijun_bounce_bull = (data['low'] <= kijun) & (data['close'] > kijun)
        kijun_bounce_bear = (data['high'] >= kijun) & (data['close'] < kijun)
        return {
            'tk_bullish_cross': tk_bullish_cross,
            'tk_bearish_cross': tk_bearish_cross,
            'price_above_cloud': price_above_cloud,
            'price_below_cloud': price_below_cloud,
            'price_in_cloud': price_in_cloud,
            'kumo_bullish_breakout': kumo_bullish_breakout,
            'kumo_bearish_breakout': kumo_bearish_breakout,
            'chikou_above_price': chikou_above_price,
            'chikou_below_price': chikou_below_price,
            'kijun_bounce_bull': kijun_bounce_bull,
            'kijun_bounce_bear': kijun_bounce_bear
        }
    def _calculate_momentum_signals(self, ichimoku_lines: Dict[str, pd.Series], 
                                  data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum-based signals.
        Args:
            ichimoku_lines: Dictionary of Ichimoku lines
            data: OHLCV data
        Returns:
            Dictionary of momentum signals
        """
        tenkan = ichimoku_lines['tenkan_sen']
        kijun = ichimoku_lines['kijun_sen']
        # Line slopes (momentum indicators)
        tenkan_slope = tenkan.diff(periods=3)
        kijun_slope = kijun.diff(periods=3)
        # Price momentum relative to lines
        price_momentum_vs_tenkan = (data['close'] - tenkan) / tenkan * 100
        price_momentum_vs_kijun = (data['close'] - kijun) / kijun * 100
        # Momentum alignment
        momentum_aligned_bull = (tenkan_slope > 0) & (kijun_slope > 0) & (tenkan > kijun)
        momentum_aligned_bear = (tenkan_slope < 0) & (kijun_slope < 0) & (tenkan < kijun)
        # Momentum divergence
        momentum_diverging = (tenkan_slope > 0) & (kijun_slope < 0) | (tenkan_slope < 0) & (kijun_slope > 0)
        # Strong momentum conditions
        strong_momentum_bull = (
            (price_momentum_vs_tenkan > 2) & 
            (price_momentum_vs_kijun > 2) & 
            momentum_aligned_bull
        )
        strong_momentum_bear = (
            (price_momentum_vs_tenkan < -2) & 
            (price_momentum_vs_kijun < -2) & 
            momentum_aligned_bear
        )
        return {
            'tenkan_slope': tenkan_slope,
            'kijun_slope': kijun_slope,
            'price_momentum_vs_tenkan': price_momentum_vs_tenkan,
            'price_momentum_vs_kijun': price_momentum_vs_kijun,
            'momentum_aligned_bull': momentum_aligned_bull,
            'momentum_aligned_bear': momentum_aligned_bear,
            'momentum_diverging': momentum_diverging,
            'strong_momentum_bull': strong_momentum_bull,
            'strong_momentum_bear': strong_momentum_bear
        }
    def _calculate_cloud_strength(self, ichimoku_lines: Dict[str, pd.Series], 
                                data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate cloud strength indicators.
        Args:
            ichimoku_lines: Dictionary of Ichimoku lines
            data: OHLCV data
        Returns:
            Dictionary of cloud strength indicators
        """
        senkou_a = ichimoku_lines['senkou_span_a']
        senkou_b = ichimoku_lines['senkou_span_b']
        # Cloud strength based on thickness and consistency
        thickness_strength = abs(senkou_a - senkou_b) / data['close']
        # Cloud consistency (how long cloud has been same color)
        cloud_direction = (senkou_a > senkou_b).astype(int)
        cloud_direction_changes = cloud_direction.diff().abs()
        # Periods since last cloud direction change
        cloud_consistency = pd.Series(index=data.index, dtype=float)
        periods_since_change = 0
        for i in range(len(cloud_direction_changes)):
            if cloud_direction_changes.iloc[i] == 1:  # Direction changed
                periods_since_change = 0
            else:
                periods_since_change += 1
            cloud_consistency.iloc[i] = periods_since_change
        # Normalize cloud consistency
        cloud_consistency_normalized = np.minimum(cloud_consistency / 26, 1.0)  # Normalize to max 26 periods
        # Overall cloud strength score
        cloud_strength_score = (thickness_strength * 100 + cloud_consistency_normalized) / 2
        return {
            'thickness_strength': thickness_strength,
            'cloud_consistency': cloud_consistency,
            'cloud_consistency_normalized': cloud_consistency_normalized,
            'cloud_strength_score': cloud_strength_score
        }
    def _analyze_price_position(self, ichimoku_lines: Dict[str, pd.Series], 
                              data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze price position relative to Ichimoku components.
        Args:
            ichimoku_lines: Dictionary of Ichimoku lines
            data: OHLCV data
        Returns:
            Dictionary of price position indicators
        """
        price = data['close']
        tenkan = ichimoku_lines['tenkan_sen']
        kijun = ichimoku_lines['kijun_sen']
        # Price position relative to lines
        price_above_tenkan = price > tenkan
        price_above_kijun = price > kijun
        # Bullish alignment: Price > Tenkan > Kijun > Cloud
        bullish_alignment = (
            price_above_tenkan & 
            price_above_kijun & 
            (tenkan > kijun)
        )
        # Bearish alignment: Price < Tenkan < Kijun < Cloud
        bearish_alignment = (
            (~price_above_tenkan) & 
            (~price_above_kijun) & 
            (tenkan < kijun)
        )
        # Mixed signals
        mixed_signals = ~(bullish_alignment | bearish_alignment)
        # Distance from key levels
        distance_from_tenkan = (price - tenkan) / tenkan * 100
        distance_from_kijun = (price - kijun) / kijun * 100
        return {
            'price_above_tenkan': price_above_tenkan,
            'price_above_kijun': price_above_kijun,
            'bullish_alignment': bullish_alignment,
            'bearish_alignment': bearish_alignment,
            'mixed_signals': mixed_signals,
            'distance_from_tenkan': distance_from_tenkan,
            'distance_from_kijun': distance_from_kijun
        }
    def _multi_timeframe_cloud_analysis(self, data: pd.DataFrame, 
                                      ichimoku_lines: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Perform multi-timeframe cloud analysis.
        Args:
            data: OHLCV data
            ichimoku_lines: Dictionary of Ichimoku lines
        Returns:
            Dictionary of multi-timeframe indicators
        """
        # Calculate longer-term Ichimoku for confirmation
        long_term_tenkan = (data['high'].rolling(window=18).max() + data['low'].rolling(window=18).min()) / 2
        long_term_kijun = (data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2
        # Long-term cloud direction
        long_term_cloud_bullish = long_term_tenkan > long_term_kijun
        # Current vs long-term alignment
        current_cloud_bullish = ichimoku_lines['senkou_span_a'] > ichimoku_lines['senkou_span_b']
        cloud_alignment = current_cloud_bullish == long_term_cloud_bullish
        # Multi-timeframe confirmation strength
        mtf_confirmation_strength = cloud_alignment.rolling(window=10).mean()
        return {
            'long_term_tenkan': long_term_tenkan,
            'long_term_kijun': long_term_kijun,
            'long_term_cloud_bullish': long_term_cloud_bullish,
            'cloud_alignment': cloud_alignment,
            'mtf_confirmation_strength': mtf_confirmation_strength
        }
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Ichimoku Cloud trading signals.
        Args:
            data: OHLCV market data
        Returns:
            Series of trading signals: 1 (buy), 0 (hold), -1 (sell)
        """
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        # Get genetic parameters
        cloud_strength_weight = self.genes.parameters['cloud_strength_weight']
        momentum_confirmation = self.genes.parameters['momentum_confirmation']
        # Initialize signals
        signals = pd.Series(0.0, index=data.index)
        # Signal 1: Crossover signals
        crossover_signals = self._generate_crossover_signals(indicators)
        # Signal 2: Cloud breakout signals
        cloud_signals = self._generate_cloud_signals(indicators)
        # Signal 3: Momentum confirmation signals
        momentum_signals = self._generate_momentum_confirmation_signals(indicators)
        # Signal 4: Alignment signals
        alignment_signals = self._generate_alignment_signals(indicators)
        # Combine signals with weights
        signals = (
            crossover_signals * 0.25 +
            cloud_signals * cloud_strength_weight * 0.35 +
            momentum_signals * momentum_confirmation * 0.25 +
            alignment_signals * (1.0 - cloud_strength_weight) * 0.15
        )
        # Apply multi-timeframe confirmation
        signals = self._apply_multi_timeframe_confirmation(signals, indicators)
        # Ensure signals are within valid range
        signals = signals.clip(-1.0, 1.0)
        # Fill any NaN values
        signals = safe_fillna_zero(signals)
        return signals
    def _generate_crossover_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate TK crossover and other line-based signals.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Crossover signals
        """
        signals = pd.Series(0.0, index=indicators['tenkan_sen'].index)
        # TK Cross signals
        tk_bull_cross = indicators['tk_bullish_cross']
        tk_bear_cross = indicators['tk_bearish_cross']
        # Kijun bounces
        kijun_bounce_bull = indicators['kijun_bounce_bull']
        kijun_bounce_bear = indicators['kijun_bounce_bear']
        # Signal strength based on cloud context
        cloud_bullish = indicators['cloud_bullish']
        cloud_strength = indicators['cloud_strength_score']
        # Bullish crossover signals
        bullish_strength = np.where(
            tk_bull_cross & cloud_bullish,
            np.minimum(0.8, 0.5 + cloud_strength),
            np.where(
                kijun_bounce_bull & cloud_bullish,
                np.minimum(0.6, 0.3 + cloud_strength),
                0.0
            )
        )
        # Bearish crossover signals
        bearish_strength = np.where(
            tk_bear_cross & (~cloud_bullish),
            -np.minimum(0.8, 0.5 + cloud_strength),
            np.where(
                kijun_bounce_bear & (~cloud_bullish),
                -np.minimum(0.6, 0.3 + cloud_strength),
                0.0
            )
        )
        signals = pd.Series(bullish_strength + bearish_strength, 
                          index=indicators['tenkan_sen'].index)
        return signals
    def _generate_cloud_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate cloud breakout and cloud-based signals.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Cloud-based signals
        """
        signals = pd.Series(0.0, index=indicators['tenkan_sen'].index)
        # Cloud breakout signals
        kumo_bull_breakout = indicators['kumo_bullish_breakout']
        kumo_bear_breakout = indicators['kumo_bearish_breakout']
        # Cloud strength and thickness
        cloud_strength = indicators['cloud_strength_score']
        cloud_thickness = indicators['cloud_thickness_pct']
        # Strong cloud signals (when cloud is thick and consistent)
        strong_cloud_conditions = (cloud_strength > 0.5) & (cloud_thickness > 1.0)
        # Bullish cloud signals
        bullish_cloud_strength = np.where(
            kumo_bull_breakout & strong_cloud_conditions,
            np.minimum(1.0, 0.7 + cloud_strength * 0.3),
            np.where(
                kumo_bull_breakout,
                0.5,
                0.0
            )
        )
        # Bearish cloud signals
        bearish_cloud_strength = np.where(
            kumo_bear_breakout & strong_cloud_conditions,
            -np.minimum(1.0, 0.7 + cloud_strength * 0.3),
            np.where(
                kumo_bear_breakout,
                -0.5,
                0.0
            )
        )
        signals = pd.Series(bullish_cloud_strength + bearish_cloud_strength,
                          index=indicators['tenkan_sen'].index)
        return signals
    def _generate_momentum_confirmation_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate momentum confirmation signals.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Momentum confirmation signals
        """
        signals = pd.Series(0.0, index=indicators['tenkan_sen'].index)
        # Strong momentum signals
        strong_momentum_bull = indicators['strong_momentum_bull']
        strong_momentum_bear = indicators['strong_momentum_bear']
        # Momentum alignment
        momentum_aligned_bull = indicators['momentum_aligned_bull']
        momentum_aligned_bear = indicators['momentum_aligned_bear']
        # Momentum signal strength
        bullish_momentum_strength = np.where(
            strong_momentum_bull,
            0.9,
            np.where(momentum_aligned_bull, 0.6, 0.0)
        )
        bearish_momentum_strength = np.where(
            strong_momentum_bear,
            -0.9,
            np.where(momentum_aligned_bear, -0.6, 0.0)
        )
        signals = pd.Series(bullish_momentum_strength + bearish_momentum_strength,
                          index=indicators['tenkan_sen'].index)
        return signals
    def _generate_alignment_signals(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Generate perfect alignment signals.
        Args:
            indicators: Dictionary of calculated indicators
        Returns:
            Alignment signals
        """
        signals = pd.Series(0.0, index=indicators['tenkan_sen'].index)
        # Perfect alignment conditions
        bullish_alignment = indicators['bullish_alignment']
        bearish_alignment = indicators['bearish_alignment']
        # Price above cloud confirmation
        price_above_cloud = indicators['price_above_cloud']
        price_below_cloud = indicators['price_below_cloud']
        # Chikou confirmation
        chikou_above = indicators['chikou_above_price']
        chikou_below = indicators['chikou_below_price']
        # Perfect bullish setup
        perfect_bullish = bullish_alignment & price_above_cloud & chikou_above
        # Perfect bearish setup
        perfect_bearish = bearish_alignment & price_below_cloud & chikou_below
        # Alignment signal strength
        bullish_alignment_strength = np.where(perfect_bullish, 0.8, np.where(bullish_alignment, 0.5, 0.0))
        bearish_alignment_strength = np.where(perfect_bearish, -0.8, np.where(bearish_alignment, -0.5, 0.0))
        signals = pd.Series(bullish_alignment_strength + bearish_alignment_strength,
                          index=indicators['tenkan_sen'].index)
        return signals
    def _apply_multi_timeframe_confirmation(self, signals: pd.Series, 
                                          indicators: Dict[str, pd.Series]) -> pd.Series:
        """Apply multi-timeframe confirmation to signals.
        Args:
            signals: Base signals
            indicators: Dictionary of indicators
        Returns:
            Confirmed signals
        """
        confirmed_signals = signals.copy()
        # Multi-timeframe confirmation strength
        mtf_strength = indicators['mtf_confirmation_strength']
        # Adjust signals based on multi-timeframe alignment
        for i in range(len(signals)):
            if abs(signals.iloc[i]) > 0.1:  # Significant signal
                current_mtf_strength = mtf_strength.iloc[i] if not pd.isna(mtf_strength.iloc[i]) else 0.5
                # Enhance or reduce signal based on MTF confirmation
                if current_mtf_strength > 0.7:  # Strong MTF confirmation
                    confirmed_signals.iloc[i] *= 1.2
                elif current_mtf_strength < 0.3:  # Weak MTF confirmation
                    confirmed_signals.iloc[i] *= 0.6
                # Ensure bounds
                confirmed_signals.iloc[i] = np.clip(confirmed_signals.iloc[i], -1.0, 1.0)
        return confirmed_signals
    def get_ichimoku_regime(self, data: pd.DataFrame) -> str:
        """Identify current Ichimoku regime.
        Args:
            data: OHLCV market data
        Returns:
            String describing current Ichimoku regime
        """
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return "unknown"
        current_alignment = indicators['bullish_alignment'].iloc[-1]
        current_cloud = indicators['cloud_bullish'].iloc[-1]
        price_vs_cloud = indicators['price_above_cloud'].iloc[-1]
        if current_alignment and current_cloud and price_vs_cloud:
            return "strong_bullish"
        elif not current_alignment and not current_cloud and not price_vs_cloud:
            return "strong_bearish"
        elif current_cloud and price_vs_cloud:
            return "moderate_bullish"
        elif not current_cloud and not price_vs_cloud:
            return "moderate_bearish"
        else:
            return "mixed_signals"
    def calculate_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """Calculate position size based on Ichimoku conditions.
        Args:
            data: Current market data
            signal: Signal strength (-1 to 1)
        Returns:
            Position size as percentage of capital
        """
        if abs(signal) < self.genes.filter_threshold:
            return 0.0
        # Get Ichimoku conditions
        indicators = self.calculate_technical_indicators(data)
        if len(data) == 0:
            return 0.0
        cloud_strength = indicators['cloud_strength_score'].iloc[-1]
        mtf_confirmation = indicators['mtf_confirmation_strength'].iloc[-1]
        # Base position size from genes
        base_size = self.genes.position_size
        # Adjust based on cloud strength and multi-timeframe confirmation
        cloud_adjustment = max(0.5, min(1.5, cloud_strength))
        mtf_adjustment = max(0.7, min(1.3, mtf_confirmation))
        # Adjust for signal strength
        signal_adjustment = abs(signal)
        # Calculate final position size
        position_size = base_size * signal_adjustment * cloud_adjustment * mtf_adjustment
        # Apply maximum position size limit
        max_size = self.settings.trading.max_position_size
        return min(position_size, max_size)
    def __str__(self) -> str:
        """String representation with genetic parameters."""
        tenkan = self.genes.parameters.get('tenkan_period', 9)
        kijun = self.genes.parameters.get('kijun_period', 26)
        cloud_weight = self.genes.parameters.get('cloud_strength_weight', 0.7)
        fitness_str = f" (fitness={self.fitness.composite_fitness:.3f})" if self.fitness else ""
        return f"Ichimoku({tenkan:.0f},{kijun:.0f})[{cloud_weight:.2f}]{fitness_str}"
