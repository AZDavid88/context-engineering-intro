from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .base_seed import BaseSeed, SeedType, SeedGenes
from .seed_registry import genetic_seed
from src.config.settings import Settings, Optional

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Nadaraya-Watson Genetic Seed - Seed #14
This seed implements Nadaraya-Watson kernel regression for nonparametric trend
estimation with genetic algorithm parameter evolution. Provides smooth trend
estimation without the lag of traditional moving averages.

Key Features:
- Custom Nadaraya-Watson kernel regression implementation
- Genetic evolution of bandwidth and kernel type
- Gaussian and Epanechnikov kernel support
- Trend consistency measurement and filtering
- Adaptive bandwidth based on market volatility
- Regime-aware trend following

Mathematical Foundation:
- Nadaraya-Watson estimator: m̂(x) = Σ(K_h(x-X_i) * Y_i) / Σ(K_h(x-X_i))
- Where K_h is the kernel function with bandwidth h
- Provides smooth, nonparametric trend estimation
"""

@genetic_seed
class NadarayaWatsonSeed(BaseSeed):
    """Nadaraya-Watson kernel regression seed for nonparametric trend estimation."""
    
    @property
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        return "Nadaraya_Watson"
    
    @property
    def seed_description(self) -> str:
        """Return detailed seed description."""
        return ("Nadaraya-Watson kernel regression for nonparametric trend estimation. "
                "Provides smooth trend signals without the lag of traditional moving averages. "
                "Uses genetic algorithms to evolve optimal bandwidth, kernel type, and trend "
                "consistency thresholds for different market conditions.")
    
    @property
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        return [
            'bandwidth',
            'kernel_type',
            'trend_threshold',
            'smoothing_factor',
            'volatility_adaptation'
        ]
    
    @property
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max) - CRYPTO-OPTIMIZED."""
        return {
            'bandwidth': (5.0, 40.0),           # Narrower BW for reactivity; crypto-optimized range
            'kernel_type': (0.0, 1.0),         # 0=Gaussian, 1=Epanechnikov (epi kernel for sharper cutoff)
            'trend_threshold': (0.002, 0.08),   # Minimum trend strength (0.2%-8%)
            'smoothing_factor': (0.1, 0.9),    # Trend smoothing factor
            'volatility_adaptation': (0.0, 1.0) # Adaptive bandwidth weight
        }
    
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize Nadaraya-Watson seed.
        
        Args:
            genes: Genetic parameters
            settings: Configuration settings
        """
        # Set seed type
        genes.seed_type = SeedType.TREND_FOLLOWING
        
        # Initialize default parameters if not provided
        if not genes.parameters:
            genes.parameters = {
                'bandwidth': 20.0,
                'kernel_type': 0.0,  # Gaussian
                'trend_threshold': 0.01,
                'smoothing_factor': 0.3,
                'volatility_adaptation': 0.5
            }
        
        super().__init__(genes, settings)
    
    def _gaussian_kernel(self, x: np.ndarray) -> np.ndarray:
        """Gaussian kernel function: K(u) = (1/√2π) * exp(-0.5 * u²)"""
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    
    def _epanechnikov_kernel(self, x: np.ndarray) -> np.ndarray:
        """Epanechnikov kernel function: K(u) = 0.75 * (1 - u²) for |u| ≤ 1, 0 otherwise"""
        abs_x = np.abs(x)
        return np.where(abs_x <= 1.0, 0.75 * (1 - x**2), 0.0)
    
    def _compute_nadaraya_watson_estimator(self, prices: pd.Series, bandwidth: float, 
                                         kernel_type: float) -> pd.Series:
        """Compute Nadaraya-Watson kernel regression estimator.
        
        Args:
            prices: Price series
            bandwidth: Kernel bandwidth parameter
            kernel_type: Kernel type (0=Gaussian, 1=Epanechnikov)
            
        Returns:
            Nadaraya-Watson trend estimate
        """
        n = len(prices)
        nw_estimate = np.zeros(n)
        
        # Convert to numpy for performance
        price_values = prices.values
        
        # Choose kernel function
        if kernel_type < 0.5:
            kernel_func = self._gaussian_kernel
        else:
            kernel_func = self._epanechnikov_kernel
        
        # Compute NW estimator for each point
        for i in range(n):
            if i < int(bandwidth):
                # Not enough data, use simple average
                nw_estimate[i] = np.mean(price_values[:i+1]) if i > 0 else price_values[i]
                continue
            
            # Create distance array
            lookback_window = min(int(bandwidth * 2), i + 1)
            start_idx = max(0, i - lookback_window + 1)
            
            # Time distances (normalized by bandwidth)
            time_distances = np.arange(start_idx, i + 1) - i
            u = time_distances / bandwidth
            
            # Kernel weights
            weights = kernel_func(u)
            
            # Weighted average (Nadaraya-Watson estimator)
            if np.sum(weights) > 0:
                nw_estimate[i] = np.sum(weights * price_values[start_idx:i+1]) / np.sum(weights)
            else:
                nw_estimate[i] = price_values[i]
        
        return pd.Series(nw_estimate, index=prices.index)
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Nadaraya-Watson trend indicators.
        
        Args:
            data: OHLCV market data
            
        Returns:
            Dictionary of indicator name -> indicator values
        """
        # Get genetic parameters
        bandwidth = self.genes.parameters['bandwidth']
        kernel_type = self.genes.parameters['kernel_type']
        smoothing_factor = self.genes.parameters['smoothing_factor']
        vol_adaptation = self.genes.parameters['volatility_adaptation']
        
        close = data['close']
        
        # Adaptive bandwidth based on volatility
        if vol_adaptation > 0.1:
            volatility = close.pct_change().rolling(window=20).std()
            vol_normalized = volatility / volatility.rolling(window=100).mean()
            # Increase bandwidth in high volatility periods
            adaptive_bandwidth = bandwidth * (1 + vol_adaptation * (vol_normalized - 1))
            adaptive_bandwidth = adaptive_bandwidth.clip(bandwidth * 0.5, bandwidth * 2.0)
        else:
            adaptive_bandwidth = pd.Series(bandwidth, index=close.index)
        
        # Primary Nadaraya-Watson trend estimation
        nw_trend = self._compute_nadaraya_watson_estimator(close, bandwidth, kernel_type)
        
        # Multi-timeframe NW analysis
        if len(data) > bandwidth * 2:
            # Shorter-term trend (more responsive)
            short_bandwidth = max(5, bandwidth * 0.5)
            nw_trend_short = self._compute_nadaraya_watson_estimator(close, short_bandwidth, kernel_type)
            
            # Longer-term trend (smoother)
            long_bandwidth = min(bandwidth * 1.5, len(data) // 3)
            nw_trend_long = self._compute_nadaraya_watson_estimator(close, long_bandwidth, kernel_type)
        else:
            nw_trend_short = nw_trend
            nw_trend_long = nw_trend
        
        # Trend direction and strength
        trend_direction = nw_trend.diff()
        trend_strength = abs(trend_direction) / close
        
        # Trend consistency (smoothness of the trend)
        trend_consistency = 1.0 - (trend_direction.diff().abs().rolling(window=10).mean() / 
                                  trend_direction.abs().rolling(window=10).mean()).fillna(0)
        
        # Price position relative to NW trend
        price_vs_trend = (close - nw_trend) / nw_trend
        
        # Trend regime classification
        uptrend = trend_direction > 0
        downtrend = trend_direction < 0
        sideways = abs(trend_direction) < (trend_direction.abs().rolling(window=20).quantile(0.3))
        
        # Multi-timeframe trend agreement
        mtf_bullish = (nw_trend_short > nw_trend) & (nw_trend > nw_trend_long)
        mtf_bearish = (nw_trend_short < nw_trend) & (nw_trend < nw_trend_long)
        
        # Trend momentum (rate of trend change)
        trend_momentum = trend_direction.diff()
        trend_acceleration = trend_momentum > 0
        trend_deceleration = trend_momentum < 0
        
        # Dynamic support/resistance based on NW trend
        nw_support = nw_trend - (nw_trend * 0.02)  # 2% below trend
        nw_resistance = nw_trend + (nw_trend * 0.02)  # 2% above trend
        
        # Price distance from trend (normalized)
        price_distance = abs(close - nw_trend) / nw_trend
        
        return {
            'nw_trend': nw_trend,
            'nw_trend_short': nw_trend_short,
            'nw_trend_long': nw_trend_long,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'trend_consistency': trend_consistency,
            'price_vs_trend': price_vs_trend,
            'uptrend': uptrend,
            'downtrend': downtrend,
            'sideways': sideways,
            'mtf_bullish': mtf_bullish,
            'mtf_bearish': mtf_bearish,
            'trend_momentum': trend_momentum,
            'trend_acceleration': trend_acceleration,
            'trend_deceleration': trend_deceleration,
            'nw_support': nw_support,
            'nw_resistance': nw_resistance,
            'price_distance': price_distance,
            'adaptive_bandwidth': adaptive_bandwidth
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Nadaraya-Watson trend following signals.
        
        Args:
            data: OHLCV market data
            
        Returns:
            Series of trading signals (-1.0 to 1.0)
        """
        # Calculate technical indicators internally (following existing pattern)
        indicators = self.calculate_technical_indicators(data)
        
        # Get genetic parameters
        trend_threshold = self.genes.parameters['trend_threshold']
        smoothing_factor = self.genes.parameters['smoothing_factor']
        
        # Extract indicators
        nw_trend = indicators['nw_trend']
        trend_strength = indicators['trend_strength']
        trend_consistency = indicators['trend_consistency']
        price_vs_trend = indicators['price_vs_trend']
        uptrend = indicators['uptrend']
        downtrend = indicators['downtrend']
        mtf_bullish = indicators['mtf_bullish']
        mtf_bearish = indicators['mtf_bearish']
        trend_acceleration = indicators['trend_acceleration']
        nw_support = indicators['nw_support']
        nw_resistance = indicators['nw_resistance']
        
        close = data['close']
        
        # Ultra-simplified signal logic (matching RSI pattern)
        
        # Basic trend following signals (following research patterns)
        # Entry long when price above trend OR strong upward trend
        entry_long = (
            (close > nw_trend) |                    # Price above trend
            (trend_strength > trend_threshold * 2)  # Very strong trend
        )
        
        # Entry short when price below trend OR strong downward trend
        entry_short = (
            (close < nw_trend) |                    # Price below trend
            (trend_strength > trend_threshold * 2)  # Very strong trend (any direction)
        )
        
        # Simple exits (opposite conditions)
        exit_long = (close < nw_trend * 0.99)      # Exit when price clearly below trend
        exit_short = (close > nw_trend * 1.01)     # Exit when price clearly above trend
        
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
    
    def calculate_trend_strength_score(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> pd.Series:
        """Calculate trend strength score for position sizing.
        
        Args:
            data: OHLCV market data
            indicators: Technical indicators
            
        Returns:
            Trend strength score (0-1)
        """
        trend_strength = indicators['trend_strength']
        trend_consistency = indicators['trend_consistency']
        
        # Normalize trend strength (0-1 scale)
        strength_normalized = trend_strength / (trend_strength.rolling(window=50).quantile(0.9) + 1e-6)
        strength_normalized = strength_normalized.clip(0, 1)
        
        # Combine strength and consistency
        trend_score = (strength_normalized * 0.7) + (trend_consistency * 0.3)
        
        return safe_fillna(trend_score, 0.5)