"""
Genetic Engine Evaluation - Fitness calculation and performance analysis.
Handles strategy evaluation, performance metrics, and market data generation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna
from src.strategy.genetic_seeds import BaseSeed
from src.strategy.genetic_seeds.base_seed import SeedFitness

# Configure logging
logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Comprehensive fitness evaluation system for trading strategies."""
    
    def __init__(self, fitness_weights: Optional[Dict[str, float]] = None):
        """Initialize fitness evaluator.
        
        Args:
            fitness_weights: Weights for multi-objective optimization
        """
        self.fitness_weights = fitness_weights or {
            "sharpe_ratio": 1.0,
            "consistency": 0.3,
            "max_drawdown": -1.0,  # Negative to minimize
            "win_rate": 0.5
        }
        
        # Cache for market data
        self._market_data_cache = {}
        self._synthetic_data_cache = {}
        
        logger.info("Fitness evaluator initialized")
    
    def evaluate_individual(self, individual: BaseSeed, 
                          market_data: Optional[pd.DataFrame] = None) -> Tuple[float, float, float, float]:
        """Evaluate fitness of an individual trading strategy.
        
        Args:
            individual: Trading strategy individual to evaluate
            market_data: Optional market data for evaluation
            
        Returns:
            Tuple of (sharpe_ratio, consistency, max_drawdown, win_rate)
        """
        try:
            # Use provided market data or generate synthetic data
            if market_data is None:
                market_data = self.generate_synthetic_market_data()
            
            # Generate trading signals from the individual
            signals = self._generate_trading_signals(individual, market_data)
            
            if signals is None or signals.empty:
                logger.warning(f"No signals generated for individual {type(individual).__name__}")
                return (0.0, 0.0, -1.0, 0.0)  # Poor fitness for failed strategies
            
            # Calculate strategy returns
            returns = self.calculate_strategy_returns(signals, market_data)
            
            if returns is None or returns.empty or returns.sum() == 0:
                logger.debug("No returns generated, assigning poor fitness")
                return (0.0, 0.0, -1.0, 0.0)
            
            # Calculate fitness metrics
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            consistency = self.calculate_consistency(returns)
            max_drawdown = self.calculate_max_drawdown(returns)
            win_rate = self.calculate_win_rate(returns)
            
            # Update individual's fitness
            if hasattr(individual, 'fitness'):
                individual.fitness.sharpe_ratio = sharpe_ratio
                individual.fitness.consistency = consistency
                individual.fitness.max_drawdown = max_drawdown
                individual.fitness.win_rate = win_rate
                individual.fitness.total_return = returns.sum()
                individual.fitness.evaluation_count += 1
            
            logger.debug(f"Evaluated {type(individual).__name__}: "
                        f"Sharpe={sharpe_ratio:.3f}, Consistency={consistency:.3f}, "
                        f"Drawdown={max_drawdown:.3f}, WinRate={win_rate:.3f}")
            
            return (sharpe_ratio, consistency, max_drawdown, win_rate)
            
        except Exception as e:
            logger.error(f"Error evaluating individual {type(individual).__name__}: {e}")
            return (0.0, 0.0, -1.0, 0.0)  # Poor fitness for failed evaluations
    
    def _generate_trading_signals(self, individual: BaseSeed, 
                                 market_data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate trading signals from individual strategy."""
        try:
            if hasattr(individual, 'generate_signals'):
                signals = individual.generate_signals(market_data)
                
                # Ensure signals are properly formatted
                if isinstance(signals, pd.Series):
                    # Clean and validate signals
                    signals = signals.fillna(0)  # Fill NaN with neutral signal
                    signals = signals.clip(-1, 1)  # Ensure signals are in [-1, 1] range
                    return signals
                elif isinstance(signals, (list, np.ndarray)):
                    # Convert to pandas Series
                    return pd.Series(signals, index=market_data.index).fillna(0).clip(-1, 1)
                else:
                    logger.warning(f"Invalid signal format from {type(individual).__name__}: {type(signals)}")
                    return None
            else:
                logger.warning(f"Individual {type(individual).__name__} has no generate_signals method")
                return None
                
        except Exception as e:
            logger.error(f"Error generating signals from {type(individual).__name__}: {e}")
            return None
    
    def generate_synthetic_market_data(self, periods: int = 252) -> pd.DataFrame:
        """Generate synthetic market data for testing strategies."""
        cache_key = f"synthetic_{periods}"
        
        if cache_key in self._synthetic_data_cache:
            return self._synthetic_data_cache[cache_key]
        
        try:
            # Generate random walk with trend and volatility
            np.random.seed(42)  # For reproducible results during testing
            
            dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
            
            # Base price movement (random walk with slight upward bias)
            returns = np.random.normal(0.0005, 0.02, periods)  # Daily returns ~0.05% mean, 2% volatility
            returns[0] = 0  # First return is 0
            
            # Calculate prices from returns
            prices = 100 * (1 + returns).cumprod()  # Start at $100
            
            # Add some realistic price patterns
            high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, periods)))
            low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, periods)))
            
            # Generate volume (somewhat correlated with price movements)
            volume = np.abs(np.random.normal(1000000, 200000, periods))
            volume = volume * (1 + 0.5 * np.abs(returns))  # Higher volume on big moves
            
            # Create DataFrame
            market_data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, periods)),
                'high': np.maximum(prices, high_prices),
                'low': np.minimum(prices, low_prices),
                'close': prices,
                'volume': volume,
                'returns': returns
            }, index=dates)
            
            # Add technical indicators
            market_data = self._add_technical_indicators(market_data)
            
            # Cache the result
            self._synthetic_data_cache[cache_key] = market_data
            
            logger.debug(f"Generated synthetic market data: {periods} periods")
            return market_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic market data: {e}")
            # Return minimal fallback data
            dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
            return pd.DataFrame({
                'open': [100] * periods,
                'high': [101] * periods,
                'low': [99] * periods,
                'close': [100] * periods,
                'volume': [1000000] * periods,
                'returns': [0] * periods
            }, index=dates)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data."""
        try:
            df = data.copy()
            
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)  # Fill NaN with neutral RSI
            
            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            df['bb_std'] = df['close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data  # Return original data if indicator calculation fails
    
    def calculate_strategy_returns(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from trading signals and market data."""
        try:
            if 'returns' in data.columns:
                market_returns = data['returns']
            else:
                # Calculate returns from close prices
                market_returns = data['close'].pct_change().fillna(0)
            
            # Align signals and returns
            aligned_signals = signals.reindex(market_returns.index, method='ffill').fillna(0)
            
            # Strategy returns = signal * market returns (with 1-day lag for realism)
            strategy_returns = aligned_signals.shift(1) * market_returns
            strategy_returns = strategy_returns.fillna(0)
            
            return strategy_returns
            
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {e}")
            return pd.Series([0] * len(signals), index=signals.index)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        try:
            if returns.empty or returns.std() == 0:
                return 0.0
            
            # Annualize returns and volatility (assuming daily data)
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            
            return float(sharpe) if not np.isnan(sharpe) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_consistency(self, returns: pd.Series, window: int = 21) -> float:
        """Calculate consistency metric (positive return periods / total periods)."""
        try:
            if returns.empty:
                return 0.0
            
            # Rolling positive return rate
            rolling_positive = (returns > 0).rolling(window=window, min_periods=1).mean()
            
            # Average consistency over time
            consistency = rolling_positive.mean()
            
            return float(consistency) if not np.isnan(consistency) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            if returns.empty:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Maximum drawdown (most negative value)
            max_dd = drawdown.min()
            
            return float(max_dd) if not np.isnan(max_dd) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)."""
        try:
            if returns.empty:
                return 0.0
            
            positive_returns = (returns > 0).sum()
            total_returns = len(returns)
            
            win_rate = positive_returns / total_returns if total_returns > 0 else 0.0
            
            return float(win_rate)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def calculate_turnover(self, signals: pd.Series) -> float:
        """Calculate portfolio turnover rate."""
        try:
            if signals.empty or len(signals) < 2:
                return 0.0
            
            # Calculate signal changes
            signal_changes = signals.diff().abs()
            
            # Average daily turnover
            turnover = signal_changes.mean()
            
            return float(turnover) if not np.isnan(turnover) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating turnover: {e}")
            return 0.0
    
    def calculate_information_ratio(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """Calculate information ratio vs benchmark."""
        try:
            if returns.empty or benchmark_returns.empty:
                return 0.0
            
            # Align returns
            aligned_returns = returns.reindex(benchmark_returns.index, method='ffill').fillna(0)
            
            # Excess returns
            excess_returns = aligned_returns - benchmark_returns
            
            if excess_returns.std() == 0:
                return 0.0
            
            # Information ratio
            info_ratio = excess_returns.mean() / excess_returns.std()
            
            return float(info_ratio) if not np.isnan(info_ratio) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def evaluate_multi_objective_fitness(self, individual: BaseSeed,
                                       market_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate weighted multi-objective fitness score."""
        try:
            # Get individual fitness components
            sharpe, consistency, max_drawdown, win_rate = self.evaluate_individual(individual, market_data)
            
            # Apply weights
            weighted_score = (
                self.fitness_weights["sharpe_ratio"] * sharpe +
                self.fitness_weights["consistency"] * consistency +
                self.fitness_weights["max_drawdown"] * max_drawdown +  # Already negative
                self.fitness_weights["win_rate"] * win_rate
            )
            
            return float(weighted_score)
            
        except Exception as e:
            logger.error(f"Error calculating multi-objective fitness: {e}")
            return 0.0
    
    def clear_cache(self) -> None:
        """Clear evaluation caches."""
        self._market_data_cache.clear()
        self._synthetic_data_cache.clear()
        logger.info("Evaluation caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'market_data_cache_size': len(self._market_data_cache),
            'synthetic_data_cache_size': len(self._synthetic_data_cache),
            'fitness_weights': self.fitness_weights.copy()
        }