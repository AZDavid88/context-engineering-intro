"""
Performance Analyzer - Multi-Objective Fitness Extraction

This module extracts genetic fitness scores from VectorBT backtest results,
providing the critical feedback loop for genetic algorithm evolution.

This addresses GAP 5 from the PRP: Missing Performance Feedback Loop.

Key Features:
- Multi-objective fitness calculation (Sharpe + Consistency + Drawdown + Turnover)
- Statistical confidence validation through multiple metrics
- Regime-based performance analysis
- Risk-adjusted return calculations
- Transaction cost impact assessment
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
import logging
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from dataclasses import dataclass

# VectorBT imports (from comprehensive research)
import vectorbt as vbt

from src.strategy.genetic_seeds.base_seed import SeedFitness
from src.config.settings import get_settings, Settings


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_drawdown: float
    
    # Consistency metrics
    win_rate: float
    profit_factor: float
    expectancy: float
    consistency_ratio: float
    
    # Trade statistics
    total_trades: int
    avg_trade_duration: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Turnover and costs
    turnover_rate: float
    transaction_costs: float
    net_profit_after_costs: float
    
    # Risk-adjusted metrics
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    
    # Regime performance
    bull_market_return: float
    bear_market_return: float
    sideways_market_return: float


class PerformanceAnalyzer:
    """Analyzes VectorBT portfolio performance and extracts genetic fitness."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize performance analyzer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.Analyzer")
        
        # Benchmark data for excess return calculation
        self.benchmark_return = 0.0  # Risk-free rate or market return
        
        # Performance tracking
        self.analysis_count = 0
        self.fitness_cache: Dict[str, SeedFitness] = {}
    
    def analyze_portfolio_performance(self, portfolio: vbt.Portfolio, 
                                    strategy_id: str = "unknown") -> PerformanceMetrics:
        """Comprehensive performance analysis of VectorBT portfolio.
        
        Args:
            portfolio: VectorBT Portfolio object
            strategy_id: Strategy identifier for caching
            
        Returns:
            Complete performance metrics
        """
        try:
            # Basic return metrics
            total_return = portfolio.total_return()
            returns = portfolio.returns()
            
            # Handle edge cases
            if len(returns) == 0 or returns.isna().all():
                return self._create_zero_performance_metrics()
            
            # Calculate annualized return
            trading_days = len(returns)
            periods_per_year = 365 * 24  # Assuming hourly data
            annualized_return = (1 + total_return) ** (periods_per_year / trading_days) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(periods_per_year)
            sharpe_ratio = self._calculate_sharpe_ratio(returns, periods_per_year)
            sortino_ratio = self._calculate_sortino_ratio(returns, periods_per_year)
            
            # Drawdown metrics
            drawdown = portfolio.drawdown()
            max_drawdown = portfolio.max_drawdown()
            avg_drawdown = drawdown.mean()
            
            # Trade analysis
            trades = portfolio.trades
            trade_metrics = self._analyze_trades(trades)
            
            # Consistency metrics
            consistency_metrics = self._calculate_consistency_metrics(returns, trades)
            
            # Turnover and cost analysis
            turnover_metrics = self._calculate_turnover_metrics(portfolio)
            
            # Risk-adjusted ratios
            risk_ratios = self._calculate_risk_adjusted_ratios(
                annualized_return, volatility, max_drawdown
            )
            
            # Regime performance analysis
            regime_performance = self._analyze_regime_performance(portfolio, returns)
            
            # Combine all metrics
            metrics = PerformanceMetrics(
                # Returns
                total_return=total_return,
                annualized_return=annualized_return,
                excess_return=annualized_return - self.benchmark_return,
                
                # Risk
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                avg_drawdown=avg_drawdown,
                
                # Consistency
                win_rate=consistency_metrics['win_rate'],
                profit_factor=consistency_metrics['profit_factor'],
                expectancy=consistency_metrics['expectancy'],
                consistency_ratio=consistency_metrics['consistency_ratio'],
                
                # Trades
                total_trades=trade_metrics['total_trades'],
                avg_trade_duration=trade_metrics['avg_duration'],
                avg_winner=trade_metrics['avg_winner'],
                avg_loser=trade_metrics['avg_loser'],
                largest_winner=trade_metrics['largest_winner'],
                largest_loser=trade_metrics['largest_loser'],
                max_consecutive_wins=trade_metrics['max_consecutive_wins'],
                max_consecutive_losses=trade_metrics['max_consecutive_losses'],
                
                # Turnover
                turnover_rate=turnover_metrics['turnover_rate'],
                transaction_costs=turnover_metrics['transaction_costs'],
                net_profit_after_costs=turnover_metrics['net_profit_after_costs'],
                
                # Risk-adjusted
                calmar_ratio=risk_ratios['calmar_ratio'],
                sterling_ratio=risk_ratios['sterling_ratio'],
                burke_ratio=risk_ratios['burke_ratio'],
                
                # Regime
                bull_market_return=regime_performance['bull_return'],
                bear_market_return=regime_performance['bear_return'],
                sideways_market_return=regime_performance['sideways_return']
            )
            
            self.analysis_count += 1
            self.logger.debug(f"Analyzed portfolio {strategy_id}: "
                            f"Return={total_return:.2%}, Sharpe={sharpe_ratio:.3f}, "
                            f"Drawdown={max_drawdown:.2%}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio {strategy_id}: {e}")
            return self._create_zero_performance_metrics()
    
    def extract_genetic_fitness(self, portfolio: vbt.Portfolio, 
                              strategy_id: str = "unknown") -> SeedFitness:
        """Extract genetic fitness from portfolio performance.
        
        Args:
            portfolio: VectorBT Portfolio object
            strategy_id: Strategy identifier
            
        Returns:
            SeedFitness object for genetic algorithm
        """
        # Check cache first
        if strategy_id in self.fitness_cache:
            return self.fitness_cache[strategy_id]
        
        # Analyze performance
        metrics = self.analyze_portfolio_performance(portfolio, strategy_id)
        
        # Calculate multi-objective fitness components
        fitness_components = self._calculate_fitness_components(metrics)
        
        # Create SeedFitness object
        fitness = SeedFitness(
            # Primary fitness metrics
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            consistency=fitness_components['consistency_score'],
            
            # Auxiliary metrics
            total_return=metrics.total_return,
            volatility=metrics.volatility,
            profit_factor=metrics.profit_factor,
            
            # Trade statistics
            total_trades=metrics.total_trades,
            avg_trade_duration=metrics.avg_trade_duration,
            max_consecutive_losses=metrics.max_consecutive_losses,
            
            # Composite fitness (calculated by SeedFitness validator)
            composite_fitness=0.0,  # Will be calculated
            
            # Validation periods (placeholder - would be calculated with walk-forward)
            in_sample_fitness=metrics.sharpe_ratio * 0.8,
            out_of_sample_fitness=metrics.sharpe_ratio * 0.6,
            walk_forward_fitness=metrics.sharpe_ratio * 0.7
        )
        
        # Cache result
        self.fitness_cache[strategy_id] = fitness
        
        return fitness
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int) -> float:
        """Calculate Sharpe ratio with proper risk-free rate adjustment.
        
        Args:
            returns: Portfolio returns
            periods_per_year: Number of periods per year for annualization
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Convert benchmark to period frequency
        risk_free_period = self.benchmark_return / periods_per_year
        excess_returns = returns - risk_free_period
        
        return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int) -> float:
        """Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Portfolio returns
            periods_per_year: Periods per year for annualization
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf if returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return np.inf if returns.mean() > 0 else 0.0
        
        risk_free_period = self.benchmark_return / periods_per_year
        excess_return = (returns.mean() - risk_free_period) * periods_per_year
        
        return excess_return / downside_deviation
    
    def _analyze_trades(self, trades) -> Dict[str, Any]:
        """Analyze individual trades for statistics.
        
        Args:
            trades: VectorBT trades object
            
        Returns:
            Dictionary of trade statistics
        """
        if trades.count() == 0:
            return {
                'total_trades': 0,
                'avg_duration': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'largest_winner': 0.0,
                'largest_loser': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        try:
            # Trade returns
            trade_returns = trades.returns.values
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            
            # Duration analysis
            durations = trades.duration.values
            avg_duration = float(durations.mean()) if len(durations) > 0 else 0.0
            
            # Win/loss analysis
            avg_winner = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
            avg_loser = float(losing_trades.mean()) if len(losing_trades) > 0 else 0.0
            largest_winner = float(winning_trades.max()) if len(winning_trades) > 0 else 0.0
            largest_loser = float(losing_trades.min()) if len(losing_trades) > 0 else 0.0
            
            # Consecutive wins/losses
            consecutive_stats = self._calculate_consecutive_trades(trade_returns)
            
            return {
                'total_trades': int(trades.count()),
                'avg_duration': avg_duration,
                'avg_winner': avg_winner,
                'avg_loser': avg_loser,
                'largest_winner': largest_winner,
                'largest_loser': largest_loser,
                'max_consecutive_wins': consecutive_stats['max_wins'],
                'max_consecutive_losses': consecutive_stats['max_losses']
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing trades: {e}")
            return {
                'total_trades': 0,
                'avg_duration': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'largest_winner': 0.0,
                'largest_loser': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
    
    def _calculate_consecutive_trades(self, trade_returns: np.ndarray) -> Dict[str, int]:
        """Calculate maximum consecutive wins and losses.
        
        Args:
            trade_returns: Array of trade returns
            
        Returns:
            Dictionary with max consecutive wins and losses
        """
        if len(trade_returns) == 0:
            return {'max_wins': 0, 'max_losses': 0}
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for ret in trade_returns:
            if ret > 0:  # Winning trade
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif ret < 0:  # Losing trade
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # Breakeven trade
                current_wins = 0
                current_losses = 0
        
        return {'max_wins': max_wins, 'max_losses': max_losses}
    
    def _calculate_consistency_metrics(self, returns: pd.Series, trades) -> Dict[str, float]:
        """Calculate consistency and reliability metrics.
        
        Args:
            returns: Portfolio returns
            trades: VectorBT trades object
            
        Returns:
            Dictionary of consistency metrics
        """
        if len(returns) == 0 or trades.count() == 0:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'consistency_ratio': 0.0
            }
        
        try:
            # Trade-based metrics
            trade_returns = trades.returns.values
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            
            # Win rate
            win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0.0
            
            # Profit factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0.0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.01  # Avoid division by zero
            profit_factor = gross_profit / gross_loss
            
            # Expectancy
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
            avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Consistency ratio (stability of returns)
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if len(monthly_returns) > 1 and monthly_returns.std() > 0:
                consistency_ratio = monthly_returns.mean() / monthly_returns.std()
            else:
                consistency_ratio = 0.0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'consistency_ratio': consistency_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating consistency metrics: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'consistency_ratio': 0.0
            }
    
    def _calculate_turnover_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """Calculate turnover and transaction cost metrics.
        
        Args:
            portfolio: VectorBT Portfolio object
            
        Returns:
            Dictionary of turnover metrics
        """
        try:
            # Get portfolio orders
            orders = portfolio.orders
            
            if orders.count() == 0:
                return {
                    'turnover_rate': 0.0,
                    'transaction_costs': 0.0,
                    'net_profit_after_costs': 0.0
                }
            
            # Calculate turnover rate
            total_value_traded = orders.value.sum()
            avg_portfolio_value = portfolio.value().mean()
            turnover_rate = total_value_traded / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
            
            # Calculate transaction costs
            # Costs = (order_value * fee_rate) for each order
            fees = self.settings.trading.taker_fee  # Use taker fee as conservative estimate
            transaction_costs = total_value_traded * fees
            
            # Net profit after costs
            gross_profit = portfolio.total_return() * portfolio.init_cash
            net_profit_after_costs = gross_profit - transaction_costs
            
            return {
                'turnover_rate': turnover_rate,
                'transaction_costs': transaction_costs,
                'net_profit_after_costs': net_profit_after_costs
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating turnover metrics: {e}")
            return {
                'turnover_rate': 0.0,
                'transaction_costs': 0.0,
                'net_profit_after_costs': 0.0
            }
    
    def _calculate_risk_adjusted_ratios(self, annual_return: float, volatility: float, 
                                      max_drawdown: float) -> Dict[str, float]:
        """Calculate various risk-adjusted performance ratios.
        
        Args:
            annual_return: Annualized return
            volatility: Return volatility
            max_drawdown: Maximum drawdown
            
        Returns:
            Dictionary of risk-adjusted ratios
        """
        # Calmar ratio (return / max drawdown)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Sterling ratio (similar to Calmar but with average drawdown)
        avg_drawdown = max_drawdown * 0.7  # Approximate average as 70% of max
        sterling_ratio = annual_return / avg_drawdown if avg_drawdown > 0 else 0.0
        
        # Burke ratio (excess return / downside deviation)
        # Simplified version using volatility as proxy
        burke_ratio = annual_return / volatility if volatility > 0 else 0.0
        
        return {
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'burke_ratio': burke_ratio
        }
    
    def _analyze_regime_performance(self, portfolio: vbt.Portfolio, 
                                  returns: pd.Series) -> Dict[str, float]:
        """Analyze performance across different market regimes.
        
        Args:
            portfolio: VectorBT Portfolio object
            returns: Portfolio returns
            
        Returns:
            Dictionary of regime-specific performance
        """
        try:
            # Simple regime classification based on rolling volatility
            rolling_vol = returns.rolling(window=30).std()
            vol_threshold_high = rolling_vol.quantile(0.75)
            vol_threshold_low = rolling_vol.quantile(0.25)
            
            # Classify regimes
            high_vol_periods = rolling_vol > vol_threshold_high  # Bear/volatile
            low_vol_periods = rolling_vol < vol_threshold_low   # Bull/stable
            medium_vol_periods = ~(high_vol_periods | low_vol_periods)  # Sideways
            
            # Calculate returns for each regime
            bull_return = returns[low_vol_periods].mean() * 252 if low_vol_periods.any() else 0.0
            bear_return = returns[high_vol_periods].mean() * 252 if high_vol_periods.any() else 0.0
            sideways_return = returns[medium_vol_periods].mean() * 252 if medium_vol_periods.any() else 0.0
            
            return {
                'bull_return': bull_return,
                'bear_return': bear_return,
                'sideways_return': sideways_return
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing regime performance: {e}")
            return {
                'bull_return': 0.0,
                'bear_return': 0.0,
                'sideways_return': 0.0
            }
    
    def _calculate_fitness_components(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Calculate fitness components for genetic algorithm.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Dictionary of fitness components
        """
        # Consistency score combines multiple stability metrics
        consistency_factors = [
            min(1.0, metrics.win_rate * 2),  # Win rate normalized
            min(1.0, metrics.profit_factor / 3),  # Profit factor normalized
            max(0.0, min(1.0, metrics.consistency_ratio + 0.5)),  # Consistency ratio
            max(0.0, 1.0 - abs(metrics.max_drawdown) * 5)  # Drawdown penalty
        ]
        
        consistency_score = sum(consistency_factors) / len(consistency_factors)
        
        # Turnover efficiency (lower turnover is better for costs)
        turnover_efficiency = max(0.0, 1.0 - metrics.turnover_rate / 10)
        
        # Risk-adjusted performance
        risk_adjusted_score = (
            metrics.sharpe_ratio / 5.0 +  # Normalize Sharpe
            metrics.calmar_ratio / 10.0   # Normalize Calmar
        ) / 2.0
        
        return {
            'consistency_score': consistency_score,
            'turnover_efficiency': turnover_efficiency,
            'risk_adjusted_score': risk_adjusted_score
        }
    
    def _create_zero_performance_metrics(self) -> PerformanceMetrics:
        """Create zero-filled performance metrics for failed strategies.
        
        Returns:
            PerformanceMetrics with zero/default values
        """
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            excess_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=1.0,  # Maximum possible drawdown
            avg_drawdown=0.5,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            consistency_ratio=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            avg_winner=0.0,
            avg_loser=0.0,
            largest_winner=0.0,
            largest_loser=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=100,  # High penalty
            turnover_rate=0.0,
            transaction_costs=0.0,
            net_profit_after_costs=0.0,
            calmar_ratio=0.0,
            sterling_ratio=0.0,
            burke_ratio=0.0,
            bull_market_return=0.0,
            bear_market_return=0.0,
            sideways_market_return=0.0
        )
    
    def batch_analyze_portfolios(self, portfolios: List[vbt.Portfolio], 
                               strategy_ids: List[str]) -> List[SeedFitness]:
        """Analyze multiple portfolios efficiently.
        
        Args:
            portfolios: List of VectorBT portfolios
            strategy_ids: List of strategy identifiers
            
        Returns:
            List of fitness results
        """
        results = []
        
        self.logger.info(f"Analyzing batch of {len(portfolios)} portfolios...")
        
        for i, (portfolio, strategy_id) in enumerate(zip(portfolios, strategy_ids)):
            try:
                fitness = self.extract_genetic_fitness(portfolio, strategy_id)
                results.append(fitness)
                
                if (i + 1) % 50 == 0:  # Progress logging
                    self.logger.info(f"Analyzed {i + 1}/{len(portfolios)} portfolios")
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze portfolio {strategy_id}: {e}")
                # Create default fitness for failed analysis
                default_fitness = SeedFitness(
                    sharpe_ratio=0.0,
                    max_drawdown=1.0,
                    win_rate=0.0,
                    consistency=0.0,
                    total_return=0.0,
                    volatility=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    avg_trade_duration=0.0,
                    max_consecutive_losses=100,
                    composite_fitness=0.0,
                    in_sample_fitness=0.0,
                    out_of_sample_fitness=0.0,
                    walk_forward_fitness=0.0
                )
                results.append(default_fitness)
        
        self.logger.info(f"Batch analysis complete: {len(results)} results")
        return results
    
    def get_performance_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Get human-readable performance summary.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Dictionary with formatted summary
        """
        return {
            'returns': {
                'total_return': f"{metrics.total_return:.2%}",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'excess_return': f"{metrics.excess_return:.2%}"
            },
            'risk': {
                'volatility': f"{metrics.volatility:.2%}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.3f}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'calmar_ratio': f"{metrics.calmar_ratio:.3f}"
            },
            'trading': {
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate:.1%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'avg_trade_duration': f"{metrics.avg_trade_duration:.1f}h"
            },
            'costs': {
                'turnover_rate': f"{metrics.turnover_rate:.2f}",
                'transaction_costs': f"${metrics.transaction_costs:.2f}",
                'net_profit_after_costs': f"${metrics.net_profit_after_costs:.2f}"
            }
        }


async def test_performance_analyzer():
    """Test function for performance analyzer."""
    
    print("=== Performance Analyzer Test ===")
    
    # Create test portfolio using VectorBT
    import numpy as np
    
    # Generate test data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    prices = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.0001, 0.02, 1000)),
        index=dates
    )
    
    # Create simple trading signals
    returns = prices.pct_change()
    signals = (returns > 0.01).astype(int) - (returns < -0.01).astype(int)
    
    # Create VectorBT portfolio
    entries = signals == 1
    exits = signals == -1
    
    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001
    )
    
    # Test performance analysis
    analyzer = PerformanceAnalyzer()
    
    print(f"Analyzing test portfolio...")
    metrics = analyzer.analyze_portfolio_performance(portfolio, "test_strategy")
    
    print(f"Performance Analysis Complete!")
    print(f"  - Total Return: {metrics.total_return:.2%}")
    print(f"  - Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  - Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  - Win Rate: {metrics.win_rate:.1%}")
    print(f"  - Total Trades: {metrics.total_trades}")
    
    # Test genetic fitness extraction
    print(f"\nExtracting genetic fitness...")
    fitness = analyzer.extract_genetic_fitness(portfolio, "test_strategy")
    
    print(f"Genetic Fitness Extracted!")
    print(f"  - Composite Fitness: {fitness.composite_fitness:.3f}")
    print(f"  - Sharpe Component: {fitness.sharpe_ratio:.3f}")
    print(f"  - Drawdown Component: {fitness.max_drawdown:.3f}")
    print(f"  - Consistency Component: {fitness.consistency:.3f}")
    
    # Test performance summary
    print(f"\nPerformance Summary:")
    summary = analyzer.get_performance_summary(metrics)
    for category, values in summary.items():
        print(f"  {category.upper()}:")
        for key, value in values.items():
            print(f"    - {key}: {value}")
    
    print(f"\n✅ Performance Analyzer test completed successfully!")


if __name__ == "__main__":
    """Test the performance analyzer."""
    
    import asyncio
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_performance_analyzer())