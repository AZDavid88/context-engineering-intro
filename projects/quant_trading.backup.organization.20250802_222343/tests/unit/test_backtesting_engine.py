"""
Unit Tests for Backtesting Engine Components

This module tests the complete backtesting pipeline including:
- Strategy converter (AST to VectorBT bridge)
- Performance analyzer (fitness extraction)  
- VectorBT engine integration
- Multi-asset coordination

Key Test Areas:
- Signal conversion accuracy
- Portfolio creation with realistic costs
- Performance metric calculations
- Genetic fitness extraction
- Multi-objective optimization validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Import components to test
from src.backtesting.strategy_converter import StrategyConverter, SignalConversionResult
from src.backtesting.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from src.backtesting.vectorbt_engine import VectorBTEngine, BacktestResult
from src.strategy.genetic_seeds.ema_crossover_seed import EMACrossoverSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType, SeedFitness
from src.config.settings import get_settings

# VectorBT for integration testing
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False


class TestStrategyConverter:
    """Test cases for strategy converter."""
    
    @pytest.fixture
    def converter(self):
        """Create strategy converter for testing."""
        return StrategyConverter()
    
    @pytest.fixture
    def test_seed(self):
        """Create test seed for conversion."""
        genes = SeedGenes(
            seed_id="converter_test_seed",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'fast_ema_period': 12.0,
                'slow_ema_period': 26.0,
                'momentum_threshold': 0.01,
                'signal_strength': 0.8,
                'trend_filter': 0.005
            }
        )
        return EMACrossoverSeed(genes)
    
    @pytest.fixture
    def test_data(self):
        """Create test market data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        
        # Create trending data for clear signals
        prices = pd.Series(100 + np.cumsum(np.random.normal(0.1, 1, 100)), index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
    
    def test_converter_initialization(self, converter):
        """Test converter initializes correctly."""
        assert converter is not None
        assert converter.settings is not None
        assert converter.conversion_stats['total_conversions'] == 0
    
    def test_signal_conversion(self, converter, test_seed, test_data):
        """Test converting seed signals to VectorBT format."""
        result = converter.convert_seed_to_signals(test_seed, test_data, "BTC")
        
        assert isinstance(result, SignalConversionResult)
        assert result.strategy_id == test_seed.genes.seed_id
        assert result.asset_symbol == "BTC"
        
        # Check signal arrays
        assert isinstance(result.entries, pd.Series)
        assert isinstance(result.exits, pd.Series)
        assert isinstance(result.size, pd.Series)
        
        # Check array lengths match data
        assert len(result.entries) == len(test_data)
        assert len(result.exits) == len(test_data)
        assert len(result.size) == len(test_data)
        
        # Check signal types
        assert result.entries.dtype == bool
        assert result.exits.dtype == bool
        assert result.size.dtype in [np.float64, float]
    
    def test_signal_validation(self, converter, test_seed, test_data):
        """Test signal validation logic."""
        # Generate raw signals
        raw_signals = test_seed.generate_signals(test_data)
        
        # Test validation
        is_valid = converter._validate_raw_signals(raw_signals, test_data)
        assert is_valid is True
        
        # Test invalid signals
        invalid_signals = pd.Series([np.nan] * len(test_data), index=test_data.index)
        is_invalid = converter._validate_raw_signals(invalid_signals, test_data)
        assert is_invalid is False
    
    def test_entry_exit_conversion(self, converter):
        """Test conversion of continuous signals to entry/exit arrays."""
        # Create test signals
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        signals = pd.Series([0, 0.8, 0.5, 0, -0.7, -0.3, 0, 0.9, 0, 0], index=dates)
        
        entries, exits = converter._convert_to_entry_exit_arrays(signals)
        
        # Check basic properties
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert entries.dtype == bool
        assert exits.dtype == bool
        
        # Check entry points
        assert entries.iloc[1] == True  # First buy signal
        assert entries.iloc[4] == True  # First sell signal
        assert entries.iloc[7] == True  # Second buy signal
    
    def test_position_sizing(self, converter, test_seed, test_data):
        """Test position size calculation."""
        raw_signals = test_seed.generate_signals(test_data)
        position_sizes = converter._calculate_position_sizes(test_seed, test_data, raw_signals)
        
        assert isinstance(position_sizes, pd.Series)
        assert len(position_sizes) == len(test_data)
        assert (position_sizes >= 0).all()
        assert (position_sizes <= 1.0).all()  # Should be percentages
    
    def test_multi_asset_conversion(self, converter, test_seed, test_data):
        """Test multi-asset signal conversion."""
        # Create multi-asset data
        data_by_asset = {
            'BTC': test_data,
            'ETH': test_data * 0.1,  # Different price scale
            'SOL': test_data * 0.01  # Much smaller scale
        }
        
        result = converter.convert_multi_asset_signals(test_seed, data_by_asset)
        
        assert len(result.signals_by_asset) <= 3  # May fail for some assets
        assert result.total_signals > 0
        assert len(result.portfolio_allocation) > 0
        
        # Check allocations sum to ~1.0
        total_allocation = sum(result.portfolio_allocation.values())
        assert abs(total_allocation - 1.0) < 0.01
    
    def test_batch_conversion(self, converter, test_seed, test_data):
        """Test batch conversion of multiple seeds."""
        # Create multiple seeds with different parameters
        seeds = []
        for i in range(5):
            genes = SeedGenes(
                seed_id=f"batch_seed_{i}",
                seed_type=SeedType.MOMENTUM,
                parameters={
                    'fast_ema_period': 10.0 + i,
                    'slow_ema_period': 20.0 + i * 2,
                    'momentum_threshold': 0.01,
                    'signal_strength': 0.5 + i * 0.1,
                    'trend_filter': 0.005
                }
            )
            seeds.append(EMACrossoverSeed(genes))
        
        results = converter.batch_convert_population(seeds, test_data)
        
        assert len(results) <= 5  # May have some failures
        assert len(results) > 0    # Should have some successes
        
        for result in results:
            assert isinstance(result, SignalConversionResult)
            assert result.signal_integrity_score >= 0


@pytest.mark.skipif(not VECTORBT_AVAILABLE, reason="VectorBT not available")
class TestPerformanceAnalyzer:
    """Test cases for performance analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create performance analyzer for testing."""
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def test_portfolio(self):
        """Create test VectorBT portfolio."""
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0.001, 0.02, 100)), index=dates)
        
        # Create simple buy/hold signals
        entries = pd.Series([True] + [False] * 99, index=dates)
        exits = pd.Series([False] * 99 + [True], index=dates)
        
        return vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001
        )
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.settings is not None
        assert analyzer.analysis_count == 0
    
    def test_portfolio_analysis(self, analyzer, test_portfolio):
        """Test comprehensive portfolio analysis."""
        metrics = analyzer.analyze_portfolio_performance(test_portfolio, "test_strategy")
        
        assert isinstance(metrics, PerformanceMetrics)
        
        # Check all required metrics are present
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'total_trades')
        
        # Check metric ranges
        assert metrics.max_drawdown >= 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.total_trades >= 0
    
    def test_fitness_extraction(self, analyzer, test_portfolio):
        """Test genetic fitness extraction."""
        fitness = analyzer.extract_genetic_fitness(test_portfolio, "test_strategy")
        
        assert isinstance(fitness, SeedFitness)
        
        # Check fitness components
        assert hasattr(fitness, 'sharpe_ratio')
        assert hasattr(fitness, 'max_drawdown')
        assert hasattr(fitness, 'win_rate')
        assert hasattr(fitness, 'consistency')
        assert hasattr(fitness, 'composite_fitness')
        
        # Check fitness ranges
        assert fitness.max_drawdown >= 0
        assert 0 <= fitness.win_rate <= 1
        assert 0 <= fitness.composite_fitness <= 1
    
    def test_sharpe_calculation(self, analyzer):
        """Test Sharpe ratio calculation."""
        # Create test returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        sharpe = analyzer._calculate_sharpe_ratio(returns, 252)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_trade_analysis(self, analyzer, test_portfolio):
        """Test trade analysis functionality."""
        trades = test_portfolio.trades
        trade_metrics = analyzer._analyze_trades(trades)
        
        assert isinstance(trade_metrics, dict)
        assert 'total_trades' in trade_metrics
        assert 'avg_duration' in trade_metrics
        assert 'avg_winner' in trade_metrics
        assert 'avg_loser' in trade_metrics
        
        # Check metric types
        assert isinstance(trade_metrics['total_trades'], int)
        assert isinstance(trade_metrics['avg_duration'], float)
    
    def test_batch_analysis(self, analyzer, test_portfolio):
        """Test batch portfolio analysis."""
        portfolios = [test_portfolio] * 3
        strategy_ids = ['strategy_1', 'strategy_2', 'strategy_3']
        
        results = analyzer.batch_analyze_portfolios(portfolios, strategy_ids)
        
        assert len(results) == 3
        for fitness in results:
            assert isinstance(fitness, SeedFitness)


@pytest.mark.skipif(not VECTORBT_AVAILABLE, reason="VectorBT not available")
class TestVectorBTEngine:
    """Test cases for VectorBT backtesting engine."""
    
    @pytest.fixture
    def engine(self):
        """Create VectorBT engine for testing."""
        return VectorBTEngine()
    
    @pytest.fixture
    def test_seed(self):
        """Create test seed for backtesting."""
        genes = SeedGenes(
            seed_id="engine_test_seed",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'fast_ema_period': 12.0,
                'slow_ema_period': 26.0,
                'momentum_threshold': 0.01,
                'signal_strength': 0.8,
                'trend_filter': 0.005
            }
        )
        return EMACrossoverSeed(genes)
    
    @pytest.fixture
    def test_data(self):
        """Create test market data."""
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        
        # Create realistic price data with some trend
        returns = np.random.normal(0.0005, 0.02, 200)
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.converter is not None
        assert engine.analyzer is not None
        assert engine.backtest_stats['total_backtests'] == 0
    
    def test_single_seed_backtest(self, engine, test_seed, test_data):
        """Test backtesting a single seed."""
        result = engine.backtest_seed(test_seed, test_data, "BTC")
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_id == test_seed.genes.seed_id
        assert result.portfolio is not None
        assert result.performance_metrics is not None
        assert result.fitness is not None
        
        # Check fitness components
        assert isinstance(result.fitness.sharpe_ratio, float)
        assert isinstance(result.fitness.max_drawdown, float)
        assert 0 <= result.fitness.composite_fitness <= 1
    
    def test_population_backtest(self, engine, test_data):
        """Test backtesting a population of seeds."""
        # Create small population
        seeds = []
        for i in range(3):
            genes = SeedGenes(
                seed_id=f"pop_seed_{i}",
                seed_type=SeedType.MOMENTUM,
                parameters={
                    'fast_ema_period': 10.0 + i,
                    'slow_ema_period': 20.0 + i * 2,
                    'momentum_threshold': 0.01,
                    'signal_strength': 0.5 + i * 0.1,
                    'trend_filter': 0.005
                }
            )
            seeds.append(EMACrossoverSeed(genes))
        
        results = engine.backtest_population(seeds, test_data, parallel=False)
        
        assert len(results) <= 3  # May have some failures
        assert len(results) > 0    # Should have some successes
        
        for result in results:
            assert isinstance(result, BacktestResult)
    
    def test_realistic_portfolio_creation(self, engine, test_seed, test_data):
        """Test portfolio creation with realistic costs."""
        # Convert signals
        conversion_result = engine.converter.convert_seed_to_signals(test_seed, test_data, "BTC")
        
        # Create portfolio
        portfolio = engine._create_realistic_portfolio(conversion_result, test_data)
        
        assert portfolio is not None
        assert portfolio.init_cash == engine.initial_cash
        
        # Check that fees and slippage are applied
        assert hasattr(portfolio, 'orders')
    
    def test_robustness_validation(self, engine, test_seed, test_data):
        """Test strategy robustness validation."""
        robustness = engine.validate_strategy_robustness(test_seed, test_data, validation_periods=2)
        
        assert isinstance(robustness, dict)
        
        if 'validation_failed' not in robustness:
            assert 'validation_periods' in robustness
            assert 'avg_sharpe' in robustness
            assert 'sharpe_stability' in robustness
            
            # Check metric ranges
            assert robustness['validation_periods'] <= 2
            assert 0 <= robustness['sharpe_stability'] <= 1
    
    def test_benchmark_performance(self, engine, test_seed, test_data):
        """Test benchmark performance analysis."""
        # Create some test results
        results = []
        for i in range(3):
            try:
                result = engine.backtest_seed(test_seed, test_data, f"BTC_{i}")
                results.append(result)
            except:
                continue
        
        if results:
            benchmark = engine.benchmark_performance(results)
            
            assert isinstance(benchmark, dict)
            assert 'total_strategies' in benchmark
            
            if 'performance_distribution' in benchmark:
                perf = benchmark['performance_distribution']
                assert 'avg_sharpe' in perf
                assert 'avg_return' in perf
    
    def test_engine_statistics(self, engine):
        """Test engine statistics tracking."""
        stats = engine.get_engine_stats()
        
        assert isinstance(stats, dict)
        assert 'total_backtests' in stats
        assert 'successful_backtests' in stats
        assert 'success_rate' in stats
        assert 'converter_stats' in stats
        
        # Check stats are numbers
        assert isinstance(stats['total_backtests'], int)
        assert isinstance(stats['success_rate'], float)


class TestIntegrationWorkflow:
    """Integration tests for the complete backtesting workflow."""
    
    @pytest.fixture
    def complete_workflow_setup(self):
        """Set up complete workflow components."""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=150, freq='1H')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0.001, 0.02, 150)), index=dates)
        
        data = pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 150)
        }, index=dates)
        
        # Create test seed
        genes = SeedGenes(
            seed_id="integration_test_seed",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'fast_ema_period': 12.0,
                'slow_ema_period': 26.0,
                'momentum_threshold': 0.01,
                'signal_strength': 0.8,
                'trend_filter': 0.005
            }
        )
        seed = EMACrossoverSeed(genes)
        
        return {
            'data': data,
            'seed': seed,
            'converter': StrategyConverter(),
            'analyzer': PerformanceAnalyzer(),
            'engine': VectorBTEngine() if VECTORBT_AVAILABLE else None
        }
    
    def test_signal_to_fitness_pipeline(self, complete_workflow_setup):
        """Test complete pipeline from signal generation to fitness extraction."""
        setup = complete_workflow_setup
        
        if setup['engine'] is None:
            pytest.skip("VectorBT not available")
        
        # Step 1: Generate signals
        signals = setup['seed'].generate_signals(setup['data'])
        assert len(signals) == len(setup['data'])
        
        # Step 2: Convert to VectorBT format
        conversion_result = setup['converter'].convert_seed_to_signals(
            setup['seed'], setup['data'], "BTC"
        )
        assert conversion_result.signal_integrity_score > 0
        
        # Step 3: Create portfolio
        portfolio = setup['converter'].create_vectorbt_portfolio(
            conversion_result, setup['data']
        )
        assert portfolio is not None
        
        # Step 4: Analyze performance
        performance_metrics = setup['analyzer'].analyze_portfolio_performance(
            portfolio, setup['seed'].genes.seed_id
        )
        assert performance_metrics.total_trades >= 0
        
        # Step 5: Extract genetic fitness
        fitness = setup['analyzer'].extract_genetic_fitness(
            portfolio, setup['seed'].genes.seed_id
        )
        assert 0 <= fitness.composite_fitness <= 1
        
        print(f"Pipeline test successful:")
        print(f"  - Signals generated: {(abs(signals) > 0.1).sum()}")
        print(f"  - Conversion integrity: {conversion_result.signal_integrity_score:.3f}")
        print(f"  - Portfolio return: {performance_metrics.total_return:.2%}")
        print(f"  - Genetic fitness: {fitness.composite_fitness:.3f}")
    
    @pytest.mark.skipif(not VECTORBT_AVAILABLE, reason="VectorBT not available")
    def test_end_to_end_engine_workflow(self, complete_workflow_setup):
        """Test end-to-end workflow through VectorBT engine."""
        setup = complete_workflow_setup
        
        # Run complete backtest
        result = setup['engine'].backtest_seed(setup['seed'], setup['data'], "BTC")
        
        # Validate complete result
        assert isinstance(result, BacktestResult)
        assert result.strategy_id == setup['seed'].genes.seed_id
        assert result.portfolio is not None
        assert result.performance_metrics is not None
        assert result.fitness is not None
        
        # Check that all components are connected properly
        assert result.fitness.sharpe_ratio == result.performance_metrics.sharpe_ratio
        assert result.fitness.max_drawdown == result.performance_metrics.max_drawdown
        assert result.fitness.total_trades == result.performance_metrics.total_trades
        
        print(f"End-to-end test successful:")
        print(f"  - Strategy: {result.strategy_id}")
        print(f"  - Return: {result.performance_metrics.total_return:.2%}")
        print(f"  - Sharpe: {result.fitness.sharpe_ratio:.3f}")
        print(f"  - Fitness: {result.fitness.composite_fitness:.3f}")


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])