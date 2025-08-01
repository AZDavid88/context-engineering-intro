"""
Unit Tests for Genetic Seed Library

This module provides comprehensive unit tests for all genetic seeds,
validating signal generation, parameter bounds, and fitness calculation
as specified in the consultant recommendations.

Key Test Areas:
- Seed registration and validation
- Signal generation with synthetic data  
- Parameter bounds validation
- Technical indicator calculations
- Risk management functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any

# Import components to test
from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes, SeedType, SeedFitness
from src.strategy.genetic_seeds.seed_registry import SeedRegistry, get_registry
from src.strategy.genetic_seeds.ema_crossover_seed import EMACrossoverSeed
from src.strategy.genetic_seeds.donchian_breakout_seed import DonchianBreakoutSeed
from src.strategy.genetic_seeds.rsi_filter_seed import RSIFilterSeed
from src.config.settings import get_settings


class TestSeedRegistry:
    """Test cases for genetic seed registry."""
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        registry = SeedRegistry()
        assert registry is not None
        assert len(registry._registry) >= 0
        assert len(registry._validation_functions) > 0
    
    def test_automatic_seed_registration(self):
        """Test that seeds are automatically registered via decorator."""
        registry = get_registry()
        
        # Check that our test seeds are registered
        expected_seeds = ['EMA_Crossover', 'Donchian_Breakout', 'RSI_Filter']
        
        for seed_name in expected_seeds:
            assert seed_name in registry._registry
            registration = registry._registry[seed_name]
            assert registration.status.value == 'registered'
    
    def test_seed_class_retrieval(self):
        """Test retrieving seed classes from registry."""
        registry = get_registry()
        
        ema_class = registry.get_seed_class('EMA_Crossover')
        assert ema_class is not None
        assert ema_class == EMACrossoverSeed
        
        # Test non-existent seed
        non_existent = registry.get_seed_class('NonExistentSeed')
        assert non_existent is None
    
    def test_seeds_by_type(self):
        """Test filtering seeds by type."""
        registry = get_registry()
        
        momentum_seeds = registry.get_seeds_by_type(SeedType.MOMENTUM)
        assert len(momentum_seeds) > 0
        assert EMACrossoverSeed in momentum_seeds
        
        breakout_seeds = registry.get_seeds_by_type(SeedType.BREAKOUT)
        assert len(breakout_seeds) > 0
        assert DonchianBreakoutSeed in breakout_seeds
    
    def test_random_population_creation(self):
        """Test creating random population from registry."""
        registry = get_registry()
        
        population = registry.create_random_population(10)
        assert len(population) <= 10  # May be less if some seeds fail
        
        for seed in population:
            assert isinstance(seed, BaseSeed)
            assert seed.genes is not None
            assert seed.genes.seed_id is not None


class TestBaseSeed:
    """Test cases for base seed functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        
        # Create realistic price data
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.02, 100)
        prices = pd.Series(base_price * np.cumprod(1 + returns), index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(base_price),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
    
    @pytest.fixture
    def test_genes(self):
        """Create test genetic parameters."""
        return SeedGenes(
            seed_id="test_seed",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'test_param1': 0.5,
                'test_param2': 10.0
            }
        )
    
    def test_seed_genes_validation(self, test_genes):
        """Test seed genes validation."""
        assert test_genes.seed_id == "test_seed"
        assert test_genes.seed_type == SeedType.MOMENTUM
        assert 'test_param1' in test_genes.parameters
        
        # Test parameter validation
        with pytest.raises(ValueError):
            SeedGenes(
                seed_id="invalid",
                seed_type=SeedType.MOMENTUM,
                parameters={'invalid_param': 'not_numeric'}
            )
    
    def test_seed_fitness_calculation(self):
        """Test fitness calculation and validation."""
        fitness = SeedFitness(
            sharpe_ratio=2.5,
            max_drawdown=0.08,
            win_rate=0.65,
            consistency=0.75,
            total_return=0.15,
            volatility=0.12,
            profit_factor=1.8,
            total_trades=50,
            avg_trade_duration=24.0,
            max_consecutive_losses=3,
            composite_fitness=0.0,  # Will be calculated
            in_sample_fitness=0.0,
            out_of_sample_fitness=0.0,
            walk_forward_fitness=0.0
        )
        
        # Check that composite fitness was calculated
        assert fitness.composite_fitness > 0
        assert 0 <= fitness.composite_fitness <= 1
        
        # Check individual components
        assert fitness.sharpe_ratio == 2.5
        assert fitness.win_rate == 0.65


class TestEMACrossoverSeed:
    """Test cases for EMA Crossover seed."""
    
    @pytest.fixture
    def ema_seed(self):
        """Create EMA crossover seed for testing."""
        genes = SeedGenes(
            seed_id="test_ema",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'fast_ema_period': 5.0,      # More sensitive
                'slow_ema_period': 20.0,     # More sensitive  
                'momentum_threshold': 0.001, # Lower threshold
                'signal_strength': 1.0,      # Maximum strength
                'trend_filter': 0.0         # No filter
            }
        )
        return EMACrossoverSeed(genes)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with clear trend for EMA testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')  # Fixed frequency
        
        # Create more volatile trending price data
        trend = np.linspace(100, 120, 100)  # Upward trend
        volatility = 3 * np.sin(np.linspace(0, 4*np.pi, 100))  # Cyclical volatility
        noise = np.random.normal(0, 2, 100)  # More noise
        prices = pd.Series(trend + volatility + noise, index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
    
    def test_ema_seed_initialization(self, ema_seed):
        """Test EMA seed initializes correctly."""
        assert ema_seed.seed_name == "EMA_Crossover"
        assert ema_seed.genes.seed_type == SeedType.MOMENTUM
        assert 'fast_ema_period' in ema_seed.genes.parameters
        assert 'slow_ema_period' in ema_seed.genes.parameters
    
    def test_parameter_bounds(self, ema_seed):
        """Test parameter bounds are correctly defined."""
        bounds = ema_seed.parameter_bounds
        
        assert 'fast_ema_period' in bounds
        assert 'slow_ema_period' in bounds
        
        # Check bounds are tuples with min < max
        for param, (min_val, max_val) in bounds.items():
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert min_val < max_val
    
    def test_required_parameters(self, ema_seed):
        """Test required parameters are correctly specified."""
        required = ema_seed.required_parameters
        
        assert isinstance(required, list)
        assert len(required) > 0
        assert 'fast_ema_period' in required
        assert 'slow_ema_period' in required
    
    def test_technical_indicators_calculation(self, ema_seed, sample_data):
        """Test technical indicator calculations."""
        indicators = ema_seed.calculate_technical_indicators(sample_data)
        
        assert isinstance(indicators, dict)
        assert 'fast_ema' in indicators
        assert 'slow_ema' in indicators
        
        # Check indicator series have correct length
        assert len(indicators['fast_ema']) == len(sample_data)
        assert len(indicators['slow_ema']) == len(sample_data)
        
        # Check no NaN values at the end
        assert not indicators['fast_ema'].iloc[-10:].isna().any()
        assert not indicators['slow_ema'].iloc[-10:].isna().any()
    
    def test_signal_generation(self, ema_seed, sample_data):
        """Test signal generation produces valid signals."""
        signals = ema_seed.generate_signals(sample_data)
        
        # Check signal format
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        
        # Check signal range
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0
        
        # Check for some non-zero signals
        non_zero_signals = (signals != 0).sum()
        assert non_zero_signals > 0  # Should have some signals
        
        # Check no NaN values
        assert not signals.isna().any()
    
    def test_position_sizing(self, ema_seed, sample_data):
        """Test position sizing calculation."""
        signal = 0.8  # Strong buy signal
        
        position_size = ema_seed.calculate_position_size(sample_data, signal)
        
        assert isinstance(position_size, float)
        assert 0 <= position_size <= 1.0  # Should be percentage
        
        # Test with weak signal
        weak_signal = 0.05
        weak_position = ema_seed.calculate_position_size(sample_data, weak_signal)
        assert weak_position < position_size  # Weaker signal = smaller position
    
    def test_risk_management(self, ema_seed, sample_data):
        """Test risk management functionality."""
        signals = ema_seed.generate_signals(sample_data)
        risk_adjusted = ema_seed.apply_risk_management(sample_data, signals)
        
        assert isinstance(risk_adjusted, pd.Series)
        assert len(risk_adjusted) == len(signals)
        
        # Risk management should not increase signal magnitude
        assert (abs(risk_adjusted) <= abs(signals)).all()
    
    def test_mutation(self, ema_seed):
        """Test genetic mutation functionality."""
        mutated_seed = ema_seed.mutate(mutation_rate=0.5)
        
        assert isinstance(mutated_seed, EMACrossoverSeed)
        assert mutated_seed.genes.generation == ema_seed.genes.generation + 1
        
        # Some parameters should be different
        original_params = ema_seed.genes.parameters
        mutated_params = mutated_seed.genes.parameters
        
        differences = sum(1 for key in original_params 
                         if abs(original_params[key] - mutated_params[key]) > 0.001)
        assert differences > 0  # At least some parameters should change
    
    def test_crossover(self, ema_seed):
        """Test genetic crossover functionality."""
        # Create another seed with different parameters
        other_genes = SeedGenes(
            seed_id="other_ema",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'fast_ema_period': 8.0,
                'slow_ema_period': 21.0,
                'momentum_threshold': 0.02,
                'signal_strength': 0.6,
                'trend_filter': 0.01
            }
        )
        other_seed = EMACrossoverSeed(other_genes)
        
        child1, child2 = ema_seed.crossover(other_seed)
        
        assert isinstance(child1, EMACrossoverSeed)
        assert isinstance(child2, EMACrossoverSeed)
        assert child1.genes.generation > ema_seed.genes.generation
        assert child2.genes.generation > other_seed.genes.generation


class TestDonchianBreakoutSeed:
    """Test cases for Donchian Breakout seed."""
    
    @pytest.fixture
    def donchian_seed(self):
        """Create Donchian breakout seed for testing."""
        genes = SeedGenes(
            seed_id="test_donchian",
            seed_type=SeedType.BREAKOUT,
            parameters={
                'channel_period': 10.0,      # Minimum period for more signals
                'breakout_threshold': 0.001,  # Minimum threshold  
                'volume_confirmation': 1.0,   # Minimum volume requirement
                'false_breakout_filter': 2.0, # Minimum filtering (within bounds)
                'trend_bias': 0.3             # More permissive
            }
        )
        return DonchianBreakoutSeed(genes)
    
    @pytest.fixture
    def breakout_data(self):
        """Create data with clear breakout pattern."""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')  # Fixed frequency
        
        # Create deterministic data with clear breakout patterns (remove randomness)
        prices = []
        for i in range(100):
            if i < 30:
                # Tight sideways movement
                prices.append(100.0)
            elif i < 50:
                # Compression before breakout
                prices.append(100.0)  
            else:
                # Strong breakout movement with momentum
                breakout_strength = (i - 50) * 1.0  # Clear breakout
                prices.append(100.0 + breakout_strength)
        
        prices = pd.Series(prices, index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
    
    def test_donchian_seed_initialization(self, donchian_seed):
        """Test Donchian seed initializes correctly."""
        assert donchian_seed.seed_name == "Donchian_Breakout"
        assert donchian_seed.genes.seed_type == SeedType.BREAKOUT
        assert 'channel_period' in donchian_seed.genes.parameters
    
    def test_donchian_indicators(self, donchian_seed, breakout_data):
        """Test Donchian channel calculations."""
        indicators = donchian_seed.calculate_technical_indicators(breakout_data)
        
        assert 'donchian_high' in indicators
        assert 'donchian_low' in indicators
        assert 'donchian_mid' in indicators
        assert 'channel_width' in indicators
        
        # Check channel properties (exclude NaN values from comparison)
        valid_mask = ~(indicators['donchian_high'].isna() | indicators['donchian_low'].isna())
        assert (indicators['donchian_high'][valid_mask] >= indicators['donchian_low'][valid_mask]).all()
        assert (indicators['donchian_mid'][valid_mask] >= indicators['donchian_low'][valid_mask]).all()
        assert (indicators['donchian_mid'][valid_mask] <= indicators['donchian_high'][valid_mask]).all()
    
    def test_breakout_detection(self, donchian_seed, breakout_data):
        """Test breakout signal detection."""
        signals = donchian_seed.generate_signals(breakout_data)
        
        # Should detect some signals with the fixed algorithm
        non_zero_signals = (abs(signals) > 0.0).sum()
        
        assert non_zero_signals > 0  # Should detect some signals (algorithm working)


class TestRSIFilterSeed:
    """Test cases for RSI Filter seed."""
    
    @pytest.fixture
    def rsi_seed(self):
        """Create RSI filter seed for testing."""
        genes = SeedGenes(
            seed_id="test_rsi",
            seed_type=SeedType.MEAN_REVERSION,
            parameters={
                'rsi_period': 14.0,
                'oversold_threshold': 25.0,
                'overbought_threshold': 75.0,
                'operation_mode': 0.7,
                'divergence_weight': 0.3
            }
        )
        return RSIFilterSeed(genes)
    
    @pytest.fixture
    def oscillating_data(self):
        """Create oscillating price data for RSI testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        
        # Create oscillating prices
        x = np.linspace(0, 4*np.pi, 100)
        trend = 100 + 10 * np.sin(x)
        noise = np.random.normal(0, 0.5, 100)
        prices = pd.Series(trend + noise, index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
    
    def test_rsi_calculation(self, rsi_seed, oscillating_data):
        """Test RSI calculation."""
        rsi = rsi_seed.calculate_rsi(oscillating_data['close'], 14)
        
        # Check RSI properties
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(oscillating_data)
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()
        
        # Check for oscillation (should have oversold and overbought levels)
        assert rsi.min() < 40  # Should reach oversold
        assert rsi.max() > 60  # Should reach overbought
    
    def test_rsi_regime_detection(self, rsi_seed, oscillating_data):
        """Test RSI regime detection."""
        regime = rsi_seed.get_rsi_regime(oscillating_data)
        
        assert regime in ['oversold', 'overbought', 'bullish', 'bearish', 'neutral']
    
    def test_rsi_signal_generation(self, rsi_seed, oscillating_data):
        """Test RSI signal generation."""
        signals = rsi_seed.generate_signals(oscillating_data)
        
        # Check signal properties
        assert isinstance(signals, pd.Series)
        assert signals.min() >= -1.0
        assert signals.max() <= 1.0
        
        # Should generate some signals in oscillating market
        non_zero_signals = (abs(signals) > 0.1).sum()
        assert non_zero_signals > 0


class TestSeedIntegration:
    """Integration tests for multiple seeds working together."""
    
    @pytest.fixture
    def mixed_population(self):
        """Create population with different seed types."""
        seeds = []
        
        # EMA seed
        ema_genes = SeedGenes(
            seed_id="integration_ema",
            seed_type=SeedType.MOMENTUM,
            parameters={
                'fast_ema_period': 10.0,
                'slow_ema_period': 20.0,
                'momentum_threshold': 0.01,
                'signal_strength': 0.7,
                'trend_filter': 0.005
            }
        )
        seeds.append(EMACrossoverSeed(ema_genes))
        
        # Donchian seed
        donchian_genes = SeedGenes(
            seed_id="integration_donchian",
            seed_type=SeedType.BREAKOUT,
            parameters={
                'channel_period': 15.0,
                'breakout_threshold': 0.008,
                'volume_confirmation': 1.3,
                'false_breakout_filter': 3.0,
                'trend_bias': 0.6
            }
        )
        seeds.append(DonchianBreakoutSeed(donchian_genes))
        
        # RSI seed
        rsi_genes = SeedGenes(
            seed_id="integration_rsi",
            seed_type=SeedType.MEAN_REVERSION,
            parameters={
                'rsi_period': 12.0,
                'oversold_threshold': 30.0,
                'overbought_threshold': 70.0,
                'operation_mode': 0.8,
                'divergence_weight': 0.2
            }
        )
        seeds.append(RSIFilterSeed(rsi_genes))
        
        return seeds
    
    @pytest.fixture
    def complex_data(self):
        """Create complex market data with multiple patterns."""
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        
        # Combine trend, oscillation, and breakout patterns
        trend = np.linspace(100, 150, 200)
        oscillation = 5 * np.sin(np.linspace(0, 8*np.pi, 200))
        breakouts = np.where((np.arange(200) % 50) < 3, np.random.uniform(5, 15, 200), 0)
        noise = np.random.normal(0, 1, 200)
        
        prices = pd.Series(trend + oscillation + breakouts + noise, index=dates)
        
        return pd.DataFrame({
            'open': prices.shift(1).fillna(100),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)
    
    def test_all_seeds_generate_signals(self, mixed_population, complex_data):
        """Test all seed types generate valid signals."""
        for seed in mixed_population:
            signals = seed.generate_signals(complex_data)
            
            # Basic validation
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(complex_data)
            assert signals.min() >= -1.0
            assert signals.max() <= 1.0
            assert not signals.isna().any()
            
            # Should generate some signals
            non_zero_signals = (abs(signals) > 0.05).sum()
            assert non_zero_signals > 0, f"Seed {seed.seed_name} generated no signals"
    
    def test_seed_diversity(self, mixed_population, complex_data):
        """Test that different seeds generate diverse signals."""
        all_signals = []
        
        for seed in mixed_population:
            signals = seed.generate_signals(complex_data)
            all_signals.append(signals)
        
        # Calculate correlation between different seeds
        signal_df = pd.DataFrame({
            f'seed_{i}': signals for i, signals in enumerate(all_signals)
        })
        
        correlation_matrix = signal_df.corr()
        
        # Seeds should not be perfectly correlated
        for i in range(len(mixed_population)):
            for j in range(i + 1, len(mixed_population)):
                correlation = correlation_matrix.iloc[i, j]
                assert abs(correlation) < 0.95, f"Seeds {i} and {j} too highly correlated: {correlation}"
    
    def test_performance_across_seed_types(self, mixed_population, complex_data):
        """Test performance characteristics across different seed types."""
        performance_by_type = {}
        
        for seed in mixed_population:
            signals = seed.generate_signals(complex_data)
            
            # Simple performance metrics
            signal_strength = abs(signals).mean()
            signal_frequency = (abs(signals) > 0.1).sum() / len(signals)
            
            seed_type = seed.genes.seed_type.value
            if seed_type not in performance_by_type:
                performance_by_type[seed_type] = []
            
            performance_by_type[seed_type].append({
                'strength': signal_strength,
                'frequency': signal_frequency
            })
        
        # All seed types should have reasonable performance
        for seed_type, performances in performance_by_type.items():
            avg_strength = np.mean([p['strength'] for p in performances])
            avg_frequency = np.mean([p['frequency'] for p in performances])
            
            assert avg_strength > 0.01, f"Seed type {seed_type} has very weak signals"
            assert avg_frequency > 0.01, f"Seed type {seed_type} has very infrequent signals"


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])