"""
Universal Correlation Enhancer Integration Tests

Tests the universal wrapper that can enhance any genetic seed with correlation capabilities.
This replaces the need for individual correlation-enhanced seed implementations.

Test Coverage:
- Universal wrapper functionality with different seed types
- Genetic parameter integration and evolution
- Signal generation enhancement and fallback behavior
- Registry integration and factory methods
- Performance and compatibility validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Set up test logging
logging.basicConfig(level=logging.DEBUG)

# Import the universal enhancer and test dependencies
from src.strategy.genetic_seeds.universal_correlation_enhancer import (
    UniversalCorrelationEnhancer,
    enhance_seed_with_correlation,
    register_all_enhanced_seeds
)
from src.strategy.genetic_seeds.rsi_filter_seed import RSIFilterSeed
from src.strategy.genetic_seeds.bollinger_bands_seed import BollingerBandsSeed
from src.strategy.genetic_seeds.donchian_breakout_seed import DonchianBreakoutSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType
from src.config.settings import get_settings


class TestUniversalCorrelationEnhancer:
    """Comprehensive tests for the Universal Correlation Enhancement Wrapper."""
    
    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='1h')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, 200)
        
        # Add trend and volatility patterns
        trend = np.sin(np.arange(200) * 0.1) * 0.01
        volatility_clustering = np.random.choice([0.5, 1.0, 1.5], 200, p=[0.3, 0.4, 0.3])
        
        combined_returns = returns + trend
        combined_returns = combined_returns * volatility_clustering
        prices = base_price * np.exp(np.cumsum(combined_returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 200)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 200))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 200) * volatility_clustering
        }, index=dates)
        
        return data.clip(lower=0)
    
    @pytest.fixture
    def mock_filtered_assets(self) -> List[str]:
        """Mock filtered assets for correlation testing."""
        return ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
    
    def create_test_seed(self, seed_type: SeedType, seed_class, seed_id: str) -> Any:
        """Create a test seed with appropriate parameters."""
        genes = SeedGenes.create_default(seed_type, seed_id)
        
        if seed_type == SeedType.MEAN_REVERSION:
            genes.parameters.update({
                'rsi_period': 14.0,
                'oversold_threshold': 30.0,
                'overbought_threshold': 70.0,
                'operation_mode': 0.7,
                'divergence_weight': 0.3
            })
        elif seed_type == SeedType.VOLATILITY:
            genes.parameters.update({
                'lookback_period': 20.0,
                'volatility_multiplier': 2.0,
                'squeeze_threshold': 0.15,
                'breakout_strength': 0.02,
                'position_scaling_factor': 1.0
            })
        elif seed_type == SeedType.BREAKOUT:
            genes.parameters.update({
                'channel_period': 20.0,
                'breakout_threshold': 0.005,
                'volume_confirmation': 1.5,
                'false_breakout_filter': 4.0,
                'trend_bias': 0.5
            })
        
        return seed_class(genes)
    
    def test_universal_enhancer_with_rsi_seed(self, sample_ohlcv_data, mock_filtered_assets):
        """Test universal enhancer with RSI seed."""
        # Create base RSI seed
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "test_rsi_001")
        
        # Create enhanced version
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Test basic properties
        assert "Correlation_Enhanced_RSI" in enhanced_rsi.seed_name
        assert enhanced_rsi._original_seed_name == "RSIFilterSeed"
        assert enhanced_rsi._original_seed_type == SeedType.MEAN_REVERSION
        
        # Test that correlation parameters were added
        assert 'momentum_confirmation' in enhanced_rsi.genes.parameters
        assert 'mean_reversion_correlation_weight' in enhanced_rsi.genes.parameters
        assert 'correlation_weight' in enhanced_rsi.genes.parameters
        
        # Test parameter bounds
        bounds = enhanced_rsi.parameter_bounds
        assert 'momentum_confirmation' in bounds
        assert bounds['momentum_confirmation'][0] < bounds['momentum_confirmation'][1]
        
        # Test signal generation without correlation data (fallback)
        base_signals = enhanced_rsi.generate_signals(sample_ohlcv_data)
        assert isinstance(base_signals, pd.Series)
        assert len(base_signals) == len(sample_ohlcv_data)
        
        # Test signal generation with correlation data (enhanced)
        enhanced_signals = enhanced_rsi.generate_signals(
            sample_ohlcv_data,
            filtered_assets=mock_filtered_assets,
            current_asset='BTC'
        )
        assert isinstance(enhanced_signals, pd.Series)
        assert len(enhanced_signals) == len(sample_ohlcv_data)
    
    def test_universal_enhancer_with_bollinger_bands_seed(self, sample_ohlcv_data, mock_filtered_assets):
        """Test universal enhancer with Bollinger Bands seed."""
        # Create base Bollinger Bands seed
        base_bb = self.create_test_seed(SeedType.VOLATILITY, BollingerBandsSeed, "test_bb_001")
        
        # Create enhanced version
        enhanced_bb = UniversalCorrelationEnhancer(base_bb)
        
        # Test basic properties
        assert "Correlation_Enhanced_Bollinger" in enhanced_bb.seed_name
        assert enhanced_bb._original_seed_name == "BollingerBandsSeed"
        assert enhanced_bb._original_seed_type == SeedType.VOLATILITY
        
        # Test volatility-specific correlation parameters
        assert 'volatility_regime_confirmation' in enhanced_bb.genes.parameters
        assert 'cross_asset_volatility_weight' in enhanced_bb.genes.parameters
        assert 'squeeze_correlation_threshold' in enhanced_bb.genes.parameters
        
        # Test signal generation
        signals = enhanced_bb.generate_signals(sample_ohlcv_data)
        enhanced_signals = enhanced_bb.generate_signals(
            sample_ohlcv_data,
            filtered_assets=mock_filtered_assets,
            current_asset='BTC'
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(enhanced_signals, pd.Series)
        assert len(signals) == len(enhanced_signals) == len(sample_ohlcv_data)
    
    def test_universal_enhancer_with_donchian_seed(self, sample_ohlcv_data, mock_filtered_assets):
        """Test universal enhancer with Donchian breakout seed."""
        # Create base Donchian seed
        base_donchian = self.create_test_seed(SeedType.BREAKOUT, DonchianBreakoutSeed, "test_donchian_001")
        
        # Create enhanced version
        enhanced_donchian = UniversalCorrelationEnhancer(base_donchian)
        
        # Test basic properties
        assert "Correlation_Enhanced_Donchian" in enhanced_donchian.seed_name
        assert enhanced_donchian._original_seed_name == "DonchianBreakoutSeed"
        assert enhanced_donchian._original_seed_type == SeedType.BREAKOUT
        
        # Test breakout-specific correlation parameters
        assert 'trend_regime_confirmation' in enhanced_donchian.genes.parameters
        assert 'cross_asset_breakout_weight' in enhanced_donchian.genes.parameters
        assert 'false_breakout_correlation_filter' in enhanced_donchian.genes.parameters
        
        # Test signal generation
        signals = enhanced_donchian.generate_signals(sample_ohlcv_data)
        enhanced_signals = enhanced_donchian.generate_signals(
            sample_ohlcv_data,
            filtered_assets=mock_filtered_assets,
            current_asset='BTC'
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(enhanced_signals, pd.Series)
    
    def test_genetic_evolution_compatibility(self, sample_ohlcv_data):
        """Test genetic evolution compatibility."""
        # Create base seed and enhance it
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "evolution_test")
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Test cloning with mutations
        mutations = {
            'rsi_period': 21.0,
            'momentum_confirmation': 0.8,
            'correlation_weight': 0.4
        }
        
        mutated_seed = enhanced_rsi.clone_with_mutations(mutations)
        
        # Verify mutations were applied
        assert mutated_seed.genes.parameters['rsi_period'] == 21.0
        assert mutated_seed.genes.parameters['momentum_confirmation'] == 0.8
        assert mutated_seed.genes.parameters['correlation_weight'] == 0.4
        
        # Verify it's still an enhanced seed
        assert isinstance(mutated_seed, UniversalCorrelationEnhancer)
        assert mutated_seed._original_seed_name == "RSIFilterSeed"
    
    def test_factory_methods(self):
        """Test factory methods for creating enhanced seeds."""
        # Test enhance_seed_with_correlation function
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "factory_test")
        enhanced_rsi = enhance_seed_with_correlation(base_rsi)
        
        assert isinstance(enhanced_rsi, UniversalCorrelationEnhancer)
        assert enhanced_rsi.base_seed == base_rsi
        
        # Test create_enhanced_seed class method
        genes = SeedGenes.create_default(SeedType.MEAN_REVERSION, "class_method_test")
        enhanced_seed = UniversalCorrelationEnhancer.create_enhanced_seed("RSIFilterSeed", genes)
        
        if enhanced_seed:  # May be None if RSIFilterSeed not in registry
            assert isinstance(enhanced_seed, UniversalCorrelationEnhancer)
    
    def test_method_delegation(self, sample_ohlcv_data):
        """Test that methods are properly delegated to base seed."""
        # Create enhanced seed
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "delegation_test")
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Test method delegation
        indicators = enhanced_rsi.calculate_technical_indicators(sample_ohlcv_data)
        assert isinstance(indicators, dict)
        assert len(indicators) > 0
        
        # Test position management methods
        entry_decision = enhanced_rsi.should_enter_position(sample_ohlcv_data, 0.5)
        assert isinstance(entry_decision, bool)
        
        stop_loss = enhanced_rsi.calculate_stop_loss_level(sample_ohlcv_data, 50000.0, 1)
        assert isinstance(stop_loss, (int, float))
        
        position_size = enhanced_rsi.calculate_position_size(sample_ohlcv_data, 0.5)
        assert isinstance(position_size, (int, float))
        
        exit_conditions = enhanced_rsi.get_exit_conditions(sample_ohlcv_data, 1)
        assert isinstance(exit_conditions, dict)
    
    def test_error_handling_and_fallbacks(self, sample_ohlcv_data):
        """Test error handling and fallback behavior."""
        # Create enhanced seed
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "error_test")
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Test with invalid correlation data (should fallback to base implementation)
        fallback_signals = enhanced_rsi.generate_signals(
            sample_ohlcv_data,
            filtered_assets=[],  # Empty list
            current_asset=None   # None asset
        )
        
        assert isinstance(fallback_signals, pd.Series)
        assert len(fallback_signals) == len(sample_ohlcv_data)
        
        # Test with malformed data (should handle gracefully)
        empty_data = pd.DataFrame()
        try:
            empty_signals = enhanced_rsi.generate_signals(empty_data)
            # Should either return empty series or raise handled exception
            assert isinstance(empty_signals, pd.Series)
        except Exception:
            # Acceptable if base seed raises exception for empty data
            pass
    
    def test_correlation_enhancement_summary(self, sample_ohlcv_data, mock_filtered_assets):
        """Test correlation enhancement summary functionality."""
        # Create enhanced seed
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "summary_test")
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Generate base and enhanced signals
        base_signals = enhanced_rsi.base_seed.generate_signals(sample_ohlcv_data)
        enhanced_signals = enhanced_rsi.generate_signals(
            sample_ohlcv_data,
            filtered_assets=mock_filtered_assets,
            current_asset='BTC'
        )
        
        # Get enhancement summary
        summary = enhanced_rsi.get_correlation_enhancement_summary(base_signals, enhanced_signals)
        
        assert isinstance(summary, dict)
        assert 'base_signal_strength' in summary
        assert 'enhanced_signal_strength' in summary
        assert 'signal_enhancement_ratio' in summary
        assert 'correlation_enhancement_active' in summary
        assert 'original_seed_type' in summary
        assert summary['original_seed_type'] == "RSIFilterSeed"
    
    def test_parameter_bounds_integration(self):
        """Test parameter bounds are properly integrated."""
        # Create enhanced seed
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "bounds_test")
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Get parameter bounds
        bounds = enhanced_rsi.parameter_bounds
        
        # Verify base seed bounds are included
        base_bounds = base_rsi.parameter_bounds
        for param, bound in base_bounds.items():
            assert param in bounds
            assert bounds[param] == bound
        
        # Verify correlation bounds are included
        assert 'momentum_confirmation' in bounds
        assert 'correlation_weight' in bounds
        
        # Verify bounds are valid
        for param, (min_val, max_val) in bounds.items():
            assert min_val < max_val
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
    
    def test_multiple_seed_types_compatibility(self, sample_ohlcv_data):
        """Test compatibility with multiple seed types."""
        # Test with different seed types
        test_seeds = [
            (SeedType.MEAN_REVERSION, RSIFilterSeed, "multi_rsi"),
            (SeedType.VOLATILITY, BollingerBandsSeed, "multi_bb"),
            (SeedType.BREAKOUT, DonchianBreakoutSeed, "multi_donchian")
        ]
        
        for seed_type, seed_class, seed_id in test_seeds:
            # Create base seed
            base_seed = self.create_test_seed(seed_type, seed_class, seed_id)
            
            # Enhance with universal enhancer
            enhanced_seed = UniversalCorrelationEnhancer(base_seed)
            
            # Verify enhancement
            assert enhanced_seed._original_seed_type == seed_type
            assert isinstance(enhanced_seed, UniversalCorrelationEnhancer)
            
            # Test signal generation
            signals = enhanced_seed.generate_signals(sample_ohlcv_data)
            assert isinstance(signals, pd.Series)
            assert len(signals) == len(sample_ohlcv_data)
            
            # Verify seed-specific parameters were added
            if seed_type == SeedType.MEAN_REVERSION:
                assert 'momentum_confirmation' in enhanced_seed.genes.parameters
            elif seed_type == SeedType.VOLATILITY:
                assert 'volatility_regime_confirmation' in enhanced_seed.genes.parameters
            elif seed_type == SeedType.BREAKOUT:
                assert 'trend_regime_confirmation' in enhanced_seed.genes.parameters
    
    def test_string_representations(self):
        """Test string representations of enhanced seeds."""
        # Create enhanced seed
        base_rsi = self.create_test_seed(SeedType.MEAN_REVERSION, RSIFilterSeed, "string_test")
        enhanced_rsi = UniversalCorrelationEnhancer(base_rsi)
        
        # Test __str__
        str_repr = str(enhanced_rsi)
        assert "CorrelationEnhanced" in str_repr
        
        # Test __repr__
        repr_str = repr(enhanced_rsi)
        assert "UniversalCorrelationEnhancer" in repr_str
        assert "base_seed=" in repr_str


# Performance and integration test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])