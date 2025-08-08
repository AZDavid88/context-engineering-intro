"""
Phase 3 Market Regime Detection Integration Tests

Comprehensive validation of multi-source regime detection enhancement
following existing test patterns and validation methodologies.

Test Coverage:
- Individual regime detector validation
- Composite regime engine integration
- Universal regime enhancer functionality
- End-to-end system integration
- Performance and reliability testing
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock

from src.analysis.regime_detection_engine import CompositeRegimeDetectionEngine, RegimeAnalysis, CompositeRegime
from src.analysis.regime_detectors import (
    VolatilityRegimeDetector, 
    CorrelationRegimeDetector, 
    VolumeRegimeDetector
)
from src.strategy.genetic_seeds.universal_regime_enhancer import (
    UniversalRegimeEnhancer,
    enhance_seeds_with_regime_awareness,
    generate_regime_aware_signals_for_seed
)
from src.data.fear_greed_client import FearGreedClient, FearGreedData, MarketRegime
from src.analysis.correlation_engine import FilteredAssetCorrelationEngine
from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes, SeedType
from src.config.settings import get_settings


class TestableBaseSeed(BaseSeed):
    """Testable seed implementation for validation."""
    
    def __init__(self, genes: SeedGenes):
        super().__init__(genes)
        self.seed_type = SeedType.MOMENTUM
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple test signals."""
        return pd.Series([1.0, -1.0, 0.0, 1.0, -1.0], index=data.index[:5])
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators for testing."""
        return data.copy()
    
    @property
    def parameter_bounds(self) -> Dict[str, tuple]:
        """Parameter bounds for genetic evolution."""
        return {"test_param": (0.0, 2.0)}
    
    @property
    def required_parameters(self) -> List[str]:
        """Required parameters for this seed."""
        return ["test_param"]
    
    @property
    def seed_name(self) -> str:
        """Human-readable seed name."""
        return "TestableBaseSeed"
    
    @property 
    def seed_description(self) -> str:
        """Description of seed functionality."""
        return "Simple testable seed for validation"


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    np.random.seed(42)  # Reproducible data
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(100) * 1000,
        'high': 50500 + np.random.randn(100) * 1000,
        'low': 49500 + np.random.randn(100) * 1000,
        'close': 50000 + np.random.randn(100) * 1000,
        'volume': 1000 + np.random.randn(100) * 100
    })
    data.index = dates
    return data


@pytest.fixture
def mock_fear_greed_client():
    """Create mock fear/greed client."""
    mock_client = Mock(spec=FearGreedClient)
    mock_data = FearGreedData(
        value=30,  # Fear regime
        value_classification="Fear",
        timestamp=datetime.now(),
        regime=MarketRegime.FEAR,
        trading_signal=None,
        contrarian_strength=0.7
    )
    mock_client.get_current_index = AsyncMock(return_value=mock_data)
    return mock_client


@pytest.fixture
def mock_correlation_engine():
    """Create mock correlation engine."""
    mock_engine = Mock(spec=FilteredAssetCorrelationEngine)
    
    from src.analysis.correlation_engine import CorrelationMetrics
    mock_metrics = CorrelationMetrics(
        correlation_pairs={('BTC', 'ETH'): 0.8},
        portfolio_correlation_score=0.6,
        regime_classification="normal_correlation"
    )
    
    mock_engine.calculate_filtered_asset_correlations = AsyncMock(return_value=mock_metrics)
    mock_engine.health_check = AsyncMock(return_value={"status": "healthy"})
    return mock_engine


@pytest.fixture
def regime_detection_engine(mock_fear_greed_client, mock_correlation_engine):
    """Create regime detection engine with mocked dependencies."""
    return CompositeRegimeDetectionEngine(
        fear_greed_client=mock_fear_greed_client,
        correlation_engine=mock_correlation_engine
    )


class TestIndividualRegimeDetectors:
    """Test individual regime detector components."""
    
    @pytest.mark.asyncio
    async def test_volatility_regime_detector(self, sample_ohlcv_data):
        """Test volatility regime detection."""
        
        # Mock storage interface
        with patch('src.analysis.regime_detectors.volatility_regime_detector.get_storage_implementation') as mock_storage:
            mock_storage.return_value.get_ohlcv_bars = AsyncMock(return_value=sample_ohlcv_data)
            mock_storage.return_value.health_check = AsyncMock(return_value={"status": "healthy", "backend": "test"})
            
            detector = VolatilityRegimeDetector()
            
            # Test regime detection
            metrics = await detector.detect_volatility_regime(['BTC', 'ETH'])
            
            assert metrics.volatility_regime in ['low_volatility', 'medium_volatility', 'high_volatility']
            assert 0 <= metrics.current_volatility <= 2.0  # Reasonable volatility range
            assert 0 <= metrics.data_quality_score <= 1.0
            assert metrics.calculation_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_correlation_regime_detector(self, mock_correlation_engine):
        """Test correlation regime detection."""
        detector = CorrelationRegimeDetector(correlation_engine=mock_correlation_engine)
        
        metrics = await detector.detect_correlation_regime(['BTC', 'ETH'])
        
        assert metrics.correlation_regime in ['correlation_breakdown', 'normal_correlation', 'high_correlation']
        assert 0 <= metrics.average_correlation <= 1.0
        assert metrics.calculation_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_volume_regime_detector(self, sample_ohlcv_data):
        """Test volume regime detection."""
        
        # Mock storage interface
        with patch('src.analysis.regime_detectors.volume_regime_detector.get_storage_implementation') as mock_storage:
            mock_storage.return_value.get_ohlcv_bars = AsyncMock(return_value=sample_ohlcv_data)
            mock_storage.return_value.health_check = AsyncMock(return_value={"status": "healthy", "backend": "test"})
            
            detector = VolumeRegimeDetector()
            
            # Test regime detection
            metrics = await detector.detect_volume_regime(['BTC', 'ETH'])
            
            assert metrics.volume_regime in ['low_volume', 'normal_volume', 'high_volume']
            assert metrics.current_volume_ratio >= 0
            assert metrics.volume_trend in ['increasing', 'decreasing', 'stable']
            assert metrics.calculation_timestamp is not None


class TestCompositeRegimeEngine:
    """Test composite regime detection engine."""
    
    @pytest.mark.asyncio
    async def test_composite_regime_detection(self, regime_detection_engine):
        """Test composite regime detection integration."""
        
        # Mock individual detector responses
        with patch.multiple(
            regime_detection_engine,
            volatility_detector=Mock(),
            correlation_detector=Mock(),
            volume_detector=Mock()
        ):
            # Setup mock responses
            from src.analysis.regime_detectors.volatility_regime_detector import VolatilityMetrics
            from src.analysis.regime_detectors.correlation_regime_detector import CorrelationRegimeMetrics
            from src.analysis.regime_detectors.volume_regime_detector import VolumeRegimeMetrics
            
            regime_detection_engine.volatility_detector.detect_volatility_regime = AsyncMock(
                return_value=VolatilityMetrics(volatility_regime="medium_volatility", data_quality_score=0.9)
            )
            regime_detection_engine.correlation_detector.detect_correlation_regime = AsyncMock(
                return_value=CorrelationRegimeMetrics(correlation_regime="normal_correlation", data_quality_score=0.8)
            )
            regime_detection_engine.volume_detector.detect_volume_regime = AsyncMock(
                return_value=VolumeRegimeMetrics(volume_regime="normal_volume", data_quality_score=0.85)
            )
            
            # Test composite detection
            analysis = await regime_detection_engine.detect_composite_regime(['BTC', 'ETH'])
            
            assert isinstance(analysis, RegimeAnalysis)
            assert analysis.composite_regime in [CompositeRegime.RISK_ON, CompositeRegime.RISK_OFF, 
                                               CompositeRegime.NEUTRAL, CompositeRegime.TRANSITIONAL]
            assert 0 <= analysis.regime_confidence <= 1.0
            assert 0 <= analysis.regime_stability <= 1.0
            assert analysis.calculation_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_pressure_generation(self, regime_detection_engine):
        """Test genetic algorithm pressure generation."""
        
        # Create test regime analysis
        analysis = RegimeAnalysis(
            composite_regime=CompositeRegime.RISK_ON,
            regime_confidence=0.8,
            regime_stability=0.9
        )
        
        # Test pressure generation
        pressure = regime_detection_engine.generate_genetic_algorithm_pressure(analysis)
        
        assert isinstance(pressure, dict)
        assert 'contrarian_bias' in pressure
        assert 'momentum_bias' in pressure
        assert 'volatility_tolerance' in pressure
        assert 'position_sizing' in pressure
        assert 'holding_period' in pressure
        
        # All values should be between 0 and 1
        for value in pressure.values():
            assert 0 <= value <= 1.0
    
    @pytest.mark.asyncio
    async def test_health_check(self, regime_detection_engine):
        """Test composite engine health check."""
        health = await regime_detection_engine.health_check()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'component' in health
        assert 'timestamp' in health


class TestUniversalRegimeEnhancer:
    """Test universal regime enhancement wrapper."""
    
    def test_enhancer_initialization(self, regime_detection_engine):
        """Test enhancer initialization and configuration."""
        # Create test seed
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({'test_param': 1.0})
        base_seed = TestableBaseSeed(genes)
        
        # Create enhancer
        enhancer = UniversalRegimeEnhancer(base_seed, regime_detection_engine)
        
        assert enhancer.is_regime_aware
        assert enhancer.base_seed is base_seed
        assert enhancer.regime_engine is regime_detection_engine
        assert 'risk_on_signal_multiplier' in enhancer.config
    
    @pytest.mark.asyncio
    async def test_enhanced_signal_generation(self, regime_detection_engine, sample_ohlcv_data):
        """Test enhanced signal generation."""
        
        # Create test seed
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({'test_param': 1.0})
        base_seed = TestableBaseSeed(genes)
        enhancer = UniversalRegimeEnhancer(base_seed, regime_detection_engine)
        
        # Mock regime analysis
        mock_analysis = RegimeAnalysis(
            composite_regime=CompositeRegime.RISK_ON,
            regime_confidence=0.8,
            regime_stability=0.7
        )
        
        with patch.object(enhancer, '_get_regime_analysis', return_value=mock_analysis):
            # Generate enhanced signals
            enhanced_signals = await enhancer.generate_enhanced_signals(
                sample_ohlcv_data, ['BTC', 'ETH']
            )
            
            assert isinstance(enhanced_signals, pd.Series)
            assert len(enhanced_signals) == 5  # Test seed generates 5 signals
    
    def test_fitness_enhancement(self, regime_detection_engine):
        """Test fitness enhancement calculations."""
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({'test_param': 1.0})
        base_seed = TestableBaseSeed(genes)
        enhancer = UniversalRegimeEnhancer(base_seed, regime_detection_engine)
        
        # Test fitness enhancement
        base_fitness = 2.5
        mock_analysis = RegimeAnalysis(
            composite_regime=CompositeRegime.RISK_ON,
            regime_confidence=0.8,
            regime_stability=0.9
        )
        
        enhanced_fitness = enhancer.calculate_enhanced_fitness(base_fitness, mock_analysis)
        
        assert enhanced_fitness > base_fitness  # Should be enhanced
        assert enhanced_fitness > 0
    
    def test_bulk_enhancement(self, regime_detection_engine):
        """Test bulk seed enhancement."""
        # Create multiple test seeds
        seeds = []
        for i in range(3):
            genes = SeedGenes.create_default(SeedType.MOMENTUM)
            genes.parameters.update({'test_param': 1.0})
            seeds.append(TestableBaseSeed(genes))
        
        # Enhance all seeds
        enhanced_seeds = enhance_seeds_with_regime_awareness(seeds, regime_detection_engine)
        
        assert len(enhanced_seeds) == 3
        for enhanced_seed in enhanced_seeds:
            assert isinstance(enhanced_seed, UniversalRegimeEnhancer)
            assert enhanced_seed.is_regime_aware


class TestSystemIntegration:
    """Test end-to-end system integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_regime_aware_trading(self, regime_detection_engine, sample_ohlcv_data):
        """Test complete end-to-end regime-aware trading workflow."""
        
        # Create regime-aware seed
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({"test_param": 1.0})
        base_seed = TestableBaseSeed(genes)
        enhanced_seed = UniversalRegimeEnhancer(base_seed, regime_detection_engine)
        
        # Mock all regime detection components
        mock_analysis = RegimeAnalysis(
            sentiment_regime="fear",
            volatility_regime="medium_volatility",
            correlation_regime="normal_correlation",
            volume_regime="normal_volume",
            composite_regime=CompositeRegime.RISK_OFF,
            regime_confidence=0.75,
            regime_stability=0.8
        )
        
        with patch.object(regime_detection_engine, 'detect_composite_regime', return_value=mock_analysis):
            # Generate signals
            signals = await enhanced_seed.generate_enhanced_signals(
                sample_ohlcv_data, ['BTC', 'ETH']
            )
            
            # Verify signals were generated and enhanced
            assert isinstance(signals, pd.Series)
            assert len(signals) > 0
            
            # Test genetic pressure generation
            pressure = regime_detection_engine.generate_genetic_algorithm_pressure(mock_analysis)
            assert isinstance(pressure, dict)
            
            # Test fitness enhancement
            enhanced_fitness = enhanced_seed.calculate_enhanced_fitness(2.0, mock_analysis)
            assert enhanced_fitness >= 2.0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, regime_detection_engine, sample_ohlcv_data):
        """Test system performance under concurrent load."""
        
        # Create multiple enhanced seeds
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({"test_param": 1.0})
        seeds = [UniversalRegimeEnhancer(TestableBaseSeed(genes), regime_detection_engine) for _ in range(5)]
        
        # Mock regime analysis for performance
        mock_analysis = RegimeAnalysis(composite_regime=CompositeRegime.NEUTRAL)
        
        with patch.object(regime_detection_engine, 'detect_composite_regime', return_value=mock_analysis):
            # Run concurrent signal generation
            tasks = []
            for seed in seeds:
                task = seed.generate_enhanced_signals(sample_ohlcv_data, ['BTC', 'ETH'])
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Verify all tasks completed successfully
            assert len(results) == 5
            for result in results:
                assert isinstance(result, pd.Series)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, regime_detection_engine, sample_ohlcv_data):
        """Test error handling and fallback mechanisms."""
        
        # Create enhanced seed
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({"test_param": 1.0})
        base_seed = TestableBaseSeed(genes)
        enhanced_seed = UniversalRegimeEnhancer(base_seed, regime_detection_engine)
        
        # Test with regime detection failure
        with patch.object(regime_detection_engine, 'detect_composite_regime', 
                         side_effect=Exception("Regime detection failed")):
            
            # Should fall back to base signals without crashing
            signals = await enhanced_seed.generate_enhanced_signals(
                sample_ohlcv_data, ['BTC', 'ETH']
            )
            
            assert isinstance(signals, pd.Series)
            assert len(signals) > 0
    
    def test_backward_compatibility(self):
        """Test that enhanced seeds maintain backward compatibility."""
        
        # Create seed without regime engine
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({"test_param": 1.0})
        base_seed = TestableBaseSeed(genes)
        enhancer = UniversalRegimeEnhancer(base_seed, regime_engine=None)
        
        # Should not be regime aware
        assert not enhancer.is_regime_aware
        
        # Should delegate to base seed correctly
        assert enhancer.seed_type == SeedType.MOMENTUM
        assert hasattr(enhancer, 'generate_signals')


class TestConfigurationAndSettings:
    """Test configuration and settings integration."""
    
    def test_settings_integration(self):
        """Test integration with settings system."""
        settings = get_settings()
        
        # Test regime detection engine initialization with settings
        engine = CompositeRegimeDetectionEngine(settings=settings)
        
        assert engine.settings is settings
        assert hasattr(engine, 'regime_weights')
        assert hasattr(engine, 'confidence_threshold')
    
    def test_custom_enhancement_config(self, regime_detection_engine):
        """Test custom enhancement configuration."""
        
        custom_config = {
            'risk_on_signal_multiplier': 1.5,
            'risk_off_signal_multiplier': 0.6,
            'regime_confidence_threshold': 0.8
        }
        
        genes = SeedGenes.create_default(SeedType.MOMENTUM)
        genes.parameters.update({"test_param": 1.0})
        base_seed = TestableBaseSeed(genes)
        enhancer = UniversalRegimeEnhancer(
            base_seed, 
            regime_detection_engine,
            regime_enhancement_config=custom_config
        )
        
        assert enhancer.config['risk_on_signal_multiplier'] == 1.5
        assert enhancer.config['risk_off_signal_multiplier'] == 0.6
        assert enhancer.config['regime_confidence_threshold'] == 0.8


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])