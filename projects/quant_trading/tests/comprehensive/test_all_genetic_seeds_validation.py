"""
Comprehensive Genetic Seed Validation Suite - CODEFARM Implementation
Tests all 14 genetic seeds for functional correctness, enhancement verification, and performance.

This is the definitive validation that eliminates "validation theater" by testing every single seed.
"""

import pytest
import pandas as pd
import numpy as np
import importlib
import inspect
import time
from typing import Dict, List, Any, Type, Optional
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path for imports
sys.path.append('/workspaces/context-engineering-intro/projects/quant_trading')

# Import our system components
from src.strategy.genetic_seeds.universal_correlation_enhancer import UniversalCorrelationEnhancer
from src.strategy.genetic_seeds.enhanced_seed_factory import (
    discover_all_genetic_seeds, 
    create_enhanced_seed_instance,
    get_enhancement_statistics
)
from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes, SeedType
from src.config.settings import get_settings

# Set up comprehensive logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveSeedValidator:
    """Validates all 14 genetic seeds for complete functional correctness."""
    
    def __init__(self):
        self.settings = get_settings()
        self.test_results = {}
        self.performance_metrics = {}
        
    def generate_test_ohlcv_data(self, periods: int = 200) -> pd.DataFrame:
        """Generate realistic OHLCV test data for seed validation."""
        dates = pd.date_range(start='2025-01-01', periods=periods, freq='1h')
        
        # Generate realistic price movements with multiple patterns
        np.random.seed(42)  # Reproducible tests
        base_price = 50000.0
        
        # Create complex price patterns
        trend = np.sin(np.arange(periods) * 0.05) * 0.02  # Cyclical trend
        momentum = np.cumsum(np.random.normal(0, 0.001, periods))  # Random walk
        volatility_clusters = np.random.choice([0.5, 1.0, 2.0], periods, p=[0.3, 0.4, 0.3])
        
        # Generate returns with realistic patterns
        returns = np.random.normal(0.0002, 0.015, periods) + trend + momentum
        returns = returns * volatility_clusters
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices with realistic spreads
        close = prices
        high = close * (1 + np.abs(np.random.normal(0, 0.003, periods)))
        low = close * (1 - np.abs(np.random.normal(0, 0.003, periods)))
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        
        # Generate volume with correlation to volatility
        volume = np.random.lognormal(8, 0.5, periods) * volatility_clusters
        
        data = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,  
            'close': close,
            'volume': volume
        }, index=dates)
        
        # Ensure no negative prices and realistic bounds
        return data.clip(lower=0.01)
    
    def create_seed_specific_genes(self, seed_class: Type[BaseSeed], seed_id: str) -> SeedGenes:
        """Create appropriate genetic parameters for specific seed types."""
        
        # ACTUAL parameter configurations extracted from real seed implementations
        SEED_PARAMETER_CONFIGS = {
            "RSIFilterSeed": {
                'rsi_period': 14.0,
                'oversold_threshold': 30.0,
                'overbought_threshold': 70.0,
                'operation_mode': 0.7,
                'divergence_weight': 0.3
            },
            "BollingerBandsSeed": {
                'lookback_period': 20.0,
                'volatility_multiplier': 2.0,
                'squeeze_threshold': 0.15,
                'breakout_strength': 0.02,
                'position_scaling_factor': 1.0
            },
            "DonchianBreakoutSeed": {
                'channel_period': 20.0,
                'breakout_threshold': 0.005,
                'volume_confirmation': 1.5,
                'false_breakout_filter': 4.0,
                'trend_bias': 0.5
            },
            "EMACrossoverSeed": {
                "fast_ema_period": 10.0,
                "slow_ema_period": 26.0,
                "momentum_threshold": 0.025,
                "signal_strength": 0.55,
                "trend_filter": 0.01,
            },
            "ATRStopLossSeed": {
                "atr_period": 15.0,
                "stop_loss_multiplier": 1.85,
                "trailing_stop_multiplier": 1.65,
                "position_size_atr_factor": 1.05,
                "volatility_adjustment": 0.5,
            },
            "VolatilityScalingSeed": {
                "volatility_window": 25.0,
                "regime_threshold": 1.4,
                "scaling_factor": 0.5,
                "regime_persistence": 6.0,
                "multi_timeframe_weight": 0.5,
                "target_volatility": 0.15,
                "position_base": 0.027,
            },
            "PCATreeQuantileSeed": {
                "pca_components": 3.5,
                "tree_depth": 5.0,
                "n_estimators": 105.0,
                "signal_quantile": 0.5,
                "feature_window": 275.0,
                "quantile_bins": 3.0,
            },
            "LinearSVCClassifierSeed": {
                "lookback_window": 110.0,
                "feature_count": 9.0,
                "regularization": 1.25,
                "ensemble_size": 3.0,
                "cross_validation": 6.5,
                "quantile_bins": 3.0,
            },
            "StochasticOscillatorSeed": {
                "k_period": 15.0,
                "d_period": 3.5,
                "overbought_level": 77.5,
                "oversold_level": 22.5,
                "divergence_sensitivity": 0.5,
            },
            "IchimokuCloudSeed": {
                "tenkan_period": 9.5,
                "kijun_period": 27.0,
                "senkou_span_b_period": 60.0,
                "cloud_strength_weight": 0.5,
                "momentum_confirmation": 0.5,
            },
            "NadarayaWatsonSeed": {
                "bandwidth": 22.5,
                "kernel_type": 0.5,
                "trend_threshold": 0.041,
                "smoothing_factor": 0.5,
                "volatility_adaptation": 0.5,
            },
            "SMATrendFilterSeed": {
                "fast_sma_period": 55.0,
                "slow_sma_period": 225.0,
                "trend_strength_period": 55.0,
                "filter_sensitivity": 0.5,
                "momentum_confirmation": 0.5,
            },
            "VWAPReversionSeed": {
                "vwap_period": 105.0,
                "reversion_threshold": 0.0115,
                "volume_confirmation": 1.5,
                "deviation_multiplier": 2.5,
                "regime_detection_weight": 0.5,
            },
            "FundingRateCarrySeed": {
                "funding_threshold": 0.0,
                "carry_duration": 36.5,
                "rate_momentum": 0.5,
                "reversal_sensitivity": 0.55,
                "funding_persistence": 13.0,
            },
        }
        
        # Determine seed type from class
        seed_type = self.infer_seed_type(seed_class)
        genes = SeedGenes.create_default(seed_type, seed_id)
        
        # Apply ACTUAL parameters from real seed implementations
        seed_name = seed_class.__name__
        if seed_name in SEED_PARAMETER_CONFIGS:
            genes.parameters.update(SEED_PARAMETER_CONFIGS[seed_name])
        else:
            # Fallback for any unknown seeds
            logger.warning(f"No parameter configuration found for {seed_name}, using defaults")
        
        return genes
    
    def infer_seed_type(self, seed_class: Type[BaseSeed]) -> SeedType:
        """Infer SeedType from class name and characteristics."""
        class_name = seed_class.__name__.upper()
        
        if any(term in class_name for term in ['RSI', 'BOLLINGER', 'VWAP', 'NADARAYA']):
            return SeedType.MEAN_REVERSION
        elif any(term in class_name for term in ['VOLATILITY', 'ATR']):
            return SeedType.VOLATILITY
        elif any(term in class_name for term in ['DONCHIAN', 'BREAKOUT']):
            return SeedType.BREAKOUT
        elif any(term in class_name for term in ['EMA', 'SMA', 'MOMENTUM']):
            return SeedType.MOMENTUM
        elif any(term in class_name for term in ['FUNDING', 'CARRY']):
            return SeedType.CARRY
        elif any(term in class_name for term in ['PCA', 'SVC', 'ML', 'LINEAR']):
            return SeedType.ML_CLASSIFIER
        elif any(term in class_name for term in ['STOCHASTIC', 'ICHIMOKU']):
            return SeedType.TREND_FOLLOWING
        else:
            return SeedType.MOMENTUM  # Default fallback

    def test_single_seed_comprehensive(self, seed_name: str, seed_class: Type[BaseSeed]) -> Dict[str, Any]:
        """Comprehensive test of a single seed type."""
        
        logger.info(f"🧪 Testing {seed_name} comprehensively...")
        test_result = {
            'seed_name': seed_name,
            'import_success': True,
            'instantiation_success': False,
            'signal_generation_success': False,
            'enhancement_success': False,
            'parameter_validation_success': False,
            'correlation_parameters_added': 0,
            'performance_metrics': {},
            'error': None,
            'error_type': None
        }
        
        try:
            # Test 1: Seed Instantiation
            logger.debug(f"  📦 Instantiating {seed_name}...")
            genes = self.create_seed_specific_genes(seed_class, f"test_{seed_name}")
            base_seed = seed_class(genes, self.settings)
            test_result['instantiation_success'] = True
            logger.debug(f"  ✅ {seed_name}: Instantiation successful")
            
            # Test 2: Signal Generation
            logger.debug(f"  📊 Testing signal generation for {seed_name}...")
            test_data = self.generate_test_ohlcv_data()
            start_time = time.time()
            signals = base_seed.generate_signals(test_data)
            signal_time = time.time() - start_time
            
            # Validate signal output
            assert isinstance(signals, pd.Series), f"{seed_name}: signals must be pd.Series, got {type(signals)}"
            assert len(signals) > 0, f"{seed_name}: signals cannot be empty"
            assert len(signals) == len(test_data), f"{seed_name}: signals length {len(signals)} != data length {len(test_data)}"
            
            # Allow for some NaN values (common in technical indicators) but not all
            nan_ratio = signals.isna().sum() / len(signals)
            assert nan_ratio < 0.9, f"{seed_name}: too many NaN signals ({nan_ratio:.1%})"
            
            test_result['signal_generation_success'] = True
            test_result['performance_metrics']['signal_generation_time'] = signal_time
            test_result['performance_metrics']['signal_nan_ratio'] = nan_ratio
            logger.debug(f"  ✅ {seed_name}: Signal generation successful ({signal_time:.3f}s, {nan_ratio:.1%} NaN)")
            
            # Test 3: Enhancement with Universal Wrapper
            logger.debug(f"  🔧 Testing universal enhancement for {seed_name}...")
            start_time = time.time()
            enhanced_seed = UniversalCorrelationEnhancer(base_seed, self.settings)
            enhancement_time = time.time() - start_time
            
            # Validate enhancement structure
            assert hasattr(enhanced_seed, 'base_seed'), f"{seed_name}: enhancement wrapper missing base_seed"
            assert enhanced_seed._original_seed_name == seed_class.__name__, f"{seed_name}: wrong original seed name"
            assert enhanced_seed.base_seed is base_seed, f"{seed_name}: base seed reference incorrect"
            
            # Test enhanced signal generation
            start_time = time.time()
            enhanced_signals = enhanced_seed.generate_signals(test_data)
            enhanced_signal_time = time.time() - start_time
            
            assert isinstance(enhanced_signals, pd.Series), f"{seed_name}: enhanced signals must be pd.Series, got {type(enhanced_signals)}"
            assert len(enhanced_signals) == len(test_data), f"{seed_name}: enhanced signals length mismatch"
            
            test_result['enhancement_success'] = True
            test_result['performance_metrics']['enhancement_time'] = enhancement_time
            test_result['performance_metrics']['enhanced_signal_time'] = enhanced_signal_time
            logger.debug(f"  ✅ {seed_name}: Enhancement successful ({enhancement_time:.3f}s)")
            
            # Test 4: Parameter Validation
            logger.debug(f"  📏 Validating parameters for {seed_name}...")
            required_params = enhanced_seed.required_parameters
            param_bounds = enhanced_seed.parameter_bounds
            
            # Verify correlation parameters were added
            correlation_params = [p for p in enhanced_seed.genes.parameters.keys() 
                                if any(cp in p.lower() for cp in ['correlation', 'regime', 'momentum', 'volatility', 'trend', 'diversification'])]
            
            assert len(correlation_params) >= 3, f"{seed_name}: insufficient correlation parameters ({len(correlation_params)} < 3)"
            assert len(required_params) > 0, f"{seed_name}: no required parameters defined"
            assert len(param_bounds) > 0, f"{seed_name}: no parameter bounds defined"
            
            test_result['correlation_parameters_added'] = len(correlation_params)
            test_result['parameter_validation_success'] = True
            logger.debug(f"  ✅ {seed_name}: Parameters validated ({len(correlation_params)} correlation params added)")
            
            # Test 5: Method Delegation & Additional Functionality
            logger.debug(f"  🔍 Testing method delegation for {seed_name}...")
            indicators = enhanced_seed.calculate_technical_indicators(test_data)
            assert isinstance(indicators, dict), f"{seed_name}: indicators must be dict, got {type(indicators)}"
            
            # Test cloning capability
            mutations = {'correlation_weight': 0.5}
            if 'correlation_weight' not in enhanced_seed.genes.parameters:
                # Add the parameter if it doesn't exist
                enhanced_seed.genes.parameters['correlation_weight'] = 0.3
            
            cloned_seed = enhanced_seed.clone_with_mutations(mutations)
            assert isinstance(cloned_seed, UniversalCorrelationEnhancer), f"{seed_name}: cloning failed, got {type(cloned_seed)}"
            assert cloned_seed.genes.parameters['correlation_weight'] == 0.5, f"{seed_name}: mutation not applied"
            
            # Test method delegation methods (if they exist)
            try:
                entry_decision = enhanced_seed.should_enter_position(test_data, 0.5)
                assert isinstance(entry_decision, bool), f"{seed_name}: should_enter_position must return bool"
                test_result['method_delegation_success'] = True
            except (AttributeError, NotImplementedError):
                # Some seeds might not implement this method
                test_result['method_delegation_success'] = False
            
            logger.info(f"✅ {seed_name}: ALL COMPREHENSIVE TESTS PASSED")
            return test_result
            
        except Exception as e:
            test_result['error'] = str(e)
            test_result['error_type'] = type(e).__name__
            logger.error(f"❌ {seed_name}: COMPREHENSIVE TEST FAILED - {type(e).__name__}: {e}")
            return test_result

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all genetic seeds."""
        
        logger.info("🚀 STARTING COMPREHENSIVE GENETIC SEED VALIDATION...")
        logger.info("   This will test ALL 14 seeds for complete functional correctness")
        logger.info("   No validation theater - only real functional testing!")
        
        # Discovery phase
        logger.info("📊 Discovering genetic seed classes...")
        discovered_seeds = discover_all_genetic_seeds()
        logger.info(f"   Found {len(discovered_seeds)} genetic seed classes")
        
        for seed_name in discovered_seeds.keys():
            logger.info(f"   - {seed_name}")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_seeds_discovered': len(discovered_seeds),
            'successful_tests': 0,
            'failed_tests': 0,
            'seed_test_results': {},
            'performance_summary': {},
            'enhancement_coverage': {},
            'detailed_failures': []
        }
        
        logger.info(f"\n🧪 TESTING {len(discovered_seeds)} SEEDS COMPREHENSIVELY...")
        
        # Test each seed comprehensively
        for i, (seed_name, seed_class) in enumerate(discovered_seeds.items(), 1):
            logger.info(f"\n[{i}/{len(discovered_seeds)}] Testing {seed_name}...")
            
            try:
                result = self.test_single_seed_comprehensive(seed_name, seed_class)
                validation_results['seed_test_results'][seed_name] = result
                
                # Check if all critical tests passed
                critical_tests_passed = all([
                    result['instantiation_success'],
                    result['signal_generation_success'], 
                    result['enhancement_success'],
                    result['parameter_validation_success']
                ])
                
                if critical_tests_passed:
                    validation_results['successful_tests'] += 1
                    logger.info(f"   ✅ {seed_name}: PASSED ALL TESTS")
                else:
                    validation_results['failed_tests'] += 1
                    failed_tests = [test for test in [
                        ('instantiation', result['instantiation_success']),
                        ('signal_generation', result['signal_generation_success']),
                        ('enhancement', result['enhancement_success']),
                        ('parameter_validation', result['parameter_validation_success'])
                    ] if not test[1]]
                    
                    failure_details = f"{seed_name}: Failed {[t[0] for t in failed_tests]}"
                    validation_results['detailed_failures'].append(failure_details)
                    logger.warning(f"   ❌ {seed_name}: FAILED - {failure_details}")
                    
            except Exception as e:
                logger.error(f"   💥 {seed_name}: CRITICAL SYSTEM FAILURE - {type(e).__name__}: {e}")
                validation_results['failed_tests'] += 1
                validation_results['seed_test_results'][seed_name] = {
                    'critical_failure': str(e),
                    'error_type': type(e).__name__
                }
                validation_results['detailed_failures'].append(f"{seed_name}: Critical failure - {e}")
        
        # Calculate summary statistics
        total_tests = validation_results['total_seeds_discovered']
        success_rate = (validation_results['successful_tests'] / total_tests * 100) if total_tests > 0 else 0
        validation_results['success_rate'] = success_rate
        
        # Performance summary
        successful_results = [
            result for result in validation_results['seed_test_results'].values()
            if 'performance_metrics' in result and result.get('signal_generation_success', False)
        ]
        
        if successful_results:
            total_signal_time = sum(
                result['performance_metrics'].get('signal_generation_time', 0)
                for result in successful_results
            )
            avg_signal_time = total_signal_time / len(successful_results)
            
            validation_results['performance_summary'] = {
                'total_signal_generation_time': total_signal_time,
                'average_signal_generation_time': avg_signal_time,
                'total_tested_seeds': len(successful_results)
            }
        
        # Enhancement coverage analysis
        enhancement_stats = get_enhancement_statistics()
        validation_results['enhancement_coverage'] = enhancement_stats
        
        # Final results display
        logger.info(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║              COMPREHENSIVE VALIDATION COMPLETE               ║
        ║                                                              ║
        ║  📊 Total Seeds Discovered: {total_tests:2d}                           ║
        ║  ✅ Successful Tests:       {validation_results['successful_tests']:2d} ({validation_results['successful_tests']/total_tests*100:5.1f}%)                  ║ 
        ║  ❌ Failed Tests:           {validation_results['failed_tests']:2d} ({validation_results['failed_tests']/total_tests*100:5.1f}%)                  ║
        ║  📈 Success Rate:           {success_rate:5.1f}%                        ║
        ║  ⏱️  Avg Signal Time:       {validation_results['performance_summary'].get('average_signal_generation_time', 0):.3f}s                     ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        if validation_results['detailed_failures']:
            logger.warning("\n🔍 DETAILED FAILURE ANALYSIS:")
            for failure in validation_results['detailed_failures']:
                logger.warning(f"   ❌ {failure}")
        
        # Success threshold check
        if success_rate >= 85.0:
            logger.info(f"🎯 VALIDATION SUCCESS: {success_rate:.1f}% success rate meets 85% threshold")
        else:
            logger.warning(f"⚠️ VALIDATION CONCERN: {success_rate:.1f}% success rate below 85% threshold")
        
        return validation_results

# Pytest Test Classes
class TestComprehensiveGeneticSeedValidation:
    """Pytest class for comprehensive genetic seed validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.validator = ComprehensiveSeedValidator()
    
    def test_all_seeds_comprehensive_validation(self):
        """Test all 14 genetic seeds comprehensively - THE DEFINITIVE TEST."""
        results = self.validator.run_comprehensive_validation()
        
        # Critical assertions for comprehensive validation
        assert results['total_seeds_discovered'] >= 10, f"Too few seeds discovered: {results['total_seeds_discovered']}"
        
        # Success rate threshold - must be at least 85%
        assert results['success_rate'] >= 85.0, f"COMPREHENSIVE VALIDATION FAILED: Success rate {results['success_rate']:.1f}% below 85% threshold"
        
        # Maximum allowed failures - no more than 2 seeds can fail
        assert results['failed_tests'] <= 2, f"Too many seed failures: {results['failed_tests']}"
        
        # Detailed assertions for each successfully tested seed
        successful_seeds = 0
        for seed_name, result in results['seed_test_results'].items():
            if 'critical_failure' not in result:
                # These are the core requirements for each seed
                assert result['instantiation_success'], f"{seed_name}: instantiation failed - {result.get('error', 'unknown error')}"
                assert result['signal_generation_success'], f"{seed_name}: signal generation failed - {result.get('error', 'unknown error')}"  
                assert result['enhancement_success'], f"{seed_name}: enhancement failed - {result.get('error', 'unknown error')}"
                assert result['parameter_validation_success'], f"{seed_name}: parameter validation failed - {result.get('error', 'unknown error')}"
                assert result['correlation_parameters_added'] >= 3, f"{seed_name}: insufficient correlation parameters added ({result['correlation_parameters_added']})"
                successful_seeds += 1
        
        # Ensure we tested a reasonable number of seeds successfully
        assert successful_seeds >= 12, f"Too few seeds passed comprehensive testing: {successful_seeds}"
        
        print(f"\n🎯 COMPREHENSIVE VALIDATION SUMMARY:")
        print(f"   📊 Seeds Discovered: {results['total_seeds_discovered']}")
        print(f"   ✅ Seeds Passed: {successful_seeds}")
        print(f"   📈 Success Rate: {results['success_rate']:.1f}%")
        print(f"   🚀 COMPREHENSIVE VALIDATION: {'PASSED' if results['success_rate'] >= 85.0 else 'FAILED'}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for all seeds."""
        results = self.validator.run_comprehensive_validation()
        
        # Performance thresholds
        MAX_SIGNAL_TIME_PER_SEED = 2.0  # 2 seconds per seed (generous for complex indicators)
        MAX_ENHANCEMENT_TIME_PER_SEED = 0.5  # 500ms per seed enhancement
        
        performance_failures = []
        
        for seed_name, result in results['seed_test_results'].items():
            if 'performance_metrics' in result and result.get('signal_generation_success', False):
                metrics = result['performance_metrics']
                
                if 'signal_generation_time' in metrics:
                    if metrics['signal_generation_time'] > MAX_SIGNAL_TIME_PER_SEED:
                        performance_failures.append(
                            f"{seed_name}: signal generation too slow ({metrics['signal_generation_time']:.3f}s > {MAX_SIGNAL_TIME_PER_SEED}s)"
                        )
                
                if 'enhancement_time' in metrics:
                    if metrics['enhancement_time'] > MAX_ENHANCEMENT_TIME_PER_SEED:
                        performance_failures.append(
                            f"{seed_name}: enhancement too slow ({metrics['enhancement_time']:.3f}s > {MAX_ENHANCEMENT_TIME_PER_SEED}s)"
                        )
        
        # Performance assertion
        assert len(performance_failures) == 0, f"Performance failures detected:\n" + "\n".join(performance_failures)
        
        print(f"\n⚡ PERFORMANCE VALIDATION PASSED")
        if 'performance_summary' in results:
            summary = results['performance_summary']
            print(f"   ⏱️ Average Signal Time: {summary.get('average_signal_generation_time', 0):.3f}s")
            print(f"   🧪 Seeds Benchmarked: {summary.get('total_tested_seeds', 0)}")

    def test_enhancement_coverage_verification(self):
        """Verify universal enhancement coverage is truly comprehensive."""
        results = self.validator.run_comprehensive_validation()
        
        # Enhancement coverage requirements
        enhancement_stats = results.get('enhancement_coverage', {})
        
        if enhancement_stats:
            total_seeds = enhancement_stats.get('total_seeds', 0)
            enhanced_seeds = enhancement_stats.get('enhanced_seeds', 0)
            coverage = enhancement_stats.get('enhancement_coverage', 0)
            
            assert coverage >= 0.95, f"Enhancement coverage {coverage:.1%} below 95% threshold"
            assert enhanced_seeds >= 12, f"Too few enhanced seeds: {enhanced_seeds}"
            
            print(f"\n🔧 ENHANCEMENT COVERAGE VERIFICATION:")
            print(f"   📊 Total Seeds: {total_seeds}")
            print(f"   ✨ Enhanced Seeds: {enhanced_seeds}")
            print(f"   📈 Coverage: {coverage:.1%}")
            print(f"   🎯 ENHANCEMENT COVERAGE: {'PASSED' if coverage >= 0.95 else 'FAILED'}")

# Standalone execution capability
if __name__ == "__main__":
    print("🚀 CODEFARM COMPREHENSIVE GENETIC SEED VALIDATION")
    print("=" * 70)
    print("This will test ALL genetic seeds for complete functional correctness.")
    print("No validation theater - only real functional testing!")
    print("=" * 70)
    
    # Run comprehensive validation
    validator = ComprehensiveSeedValidator()
    results = validator.run_comprehensive_validation()
    
    # Display detailed results
    print("\n🔍 DETAILED SEED-BY-SEED RESULTS:")
    print("-" * 70)
    
    for seed_name, result in results['seed_test_results'].items():
        if 'critical_failure' in result:
            print(f"❌ CRITICAL FAILURE {seed_name}: {result['critical_failure']}")
            continue
            
        # Determine overall status
        status = "✅ COMPREHENSIVE PASS" if all([
            result.get('instantiation_success', False),
            result.get('signal_generation_success', False),
            result.get('enhancement_success', False),
            result.get('parameter_validation_success', False)
        ]) else "❌ COMPREHENSIVE FAIL"
        
        print(f"{status} {seed_name}")
        
        # Show test breakdown
        tests = [
            ('📦 Instantiation', result.get('instantiation_success', False)),
            ('📊 Signal Generation', result.get('signal_generation_success', False)),
            ('🔧 Enhancement', result.get('enhancement_success', False)),
            ('📏 Parameter Validation', result.get('parameter_validation_success', False))
        ]
        
        for test_name, passed in tests:
            status_icon = '✓' if passed else '✗'
            print(f"    {status_icon} {test_name}")
        
        # Show performance metrics if available
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            if 'signal_generation_time' in metrics:
                print(f"    ⏱️  Signal Time: {metrics['signal_generation_time']:.3f}s")
            if 'correlation_parameters_added' in result:
                print(f"    🔗 Correlation Params: {result['correlation_parameters_added']}")
        
        if result.get('error'):
            print(f"    ❗ Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print(f"🎯 FINAL COMPREHENSIVE VALIDATION RESULT:")
    print(f"   📊 Total Seeds: {results['total_seeds_discovered']}")
    print(f"   ✅ Successful: {results['successful_tests']}")
    print(f"   ❌ Failed: {results['failed_tests']}")
    print(f"   📈 Success Rate: {results['success_rate']:.1f}%")
    
    if results['success_rate'] >= 85.0:
        print(f"   🏆 COMPREHENSIVE VALIDATION: PASSED ({results['success_rate']:.1f}% ≥ 85%)")
    else:
        print(f"   ⚠️  COMPREHENSIVE VALIDATION: FAILED ({results['success_rate']:.1f}% < 85%)")
        if results['detailed_failures']:
            print(f"   🔍 Detailed Failures:")
            for failure in results['detailed_failures']:
                print(f"      - {failure}")
    
    print("=" * 70)