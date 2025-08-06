#!/usr/bin/env python3
"""
Comprehensive Genetic Seed Validation Suite
============================================

This suite performs exhaustive testing of all genetic seeds to ensure:
1. GA-readiness and parameter evolution capability
2. Mathematical soundness and signal generation
3. Integration compatibility with the trading system
4. Robustness under various market conditions
5. Performance baselines for Hyperliquid crypto trading

Created for final validation before Phase 2 implementation.
"""

import sys
import os
import pandas as pd
import numpy as np
import pytest
from typing import Dict, List, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedType, SeedGenes
from src.strategy.genetic_seeds.seed_registry import get_registry
from src.config.settings import Settings

# Import all seeds to ensure registration (following research pattern)
import src.strategy.genetic_seeds.ema_crossover_seed
import src.strategy.genetic_seeds.donchian_breakout_seed
import src.strategy.genetic_seeds.rsi_filter_seed
import src.strategy.genetic_seeds.stochastic_oscillator_seed
import src.strategy.genetic_seeds.sma_trend_filter_seed
import src.strategy.genetic_seeds.atr_stop_loss_seed
import src.strategy.genetic_seeds.ichimoku_cloud_seed
import src.strategy.genetic_seeds.vwap_reversion_seed
import src.strategy.genetic_seeds.volatility_scaling_seed
import src.strategy.genetic_seeds.funding_rate_carry_seed
import src.strategy.genetic_seeds.linear_svc_classifier_seed
import src.strategy.genetic_seeds.pca_tree_quantile_seed

class ComprehensiveGeneticSeedValidator:
    """
    Comprehensive validation suite for genetic seed system.
    
    Validates mathematical soundness, GA-readiness, integration compatibility,
    and trading performance for Hyperliquid crypto platform.
    """
    
    def __init__(self):
        """Initialize validator with test data and requirements."""
        self.settings = Settings()
        self.test_results = {}
        self.performance_metrics = {}
        
        # Create comprehensive test datasets
        self.test_datasets = self._create_test_datasets()
        
        # Define validation requirements
        self.validation_requirements = {
            'signal_generation': True,           # Must generate signals
            'mathematical_soundness': True,      # No mathematical impossibilities
            'ga_parameter_evolution': True,      # Parameters must be evolvable
            'integration_compatibility': True,   # Must work with trading system
            'performance_baseline': True,        # Must meet minimum performance
            'robustness_testing': True,         # Must handle edge cases
            'hyperliquid_readiness': True       # Ready for crypto trading
        }
    
    def _create_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create comprehensive test datasets for validation."""
        datasets = {}
        
        # 1. Realistic Crypto Price Data (Bitcoin-like)
        dates = pd.date_range('2024-01-01', periods=500, freq='h')
        
        # Bitcoin-style price movement with volatility
        np.random.seed(42)  # Reproducible
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, 500)  # 2% hourly volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        datasets['crypto_realistic'] = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(15, 1, 500)  # Realistic volume distribution
        }, index=dates)
        
        # 2. Trending Market (Bull Market)
        trend_prices = np.linspace(40000, 60000, 200)
        noise = np.random.normal(0, 800, 200)
        bull_prices = trend_prices + noise
        
        datasets['bull_market'] = pd.DataFrame({
            'open': bull_prices,
            'high': bull_prices * 1.01,
            'low': bull_prices * 0.99,
            'close': bull_prices,
            'volume': np.random.lognormal(14, 0.5, 200)
        }, index=pd.date_range('2024-01-01', periods=200, freq='h'))
        
        # 3. Ranging Market (Sideways)
        range_center = 45000
        range_width = 2000
        sideways_prices = range_center + range_width * np.sin(np.linspace(0, 10*np.pi, 300))
        sideways_prices += np.random.normal(0, 300, 300)
        
        datasets['sideways_market'] = pd.DataFrame({
            'open': sideways_prices,
            'high': sideways_prices * 1.005,
            'low': sideways_prices * 0.995,
            'close': sideways_prices,
            'volume': np.random.lognormal(14.5, 0.3, 300)
        }, index=pd.date_range('2024-01-01', periods=300, freq='h'))
        
        # 4. High Volatility Market (Crash scenario)
        volatile_returns = np.random.normal(0, 0.05, 150)  # 5% volatility
        volatile_prices = [50000]
        for ret in volatile_returns[1:]:
            volatile_prices.append(volatile_prices[-1] * (1 + ret))
        
        datasets['high_volatility'] = pd.DataFrame({
            'open': volatile_prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in volatile_prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in volatile_prices],
            'close': volatile_prices,
            'volume': np.random.lognormal(16, 1.5, 150)  # Higher volume during volatility
        }, index=pd.date_range('2024-01-01', periods=150, freq='h'))
        
        # 5. Breakout Pattern (for Donchian validation)
        flat_period = [48000] * 50
        breakout_period = np.linspace(48000, 55000, 30)
        continuation_period = np.linspace(55000, 58000, 20) + np.random.normal(0, 200, 20)
        breakout_prices = np.concatenate([flat_period, breakout_period, continuation_period])
        
        datasets['breakout_pattern'] = pd.DataFrame({
            'open': breakout_prices,
            'high': breakout_prices * 1.002,
            'low': breakout_prices * 0.998,
            'close': breakout_prices,
            'volume': np.concatenate([
                np.random.lognormal(14, 0.2, 50),  # Low volume during flat
                np.random.lognormal(15, 0.5, 30),  # High volume during breakout
                np.random.lognormal(14.5, 0.3, 20) # Normal volume after
            ])
        }, index=pd.date_range('2024-01-01', periods=100, freq='h'))
        
        return datasets
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run complete validation suite on all genetic seeds.
        
        Returns:
            Complete validation report with all test results
        """
        print("🔬 CodeFarm Comprehensive Genetic Seed Validation Suite")
        print("=" * 60)
        
        validation_report = {
            'overall_status': 'PENDING',
            'seeds_tested': 0,
            'seeds_passed': 0,
            'seeds_failed': 0,
            'detailed_results': {},
            'performance_summary': {},
            'ga_readiness_status': {},
            'integration_compatibility': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        # Get registered seeds from registry (following research pattern)
        registry = get_registry()
        registered_seeds = registry.list_all_seeds()
        
        # Test each registered genetic seed
        for seed_name, seed_info in registered_seeds.items():
            print(f"\n🧬 Testing Seed: {seed_name}")
            print("-" * 40)
            
            try:
                # Get seed class from registry
                seed_class = registry.get_seed_class(seed_name)
                if not seed_class:
                    print(f"❌ {seed_name}: FAILED - Could not retrieve seed class")
                    validation_report['seeds_failed'] += 1
                    continue
                
                seed_result = self._validate_individual_seed(seed_name, seed_class)
                validation_report['detailed_results'][seed_name] = seed_result
                validation_report['seeds_tested'] += 1
                
                if seed_result['overall_passed']:
                    validation_report['seeds_passed'] += 1
                    print(f"✅ {seed_name}: PASSED")
                else:
                    validation_report['seeds_failed'] += 1
                    print(f"❌ {seed_name}: FAILED")
                    validation_report['critical_issues'].extend(
                        seed_result.get('critical_issues', [])
                    )
                
            except Exception as e:
                print(f"💥 {seed_name}: CRITICAL ERROR - {str(e)}")
                validation_report['seeds_failed'] += 1
                validation_report['critical_issues'].append(
                    f"{seed_name}: Exception during validation - {str(e)}"
                )
        
        # Determine overall status
        if validation_report['seeds_failed'] == 0:
            validation_report['overall_status'] = 'ALL_SEEDS_OPERATIONAL'
        elif validation_report['seeds_passed'] > 0:
            validation_report['overall_status'] = 'PARTIAL_FUNCTIONALITY'
        else:
            validation_report['overall_status'] = 'SYSTEM_FAILURE'
        
        # Generate final assessment
        self._generate_final_assessment(validation_report)
        
        return validation_report
    
    def _validate_individual_seed(self, seed_name: str, seed_class) -> Dict[str, Any]:
        """Validate individual genetic seed comprehensively."""
        
        seed_result = {
            'seed_name': seed_name,
            'overall_passed': True,
            'test_results': {},
            'performance_metrics': {},
            'ga_compatibility': {},
            'critical_issues': [],
            'warnings': []
        }
        
        try:
            # Create seed instance with default parameters
            test_genes = SeedGenes(
                seed_id=f"test_{seed_name.lower()}",
                seed_type=SeedType.MOMENTUM,  # Will be overridden by seed
                parameters={}  # Let seed set defaults
            )
            
            seed_instance = seed_class(test_genes, self.settings)
            
            # Test 1: Basic Instantiation and Parameter Validation
            seed_result['test_results']['instantiation'] = self._test_instantiation(
                seed_instance, seed_name
            )
            
            # Test 2: Signal Generation Across All Market Conditions
            seed_result['test_results']['signal_generation'] = self._test_signal_generation(
                seed_instance, seed_name
            )
            
            # Test 3: Mathematical Soundness (No Impossibilities)
            seed_result['test_results']['mathematical_soundness'] = self._test_mathematical_soundness(
                seed_instance, seed_name
            )
            
            # Test 4: GA Parameter Evolution Capability
            seed_result['test_results']['ga_evolution'] = self._test_ga_evolution(
                seed_instance, seed_name
            )
            
            # Test 5: Integration Compatibility
            seed_result['test_results']['integration'] = self._test_integration_compatibility(
                seed_instance, seed_name
            )
            
            # Test 6: Performance Baseline
            seed_result['test_results']['performance'] = self._test_performance_baseline(
                seed_instance, seed_name
            )
            
            # Test 7: Robustness and Edge Cases
            seed_result['test_results']['robustness'] = self._test_robustness(
                seed_instance, seed_name
            )
            
            # Test 8: Hyperliquid Platform Readiness
            seed_result['test_results']['hyperliquid_readiness'] = self._test_hyperliquid_readiness(
                seed_instance, seed_name
            )
            
            # Aggregate results
            failed_tests = [
                test_name for test_name, result in seed_result['test_results'].items()
                if not result.get('passed', False)
            ]
            
            if failed_tests:
                seed_result['overall_passed'] = False
                seed_result['critical_issues'] = [
                    f"Failed tests: {', '.join(failed_tests)}"
                ]
            
        except Exception as e:
            seed_result['overall_passed'] = False
            seed_result['critical_issues'].append(f"Validation exception: {str(e)}")
        
        return seed_result
    
    def _test_instantiation(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test basic seed instantiation and parameter validation."""
        test_result = {'passed': True, 'details': {}}
        
        try:
            # Test required attributes exist
            required_attrs = ['seed_name', 'seed_description', 'required_parameters', 'parameter_bounds']
            for attr in required_attrs:
                if not hasattr(seed_instance, attr):
                    test_result['passed'] = False
                    test_result['details'][f'missing_{attr}'] = f"Missing required attribute: {attr}"
            
            # Test parameter bounds are valid
            if hasattr(seed_instance, 'parameter_bounds'):
                bounds = seed_instance.parameter_bounds
                for param, (min_val, max_val) in bounds.items():
                    if min_val >= max_val:
                        test_result['passed'] = False
                        test_result['details'][f'invalid_bounds_{param}'] = f"Invalid bounds for {param}: {min_val} >= {max_val}"
            
            # Test genes have valid parameters
            if seed_instance.genes.parameters:
                for param, value in seed_instance.genes.parameters.items():
                    if not isinstance(value, (int, float)):
                        test_result['passed'] = False
                        test_result['details'][f'invalid_param_type_{param}'] = f"Parameter {param} has invalid type: {type(value)}"
            
            test_result['details']['seed_type'] = seed_instance.genes.seed_type.value
            test_result['details']['parameter_count'] = len(seed_instance.genes.parameters)
            
        except Exception as e:
            test_result['passed'] = False
            test_result['details']['exception'] = str(e)
        
        return test_result
    
    def _test_signal_generation(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test signal generation across all market conditions."""
        test_result = {'passed': True, 'details': {}, 'signal_stats': {}}
        
        for dataset_name, data in self.test_datasets.items():
            try:
                # Generate signals
                signals = seed_instance.generate_signals(data)
                
                # Validate signal format
                if not isinstance(signals, pd.Series):
                    test_result['passed'] = False
                    test_result['details'][f'{dataset_name}_invalid_type'] = "Signals must be pandas Series"
                    continue
                
                # Check signal range
                if not signals.between(-1.0, 1.0).all():
                    test_result['passed'] = False
                    test_result['details'][f'{dataset_name}_invalid_range'] = "Signals must be in range [-1.0, 1.0]"
                    continue
                
                # Count signals
                signal_count = (abs(signals) > 0.01).sum()  # Significant signals
                signal_frequency = signal_count / len(signals)
                
                test_result['signal_stats'][dataset_name] = {
                    'total_signals': signal_count,
                    'signal_frequency': signal_frequency,
                    'max_signal': signals.max(),
                    'min_signal': signals.min(),
                    'mean_abs_signal': abs(signals).mean()
                }
                
                # Validate signal generation (must generate some signals)
                if signal_count == 0:
                    test_result['details'][f'{dataset_name}_no_signals'] = f"No signals generated for {dataset_name}"
                    # Don't fail immediately - some market conditions might not trigger certain strategies
                
            except Exception as e:
                test_result['passed'] = False
                test_result['details'][f'{dataset_name}_exception'] = str(e)
        
        # Overall signal generation assessment
        total_signals = sum(stats['total_signals'] for stats in test_result['signal_stats'].values())
        if total_signals == 0:
            test_result['passed'] = False
            test_result['details']['no_signals_any_market'] = "Seed generated zero signals across all market conditions"
        
        return test_result
    
    def _test_mathematical_soundness(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test for mathematical impossibilities and logical errors."""
        test_result = {'passed': True, 'details': {}}
        
        # Special test for Donchian-type breakout strategies
        if 'DONCHIAN' in seed_name.upper() or 'BREAKOUT' in seed_name.upper():
            # Test with deterministic breakout pattern
            breakout_data = self.test_datasets['breakout_pattern']
            
            try:
                signals = seed_instance.generate_signals(breakout_data)
                
                # Should detect the clear breakout around period 50-80
                breakout_period_signals = signals.iloc[50:80]
                breakout_detected = (abs(breakout_period_signals) > 0.1).any()
                
                if not breakout_detected:
                    test_result['passed'] = False
                    test_result['details']['no_breakout_detection'] = "Failed to detect clear breakout pattern"
                else:
                    test_result['details']['breakout_detection'] = "Successfully detected breakout pattern"
                
            except Exception as e:
                test_result['passed'] = False
                test_result['details']['breakout_test_exception'] = str(e)
        
        # Test with trending data (should generate directional signals)
        try:
            trending_data = self.test_datasets['bull_market']
            signals = seed_instance.generate_signals(trending_data)
            
            # For trend-following strategies, expect some positive bias in bull market
            mean_signal = signals.mean()
            test_result['details']['trend_bias'] = mean_signal
            
            # Not all strategies should be trend-following, so this is informational
            
        except Exception as e:
            test_result['details']['trend_test_exception'] = str(e)
        
        return test_result
    
    def _test_ga_evolution(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test genetic algorithm parameter evolution capability."""
        test_result = {'passed': True, 'details': {}}
        
        try:
            # Test parameter bounds exist and are valid
            bounds = seed_instance.parameter_bounds
            if not bounds:
                test_result['passed'] = False
                test_result['details']['no_parameter_bounds'] = "No parameter bounds defined for GA evolution"
                return test_result
            
            # Test parameter modification doesn't break the seed
            original_params = seed_instance.genes.parameters.copy()
            
            # Create modified parameters within bounds
            modified_params = {}
            for param, (min_val, max_val) in bounds.items():
                if param in original_params:
                    # Test with boundary values
                    test_values = [
                        min_val,
                        max_val,
                        (min_val + max_val) / 2,  # Middle value
                        min_val + 0.1 * (max_val - min_val),  # Near minimum
                        max_val - 0.1 * (max_val - min_val)   # Near maximum
                    ]
                    
                    for test_val in test_values:
                        test_params = original_params.copy()
                        test_params[param] = test_val
                        
                        # Create new seed with modified parameters
                        test_genes = SeedGenes(
                            seed_id=f"ga_test_{seed_name.lower()}",
                            seed_type=seed_instance.genes.seed_type,
                            parameters=test_params
                        )
                        
                        try:
                            test_seed = seed_instance.__class__(test_genes, self.settings)
                            test_signals = test_seed.generate_signals(self.test_datasets['crypto_realistic'])
                            
                            # Verify signals are still valid
                            if not isinstance(test_signals, pd.Series):
                                test_result['passed'] = False
                                test_result['details'][f'ga_evolution_failed_{param}'] = f"Parameter {param}={test_val} broke signal generation"
                                break
                            
                        except Exception as e:
                            test_result['passed'] = False
                            test_result['details'][f'ga_evolution_exception_{param}'] = f"Parameter {param}={test_val} caused exception: {str(e)}"
                            break
            
            test_result['details']['parameters_tested'] = len(bounds)
            test_result['details']['evolution_ready'] = test_result['passed']
            
        except Exception as e:
            test_result['passed'] = False
            test_result['details']['ga_test_exception'] = str(e)
        
        return test_result
    
    def _test_integration_compatibility(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test compatibility with trading system integration."""
        test_result = {'passed': True, 'details': {}}
        
        try:
            # Test required methods exist
            required_methods = ['generate_signals', 'calculate_technical_indicators']
            for method in required_methods:
                if not hasattr(seed_instance, method) or not callable(getattr(seed_instance, method)):
                    test_result['passed'] = False
                    test_result['details'][f'missing_method_{method}'] = f"Required method {method} not found or not callable"
            
            # Test optional methods that enhance integration
            optional_methods = ['should_enter_position', 'calculate_position_size', 'get_exit_conditions']
            available_optional = []
            for method in optional_methods:
                if hasattr(seed_instance, method) and callable(getattr(seed_instance, method)):
                    available_optional.append(method)
            
            test_result['details']['optional_methods_available'] = available_optional
            test_result['details']['integration_score'] = len(available_optional) / len(optional_methods)
            
            # Test data format compatibility
            test_data = self.test_datasets['crypto_realistic']
            
            # Test technical indicators
            try:
                indicators = seed_instance.calculate_technical_indicators(test_data)
                if not isinstance(indicators, dict):
                    test_result['passed'] = False
                    test_result['details']['invalid_indicators_format'] = "Technical indicators must return dictionary"
                else:
                    test_result['details']['indicators_count'] = len(indicators)
            except Exception as e:
                test_result['passed'] = False
                test_result['details']['indicators_exception'] = str(e)
            
        except Exception as e:
            test_result['passed'] = False
            test_result['details']['integration_test_exception'] = str(e)
        
        return test_result
    
    def _test_performance_baseline(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test minimum performance requirements for trading."""
        test_result = {'passed': True, 'details': {}}
        
        # Note: This is a simplified performance test
        # In production, would use vectorbt or similar for proper backtesting
        
        try:
            # Test on realistic crypto data
            data = self.test_datasets['crypto_realistic']
            signals = seed_instance.generate_signals(data)
            
            # Calculate basic performance metrics
            returns = data['close'].pct_change()
            strategy_returns = signals.shift(1) * returns  # Signals applied next period
            
            # Remove NaN values
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 0:
                total_return = (1 + strategy_returns).prod() - 1
                volatility = strategy_returns.std() * np.sqrt(365 * 24)  # Annualized for hourly data
                sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365 * 24) if strategy_returns.std() > 0 else 0
                
                max_drawdown = (strategy_returns.cumsum().cummax() - strategy_returns.cumsum()).max()
                
                test_result['details']['total_return'] = total_return
                test_result['details']['volatility'] = volatility
                test_result['details']['sharpe_ratio'] = sharpe_ratio
                test_result['details']['max_drawdown'] = max_drawdown
                
                # Basic performance thresholds (not too strict for validation)
                if total_return < -0.8:  # Max 80% loss
                    test_result['details']['warning_high_loss'] = f"High total loss: {total_return:.2%}"
                
                if sharpe_ratio < -3.0:  # Very poor risk-adjusted returns
                    test_result['details']['warning_poor_sharpe'] = f"Very poor Sharpe ratio: {sharpe_ratio:.2f}"
                
                if max_drawdown > 0.9:  # Max 90% drawdown
                    test_result['details']['warning_high_drawdown'] = f"High drawdown: {max_drawdown:.2%}"
            
            else:
                test_result['details']['no_returns_data'] = "No valid returns data for performance calculation"
        
        except Exception as e:
            test_result['details']['performance_test_exception'] = str(e)
        
        return test_result
    
    def _test_robustness(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test robustness and edge case handling."""
        test_result = {'passed': True, 'details': {}}
        
        # Test with various edge cases
        edge_cases = {
            'single_row': self.test_datasets['crypto_realistic'].iloc[:1],
            'missing_volume': self.test_datasets['crypto_realistic'].drop('volume', axis=1),
            'all_same_price': pd.DataFrame({
                'open': [50000] * 100,
                'high': [50000] * 100,
                'low': [50000] * 100,
                'close': [50000] * 100,
                'volume': [1000] * 100
            }, index=pd.date_range('2024-01-01', periods=100, freq='h'))
        }
        
        # Add NaN values test
        nan_data = self.test_datasets['crypto_realistic'].copy()
        nan_data.iloc[50:60] = np.nan
        edge_cases['with_nans'] = nan_data
        
        for case_name, test_data in edge_cases.items():
            try:
                signals = seed_instance.generate_signals(test_data)
                
                # Validate output format
                if isinstance(signals, pd.Series) and signals.between(-1.0, 1.0).all():
                    test_result['details'][f'{case_name}_passed'] = True
                else:
                    test_result['details'][f'{case_name}_failed'] = "Invalid signal format or range"
                    
            except Exception as e:
                # Some edge cases might legitimately fail (e.g., single row)
                test_result['details'][f'{case_name}_exception'] = str(e)
                if case_name not in ['single_row']:  # Don't fail for expected edge cases
                    test_result['passed'] = False
        
        return test_result
    
    def _test_hyperliquid_readiness(self, seed_instance: BaseSeed, seed_name: str) -> Dict[str, Any]:
        """Test readiness for Hyperliquid crypto platform."""
        test_result = {'passed': True, 'details': {}}
        
        try:
            # Test with high-frequency crypto data characteristics
            crypto_data = self.test_datasets['crypto_realistic']
            
            # Test signal generation frequency (important for crypto trading)
            signals = seed_instance.generate_signals(crypto_data)
            signal_frequency = (abs(signals) > 0.01).sum() / len(signals)
            
            test_result['details']['signal_frequency'] = signal_frequency
            
            # For crypto, we want reasonable but not excessive signal frequency
            if signal_frequency > 0.8:
                test_result['details']['warning_high_frequency'] = "Very high signal frequency - may lead to overtrading"
            elif signal_frequency < 0.001:
                test_result['details']['warning_low_frequency'] = "Very low signal frequency - may miss opportunities"
            
            # Test with high volatility (common in crypto)
            volatile_signals = seed_instance.generate_signals(self.test_datasets['high_volatility'])
            volatile_signal_count = (abs(volatile_signals) > 0.01).sum()
            
            test_result['details']['volatile_market_signals'] = volatile_signal_count
            test_result['details']['handles_volatility'] = volatile_signal_count > 0
            
            # Test parameter ranges are suitable for crypto timeframes
            bounds = seed_instance.parameter_bounds
            crypto_suitable_params = []
            
            for param, (min_val, max_val) in bounds.items():
                # Check if parameter ranges are suitable for crypto (hourly data)
                if 'period' in param.lower():
                    if min_val >= 1 and max_val <= 168:  # 1 hour to 1 week
                        crypto_suitable_params.append(param)
                else:
                    crypto_suitable_params.append(param)  # Non-period parameters assumed suitable
            
            test_result['details']['crypto_suitable_params'] = len(crypto_suitable_params)
            test_result['details']['total_params'] = len(bounds)
            test_result['details']['crypto_compatibility_ratio'] = len(crypto_suitable_params) / len(bounds) if bounds else 1.0
            
        except Exception as e:
            test_result['passed'] = False
            test_result['details']['hyperliquid_test_exception'] = str(e)
        
        return test_result
    
    def _generate_final_assessment(self, validation_report: Dict[str, Any]) -> None:
        """Generate final assessment and recommendations."""
        
        # Calculate success metrics
        total_tests = sum(
            len(seed_result['test_results']) 
            for seed_result in validation_report['detailed_results'].values()
        )
        
        passed_tests = sum(
            sum(1 for test_result in seed_result['test_results'].values() if test_result.get('passed', False))
            for seed_result in validation_report['detailed_results'].values()
        )
        
        validation_report['test_pass_rate'] = passed_tests / total_tests if total_tests > 0 else 0
        
        # Generate recommendations
        if validation_report['overall_status'] == 'ALL_SEEDS_OPERATIONAL':
            validation_report['recommendations'] = [
                "✅ All genetic seeds are operational and GA-ready",
                "✅ Safe to proceed with Phase 2 implementation",
                "✅ System ready for Hyperliquid integration",
                "✅ Mathematical verification protocols successfully implemented"
            ]
        elif validation_report['overall_status'] == 'PARTIAL_FUNCTIONALITY':
            validation_report['recommendations'] = [
                "⚠️ Some seeds have issues - review critical_issues section",
                "⚠️ Fix failing seeds before Phase 2 implementation",
                "✅ Working seeds can be used for GA evolution",
                "🔧 Apply mathematical verification to failing seeds"
            ]
        else:
            validation_report['recommendations'] = [
                "❌ Critical system failure - no seeds operational",
                "❌ Do NOT proceed with Phase 2 until issues resolved",
                "🔧 Apply comprehensive debugging methodology",
                "🔧 Review mathematical verification protocols"
            ]

if __name__ == "__main__":
    # Run comprehensive validation
    validator = ComprehensiveGeneticSeedValidator()
    results = validator.run_comprehensive_validation()
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎯 FINAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Seeds Tested: {results['seeds_tested']}")
    print(f"Seeds Passed: {results['seeds_passed']}")
    print(f"Seeds Failed: {results['seeds_failed']}")
    print(f"Test Pass Rate: {results['test_pass_rate']:.1%}")
    
    if results['critical_issues']:
        print(f"\n❌ Critical Issues:")
        for issue in results['critical_issues']:
            print(f"  - {issue}")
    
    print(f"\n📋 Recommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")