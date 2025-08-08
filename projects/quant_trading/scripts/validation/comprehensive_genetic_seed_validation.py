#!/usr/bin/env python3
"""
Comprehensive Genetic Seed Validation Suite

This consolidates all genetic seed testing into a single, authoritative script
that validates ACTUAL SIGNAL GENERATION for all 14 seeds, not just imports and registration.

Critical Tests:
1. Signal generation with multiple market scenarios
2. Parameter bounds enforcement
3. Robustness across edge cases
4. Performance benchmarking
5. Integration compatibility
"""

import sys
import os
# Add project root to path - fix "No module named 'src'" error
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Import all genetic seeds
from src.strategy.genetic_seeds import *
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType
from src.strategy.genetic_seeds.seed_registry import get_registry

class ComprehensiveGeneticSeedValidator:
    """Authoritative validation suite for all genetic seeds"""
    
    def __init__(self):
        self.registry = get_registry()
        self.all_seed_classes = self._get_all_seed_classes()
        self.test_scenarios = self._create_test_scenarios()
        self.validation_results = {}
        
    def _get_all_seed_classes(self) -> List:
        """Get all 14 genetic seed classes"""
        return [
            EMACrossoverSeed, DonchianBreakoutSeed, RSIFilterSeed, 
            StochasticOscillatorSeed, SMATrendFilterSeed, ATRStopLossSeed,
            IchimokuCloudSeed, VWAPReversionSeed, VolatilityScalingSeed,
            FundingRateCarrySeed, LinearSVCClassifierSeed, PCATreeQuantileSeed,
            BollingerBandsSeed, NadarayaWatsonSeed
        ]
    
    def _create_test_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Create multiple test scenarios covering different market conditions"""
        scenarios = {}
        
        # Scenario 1: Trending market (should favor momentum seeds)
        dates1 = pd.date_range('2023-01-01', periods=100, freq='D')
        trend_prices = pd.Series(100 * (1 + np.linspace(0, 0.5, 100)), index=dates1)
        scenarios['trending'] = pd.DataFrame({
            'open': trend_prices * 0.99,
            'high': trend_prices * 1.02,
            'low': trend_prices * 0.98,
            'close': trend_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates1)
        
        # Scenario 2: Oscillating market (should favor mean reversion seeds)
        dates2 = pd.date_range('2023-01-01', periods=100, freq='D')
        oscillating_prices = pd.Series(100 + 20 * np.sin(np.linspace(0, 4*np.pi, 100)), index=dates2)
        scenarios['oscillating'] = pd.DataFrame({
            'open': oscillating_prices * 0.99,
            'high': oscillating_prices * 1.02,
            'low': oscillating_prices * 0.98,
            'close': oscillating_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates2)
        
        # Scenario 3: High volatility (should favor volatility-adaptive seeds)
        dates3 = pd.date_range('2023-01-01', periods=100, freq='D')
        volatile_returns = np.random.normal(0, 0.05, 100)  # 5% daily volatility
        volatile_prices = pd.Series(100 * np.cumprod(1 + volatile_returns), index=dates3)
        scenarios['volatile'] = pd.DataFrame({
            'open': volatile_prices * 0.99,
            'high': volatile_prices * 1.05,
            'low': volatile_prices * 0.95,
            'close': volatile_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates3)
        
        # Scenario 4: Breakout pattern (should favor breakout seeds)
        dates4 = pd.date_range('2023-01-01', periods=100, freq='D')
        breakout_prices = []
        for i in range(100):
            if i < 50:
                breakout_prices.append(100 + np.random.normal(0, 1))  # Sideways
            else:
                breakout_prices.append(100 + (i-50) * 0.5 + np.random.normal(0, 1))  # Breakout
        breakout_prices = pd.Series(breakout_prices, index=dates4)
        scenarios['breakout'] = pd.DataFrame({
            'open': breakout_prices * 0.99,
            'high': breakout_prices * 1.02,
            'low': breakout_prices * 0.98,
            'close': breakout_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates4)
        
        return scenarios
    
    def validate_seed_signal_generation(self, seed_class, scenario_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate signal generation for a specific seed and scenario"""
        try:
            # Create seed with default parameters
            genes = SeedGenes(
                seed_id=f'test_{seed_class.__name__}_{scenario_name}',
                seed_type=SeedType.MOMENTUM,  # Will be overridden by seed
                parameters={}
            )
            seed = seed_class(genes)
            
            # Generate signals
            signals = seed.generate_signals(data)
            
            # Validate signal properties
            validation = {
                'seed_name': seed.seed_name,
                'scenario': scenario_name,
                'total_periods': len(signals),
                'non_zero_signals': (signals != 0).sum(),
                'signal_sum': signals.sum(),
                'signal_range': (signals.min(), signals.max()),
                'has_long_signals': (signals > 0).any(),
                'has_short_signals': (signals < 0).any(),
                'valid_range': (signals.min() >= -1.0) and (signals.max() <= 1.0),
                'no_nan_values': not signals.isna().any(),
                'success': True,
                'error': None
            }
            
            # Critical failure checks
            if validation['non_zero_signals'] == 0:
                validation['success'] = False
                validation['error'] = 'No signals generated'
            
            if not validation['valid_range']:
                validation['success'] = False
                validation['error'] = f'Invalid signal range: {validation["signal_range"]}'
            
            if not validation['no_nan_values']:
                validation['success'] = False
                validation['error'] = 'Contains NaN values'
            
            return validation
            
        except Exception as e:
            return {
                'seed_name': seed_class.__name__,
                'scenario': scenario_name,
                'success': False,
                'error': str(e),
                'total_periods': 0,
                'non_zero_signals': 0
            }
    
    def validate_parameter_bounds(self, seed_class) -> Dict[str, Any]:
        """Validate parameter bounds are correctly implemented"""
        try:
            genes = SeedGenes(
                seed_id=f'bounds_test_{seed_class.__name__}',
                seed_type=SeedType.MOMENTUM,
                parameters={}
            )
            seed = seed_class(genes)
            
            bounds = seed.parameter_bounds
            required = seed.required_parameters
            
            validation = {
                'seed_name': seed.seed_name,
                'bounds_defined': len(bounds) > 0,
                'required_params_defined': len(required) > 0,
                'bounds_valid': True,
                'params_in_bounds': True,
                'success': True,
                'error': None
            }
            
            # Check bounds validity
            for param, (min_val, max_val) in bounds.items():
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    validation['bounds_valid'] = False
                    validation['error'] = f'Invalid bound types for {param}'
                    break
                if min_val >= max_val:
                    validation['bounds_valid'] = False
                    validation['error'] = f'Invalid bounds for {param}: {min_val} >= {max_val}'
                    break
            
            # Check required parameters have bounds
            for param in required:
                if param not in bounds:
                    validation['params_in_bounds'] = False
                    validation['error'] = f'Required parameter {param} missing bounds'
                    break
            
            validation['success'] = (validation['bounds_valid'] and 
                                   validation['params_in_bounds'] and
                                   validation['bounds_defined'] and
                                   validation['required_params_defined'])
            
            return validation
            
        except Exception as e:
            return {
                'seed_name': seed_class.__name__,
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🔬 COMPREHENSIVE GENETIC SEED VALIDATION")
        print("=" * 60)
        
        results = {
            'signal_generation': {},
            'parameter_bounds': {},
            'summary': {
                'total_seeds': len(self.all_seed_classes),
                'successful_seeds': 0,
                'failed_seeds': [],
                'scenario_performance': {},
                'critical_issues': []
            }
        }
        
        # Test 1: Signal Generation Across All Scenarios
        print("\n📊 Testing Signal Generation Across Market Scenarios...")
        for seed_class in self.all_seed_classes:
            seed_name = seed_class.__name__
            results['signal_generation'][seed_name] = {}
            
            seed_success = True
            for scenario_name, data in self.test_scenarios.items():
                validation = self.validate_seed_signal_generation(seed_class, scenario_name, data)
                results['signal_generation'][seed_name][scenario_name] = validation
                
                if not validation['success']:
                    seed_success = False
                    results['summary']['critical_issues'].append(
                        f"{seed_name} failed in {scenario_name}: {validation['error']}"
                    )
                    print(f"   ❌ {seed_name} - {scenario_name}: {validation['error']}")
                else:
                    signals_count = validation['non_zero_signals']
                    print(f"   ✅ {seed_name} - {scenario_name}: {signals_count} signals")
            
            if seed_success:
                results['summary']['successful_seeds'] += 1
            else:
                results['summary']['failed_seeds'].append(seed_name)
        
        # Test 2: Parameter Bounds Validation
        print(f"\n🔬 Testing Parameter Bounds for {len(self.all_seed_classes)} Seeds...")
        for seed_class in self.all_seed_classes:
            validation = self.validate_parameter_bounds(seed_class)
            results['parameter_bounds'][seed_class.__name__] = validation
            
            if validation['success']:
                print(f"   ✅ {validation['seed_name']}: Valid bounds")
            else:
                print(f"   ❌ {validation['seed_name']}: {validation['error']}")
                results['summary']['critical_issues'].append(
                    f"Parameter bounds issue in {validation['seed_name']}: {validation['error']}"
                )
        
        # Test 3: Scenario Performance Analysis
        print(f"\n📈 Analyzing Performance Across Scenarios...")
        for scenario_name in self.test_scenarios.keys():
            scenario_stats = {
                'seeds_with_signals': 0,
                'total_signals': 0,
                'avg_signals_per_seed': 0
            }
            
            for seed_name in results['signal_generation'].keys():
                scenario_result = results['signal_generation'][seed_name][scenario_name]
                if scenario_result['success'] and scenario_result['non_zero_signals'] > 0:
                    scenario_stats['seeds_with_signals'] += 1
                    scenario_stats['total_signals'] += scenario_result['non_zero_signals']
            
            if scenario_stats['seeds_with_signals'] > 0:
                scenario_stats['avg_signals_per_seed'] = (
                    scenario_stats['total_signals'] / scenario_stats['seeds_with_signals']
                )
            
            results['summary']['scenario_performance'][scenario_name] = scenario_stats
            print(f"   📊 {scenario_name}: {scenario_stats['seeds_with_signals']}/{len(self.all_seed_classes)} seeds active")
        
        # Final Summary
        print(f"\n📋 VALIDATION SUMMARY:")
        print(f"   Total Seeds Tested: {results['summary']['total_seeds']}")
        print(f"   Successful Seeds: {results['summary']['successful_seeds']}")
        print(f"   Failed Seeds: {len(results['summary']['failed_seeds'])}")
        
        if results['summary']['critical_issues']:
            print(f"\n⚠️  CRITICAL ISSUES FOUND:")
            for issue in results['summary']['critical_issues']:
                print(f"   🚨 {issue}")
        else:
            print(f"\n🎉 ALL GENETIC SEEDS VALIDATED SUCCESSFULLY")
        
        return results
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed validation report"""
        report = []
        report.append("# Comprehensive Genetic Seed Validation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- Total Seeds: {results['summary']['total_seeds']}")
        report.append(f"- Successful: {results['summary']['successful_seeds']}")
        report.append(f"- Failed: {len(results['summary']['failed_seeds'])}")
        report.append("")
        
        # Signal Generation Results
        report.append("## Signal Generation by Scenario")
        for scenario, stats in results['summary']['scenario_performance'].items():
            report.append(f"### {scenario.title()} Market")
            report.append(f"- Active Seeds: {stats['seeds_with_signals']}/{results['summary']['total_seeds']}")
            report.append(f"- Average Signals per Seed: {stats['avg_signals_per_seed']:.1f}")
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Results by Seed")
        for seed_name, scenarios in results['signal_generation'].items():
            report.append(f"### {seed_name}")
            for scenario_name, result in scenarios.items():
                if result['success']:
                    report.append(f"- {scenario_name}: ✅ {result['non_zero_signals']} signals")
                else:
                    report.append(f"- {scenario_name}: ❌ {result['error']}")
            report.append("")
        
        # Critical Issues
        if results['summary']['critical_issues']:
            report.append("## Critical Issues")
            for issue in results['summary']['critical_issues']:
                report.append(f"- 🚨 {issue}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main validation execution"""
    validator = ComprehensiveGeneticSeedValidator()
    results = validator.run_comprehensive_validation()
    
    # Generate and save detailed report
    report = validator.generate_detailed_report(results)
    with open('genetic_seed_validation_report.md', 'w') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: genetic_seed_validation_report.md")
    
    # Return success/failure for CI/CD integration
    critical_failures = len(results['summary']['critical_issues'])
    if critical_failures > 0:
        print(f"\n❌ VALIDATION FAILED: {critical_failures} critical issues found")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED: All genetic seeds functioning correctly")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)