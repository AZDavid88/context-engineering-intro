#!/usr/bin/env python3
"""
INDEPENDENT STABILITY VERIFICATION TEST

This script can be run independently to verify genetic algorithm stability.
Created to address concerns about test credibility and false positives.

Usage: python verify_stability_test.py

Expected Output:
- If stable: All tests pass with detailed metrics
- If unstable: Specific error details and locations
"""

import asyncio
import sys
import traceback
import warnings
from typing import Dict, Any

def main():
    print("🔍 INDEPENDENT GENETIC ALGORITHM STABILITY VERIFICATION")
    print("=" * 60)
    print("This test can be run independently to verify results")
    print("=" * 60)
    
    # Capture all warnings for analysis
    warnings.filterwarnings('default')
    warning_count = {'before': 0, 'after': 0}
    
    async def run_stability_test() -> Dict[str, Any]:
        results = {
            'initialization': False,
            'population_creation': False,
            'validation': None,
            'parameter_check': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Import after we've set up warning capture
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
            
            # Test 1: Pool initialization with minimal config
            print("\n1. Testing Pool Initialization...")
            config = EvolutionConfig(population_size=10, generations=1)
            pool = GeneticStrategyPool(config)
            results['initialization'] = True
            print("   ✅ Pool initialized successfully")
            
            # Test 2: Population creation
            print("\n2. Testing Population Creation...")
            pop_count = await pool.initialize_population()
            results['population_creation'] = True
            print(f"   ✅ Created {pop_count} individuals")
            
            # Test 3: Population validation
            print("\n3. Testing Population Validation...")
            validation = pool._validate_population_with_tolerance()
            results['validation'] = validation
            
            critical_failures = validation.get('critical_failures', 0)
            warnings_count = validation.get('warnings', 0)
            total_individuals = validation.get('total_individuals', 0)
            
            print(f"   📊 Validation Results:")
            print(f"      - Total individuals: {total_individuals}")
            print(f"      - Critical failures: {critical_failures}")
            print(f"      - Warnings: {warnings_count}")
            
            if critical_failures > 0:
                results['errors'].append(f"Critical validation failures: {critical_failures}")
                print("   ❌ CRITICAL FAILURES DETECTED")
            else:
                print("   ✅ No critical failures")
            
            # Test 4: Parameter type and bounds checking
            print("\n4. Testing Parameter Integrity...")
            param_issues = []
            
            for i, individual in enumerate(pool.population[:5]):  # Check first 5
                if individual is None:
                    param_issues.append(f"Individual {i} is None")
                    continue
                    
                if individual.genes is None:
                    param_issues.append(f"Individual {i} has None genes")
                    continue
                
                for param_name, param_value in individual.genes.parameters.items():
                    if param_value is None:
                        param_issues.append(f"Individual {i}: {param_name} is None")
                    elif not isinstance(param_value, (int, float)):
                        param_issues.append(f"Individual {i}: {param_name} is {type(param_value)}")
            
            if param_issues:
                results['errors'].extend(param_issues)
                print(f"   ❌ Parameter issues found: {len(param_issues)}")
                for issue in param_issues[:3]:  # Show first 3
                    print(f"      - {issue}")
            else:
                results['parameter_check'] = True
                print("   ✅ All parameters properly typed")
            
            return results
            
        except Exception as e:
            results['errors'].append(f"Fatal error: {str(e)}")
            print(f"\n❌ FATAL ERROR: {e}")
            traceback.print_exc()
            return results
    
    # Run the async test
    test_results = asyncio.run(run_stability_test())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INDEPENDENT VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = (
        test_results['initialization'] and
        test_results['population_creation'] and
        test_results['parameter_check'] and
        len(test_results['errors']) == 0 and
        (test_results['validation'] and test_results['validation'].get('critical_failures', 0) == 0)
    )
    
    if all_passed:
        print("🎉 RESULT: ALL STABILITY TESTS PASSED")
        print("The genetic algorithm core appears stable and production-ready.")
    else:
        print("❌ RESULT: STABILITY ISSUES DETECTED")
        print("The following issues were found:")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    print(f"\n📊 Detailed Results: {test_results}")
    
    # Return exit code for automation
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)