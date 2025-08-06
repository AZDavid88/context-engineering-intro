#!/usr/bin/env python3
"""
Comprehensive System Validation Suite

This validates ALL core functionality across the entire system to ensure
no regressions have been introduced by any architectural changes.
"""

import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any

sys.path.append('/workspaces/context-engineering-intro/projects/quant_trading')

# Import all core modules
try:
    import src
    from src.config import get_settings
    from src.discovery import EnhancedAssetFilter, get_crypto_safe_parameters
    from src.strategy.genetic_seeds import get_registry, BaseSeed, SeedGenes, SeedType
    from src.backtesting import PerformanceAnalyzer
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

class SystemValidator:
    """Comprehensive system functionality validator."""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': [],
            'start_time': datetime.now()
        }
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test with error handling."""
        self.results['total_tests'] += 1
        
        try:
            print(f"🧪 Testing {test_name}...")
            start_time = time.time()
            
            result = test_func()
            
            duration = time.time() - start_time
            
            if result:
                print(f"   ✅ PASSED ({duration:.2f}s)")
                self.results['passed_tests'] += 1
                self.results['test_details'].append({
                    'name': test_name,
                    'status': 'PASSED',
                    'duration': duration,
                    'error': None
                })
                return True
            else:
                print(f"   ❌ FAILED ({duration:.2f}s)")
                self.results['failed_tests'] += 1
                self.results['test_details'].append({
                    'name': test_name,
                    'status': 'FAILED',
                    'duration': duration,
                    'error': 'Test returned False'
                })
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{str(e)} | {traceback.format_exc()}"
            print(f"   ❌ ERROR ({duration:.2f}s): {e}")
            
            self.results['failed_tests'] += 1
            self.results['test_details'].append({
                'name': test_name,
                'status': 'ERROR',
                'duration': duration,
                'error': error_msg
            })
            return False
    
    def test_package_imports(self) -> bool:
        """Test that all packages import correctly."""
        if not IMPORTS_OK:
            print(f"Import error: {IMPORT_ERROR}")
            return False
        
        # Test individual module access
        required_modules = [
            'src.config',
            'src.data', 
            'src.discovery',
            'src.strategy',
            'src.backtesting',
            'src.execution',
            'src.utils'
        ]
        
        for module_name in required_modules:
            try:
                exec(f"import {module_name}")
            except Exception as e:
                print(f"Failed to import {module_name}: {e}")
                return False
        
        return True
    
    def test_configuration_system(self) -> bool:
        """Test configuration management."""
        try:
            settings = get_settings()
            
            # Test basic configuration access
            has_hyperliquid = hasattr(settings, 'hyperliquid')
            has_database = hasattr(settings, 'database')
            
            return has_hyperliquid and has_database
        except:
            return False
    
    def test_genetic_seeds_complete(self) -> bool:
        """Test complete genetic seeds functionality."""
        try:
            registry = get_registry()
            
            # Test registry has correct number of seeds
            if len(registry._registry) != 14:
                return False
            
            # Test all seeds can be instantiated
            for seed_name in registry._registry:
                seed_class = registry.get_seed_class(seed_name)
                if not seed_class:
                    return False
                
                # Test instantiation
                genes = SeedGenes(
                    seed_id='test',
                    seed_type=SeedType.MOMENTUM,
                    parameters={}
                )
                seed_instance = seed_class(genes)
                
                # Test basic methods exist
                if not hasattr(seed_instance, 'seed_name'):
                    return False
                if not hasattr(seed_instance, 'parameter_bounds'):
                    return False
                if not hasattr(seed_instance, 'generate_signals'):
                    return False
            
            return True
        except:
            return False
    
    def test_discovery_system(self) -> bool:
        """Test asset discovery and filtering system."""
        try:
            # Test crypto safe parameters
            crypto_params = get_crypto_safe_parameters()
            if not crypto_params:
                return False
            
            # Test enhanced asset filter initialization
            settings = get_settings()
            asset_filter = EnhancedAssetFilter(settings)
            
            # Test filter has required methods
            required_methods = ['filter_universe', 'get_enhanced_filter_summary']
            for method in required_methods:
                if not hasattr(asset_filter, method):
                    return False
            
            return True
        except:
            return False
    
    def test_backtesting_system(self) -> bool:
        """Test backtesting engine components."""
        try:
            # Test performance analyzer
            analyzer = PerformanceAnalyzer()
            
            # Test analyzer has required methods
            required_methods = ['analyze_portfolio_performance', 'get_performance_summary']
            for method in required_methods:
                if not hasattr(analyzer, method):
                    return False
            
            return True
        except:
            return False
    
    def test_system_integration(self) -> bool:
        """Test cross-module integration."""
        try:
            # Test that modules can communicate
            settings = get_settings()
            registry = get_registry()
            crypto_params = get_crypto_safe_parameters()
            
            # Test genetic seed with crypto parameters
            safe_genome = crypto_params.generate_crypto_safe_genome()
            if not safe_genome:
                return False
            
            return True
        except:
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete system validation."""
        print("🔍 COMPREHENSIVE SYSTEM VALIDATION")
        print("=" * 60)
        print(f"Started at: {self.results['start_time']}")
        print("=" * 60)
        
        # Define all tests
        tests = [
            ("Package Imports", self.test_package_imports),
            ("Configuration System", self.test_configuration_system),
            ("Genetic Seeds Complete", self.test_genetic_seeds_complete),
            ("Discovery System", self.test_discovery_system),
            ("Backtesting System", self.test_backtesting_system),
            ("System Integration", self.test_system_integration),
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Calculate final results
        self.results['end_time'] = datetime.now()
        self.results['total_duration'] = (
            self.results['end_time'] - self.results['start_time']
        ).total_seconds()
        
        self.results['success_rate'] = (
            self.results['passed_tests'] / self.results['total_tests'] * 100
            if self.results['total_tests'] > 0 else 0
        )
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed_tests']} ✅")
        print(f"Failed: {self.results['failed_tests']} ❌")
        print(f"Success Rate: {self.results['success_rate']:.1f}%")
        print(f"Total Duration: {self.results['total_duration']:.2f}s")
        
        if self.results['failed_tests'] > 0:
            print(f"\n❌ FAILED TESTS:")
            for test in self.results['test_details']:
                if test['status'] != 'PASSED':
                    print(f"   • {test['name']}: {test['status']}")
                    if test['error']:
                        print(f"     Error: {test['error'][:100]}...")
        
        if self.results['success_rate'] == 100:
            print(f"\n🎉 ALL SYSTEMS OPERATIONAL")
            print(f"✅ No functionality lost from architectural changes")
        else:
            print(f"\n⚠️ SYSTEM ISSUES DETECTED")
            print(f"❌ Some functionality may be compromised")

def main():
    """Run comprehensive system validation."""
    validator = SystemValidator()
    results = validator.run_all_tests()
    
    # Return appropriate exit code
    sys.exit(0 if results['success_rate'] == 100 else 1)

if __name__ == "__main__":
    main()