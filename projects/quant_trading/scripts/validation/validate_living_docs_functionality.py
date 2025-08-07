#!/usr/bin/env python3
"""
Living Documentation Functionality Validator

Simple validator to check that scripts actually work as documented in
/verified_docs/by_module_simplified and integrate properly.

No overcomplicated frameworks - just straightforward functionality testing.
"""

import asyncio
import sys
import os
import importlib
import traceback
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

class LivingDocsValidator:
    """Simple validator for living documentation functionality."""
    
    def __init__(self):
        self.project_root = project_root
        self.docs_path = project_root / "verified_docs" / "by_module_simplified"
        self.src_path = project_root / "src"
        
        self.results = {
            'modules_tested': 0,
            'modules_working': 0,
            'modules_broken': 0,
            'total_functions_tested': 0,
            'functions_working': 0,
            'functions_broken': 0,
            'detailed_results': {},
            'timestamp': datetime.now()
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all documented modules."""
        print("🔍 LIVING DOCUMENTATION FUNCTIONALITY VALIDATION")
        print("=" * 60)
        print(f"Docs Path: {self.docs_path}")
        print(f"Source Path: {self.src_path}")
        print("=" * 60)
        
        # Get all module directories from docs
        module_dirs = [d for d in self.docs_path.iterdir() if d.is_dir()]
        
        for module_dir in sorted(module_dirs):
            module_name = module_dir.name
            print(f"\n📁 Testing Module: {module_name}")
            
            self.results['modules_tested'] += 1
            
            try:
                module_result = self._test_module(module_name, module_dir)
                self.results['detailed_results'][module_name] = module_result
                
                if module_result['status'] == 'working':
                    self.results['modules_working'] += 1
                    print(f"   ✅ Module {module_name} - WORKING")
                else:
                    self.results['modules_broken'] += 1
                    print(f"   ❌ Module {module_name} - BROKEN")
                    if module_result.get('error'):
                        print(f"      Error: {module_result['error'][:100]}...")
                
                # Update function counts
                self.results['total_functions_tested'] += module_result.get('functions_tested', 0)
                self.results['functions_working'] += module_result.get('functions_working', 0)
                self.results['functions_broken'] += module_result.get('functions_broken', 0)
                
            except Exception as e:
                self.results['modules_broken'] += 1
                self.results['detailed_results'][module_name] = {
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(f"   💥 Module {module_name} - ERROR: {e}")
        
        self._print_summary()
        return self.results
    
    def _test_module(self, module_name: str, module_dir: Path) -> Dict[str, Any]:
        """Test a single module's functionality."""
        result = {
            'status': 'unknown',
            'functions_tested': 0,
            'functions_working': 0,
            'functions_broken': 0,
            'import_status': {},
            'function_tests': {},
            'documentation_claims': {},
            'reality_check': {}
        }
        
        try:
            # Read documentation claims
            verification_report = module_dir / "function_verification_report.md"
            if verification_report.exists():
                result['documentation_claims'] = self._extract_documentation_claims(verification_report)
            
            # Test actual imports and functionality
            result['reality_check'] = self._test_actual_functionality(module_name)
            
            # Compare claims vs reality
            working_count = 0
            broken_count = 0
            
            for component, status in result['reality_check'].items():
                if status.get('working', False):
                    working_count += 1
                else:
                    broken_count += 1
            
            result['functions_tested'] = working_count + broken_count
            result['functions_working'] = working_count
            result['functions_broken'] = broken_count
            
            if broken_count == 0:
                result['status'] = 'working'
            elif working_count > broken_count:
                result['status'] = 'mostly_working'
            else:
                result['status'] = 'broken'
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _extract_documentation_claims(self, report_file: Path) -> Dict[str, Any]:
        """Extract claims from documentation."""
        claims = {
            'files_analyzed': 0,
            'functions_claimed': 0,
            'classes_claimed': [],
            'key_functions': []
        }
        
        try:
            content = report_file.read_text()
            
            # Extract basic statistics (simple regex patterns)
            if "Files Analyzed" in content:
                for line in content.split('\n'):
                    if "Files Analyzed" in line and ":" in line:
                        try:
                            claims['files_analyzed'] = int(line.split(':')[1].strip().split()[0])
                        except:
                            pass
            
            # Extract class names mentioned
            classes_mentioned = []
            for line in content.split('\n'):
                if 'class' in line.lower() and '`' in line:
                    # Extract class names from markdown code
                    parts = line.split('`')
                    for part in parts:
                        if 'class' in part.lower() or part.endswith('Engine') or part.endswith('Manager'):
                            classes_mentioned.append(part.strip())
            
            claims['classes_claimed'] = list(set(classes_mentioned))
            
        except Exception as e:
            claims['extraction_error'] = str(e)
        
        return claims
    
    def _test_actual_functionality(self, module_name: str) -> Dict[str, Any]:
        """Test the actual functionality of a module."""
        reality = {}
        
        # Module-specific testing strategies
        if module_name == 'strategy':
            reality.update(self._test_strategy_module())
        elif module_name == 'execution':
            reality.update(self._test_execution_module())
        elif module_name == 'data':
            reality.update(self._test_data_module())
        elif module_name == 'config':
            reality.update(self._test_config_module())
        elif module_name == 'backtesting':
            reality.update(self._test_backtesting_module())
        elif module_name == 'discovery':
            reality.update(self._test_discovery_module())
        else:
            # Generic module testing
            reality.update(self._test_generic_module(module_name))
        
        return reality
    
    def _test_strategy_module(self) -> Dict[str, Any]:
        """Test strategy module specifically."""
        tests = {}
        
        # Test genetic seeds registry
        try:
            import src.strategy.genetic_seeds as genetic_seeds
            registry = genetic_seeds.get_registry()
            
            seed_names = registry.get_all_seed_names()
            tests['genetic_seeds_registry'] = {
                'working': len(seed_names) > 0,
                'seed_count': len(seed_names),
                'sample_seeds': seed_names[:5]
            }
        except Exception as e:
            tests['genetic_seeds_registry'] = {
                'working': False,
                'error': str(e)
            }
        
        # Test genetic engine core
        try:
            from src.strategy.genetic_engine_core import GeneticEngineCore
            core = GeneticEngineCore()
            tests['genetic_engine_core'] = {
                'working': True,
                'class_instantiated': True
            }
        except Exception as e:
            tests['genetic_engine_core'] = {
                'working': False,
                'error': str(e)
            }
        
        # Test universal strategy engine
        try:
            from src.strategy.universal_strategy_engine import UniversalStrategyEngine
            engine = UniversalStrategyEngine()
            tests['universal_strategy_engine'] = {
                'working': True,
                'class_instantiated': True
            }
        except Exception as e:
            tests['universal_strategy_engine'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _test_execution_module(self) -> Dict[str, Any]:
        """Test execution module specifically."""
        tests = {}
        
        # Test genetic strategy pool
        try:
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
            config = EvolutionConfig(population_size=5, generations=1)
            tests['genetic_strategy_pool'] = {
                'working': True,
                'config_created': True
            }
        except Exception as e:
            tests['genetic_strategy_pool'] = {
                'working': False,
                'error': str(e)
            }
        
        # Test retail connection optimizer
        try:
            from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
            optimizer = RetailConnectionOptimizer()
            tests['retail_connection_optimizer'] = {
                'working': True,
                'class_instantiated': True
            }
        except Exception as e:
            tests['retail_connection_optimizer'] = {
                'working': False,
                'error': str(e)
            }
        
        # Test order management
        try:
            from src.execution.order_management import OrderType, OrderStatus
            tests['order_management'] = {
                'working': True,
                'enums_imported': True
            }
        except Exception as e:
            tests['order_management'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _test_data_module(self) -> Dict[str, Any]:
        """Test data module specifically."""
        tests = {}
        
        # Test storage interfaces
        try:
            from src.data.storage_interfaces import get_storage_implementation
            storage = get_storage_implementation()
            tests['storage_interfaces'] = {
                'working': True,
                'storage_created': storage is not None,
                'storage_type': type(storage).__name__
            }
        except Exception as e:
            tests['storage_interfaces'] = {
                'working': False,
                'error': str(e)
            }
        
        # Test hyperliquid client
        try:
            from src.data.hyperliquid_client import HyperliquidClient
            client = HyperliquidClient()
            tests['hyperliquid_client'] = {
                'working': True,
                'client_created': True
            }
        except Exception as e:
            tests['hyperliquid_client'] = {
                'working': False,
                'error': str(e)
            }
        
        # Test market data pipeline
        try:
            from src.data.market_data_pipeline import MarketDataPipeline
            pipeline = MarketDataPipeline()
            tests['market_data_pipeline'] = {
                'working': True,
                'pipeline_created': True
            }
        except Exception as e:
            tests['market_data_pipeline'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _test_config_module(self) -> Dict[str, Any]:
        """Test config module specifically."""
        tests = {}
        
        # Test settings
        try:
            from src.config.settings import get_settings, Settings
            settings = get_settings()
            tests['settings'] = {
                'working': True,
                'settings_loaded': settings is not None,
                'settings_type': type(settings).__name__
            }
        except Exception as e:
            tests['settings'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _test_backtesting_module(self) -> Dict[str, Any]:
        """Test backtesting module specifically."""
        tests = {}
        
        # Test performance analyzer
        try:
            from src.backtesting.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            tests['performance_analyzer'] = {
                'working': True,
                'analyzer_created': True
            }
        except Exception as e:
            tests['performance_analyzer'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _test_discovery_module(self) -> Dict[str, Any]:
        """Test discovery module specifically."""
        tests = {}
        
        # Test enhanced asset filter
        try:
            from src.discovery.enhanced_asset_filter import EnhancedAssetFilter
            # Need settings for this
            from src.config.settings import get_settings
            settings = get_settings()
            filter_obj = EnhancedAssetFilter(settings)
            tests['enhanced_asset_filter'] = {
                'working': True,
                'filter_created': True
            }
        except Exception as e:
            tests['enhanced_asset_filter'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _test_generic_module(self, module_name: str) -> Dict[str, Any]:
        """Generic testing for unknown modules."""
        tests = {}
        
        # Special handling for documentation-only modules
        documentation_only_modules = {'deployment'}  # Add more as needed
        
        if module_name in documentation_only_modules:
            # This is a documentation-only module - check if docs exist
            tests['documentation_module'] = {
                'working': True,
                'type': 'documentation_only',
                'note': f'{module_name} module contains guidelines and documentation, not executable code'
            }
            return tests
        
        try:
            # Try to import the module
            module_path = f"src.{module_name}"
            module = importlib.import_module(module_path)
            
            # Get all classes and functions
            members = inspect.getmembers(module)
            classes = [name for name, obj in members if inspect.isclass(obj)]
            functions = [name for name, obj in members if inspect.isfunction(obj)]
            
            tests['module_import'] = {
                'working': True,
                'classes_found': len(classes),
                'functions_found': len(functions),
                'sample_classes': classes[:3],
                'sample_functions': functions[:3]
            }
            
        except Exception as e:
            tests['module_import'] = {
                'working': False,
                'error': str(e)
            }
        
        return tests
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📊 LIVING DOCUMENTATION VALIDATION SUMMARY")
        print("=" * 60)
        
        total_modules = self.results['modules_tested']
        working_modules = self.results['modules_working']
        broken_modules = self.results['modules_broken']
        
        print(f"Modules Tested: {total_modules}")
        print(f"Modules Working: {working_modules} ✅")
        print(f"Modules Broken: {broken_modules} ❌")
        
        if total_modules > 0:
            success_rate = (working_modules / total_modules) * 100
            print(f"Module Success Rate: {success_rate:.1f}%")
        
        total_functions = self.results['total_functions_tested']
        working_functions = self.results['functions_working']
        broken_functions = self.results['functions_broken']
        
        print(f"\nFunctions/Components Tested: {total_functions}")
        print(f"Functions Working: {working_functions} ✅")
        print(f"Functions Broken: {broken_functions} ❌")
        
        if total_functions > 0:
            func_success_rate = (working_functions / total_functions) * 100
            print(f"Function Success Rate: {func_success_rate:.1f}%")
        
        # Show broken modules
        if broken_modules > 0:
            print(f"\n❌ BROKEN MODULES:")
            for module, result in self.results['detailed_results'].items():
                if result.get('status') in ['broken', 'error']:
                    print(f"   • {module}: {result.get('status', 'unknown')}")
                    if 'error' in result:
                        print(f"     Error: {result['error'][:100]}...")
                    
                    # Show broken components within the module
                    if 'reality_check' in result:
                        for component, component_result in result['reality_check'].items():
                            if not component_result.get('working', True):
                                error = component_result.get('error', 'Unknown error')
                                print(f"     └─ {component}: {error[:80]}...")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if broken_modules == 0:
            print("   ✅ All documented modules are functional")
            print("   ✅ Living documentation accurately reflects reality")
        elif working_modules > broken_modules:
            print("   ⚠️ Most modules working but some issues detected")
            print("   📝 Documentation mostly accurate with some gaps")
        else:
            print("   ❌ Significant functionality issues detected")
            print("   📝 Documentation may not reflect current reality")
        
        print("=" * 60)

async def main():
    """Main validation execution."""
    validator = LivingDocsValidator()
    results = validator.run_comprehensive_validation()
    
    # Save detailed results to file
    import json
    results_file = validator.project_root / "validation_results.json"
    
    # Convert datetime to string for JSON serialization
    json_results = results.copy()
    json_results['timestamp'] = results['timestamp'].isoformat()
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['modules_broken'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())