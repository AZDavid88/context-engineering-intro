#!/usr/bin/env python3
"""
Complete System Validation Script

This script validates the entire integrated system end-to-end:
- Import validation for all components
- Configuration validation
- API connectivity testing
- Memory usage validation
- Integration pipeline testing

Usage:
    python validate_complete_system.py [--testnet] [--verbose]
"""

import asyncio
import sys
import os
import logging
import importlib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


class SystemValidator:
    """Comprehensive system validation."""
    
    def __init__(self, use_testnet: bool = True, verbose: bool = False):
        self.use_testnet = use_testnet
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.validation_results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        return logging.getLogger(__name__)
    
    async def run_complete_validation(self) -> bool:
        """Run complete system validation."""
        
        self.logger.info("🔍 Starting Complete System Validation")
        self.logger.info("=" * 50)
        
        validation_tests = [
            ("Import Validation", self._validate_imports),
            ("Configuration Validation", self._validate_configuration),
            ("API Connectivity", self._validate_api_connectivity),
            ("Memory Management", self._validate_memory_management), 
            ("Integration Pipeline", self._validate_integration_pipeline),
            ("Genetic Engine", self._validate_genetic_engine),
            ("Research Compliance", self._validate_research_compliance)
        ]
        
        all_passed = True
        
        for test_name, test_func in validation_tests:
            self.logger.info(f"🧪 Running {test_name}...")
            try:
                result = await test_func()
                if result:
                    self.logger.info(f"   ✅ {test_name} PASSED")
                    self.validation_results[test_name] = "PASSED"
                else:
                    self.logger.error(f"   ❌ {test_name} FAILED")
                    self.validation_results[test_name] = "FAILED"
                    all_passed = False
            except Exception as e:
                self.logger.error(f"   ❌ {test_name} ERROR: {e}")
                self.validation_results[test_name] = f"ERROR: {e}"
                all_passed = False
        
        self.logger.info("")
        self._generate_validation_summary(all_passed)
        
        return all_passed
    
    async def _validate_imports(self) -> bool:
        """Validate all critical imports."""
        
        critical_imports = [
            'src.config.settings',
            'src.data.dynamic_asset_data_collector',
            'src.discovery.enhanced_asset_filter',
            'src.strategy.genetic_engine',
            'src.data.hyperliquid_client',
            'src.backtesting.performance_analyzer'
        ]
        
        import_results = []
        
        for module_name in critical_imports:
            try:
                importlib.import_module(module_name)
                import_results.append(True)
                self.logger.debug(f"   ✅ {module_name}")
            except ImportError as e:
                self.logger.error(f"   ❌ {module_name}: {e}")
                import_results.append(False)
        
        return all(import_results)
    
    async def _validate_configuration(self) -> bool:
        """Validate configuration system."""
        
        try:
            from src.config.settings import Settings
            
            # Test configuration loading
            settings = Settings()
            
            # Validate critical settings exist
            required_settings = [
                'hyperliquid',
                'trading',
                'genetic_algorithm'
            ]
            
            for setting in required_settings:
                if not hasattr(settings, setting):
                    self.logger.error(f"   ❌ Missing setting: {setting}")
                    return False
            
            # Test testnet/mainnet configuration
            if self.use_testnet:
                settings.environment = "testnet"
                self.logger.debug("   🧪 Testnet configuration validated")
            else:
                settings.environment = "mainnet"
                self.logger.debug("   🏭 Mainnet configuration validated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"   ❌ Configuration validation failed: {e}")
            return False
    
    async def _validate_api_connectivity(self) -> bool:
        """Validate API connectivity."""
        
        try:
            from src.config.settings import Settings
            from src.data.hyperliquid_client import HyperliquidClient
            
            settings = Settings()
            if self.use_testnet:
                settings.environment = "testnet"
            else:
                settings.environment = "mainnet"
            
            client = HyperliquidClient(settings)
            
            # Test connection
            await client.connect()
            
            # Test basic API call
            try:
                all_mids = await client.get_all_mids()
                
                if all_mids and isinstance(all_mids, dict):
                    self.logger.debug(f"   ✅ API connectivity confirmed - {len(all_mids)} assets available")
                    await client.disconnect()
                    return True
                else:
                    self.logger.error("   ❌ API returned invalid data")
                    await client.disconnect()
                    return False
                    
            except Exception as api_error:
                self.logger.error(f"   ❌ API call failed: {api_error}")
                await client.disconnect()
                return False
            
        except Exception as e:
            self.logger.error(f"   ❌ API connectivity validation failed: {e}")
            return False
    
    async def _validate_memory_management(self) -> bool:
        """Validate memory management capabilities."""
        
        try:
            import psutil
            import pandas as pd
            import numpy as np
            
            # Get baseline memory
            baseline_memory = psutil.Process().memory_info().rss / (1024 ** 3)
            
            # Create large dataset to test memory handling
            large_dataset = {}
            for i in range(10):  # 10 assets
                df = pd.DataFrame({
                    'open': np.random.randn(5000),
                    'high': np.random.randn(5000),
                    'low': np.random.randn(5000),
                    'close': np.random.randn(5000),
                    'volume': np.random.randn(5000)
                })
                large_dataset[f'ASSET_{i}'] = df
            
            # Check memory usage
            peak_memory = psutil.Process().memory_info().rss / (1024 ** 3)
            memory_increase = peak_memory - baseline_memory
            
            # Cleanup
            del large_dataset
            import gc
            gc.collect()
            
            self.logger.debug(f"   💾 Memory test: {memory_increase:.2f}GB increase for 50K data points")
            
            # Memory increase should be reasonable (less than 2GB for test data)
            if memory_increase < 2.0:
                return True
            else:
                self.logger.error(f"   ❌ Excessive memory usage: {memory_increase:.2f}GB")
                return False
            
        except Exception as e:
            self.logger.error(f"   ❌ Memory validation failed: {e}")
            return False
    
    async def _validate_integration_pipeline(self) -> bool:
        """Validate integration pipeline components."""
        
        try:
            from src.config.settings import Settings
            from src.data.dynamic_asset_data_collector import IntegratedPipelineOrchestrator
            
            settings = Settings()
            if self.use_testnet:
                settings.environment = "testnet"
            else:
                settings.environment = "mainnet"
            
            # Initialize orchestrator
            orchestrator = IntegratedPipelineOrchestrator(settings)
            
            # Validate components are initialized
            if orchestrator.asset_filter is None:
                self.logger.error("   ❌ Asset filter not initialized")
                return False
            
            if orchestrator.data_collector is None:
                self.logger.error("   ❌ Data collector not initialized")
                return False
            
            # Test pipeline status
            status = orchestrator.get_pipeline_status()
            
            if not status['components_status']['asset_filter_ready']:
                self.logger.error("   ❌ Asset filter not ready")
                return False
            
            if not status['components_status']['data_collector_ready']:
                self.logger.error("   ❌ Data collector not ready")
                return False
            
            self.logger.debug("   ✅ Pipeline components validated")
            return True
            
        except Exception as e:
            self.logger.error(f"   ❌ Integration pipeline validation failed: {e}")
            return False
    
    async def _validate_genetic_engine(self) -> bool:
        """Validate genetic engine initialization."""
        
        try:
            from src.config.settings import Settings
            from src.strategy.genetic_engine import GeneticEngine
            
            settings = Settings()
            genetic_engine = GeneticEngine(settings=settings)
            
            # Validate configuration using existing interface
            if genetic_engine.config.population_size <= 0:
                self.logger.error("   ❌ Invalid population size")
                return False
            
            if genetic_engine.config.n_generations <= 0:
                self.logger.error("   ❌ Invalid generations count")
                return False
            
            # Validate registry is available
            seed_registry = genetic_engine.seed_registry
            if not seed_registry:
                self.logger.error("   ❌ No genetic seed registry available")
                return False
                
            available_seeds = seed_registry.list_all_seeds()
            if not available_seeds:
                self.logger.error("   ❌ No genetic seeds registered")
                return False
            
            self.logger.debug(f"   ✅ Genetic engine validated - {len(available_seeds)} seeds available")
            return True
            
        except Exception as e:
            self.logger.error(f"   ❌ Genetic engine validation failed: {e}")
            return False
    
    async def _validate_research_compliance(self) -> bool:
        """Validate research compliance and documentation."""
        
        try:
            research_directories = [
                'research/hyperliquid_documentation',
                'research/vectorbt_comprehensive',
                'research/deap'
            ]
            
            missing_research = []
            
            for research_dir in research_directories:
                full_path = os.path.join(os.path.dirname(__file__), research_dir)
                if not os.path.exists(full_path):
                    missing_research.append(research_dir)
            
            if missing_research:
                self.logger.error(f"   ❌ Missing research directories: {missing_research}")
                return False
            
            # Check for critical research files
            critical_files = [
                'research/hyperliquid_documentation/3_info_endpoint.md',
                'research/vectorbt_comprehensive/page_4_memory_management_large_scale.md',
                'research/deap/3_genetic_programming_comprehensive.md'
            ]
            
            missing_files = []
            
            for file_path in critical_files:
                full_path = os.path.join(os.path.dirname(__file__), file_path)
                if not os.path.exists(full_path):
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"   ❌ Missing critical research files: {missing_files}")
                return False
            
            self.logger.debug("   ✅ Research compliance validated")
            return True
            
        except Exception as e:
            self.logger.error(f"   ❌ Research compliance validation failed: {e}")
            return False
    
    def _generate_validation_summary(self, all_passed: bool):
        """Generate validation summary."""
        
        self.logger.info("📊 VALIDATION SUMMARY")
        self.logger.info("=" * 50)
        
        for test_name, result in self.validation_results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            self.logger.info(f"   {status_icon} {test_name}: {result}")
        
        self.logger.info("")
        
        if all_passed:
            self.logger.info("🎉 ALL VALIDATIONS PASSED")
            self.logger.info("🚀 System is ready for integrated pipeline execution")
            self.logger.info("💡 Next step: Run 'python run_integrated_pipeline.py'")
        else:
            self.logger.error("❌ SOME VALIDATIONS FAILED")
            self.logger.error("🔧 Please fix the issues above before running the pipeline")
        
        self.logger.info("")
        self.logger.info(f"📅 Validation completed at: {datetime.now().isoformat()}")


async def main():
    """Main validation function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Complete System")
    parser.add_argument("--testnet", action="store_true", default=True,
                       help="Use testnet for validation (default: True)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    validator = SystemValidator(use_testnet=args.testnet, verbose=args.verbose)
    success = await validator.run_complete_validation()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)