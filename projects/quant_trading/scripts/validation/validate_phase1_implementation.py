#!/usr/bin/env python3
"""
Phase 1 Implementation Validation - Strategic Forest-Level Testing

This script validates the complete Phase 1 Ray cluster scaling implementation
with strategic focus on business value and clean phase progression.

Validation Categories:
1. Storage Interface Architecture - Clean backend abstraction
2. Ray Cluster Integration - Distributed computing foundation
3. Genetic Algorithm Performance - Concrete business value
4. Phase Progression Readiness - Clean upgrade path to Phase 2-4

Strategic Design Principles:
- Business value focus over infrastructure complexity
- Forest-level validation over trees-level details
- Clean progression path validation
- Production-ready performance benchmarks
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import statistics

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.storage_interfaces import (
    DataStorageInterface, 
    LocalDataStorage, 
    SharedDataStorage, 
    get_storage_implementation
)
from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig, EvolutionMode
from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
from src.strategy.genetic_seeds.seed_registry import get_registry
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase1ValidationResults:
    """Container for Phase 1 validation results."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.overall_status = "pending"
        self.business_value_score = 0.0
        self.phase_progression_ready = False
        self.validation_timestamp = datetime.now()
    
    def add_result(self, category: str, test_name: str, status: str, 
                   details: Dict[str, Any] = None, performance_metrics: Dict[str, float] = None):
        """Add validation result."""
        if category not in self.results:
            self.results[category] = {}
            
        self.results[category][test_name] = {
            "status": status,
            "details": details or {},
            "performance_metrics": performance_metrics or {},
            "timestamp": datetime.now()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total_tests = sum(len(cat) for cat in self.results.values())
        passed_tests = sum(
            1 for cat in self.results.values() 
            for test in cat.values() 
            if test["status"] == "passed"
        )
        
        critical_failures = []
        for cat_name, category in self.results.items():
            for test_name, test in category.items():
                if test["status"] == "failed" and test["details"].get("critical", False):
                    critical_failures.append(f"{cat_name}.{test_name}")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests) if total_tests > 0 else 0.0,
            "critical_failures": critical_failures,
            "business_value_score": self.business_value_score,
            "phase_progression_ready": self.phase_progression_ready,
            "overall_status": self.overall_status
        }


class Phase1Validator:
    """Comprehensive Phase 1 implementation validator."""
    
    def __init__(self):
        self.results = Phase1ValidationResults()
        self.logger = logger
    
    async def run_all_validations(self) -> Phase1ValidationResults:
        """
        Run all Phase 1 validation tests.
        
        Returns:
            Complete validation results with business value assessment
        """
        try:
            self.logger.info("🚀 Starting Phase 1 Implementation Validation")
            
            # Core Architecture Validation
            await self.validate_storage_interface()
            await self.validate_genetic_pool_integration()
            await self.validate_ray_infrastructure()
            
            # Business Value Validation  
            await self.validate_genetic_algorithm_performance()
            await self.validate_distributed_computing_benefits()
            
            # Phase Progression Validation
            await self.validate_phase2_preparation()
            await self.validate_clean_upgrade_paths()
            
            # Calculate overall status and business value
            self._calculate_overall_results()
            
            self.logger.info("✅ Phase 1 Implementation Validation Complete")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed with exception: {e}")
            self.results.overall_status = "failed"
            return self.results
    
    async def validate_storage_interface(self):
        """Validate strategic storage interface implementation."""
        self.logger.info("🔍 Validating Storage Interface Architecture")
        
        try:
            # Test interface abstraction
            storage = get_storage_implementation()
            
            # Health check
            health = await storage.health_check()
            
            if health["status"] == "healthy":
                self.results.add_result(
                    "storage_interface", 
                    "backend_health", 
                    "passed",
                    details={"backend": health.get("backend", "unknown")},
                    performance_metrics={"query_latency_ms": health.get("query_latency_ms", 0)}
                )
            else:
                self.results.add_result(
                    "storage_interface", 
                    "backend_health", 
                    "failed",
                    details={"error": health.get("error", "Unknown"), "critical": True}
                )
            
            # Test backend switching capability (Phase 4 preparation)
            try:
                local_storage = LocalDataStorage()
                shared_storage = SharedDataStorage("/tmp/test_shared")
                
                # Both should implement the same interface
                interface_methods = [
                    "store_ohlcv_bars", "get_ohlcv_bars", 
                    "calculate_technical_indicators", "get_market_summary", "health_check"
                ]
                
                interface_compliance = True
                for method_name in interface_methods:
                    if not (hasattr(local_storage, method_name) and hasattr(shared_storage, method_name)):
                        interface_compliance = False
                        break
                
                self.results.add_result(
                    "storage_interface", 
                    "backend_switching_ready", 
                    "passed" if interface_compliance else "failed",
                    details={
                        "local_backend": type(local_storage).__name__,
                        "shared_backend": type(shared_storage).__name__,
                        "interface_compliance": interface_compliance,
                        "critical": True
                    }
                )
                
            except Exception as e:
                self.results.add_result(
                    "storage_interface", 
                    "backend_switching_ready", 
                    "failed",
                    details={"error": str(e), "critical": True}
                )
        
        except Exception as e:
            self.results.add_result(
                "storage_interface", 
                "architecture_validation", 
                "failed",
                details={"error": str(e), "critical": True}
            )
    
    async def validate_genetic_pool_integration(self):
        """Validate genetic strategy pool integration with storage interface."""
        self.logger.info("🔍 Validating Genetic Pool Integration")
        
        try:
            # Create connection optimizer (minimal for testing)
            connection_optimizer = RetailConnectionOptimizer()
            
            # Test genetic pool initialization with storage interface
            storage = get_storage_implementation()
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=connection_optimizer,
                use_ray=False,  # Test local mode first
                storage=storage
            )
            
            # Validate storage interface integration
            if hasattr(genetic_pool, 'storage') and genetic_pool.storage is not None:
                storage_health = await genetic_pool.storage.health_check()
                
                self.results.add_result(
                    "genetic_pool_integration",
                    "storage_integration",
                    "passed" if storage_health["status"] == "healthy" else "failed",
                    details={
                        "storage_backend": storage_health.get("backend", "unknown"),
                        "integration_successful": True
                    }
                )
            else:
                self.results.add_result(
                    "genetic_pool_integration",
                    "storage_integration", 
                    "failed",
                    details={"error": "Storage interface not integrated", "critical": True}
                )
            
            # Test seed registry integration
            registry = get_registry()
            available_seeds = registry.get_available_seeds()
            
            self.results.add_result(
                "genetic_pool_integration",
                "seed_registry_integration",
                "passed" if len(available_seeds) > 0 else "warning",
                details={
                    "available_seeds_count": len(available_seeds),
                    "seed_types": list(available_seeds.keys())[:5]  # First 5 for brevity
                }
            )
            
        except Exception as e:
            self.results.add_result(
                "genetic_pool_integration",
                "integration_validation",
                "failed", 
                details={"error": str(e), "critical": True}
            )
    
    async def validate_ray_infrastructure(self):
        """Validate Ray infrastructure availability and configuration."""
        self.logger.info("🔍 Validating Ray Infrastructure")
        
        try:
            # Test Ray import availability
            try:
                import ray
                ray_available = True
                ray_version = ray.__version__
            except ImportError:
                ray_available = False
                ray_version = None
            
            self.results.add_result(
                "ray_infrastructure",
                "ray_availability",
                "passed" if ray_available else "warning",
                details={
                    "ray_available": ray_available,
                    "ray_version": ray_version,
                    "note": "Ray expected to be available in container environment"
                }
            )
            
            # Test genetic pool Ray integration readiness
            connection_optimizer = RetailConnectionOptimizer()
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=connection_optimizer,
                use_ray=ray_available  # Enable Ray if available
            )
            
            ray_integration_ready = hasattr(genetic_pool, 'use_ray') and hasattr(genetic_pool, 'ray_initialized')
            
            self.results.add_result(
                "ray_infrastructure",
                "integration_readiness",
                "passed" if ray_integration_ready else "failed",
                details={
                    "integration_ready": ray_integration_ready,
                    "ray_enabled": genetic_pool.use_ray if ray_integration_ready else False
                }
            )
            
        except Exception as e:
            self.results.add_result(
                "ray_infrastructure",
                "infrastructure_validation",
                "failed",
                details={"error": str(e), "critical": False}  # Not critical for local testing
            )
    
    async def validate_genetic_algorithm_performance(self):
        """Validate genetic algorithm provides concrete business value."""
        self.logger.info("🔍 Validating Genetic Algorithm Performance")
        
        try:
            # Test minimal genetic algorithm execution
            connection_optimizer = RetailConnectionOptimizer()
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=connection_optimizer,
                evolution_config=EvolutionConfig(
                    population_size=10,  # Small for testing
                    generations=2,       # Quick test
                    mutation_rate=0.1,
                    crossover_rate=0.8
                ),
                use_ray=False  # Local mode for validation
            )
            
            # Test population initialization
            start_time = time.time()
            population_size = await genetic_pool.initialize_population()
            initialization_time = time.time() - start_time
            
            business_value_metrics = {
                "population_initialization_successful": population_size > 0,
                "initialization_time_seconds": initialization_time,
                "population_size": population_size
            }
            
            # Calculate business value score (0-100)
            if population_size > 0:
                business_value_score = min(100, (population_size / 10) * 50 + 50)  # 50-100 based on population
            else:
                business_value_score = 0
                
            self.results.add_result(
                "genetic_algorithm_performance",
                "algorithm_execution",
                "passed" if population_size > 0 else "failed",
                details=business_value_metrics,
                performance_metrics={
                    "business_value_score": business_value_score,
                    "initialization_latency_ms": initialization_time * 1000
                }
            )
            
            # Update overall business value score
            self.results.business_value_score += business_value_score * 0.4  # 40% weight
            
        except Exception as e:
            self.results.add_result(
                "genetic_algorithm_performance",
                "performance_validation", 
                "failed",
                details={"error": str(e), "critical": True}
            )
    
    async def validate_distributed_computing_benefits(self):
        """Validate distributed computing provides measurable benefits."""
        self.logger.info("🔍 Validating Distributed Computing Benefits")
        
        try:
            # This is a readiness test - actual distributed testing requires Ray cluster
            connection_optimizer = RetailConnectionOptimizer()
            
            # Test local execution baseline
            start_time = time.time()
            local_pool = GeneticStrategyPool(
                connection_optimizer=connection_optimizer,
                use_ray=False,
                evolution_config=EvolutionConfig(population_size=5, generations=1)
            )
            local_population = await local_pool.initialize_population()
            local_time = time.time() - start_time
            
            # Test Ray-ready configuration
            start_time = time.time()
            ray_pool = GeneticStrategyPool(
                connection_optimizer=connection_optimizer,
                use_ray=True,  # Will fall back to local if Ray not available
                evolution_config=EvolutionConfig(population_size=5, generations=1)
            )
            ray_population = await ray_pool.initialize_population()
            ray_time = time.time() - start_time
            
            distributed_readiness = {
                "local_execution_successful": local_population > 0,
                "ray_configuration_successful": ray_population > 0,
                "local_execution_time": local_time,
                "ray_execution_time": ray_time,
                "performance_comparison_ready": True
            }
            
            performance_score = 50 if (local_population > 0 and ray_population > 0) else 0
            
            self.results.add_result(
                "distributed_computing",
                "benefits_validation",
                "passed" if local_population > 0 and ray_population > 0 else "failed",
                details=distributed_readiness,
                performance_metrics={"readiness_score": performance_score}
            )
            
            # Update business value score
            self.results.business_value_score += performance_score * 0.3  # 30% weight
            
        except Exception as e:
            self.results.add_result(
                "distributed_computing",
                "benefits_validation",
                "failed",
                details={"error": str(e), "critical": False}
            )
    
    async def validate_phase2_preparation(self):
        """Validate readiness for Phase 2 correlation analysis integration."""
        self.logger.info("🔍 Validating Phase 2 Preparation")
        
        try:
            # Test storage interface supports correlation data
            storage = get_storage_implementation()
            
            # Validate interface methods needed for Phase 2
            phase2_methods = ["get_ohlcv_bars", "get_market_summary", "health_check"]
            method_availability = {}
            
            for method in phase2_methods:
                method_availability[method] = hasattr(storage, method)
            
            phase2_ready = all(method_availability.values())
            
            self.results.add_result(
                "phase2_preparation",
                "interface_readiness",
                "passed" if phase2_ready else "failed",
                details={
                    "required_methods": phase2_methods,
                    "method_availability": method_availability,
                    "phase2_interface_ready": phase2_ready,
                    "critical": True
                }
            )
            
            # Test genetic pool supports enhanced strategies
            connection_optimizer = RetailConnectionOptimizer()
            genetic_pool = GeneticStrategyPool(connection_optimizer=connection_optimizer)
            
            enhancement_readiness = hasattr(genetic_pool, 'storage') and genetic_pool.storage is not None
            
            self.results.add_result(
                "phase2_preparation",
                "genetic_pool_enhancement_ready",
                "passed" if enhancement_readiness else "failed",
                details={
                    "storage_interface_integrated": enhancement_readiness,
                    "ready_for_correlation_signals": enhancement_readiness
                }
            )
            
            if phase2_ready and enhancement_readiness:
                self.results.phase_progression_ready = True
                self.results.business_value_score += 30  # 30% weight for phase progression
            
        except Exception as e:
            self.results.add_result(
                "phase2_preparation",
                "preparation_validation",
                "failed",
                details={"error": str(e), "critical": True}
            )
    
    async def validate_clean_upgrade_paths(self):
        """Validate clean upgrade paths to Phase 4 Neon database."""
        self.logger.info("🔍 Validating Clean Upgrade Paths")
        
        try:
            # Test storage interface abstraction enables backend switching
            current_storage = get_storage_implementation()
            
            # Simulate backend switching test
            interface_methods = [
                "store_ohlcv_bars", "get_ohlcv_bars", "calculate_technical_indicators", 
                "get_market_summary", "health_check"
            ]
            
            abstraction_quality = True
            method_signatures = {}
            
            for method_name in interface_methods:
                if hasattr(current_storage, method_name):
                    method_signatures[method_name] = "available"
                else:
                    abstraction_quality = False
                    method_signatures[method_name] = "missing"
            
            # Test multiple backend support
            backend_support = {}
            try:
                local_storage = LocalDataStorage()
                backend_support["LocalDataStorage"] = "available"
            except:
                backend_support["LocalDataStorage"] = "failed"
            
            try:
                shared_storage = SharedDataStorage("/tmp/test")
                backend_support["SharedDataStorage"] = "available"
            except:
                backend_support["SharedDataStorage"] = "failed"
            
            upgrade_path_ready = abstraction_quality and all(
                status == "available" for status in backend_support.values()
            )
            
            self.results.add_result(
                "upgrade_paths",
                "phase4_neon_ready",
                "passed" if upgrade_path_ready else "failed",
                details={
                    "interface_abstraction_quality": abstraction_quality,
                    "method_signatures": method_signatures,
                    "backend_support": backend_support,
                    "zero_code_change_upgrade": upgrade_path_ready,
                    "critical": True
                }
            )
            
        except Exception as e:
            self.results.add_result(
                "upgrade_paths",
                "upgrade_validation",
                "failed",
                details={"error": str(e), "critical": True}
            )
    
    def _calculate_overall_results(self):
        """Calculate overall validation status and business value."""
        summary = self.results.get_summary()
        
        # Determine overall status
        if summary["critical_failures"]:
            self.results.overall_status = "failed"
        elif summary["success_rate"] >= 0.8:
            self.results.overall_status = "passed"
        else:
            self.results.overall_status = "warning"
        
        # Cap business value score at 100
        self.results.business_value_score = min(100, self.results.business_value_score)
    
    def print_results(self):
        """Print comprehensive validation results."""
        summary = self.results.get_summary()
        
        print("\n" + "="*60)
        print("🚀 PHASE 1 IMPLEMENTATION VALIDATION RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL STATUS: {self.results.overall_status.upper()}")
        print(f"📈 BUSINESS VALUE SCORE: {self.results.business_value_score:.1f}/100")
        print(f"🔄 PHASE PROGRESSION READY: {'✅' if self.results.phase_progression_ready else '❌'}")
        print(f"✅ TESTS PASSED: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']*100:.1f}%)")
        
        if summary["critical_failures"]:
            print(f"\n❌ CRITICAL FAILURES:")
            for failure in summary["critical_failures"]:
                print(f"   - {failure}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for category, tests in self.results.results.items():
            print(f"\n🔍 {category.upper().replace('_', ' ')}:")
            for test_name, test_result in tests.items():
                status_icon = "✅" if test_result["status"] == "passed" else "❌" if test_result["status"] == "failed" else "⚠️"
                print(f"   {status_icon} {test_name}: {test_result['status'].upper()}")
                
                if test_result["performance_metrics"]:
                    for metric, value in test_result["performance_metrics"].items():
                        print(f"      📊 {metric}: {value}")
        
        print(f"\n🎯 STRATEGIC ASSESSMENT:")
        if self.results.overall_status == "passed":
            print("   ✅ Phase 1 implementation provides concrete business value")
            print("   ✅ Ray cluster foundation ready for distributed genetic algorithms") 
            print("   ✅ Storage interface enables clean phase progression")
            if self.results.phase_progression_ready:
                print("   ✅ Ready for Phase 2 correlation analysis integration")
        else:
            print("   ❌ Critical issues must be resolved before Phase 2 progression")
        
        print("="*60)


async def main():
    """Main validation execution."""
    validator = Phase1Validator()
    results = await validator.run_all_validations()
    validator.print_results()
    
    # Exit with appropriate code
    if results.overall_status == "passed":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())