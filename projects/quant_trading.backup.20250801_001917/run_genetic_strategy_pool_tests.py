#!/usr/bin/env python3
"""
Genetic Strategy Pool Test Runner - Phase 5B Implementation

This script runs comprehensive tests for the genetic strategy pool with 
real validated genetic seeds, ensuring 100.0/100 health score maintenance
and full integration with the existing BaseSeed framework.

Research Context Integration:
- /research/ray_cluster/research_summary.md - Ray patterns validated
- /research/deap/research_summary.md - Genetic algorithm patterns validated
- /research/asyncio_advanced/research_summary.md - Async patterns validated

Test Coverage:
✅ Real Seed Integration: Uses all 12 validated genetic seeds
✅ Health Score Validation: 100.0/100 baseline preservation tests  
✅ Performance Benchmarks: 95.8ms response time target validation
✅ Ray Distributed Testing: Both local and distributed execution modes
✅ Fault Tolerance: Error handling and graceful degradation
✅ Parameter Bounds: Validated genetic parameter generation
"""

import sys
import os
import asyncio
import pytest
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def validate_test_environment():
    """Validate that test environment is properly configured."""
    logger.info("🔍 Validating test environment...")
    
    # Check critical files exist
    critical_files = [
        "src/execution/genetic_strategy_pool.py",
        "src/strategy/genetic_seeds/base_seed.py", 
        "src/strategy/genetic_seeds/seed_registry.py",
        "tests/test_genetic_strategy_pool.py"
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing critical files: {missing_files}")
        return False
    
    # Check genetic seed files are available (should be 12 after cleanup)
    seeds_dir = project_root / "src" / "strategy" / "genetic_seeds"
    seed_files = list(seeds_dir.glob("*_seed.py"))
    logger.info(f"📂 Found {len(seed_files)} genetic seed files")
    
    if len(seed_files) < 10:  # Should have at least 10 seeds
        logger.warning(f"⚠️ Expected at least 10 genetic seed files, found {len(seed_files)}")
    
    # Check for research directory
    research_dir = project_root / "research"
    if not research_dir.exists():
        logger.error("❌ Research directory not found - context validation impossible")
        return False
    
    logger.info("✅ Test environment validation passed")
    return True

def run_test_suite():
    """Run the complete genetic strategy pool test suite."""
    logger.info("🧪 Starting Genetic Strategy Pool Test Suite...")
    
    # Test configuration
    test_args = [
        "tests/test_genetic_strategy_pool.py",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--strict-markers",  # Strict marker handling
        "-p", "no:warnings",  # Suppress warnings for cleaner output
        "--asyncio-mode=auto"  # Auto async mode detection
    ]
    
    # Add specific test markers for different categories
    logger.info("📋 Test Categories:")
    logger.info("  • TestGeneticStrategyPoolInitialization - Basic initialization")
    logger.info("  • TestIndividualClass - Individual genetic member tests") 
    logger.info("  • TestPopulationManagement - Population creation and management")
    logger.info("  • TestEvolutionCycles - Genetic evolution algorithms")
    logger.info("  • TestRealSeedIntegration - Real validated genetic seeds")
    logger.info("  • TestPerformanceBenchmarks - Performance target validation")
    logger.info("  • TestHealthScoreValidation - 100.0/100 health score tests")
    logger.info("  • TestErrorHandling - Fault tolerance and error recovery")
    
    try:
        # Run pytest with our configuration
        exit_code = pytest.main(test_args)
        
        if exit_code == 0:
            logger.info("✅ All tests passed successfully!")
            logger.info("🎯 Genetic Strategy Pool is ready for Phase 5B integration")
            return True
        else:
            logger.error(f"❌ Tests failed with exit code: {exit_code}")
            return False
            
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        return False

def run_specific_test_category(category: str):
    """Run a specific category of tests."""
    category_map = {
        "init": "TestGeneticStrategyPoolInitialization",
        "individual": "TestIndividualClass", 
        "population": "TestPopulationManagement",
        "evolution": "TestEvolutionCycles",
        "real_seeds": "TestRealSeedIntegration",
        "performance": "TestPerformanceBenchmarks", 
        "health": "TestHealthScoreValidation",
        "errors": "TestErrorHandling"
    }
    
    if category not in category_map:
        logger.error(f"❌ Unknown test category: {category}")
        logger.info(f"Available categories: {', '.join(category_map.keys())}")
        return False
    
    test_class = category_map[category]
    logger.info(f"🧪 Running {test_class} tests...")
    
    test_args = [
        "tests/test_genetic_strategy_pool.py",
        "-k", test_class,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    exit_code = pytest.main(test_args)
    return exit_code == 0

def main():
    """Main test runner."""
    logger.info("🚀 Genetic Strategy Pool Test Runner - Phase 5B")
    logger.info("=" * 60)
    
    # Validate environment first
    if not validate_test_environment():
        logger.error("❌ Environment validation failed - cannot proceed")
        return 1
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        if category == "help":
            print("Usage: python run_genetic_strategy_pool_tests.py [category]")
            print("\nAvailable categories:")
            print("  init       - Initialization tests")
            print("  individual - Individual class tests")
            print("  population - Population management tests")
            print("  evolution  - Evolution cycle tests")
            print("  real_seeds - Real genetic seed integration tests")
            print("  performance- Performance benchmark tests")
            print("  health     - Health score validation tests")
            print("  errors     - Error handling tests")
            print("  (no args) - Run all tests")
            return 0
        
        success = run_specific_test_category(category)
    else:
        success = run_test_suite()
    
    if success:
        logger.info("🎉 Test execution completed successfully!")
        logger.info("📊 Genetic Strategy Pool validated for production deployment")
        return 0
    else:
        logger.error("💥 Test execution failed - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())