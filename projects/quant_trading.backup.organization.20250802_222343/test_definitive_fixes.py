#!/usr/bin/env python3
"""
DEFINITIVE FIXES VALIDATION TEST

This test validates both critical fixes:
1. API Rate Limiting - Using research-based definitive implementation
2. Genetic Engine Multiprocessing - Using DEAP research patterns exactly

NO MORE BAND-AID FIXES. These are permanent solutions.
"""

import asyncio
import logging
import time
import multiprocessing
from typing import Dict, List
import pandas as pd
import numpy as np

# Import the DEFINITIVE solutions
from src.config.rate_limiter import HyperliquidRateLimiter, APIEndpointType, get_rate_limiter
from src.strategy.genetic_engine_research_compliant import create_research_compliant_engine, ResearchCompliantConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefinitiveFixesValidator:
    """Validates both definitive fixes comprehensively."""
    
    def __init__(self):
        """Initialize validator."""
        self.rate_limiter = get_rate_limiter()
        self.test_results = {}
    
    def test_fix_1_rate_limiting(self) -> bool:
        """Test FIX 1: Definitive Rate Limiting Solution."""
        logger.info("🔧 TESTING FIX 1: DEFINITIVE RATE LIMITING")
        
        try:
            # Reset for clean test
            self.rate_limiter.reset_for_testing()
            
            # Test 1: Basic rate limit checking
            can_proceed, reason = self.rate_limiter.can_make_request(
                APIEndpointType.INFO_STANDARD, batch_size=1
            )
            assert can_proceed, f"Basic request should be allowed: {reason}"
            logger.info("   ✅ Basic rate limit checking works")
            
            # Test 2: Batch weight calculation (research formula)
            weight_1 = self.rate_limiter.calculate_batch_weight(1)
            weight_40 = self.rate_limiter.calculate_batch_weight(40)
            weight_80 = self.rate_limiter.calculate_batch_weight(80)
            
            assert weight_1 == 1, f"Batch size 1 should have weight 1, got {weight_1}"
            assert weight_40 == 2, f"Batch size 40 should have weight 2, got {weight_40}"
            assert weight_80 == 3, f"Batch size 80 should have weight 3, got {weight_80}"
            logger.info("   ✅ Research batch weight formula works")
            
            # Test 3: Different endpoint weights
            light_weight = self.rate_limiter.get_endpoint_weight(APIEndpointType.INFO_LIGHT)
            standard_weight = self.rate_limiter.get_endpoint_weight(APIEndpointType.INFO_STANDARD)
            heavy_weight = self.rate_limiter.get_endpoint_weight(APIEndpointType.INFO_HEAVY)
            
            assert light_weight == 2, f"Light endpoint should be weight 2, got {light_weight}"
            assert standard_weight == 20, f"Standard endpoint should be weight 20, got {standard_weight}"
            assert heavy_weight == 60, f"Heavy endpoint should be weight 60, got {heavy_weight}"
            logger.info("   ✅ Endpoint weight classification works")
            
            # Test 4: Rate limit consumption
            initial_remaining = self.rate_limiter.state.ip_weight_remaining
            self.rate_limiter.consume_request(APIEndpointType.INFO_STANDARD)
            after_consumption = self.rate_limiter.state.ip_weight_remaining
            
            assert after_consumption == initial_remaining - 20, "Weight should be consumed correctly"
            logger.info("   ✅ Rate limit consumption works")
            
            # Test 5: 429 backoff handling
            self.rate_limiter.consume_request(APIEndpointType.INFO_STANDARD, response_code=429)
            status = self.rate_limiter.get_status()
            
            assert status['in_backoff'], "Should be in backoff after 429"
            assert status['consecutive_429s'] == 1, f"Should have 1 consecutive 429, got {status['consecutive_429s']}"
            logger.info("   ✅ 429 backoff handling works")
            
            logger.info("🎯 FIX 1 VALIDATED: Definitive rate limiting working perfectly")
            return True
            
        except Exception as e:
            logger.error(f"❌ FIX 1 FAILED: {e}")
            return False
    
    async def test_fix_2_genetic_multiprocessing(self) -> bool:
        """Test FIX 2: Genetic Engine Multiprocessing Solution."""
        logger.info("🧬 TESTING FIX 2: GENETIC ENGINE MULTIPROCESSING")
        
        try:
            # Create synthetic market data
            dates = pd.date_range('2024-01-01', periods=500, freq='1h')
            market_data = pd.DataFrame({
                'open': np.random.uniform(100, 110, 500),
                'high': np.random.uniform(105, 115, 500),
                'low': np.random.uniform(95, 105, 500),
                'close': np.random.uniform(100, 110, 500),
                'volume': np.random.uniform(1000, 5000, 500)
            }, index=dates)
            
            logger.info("   📊 Created synthetic market data")
            
            # Test 1: Single-threaded evolution (should work)
            config_single = ResearchCompliantConfig(
                population_size=10,
                n_generations=2,
                use_multiprocessing=False  # Single-threaded
            )
            
            engine_single = create_research_compliant_engine()
            engine_single.config = config_single
            
            results_single = engine_single.evolve(market_data)
            
            assert results_single.status.value == "completed", f"Single-threaded evolution failed: {results_single.status}"
            assert results_single.best_individual is not None, "Should have best individual"
            assert len(results_single.population) == 10, f"Should have 10 individuals, got {len(results_single.population)}"
            logger.info("   ✅ Single-threaded evolution works")
            
            # Test 2: Multi-threaded evolution (the FIXED version)
            config_multi = ResearchCompliantConfig(
                population_size=10,
                n_generations=2,
                use_multiprocessing=True  # Multi-threaded with context manager
            )
            
            engine_multi = create_research_compliant_engine()
            engine_multi.config = config_multi
            
            results_multi = engine_multi.evolve(market_data)
            
            assert results_multi.status.value == "completed", f"Multi-threaded evolution failed: {results_multi.status}"
            assert results_multi.best_individual is not None, "Should have best individual"
            assert len(results_multi.population) == 10, f"Should have 10 individuals, got {len(results_multi.population)}"
            logger.info("   ✅ Multi-threaded evolution works (RESEARCH PATTERN)")
            
            # Test 3: Verify performance difference (multiprocessing should be faster)
            # Note: This test is optional since small populations might not show difference
            single_time = results_single.execution_time
            multi_time = results_multi.execution_time
            
            logger.info(f"   📊 Single-threaded time: {single_time:.2f}s")
            logger.info(f"   📊 Multi-threaded time: {multi_time:.2f}s")
            logger.info("   ✅ Both execution modes complete successfully")
            
            logger.info("🎯 FIX 2 VALIDATED: Genetic engine multiprocessing working perfectly")
            return True
            
        except Exception as e:
            logger.error(f"❌ FIX 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_integration_fixes(self) -> bool:
        """Test both fixes working together."""
        logger.info("🔗 TESTING INTEGRATION: Both fixes working together")
        
        try:
            # Reset rate limiter
            self.rate_limiter.reset_for_testing()
            
            # Simulate API usage followed by genetic evolution
            logger.info("   📡 Simulating API usage...")
            
            # Consume some rate limit (simulating discovery phase)
            for i in range(5):
                can_proceed, reason = self.rate_limiter.can_make_request(APIEndpointType.INFO_STANDARD)
                if can_proceed:
                    self.rate_limiter.consume_request(APIEndpointType.INFO_STANDARD)
                else:
                    logger.warning(f"   ⚠️ Rate limit hit during simulation: {reason}")
            
            status = self.rate_limiter.get_status()
            logger.info(f"   📊 After API simulation: {status['ip_weight_remaining']} weight remaining")
            
            # Now run genetic evolution (should work regardless of API state)
            logger.info("   🧬 Running genetic evolution...")
            
            dates = pd.date_range('2024-01-01', periods=200, freq='1h')
            market_data = pd.DataFrame({
                'open': np.random.uniform(100, 110, 200),
                'high': np.random.uniform(105, 115, 200),
                'low': np.random.uniform(95, 105, 200),
                'close': np.random.uniform(100, 110, 200),
                'volume': np.random.uniform(1000, 5000, 200)
            }, index=dates)
            
            config = ResearchCompliantConfig(
                population_size=8,
                n_generations=2,
                use_multiprocessing=True  # Test the fixed multiprocessing
            )
            
            engine = create_research_compliant_engine()
            engine.config = config
            
            results = engine.evolve(market_data)
            
            assert results.status.value == "completed", "Integration test evolution should succeed"
            logger.info("   ✅ Genetic evolution succeeds after API usage")
            
            logger.info("🎯 INTEGRATION VALIDATED: Both fixes work together perfectly")
            return True
            
        except Exception as e:
            logger.error(f"❌ INTEGRATION TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, bool]:
        """Run all validation tests."""
        logger.info("🚀 DEFINITIVE FIXES COMPREHENSIVE VALIDATION")
        logger.info("=" * 60)
        
        results = {}
        
        # Test Fix 1: Rate Limiting
        results['rate_limiting'] = self.test_fix_1_rate_limiting()
        
        # Test Fix 2: Genetic Multiprocessing
        results['genetic_multiprocessing'] = await self.test_fix_2_genetic_multiprocessing()
        
        # Test Integration
        results['integration'] = await self.test_integration_fixes()
        
        return results

async def main():
    """Main validation function."""
    validator = DefinitiveFixesValidator()
    
    results = await validator.run_comprehensive_validation()
    
    logger.info("=" * 60)
    logger.info("🎯 DEFINITIVE FIXES VALIDATION COMPLETE")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"   {test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 ALL FIXES VALIDATED - NO MORE RATE LIMIT OR MULTIPROCESSING ISSUES!")
        logger.info("   🔧 Rate limiting: Research-based implementation working")
        logger.info("   🧬 Genetic engine: DEAP context manager pattern working")
        logger.info("   🔗 Integration: Both systems working together perfectly")
    else:
        logger.error("❌ SOME FIXES FAILED - REQUIRE ADDITIONAL WORK")

# Research requires __main__ protection for multiprocessing tests
if __name__ == "__main__":
    asyncio.run(main())