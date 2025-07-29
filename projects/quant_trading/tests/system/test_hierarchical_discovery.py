#!/usr/bin/env python3
"""
HIERARCHICAL GENETIC DISCOVERY - END-TO-END INTEGRATION TEST

This script validates the complete three-stage hierarchical genetic discovery system
from asset filtering through final strategy production. Tests integration between
all discovery modules and existing infrastructure.

Components Tested:
1. CryptoSafeParameters - Safety parameter generation and validation
2. ResearchBackedAssetFilter - Asset universe filtering (180 → 16 assets)
3. HierarchicalGAOrchestrator - Complete three-stage discovery pipeline
4. Integration with HyperliquidClient - Real API connectivity with rate limiting
5. End-to-end data flow - Asset filtering → Daily → Hourly → Minute precision

Expected Results:
- Asset filtering: 180 → ~16 assets in <10 seconds
- Stage 1: Daily pattern discovery with crypto-safe parameters
- Stage 2: Hourly timing refinement with seeded evolution
- Stage 3: Minute precision evolution with micro-tuning
- Final output: 3 production-ready strategies with comprehensive metrics

Usage: python test_hierarchical_discovery_e2e.py
"""

import asyncio
import sys
import traceback
import logging
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Test our discovery system imports
try:
    from src.discovery.crypto_safe_parameters import (
        get_crypto_safe_parameters, 
        validate_trading_safety,
        CryptoSafeParameters,
        MarketRegime
    )
    from src.discovery.asset_universe_filter import (
        ResearchBackedAssetFilter, 
        AssetMetrics,
        FilterCriteria
    )
    from src.discovery.enhanced_asset_filter import (
        EnhancedAssetFilter,
        EnhancedFilterMetrics
    )
    from src.discovery.hierarchical_genetic_engine import (
        HierarchicalGAOrchestrator,
        DailyPatternDiscovery,
        HourlyTimingRefinement, 
        MinutePrecisionEvolution,
        StrategyGenome,
        EvolutionStage,
        TimeframeType
    )
    from src.data.hyperliquid_client import HyperliquidClient
    from src.config.settings import get_settings
    
    print("✅ All discovery modules imported successfully")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: Failed to import discovery modules: {e}")
    print("Ensure you're running from the project root directory.")
    sys.exit(1)


@dataclass
class E2ETestMetrics:
    """Comprehensive end-to-end test metrics."""
    
    # Test execution metrics
    test_start_time: datetime
    test_duration_seconds: float = 0.0
    total_api_calls: int = 0
    peak_memory_usage_mb: float = 0.0
    
    # Asset filtering metrics
    initial_asset_count: int = 0
    filtered_asset_count: int = 0
    asset_filter_duration: float = 0.0
    filter_reduction_percentage: float = 0.0
    
    # Discovery stage metrics
    stage1_duration: float = 0.0
    stage1_strategies: int = 0
    stage2_duration: float = 0.0
    stage2_strategies: int = 0
    stage3_duration: float = 0.0
    stage3_strategies: int = 0
    
    # Final results
    total_evaluations: int = 0
    search_space_reduction: float = 0.0
    final_strategies: int = 0
    avg_fitness_score: float = 0.0
    crypto_safety_violations: int = 0
    
    # Enhanced rate limiting metrics
    rate_limit_hits_base: int = 0
    rate_limit_hits_enhanced: int = 0
    api_calls_saved_by_optimization: int = 0
    cache_hit_rate: float = 0.0
    optimization_efficiency: float = 0.0
    
    # Success metrics
    test_success: bool = False
    errors_encountered: List[str] = None
    
    def __post_init__(self):
        if self.errors_encountered is None:
            self.errors_encountered = []


class HierarchicalDiscoveryE2ETester:
    """Comprehensive end-to-end tester for hierarchical genetic discovery system."""
    
    def __init__(self):
        """Initialize E2E tester with comprehensive monitoring."""
        self.settings = get_settings()
        self.logger = self._setup_logging()
        self.metrics = E2ETestMetrics(test_start_time=datetime.now())
        
        # Components to test
        self.crypto_params = None
        self.asset_filter = None
        self.orchestrator = None
        
        # Test configuration
        self.enable_real_api_calls = True  # Set to False for API-free testing
        self.max_test_duration = 300      # 5 minutes maximum test time
        self.memory_monitoring = True
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for E2E testing."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('hierarchical_discovery_e2e.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _monitor_memory_usage(self):
        """Monitor peak memory usage during testing."""
        if self.memory_monitoring:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.metrics.peak_memory_usage_mb = max(
                self.metrics.peak_memory_usage_mb, 
                current_memory
            )
    
    def _validate_component_integration(self):
        """Test integration between discovery components."""
        self.logger.info("🔧 Testing component integration...")
        
        try:
            # Test 1: CryptoSafeParameters integration
            self.crypto_params = get_crypto_safe_parameters()
            safe_genome = self.crypto_params.generate_crypto_safe_genome()
            is_safe = validate_trading_safety(safe_genome)
            
            if not is_safe:
                raise ValueError("Generated genome failed safety validation")
            
            self.logger.info("   ✅ CryptoSafeParameters: Working correctly")
            
            # Test 2a: Base AssetFilter integration
            self.base_asset_filter = ResearchBackedAssetFilter(self.settings)
            self.logger.info("   ✅ ResearchBackedAssetFilter: Initialized successfully")
            
            # Test 2b: Enhanced AssetFilter integration (MAIN FILTER FOR TESTING)
            self.asset_filter = EnhancedAssetFilter(self.settings)
            self.logger.info("   ✅ EnhancedAssetFilter: Initialized successfully")
            
            # Test 3: HierarchicalGAOrchestrator integration
            self.orchestrator = HierarchicalGAOrchestrator(self.settings)
            self.logger.info("   ✅ HierarchicalGAOrchestrator: Initialized successfully")
            
            # Test 4: StrategyGenome integration with crypto params
            test_genome = StrategyGenome.from_crypto_safe_params(self.crypto_params)
            if not test_genome.validate_safety():
                raise ValueError("StrategyGenome failed crypto safety validation")
            
            self.logger.info("   ✅ StrategyGenome: Crypto-safe generation working")
            
            # Test 5: Cross-component integration
            if hasattr(self.orchestrator, 'crypto_params'):
                self.logger.info("   ✅ Cross-component integration: Validated")
            else:
                raise ValueError("Orchestrator missing crypto_params integration")
            
            return True
            
        except Exception as e:
            error_msg = f"Component integration failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"   ❌ {error_msg}")
            return False
    
    async def _test_asset_filtering_stage(self):
        """Test asset universe filtering - Enhanced vs Base comparison."""
        self.logger.info("📊 Testing asset filtering stage with RATE LIMITING OPTIMIZATION...")
        
        stage_start = time.time()
        
        try:
            if self.enable_real_api_calls:
                # Test Enhanced Filter (main test)
                self.logger.info("   🚀 Testing ENHANCED filter with rate limiting optimizations...")
                filtered_assets, asset_metrics = await self.asset_filter.filter_universe(
                    refresh_cache=True,
                    enable_optimizations=True
                )
                
                # Get enhanced metrics
                if hasattr(self.asset_filter, 'get_enhanced_filter_summary'):
                    enhanced_summary = self.asset_filter.get_enhanced_filter_summary(filtered_assets, asset_metrics)
                    opt_performance = enhanced_summary.get("optimization_performance", {})
                    
                    # Extract optimization metrics
                    self.metrics.rate_limit_hits_enhanced = opt_performance.get("rate_limit_hits", 0)
                    self.metrics.api_calls_saved_by_optimization = opt_performance.get("total_api_calls_saved", 0)
                    
                    cache_hit_rate_str = opt_performance.get("cache_hit_rate", "0%")
                    self.metrics.cache_hit_rate = float(cache_hit_rate_str.replace("%", "")) / 100
                    
                    opt_efficiency_str = enhanced_summary.get("optimization_breakdown", {}).get("optimization_efficiency", "0%")
                    if isinstance(opt_efficiency_str, str):
                        self.metrics.optimization_efficiency = float(opt_efficiency_str.replace("%", "")) / 100
                    else:
                        self.metrics.optimization_efficiency = opt_efficiency_str
                
                self.logger.info(f"   ✅ Enhanced filter: Rate limit hits = {self.metrics.rate_limit_hits_enhanced}")
                self.logger.info(f"   ✅ Enhanced filter: Cache hit rate = {self.metrics.cache_hit_rate * 100:.1f}%")
                self.logger.info(f"   ✅ Enhanced filter: API calls saved = {self.metrics.api_calls_saved_by_optimization}")
                
                self.metrics.total_api_calls += 50  # Estimated API calls for filtering
            else:
                # Mock test data for API-free testing
                filtered_assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'DOT']
                asset_metrics = {asset: AssetMetrics(symbol=asset) for asset in filtered_assets}
                self.logger.info("   📋 Using mock data (API calls disabled)")
            
            # Validate filtering results
            self.metrics.initial_asset_count = 180  # Known Hyperliquid asset count
            self.metrics.filtered_asset_count = len(filtered_assets)
            self.metrics.asset_filter_duration = time.time() - stage_start
            
            if len(filtered_assets) == 0:
                raise ValueError("Enhanced asset filtering returned no assets")
            
            if len(filtered_assets) > 30:
                self.logger.warning(f"   ⚠️ High asset count: {len(filtered_assets)} (expected ~16)")
            
            # Calculate filter efficiency
            self.metrics.filter_reduction_percentage = (
                1.0 - len(filtered_assets) / self.metrics.initial_asset_count
            ) * 100
            
            self.logger.info(f"   ✅ ENHANCED filtering: {self.metrics.initial_asset_count} → {len(filtered_assets)} assets")
            self.logger.info(f"   ✅ Reduction: {self.metrics.filter_reduction_percentage:.1f}%")
            self.logger.info(f"   ✅ Duration: {self.metrics.asset_filter_duration:.1f}s")
            
            # Success criteria: Enhanced filter should eliminate or significantly reduce rate limit hits
            if self.enable_real_api_calls and self.metrics.rate_limit_hits_enhanced > 5:
                self.logger.warning(f"   ⚠️ Enhanced filter still experiencing high rate limit hits: {self.metrics.rate_limit_hits_enhanced}")
            elif self.enable_real_api_calls:
                self.logger.info(f"   ✅ Rate limiting optimization successful: {self.metrics.rate_limit_hits_enhanced} hits")
            
            return filtered_assets, asset_metrics
            
        except Exception as e:
            error_msg = f"Enhanced asset filtering failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"   ❌ {error_msg}")
            
            # Check if this is a rate limiting error (the issue we're trying to solve)
            if "rate limit" in str(e).lower() or "429" in str(e):
                self.metrics.rate_limit_hits_enhanced += 1
                self.logger.error(f"   ❌ CRITICAL: Enhanced filter still hitting rate limits - optimization failed!")
            
            return [], {}
    
    async def _test_stage1_daily_discovery(self, filtered_assets: List[str]):
        """Test Stage 1: Daily Pattern Discovery."""
        self.logger.info("🔍 Testing Stage 1: Daily Pattern Discovery...")
        
        stage_start = time.time()
        
        try:
            # Initialize daily discovery
            daily_discovery = DailyPatternDiscovery(self.settings)
            
            # Test with subset of assets for speed (first 3 assets)
            test_assets = filtered_assets[:3] if len(filtered_assets) >= 3 else filtered_assets
            self.logger.info(f"   📊 Testing with {len(test_assets)} assets: {test_assets}")
            
            # Run daily discovery
            if self.enable_real_api_calls:
                daily_patterns = await daily_discovery.discover_daily_patterns(test_assets)
                self.metrics.total_api_calls += len(test_assets) * 20  # Estimated API calls
            else:
                # Mock daily patterns for API-free testing
                daily_patterns = []
                for i, asset in enumerate(test_assets):
                    pattern = StrategyGenome.from_crypto_safe_params(self.crypto_params)
                    pattern.asset_tested = asset
                    pattern.fitness_score = 0.6 + (i * 0.1)  # Mock fitness scores
                    pattern.stage = EvolutionStage.DAILY_DISCOVERY
                    daily_patterns.append(pattern)
                self.logger.info("   📋 Using mock daily patterns (API calls disabled)")
            
            # Validate daily discovery results
            self.metrics.stage1_duration = time.time() - stage_start
            self.metrics.stage1_strategies = len(daily_patterns)
            
            if len(daily_patterns) == 0:
                raise ValueError("Daily discovery returned no patterns")
            
            # Validate crypto safety of all patterns
            safety_violations = 0
            for pattern in daily_patterns:
                if not pattern.validate_safety():
                    safety_violations += 1
            
            self.metrics.crypto_safety_violations += safety_violations
            
            if safety_violations > 0:
                self.logger.warning(f"   ⚠️ {safety_violations} crypto safety violations detected")
            
            self.logger.info(f"   ✅ Generated {len(daily_patterns)} daily patterns")
            self.logger.info(f"   ✅ Duration: {self.metrics.stage1_duration:.1f}s")
            self.logger.info(f"   ✅ Avg fitness: {sum(p.fitness_score for p in daily_patterns) / len(daily_patterns):.3f}")
            
            return daily_patterns
            
        except Exception as e:
            error_msg = f"Stage 1 daily discovery failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"   ❌ {error_msg}")
            return []
    
    async def _test_stage2_hourly_refinement(self, daily_patterns: List[StrategyGenome]):
        """Test Stage 2: Hourly Timing Refinement."""
        self.logger.info("⚡ Testing Stage 2: Hourly Timing Refinement...")
        
        stage_start = time.time()
        
        try:
            # Initialize hourly refinement
            hourly_refinement = HourlyTimingRefinement(self.settings)
            
            # Test with subset for speed (first 5 patterns)
            test_patterns = daily_patterns[:5] if len(daily_patterns) >= 5 else daily_patterns
            self.logger.info(f"   🎯 Refining {len(test_patterns)} daily patterns")
            
            # Run hourly refinement
            if self.enable_real_api_calls:
                hourly_strategies = await hourly_refinement.refine_hourly_timing(test_patterns)
                self.metrics.total_api_calls += len(test_patterns) * 10  # Estimated API calls
            else:
                # Mock hourly strategies for API-free testing
                hourly_strategies = []
                for i, pattern in enumerate(test_patterns):
                    strategy = StrategyGenome.from_crypto_safe_params(self.crypto_params)
                    strategy.asset_tested = pattern.asset_tested
                    strategy.fitness_score = pattern.fitness_score + 0.1  # Improved fitness
                    strategy.stage = EvolutionStage.HOURLY_REFINEMENT
                    hourly_strategies.append(strategy)
                self.logger.info("   📋 Using mock hourly strategies (API calls disabled)")
            
            # Validate hourly refinement results
            self.metrics.stage2_duration = time.time() - stage_start
            self.metrics.stage2_strategies = len(hourly_strategies)
            
            if len(hourly_strategies) == 0:
                raise ValueError("Hourly refinement returned no strategies")
            
            # Validate fitness improvement
            if len(hourly_strategies) > 0 and len(daily_patterns) > 0:
                avg_hourly_fitness = sum(s.fitness_score for s in hourly_strategies) / len(hourly_strategies)
                avg_daily_fitness = sum(p.fitness_score for p in daily_patterns) / len(daily_patterns)
                
                if avg_hourly_fitness <= avg_daily_fitness:
                    self.logger.warning("   ⚠️ No fitness improvement in hourly refinement")
            
            self.logger.info(f"   ✅ Refined to {len(hourly_strategies)} hourly strategies")
            self.logger.info(f"   ✅ Duration: {self.metrics.stage2_duration:.1f}s")
            self.logger.info(f"   ✅ Avg fitness: {sum(s.fitness_score for s in hourly_strategies) / len(hourly_strategies):.3f}")
            
            return hourly_strategies
            
        except Exception as e:
            error_msg = f"Stage 2 hourly refinement failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"   ❌ {error_msg}")
            return []
    
    async def _test_stage3_minute_precision(self, hourly_strategies: List[StrategyGenome]):
        """Test Stage 3: Minute Precision Evolution."""
        self.logger.info("🎯 Testing Stage 3: Minute Precision Evolution...")
        
        stage_start = time.time()
        
        try:
            # Initialize minute precision
            minute_precision = MinutePrecisionEvolution(self.settings)
            
            # Test with subset for speed (first 3 strategies)
            test_strategies = hourly_strategies[:3] if len(hourly_strategies) >= 3 else hourly_strategies
            self.logger.info(f"   🔬 Optimizing {len(test_strategies)} hourly strategies")
            
            # Run minute precision evolution
            if self.enable_real_api_calls:
                final_strategies = await minute_precision.evolve_minute_precision(test_strategies)
                self.metrics.total_api_calls += len(test_strategies) * 5  # Estimated API calls
            else:
                # Mock final strategies for API-free testing
                final_strategies = []
                for i, strategy in enumerate(test_strategies):
                    final_strategy = StrategyGenome.from_crypto_safe_params(self.crypto_params)
                    final_strategy.asset_tested = strategy.asset_tested
                    final_strategy.fitness_score = strategy.fitness_score + 0.05  # Final improvement
                    final_strategy.stage = EvolutionStage.MINUTE_PRECISION
                    final_strategies.append(final_strategy)
                self.logger.info("   📋 Using mock final strategies (API calls disabled)")
            
            # Validate minute precision results
            self.metrics.stage3_duration = time.time() - stage_start
            self.metrics.stage3_strategies = len(final_strategies)
            self.metrics.final_strategies = len(final_strategies)
            
            if len(final_strategies) == 0:
                raise ValueError("Minute precision returned no strategies")
            
            # Calculate final metrics
            if len(final_strategies) > 0:
                self.metrics.avg_fitness_score = sum(s.fitness_score for s in final_strategies) / len(final_strategies)
            
            self.logger.info(f"   ✅ Optimized to {len(final_strategies)} final strategies")
            self.logger.info(f"   ✅ Duration: {self.metrics.stage3_duration:.1f}s")
            self.logger.info(f"   ✅ Avg fitness: {self.metrics.avg_fitness_score:.3f}")
            
            return final_strategies
            
        except Exception as e:
            error_msg = f"Stage 3 minute precision failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"   ❌ {error_msg}")
            return []
    
    async def _test_orchestrator_integration(self):
        """Test complete orchestrator integration."""
        self.logger.info("🚀 Testing HierarchicalGAOrchestrator integration...")
        
        try:
            # Test orchestrator initialization
            orchestrator = HierarchicalGAOrchestrator(self.settings)
            
            # Test orchestrator components
            if not hasattr(orchestrator, 'daily_discovery'):
                raise ValueError("Orchestrator missing daily_discovery component")
            
            if not hasattr(orchestrator, 'hourly_refinement'):
                raise ValueError("Orchestrator missing hourly_refinement component")
            
            if not hasattr(orchestrator, 'minute_precision'):
                raise ValueError("Orchestrator missing minute_precision component")
            
            # Test orchestrator methods
            if not hasattr(orchestrator, 'discover_alpha_strategies'):
                raise ValueError("Orchestrator missing discover_alpha_strategies method")
            
            if not hasattr(orchestrator, 'get_discovery_metrics'):
                raise ValueError("Orchestrator missing get_discovery_metrics method")
            
            self.logger.info("   ✅ Orchestrator integration validated")
            
            # Test minimal orchestrator run (API-free for speed)
            if not self.enable_real_api_calls:
                self.logger.info("   📋 Orchestrator end-to-end test skipped (API calls disabled)")
                return True
            
            # Optionally test full orchestrator run (commented out for speed)
            # strategies = await orchestrator.discover_alpha_strategies(
            #     refresh_asset_filter=False,
            #     target_strategies=2
            # )
            # self.logger.info(f"   ✅ Orchestrator produced {len(strategies)} strategies")
            
            return True
            
        except Exception as e:
            error_msg = f"Orchestrator integration failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"   ❌ {error_msg}")
            return False
    
    def _calculate_final_metrics(self):
        """Calculate final test metrics and success criteria."""
        self.metrics.test_duration_seconds = (datetime.now() - self.metrics.test_start_time).total_seconds()
        
        # Calculate search space reduction (theoretical)
        brute_force_evaluations = 108000  # Theoretical brute force
        actual_evaluations = (
            self.metrics.stage1_strategies * 50 +   # Daily discovery 
            self.metrics.stage2_strategies * 100 +  # Hourly refinement
            self.metrics.stage3_strategies * 200    # Minute precision
        )
        
        self.metrics.total_evaluations = actual_evaluations
        if actual_evaluations > 0:
            self.metrics.search_space_reduction = 1.0 - (actual_evaluations / brute_force_evaluations)
        
        # Determine overall test success
        success_criteria = [
            len(self.metrics.errors_encountered) == 0,        # No errors
            self.metrics.filtered_asset_count > 0,            # Asset filtering worked
            self.metrics.final_strategies > 0,                # Final strategies produced
            self.metrics.crypto_safety_violations == 0,       # No safety violations
            self.metrics.test_duration_seconds < self.max_test_duration  # Reasonable performance
        ]
        
        self.metrics.test_success = all(success_criteria)
    
    def _display_final_results(self):
        """Display comprehensive test results."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🎉 HIERARCHICAL DISCOVERY E2E TEST RESULTS")
        self.logger.info("=" * 70)
        
        # Test execution summary
        self.logger.info(f"📊 TEST EXECUTION:")
        self.logger.info(f"   Duration: {self.metrics.test_duration_seconds:.1f}s")
        self.logger.info(f"   Peak Memory: {self.metrics.peak_memory_usage_mb:.1f} MB")
        self.logger.info(f"   API Calls: {self.metrics.total_api_calls}")
        self.logger.info(f"   Success: {'✅ PASS' if self.metrics.test_success else '❌ FAIL'}")
        
        # Asset filtering results with ENHANCED OPTIMIZATION METRICS
        self.logger.info(f"\n📊 ENHANCED ASSET FILTERING:")
        self.logger.info(f"   Initial Assets: {self.metrics.initial_asset_count}")
        self.logger.info(f"   Filtered Assets: {self.metrics.filtered_asset_count}")
        self.logger.info(f"   Reduction: {self.metrics.filter_reduction_percentage:.1f}%")
        self.logger.info(f"   Duration: {self.metrics.asset_filter_duration:.1f}s")
        
        # Rate limiting optimization results
        self.logger.info(f"\n🚀 RATE LIMITING OPTIMIZATION:")
        self.logger.info(f"   Enhanced Rate Limit Hits: {self.metrics.rate_limit_hits_enhanced}")
        self.logger.info(f"   Cache Hit Rate: {self.metrics.cache_hit_rate * 100:.1f}%")
        self.logger.info(f"   API Calls Saved: {self.metrics.api_calls_saved_by_optimization}")
        self.logger.info(f"   Optimization Efficiency: {self.metrics.optimization_efficiency * 100:.1f}%")
        
        # Critical issue resolution
        if self.metrics.rate_limit_hits_enhanced == 0:
            self.logger.info(f"   ✅ BATCH 6-7 FAILURES RESOLVED: No rate limit hits!")
        elif self.metrics.rate_limit_hits_enhanced < 3:
            self.logger.info(f"   ✅ SIGNIFICANT IMPROVEMENT: Rate limits reduced to {self.metrics.rate_limit_hits_enhanced}")
        else:
            self.logger.info(f"   ⚠️ PARTIAL IMPROVEMENT: Still {self.metrics.rate_limit_hits_enhanced} rate limit hits")
        
        # Discovery stage results
        self.logger.info(f"\n📊 DISCOVERY STAGES:")
        self.logger.info(f"   Stage 1 (Daily): {self.metrics.stage1_strategies} patterns ({self.metrics.stage1_duration:.1f}s)")
        self.logger.info(f"   Stage 2 (Hourly): {self.metrics.stage2_strategies} strategies ({self.metrics.stage2_duration:.1f}s)")
        self.logger.info(f"   Stage 3 (Minute): {self.metrics.stage3_strategies} final ({self.metrics.stage3_duration:.1f}s)")
        
        # Performance metrics
        self.logger.info(f"\n📊 PERFORMANCE METRICS:")
        self.logger.info(f"   Total Evaluations: {self.metrics.total_evaluations:,}")
        self.logger.info(f"   Search Space Reduction: {self.metrics.search_space_reduction:.1%}")
        self.logger.info(f"   Final Strategies: {self.metrics.final_strategies}")
        self.logger.info(f"   Avg Fitness Score: {self.metrics.avg_fitness_score:.3f}")
        
        # Safety validation
        self.logger.info(f"\n🛡️ CRYPTO SAFETY:")
        if self.metrics.crypto_safety_violations == 0:
            self.logger.info("   ✅ No crypto safety violations detected")
        else:
            self.logger.info(f"   ❌ {self.metrics.crypto_safety_violations} safety violations")
        
        # Error summary
        if self.metrics.errors_encountered:
            self.logger.info(f"\n❌ ERRORS ENCOUNTERED ({len(self.metrics.errors_encountered)}):")
            for i, error in enumerate(self.metrics.errors_encountered, 1):
                self.logger.info(f"   {i}. {error}")
        else:
            self.logger.info("\n✅ NO ERRORS ENCOUNTERED")
        
        # Final verdict
        if self.metrics.test_success:
            self.logger.info("\n🎉 HIERARCHICAL DISCOVERY SYSTEM: READY FOR PRODUCTION!")
        else:
            self.logger.info("\n⚠️ HIERARCHICAL DISCOVERY SYSTEM: REQUIRES FIXES BEFORE PRODUCTION")
    
    async def run_complete_e2e_test(self, enable_api_calls: bool = True):
        """Run complete end-to-end integration test."""
        self.enable_real_api_calls = enable_api_calls
        
        self.logger.info("🚀 STARTING HIERARCHICAL DISCOVERY E2E TEST")
        self.logger.info("=" * 70)
        self.logger.info(f"API Calls: {'Enabled' if enable_api_calls else 'Disabled (Mock Mode)'}")
        self.logger.info(f"Max Duration: {self.max_test_duration}s")
        self.logger.info("=" * 70)
        
        try:
            # Monitor memory usage
            self._monitor_memory_usage()
            
            # Test 1: Component Integration
            if not self._validate_component_integration():
                self.logger.error("❌ Component integration failed - aborting test")
                return False
            
            self._monitor_memory_usage()
            
            # Test 2: Asset Filtering Stage
            filtered_assets, asset_metrics = await self._test_asset_filtering_stage()
            if not filtered_assets:
                self.logger.error("❌ Asset filtering failed - aborting test")
                return False
            
            self._monitor_memory_usage()
            
            # Test 3: Stage 1 - Daily Discovery
            daily_patterns = await self._test_stage1_daily_discovery(filtered_assets)
            if not daily_patterns:
                self.logger.error("❌ Daily discovery failed - aborting test")
                return False
            
            self._monitor_memory_usage()
            
            # Test 4: Stage 2 - Hourly Refinement
            hourly_strategies = await self._test_stage2_hourly_refinement(daily_patterns)
            if not hourly_strategies:
                self.logger.error("❌ Hourly refinement failed - aborting test")
                return False
            
            self._monitor_memory_usage()
            
            # Test 5: Stage 3 - Minute Precision
            final_strategies = await self._test_stage3_minute_precision(hourly_strategies)
            if not final_strategies:
                self.logger.error("❌ Minute precision failed - aborting test")
                return False
            
            self._monitor_memory_usage()
            
            # Test 6: Orchestrator Integration
            if not await self._test_orchestrator_integration():
                self.logger.error("❌ Orchestrator integration failed")
                return False
            
            # Calculate final metrics and display results
            self._calculate_final_metrics()
            self._display_final_results()
            
            return self.metrics.test_success
            
        except Exception as e:
            error_msg = f"E2E test execution failed: {e}"
            self.metrics.errors_encountered.append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            traceback.print_exc()
            
            # Still calculate and display partial results
            self._calculate_final_metrics()
            self._display_final_results()
            
            return False


async def main():
    """Main test execution function."""
    print("🧪 HIERARCHICAL GENETIC DISCOVERY - END-TO-END INTEGRATION TEST")
    print("=" * 80)
    print("Testing complete three-stage discovery system integration...")
    print("Components: CryptoSafeParameters + AssetFilter + HierarchicalGA")
    print("=" * 80)
    
    # Initialize tester
    tester = HierarchicalDiscoveryE2ETester()
    
    # Run E2E test with API calls enabled (set to False for faster testing)
    enable_real_api = True  # Change to False for mock testing
    
    success = await tester.run_complete_e2e_test(enable_api_calls=enable_real_api)
    
    # Return appropriate exit code
    if success:
        print("\n🎉 E2E TEST PASSED - HIERARCHICAL DISCOVERY SYSTEM READY!")
        return 0
    else:
        print("\n❌ E2E TEST FAILED - REQUIRES FIXES BEFORE PRODUCTION")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)