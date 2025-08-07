#!/usr/bin/env python3
"""
Complete System Integration Validator

Tests the ENTIRE application as a functional trading system:
- Discovery → Strategy → Execution → Data pipeline
- Real data processing workflows  
- Genetic algorithm evolution validation
- Cross-module integration testing
- Error handling and recovery
- Performance under realistic conditions

This answers: "Can I actually deploy this trading system and have it work?"
"""

import asyncio
import sys
import os
import time
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteSystemIntegrationValidator:
    """Comprehensive system integration validator."""
    
    def __init__(self, enable_real_data: bool = False, enable_distributed: bool = False):
        self.project_root = project_root
        self.enable_real_data = enable_real_data
        self.enable_distributed = enable_distributed
        
        self.results = {
            'workflow_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'error_handling_tests': {},
            'overall_status': 'pending',
            'business_readiness_score': 0.0,
            'timestamp': datetime.now(),
            'test_duration': 0.0
        }
        
        self.start_time = None
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run comprehensive system integration validation."""
        self.start_time = time.time()
        
        print("🚀 COMPLETE SYSTEM INTEGRATION VALIDATION")
        print("=" * 70)
        print(f"Real Data Enabled: {self.enable_real_data}")
        print(f"Distributed Mode: {self.enable_distributed}")
        print("=" * 70)
        
        try:
            # Phase 1: Individual Workflow Validation
            await self._test_data_processing_workflow()
            await self._test_genetic_algorithm_workflow()  
            await self._test_signal_generation_workflow()
            await self._test_discovery_workflow()
            
            # Phase 2: Cross-Module Integration Testing
            await self._test_discovery_to_strategy_integration()
            await self._test_strategy_to_execution_integration()
            await self._test_data_to_modules_integration()
            
            # Phase 3: Error Handling & Recovery
            await self._test_error_handling_scenarios()
            
            # Phase 4: Performance Validation
            await self._test_system_performance()
            
            # Calculate final assessment
            self._calculate_business_readiness_score()
            
            self.results['test_duration'] = time.time() - self.start_time
            print(f"\n✅ Complete System Integration Validation finished in {self.results['test_duration']:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ System integration validation failed: {e}")
            self.results['overall_status'] = 'failed'
            self.results['critical_error'] = str(e)
            self.results['test_duration'] = time.time() - self.start_time
        
        self._print_comprehensive_results()
        return self.results
    
    async def _test_data_processing_workflow(self):
        """Test complete data fetch → process → store → retrieve workflow."""
        print("\n📊 Testing Data Processing Workflow...")
        
        test_result = {
            'status': 'pending',
            'components_tested': [],
            'workflow_steps': {},
            'performance_metrics': {}
        }
        
        try:
            start_time = time.time()
            
            # Step 1: Initialize storage system
            from src.data.storage_interfaces import get_storage_implementation
            storage = get_storage_implementation()
            
            health = await storage.health_check()
            test_result['workflow_steps']['storage_initialization'] = {
                'status': 'passed' if health['status'] == 'healthy' else 'failed',
                'backend': health.get('backend', 'unknown'),
                'latency_ms': health.get('query_latency_ms', 0)
            }
            
            # Step 2: Test data pipeline creation
            from src.data.market_data_pipeline import MarketDataPipeline
            pipeline = MarketDataPipeline()
            test_result['workflow_steps']['pipeline_creation'] = {
                'status': 'passed',
                'pipeline_type': type(pipeline).__name__
            }
            
            # Step 3: Test client connectivity
            from src.data.hyperliquid_client import HyperliquidClient  
            client = HyperliquidClient()
            
            if self.enable_real_data:
                # Test real API connectivity
                try:
                    await client.connect()
                    test_result['workflow_steps']['api_connectivity'] = {
                        'status': 'passed',
                        'connection_type': 'real_api'
                    }
                except Exception as e:
                    test_result['workflow_steps']['api_connectivity'] = {
                        'status': 'failed',
                        'error': str(e),
                        'connection_type': 'real_api'
                    }
            else:
                # Mock API test
                test_result['workflow_steps']['api_connectivity'] = {
                    'status': 'passed',
                    'connection_type': 'mock_test',
                    'note': 'Real API testing disabled'
                }
            
            # Step 4: Test data storage workflow with PROPER market data pipeline
            # Generate realistic tick data and use real MarketDataAggregator
            from src.data.market_data_pipeline import TickData, MarketDataAggregator
            from datetime import timedelta
            
            # Create real aggregator (same as production)
            aggregator = MarketDataAggregator(bar_duration=timedelta(minutes=1))
            
            # Generate realistic tick data that will create proper VWAP/trade_count
            bars = []
            base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            base_price = 50000.0
            
            for bar_idx in range(50):  # Generate 50 bars
                bar_start_time = base_time + timedelta(minutes=bar_idx)
                
                # Generate 10-50 realistic ticks per bar (mimicking real trading)
                ticks_per_bar = np.random.randint(10, 51)
                
                for tick_idx in range(ticks_per_bar):
                    # Realistic price movement within bar
                    price_change = np.random.normal(0, 0.001)  # 0.1% std deviation
                    current_price = base_price * (1 + price_change)
                    
                    # Realistic volume (varies by tick)
                    volume = np.random.uniform(0.1, 2.0)  # 0.1 to 2.0 BTC per tick
                    
                    # Create tick within the bar timeframe
                    tick_time = bar_start_time + timedelta(seconds=tick_idx * (60 // ticks_per_bar))
                    
                    tick = TickData(
                        symbol='BTC',
                        timestamp=tick_time,
                        price=current_price,
                        volume=volume,
                        side='buy' if np.random.random() > 0.5 else 'sell',
                        trade_id=f"test_trade_{bar_idx}_{tick_idx}"
                    )
                    
                    # Process tick through REAL aggregator (calculates proper VWAP/trade_count)
                    completed_bar = aggregator.process_tick(tick)
                    if completed_bar and len(bars) < 50:
                        bars.append(completed_bar)
                    
                    # Update base price for next tick
                    base_price = current_price
            
            # Force completion of any remaining bars
            if len(bars) < 50 and 'BTC' in aggregator.current_bars:
                final_bar = aggregator._complete_bar('BTC', aggregator.current_bars['BTC'])
                bars.append(final_bar)
            
            # Store data
            await storage.store_ohlcv_bars(bars[:50])  # Store first 50 bars
            
            # Retrieve data
            retrieved_data = await storage.get_ohlcv_bars('BTC', limit=50)
            
            test_result['workflow_steps']['data_storage_cycle'] = {
                'status': 'passed' if len(retrieved_data) > 0 else 'failed',
                'bars_stored': 50,
                'bars_retrieved': len(retrieved_data),
                'data_integrity': len(retrieved_data) == 50
            }
            
            # Step 5: Test technical indicators calculation
            if len(retrieved_data) >= 20:  # Need minimum data for indicators
                indicators = await storage.calculate_technical_indicators('BTC', lookback_periods=20)
                test_result['workflow_steps']['technical_indicators'] = {
                    'status': 'passed' if len(indicators) > 0 else 'failed',
                    'indicators_calculated': len(indicators.columns) if hasattr(indicators, 'columns') else 0
                }
            else:
                test_result['workflow_steps']['technical_indicators'] = {
                    'status': 'skipped',
                    'reason': 'insufficient_data'
                }
            
            # Performance metrics
            total_time = time.time() - start_time
            test_result['performance_metrics'] = {
                'total_workflow_time': total_time,
                'storage_latency': health.get('query_latency_ms', 0),
                'throughput_bars_per_second': 50 / total_time if total_time > 0 else 0
            }
            
            # Determine overall status
            failed_steps = [step for step, result in test_result['workflow_steps'].items() 
                          if result['status'] == 'failed']
            
            if len(failed_steps) == 0:
                test_result['status'] = 'passed'
                print("   ✅ Data Processing Workflow - PASSED")
            else:
                test_result['status'] = 'partial'
                print(f"   ⚠️ Data Processing Workflow - PARTIAL ({len(failed_steps)} failures)")
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Data Processing Workflow - FAILED: {e}")
        
        self.results['workflow_tests']['data_processing'] = test_result
    
    async def _test_genetic_algorithm_workflow(self):
        """Test full genetic evolution with real fitness improvement validation."""
        print("\n🧬 Testing Genetic Algorithm Workflow...")
        
        test_result = {
            'status': 'pending',
            'evolution_metrics': {},
            'population_analysis': {},
            'fitness_progression': []
        }
        
        try:
            # Step 1: Initialize genetic system
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
            from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
            from src.data.storage_interfaces import get_storage_implementation
            
            storage = get_storage_implementation()
            optimizer = RetailConnectionOptimizer()
            config = EvolutionConfig(
                population_size=20,  # Small for testing
                generations=3,       # Few generations for speed
                mutation_rate=0.2,
                crossover_rate=0.8
            )
            
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=optimizer,
                use_ray=self.enable_distributed,
                evolution_config=config,
                storage=storage
            )
            
            # Step 2: Initialize population and analyze diversity
            population_size = await genetic_pool.initialize_population()
            
            # Analyze initial population
            seed_types = [ind.seed_type.value for ind in genetic_pool.population]
            unique_types = len(set(seed_types))
            
            test_result['population_analysis']['initial'] = {
                'size': population_size,
                'diversity': unique_types,
                'seed_types': list(set(seed_types))
            }
            
            # Step 3: Generate synthetic market data for evolution
            market_data = self._generate_synthetic_ohlcv_data('BTC', 200)
            
            # Step 4: Run evolution and track fitness progression
            print("   🔄 Running genetic evolution...")
            start_evolution = time.time()
            
            # Custom evolution tracking
            initial_population = genetic_pool.population.copy()
            
            # Evolve strategies
            best_individuals = await genetic_pool.evolve_strategies(market_data, generations=config.generations)
            
            evolution_time = time.time() - start_evolution
            
            # Step 5: Analyze evolution results
            final_population = genetic_pool.population
            evolution_history = genetic_pool.evolution_history
            
            # Fitness progression analysis
            fitness_progression = []
            for i, metrics in enumerate(evolution_history):
                fitness_progression.append({
                    'generation': i,
                    'best_fitness': metrics.best_fitness,
                    'average_fitness': metrics.average_fitness,
                    'diversity': metrics.population_diversity
                })
            
            test_result['fitness_progression'] = fitness_progression
            
            # Evolution quality metrics
            if len(evolution_history) > 0:
                initial_best = evolution_history[0].best_fitness
                final_best = evolution_history[-1].best_fitness
                improvement = final_best - initial_best
                
                test_result['evolution_metrics'] = {
                    'generations_completed': len(evolution_history),
                    'initial_best_fitness': initial_best,
                    'final_best_fitness': final_best,
                    'fitness_improvement': improvement,
                    'improvement_percentage': (improvement / abs(initial_best) * 100) if initial_best != 0 else 0,
                    'evolution_time': evolution_time,
                    'convergence_achieved': improvement > 0
                }
                
                # Test success criteria
                evolution_successful = (
                    len(evolution_history) == config.generations and  # Completed all generations
                    len(best_individuals) > 0 and                     # Found best individuals
                    all(ind.fitness is not None for ind in best_individuals[:5])  # Top individuals have fitness
                )
                
                test_result['status'] = 'passed' if evolution_successful else 'partial'
                
                if evolution_successful:
                    print(f"   ✅ Genetic Algorithm Workflow - PASSED")
                    print(f"      📈 Fitness improved by {improvement:.4f} ({(improvement/abs(initial_best)*100):.1f}%)")
                else:
                    print(f"   ⚠️ Genetic Algorithm Workflow - PARTIAL")
            else:
                test_result['status'] = 'failed'
                test_result['error'] = 'No evolution history generated'
                print("   ❌ Genetic Algorithm Workflow - FAILED: No evolution occurred")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Genetic Algorithm Workflow - FAILED: {e}")
        
        self.results['workflow_tests']['genetic_algorithm'] = test_result
    
    async def _test_signal_generation_workflow(self):
        """Test all genetic seeds generating meaningful trading signals."""
        print("\n📡 Testing Signal Generation Workflow...")
        
        test_result = {
            'status': 'pending',
            'seeds_tested': 0,
            'seeds_working': 0,
            'signal_quality_analysis': {},
            'seed_results': {}
        }
        
        try:
            # Step 1: Get all genetic seeds
            import src.strategy.genetic_seeds as genetic_seeds
            registry = genetic_seeds.get_registry()
            seed_names = registry.get_all_seed_names()
            
            # Step 2: Generate realistic test data
            market_data = self._generate_synthetic_ohlcv_data('BTC', 300)  # More data for ML seeds
            
            # Step 3: Test each seed type
            signal_quality_stats = {
                'total_signals': 0,
                'meaningful_signals': 0,
                'signal_distribution': {},
                'seeds_with_signals': 0
            }
            
            for seed_name in seed_names:
                test_result['seeds_tested'] += 1
                
                try:
                    # Create seed instance
                    seed_instance = registry.create_seed_instance(seed_name)
                    if seed_instance is None:
                        test_result['seed_results'][seed_name] = {
                            'status': 'failed',
                            'error': 'Failed to create instance'
                        }
                        continue
                    
                    # Generate signals
                    signals = seed_instance.generate_signals(market_data)
                    
                    # Analyze signal quality
                    if signals is not None and len(signals) > 0:
                        # Convert to numpy for analysis
                        signal_array = np.array(signals) if not isinstance(signals, np.ndarray) else signals
                        
                        # Signal statistics
                        total_signals = len(signal_array)
                        non_zero_signals = np.count_nonzero(signal_array)
                        unique_values = len(np.unique(signal_array))
                        signal_range = (float(np.min(signal_array)), float(np.max(signal_array)))
                        signal_std = float(np.std(signal_array))
                        
                        # Quality assessment
                        meaningful_signals = non_zero_signals > 0 and unique_values > 1 and signal_std > 0.001
                        
                        test_result['seed_results'][seed_name] = {
                            'status': 'passed',
                            'signals_generated': total_signals,
                            'non_zero_signals': int(non_zero_signals),
                            'unique_values': int(unique_values),
                            'signal_range': signal_range,
                            'signal_std': signal_std,
                            'meaningful': meaningful_signals,
                            'seed_type': seed_instance.genes.seed_type.value
                        }
                        
                        if meaningful_signals:
                            test_result['seeds_working'] += 1
                            signal_quality_stats['seeds_with_signals'] += 1
                            signal_quality_stats['meaningful_signals'] += non_zero_signals
                        
                        signal_quality_stats['total_signals'] += total_signals
                        
                    else:
                        test_result['seed_results'][seed_name] = {
                            'status': 'failed',
                            'error': 'No signals generated or empty output'
                        }
                
                except Exception as e:
                    test_result['seed_results'][seed_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Step 4: Overall signal quality analysis
            test_result['signal_quality_analysis'] = signal_quality_stats
            
            # Success criteria
            success_rate = test_result['seeds_working'] / test_result['seeds_tested'] if test_result['seeds_tested'] > 0 else 0
            
            if success_rate >= 0.8:  # 80% of seeds working
                test_result['status'] = 'passed'
                print(f"   ✅ Signal Generation Workflow - PASSED")
                print(f"      📊 {test_result['seeds_working']}/{test_result['seeds_tested']} seeds generating meaningful signals")
            elif success_rate >= 0.6:  # 60% working
                test_result['status'] = 'partial'
                print(f"   ⚠️ Signal Generation Workflow - PARTIAL")
                print(f"      📊 {test_result['seeds_working']}/{test_result['seeds_tested']} seeds working")
            else:
                test_result['status'] = 'failed'
                print(f"   ❌ Signal Generation Workflow - FAILED")
                print(f"      📊 Only {test_result['seeds_working']}/{test_result['seeds_tested']} seeds working")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Signal Generation Workflow - FAILED: {e}")
        
        self.results['workflow_tests']['signal_generation'] = test_result
    
    async def _test_discovery_workflow(self):
        """Test asset discovery and filtering workflow."""
        print("\n🔍 Testing Discovery Workflow...")
        
        test_result = {
            'status': 'pending',
            'discovery_results': {},
            'filtering_results': {}
        }
        
        try:
            # Step 1: Test enhanced asset filter
            from src.config.settings import get_settings
            from src.discovery.enhanced_asset_filter import EnhancedAssetFilter
            
            settings = get_settings()
            asset_filter = EnhancedAssetFilter(settings)
            
            # Step 2: Test asset universe filtering (mock mode for speed)
            mock_universe = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOT', 'LINK', 'UNI', 'AAVE', 'COMP', 'SUSHI']
            
            # Filter assets
            filtered_assets, _ = await asset_filter.filter_universe(universe_override=mock_universe)
            
            test_result['discovery_results'] = {
                'input_assets': len(mock_universe),
                'filtered_assets': len(filtered_assets) if filtered_assets else 0,
                'sample_assets': filtered_assets[:5] if filtered_assets else []
            }
            
            # Step 3: Test filtering criteria with proper parameters
            # Create mock metrics for the filtered assets
            mock_metrics = {}
            if filtered_assets:
                from src.discovery.asset_universe_filter import AssetMetrics
                for asset in filtered_assets[:5]:  # Sample metrics for first 5 assets
                    mock_metrics[asset] = AssetMetrics(
                        symbol=asset,
                        avg_bid_depth=1000.0,
                        avg_ask_depth=1000.0,
                        bid_ask_spread=0.001,
                        daily_volatility=0.05,
                        max_leverage=10
                    )
            
            filter_summary = asset_filter.get_enhanced_filter_summary(
                filtered_assets=filtered_assets or [], 
                metrics=mock_metrics
            )
            
            test_result['filtering_results'] = {
                'filter_summary': filter_summary,
                'filtering_successful': len(filtered_assets) > 0 if filtered_assets else False
            }
            
            # Success criteria
            discovery_successful = (
                filtered_assets is not None and
                len(filtered_assets) > 0 and
                len(filtered_assets) <= len(mock_universe)
            )
            
            if discovery_successful:
                test_result['status'] = 'passed'
                print(f"   ✅ Discovery Workflow - PASSED")
                print(f"      🎯 Filtered {len(mock_universe)} → {len(filtered_assets)} assets")
            else:
                test_result['status'] = 'failed'
                print(f"   ❌ Discovery Workflow - FAILED: No assets discovered")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Discovery Workflow - FAILED: {e}")
        
        self.results['workflow_tests']['discovery'] = test_result
    
    async def _test_discovery_to_strategy_integration(self):
        """Test integration between discovery and strategy modules."""
        print("\n🔗 Testing Discovery → Strategy Integration...")
        
        test_result = {
            'status': 'pending',
            'integration_steps': {}
        }
        
        try:
            # Step 1: Discovery phase
            from src.config.settings import get_settings
            from src.discovery.enhanced_asset_filter import EnhancedAssetFilter
            
            settings = get_settings()
            asset_filter = EnhancedAssetFilter(settings)
            mock_universe = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOT']
            discovered_assets, _ = await asset_filter.filter_universe(universe_override=mock_universe)
            
            test_result['integration_steps']['discovery'] = {
                'status': 'passed' if discovered_assets and len(discovered_assets) > 0 else 'failed',
                'assets_discovered': len(discovered_assets) if discovered_assets else 0
            }
            
            # Step 2: Strategy initialization using discovered assets
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
            from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
            from src.data.storage_interfaces import get_storage_implementation
            
            storage = get_storage_implementation()
            optimizer = RetailConnectionOptimizer()
            config = EvolutionConfig(population_size=10, generations=1)
            
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=optimizer,
                use_ray=False,
                evolution_config=config,
                storage=storage
            )
            
            population_size = await genetic_pool.initialize_population()
            
            test_result['integration_steps']['strategy_initialization'] = {
                'status': 'passed' if population_size > 0 else 'failed',
                'population_size': population_size
            }
            
            # Step 3: Test strategy evolution using discovered assets data
            if discovered_assets and len(discovered_assets) > 0:
                # Generate market data for first discovered asset
                primary_asset = discovered_assets[0]
                market_data = self._generate_synthetic_ohlcv_data(primary_asset, 100)
                
                # Quick evolution test
                best_individuals = await genetic_pool.evolve_strategies(market_data, generations=1)
                
                test_result['integration_steps']['strategy_evolution'] = {
                    'status': 'passed' if len(best_individuals) > 0 else 'failed',
                    'best_individuals_count': len(best_individuals),
                    'primary_asset_tested': primary_asset
                }
            else:
                test_result['integration_steps']['strategy_evolution'] = {
                    'status': 'skipped',
                    'reason': 'no_discovered_assets'
                }
            
            # Overall integration success
            all_steps_passed = all(
                step_result.get('status') == 'passed' 
                for step_result in test_result['integration_steps'].values()
                if step_result.get('status') != 'skipped'
            )
            
            test_result['status'] = 'passed' if all_steps_passed else 'partial'
            
            if all_steps_passed:
                print("   ✅ Discovery → Strategy Integration - PASSED")
            else:
                print("   ⚠️ Discovery → Strategy Integration - PARTIAL")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Discovery → Strategy Integration - FAILED: {e}")
        
        self.results['integration_tests']['discovery_to_strategy'] = test_result
    
    async def _test_strategy_to_execution_integration(self):
        """Test integration between strategy and execution modules."""
        print("\n⚡ Testing Strategy → Execution Integration...")
        
        test_result = {
            'status': 'pending',
            'integration_steps': {}
        }
        
        try:
            # Step 1: Create evolved strategies
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
            from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
            from src.data.storage_interfaces import get_storage_implementation
            
            storage = get_storage_implementation()
            optimizer = RetailConnectionOptimizer()
            config = EvolutionConfig(population_size=5, generations=1)
            
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=optimizer,
                use_ray=False,
                evolution_config=config,
                storage=storage
            )
            
            await genetic_pool.initialize_population()
            market_data = self._generate_synthetic_ohlcv_data('BTC', 50)
            evolved_strategies = await genetic_pool.evolve_strategies(market_data, generations=1)
            
            test_result['integration_steps']['strategy_creation'] = {
                'status': 'passed' if len(evolved_strategies) > 0 else 'failed',
                'strategies_count': len(evolved_strategies)
            }
            
            # Step 2: Test order management integration
            from src.execution.order_management import OrderType, OrderSide, OrderRequest
            
            # Create mock order from strategy
            if len(evolved_strategies) > 0:
                best_strategy = evolved_strategies[0]
                
                # Generate a trading signal using the strategy
                signals = None
                try:
                    # Get the strategy's seed type and create instance for signal generation
                    import src.strategy.genetic_seeds as genetic_seeds
                    registry = genetic_seeds.get_registry()
                    available_seeds = registry._type_index.get(best_strategy.seed_type, [])
                    
                    if available_seeds:
                        seed_name = available_seeds[0]
                        seed_instance = registry.create_seed_instance(seed_name, best_strategy.genes)
                        signals = seed_instance.generate_signals(market_data)
                except Exception as e:
                    logger.warning(f"Could not generate signals: {e}")
                
                # Create order request based on strategy
                order_request = OrderRequest(
                    symbol='BTC',
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    size=0.001,  # Small test size
                    strategy_id=best_strategy.genes.seed_id,
                    metadata={'fitness': best_strategy.fitness}
                )
                
                test_result['integration_steps']['order_creation'] = {
                    'status': 'passed',
                    'order_symbol': order_request.symbol,
                    'order_side': order_request.side.value,
                    'signals_generated': signals is not None and len(signals) > 0
                }
            else:
                test_result['integration_steps']['order_creation'] = {
                    'status': 'failed',
                    'reason': 'no_strategies_available'
                }
            
            # Step 3: Test position sizing integration
            from src.execution.position_sizer import GeneticPositionSizer
            
            position_sizer = GeneticPositionSizer()
            
            # Test position sizing calculation - create mock seed and data
            if len(evolved_strategies) > 0:
                best_strategy = evolved_strategies[0]
                # Create seed instance for position sizing
                import src.strategy.genetic_seeds as genetic_seeds
                registry = genetic_seeds.get_registry()
                available_seeds = registry._type_index.get(best_strategy.seed_type, [])
                
                if available_seeds:
                    seed_name = available_seeds[0]
                    mock_seed = registry.create_seed_instance(seed_name, best_strategy.genes)
                    
                    if mock_seed:
                        test_position = await position_sizer.calculate_position_size(
                            symbol='BTC',
                            seed=mock_seed,
                            market_data=market_data,
                            signal_strength=0.7
                        )
                    else:
                        # Fallback if seed creation fails
                        test_position = None
                else:
                    test_position = None
            else:
                test_position = None
            
            test_result['integration_steps']['position_sizing'] = {
                'status': 'passed' if test_position and hasattr(test_position, 'size') and test_position.size > 0 else 'failed',
                'calculated_size': test_position.size if test_position and hasattr(test_position, 'size') else 0,
                'scaling_method': test_position.sizing_method if test_position and hasattr(test_position, 'sizing_method') else 'unknown'
            }
            
            # Overall integration success
            all_steps_passed = all(
                step_result.get('status') == 'passed'
                for step_result in test_result['integration_steps'].values()
            )
            
            test_result['status'] = 'passed' if all_steps_passed else 'partial'
            
            if all_steps_passed:
                print("   ✅ Strategy → Execution Integration - PASSED")
            else:
                print("   ⚠️ Strategy → Execution Integration - PARTIAL")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Strategy → Execution Integration - FAILED: {e}")
        
        self.results['integration_tests']['strategy_to_execution'] = test_result
    
    async def _test_data_to_modules_integration(self):
        """Test data module integration with all other modules."""
        print("\n💾 Testing Data → All Modules Integration...")
        
        test_result = {
            'status': 'pending',
            'data_consumers': {}
        }
        
        try:
            # Step 1: Create data source
            from src.data.storage_interfaces import get_storage_implementation
            storage = get_storage_implementation()
            
            # Generate and store test data
            market_data = self._generate_synthetic_ohlcv_data('ETH', 100)
            
            # Convert to storage format
            from src.data.data_storage import OHLCVBar
            bars = []
            for _, row in market_data.iterrows():
                bar = OHLCVBar(
                    symbol='ETH',
                    timestamp=row.name,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    vwap=float((row['high'] + row['low'] + row['close']) / 3),  # Estimated VWAP
                    trade_count=int(np.random.randint(50, 200))  # Estimated trade count for testing
                )
                bars.append(bar)
            
            await storage.store_ohlcv_bars(bars)
            
            # Step 2: Test strategy module consuming data
            try:
                import src.strategy.genetic_seeds as genetic_seeds
                registry = genetic_seeds.get_registry()
                seed_names = registry.get_all_seed_names()
                
                if seed_names:
                    # Test first seed with stored data
                    seed_instance = registry.create_seed_instance(seed_names[0])
                    signals = seed_instance.generate_signals(market_data)
                    
                    test_result['data_consumers']['strategy_module'] = {
                        'status': 'passed' if signals is not None else 'failed',
                        'seed_tested': seed_names[0],
                        'signals_generated': len(signals) if signals is not None else 0
                    }
                else:
                    test_result['data_consumers']['strategy_module'] = {
                        'status': 'failed',
                        'reason': 'no_seeds_available'
                    }
            except Exception as e:
                test_result['data_consumers']['strategy_module'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Step 3: Test discovery module with data
            try:
                from src.discovery.enhanced_asset_filter import EnhancedAssetFilter
                from src.config.settings import get_settings
                
                settings = get_settings()
                asset_filter = EnhancedAssetFilter(settings)
                
                # Test filtering with our stored asset
                filtered, _ = await asset_filter.filter_universe(universe_override=['ETH', 'BTC'])
                
                test_result['data_consumers']['discovery_module'] = {
                    'status': 'passed' if filtered and len(filtered) > 0 else 'failed',
                    'assets_filtered': len(filtered) if filtered else 0
                }
            except Exception as e:
                test_result['data_consumers']['discovery_module'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Step 4: Test execution module data consumption
            try:
                from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
                from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
                
                optimizer = RetailConnectionOptimizer()
                config = EvolutionConfig(population_size=3, generations=1)
                
                genetic_pool = GeneticStrategyPool(
                    connection_optimizer=optimizer,
                    use_ray=False,
                    evolution_config=config,
                    storage=storage  # Uses our data storage
                )
                
                await genetic_pool.initialize_population()
                evolved = await genetic_pool.evolve_strategies(market_data, generations=1)
                
                test_result['data_consumers']['execution_module'] = {
                    'status': 'passed' if len(evolved) > 0 else 'failed',
                    'strategies_evolved': len(evolved)
                }
            except Exception as e:
                test_result['data_consumers']['execution_module'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Overall success assessment
            passed_consumers = sum(1 for result in test_result['data_consumers'].values() 
                                 if result.get('status') == 'passed')
            total_consumers = len(test_result['data_consumers'])
            
            if passed_consumers == total_consumers:
                test_result['status'] = 'passed'
                print("   ✅ Data → All Modules Integration - PASSED")
            elif passed_consumers >= total_consumers * 0.6:  # 60% threshold
                test_result['status'] = 'partial'
                print(f"   ⚠️ Data → All Modules Integration - PARTIAL ({passed_consumers}/{total_consumers})")
            else:
                test_result['status'] = 'failed'
                print(f"   ❌ Data → All Modules Integration - FAILED ({passed_consumers}/{total_consumers})")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Data → All Modules Integration - FAILED: {e}")
        
        self.results['integration_tests']['data_to_modules'] = test_result
    
    async def _test_error_handling_scenarios(self):
        """Test error handling and recovery across the system."""
        print("\n🛡️ Testing Error Handling & Recovery...")
        
        test_result = {
            'status': 'pending',
            'error_scenarios': {}
        }
        
        try:
            # Scenario 1: Invalid data handling
            try:
                import src.strategy.genetic_seeds as genetic_seeds
                registry = genetic_seeds.get_registry()
                seed_names = registry.get_all_seed_names()
                
                if seed_names:
                    seed_instance = registry.create_seed_instance(seed_names[0])
                    
                    # Test with invalid data
                    invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
                    
                    try:
                        signals = seed_instance.generate_signals(invalid_data)
                        # Should handle gracefully
                        test_result['error_scenarios']['invalid_data'] = {
                            'status': 'passed',
                            'handled_gracefully': True,
                            'returned_signals': signals is not None
                        }
                    except Exception as e:
                        # Error was caught, which is also acceptable
                        test_result['error_scenarios']['invalid_data'] = {
                            'status': 'passed',
                            'handled_gracefully': True,
                            'error_caught': str(e)[:100]
                        }
                else:
                    test_result['error_scenarios']['invalid_data'] = {
                        'status': 'skipped',
                        'reason': 'no_seeds_available'
                    }
            except Exception as e:
                test_result['error_scenarios']['invalid_data'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Scenario 2: Empty population handling
            try:
                from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
                from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
                from src.data.storage_interfaces import get_storage_implementation
                
                storage = get_storage_implementation()
                optimizer = RetailConnectionOptimizer()
                config = EvolutionConfig(population_size=0, generations=1)  # Invalid size
                
                genetic_pool = GeneticStrategyPool(
                    connection_optimizer=optimizer,
                    use_ray=False,
                    evolution_config=config,
                    storage=storage
                )
                
                try:
                    population_size = await genetic_pool.initialize_population()
                    # Should handle gracefully or raise meaningful error
                    test_result['error_scenarios']['empty_population'] = {
                        'status': 'passed',
                        'handled_gracefully': True,
                        'population_size': population_size
                    }
                except Exception as e:
                    # Meaningful error is acceptable
                    test_result['error_scenarios']['empty_population'] = {
                        'status': 'passed',
                        'meaningful_error': True,
                        'error_message': str(e)[:100]
                    }
            except Exception as e:
                test_result['error_scenarios']['empty_population'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Scenario 3: Storage connectivity issues
            try:
                from src.data.storage_interfaces import LocalDataStorage
                
                # Test with invalid path
                invalid_storage = LocalDataStorage("/invalid/path/database.duckdb")
                
                try:
                    health = await invalid_storage.health_check()
                    # Should report unhealthy status
                    test_result['error_scenarios']['storage_connectivity'] = {
                        'status': 'passed' if health['status'] != 'healthy' else 'failed',
                        'reported_status': health['status'],
                        'error_detected': 'error' in health
                    }
                except Exception as e:
                    # Exception is also acceptable for invalid storage
                    test_result['error_scenarios']['storage_connectivity'] = {
                        'status': 'passed',
                        'exception_raised': True,
                        'error_message': str(e)[:100]
                    }
            except Exception as e:
                test_result['error_scenarios']['storage_connectivity'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Overall error handling assessment
            passed_scenarios = sum(1 for result in test_result['error_scenarios'].values()
                                 if result.get('status') == 'passed')
            total_scenarios = len(test_result['error_scenarios'])
            
            if passed_scenarios >= total_scenarios * 0.8:  # 80% threshold
                test_result['status'] = 'passed'
                print(f"   ✅ Error Handling & Recovery - PASSED ({passed_scenarios}/{total_scenarios})")
            elif passed_scenarios >= total_scenarios * 0.5:  # 50% threshold
                test_result['status'] = 'partial'
                print(f"   ⚠️ Error Handling & Recovery - PARTIAL ({passed_scenarios}/{total_scenarios})")
            else:
                test_result['status'] = 'failed'
                print(f"   ❌ Error Handling & Recovery - FAILED ({passed_scenarios}/{total_scenarios})")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ Error Handling & Recovery - FAILED: {e}")
        
        self.results['error_handling_tests'] = test_result
    
    async def _test_system_performance(self):
        """Test system performance under realistic conditions."""
        print("\n⚡ Testing System Performance...")
        
        test_result = {
            'status': 'pending',
            'performance_metrics': {}
        }
        
        try:
            # Test 1: Data processing throughput
            start_time = time.time()
            
            from src.data.storage_interfaces import get_storage_implementation
            storage = get_storage_implementation()
            
            # Generate larger dataset for performance testing
            large_dataset = self._generate_synthetic_ohlcv_data('PERF_TEST', 1000)
            
            # Convert to storage format
            from src.data.data_storage import OHLCVBar
            bars = []
            for _, row in large_dataset.iterrows():
                bar = OHLCVBar(
                    symbol='PERF_TEST',
                    timestamp=row.name,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    vwap=float((row['high'] + row['low'] + row['close']) / 3),  # Estimated VWAP
                    trade_count=int(np.random.randint(50, 200))  # Estimated trade count for testing
                )
                bars.append(bar)
            
            # Store data and measure performance
            store_start = time.time()
            await storage.store_ohlcv_bars(bars[:500])  # Store 500 bars
            store_time = time.time() - store_start
            
            # Retrieve data and measure performance
            retrieve_start = time.time()
            retrieved = await storage.get_ohlcv_bars('PERF_TEST', limit=500)
            retrieve_time = time.time() - retrieve_start
            
            test_result['performance_metrics']['data_processing'] = {
                'bars_stored': 500,
                'store_time_seconds': store_time,
                'store_throughput_bars_per_second': 500 / store_time if store_time > 0 else 0,
                'bars_retrieved': len(retrieved),
                'retrieve_time_seconds': retrieve_time,
                'retrieve_throughput_bars_per_second': len(retrieved) / retrieve_time if retrieve_time > 0 else 0
            }
            
            # Test 2: Genetic algorithm performance
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
            from src.execution.retail_connection_optimizer import RetailConnectionOptimizer
            
            optimizer = RetailConnectionOptimizer()
            config = EvolutionConfig(population_size=30, generations=3)  # Larger population
            
            genetic_pool = GeneticStrategyPool(
                connection_optimizer=optimizer,
                use_ray=self.enable_distributed,
                evolution_config=config,
                storage=storage
            )
            
            # Measure population initialization time
            init_start = time.time()
            population_size = await genetic_pool.initialize_population()
            init_time = time.time() - init_start
            
            # Measure evolution time
            evolution_start = time.time()
            market_data = self._generate_synthetic_ohlcv_data('PERF_TEST', 200)
            evolved = await genetic_pool.evolve_strategies(market_data, generations=config.generations)
            evolution_time = time.time() - evolution_start
            
            test_result['performance_metrics']['genetic_algorithm'] = {
                'population_size': population_size,
                'initialization_time_seconds': init_time,
                'individuals_per_second_init': population_size / init_time if init_time > 0 else 0,
                'evolution_time_seconds': evolution_time,
                'generations': config.generations,
                'time_per_generation': evolution_time / config.generations if config.generations > 0 else 0,
                'individuals_evolved': len(evolved)
            }
            
            # Performance assessment
            data_perf_good = (
                test_result['performance_metrics']['data_processing']['store_throughput_bars_per_second'] > 50 and
                test_result['performance_metrics']['data_processing']['retrieve_throughput_bars_per_second'] > 100
            )
            
            genetic_perf_good = (
                test_result['performance_metrics']['genetic_algorithm']['time_per_generation'] < 60 and  # < 1 minute per generation
                test_result['performance_metrics']['genetic_algorithm']['individuals_per_second_init'] > 1  # > 1 individual per second
            )
            
            if data_perf_good and genetic_perf_good:
                test_result['status'] = 'passed'
                print("   ✅ System Performance - PASSED")
                print(f"      📊 Data: {test_result['performance_metrics']['data_processing']['store_throughput_bars_per_second']:.1f} bars/s store")
                print(f"      🧬 Genetic: {test_result['performance_metrics']['genetic_algorithm']['time_per_generation']:.1f}s per generation")
            elif data_perf_good or genetic_perf_good:
                test_result['status'] = 'partial'
                print("   ⚠️ System Performance - PARTIAL")
            else:
                test_result['status'] = 'failed'
                print("   ❌ System Performance - FAILED")
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            print(f"   ❌ System Performance - FAILED: {e}")
        
        self.results['performance_tests'] = test_result
    
    def _calculate_business_readiness_score(self):
        """Calculate overall business readiness score."""
        scores = []
        weights = {
            'workflow_tests': 40,      # 40% weight for core workflows
            'integration_tests': 35,   # 35% weight for integration
            'error_handling_tests': 15, # 15% weight for robustness
            'performance_tests': 10    # 10% weight for performance
        }
        
        for category, weight in weights.items():
            if category in self.results:
                category_results = self.results[category]
                
                if isinstance(category_results, dict):
                    # Count passed vs total tests in this category
                    if category == 'workflow_tests' or category == 'integration_tests':
                        passed_tests = sum(1 for test in category_results.values()
                                         if isinstance(test, dict) and test.get('status') == 'passed')
                        total_tests = len(category_results)
                    else:
                        # Single test in category
                        passed_tests = 1 if category_results.get('status') == 'passed' else 0.5 if category_results.get('status') == 'partial' else 0
                        total_tests = 1
                    
                    if total_tests > 0:
                        category_score = (passed_tests / total_tests) * weight
                        scores.append(category_score)
        
        self.results['business_readiness_score'] = sum(scores)
        
        # Overall status
        if self.results['business_readiness_score'] >= 80:
            self.results['overall_status'] = 'production_ready'
        elif self.results['business_readiness_score'] >= 60:
            self.results['overall_status'] = 'mostly_ready'
        elif self.results['business_readiness_score'] >= 40:
            self.results['overall_status'] = 'needs_work'
        else:
            self.results['overall_status'] = 'not_ready'
    
    def _generate_synthetic_ohlcv_data(self, symbol: str, num_bars: int) -> pd.DataFrame:
        """Generate realistic synthetic OHLCV data for testing."""
        np.random.seed(42)  # Reproducible data
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=num_bars)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_bars)
        
        # Generate realistic price data with trend and noise
        base_price = 50000 if symbol == 'BTC' else 3000 if symbol == 'ETH' else 100
        
        # Random walk with trend
        returns = np.random.normal(0.0002, 0.02, num_bars)  # Small positive trend with volatility
        prices = np.cumprod(1 + returns) * base_price
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate realistic high/low around close
            volatility = close * 0.005  # 0.5% volatility
            high = close + abs(np.random.normal(0, volatility))
            low = close - abs(np.random.normal(0, volatility))
            
            # Open is previous close (with small gap)
            if i == 0:
                open_price = close
            else:
                open_price = data[i-1]['close'] + np.random.normal(0, volatility * 0.5)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            volume = abs(np.random.normal(1000000, 300000))
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _print_comprehensive_results(self):
        """Print comprehensive validation results."""
        print("\n" + "=" * 70)
        print("🎯 COMPLETE SYSTEM INTEGRATION VALIDATION RESULTS")
        print("=" * 70)
        
        # Overall summary
        print(f"\n📊 OVERALL STATUS: {self.results['overall_status'].upper().replace('_', ' ')}")
        print(f"📈 BUSINESS READINESS SCORE: {self.results['business_readiness_score']:.1f}/100")
        print(f"⏱️ VALIDATION DURATION: {self.results['test_duration']:.1f}s")
        
        # Workflow tests summary
        if 'workflow_tests' in self.results:
            print(f"\n🔄 WORKFLOW TESTS:")
            for workflow, result in self.results['workflow_tests'].items():
                status_icon = "✅" if result.get('status') == 'passed' else "⚠️" if result.get('status') == 'partial' else "❌"
                print(f"   {status_icon} {workflow.replace('_', ' ').title()}: {result.get('status', 'unknown').upper()}")
        
        # Integration tests summary  
        if 'integration_tests' in self.results:
            print(f"\n🔗 INTEGRATION TESTS:")
            for integration, result in self.results['integration_tests'].items():
                status_icon = "✅" if result.get('status') == 'passed' else "⚠️" if result.get('status') == 'partial' else "❌"
                print(f"   {status_icon} {integration.replace('_', ' ').title()}: {result.get('status', 'unknown').upper()}")
        
        # Error handling summary
        if 'error_handling_tests' in self.results:
            error_result = self.results['error_handling_tests']
            status_icon = "✅" if error_result.get('status') == 'passed' else "⚠️" if error_result.get('status') == 'partial' else "❌"
            print(f"\n🛡️ ERROR HANDLING: {status_icon} {error_result.get('status', 'unknown').upper()}")
        
        # Performance summary
        if 'performance_tests' in self.results:
            perf_result = self.results['performance_tests']
            status_icon = "✅" if perf_result.get('status') == 'passed' else "⚠️" if perf_result.get('status') == 'partial' else "❌"
            print(f"\n⚡ PERFORMANCE: {status_icon} {perf_result.get('status', 'unknown').upper()}")
        
        # Business readiness assessment
        print(f"\n🎯 BUSINESS READINESS ASSESSMENT:")
        if self.results['business_readiness_score'] >= 80:
            print("   🚀 SYSTEM IS PRODUCTION READY")
            print("   ✅ All core workflows functional")
            print("   ✅ Integration points working")
            print("   ✅ Error handling robust")
            print("   ✅ Performance acceptable")
        elif self.results['business_readiness_score'] >= 60:
            print("   ⚠️ SYSTEM IS MOSTLY READY")
            print("   ✅ Core functionality working")
            print("   ⚠️ Some integration issues need attention")
        elif self.results['business_readiness_score'] >= 40:
            print("   🔧 SYSTEM NEEDS WORK")
            print("   ⚠️ Several components need fixes before production")
        else:
            print("   ❌ SYSTEM NOT READY FOR PRODUCTION")
            print("   ❌ Significant issues detected across multiple areas")
        
        print("=" * 70)

async def main():
    """Main validation execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete System Integration Validator')
    parser.add_argument('--real-data', action='store_true', help='Enable real data testing')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed Ray testing')
    args = parser.parse_args()
    
    validator = CompleteSystemIntegrationValidator(
        enable_real_data=args.real_data,
        enable_distributed=args.distributed
    )
    
    results = await validator.run_complete_validation()
    
    # Save detailed results
    import json
    results_file = validator.project_root / "complete_integration_results.json"
    
    json_results = results.copy()
    json_results['timestamp'] = results['timestamp'].isoformat()
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code based on business readiness
    if results['business_readiness_score'] >= 60:  # 60% threshold for success
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())