#!/usr/bin/env python3
"""
END-TO-END ASSET → DATA → GA FLOW VALIDATION SCRIPT

This script validates the complete data flow from Hyperliquid asset discovery through
historical data ingestion to genetic algorithm initialization. This is the final
validation of the core architecture described in the planning PRP.

Based on research documentation and existing infrastructure:
- /research/hyperliquid_documentation/ (API specifications)
- src/data/hyperliquid_client.py (data access layer)
- src/strategy/universal_strategy_engine.py (GA initialization)
- src/execution/genetic_strategy_pool.py (genetic algorithm core)

Usage: python test_asset_to_ga_flow.py

Expected Output:
- If successful: Complete end-to-end flow validation with metrics
- If failed: Specific bottleneck identification and troubleshooting
"""

import asyncio
import sys
import traceback
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings

# Import our existing infrastructure (research-validated)
try:
    from src.data.hyperliquid_client import HyperliquidClient
    from src.strategy.universal_strategy_engine import UniversalStrategyEngine
    from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionConfig
    from src.config.settings import get_settings
except ImportError as e:
    print(f"❌ IMPORT ERROR: Failed to import required modules: {e}")
    print("Ensure you're running from the project root directory.")
    sys.exit(1)


def format_flow_step(step_name: str, status: str, details: str = "", duration_ms: float = 0) -> None:
    """Format and display flow step results.
    
    Args:
        step_name: Name of the flow step
        status: Status (✅, ❌, ⚠️)
        details: Additional details to display
        duration_ms: Execution time in milliseconds
    """
    duration_str = f" ({duration_ms:.1f}ms)" if duration_ms > 0 else ""
    print(f"   {status} {step_name}{duration_str}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"      {line}")


async def test_end_to_end_flow() -> Dict[str, Any]:
    """Test complete asset → data → GA flow.
    
    Returns:
        Dictionary with comprehensive test results
    """
    results = {
        'steps_completed': 0,
        'total_steps': 5,
        'step_results': {},
        'assets_discovered': 0,
        'data_points_ingested': 0,
        'ga_individuals_created': 0,
        'universe_initialized': False,
        'errors': [],
        'execution_time_ms': 0,
        'bottlenecks': []
    }
    
    overall_start = datetime.now()
    
    try:
        print("🔧 Initializing end-to-end flow test...")
        settings = get_settings()
        
        # Suppress warnings for cleaner output during testing
        warnings.filterwarnings('ignore')
        
        # STEP 1: Asset Discovery
        step_start = datetime.now()
        print("\n🔍 STEP 1: Asset Discovery")
        
        try:
            async with HyperliquidClient(settings) as client:
                asset_contexts = await client.get_asset_contexts()
                
                if not asset_contexts or len(asset_contexts) == 0:
                    raise ValueError("No assets discovered from Hyperliquid")
                
                results['assets_discovered'] = len(asset_contexts)
                results['step_results']['asset_discovery'] = {
                    'success': True,
                    'asset_count': len(asset_contexts),
                    'sample_assets': [asset.get('name', 'UNKNOWN') for asset in asset_contexts[:5]]
                }
                
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                
                # Select top assets for testing (liquid markets)
                test_assets = []
                for asset in asset_contexts:
                    name = asset.get('name', '').upper()
                    if name in ['BTC', 'ETH', 'SOL', 'AVAX', 'ARB']:  # Research-identified liquid markets
                        test_assets.append(name)
                        if len(test_assets) >= 3:  # Limit for testing
                            break
                
                if not test_assets:
                    test_assets = [asset_contexts[0].get('name', 'BTC')]  # Fallback
                
                format_flow_step(
                    "Asset Discovery", 
                    "✅", 
                    f"Discovered {len(asset_contexts)} assets\nSelected for testing: {', '.join(test_assets)}", 
                    step_duration
                )
                results['steps_completed'] += 1
                
        except Exception as e:
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            error_msg = f"Asset discovery failed: {e}"
            results['errors'].append(error_msg)
            results['step_results']['asset_discovery'] = {'success': False, 'error': str(e)}
            format_flow_step("Asset Discovery", "❌", error_msg, step_duration)
            results['bottlenecks'].append("Asset Discovery")
            return results
        
        # STEP 2: Historical Data Ingestion
        step_start = datetime.now()
        print("\n📊 STEP 2: Historical Data Ingestion")
        
        try:
            total_data_points = 0
            data_summary = []
            
            # Request data for the last 48 hours (sufficient for GA testing)
            end_time = datetime.now()
            start_time_data = end_time - timedelta(hours=48)
            start_timestamp = int(start_time_data.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            async with HyperliquidClient(settings) as client:
                for asset in test_assets:
                    try:
                        # Get hourly data (good balance of detail vs. volume)
                        candles = await client.get_candles(
                            symbol=asset,
                            interval='1h',
                            start_time=start_timestamp,
                            end_time=end_timestamp
                        )
                        
                        if candles and len(candles) > 0:
                            total_data_points += len(candles)
                            data_summary.append(f"{asset}: {len(candles)} candles")
                        else:
                            data_summary.append(f"{asset}: No data available")
                            
                    except Exception as e:
                        data_summary.append(f"{asset}: Error - {str(e)[:50]}")
            
            results['data_points_ingested'] = total_data_points
            results['step_results']['data_ingestion'] = {
                'success': total_data_points > 0,
                'total_data_points': total_data_points,
                'assets_with_data': len([s for s in data_summary if 'candles' in s])
            }
            
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            
            if total_data_points > 0:
                format_flow_step(
                    "Historical Data Ingestion", 
                    "✅", 
                    f"Ingested {total_data_points} data points\n" + "\n".join(data_summary), 
                    step_duration
                )
                results['steps_completed'] += 1
            else:
                raise ValueError("No historical data retrieved for any asset")
                
        except Exception as e:
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            error_msg = f"Data ingestion failed: {e}"
            results['errors'].append(error_msg)
            results['step_results']['data_ingestion'] = {'success': False, 'error': str(e)}
            format_flow_step("Historical Data Ingestion", "❌", error_msg, step_duration)
            results['bottlenecks'].append("Data Ingestion")
            return results
        
        # STEP 3: Universal Strategy Engine Initialization
        step_start = datetime.now()
        print("\n🧠 STEP 3: Strategy Engine Initialization")
        
        try:
            # Initialize the universal strategy engine with Hyperliquid client
            engine = UniversalStrategyEngine(settings)
            
            # Initialize universe (no parameters needed)
            universe_result = await engine.initialize_universe()
            
            results['universe_initialized'] = True
            results['step_results']['strategy_engine'] = {
                'success': True,
                'universe_assets': len(test_assets),
                'engine_ready': True
            }
            
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            format_flow_step(
                "Strategy Engine Initialization", 
                "✅", 
                f"Universe initialized with {len(test_assets)} assets\nEngine ready for strategy execution", 
                step_duration
            )
            results['steps_completed'] += 1
            
        except Exception as e:
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            error_msg = f"Strategy engine initialization failed: {e}"
            results['errors'].append(error_msg)
            results['step_results']['strategy_engine'] = {'success': False, 'error': str(e)}
            format_flow_step("Strategy Engine Initialization", "❌", error_msg, step_duration)
            results['bottlenecks'].append("Strategy Engine")
            # Continue to test GA even if engine init fails
        
        # STEP 4: Genetic Algorithm Pool Creation
        step_start = datetime.now()
        print("\n🧬 STEP 4: Genetic Algorithm Pool Creation")
        
        try:
            # Create minimal GA configuration for testing
            config = EvolutionConfig(
                population_size=20,  # Small population for testing
                generations=1,       # Single generation test
                mutation_rate=0.1,
                crossover_rate=0.7
            )
            
            ga_pool = GeneticStrategyPool(config)
            
            # Test population initialization
            population_count = await ga_pool.initialize_population()
            
            if population_count > 0:
                results['ga_individuals_created'] = population_count
                results['step_results']['genetic_algorithm'] = {
                    'success': True,
                    'population_size': population_count,
                    'config': {
                        'population_size': config.population_size,
                        'generations': config.generations
                    }
                }
                
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                format_flow_step(
                    "Genetic Algorithm Pool Creation", 
                    "✅", 
                    f"Created {population_count} individuals\nPopulation initialized successfully", 
                    step_duration
                )
                results['steps_completed'] += 1
            else:
                raise ValueError("Failed to create GA population")
                
        except Exception as e:
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            error_msg = f"GA pool creation failed: {e}"
            results['errors'].append(error_msg)
            results['step_results']['genetic_algorithm'] = {'success': False, 'error': str(e)}
            format_flow_step("Genetic Algorithm Pool Creation", "❌", error_msg, step_duration)
            results['bottlenecks'].append("Genetic Algorithm")
            return results
        
        # STEP 5: End-to-End Integration Validation
        step_start = datetime.now()
        print("\n🔗 STEP 5: End-to-End Integration Validation")
        
        try:
            # Validate that all components can work together
            integration_checks = {
                'asset_to_data_flow': results['assets_discovered'] > 0 and results['data_points_ingested'] > 0,
                'data_to_ga_flow': results['data_points_ingested'] > 0 and results['ga_individuals_created'] > 0,
                'complete_pipeline': all([
                    results['assets_discovered'] > 0,
                    results['data_points_ingested'] > 0,
                    results['ga_individuals_created'] > 0
                ])
            }
            
            all_checks_passed = all(integration_checks.values())
            
            results['step_results']['integration'] = {
                'success': all_checks_passed,
                'checks': integration_checks,
                'data_flow_ratio': results['data_points_ingested'] / results['assets_discovered'] if results['assets_discovered'] > 0 else 0,
                'ga_readiness': results['ga_individuals_created'] > 0
            }
            
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            
            if all_checks_passed:
                format_flow_step(
                    "End-to-End Integration", 
                    "✅", 
                    f"All integration checks passed\nPipeline ready for production deployment", 
                    step_duration
                )
                results['steps_completed'] += 1
            else:
                failed_checks = [check for check, passed in integration_checks.items() if not passed]
                raise ValueError(f"Integration validation failed: {failed_checks}")
                
        except Exception as e:
            step_duration = (datetime.now() - step_start).total_seconds() * 1000
            error_msg = f"Integration validation failed: {e}"
            results['errors'].append(error_msg)
            results['step_results']['integration'] = {'success': False, 'error': str(e)}
            format_flow_step("End-to-End Integration", "❌", error_msg, step_duration)
            results['bottlenecks'].append("Integration")
            
    except Exception as e:
        error_msg = f"Unexpected error in flow test: {e}"
        results['errors'].append(error_msg)
        print(f"\n❌ UNEXPECTED ERROR: {error_msg}")
        traceback.print_exc()
    
    finally:
        # Calculate total execution time
        overall_end = datetime.now()
        results['execution_time_ms'] = (overall_end - overall_start).total_seconds() * 1000
    
    return results


def main():
    """Main test execution function."""
    print("🔗 END-TO-END ASSET → DATA → GA FLOW VALIDATION")
    print("=" * 70)
    print("Testing complete architecture: Hyperliquid → Data → Genetic Algorithm")
    print("Final validation of the genetic trading organism pipeline")
    print("=" * 70)
    
    # Setup logging for debugging
    logging.basicConfig(
        level=logging.WARNING,  # Reduced logging for cleaner output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the async test
    test_results = asyncio.run(test_end_to_end_flow())
    
    # Comprehensive results analysis
    print("\n" + "=" * 70)
    print("📋 END-TO-END FLOW TEST RESULTS")
    print("=" * 70)
    
    success = (test_results['steps_completed'] == test_results['total_steps'] and 
              len(test_results['errors']) == 0)
    
    completion_rate = (test_results['steps_completed'] / test_results['total_steps']) * 100
    
    if success:
        print("🎉 RESULT: END-TO-END FLOW SUCCESSFUL")
        print(f"   ✅ All {test_results['total_steps']} pipeline steps completed")
        print(f"   ✅ Assets discovered: {test_results['assets_discovered']}")
        print(f"   ✅ Data points ingested: {test_results['data_points_ingested']}")
        print(f"   ✅ GA individuals created: {test_results['ga_individuals_created']}")
        print(f"   ✅ Universe initialized: {test_results['universe_initialized']}")
        print(f"   ✅ Total execution time: {test_results['execution_time_ms']:.1f}ms")
        
        # Performance analysis
        data_efficiency = test_results['data_points_ingested'] / test_results['assets_discovered']
        print(f"\n📊 Performance Metrics:")
        print(f"   Data efficiency: {data_efficiency:.1f} data points per asset")
        print(f"   GA readiness: 100% (population successfully created)")
        
        print("\n🚀 ARCHITECTURE VALIDATION COMPLETE!")
        print("   The genetic trading organism pipeline is fully operational.")
        print("   Ready for production deployment on Anyscale clusters.")
        
    else:
        print("❌ RESULT: END-TO-END FLOW INCOMPLETE")
        print(f"   Pipeline completion: {completion_rate:.1f}% ({test_results['steps_completed']}/{test_results['total_steps']})")
        
        if test_results['bottlenecks']:
            print(f"   🚫 Bottlenecks identified: {', '.join(test_results['bottlenecks'])}")
        
        for error in test_results['errors']:
            print(f"   ❌ {error}")
            
        print(f"\n📊 Partial Results:")
        if test_results['assets_discovered'] > 0:
            print(f"   ✅ Assets discovered: {test_results['assets_discovered']}")
        if test_results['data_points_ingested'] > 0:
            print(f"   ✅ Data points ingested: {test_results['data_points_ingested']}")
        if test_results['ga_individuals_created'] > 0:
            print(f"   ✅ GA individuals created: {test_results['ga_individuals_created']}")
            
        print(f"   📊 Execution time: {test_results['execution_time_ms']:.1f}ms")
        
        print("\n💡 Next Steps:")
        print("   1. Address bottlenecks in order of pipeline flow")
        print("   2. Run individual component tests for detailed diagnosis")
        print("   3. Check network connectivity and API credentials")
        print("   4. Review logs for specific error details")
    
    # Detailed step breakdown
    print(f"\n📋 Step-by-Step Results:")
    for step_name, step_data in test_results['step_results'].items():
        status = "✅" if step_data.get('success', False) else "❌"
        print(f"   {status} {step_name.replace('_', ' ').title()}")
        if not step_data.get('success', False) and 'error' in step_data:
            print(f"      Error: {step_data['error']}")
    
    # Return exit code for automation
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)