#!/usr/bin/env python3
"""
Phase 4B Connection Optimization Validation Test

This test validates the retail trading connection optimization improvements
introduced in Phase 4B, comparing performance across different trading
session profiles (scalping, intraday, swing).

Target Improvements:
- API Response Time: Maintain <150ms average
- Resource Efficiency: Optimized pool sizing per trading style
- Session Management: Zero warnings with enhanced performance monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.execution.trading_system_manager import (
    create_scalping_trading_system, 
    create_intraday_trading_system,
    create_swing_trading_system,
    TradingSystemManager
)
from src.execution.retail_connection_optimizer import (
    SCALPING_SESSION, INTRADAY_SESSION, SWING_SESSION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_trading_session_optimization(session_name: str, 
                                          create_system_func) -> Dict:
    """Test a specific trading session optimization.
    
    Args:
        session_name: Name of the trading session (e.g., "Scalping")
        create_system_func: Factory function to create the trading system
        
    Returns:
        Performance metrics dictionary
    """
    logger.info(f"🧪 Testing {session_name} Trading Session Optimization")
    
    start_time = time.time()
    
    async with create_system_func() as manager:
        logger.info(f"✅ {session_name} system initialized")
        
        # Test multiple operations to gather performance data
        test_operations = [
            "get_fear_greed_index",
            "collect_monitoring_snapshot",
            "get_fear_greed_index",  # Test caching
            "collect_monitoring_snapshot"
        ]
        
        operation_times = []
        
        for operation in test_operations:
            op_start = time.time()
            try:
                result = await manager.execute_trading_operation(operation)
                op_time = (time.time() - op_start) * 1000  # Convert to ms
                operation_times.append(op_time)
                logger.info(f"  ⚡ {operation}: {op_time:.1f}ms")
            except Exception as e:
                logger.error(f"  ❌ {operation} failed: {e}")
                operation_times.append(999)  # Penalty for failed operations
        
        # Get comprehensive system health
        health_summary = manager.get_system_health_summary()
        
        # Calculate session metrics
        total_time = (time.time() - start_time) * 1000
        avg_operation_time = sum(operation_times) / len(operation_times)
        
        session_metrics = {
            'session_name': session_name,
            'total_initialization_time_ms': total_time - sum(operation_times),
            'average_operation_time_ms': avg_operation_time,
            'fastest_operation_ms': min(operation_times),
            'slowest_operation_ms': max(operation_times),
            'health_score': health_summary['health_score'],
            'total_operations': len(test_operations),
            'connection_optimization': health_summary.get('connection_optimization', {}),
            'performance_rating': 'excellent' if avg_operation_time < 100 else 
                                'good' if avg_operation_time < 150 else 
                                'acceptable' if avg_operation_time < 200 else 'poor'
        }
        
        logger.info(f"📊 {session_name} Performance: {avg_operation_time:.1f}ms avg, "
                   f"Health: {health_summary['health_score']:.1f}%, "
                   f"Rating: {session_metrics['performance_rating']}")
        
        return session_metrics


async def run_phase_4b_validation():
    """Run comprehensive Phase 4B optimization validation."""
    
    logger.info("🚀 Starting Phase 4B Connection Optimization Validation")
    logger.info("=" * 60)
    
    # Test different trading session profiles
    trading_sessions = [
        ("Scalping", create_scalping_trading_system),
        ("Intraday", create_intraday_trading_system), 
        ("Swing", create_swing_trading_system)
    ]
    
    all_results = []
    
    for session_name, create_func in trading_sessions:
        try:
            result = await test_trading_session_optimization(session_name, create_func)
            all_results.append(result)
            logger.info("")  # Empty line for readability
        except Exception as e:
            logger.error(f"❌ {session_name} session test failed: {e}")
            all_results.append({
                'session_name': session_name,
                'error': str(e),
                'performance_rating': 'failed'
            })
    
    # Performance comparison analysis
    logger.info("📈 PHASE 4B OPTIMIZATION RESULTS SUMMARY")
    logger.info("=" * 60)
    
    successful_tests = [r for r in all_results if 'error' not in r]
    
    if successful_tests:
        # Find best and worst performing sessions
        best_session = min(successful_tests, key=lambda x: x['average_operation_time_ms'])
        worst_session = max(successful_tests, key=lambda x: x['average_operation_time_ms'])
        
        # Calculate improvement metrics
        avg_response_time = sum(r['average_operation_time_ms'] for r in successful_tests) / len(successful_tests)
        excellent_count = sum(1 for r in successful_tests if r['performance_rating'] == 'excellent')
        good_count = sum(1 for r in successful_tests if r['performance_rating'] == 'good')
        
        logger.info(f"✅ Overall Average Response Time: {avg_response_time:.1f}ms")
        logger.info(f"⭐ Best Performing Session: {best_session['session_name']} ({best_session['average_operation_time_ms']:.1f}ms)")
        logger.info(f"📊 Performance Distribution: {excellent_count} excellent, {good_count} good")
        
        # Check if we met Phase 4B targets
        target_met = avg_response_time <= 150  # Target: <150ms average
        health_scores = [r['health_score'] for r in successful_tests if 'health_score' in r]
        health_target_met = all(score >= 100 for score in health_scores)
        
        logger.info("")
        logger.info("🎯 PHASE 4B TARGET ACHIEVEMENT")
        logger.info("-" * 40)
        logger.info(f"📈 Response Time Target (<150ms): {'✅ ACHIEVED' if target_met else '❌ NOT MET'}")
        logger.info(f"💚 Health Score Target (100%): {'✅ ACHIEVED' if health_target_met else '❌ NOT MET'}")
        
        if target_met and health_target_met:
            logger.info("🎉 PHASE 4B OPTIMIZATION: SUCCESSFUL!")
            logger.info("   - Connection optimization implemented successfully")
            logger.info("   - All trading session profiles performing optimally") 
            logger.info("   - Ready for production retail trading")
        else:
            logger.info("⚠️  PHASE 4B OPTIMIZATION: NEEDS IMPROVEMENT")
            logger.info(f"   - Current avg response time: {avg_response_time:.1f}ms")
            logger.info("   - Consider further connection pool tuning")
    
    else:
        logger.error("❌ No successful tests completed - optimization validation failed")
    
    # Detailed session analysis
    logger.info("")
    logger.info("📋 DETAILED SESSION ANALYSIS")
    logger.info("-" * 40)
    
    for result in all_results:
        if 'error' in result:
            logger.error(f"❌ {result['session_name']}: {result['error']}")
        else:
            conn_opt = result.get('connection_optimization', {})
            logger.info(f"📊 {result['session_name']}:")
            logger.info(f"   - Avg Response: {result['average_operation_time_ms']:.1f}ms")
            logger.info(f"   - Health Score: {result.get('health_score', 'N/A')}%")
            logger.info(f"   - Performance: {result['performance_rating']}")
            if conn_opt.get('overall'):
                overall = conn_opt['overall']
                logger.info(f"   - Total Requests: {overall.get('total_requests', 0)}")
                logger.info(f"   - Session Duration: {conn_opt.get('session_duration_minutes', 0):.1f} min")
    
    logger.info("")
    logger.info("✅ Phase 4B Connection Optimization Validation Complete")
    
    return all_results


if __name__ == "__main__":
    """Run Phase 4B optimization validation test."""
    
    print("=== Phase 4B: Retail Trading Connection Optimization Validation ===")
    print("Testing optimized connection pools for scalping, intraday, and swing trading")
    print("")
    
    # Run the validation
    results = asyncio.run(run_phase_4b_validation())
    
    # Final summary
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        avg_performance = sum(r['average_operation_time_ms'] for r in successful_results) / len(successful_results)
        print(f"\\nFINAL RESULT: Average Response Time = {avg_performance:.1f}ms")
        print("TARGET: <150ms for retail trading optimization")
        
        if avg_performance <= 150:
            print("🎉 PHASE 4B OPTIMIZATION: SUCCESS!")
        else:
            print("⚠️  PHASE 4B OPTIMIZATION: NEEDS FURTHER WORK")
    else:
        print("❌ PHASE 4B VALIDATION FAILED - No successful tests")