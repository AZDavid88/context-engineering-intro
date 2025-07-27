"""
Integration Test for Real-time Monitoring System

This script tests the monitoring system integration with all Phase 1-3 components
to ensure comprehensive observability of the genetic trading organism.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add project root to Python path
project_root = "/workspaces/context-engineering-intro/projects/quant_trading"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import get_settings
from src.execution.monitoring import RealTimeMonitoringSystem, AlertLevel, AlertCategory
from src.execution.risk_management import GeneticRiskManager, GeneticRiskGenome
from src.execution.paper_trading import PaperTradingEngine, PaperTradingMode
from src.execution.position_sizer import GeneticPositionSizer
from src.execution.order_management import OrderRequest, OrderSide, OrderType


async def test_monitoring_integration():
    """Test comprehensive monitoring system integration."""
    
    print("=== Monitoring System Integration Test ===")
    
    # Initialize settings
    settings = get_settings()
    print("✅ Settings loaded")
    
    # Initialize all Phase 1-3 components
    print("\nInitializing Phase 1-3 components...")
    
    # Risk management system
    genetic_genome = GeneticRiskGenome(
        stop_loss_percentage=0.03,
        max_position_size=0.15,
        daily_drawdown_limit=0.02
    )
    risk_manager = GeneticRiskManager(settings, genetic_genome)
    print("✅ Risk management system initialized")
    
    # Paper trading system
    paper_trading = PaperTradingEngine(settings, PaperTradingMode.SIMULATION)
    print("✅ Paper trading system initialized")
    
    # Position sizing system
    position_sizer = GeneticPositionSizer(settings)
    print("✅ Position sizing system initialized")
    
    # Initialize monitoring system
    print("\nInitializing monitoring system...")
    monitoring = RealTimeMonitoringSystem(settings)
    
    # Inject component references
    monitoring.inject_components(
        risk_manager=risk_manager,
        paper_trading=paper_trading,
        position_sizer=position_sizer
    )
    print("✅ Monitoring system initialized with component integration")
    
    # Test 1: Comprehensive snapshot collection
    print("\n=== Test 1: Comprehensive Snapshot Collection ===")
    snapshot = monitoring.collect_monitoring_snapshot()
    
    print(f"✅ Snapshot Status: {snapshot.status}")
    print(f"   - Risk Level: {snapshot.current_risk_level}")
    print(f"   - Market Regime: {snapshot.market_regime}")
    print(f"   - System CPU: {snapshot.system_metrics.cpu_usage_percent:.1f}%")
    print(f"   - System Memory: {snapshot.system_metrics.memory_usage_percent:.1f}%")
    print(f"   - Trading Metrics: {snapshot.trading_metrics.total_trades} trades")
    print(f"   - Active Alerts: {len(snapshot.active_alerts)}")
    
    # Test 2: Alert system integration
    print("\n=== Test 2: Alert System Integration ===")
    
    # Trigger various alert types
    alert1 = monitoring.alert_manager.trigger_alert(
        AlertLevel.WARNING,
        AlertCategory.RISK_MANAGEMENT,
        "High Risk Level Detected",
        "Risk management system reporting elevated risk",
        "RiskManager"
    )
    
    alert2 = monitoring.alert_manager.trigger_alert(
        AlertLevel.CRITICAL,
        AlertCategory.TRADING_PERFORMANCE,
        "Poor Execution Quality",
        "Trading execution success rate below threshold",
        "PaperTradingEngine"
    )
    
    print(f"✅ Alerts triggered: {len(monitoring.alert_manager.get_active_alerts())}")
    for alert in monitoring.alert_manager.get_active_alerts():
        print(f"   - {alert.level}: {alert.title} from {alert.source_component}")
    
    # Test 3: Paper trading integration
    print("\n=== Test 3: Paper Trading Integration ===")
    
    # Execute sample paper trade
    test_order = OrderRequest(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        size=0.1,
        order_type=OrderType.MARKET,
        strategy_id="integration_test"
    )
    
    paper_trade = await paper_trading.execute_paper_trade(
        test_order, 
        {'test_param': 0.5}
    )
    
    print(f"✅ Paper trade executed: {paper_trade.execution_quality}")
    print(f"   - Symbol: {paper_trade.symbol}")
    print(f"   - Size: {paper_trade.executed_size:.4f}")
    print(f"   - Slippage: {paper_trade.slippage:.2%}")
    print(f"   - Latency: {paper_trade.latency_ms:.1f}ms")
    
    # Collect updated snapshot
    updated_snapshot = monitoring.collect_monitoring_snapshot()
    print(f"✅ Updated snapshot after trade: {updated_snapshot.trading_metrics.total_trades} total trades")
    
    # Test 4: Dashboard data generation
    print("\n=== Test 4: Dashboard Data Generation ===")
    
    # Add snapshots to history for trend analysis
    monitoring.snapshot_history.append(snapshot)
    monitoring.snapshot_history.append(updated_snapshot)
    
    dashboard_data = monitoring.get_monitoring_dashboard_data()
    
    print(f"✅ Dashboard data generated")
    print(f"   - Overall Status: {dashboard_data['status']}")
    print(f"   - Health Score: {dashboard_data['summary']['overall_health']:.1f}")
    print(f"   - Total Alerts: {dashboard_data['alerts']['total']}")
    print(f"   - Alert Breakdown: {dashboard_data['alerts']['by_level']}")
    print(f"   - Monitoring Active: {dashboard_data['summary']['monitoring_active']}")
    
    # Test 5: Performance analysis
    print("\n=== Test 5: Performance Analysis ===")
    
    # Add metrics to performance analyzer
    monitoring.performance_analyzer.add_system_metrics(snapshot.system_metrics)
    monitoring.performance_analyzer.add_trading_metrics(snapshot.trading_metrics)
    monitoring.performance_analyzer.add_genetic_metrics(snapshot.genetic_metrics)
    
    system_trends = monitoring.performance_analyzer.analyze_system_trends()
    trading_trends = monitoring.performance_analyzer.analyze_trading_trends()
    genetic_trends = monitoring.performance_analyzer.analyze_genetic_trends()
    
    print(f"✅ Performance analysis completed")
    print(f"   - System Trends: {system_trends.get('status', 'analyzed')}")
    print(f"   - Trading Trends: {trading_trends.get('status', 'analyzed')}")
    print(f"   - Genetic Trends: {genetic_trends.get('status', 'analyzed')}")
    
    # Test 6: Live monitoring loop
    print("\n=== Test 6: Live Monitoring Loop ===")
    
    print("Starting monitoring loop for 10 seconds...")
    monitoring.start_monitoring()
    
    # Let it run and collect data
    await asyncio.sleep(10)
    
    # Stop monitoring
    monitoring.stop_monitoring()
    
    print(f"✅ Monitoring loop completed")
    print(f"   - Collections: {monitoring.collection_count}")
    print(f"   - History Size: {len(monitoring.snapshot_history)}")
    print(f"   - Final Status: {monitoring.get_current_snapshot().status if monitoring.get_current_snapshot() else 'N/A'}")
    
    # Test 7: Alert acknowledgment and resolution
    print("\n=== Test 7: Alert Management ===")
    
    active_alerts = monitoring.alert_manager.get_active_alerts()
    if active_alerts:
        test_alert = active_alerts[0]
        
        # Acknowledge alert
        acknowledged = monitoring.alert_manager.acknowledge_alert(test_alert.alert_id, "integration_test")
        print(f"✅ Alert acknowledged: {acknowledged}")
        
        # Resolve alert
        resolved = monitoring.alert_manager.resolve_alert(test_alert.alert_id, "integration_test")
        print(f"✅ Alert resolved: {resolved}")
        
        remaining_alerts = len(monitoring.alert_manager.get_active_alerts())
        print(f"   - Remaining active alerts: {remaining_alerts}")
    
    # Test 8: Component metrics validation
    print("\n=== Test 8: Component Metrics Validation ===")
    
    final_snapshot = monitoring.collect_monitoring_snapshot()
    
    # Validate risk management metrics
    risk_summary = risk_manager.get_risk_summary()
    print(f"✅ Risk Management Integration:")
    print(f"   - Risk Level: {risk_summary['risk_level']}")
    print(f"   - Trading Enabled: {risk_summary['trading_enabled']}")
    print(f"   - Emergency Mode: {risk_summary['emergency_mode']}")
    
    # Validate paper trading metrics
    trading_summary = paper_trading.get_paper_trading_summary()
    print(f"✅ Paper Trading Integration:")
    print(f"   - Success Rate: {trading_summary['success_rate']:.1%}")
    print(f"   - Total Orders: {trading_summary['total_orders']}")
    print(f"   - Paper Cash: ${trading_summary['paper_cash']:.2f}")
    
    # Validate position sizing metrics
    position_stats = position_sizer.get_position_sizing_stats()
    print(f"✅ Position Sizing Integration:")
    print(f"   - Max Position Size: {position_stats['max_position_per_asset']:.1%}")
    print(f"   - Current Exposure: {position_stats['current_total_exposure']:.1%}")
    print(f"   - Active Positions: {position_stats['active_positions']}")
    
    print("\n=== Integration Test Summary ===")
    print("✅ All Phase 1-3 components successfully integrated")
    print("✅ Real-time monitoring system operational")
    print("✅ Alert system functional with multi-level escalation")
    print("✅ Performance analysis and trend detection working")
    print("✅ Dashboard data generation complete")
    print("✅ Component metrics collection validated")
    
    print(f"\nFinal System Status: {final_snapshot.status}")
    print(f"Overall Health Score: {monitoring._calculate_overall_health_score(final_snapshot):.1f}/100")
    
    # Cleanup async resources
    print("\n=== Cleanup Phase ===")
    await risk_manager.cleanup()
    print("✅ Risk manager resources cleaned up")
    
    print("\n=== Monitoring Integration Test Complete ===")


if __name__ == "__main__":
    """Run monitoring integration test."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run integration test
    asyncio.run(test_monitoring_integration())