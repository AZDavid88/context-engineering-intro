"""
Unified Monitoring System Interface - Backward compatibility layer.
Provides unified access to modular monitoring components while maintaining existing API.
"""

# Import all components from modular structure
from .monitoring_core import (
    MonitoringEngine, MetricCollector, SystemHealthTracker,
    MonitoringSnapshot, SystemHealthMetrics, GeneticEvolutionMetrics, 
    TradingPerformanceMetrics, MonitoringStatus, AlertLevel, AlertCategory
)
from .monitoring_dashboard import DashboardInterface, DataVisualization
from .monitoring_alerts import (
    AlertManager, NotificationDispatcher, EscalationManager, 
    AlertChecker, MonitoringAlert, NotificationChannel
)

# Backward compatibility aliases for existing imports
RealTimeMonitoringSystem = MonitoringEngine
MonitoringSystem = MonitoringEngine

# Export all components for unified access
__all__ = [
    # Core monitoring
    'MonitoringEngine', 'MetricCollector', 'SystemHealthTracker',
    'MonitoringSnapshot', 'SystemHealthMetrics', 'GeneticEvolutionMetrics',
    'TradingPerformanceMetrics', 'MonitoringStatus', 'AlertLevel', 'AlertCategory',
    
    # Dashboard components
    'DashboardInterface', 'DataVisualization',
    
    # Alert components
    'AlertManager', 'NotificationDispatcher', 'EscalationManager',
    'AlertChecker', 'MonitoringAlert', 'NotificationChannel',
    
    # Backward compatibility aliases
    'RealTimeMonitoringSystem', 'MonitoringSystem'
]


class UnifiedMonitoringSystem:
    """
    Unified monitoring system that integrates all monitoring components.
    Provides a single interface for complete monitoring functionality.
    """
    
    def __init__(self, settings=None):
        """Initialize unified monitoring system."""
        # Core monitoring engine
        self.engine = MonitoringEngine(settings)
        
        # Dashboard interface
        self.dashboard = DashboardInterface(self.engine.monitoring_snapshots)
        
        # Alert management
        self.alert_manager = AlertManager(settings)
        self.notification_dispatcher = NotificationDispatcher(settings)
        self.escalation_manager = EscalationManager(self.alert_manager, self.notification_dispatcher)
        self.alert_checker = AlertChecker(self.alert_manager)
        
        # Data visualization
        self.visualization = DataVisualization()
        
        # Setup alert callbacks
        self._setup_alert_integration()
    
    def _setup_alert_integration(self):
        """Setup integration between monitoring and alerting systems."""
        # Register alert checker with monitoring engine
        def check_alerts_callback(snapshot):
            self.alert_checker.check_all_conditions(snapshot)
            
        # Register notification dispatcher with alert manager
        def dispatch_notification_callback(alert):
            self.notification_dispatcher.dispatch_alert(alert)
            self.escalation_manager.start_escalation_timer(alert)
        
        # Register callbacks
        self.alert_manager.register_alert_callback(AlertLevel.WARNING, dispatch_notification_callback)
        self.alert_manager.register_alert_callback(AlertLevel.CRITICAL, dispatch_notification_callback)
        self.alert_manager.register_alert_callback(AlertLevel.EMERGENCY, dispatch_notification_callback)
    
    def start_monitoring(self, risk_manager=None, paper_trading_engine=None, position_sizer=None):
        """Start the unified monitoring system."""
        return self.engine.start_monitoring(risk_manager, paper_trading_engine, position_sizer)
    
    def stop_monitoring(self):
        """Stop the unified monitoring system."""
        return self.engine.stop_monitoring()
    
    def get_system_health(self):
        """Get current system health status."""
        return self.engine.get_system_health()
    
    def get_latest_snapshot(self):
        """Get the latest monitoring snapshot."""
        return self.engine.get_latest_snapshot()
    
    def get_dashboard_data(self):
        """Get complete dashboard data."""
        return self.dashboard.get_dashboard_data()
    
    def get_active_alerts(self, level=None, category=None):
        """Get active alerts with optional filtering."""
        return self.alert_manager.get_active_alerts(level, category)
    
    def acknowledge_alert(self, alert_id, acknowledged_by="system"):
        """Acknowledge an active alert."""
        result = self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        if result:
            self.escalation_manager.stop_escalation_timer(alert_id)
        return result
    
    def resolve_alert(self, alert_id, resolved_by="system"):
        """Resolve an active alert."""
        result = self.alert_manager.resolve_alert(alert_id, resolved_by)
        if result:
            self.escalation_manager.stop_escalation_timer(alert_id)
        return result
    
    def create_alert(self, level, category, title, message, source_component, 
                    threshold_value=None, current_value=None, context_data=None):
        """Create a new alert."""
        return self.alert_manager.trigger_alert(
            level, category, title, message, source_component,
            threshold_value, current_value, context_data
        )


# For maximum backward compatibility, also provide the unified system as default aliases
RealTimeMonitoringSystem = UnifiedMonitoringSystem
MonitoringSystem = UnifiedMonitoringSystem