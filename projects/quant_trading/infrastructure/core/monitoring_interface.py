"""
Production Infrastructure - Core Monitoring Interface

This module defines universal monitoring contracts for genetic algorithm
infrastructure, enabling platform-agnostic observability with platform-specific
optimizations and integration with existing monitoring systems.

Research-Based Implementation:
- /research/anyscale/monitoring_cost_management.md - Anyscale monitoring patterns
- Existing src/execution/monitoring.py integration patterns
- PHASE_5B5_INFRASTRUCTURE_ARCHITECTURE.md - Monitoring requirements

Key Features:
- Platform-agnostic monitoring interface
- Genetic algorithm specific metrics collection
- Cost tracking and optimization alerts
- Integration with existing RealTimeMonitoringSystem
- Health check and alerting capabilities
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

# Integration with existing system
from .deployment_interface import PlatformType, DeploymentError
from .cluster_manager import ClusterState, WorkloadType

# Set up logging
logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected from genetic algorithm infrastructure"""
    # Resource metrics
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    
    # Genetic algorithm specific metrics
    GENETIC_EVALUATIONS_PER_SECOND = "genetic_evaluations_per_second"
    POPULATION_SIZE = "population_size"
    GENERATION_TIME = "generation_time"
    FITNESS_CONVERGENCE_RATE = "fitness_convergence_rate"
    FAILED_EVALUATIONS = "failed_evaluations"
    
    # Cost and efficiency metrics
    COST_PER_HOUR = "cost_per_hour"
    COST_PER_EVALUATION = "cost_per_evaluation"
    COST_EFFICIENCY_RATIO = "cost_efficiency_ratio"
    SPOT_INSTANCE_INTERRUPTIONS = "spot_instance_interruptions"
    
    # Ray cluster metrics
    RAY_TASKS_ACTIVE = "ray_tasks_active"
    RAY_TASKS_QUEUED = "ray_tasks_queued"
    RAY_OBJECT_STORE_USAGE = "ray_object_store_usage"
    RAY_WORKER_NODES = "ray_worker_nodes"
    
    # Health and availability metrics
    CLUSTER_HEALTH_SCORE = "cluster_health_score"
    NODE_FAILURE_RATE = "node_failure_rate"
    SERVICE_AVAILABILITY = "service_availability"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    """Types of alerts for genetic algorithm infrastructure"""
    # Cost alerts
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    COST_TREND_WARNING = "cost_trend_warning"
    SPOT_INTERRUPTION_HIGH = "spot_interruption_high"
    
    # Performance alerts
    HIGH_RESOURCE_UTILIZATION = "high_resource_utilization"
    LOW_GENETIC_THROUGHPUT = "low_genetic_throughput"
    EVALUATION_FAILURE_SPIKE = "evaluation_failure_spike"
    
    # Health alerts
    CLUSTER_UNHEALTHY = "cluster_unhealthy"
    NODE_FAILURES = "node_failures"
    RAY_CLUSTER_DEGRADED = "ray_cluster_degraded"
    
    # Integration alerts
    GENETIC_POOL_DISCONNECTED = "genetic_pool_disconnected"
    MONITORING_DATA_STALE = "monitoring_data_stale"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class AlertRule:
    """Configuration for monitoring alerts"""
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    metric_type: MetricType
    
    # Threshold configuration
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    evaluation_window: int  # seconds
    
    # Alert behavior
    notification_channels: List[str] = field(default_factory=list)
    suppress_duration: int = 300  # seconds to suppress duplicate alerts
    auto_resolve: bool = True
    
    # Genetic algorithm context
    applies_to_workloads: List[WorkloadType] = field(default_factory=list)
    cluster_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Alert details
    title: str
    description: str
    current_value: float
    threshold_value: float
    
    # Context information
    cluster_id: str
    platform: PlatformType
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Timing information
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    
    # Integration information
    genetic_pool_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringDashboard:
    """Configuration for monitoring dashboards"""
    dashboard_id: str
    title: str
    platform: PlatformType
    
    # Dashboard configuration
    refresh_interval: int = 30  # seconds
    time_range: int = 3600  # seconds (1 hour default)
    
    # Genetic algorithm specific panels
    genetic_metrics_panels: List[Dict[str, Any]] = field(default_factory=list)
    cost_optimization_panels: List[Dict[str, Any]] = field(default_factory=list)
    cluster_health_panels: List[Dict[str, Any]] = field(default_factory=list)
    
    # Integration panels
    trading_system_integration_panels: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dashboard metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MonitoringManager(ABC):
    """
    Universal monitoring interface for genetic algorithm infrastructure.
    
    This abstract base class defines the contract for monitoring genetic
    algorithm workloads across different platforms, with integration
    capabilities for existing monitoring systems.
    """
    
    def __init__(self, platform: PlatformType, config: Dict[str, Any]):
        """
        Initialize monitoring manager for specific platform.
        
        Args:
            platform: Target monitoring platform
            config: Platform-specific monitoring configuration
        """
        self.platform = platform
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{platform}")
        self._active_alerts: Dict[str, Alert] = {}
        self._metric_buffer: List[MetricPoint] = []
    
    @abstractmethod
    async def collect_metrics(self, cluster_id: str, 
                            metric_types: List[MetricType]) -> List[MetricPoint]:
        """
        Collect metrics from genetic algorithm infrastructure.
        
        This method gathers real-time metrics from the Ray cluster and
        genetic algorithm workloads, with platform-specific optimizations
        for efficient data collection.
        
        Args:
            cluster_id: Unique cluster identifier
            metric_types: List of metric types to collect
            
        Returns:
            List of MetricPoint instances with current values
            
        Raises:
            MonitoringError: If metric collection fails
        """
        pass
    
    @abstractmethod
    async def store_metrics(self, metrics: List[MetricPoint]) -> bool:
        """
        Store metrics in platform-specific monitoring backend.
        
        Args:
            metrics: List of metric points to store
            
        Returns:
            True if storage successful
        """
        pass
    
    @abstractmethod
    async def query_metrics(self, cluster_id: str,
                          metric_type: MetricType,
                          start_time: datetime,
                          end_time: datetime,
                          aggregation: str = "avg") -> List[MetricPoint]:
        """
        Query historical metrics with time range and aggregation.
        
        Args:
            cluster_id: Unique cluster identifier
            metric_type: Type of metric to query
            start_time: Query start time
            end_time: Query end time
            aggregation: Aggregation method ("avg", "sum", "max", "min")
            
        Returns:
            List of aggregated metric points
        """
        pass
    
    @abstractmethod
    async def create_alert_rule(self, rule: AlertRule) -> bool:
        """
        Create monitoring alert rule for genetic algorithm infrastructure.
        
        Args:
            rule: Alert rule configuration
            
        Returns:
            True if rule creation successful
        """
        pass
    
    @abstractmethod
    async def evaluate_alert_rules(self, cluster_id: str) -> List[Alert]:
        """
        Evaluate alert rules against current metrics.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            List of triggered alerts
        """
        pass
    
    @abstractmethod
    async def create_dashboard(self, dashboard: MonitoringDashboard) -> str:
        """
        Create monitoring dashboard for genetic algorithm workloads.
        
        Args:
            dashboard: Dashboard configuration
            
        Returns:
            Dashboard URL or identifier
        """
        pass
    
    @abstractmethod
    async def get_cluster_health_summary(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get comprehensive health summary for genetic algorithm cluster.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            Dictionary with health summary
        """
        pass
    
    # Integration methods for existing monitoring system
    
    async def integrate_with_trading_monitor(self, 
                                           trading_monitor_config: Dict[str, Any]) -> bool:
        """
        Integrate with existing RealTimeMonitoringSystem.
        
        This method establishes integration with the existing monitoring
        system in src/execution/monitoring.py to provide unified observability
        across the entire trading system.
        
        Args:
            trading_monitor_config: Configuration for trading system integration
            
        Returns:
            True if integration successful
        """
        try:
            # Integration logic would connect with existing RealTimeMonitoringSystem
            self.logger.info("Integrating with existing trading system monitoring")
            
            # Set up metric forwarding for genetic algorithm metrics
            genetic_metrics = [
                MetricType.GENETIC_EVALUATIONS_PER_SECOND,
                MetricType.POPULATION_SIZE,
                MetricType.GENERATION_TIME,
                MetricType.FITNESS_CONVERGENCE_RATE,
                MetricType.COST_PER_EVALUATION
            ]
            
            # Configure alerts that align with trading system monitoring
            trading_alert_rules = [
                AlertRule(
                    rule_id="genetic_low_throughput",
                    alert_type=AlertType.LOW_GENETIC_THROUGHPUT,
                    severity=AlertSeverity.WARNING,
                    metric_type=MetricType.GENETIC_EVALUATIONS_PER_SECOND,
                    threshold_value=1.0,
                    comparison_operator="lt",
                    evaluation_window=300,
                    notification_channels=trading_monitor_config.get("alert_channels", [])
                ),
                AlertRule(
                    rule_id="genetic_cost_budget_exceeded",
                    alert_type=AlertType.COST_BUDGET_EXCEEDED,
                    severity=AlertSeverity.CRITICAL,
                    metric_type=MetricType.COST_PER_HOUR,
                    threshold_value=trading_monitor_config.get("max_hourly_cost", 50.0),
                    comparison_operator="gt",
                    evaluation_window=60,
                    notification_channels=trading_monitor_config.get("cost_alert_channels", [])
                )
            ]
            
            # Create alert rules
            for rule in trading_alert_rules:
                await self.create_alert_rule(rule)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with trading monitor: {e}")
            return False
    
    async def stream_genetic_metrics(self, cluster_id: str) -> AsyncGenerator[MetricPoint, None]:
        """
        Stream real-time genetic algorithm metrics.
        
        This method provides a continuous stream of genetic algorithm
        metrics for integration with existing monitoring dashboards.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Yields:
            MetricPoint instances with real-time values
        """
        try:
            genetic_metric_types = [
                MetricType.GENETIC_EVALUATIONS_PER_SECOND,
                MetricType.POPULATION_SIZE,
                MetricType.GENERATION_TIME,
                MetricType.FITNESS_CONVERGENCE_RATE,
                MetricType.FAILED_EVALUATIONS,
                MetricType.COST_PER_HOUR,
                MetricType.CLUSTER_HEALTH_SCORE
            ]
            
            while True:
                try:
                    metrics = await self.collect_metrics(cluster_id, genetic_metric_types)
                    for metric in metrics:
                        yield metric
                    
                    await asyncio.sleep(10)  # Stream every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in metric streaming: {e}")
                    await asyncio.sleep(30)  # Back off on error
                    
        except asyncio.CancelledError:
            self.logger.info(f"Metric streaming cancelled for cluster {cluster_id}")
            return
    
    def calculate_genetic_efficiency_metrics(self, 
                                           metrics: List[MetricPoint]) -> Dict[str, float]:
        """
        Calculate efficiency metrics specific to genetic algorithm workloads.
        
        Args:
            metrics: List of collected metric points
            
        Returns:
            Dictionary with efficiency calculations
        """
        metric_dict = {m.metric_type: m.value for m in metrics}
        
        # Calculate genetic algorithm efficiency
        evaluations_per_second = metric_dict.get(MetricType.GENETIC_EVALUATIONS_PER_SECOND, 0)
        cost_per_hour = metric_dict.get(MetricType.COST_PER_HOUR, 0)
        cpu_utilization = metric_dict.get(MetricType.CPU_UTILIZATION, 0)
        
        # Efficiency calculations
        cost_per_evaluation = (cost_per_hour / 3600 / evaluations_per_second 
                              if evaluations_per_second > 0 else float('inf'))
        
        resource_efficiency = cpu_utilization / 100.0  # Convert to ratio
        
        cost_efficiency = (1.0 / cost_per_evaluation 
                          if cost_per_evaluation > 0 and cost_per_evaluation != float('inf') 
                          else 0.0)
        
        # Overall efficiency score (0.0 to 1.0)
        efficiency_score = (resource_efficiency + min(cost_efficiency, 1.0)) / 2.0
        
        return {
            "cost_per_evaluation": cost_per_evaluation,
            "resource_efficiency": resource_efficiency,
            "cost_efficiency": min(cost_efficiency, 1.0),
            "overall_efficiency_score": efficiency_score,
            "evaluations_per_dollar": (evaluations_per_second * 3600 / cost_per_hour 
                                     if cost_per_hour > 0 else 0)
        }
    
    async def generate_cost_optimization_report(self, cluster_id: str,
                                              time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Generate cost optimization report for genetic algorithm workloads.
        
        Args:
            cluster_id: Unique cluster identifier
            time_range_hours: Time range for analysis in hours
            
        Returns:
            Dictionary with cost optimization recommendations
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Query cost and performance metrics
            cost_metrics = await self.query_metrics(
                cluster_id, MetricType.COST_PER_HOUR, start_time, end_time, "avg"
            )
            
            evaluation_metrics = await self.query_metrics(
                cluster_id, MetricType.GENETIC_EVALUATIONS_PER_SECOND, start_time, end_time, "avg"
            )
            
            utilization_metrics = await self.query_metrics(
                cluster_id, MetricType.CPU_UTILIZATION, start_time, end_time, "avg"
            )
            
            # Calculate optimization opportunities
            avg_cost = sum(m.value for m in cost_metrics) / len(cost_metrics) if cost_metrics else 0
            avg_evaluations = (sum(m.value for m in evaluation_metrics) / len(evaluation_metrics) 
                             if evaluation_metrics else 0)
            avg_utilization = (sum(m.value for m in utilization_metrics) / len(utilization_metrics) 
                             if utilization_metrics else 0)
            
            recommendations = []
            potential_savings = 0.0
            
            # Analyze utilization for right-sizing recommendations
            if avg_utilization < 50:
                recommendations.append({
                    "type": "right_sizing",
                    "description": "Consider reducing cluster size due to low utilization",
                    "potential_savings_percent": 30,
                    "confidence": 0.8
                })
                potential_savings += avg_cost * 0.3
            
            # Analyze spot instance opportunities
            if avg_cost > 10:  # Threshold for spot consideration
                recommendations.append({
                    "type": "spot_instances",
                    "description": "Consider using more spot instances for cost reduction",
                    "potential_savings_percent": 60,
                    "confidence": 0.7
                })
                potential_savings += avg_cost * 0.6
            
            # Analyze scheduling opportunities
            recommendations.append({
                "type": "scheduling",
                "description": "Consider scheduling genetic workloads during off-peak hours",
                "potential_savings_percent": 20,
                "confidence": 0.9
            })
            
            return {
                "cluster_id": cluster_id,
                "analysis_period_hours": time_range_hours,
                "current_metrics": {
                    "avg_cost_per_hour": avg_cost,
                    "avg_evaluations_per_second": avg_evaluations,
                    "avg_cpu_utilization": avg_utilization,
                    "cost_per_evaluation": avg_cost / (avg_evaluations * 3600) if avg_evaluations > 0 else 0
                },
                "optimization_recommendations": recommendations,
                "potential_monthly_savings": potential_savings * 24 * 30,
                "generated_at": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate cost optimization report: {e}")
            return {"error": str(e)}


class MonitoringError(Exception):
    """Base exception for monitoring operations"""
    pass


class MetricCollectionError(MonitoringError):
    """Raised when metric collection fails"""
    pass


class AlertRuleError(MonitoringError):
    """Raised when alert rule operations fail"""
    pass


class DashboardError(MonitoringError):
    """Raised when dashboard operations fail"""
    pass