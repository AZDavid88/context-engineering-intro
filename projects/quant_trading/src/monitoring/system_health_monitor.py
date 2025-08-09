"""
System Health Monitor - Comprehensive Distributed System Health Assessment

Provides comprehensive system health monitoring for the ultra-compressed evolution
system, enhancing the existing RealTimeMonitoringSystem with distributed health
assessment, predictive failure detection, and system-wide coordination.

Integration Architecture:
- Enhances RealTimeMonitoringSystem for monitoring coordination
- Integrates with ResilienceManager for failure pattern analysis
- Works with SessionHealth/SessionStatus from TradingSystemManager
- Coordinates with AlertingSystem for health notifications
- Monitors Ray cluster health and distributed operation status

This health monitor follows established monitoring patterns while adding
enterprise-grade health assessment for industrial-scale genetic evolution.
"""

import asyncio
import logging
import time
import statistics
import psutil
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import platform
from pathlib import Path
from collections import defaultdict, deque

# Verified imports from architecture analysis
from src.config.settings import get_settings, Settings
from src.execution.monitoring import RealTimeMonitoringSystem
from src.execution.trading_system_manager import SessionHealth, SessionStatus
from src.execution.alerting_system import AlertingSystem, AlertPriority
from src.execution.resilience_manager import ResilienceManager, ResilienceState

# Ray imports (conditional for distributed monitoring)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """System health status levels."""
    EXCELLENT = "excellent"    # All metrics green, system performing optimally
    GOOD = "good"             # Minor issues, system stable
    WARNING = "warning"       # Issues detected, may impact performance
    CRITICAL = "critical"     # Serious issues, system functionality at risk
    FAILING = "failing"       # System failure imminent or occurring


class ComponentType(str, Enum):
    """Types of system components to monitor."""
    SYSTEM_RESOURCES = "system_resources"
    RAY_CLUSTER = "ray_cluster"
    DATABASE = "database"
    GENETIC_EVOLUTION = "genetic_evolution"
    VALIDATION_PIPELINE = "validation_pipeline"
    DEPLOYMENT_SYSTEM = "deployment_system"
    EXTERNAL_APIs = "external_apis"
    MONITORING_SYSTEM = "monitoring_system"


@dataclass
class HealthMetric:
    """Individual health metric measurement."""
    
    name: str
    value: float
    unit: str = ""
    status: HealthStatus = HealthStatus.GOOD
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ComponentHealth:
    """Health assessment for a system component."""
    
    component_name: str
    component_type: ComponentType
    overall_status: HealthStatus = HealthStatus.GOOD
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    check_duration_ms: float = 0.0
    issues_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "overall_status": self.overall_status.value,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "last_check": self.last_check.isoformat(),
            "check_duration_ms": self.check_duration_ms,
            "issues_detected": self.issues_detected,
            "recommendations": self.recommendations
        }


@dataclass
class SystemHealthSnapshot:
    """Complete system health snapshot."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_status: HealthStatus = HealthStatus.GOOD
    overall_score: float = 1.0  # 0.0 = critical failure, 1.0 = excellent health
    
    # Component health summaries
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    
    # System-wide metrics
    system_uptime: float = 0.0
    total_memory_usage_percent: float = 0.0
    total_cpu_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Distributed system metrics (if Ray available)
    ray_cluster_size: int = 0
    ray_available_resources: Dict[str, float] = field(default_factory=dict)
    ray_cluster_healthy: bool = True
    
    # Health trends
    health_trend: str = "stable"  # improving, stable, degrading
    predicted_issues: List[str] = field(default_factory=list)
    
    # Performance indicators
    average_response_time_ms: float = 0.0
    system_load_average: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "system_metrics": {
                "uptime": self.system_uptime,
                "memory_usage_percent": self.total_memory_usage_percent,
                "cpu_usage_percent": self.total_cpu_usage_percent,
                "disk_usage_percent": self.disk_usage_percent,
                "system_load_average": self.system_load_average,
                "average_response_time_ms": self.average_response_time_ms,
                "error_rate": self.error_rate
            },
            "ray_cluster": {
                "size": self.ray_cluster_size,
                "available_resources": self.ray_available_resources,
                "healthy": self.ray_cluster_healthy
            },
            "health_analysis": {
                "trend": self.health_trend,
                "predicted_issues": self.predicted_issues
            }
        }


class SystemHealthMonitor:
    """Comprehensive system health monitor for distributed evolution system."""
    
    def __init__(self,
                 settings: Optional[Settings] = None,
                 monitoring_system: Optional[RealTimeMonitoringSystem] = None,
                 resilience_manager: Optional[ResilienceManager] = None):
        """
        Initialize system health monitor.
        
        Args:
            settings: System settings
            monitoring_system: Existing monitoring system to enhance
            resilience_manager: Resilience manager for failure coordination
        """
        
        self.settings = settings or get_settings()
        
        # Integration with existing systems (verified available)
        self.monitoring_system = monitoring_system or RealTimeMonitoringSystem()
        self.alerting = AlertingSystem()
        self.resilience_manager = resilience_manager
        
        # Health monitoring state
        self.health_history: deque = deque(maxlen=1000)  # Last 1000 health snapshots
        self.component_checkers: Dict[str, Callable] = {}
        self.health_thresholds: Dict[str, Dict[str, float]] = self._initialize_health_thresholds()
        
        # Monitoring configuration
        self.check_interval_seconds = 60  # Health check every minute
        self.health_monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.system_start_time = time.time()
        self.last_health_check: Optional[datetime] = None
        
        # Predictive analysis
        self.issue_prediction_enabled = True
        self.trend_analysis_window_minutes = 30
        
        logger.info("SystemHealthMonitor initialized with comprehensive health assessment")
    
    def _initialize_health_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize health thresholds for different metrics."""
        
        return {
            "system_resources": {
                "memory_usage_warning": 80.0,      # 80% memory usage
                "memory_usage_critical": 95.0,     # 95% memory usage
                "cpu_usage_warning": 80.0,         # 80% CPU usage
                "cpu_usage_critical": 95.0,        # 95% CPU usage
                "disk_usage_warning": 85.0,        # 85% disk usage
                "disk_usage_critical": 95.0,       # 95% disk usage
                "load_average_warning": 8.0,       # Load average > 8
                "load_average_critical": 16.0      # Load average > 16
            },
            "ray_cluster": {
                "node_failure_warning": 0.1,       # >10% nodes down
                "node_failure_critical": 0.25,     # >25% nodes down
                "resource_usage_warning": 0.8,     # >80% resources used
                "resource_usage_critical": 0.95    # >95% resources used
            },
            "performance": {
                "response_time_warning": 1000.0,   # >1 second response time
                "response_time_critical": 5000.0,  # >5 second response time
                "error_rate_warning": 0.01,        # >1% error rate
                "error_rate_critical": 0.05        # >5% error rate
            }
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        
        if self.health_monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.health_monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"🏥 Started continuous health monitoring (interval: {self.check_interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        
        self.health_monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("🏥 Stopped continuous health monitoring")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        
        try:
            while self.health_monitoring_active:
                try:
                    # Perform comprehensive health check
                    health_snapshot = await self.get_system_health()
                    
                    # Store in history
                    self.health_history.append(health_snapshot)
                    self.last_health_check = datetime.now(timezone.utc)
                    
                    # Check for health alerts
                    await self._check_health_alerts(health_snapshot)
                    
                    # Sleep until next check
                    await asyncio.sleep(self.check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"❌ Health monitoring iteration failed: {e}")
                    await asyncio.sleep(self.check_interval_seconds)
                    
        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Health monitoring loop failed: {e}")
    
    async def get_system_health(self) -> SystemHealthSnapshot:
        """Get comprehensive system health snapshot."""
        
        start_time = time.time()
        logger.debug("🏥 Performing comprehensive system health check")
        
        # Initialize health snapshot
        snapshot = SystemHealthSnapshot()
        
        try:
            # Check system resources
            await self._check_system_resources(snapshot)
            
            # Check Ray cluster health (if available)
            await self._check_ray_cluster_health(snapshot)
            
            # Check genetic evolution system
            await self._check_genetic_evolution_health(snapshot)
            
            # Check validation pipeline health
            await self._check_validation_pipeline_health(snapshot)
            
            # Check deployment system health
            await self._check_deployment_system_health(snapshot)
            
            # Check external APIs health
            await self._check_external_apis_health(snapshot)
            
            # Run custom component checkers
            await self._run_custom_health_checks(snapshot)
            
            # Calculate overall health
            self._calculate_overall_health(snapshot)
            
            # Perform trend analysis and predictions
            if self.issue_prediction_enabled:
                self._analyze_health_trends(snapshot)
            
            check_duration = time.time() - start_time
            logger.debug(f"🏥 Health check completed in {check_duration:.2f}s - Status: {snapshot.overall_status.value}")
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            snapshot.overall_status = HealthStatus.CRITICAL
            snapshot.overall_score = 0.0
        
        return snapshot
    
    async def _check_system_resources(self, snapshot: SystemHealthSnapshot):
        """Check system resource health (CPU, memory, disk)."""
        
        try:
            component = ComponentHealth(
                component_name="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCES
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                threshold_warning=self.health_thresholds["system_resources"]["memory_usage_warning"],
                threshold_critical=self.health_thresholds["system_resources"]["memory_usage_critical"],
                metadata={"available_gb": memory.available / (1024**3)}
            )
            
            if memory.percent >= memory_metric.threshold_critical:
                memory_metric.status = HealthStatus.CRITICAL
                component.issues_detected.append(f"Critical memory usage: {memory.percent:.1f}%")
                component.recommendations.append("Consider increasing system memory or reducing workload")
            elif memory.percent >= memory_metric.threshold_warning:
                memory_metric.status = HealthStatus.WARNING
                component.issues_detected.append(f"High memory usage: {memory.percent:.1f}%")
            
            component.metrics.append(memory_metric)
            snapshot.total_memory_usage_percent = memory.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                threshold_warning=self.health_thresholds["system_resources"]["cpu_usage_warning"],
                threshold_critical=self.health_thresholds["system_resources"]["cpu_usage_critical"],
                metadata={"cpu_count": psutil.cpu_count()}
            )
            
            if cpu_percent >= cpu_metric.threshold_critical:
                cpu_metric.status = HealthStatus.CRITICAL
                component.issues_detected.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent >= cpu_metric.threshold_warning:
                cpu_metric.status = HealthStatus.WARNING
                component.issues_detected.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            component.metrics.append(cpu_metric)
            snapshot.total_cpu_usage_percent = cpu_percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="%",
                threshold_warning=self.health_thresholds["system_resources"]["disk_usage_warning"],
                threshold_critical=self.health_thresholds["system_resources"]["disk_usage_critical"],
                metadata={"free_gb": disk.free / (1024**3)}
            )
            
            if disk_percent >= disk_metric.threshold_critical:
                disk_metric.status = HealthStatus.CRITICAL
                component.issues_detected.append(f"Critical disk usage: {disk_percent:.1f}%")
                component.recommendations.append("Free up disk space immediately")
            elif disk_percent >= disk_metric.threshold_warning:
                disk_metric.status = HealthStatus.WARNING
                component.issues_detected.append(f"High disk usage: {disk_percent:.1f}%")
            
            component.metrics.append(disk_metric)
            snapshot.disk_usage_percent = disk_percent
            
            # Load average (Unix/Linux systems)
            try:
                load_avg = psutil.getloadavg()
                load_metric = HealthMetric(
                    name="load_average",
                    value=load_avg[0],  # 1-minute load average
                    unit="",
                    threshold_warning=self.health_thresholds["system_resources"]["load_average_warning"],
                    threshold_critical=self.health_thresholds["system_resources"]["load_average_critical"],
                    metadata={"load_5min": load_avg[1], "load_15min": load_avg[2]}
                )
                
                if load_avg[0] >= load_metric.threshold_critical:
                    load_metric.status = HealthStatus.CRITICAL
                    component.issues_detected.append(f"Critical system load: {load_avg[0]:.2f}")
                elif load_avg[0] >= load_metric.threshold_warning:
                    load_metric.status = HealthStatus.WARNING
                    component.issues_detected.append(f"High system load: {load_avg[0]:.2f}")
                
                component.metrics.append(load_metric)
                snapshot.system_load_average = load_avg[0]
                
            except (AttributeError, OSError):
                # Load average not available on Windows
                pass
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            snapshot.system_uptime = uptime_seconds
            
            # Calculate overall component status
            critical_metrics = [m for m in component.metrics if m.status == HealthStatus.CRITICAL]
            warning_metrics = [m for m in component.metrics if m.status == HealthStatus.WARNING]
            
            if critical_metrics:
                component.overall_status = HealthStatus.CRITICAL
            elif warning_metrics:
                component.overall_status = HealthStatus.WARNING
            else:
                component.overall_status = HealthStatus.GOOD
            
            snapshot.components["system_resources"] = component
            
        except Exception as e:
            logger.error(f"❌ System resources health check failed: {e}")
    
    async def _check_ray_cluster_health(self, snapshot: SystemHealthSnapshot):
        """Check Ray cluster health (if available)."""
        
        if not RAY_AVAILABLE:
            logger.debug("Ray not available - skipping Ray cluster health check")
            return
        
        try:
            component = ComponentHealth(
                component_name="ray_cluster",
                component_type=ComponentType.RAY_CLUSTER
            )
            
            if ray.is_initialized():
                # Get cluster resources
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()
                
                snapshot.ray_cluster_size = len(ray.nodes())
                snapshot.ray_available_resources = dict(available_resources)
                
                # Check node health
                nodes = ray.nodes()
                alive_nodes = [node for node in nodes if node['Alive']]
                node_health_ratio = len(alive_nodes) / len(nodes) if nodes else 0
                
                node_health_metric = HealthMetric(
                    name="node_health_ratio",
                    value=node_health_ratio,
                    unit="ratio",
                    threshold_warning=1.0 - self.health_thresholds["ray_cluster"]["node_failure_warning"],
                    threshold_critical=1.0 - self.health_thresholds["ray_cluster"]["node_failure_critical"],
                    metadata={"total_nodes": len(nodes), "alive_nodes": len(alive_nodes)}
                )
                
                if node_health_ratio <= node_health_metric.threshold_critical:
                    node_health_metric.status = HealthStatus.CRITICAL
                    component.issues_detected.append(f"Critical Ray node failures: {len(nodes) - len(alive_nodes)}/{len(nodes)} nodes down")
                    snapshot.ray_cluster_healthy = False
                elif node_health_ratio <= node_health_metric.threshold_warning:
                    node_health_metric.status = HealthStatus.WARNING
                    component.issues_detected.append(f"Ray node issues: {len(nodes) - len(alive_nodes)}/{len(nodes)} nodes down")
                
                component.metrics.append(node_health_metric)
                
                # Check resource utilization
                if "CPU" in cluster_resources and "CPU" in available_resources:
                    cpu_usage_ratio = 1.0 - (available_resources["CPU"] / cluster_resources["CPU"])
                    
                    cpu_usage_metric = HealthMetric(
                        name="cpu_utilization",
                        value=cpu_usage_ratio,
                        unit="ratio",
                        threshold_warning=self.health_thresholds["ray_cluster"]["resource_usage_warning"],
                        threshold_critical=self.health_thresholds["ray_cluster"]["resource_usage_critical"],
                        metadata={"total_cpu": cluster_resources["CPU"], "available_cpu": available_resources["CPU"]}
                    )
                    
                    if cpu_usage_ratio >= cpu_usage_metric.threshold_critical:
                        cpu_usage_metric.status = HealthStatus.CRITICAL
                        component.issues_detected.append(f"Critical Ray CPU usage: {cpu_usage_ratio:.1%}")
                    elif cpu_usage_ratio >= cpu_usage_metric.threshold_warning:
                        cpu_usage_metric.status = HealthStatus.WARNING
                        component.issues_detected.append(f"High Ray CPU usage: {cpu_usage_ratio:.1%}")
                    
                    component.metrics.append(cpu_usage_metric)
                
                # Calculate overall Ray health
                if component.issues_detected:
                    critical_issues = [issue for issue in component.issues_detected if "Critical" in issue]
                    if critical_issues:
                        component.overall_status = HealthStatus.CRITICAL
                    else:
                        component.overall_status = HealthStatus.WARNING
                else:
                    component.overall_status = HealthStatus.GOOD
            
            else:
                # Ray not initialized
                component.overall_status = HealthStatus.WARNING
                component.issues_detected.append("Ray cluster not initialized")
                snapshot.ray_cluster_healthy = False
            
            snapshot.components["ray_cluster"] = component
            
        except Exception as e:
            logger.error(f"❌ Ray cluster health check failed: {e}")
            snapshot.ray_cluster_healthy = False
    
    async def _check_genetic_evolution_health(self, snapshot: SystemHealthSnapshot):
        """Check genetic evolution system health."""
        
        try:
            component = ComponentHealth(
                component_name="genetic_evolution",
                component_type=ComponentType.GENETIC_EVOLUTION
            )
            
            # This would integrate with actual genetic evolution metrics
            # For now, simulate basic health checks
            
            evolution_active_metric = HealthMetric(
                name="evolution_processes_active",
                value=1.0,  # Simulate active evolution
                unit="count",
                metadata={"last_evolution": "recently"}
            )
            
            component.metrics.append(evolution_active_metric)
            component.overall_status = HealthStatus.GOOD
            
            snapshot.components["genetic_evolution"] = component
            
        except Exception as e:
            logger.error(f"❌ Genetic evolution health check failed: {e}")
    
    async def _check_validation_pipeline_health(self, snapshot: SystemHealthSnapshot):
        """Check validation pipeline health."""
        
        try:
            component = ComponentHealth(
                component_name="validation_pipeline",
                component_type=ComponentType.VALIDATION_PIPELINE
            )
            
            # This would integrate with actual validation pipeline metrics
            validation_ready_metric = HealthMetric(
                name="validation_pipeline_ready",
                value=1.0,  # Simulate ready state
                unit="boolean",
                metadata={"components_ready": ["backtesting", "paper_trading", "testnet"]}
            )
            
            component.metrics.append(validation_ready_metric)
            component.overall_status = HealthStatus.GOOD
            
            snapshot.components["validation_pipeline"] = component
            
        except Exception as e:
            logger.error(f"❌ Validation pipeline health check failed: {e}")
    
    async def _check_deployment_system_health(self, snapshot: SystemHealthSnapshot):
        """Check deployment system health."""
        
        try:
            component = ComponentHealth(
                component_name="deployment_system",
                component_type=ComponentType.DEPLOYMENT_SYSTEM
            )
            
            # This would integrate with actual deployment system metrics
            deployment_ready_metric = HealthMetric(
                name="deployment_system_ready",
                value=1.0,  # Simulate ready state
                unit="boolean",
                metadata={"active_deployments": 0, "capacity_remaining": 10}
            )
            
            component.metrics.append(deployment_ready_metric)
            component.overall_status = HealthStatus.GOOD
            
            snapshot.components["deployment_system"] = component
            
        except Exception as e:
            logger.error(f"❌ Deployment system health check failed: {e}")
    
    async def _check_external_apis_health(self, snapshot: SystemHealthSnapshot):
        """Check external APIs health."""
        
        try:
            component = ComponentHealth(
                component_name="external_apis",
                component_type=ComponentType.EXTERNAL_APIs
            )
            
            # This would check actual external API connectivity
            # For now, simulate basic connectivity checks
            
            api_connectivity_metric = HealthMetric(
                name="api_connectivity",
                value=1.0,  # Simulate good connectivity
                unit="ratio",
                metadata={"apis_checked": ["hyperliquid", "market_data"]}
            )
            
            component.metrics.append(api_connectivity_metric)
            component.overall_status = HealthStatus.GOOD
            
            snapshot.components["external_apis"] = component
            
        except Exception as e:
            logger.error(f"❌ External APIs health check failed: {e}")
    
    async def _run_custom_health_checks(self, snapshot: SystemHealthSnapshot):
        """Run custom registered health checks."""
        
        for check_name, check_func in self.component_checkers.items():
            try:
                check_start = time.time()
                
                # Execute custom check
                check_result = await check_func()
                
                check_duration = (time.time() - check_start) * 1000  # Convert to ms
                
                if isinstance(check_result, ComponentHealth):
                    check_result.check_duration_ms = check_duration
                    snapshot.components[check_name] = check_result
                elif isinstance(check_result, dict):
                    # Convert dict result to ComponentHealth
                    component = ComponentHealth(
                        component_name=check_name,
                        component_type=ComponentType.MONITORING_SYSTEM,
                        overall_status=HealthStatus(check_result.get("status", "good")),
                        check_duration_ms=check_duration,
                        issues_detected=check_result.get("issues", []),
                        recommendations=check_result.get("recommendations", [])
                    )
                    snapshot.components[check_name] = component
                
            except Exception as e:
                logger.error(f"❌ Custom health check '{check_name}' failed: {e}")
                
                # Create failed check component
                failed_component = ComponentHealth(
                    component_name=check_name,
                    component_type=ComponentType.MONITORING_SYSTEM,
                    overall_status=HealthStatus.CRITICAL,
                    issues_detected=[f"Health check failed: {str(e)}"]
                )
                snapshot.components[check_name] = failed_component
    
    def _calculate_overall_health(self, snapshot: SystemHealthSnapshot):
        """Calculate overall system health based on component health."""
        
        if not snapshot.components:
            snapshot.overall_status = HealthStatus.WARNING
            snapshot.overall_score = 0.5
            return
        
        # Count components by status
        status_counts = defaultdict(int)
        total_components = len(snapshot.components)
        
        for component in snapshot.components.values():
            status_counts[component.overall_status] += 1
        
        # Calculate health score
        score_weights = {
            HealthStatus.EXCELLENT: 1.0,
            HealthStatus.GOOD: 0.8,
            HealthStatus.WARNING: 0.5,
            HealthStatus.CRITICAL: 0.2,
            HealthStatus.FAILING: 0.0
        }
        
        weighted_score = sum(
            status_counts[status] * weight
            for status, weight in score_weights.items()
        ) / total_components
        
        snapshot.overall_score = weighted_score
        
        # Determine overall status
        if status_counts[HealthStatus.FAILING] > 0:
            snapshot.overall_status = HealthStatus.FAILING
        elif status_counts[HealthStatus.CRITICAL] >= total_components * 0.3:  # 30% critical
            snapshot.overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] >= total_components * 0.5:  # 50% warning
            snapshot.overall_status = HealthStatus.WARNING
        elif weighted_score >= 0.9:
            snapshot.overall_status = HealthStatus.EXCELLENT
        else:
            snapshot.overall_status = HealthStatus.GOOD
    
    def _analyze_health_trends(self, snapshot: SystemHealthSnapshot):
        """Analyze health trends and predict potential issues."""
        
        if len(self.health_history) < 5:  # Need at least 5 snapshots for trend analysis
            snapshot.health_trend = "insufficient_data"
            return
        
        try:
            # Get recent health scores
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.trend_analysis_window_minutes)
            recent_snapshots = [
                s for s in self.health_history
                if s.timestamp >= cutoff_time
            ]
            
            if len(recent_snapshots) < 3:
                snapshot.health_trend = "stable"
                return
            
            # Analyze health score trend
            health_scores = [s.overall_score for s in recent_snapshots]
            
            if len(health_scores) >= 3:
                # Simple trend analysis using linear regression slope
                x = list(range(len(health_scores)))
                y = health_scores
                
                n = len(x)
                slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
                
                if slope > 0.01:
                    snapshot.health_trend = "improving"
                elif slope < -0.01:
                    snapshot.health_trend = "degrading"
                    
                    # Predict potential issues based on degrading trend
                    if slope < -0.05:
                        snapshot.predicted_issues.append("Rapid health degradation detected - investigate immediately")
                    
                    # Check specific metrics showing degradation
                    if len(recent_snapshots) >= 2:
                        latest = recent_snapshots[-1]
                        previous = recent_snapshots[-2]
                        
                        if latest.total_memory_usage_percent > previous.total_memory_usage_percent + 10:
                            snapshot.predicted_issues.append("Memory usage increasing rapidly")
                        
                        if latest.total_cpu_usage_percent > previous.total_cpu_usage_percent + 20:
                            snapshot.predicted_issues.append("CPU usage spiking - possible runaway process")
                        
                        if latest.system_load_average > previous.system_load_average + 2:
                            snapshot.predicted_issues.append("System load increasing - performance may degrade")
                else:
                    snapshot.health_trend = "stable"
            
        except Exception as e:
            logger.error(f"❌ Health trend analysis failed: {e}")
            snapshot.health_trend = "analysis_failed"
    
    async def _check_health_alerts(self, snapshot: SystemHealthSnapshot):
        """Check if health status requires alerts."""
        
        try:
            # Alert on overall status changes
            if len(self.health_history) > 0:
                previous_snapshot = self.health_history[-1]
                
                if snapshot.overall_status != previous_snapshot.overall_status:
                    # Overall health status changed
                    priority = AlertPriority.CRITICAL if snapshot.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILING] else AlertPriority.WARNING
                    
                    await self.alerting.send_system_alert(
                        alert_type="system_health_change",
                        message=f"System health status changed: {previous_snapshot.overall_status.value} → {snapshot.overall_status.value}",
                        priority=priority,
                        metadata={
                            "old_status": previous_snapshot.overall_status.value,
                            "new_status": snapshot.overall_status.value,
                            "health_score": snapshot.overall_score,
                            "component_count": len(snapshot.components)
                        }
                    )
            
            # Alert on critical component issues
            critical_components = [
                comp for comp in snapshot.components.values()
                if comp.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILING]
            ]
            
            if critical_components:
                await self.alerting.send_system_alert(
                    alert_type="critical_component_health",
                    message=f"Critical health issues in {len(critical_components)} components",
                    priority=AlertPriority.CRITICAL,
                    metadata={
                        "critical_components": [comp.component_name for comp in critical_components],
                        "total_issues": sum(len(comp.issues_detected) for comp in critical_components)
                    }
                )
            
            # Alert on predicted issues
            if snapshot.predicted_issues:
                await self.alerting.send_system_alert(
                    alert_type="predicted_health_issues",
                    message=f"Predicted health issues detected: {', '.join(snapshot.predicted_issues[:3])}",
                    priority=AlertPriority.WARNING,
                    metadata={
                        "predicted_issues": snapshot.predicted_issues,
                        "health_trend": snapshot.health_trend
                    }
                )
        
        except Exception as e:
            logger.error(f"❌ Health alert check failed: {e}")
    
    async def register_component_checker(self, name: str, check_func: Callable):
        """Register custom component health checker."""
        
        self.component_checkers[name] = check_func
        logger.info(f"🏥 Registered health checker: {name}")
    
    def get_health_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get health summary for specified time period."""
        
        if not self.health_history:
            return {
                "period_hours": hours_back,
                "snapshots": 0,
                "average_health_score": 0.0,
                "status_distribution": {},
                "trend": "no_data"
            }
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        relevant_snapshots = [
            snapshot for snapshot in self.health_history
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not relevant_snapshots:
            return {
                "period_hours": hours_back,
                "snapshots": 0,
                "average_health_score": 0.0,
                "status_distribution": {},
                "trend": "no_data"
            }
        
        # Calculate statistics
        health_scores = [s.overall_score for s in relevant_snapshots]
        average_score = statistics.mean(health_scores)
        
        status_counts = defaultdict(int)
        for snapshot in relevant_snapshots:
            status_counts[snapshot.overall_status.value] += 1
        
        return {
            "period_hours": hours_back,
            "snapshots": len(relevant_snapshots),
            "average_health_score": average_score,
            "min_health_score": min(health_scores),
            "max_health_score": max(health_scores),
            "status_distribution": dict(status_counts),
            "trend": relevant_snapshots[-1].health_trend if relevant_snapshots else "unknown",
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get current health for specific component."""
        
        if not self.health_history:
            return None
        
        latest_snapshot = self.health_history[-1]
        return latest_snapshot.components.get(component_name)


# Integration with resilience manager
async def integrate_with_resilience_manager(health_monitor: SystemHealthMonitor,
                                          resilience_manager: ResilienceManager):
    """Integrate health monitor with resilience manager."""
    
    async def health_check_for_resilience():
        """Health check function for resilience manager integration."""
        
        health_snapshot = await health_monitor.get_system_health()
        
        return {
            "healthy": health_snapshot.overall_status not in [HealthStatus.CRITICAL, HealthStatus.FAILING],
            "health_score": health_snapshot.overall_score,
            "status": health_snapshot.overall_status.value,
            "issues": sum(len(comp.issues_detected) for comp in health_snapshot.components.values())
        }
    
    await resilience_manager.register_health_check("system_health_monitor", health_check_for_resilience)
    logger.info("🤝 Integrated SystemHealthMonitor with ResilienceManager")


# Factory functions for easy integration
def get_system_health_monitor(settings: Optional[Settings] = None,
                            monitoring_system: Optional[RealTimeMonitoringSystem] = None,
                            resilience_manager: Optional[ResilienceManager] = None) -> SystemHealthMonitor:
    """Factory function to get SystemHealthMonitor instance."""
    return SystemHealthMonitor(
        settings=settings,
        monitoring_system=monitoring_system,
        resilience_manager=resilience_manager
    )


if __name__ == "__main__":
    """Test the system health monitor with sample checks."""
    
    async def test_system_health_monitor():
        """Test function for development."""
        
        logger.info("🧪 Testing System Health Monitor")
        
        monitor = get_system_health_monitor()
        logger.info("✅ System health monitor initialized successfully")
        
        # Test health check
        health_snapshot = await monitor.get_system_health()
        logger.info(f"🏥 System health check: {health_snapshot.overall_status.value} (score: {health_snapshot.overall_score:.3f})")
        
        # Test health summary
        summary = monitor.get_health_summary(hours_back=1)
        logger.info(f"📊 Health summary: {summary['snapshots']} snapshots, avg score: {summary.get('average_health_score', 0):.3f}")
        
        logger.info("✅ System Health Monitor test completed")
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_system_health_monitor())