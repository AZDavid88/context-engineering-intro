"""
Real-time Monitoring System - Central Observability Hub for Genetic Trading Organism

This module implements a comprehensive monitoring system that provides real-time
performance tracking, automated alerts, and system health monitoring for the
entire genetic trading ecosystem.

Based on research from:
- VectorBT Performance Optimization Patterns (real-time monitoring)
- Genetic Algorithm Evolution Tracking
- System Health Monitoring Best Practices

Key Features:
- Real-time genetic algorithm evolution tracking
- Trading performance analytics with Sharpe ratio monitoring
- System health metrics (memory, CPU, latency tracking)
- Automated alert system with escalation levels
- Dashboard-ready data export for operators
- Historical analysis and trend detection
- Integration with all Phase 1-3 components
"""

import asyncio
import logging
import time
import sys
import os
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import warnings

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import get_settings, Settings
from src.execution.risk_management import GeneticRiskManager, RiskLevel, RiskMetrics, MarketRegime
from src.execution.paper_trading import PaperTradingEngine, StrategyPerformance, TradeExecutionQuality
from src.execution.position_sizer import GeneticPositionSizer, PositionSizeResult
from src.execution.order_management import OrderRequest, OrderStatus

# Configure logging
logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels for monitoring system."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    """Categories of monitoring alerts."""
    GENETIC_EVOLUTION = "genetic_evolution"
    TRADING_PERFORMANCE = "trading_performance"
    SYSTEM_HEALTH = "system_health"
    RISK_MANAGEMENT = "risk_management"
    DATA_PIPELINE = "data_pipeline"
    ORDER_EXECUTION = "order_execution"


class MonitoringStatus(str, Enum):
    """Overall monitoring system status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class GeneticEvolutionMetrics:
    """Genetic algorithm evolution tracking metrics."""
    
    generation: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    fitness_std: float = 0.0
    diversity_score: float = 0.0
    convergence_rate: float = 0.0
    mutation_rate: float = 0.0
    crossover_rate: float = 0.0
    selection_pressure: float = 0.0
    
    # Performance metrics
    evaluation_time: float = 0.0
    memory_usage_gb: float = 0.0
    cache_hit_rate: float = 0.0
    vectorization_speedup: float = 0.0
    
    # Strategy metrics
    active_strategies: int = 0
    successful_strategies: int = 0
    failed_strategies: int = 0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TradingPerformanceMetrics:
    """Trading system performance metrics."""
    
    # Portfolio metrics
    total_return: float = 0.0
    daily_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    
    # Trade execution metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_amount: float = 0.0
    avg_loss_amount: float = 0.0
    profit_factor: float = 0.0
    
    # Execution quality metrics
    avg_slippage: float = 0.0
    avg_latency_ms: float = 0.0
    execution_success_rate: float = 0.0
    
    # Position metrics
    total_exposure: float = 0.0
    active_positions: int = 0
    portfolio_concentration: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_gb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    
    # Application metrics
    active_threads: int = 0
    open_connections: int = 0
    pending_orders: int = 0
    message_queue_size: int = 0
    error_count: int = 0
    warning_count: int = 0
    
    # Performance metrics
    requests_per_second: float = 0.0
    avg_response_time: float = 0.0
    throughput_mbps: float = 0.0
    uptime_hours: float = 0.0
    
    # Data pipeline metrics
    market_data_lag_ms: float = 0.0
    data_processing_rate: float = 0.0
    storage_write_rate: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MonitoringAlert:
    """Individual monitoring alert."""
    
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: AlertLevel = AlertLevel.INFO
    category: AlertCategory = AlertCategory.SYSTEM_HEALTH
    title: str = ""
    message: str = ""
    source_component: str = ""
    
    # Alert metadata
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    trigger_count: int = 1
    
    # Timing
    first_triggered: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_triggered: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False
    
    # Context data
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringSnapshot:
    """Complete monitoring system snapshot."""
    
    status: MonitoringStatus = MonitoringStatus.HEALTHY
    genetic_metrics: GeneticEvolutionMetrics = field(default_factory=GeneticEvolutionMetrics)
    trading_metrics: TradingPerformanceMetrics = field(default_factory=TradingPerformanceMetrics)
    system_metrics: SystemHealthMetrics = field(default_factory=SystemHealthMetrics)
    
    # Risk metrics
    current_risk_level: RiskLevel = RiskLevel.LOW
    active_circuit_breakers: List[str] = field(default_factory=list)
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    
    # Active alerts
    active_alerts: List[MonitoringAlert] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AlertManager:
    """Manages monitoring alerts with escalation and acknowledgment."""
    
    def __init__(self, settings: Settings):
        """Initialize alert manager.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.alert_callbacks: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        
        # Alert suppression to prevent spam
        self.alert_suppression = {}  # alert_key -> last_sent_time
        self.suppression_window = 300  # 5 minutes
        
        logger.info("Alert manager initialized")
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds."""
        return {
            'genetic_evolution': {
                'fitness_stagnation_generations': 10,
                'low_diversity_threshold': 0.1,
                'slow_evaluation_time': 30.0,
                'high_memory_usage': 12.0
            },
            'trading_performance': {
                'max_drawdown_warning': 0.05,  # 5%
                'max_drawdown_critical': 0.10,  # 10%
                'negative_sharpe_threshold': -0.5,
                'low_win_rate_threshold': 0.3,
                'execution_success_threshold': 0.95
            },
            'system_health': {
                'cpu_usage_warning': 80.0,
                'cpu_usage_critical': 95.0,
                'memory_usage_warning': 80.0,
                'memory_usage_critical': 95.0,
                'disk_usage_warning': 85.0,
                'network_latency_warning': 1000.0,
                'error_rate_threshold': 0.05
            }
        }
    
    def trigger_alert(self, level: AlertLevel, category: AlertCategory, 
                     title: str, message: str, source_component: str,
                     threshold_value: Optional[float] = None,
                     current_value: Optional[float] = None,
                     context_data: Optional[Dict[str, Any]] = None) -> MonitoringAlert:
        """Trigger a monitoring alert.
        
        Args:
            level: Alert severity level
            category: Alert category
            title: Alert title
            message: Alert message
            source_component: Component that triggered alert
            threshold_value: Threshold that was breached
            current_value: Current value that triggered alert
            context_data: Additional context data
            
        Returns:
            Created or updated monitoring alert
        """
        
        # Create unique alert key for deduplication
        alert_key = f"{category}_{title}_{source_component}"
        
        # Check alert suppression
        if self._is_alert_suppressed(alert_key):
            logger.debug(f"Alert suppressed: {alert_key}")
            return self.active_alerts.get(alert_key)
        
        current_time = datetime.now(timezone.utc)
        
        # Check if alert already exists
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            existing_alert.trigger_count += 1
            existing_alert.last_triggered = current_time
            existing_alert.current_value = current_value
            if context_data:
                existing_alert.context_data.update(context_data)
            alert = existing_alert
        else:
            # Create new alert
            alert = MonitoringAlert(
                level=level,
                category=category,
                title=title,
                message=message,
                source_component=source_component,
                threshold_value=threshold_value,
                current_value=current_value,
                context_data=context_data or {}
            )
            self.active_alerts[alert_key] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Update suppression
        self.alert_suppression[alert_key] = current_time
        
        # Execute alert callbacks
        self._execute_alert_callbacks(alert)
        
        logger.warning(f"Alert triggered: {level} - {title} from {source_component}")
        
        return alert
    
    def _is_alert_suppressed(self, alert_key: str) -> bool:
        """Check if alert is suppressed to prevent spam."""
        if alert_key not in self.alert_suppression:
            return False
        
        last_sent = self.alert_suppression[alert_key]
        time_since_last = (datetime.now(timezone.utc) - last_sent).total_seconds()
        
        return time_since_last < self.suppression_window
    
    def _execute_alert_callbacks(self, alert: MonitoringAlert):
        """Execute registered callbacks for alert level."""
        callbacks = self.alert_callbacks.get(alert.level, [])
        
        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error executing alert callback: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if alert was acknowledged successfully
        """
        for alert_key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.context_data['acknowledged_by'] = acknowledged_by
                alert.context_data['acknowledged_at'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved successfully
        """
        alert_to_remove = None
        
        for alert_key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.context_data['resolved_by'] = resolved_by
                alert.context_data['resolved_at'] = datetime.now(timezone.utc).isoformat()
                alert_to_remove = alert_key
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                break
        
        if alert_to_remove:
            del self.active_alerts[alert_to_remove]
            return True
        
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None,
                         category: Optional[AlertCategory] = None) -> List[MonitoringAlert]:
        """Get active alerts with optional filtering.
        
        Args:
            level: Filter by alert level
            category: Filter by alert category
            
        Returns:
            List of filtered active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        # Sort by level priority and timestamp
        level_priority = {
            AlertLevel.EMERGENCY: 4,
            AlertLevel.CRITICAL: 3,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 1
        }
        
        alerts.sort(key=lambda a: (-level_priority.get(a.level, 0), -a.last_triggered.timestamp()))
        
        return alerts
    
    def register_alert_callback(self, level: AlertLevel, callback: Callable):
        """Register callback for alert level.
        
        Args:
            level: Alert level to register callback for
            callback: Callback function to execute
        """
        self.alert_callbacks[level].append(callback)
        logger.debug(f"Alert callback registered for level: {level}")


class PerformanceAnalyzer:
    """Analyzes performance trends and detects anomalies."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize performance analyzer.
        
        Args:
            history_size: Maximum history size to maintain
        """
        self.history_size = history_size
        
        # Performance history
        self.genetic_history = deque(maxlen=history_size)
        self.trading_history = deque(maxlen=history_size)
        self.system_history = deque(maxlen=history_size)
        
        # Trend analysis
        self.trend_window = 20  # Data points for trend analysis
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        
        logger.info("Performance analyzer initialized")
    
    def add_genetic_metrics(self, metrics: GeneticEvolutionMetrics):
        """Add genetic evolution metrics to history."""
        self.genetic_history.append(metrics)
    
    def add_trading_metrics(self, metrics: TradingPerformanceMetrics):
        """Add trading performance metrics to history."""
        self.trading_history.append(metrics)
    
    def add_system_metrics(self, metrics: SystemHealthMetrics):
        """Add system health metrics to history."""
        self.system_history.append(metrics)
    
    def analyze_genetic_trends(self) -> Dict[str, Any]:
        """Analyze genetic algorithm evolution trends."""
        if len(self.genetic_history) < self.trend_window:
            return {'status': 'insufficient_data', 'data_points': len(self.genetic_history)}
        
        recent_metrics = list(self.genetic_history)[-self.trend_window:]
        
        # Fitness trend analysis
        fitness_values = [m.best_fitness for m in recent_metrics]
        fitness_trend = self._calculate_trend(fitness_values)
        
        # Diversity trend analysis
        diversity_values = [m.diversity_score for m in recent_metrics]
        diversity_trend = self._calculate_trend(diversity_values)
        
        # Performance trend analysis
        eval_times = [m.evaluation_time for m in recent_metrics]
        performance_trend = self._calculate_trend(eval_times, inverted=True)  # Lower is better
        
        # Convergence analysis
        convergence_analysis = self._analyze_convergence(recent_metrics)
        
        return {
            'fitness_trend': fitness_trend,
            'diversity_trend': diversity_trend,
            'performance_trend': performance_trend,
            'convergence_analysis': convergence_analysis,
            'current_generation': recent_metrics[-1].generation,
            'evaluation_efficiency': self._calculate_evaluation_efficiency(recent_metrics)
        }
    
    def analyze_trading_trends(self) -> Dict[str, Any]:
        """Analyze trading performance trends."""
        if len(self.trading_history) < self.trend_window:
            return {'status': 'insufficient_data', 'data_points': len(self.trading_history)}
        
        recent_metrics = list(self.trading_history)[-self.trend_window:]
        
        # Return trend analysis
        returns = [m.daily_return for m in recent_metrics]
        return_trend = self._calculate_trend(returns)
        
        # Sharpe ratio trend
        sharpe_values = [m.sharpe_ratio for m in recent_metrics]
        sharpe_trend = self._calculate_trend(sharpe_values)
        
        # Drawdown analysis
        drawdowns = [m.current_drawdown for m in recent_metrics]
        drawdown_trend = self._calculate_trend(drawdowns, inverted=True)  # Lower is better
        
        # Win rate trend
        win_rates = [m.win_rate for m in recent_metrics]
        win_rate_trend = self._calculate_trend(win_rates)
        
        return {
            'return_trend': return_trend,
            'sharpe_trend': sharpe_trend,
            'drawdown_trend': drawdown_trend,
            'win_rate_trend': win_rate_trend,
            'risk_adjusted_performance': self._calculate_risk_adjusted_performance(recent_metrics),
            'execution_quality': self._analyze_execution_quality(recent_metrics)
        }
    
    def analyze_system_trends(self) -> Dict[str, Any]:
        """Analyze system health trends."""
        if len(self.system_history) < self.trend_window:
            return {'status': 'insufficient_data', 'data_points': len(self.system_history)}
        
        recent_metrics = list(self.system_history)[-self.trend_window:]
        
        # Resource usage trends
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        cpu_trend = self._calculate_trend(cpu_values, inverted=True)  # Lower is better
        
        memory_values = [m.memory_usage_percent for m in recent_metrics]
        memory_trend = self._calculate_trend(memory_values, inverted=True)
        
        # Performance trends
        latency_values = [m.network_latency_ms for m in recent_metrics]
        latency_trend = self._calculate_trend(latency_values, inverted=True)
        
        # Throughput trends
        throughput_values = [m.throughput_mbps for m in recent_metrics]
        throughput_trend = self._calculate_trend(throughput_values)
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'latency_trend': latency_trend,
            'throughput_trend': throughput_trend,
            'system_stability': self._calculate_system_stability(recent_metrics),
            'resource_efficiency': self._calculate_resource_efficiency(recent_metrics)
        }
    
    def _calculate_trend(self, values: List[float], inverted: bool = False) -> Dict[str, Any]:
        """Calculate trend analysis for a series of values."""
        if len(values) < 2:
            return {'trend': 'unknown', 'slope': 0.0, 'confidence': 0.0}
        
        # Convert to numpy for calculations
        y = np.array(values)
        x = np.arange(len(y))
        
        # Calculate linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Adjust slope if inverted (for metrics where lower is better)
        if inverted:
            slope = -slope
        
        # Calculate confidence based on R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'confidence': float(r_squared),
            'recent_value': float(values[-1]),
            'change_rate': float(slope * len(values))
        }
    
    def _analyze_convergence(self, metrics: List[GeneticEvolutionMetrics]) -> Dict[str, Any]:
        """Analyze genetic algorithm convergence."""
        if len(metrics) < 5:
            return {'status': 'insufficient_data'}
        
        # Calculate convergence rate
        fitness_improvements = []
        for i in range(1, len(metrics)):
            improvement = metrics[i].best_fitness - metrics[i-1].best_fitness
            fitness_improvements.append(improvement)
        
        # Recent improvement rate
        recent_improvements = fitness_improvements[-5:]
        avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
        
        # Stagnation detection
        stagnation_threshold = 0.001
        stagnant_generations = 0
        for improvement in reversed(recent_improvements):
            if abs(improvement) < stagnation_threshold:
                stagnant_generations += 1
            else:
                break
        
        # Diversity analysis
        recent_diversity = [m.diversity_score for m in metrics[-5:]]
        avg_diversity = np.mean(recent_diversity) if recent_diversity else 0
        
        return {
            'avg_improvement_rate': float(avg_improvement),
            'stagnant_generations': stagnant_generations,
            'avg_diversity': float(avg_diversity),
            'convergence_risk': 'high' if stagnant_generations >= 3 else 'low',
            'premature_convergence': avg_diversity < 0.1 and stagnant_generations >= 2
        }
    
    def _calculate_evaluation_efficiency(self, metrics: List[GeneticEvolutionMetrics]) -> Dict[str, float]:
        """Calculate genetic algorithm evaluation efficiency."""
        if not metrics:
            return {}
        
        recent_metric = metrics[-1]
        
        # Strategies per second
        strategies_per_second = (recent_metric.population_size / 
                               max(recent_metric.evaluation_time, 0.001))
        
        # Memory efficiency (strategies per GB)
        memory_efficiency = (recent_metric.population_size / 
                           max(recent_metric.memory_usage_gb, 0.001))
        
        return {
            'strategies_per_second': float(strategies_per_second),
            'memory_efficiency': float(memory_efficiency),
            'cache_hit_rate': float(recent_metric.cache_hit_rate),
            'vectorization_speedup': float(recent_metric.vectorization_speedup)
        }
    
    def _calculate_risk_adjusted_performance(self, metrics: List[TradingPerformanceMetrics]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        if not metrics:
            return {}
        
        recent_metric = metrics[-1]
        
        # Risk-adjusted return
        risk_adjusted_return = (recent_metric.total_return / 
                              max(recent_metric.volatility, 0.001))
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = (recent_metric.total_return / 
                       max(abs(recent_metric.max_drawdown), 0.001))
        
        return {
            'risk_adjusted_return': float(risk_adjusted_return),
            'calmar_ratio': float(calmar_ratio),
            'sharpe_ratio': float(recent_metric.sharpe_ratio),
            'sortino_ratio': float(recent_metric.sortino_ratio)
        }
    
    def _analyze_execution_quality(self, metrics: List[TradingPerformanceMetrics]) -> Dict[str, float]:
        """Analyze trade execution quality."""
        if not metrics:
            return {}
        
        recent_metric = metrics[-1]
        
        return {
            'execution_success_rate': float(recent_metric.execution_success_rate),
            'avg_slippage': float(recent_metric.avg_slippage),
            'avg_latency_ms': float(recent_metric.avg_latency_ms),
            'profit_factor': float(recent_metric.profit_factor)
        }
    
    def _calculate_system_stability(self, metrics: List[SystemHealthMetrics]) -> Dict[str, float]:
        """Calculate system stability metrics."""
        if not metrics:
            return {}
        
        # Resource usage stability (lower variance is better)
        cpu_values = [m.cpu_usage_percent for m in metrics]
        memory_values = [m.memory_usage_percent for m in metrics]
        
        cpu_stability = 1.0 / (1.0 + np.std(cpu_values))
        memory_stability = 1.0 / (1.0 + np.std(memory_values))
        
        # Error rate analysis
        error_counts = [m.error_count for m in metrics]
        total_errors = sum(error_counts)
        error_rate = total_errors / len(metrics) if metrics else 0
        
        return {
            'cpu_stability': float(cpu_stability),
            'memory_stability': float(memory_stability),
            'error_rate': float(error_rate),
            'uptime_hours': float(metrics[-1].uptime_hours) if metrics else 0
        }
    
    def _calculate_resource_efficiency(self, metrics: List[SystemHealthMetrics]) -> Dict[str, float]:
        """Calculate resource utilization efficiency."""
        if not metrics:
            return {}
        
        recent_metric = metrics[-1]
        
        # Throughput per CPU usage
        cpu_efficiency = (recent_metric.throughput_mbps / 
                         max(recent_metric.cpu_usage_percent, 1.0))
        
        # Requests per memory usage
        memory_efficiency = (recent_metric.requests_per_second / 
                           max(recent_metric.memory_usage_gb, 0.1))
        
        return {
            'cpu_efficiency': float(cpu_efficiency),
            'memory_efficiency': float(memory_efficiency),
            'network_efficiency': float(recent_metric.throughput_mbps / 
                                      max(recent_metric.network_latency_ms, 1.0))
        }


class SystemHealthMonitor:
    """Monitors system health and resource usage."""
    
    def __init__(self):
        """Initialize system health monitor."""
        self.start_time = time.time()
        self.process = psutil.Process()
        
        # Metric collection counters
        self.request_count = 0
        self.error_count = 0
        self.warning_count = 0
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        
        logger.info("System health monitor initialized")
    
    def collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect current system health metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            # Process-specific metrics
            process_memory = self.process.memory_info()
            process_threads = self.process.num_threads()
            
            # Network metrics (simplified)
            network_latency = self._measure_network_latency()
            
            # Application metrics
            uptime_hours = (time.time() - self.start_time) / 3600
            avg_response_time = np.mean(self.response_times) if self.response_times else 0
            
            # Throughput calculation
            current_throughput = self._calculate_current_throughput()
            
            metrics = SystemHealthMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_gb=memory_info.used / (1024**3),
                memory_usage_percent=memory_info.percent,
                disk_usage_percent=disk_info.percent,
                network_latency_ms=network_latency,
                
                active_threads=process_threads,
                open_connections=len(self.process.connections()),
                pending_orders=0,  # Would be updated by order management system
                message_queue_size=0,  # Would be updated by message queue
                error_count=self.error_count,
                warning_count=self.warning_count,
                
                requests_per_second=self.request_count,
                avg_response_time=avg_response_time,
                throughput_mbps=current_throughput,
                uptime_hours=uptime_hours,
                
                market_data_lag_ms=0,  # Would be updated by data pipeline
                data_processing_rate=0,  # Would be updated by data pipeline
                storage_write_rate=0  # Would be updated by storage system
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemHealthMetrics()
    
    def _measure_network_latency(self) -> float:
        """Measure network latency (simplified implementation)."""
        try:
            import socket
            
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            
            # Test connection to a reliable server
            result = sock.connect_ex(('8.8.8.8', 53))
            sock.close()
            
            if result == 0:
                return (time.time() - start_time) * 1000
            else:
                return 1000.0  # Default high latency on failure
                
        except Exception:
            return 1000.0
    
    def _calculate_current_throughput(self) -> float:
        """Calculate current system throughput."""
        if len(self.throughput_samples) < 2:
            return 0.0
        
        return np.mean(self.throughput_samples)
    
    def record_request(self, response_time: float):
        """Record a request for performance tracking."""
        self.request_count += 1
        self.response_times.append(response_time)
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def record_warning(self):
        """Record a warning occurrence."""
        self.warning_count += 1
    
    def record_throughput(self, mbps: float):
        """Record throughput sample."""
        self.throughput_samples.append(mbps)


class RealTimeMonitoringSystem:
    """Central real-time monitoring system for the genetic trading organism."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the real-time monitoring system.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        
        # Initialize components
        self.alert_manager = AlertManager(self.settings)
        self.performance_analyzer = PerformanceAnalyzer()
        self.system_health = SystemHealthMonitor()
        
        # Component references (to be injected)
        self.risk_manager: Optional[GeneticRiskManager] = None
        self.paper_trading: Optional[PaperTradingEngine] = None
        self.position_sizer: Optional[GeneticPositionSizer] = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 5.0  # seconds
        self.snapshot_history = deque(maxlen=1000)
        
        # Threading for background monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Metrics collection
        self.last_snapshot_time = time.time()
        self.collection_count = 0
        
        logger.info("Real-time monitoring system initialized")
    
    def inject_components(self, 
                         risk_manager: Optional[GeneticRiskManager] = None,
                         paper_trading: Optional[PaperTradingEngine] = None,
                         position_sizer: Optional[GeneticPositionSizer] = None):
        """Inject component references for monitoring.
        
        Args:
            risk_manager: Risk management system
            paper_trading: Paper trading system
            position_sizer: Position sizing system
        """
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading
        self.position_sizer = position_sizer
        
        logger.info("Monitoring system components injected")
    
    def start_monitoring(self):
        """Start real-time monitoring in background thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MonitoringThread",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info("Monitoring loop started")
        
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Collect monitoring snapshot
                snapshot = self.collect_monitoring_snapshot()
                
                # Store snapshot
                self.snapshot_history.append(snapshot)
                
                # Update performance analyzer
                self.performance_analyzer.add_genetic_metrics(snapshot.genetic_metrics)
                self.performance_analyzer.add_trading_metrics(snapshot.trading_metrics)
                self.performance_analyzer.add_system_metrics(snapshot.system_metrics)
                
                # Check for alerts
                self._check_alert_conditions(snapshot)
                
                # Increment collection count
                self.collection_count += 1
                
                # Log periodic status
                if self.collection_count % 12 == 0:  # Every minute at 5-second intervals
                    self._log_monitoring_status(snapshot)
                
                # Wait for next collection
                self.shutdown_event.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.system_health.record_error()
                
                # Wait before retrying
                self.shutdown_event.wait(self.monitoring_interval)
        
        logger.info("Monitoring loop stopped")
    
    def collect_monitoring_snapshot(self) -> MonitoringSnapshot:
        """Collect a complete monitoring snapshot."""
        try:
            # Collect system health metrics
            system_metrics = self.system_health.collect_system_metrics()
            
            # Collect genetic evolution metrics
            genetic_metrics = self._collect_genetic_metrics()
            
            # Collect trading performance metrics
            trading_metrics = self._collect_trading_metrics()
            
            # Collect risk management metrics
            risk_level, circuit_breakers, market_regime = self._collect_risk_metrics()
            
            # Determine overall status
            overall_status = self._determine_monitoring_status(
                system_metrics, genetic_metrics, trading_metrics, risk_level
            )
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            snapshot = MonitoringSnapshot(
                status=overall_status,
                genetic_metrics=genetic_metrics,
                trading_metrics=trading_metrics,
                system_metrics=system_metrics,
                current_risk_level=risk_level,
                active_circuit_breakers=circuit_breakers,
                market_regime=market_regime,
                active_alerts=active_alerts
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error collecting monitoring snapshot: {e}")
            return MonitoringSnapshot(status=MonitoringStatus.CRITICAL)
    
    def _collect_genetic_metrics(self) -> GeneticEvolutionMetrics:
        """Collect genetic algorithm evolution metrics."""
        # This would be populated by the genetic engine
        # For now, return default metrics
        return GeneticEvolutionMetrics(
            generation=0,
            population_size=0,
            best_fitness=0.0,
            average_fitness=0.0,
            fitness_std=0.0,
            diversity_score=0.0,
            convergence_rate=0.0,
            evaluation_time=0.0,
            memory_usage_gb=0.0,
            active_strategies=0
        )
    
    def _collect_trading_metrics(self) -> TradingPerformanceMetrics:
        """Collect trading performance metrics."""
        if not self.paper_trading:
            return TradingPerformanceMetrics()
        
        try:
            # Get paper trading summary
            summary = self.paper_trading.get_paper_trading_summary()
            
            # Get top strategies for performance metrics
            top_strategies = self.paper_trading.get_top_strategies(10)
            
            # Calculate aggregate metrics
            if top_strategies:
                avg_win_rate = np.mean([s.win_rate for s in top_strategies])
                avg_fitness = np.mean([s.fitness_score for s in top_strategies])
                total_trades = sum([s.total_trades for s in top_strategies])
            else:
                avg_win_rate = 0.0
                avg_fitness = 0.0
                total_trades = 0
            
            return TradingPerformanceMetrics(
                total_return=0.0,  # Would be calculated from portfolio value
                daily_return=0.0,
                sharpe_ratio=avg_fitness,  # Using fitness as proxy
                max_drawdown=0.0,
                win_rate=avg_win_rate,
                total_trades=total_trades,
                avg_slippage=summary.get('avg_latency_ms', 0) / 1000,  # Convert to percentage
                avg_latency_ms=summary.get('avg_latency_ms', 0),
                execution_success_rate=summary.get('success_rate', 0),
                total_exposure=summary.get('portfolio_value', 0),
                active_positions=len([s for s in top_strategies if s.total_trades > 0])
            )
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return TradingPerformanceMetrics()
    
    def _collect_risk_metrics(self) -> Tuple[RiskLevel, List[str], MarketRegime]:
        """Collect risk management metrics."""
        if not self.risk_manager:
            return RiskLevel.LOW, [], MarketRegime.UNKNOWN
        
        try:
            risk_metrics = self.risk_manager.get_risk_metrics()
            return (
                risk_metrics.risk_level,
                [str(cb) for cb in risk_metrics.active_circuit_breakers],
                risk_metrics.current_regime
            )
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
            return RiskLevel.CRITICAL, [], MarketRegime.UNKNOWN
    
    def _determine_monitoring_status(self, system_metrics: SystemHealthMetrics,
                                   genetic_metrics: GeneticEvolutionMetrics,
                                   trading_metrics: TradingPerformanceMetrics,
                                   risk_level: RiskLevel) -> MonitoringStatus:
        """Determine overall monitoring system status."""
        
        # Critical conditions
        if (risk_level == RiskLevel.EMERGENCY or
            system_metrics.cpu_usage_percent > 95 or
            system_metrics.memory_usage_percent > 95):
            return MonitoringStatus.CRITICAL
        
        # Degraded conditions
        if (risk_level == RiskLevel.CRITICAL or
            system_metrics.cpu_usage_percent > 80 or
            system_metrics.memory_usage_percent > 80 or
            trading_metrics.execution_success_rate < 0.9):
            return MonitoringStatus.DEGRADED
        
        # Warning conditions
        if (risk_level == RiskLevel.HIGH or
            system_metrics.cpu_usage_percent > 70 or
            system_metrics.memory_usage_percent > 70 or
            trading_metrics.execution_success_rate < 0.95):
            return MonitoringStatus.WARNING
        
        return MonitoringStatus.HEALTHY
    
    def _check_alert_conditions(self, snapshot: MonitoringSnapshot):
        """Check for alert conditions in the monitoring snapshot."""
        
        # System health alerts
        self._check_system_health_alerts(snapshot.system_metrics)
        
        # Trading performance alerts
        self._check_trading_performance_alerts(snapshot.trading_metrics)
        
        # Risk management alerts
        self._check_risk_management_alerts(snapshot.current_risk_level, 
                                         snapshot.active_circuit_breakers)
        
        # Genetic evolution alerts
        self._check_genetic_evolution_alerts(snapshot.genetic_metrics)
    
    def _check_system_health_alerts(self, metrics: SystemHealthMetrics):
        """Check system health alert conditions."""
        thresholds = self.alert_manager.alert_thresholds['system_health']
        
        # CPU usage alerts
        if metrics.cpu_usage_percent > thresholds['cpu_usage_critical']:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL,
                AlertCategory.SYSTEM_HEALTH,
                "Critical CPU Usage",
                f"CPU usage at {metrics.cpu_usage_percent:.1f}%",
                "SystemHealthMonitor",
                thresholds['cpu_usage_critical'],
                metrics.cpu_usage_percent
            )
        elif metrics.cpu_usage_percent > thresholds['cpu_usage_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.SYSTEM_HEALTH,
                "High CPU Usage",
                f"CPU usage at {metrics.cpu_usage_percent:.1f}%",
                "SystemHealthMonitor",
                thresholds['cpu_usage_warning'],
                metrics.cpu_usage_percent
            )
        
        # Memory usage alerts
        if metrics.memory_usage_percent > thresholds['memory_usage_critical']:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL,
                AlertCategory.SYSTEM_HEALTH,
                "Critical Memory Usage",
                f"Memory usage at {metrics.memory_usage_percent:.1f}%",
                "SystemHealthMonitor",
                thresholds['memory_usage_critical'],
                metrics.memory_usage_percent
            )
        elif metrics.memory_usage_percent > thresholds['memory_usage_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.SYSTEM_HEALTH,
                "High Memory Usage",
                f"Memory usage at {metrics.memory_usage_percent:.1f}%",
                "SystemHealthMonitor",
                thresholds['memory_usage_warning'],
                metrics.memory_usage_percent
            )
        
        # Network latency alerts
        if metrics.network_latency_ms > thresholds['network_latency_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.SYSTEM_HEALTH,
                "High Network Latency",
                f"Network latency at {metrics.network_latency_ms:.0f}ms",
                "SystemHealthMonitor",
                thresholds['network_latency_warning'],
                metrics.network_latency_ms
            )
    
    def _check_trading_performance_alerts(self, metrics: TradingPerformanceMetrics):
        """Check trading performance alert conditions."""
        thresholds = self.alert_manager.alert_thresholds['trading_performance']
        
        # Drawdown alerts
        if abs(metrics.max_drawdown) > thresholds['max_drawdown_critical']:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL,
                AlertCategory.TRADING_PERFORMANCE,
                "Critical Drawdown",
                f"Maximum drawdown at {metrics.max_drawdown:.1%}",
                "TradingPerformanceMonitor",
                thresholds['max_drawdown_critical'],
                abs(metrics.max_drawdown)
            )
        elif abs(metrics.max_drawdown) > thresholds['max_drawdown_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.TRADING_PERFORMANCE,
                "High Drawdown",
                f"Maximum drawdown at {metrics.max_drawdown:.1%}",
                "TradingPerformanceMonitor",
                thresholds['max_drawdown_warning'],
                abs(metrics.max_drawdown)
            )
        
        # Sharpe ratio alerts
        if metrics.sharpe_ratio < thresholds['negative_sharpe_threshold']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.TRADING_PERFORMANCE,
                "Negative Sharpe Ratio",
                f"Sharpe ratio at {metrics.sharpe_ratio:.2f}",
                "TradingPerformanceMonitor",
                thresholds['negative_sharpe_threshold'],
                metrics.sharpe_ratio
            )
        
        # Execution success rate alerts
        if metrics.execution_success_rate < thresholds['execution_success_threshold']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.ORDER_EXECUTION,
                "Low Execution Success Rate",
                f"Execution success rate at {metrics.execution_success_rate:.1%}",
                "TradingPerformanceMonitor",
                thresholds['execution_success_threshold'],
                metrics.execution_success_rate
            )
    
    def _check_risk_management_alerts(self, risk_level: RiskLevel, 
                                    circuit_breakers: List[str]):
        """Check risk management alert conditions."""
        
        # Risk level alerts
        if risk_level == RiskLevel.EMERGENCY:
            self.alert_manager.trigger_alert(
                AlertLevel.EMERGENCY,
                AlertCategory.RISK_MANAGEMENT,
                "Emergency Risk Level",
                "Risk management system in emergency mode",
                "RiskManager"
            )
        elif risk_level == RiskLevel.CRITICAL:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL,
                AlertCategory.RISK_MANAGEMENT,
                "Critical Risk Level",
                "Risk management system at critical level",
                "RiskManager"
            )
        elif risk_level == RiskLevel.HIGH:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.RISK_MANAGEMENT,
                "High Risk Level",
                "Risk management system at high risk level",
                "RiskManager"
            )
        
        # Circuit breaker alerts
        if circuit_breakers:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL,
                AlertCategory.RISK_MANAGEMENT,
                "Circuit Breakers Active",
                f"Active circuit breakers: {', '.join(circuit_breakers)}",
                "RiskManager",
                context_data={'active_breakers': circuit_breakers}
            )
    
    def _check_genetic_evolution_alerts(self, metrics: GeneticEvolutionMetrics):
        """Check genetic evolution alert conditions."""
        thresholds = self.alert_manager.alert_thresholds['genetic_evolution']
        
        # Slow evaluation alerts
        if metrics.evaluation_time > thresholds['slow_evaluation_time']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.GENETIC_EVOLUTION,
                "Slow Genetic Evaluation",
                f"Evaluation time at {metrics.evaluation_time:.1f}s",
                "GeneticEngine",
                thresholds['slow_evaluation_time'],
                metrics.evaluation_time
            )
        
        # Low diversity alerts
        if metrics.diversity_score < thresholds['low_diversity_threshold']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.GENETIC_EVOLUTION,
                "Low Population Diversity",
                f"Diversity score at {metrics.diversity_score:.3f} - risk of premature convergence",
                "GeneticEngine",
                thresholds['low_diversity_threshold'],
                metrics.diversity_score
            )
        
        # High memory usage alerts
        if metrics.memory_usage_gb > thresholds['high_memory_usage']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING,
                AlertCategory.GENETIC_EVOLUTION,
                "High Memory Usage",
                f"Genetic evaluation using {metrics.memory_usage_gb:.1f}GB",
                "GeneticEngine",
                thresholds['high_memory_usage'],
                metrics.memory_usage_gb
            )
    
    def _log_monitoring_status(self, snapshot: MonitoringSnapshot):
        """Log periodic monitoring status."""
        logger.info(f"Monitoring Status: {snapshot.status}")
        logger.info(f"  - Risk Level: {snapshot.current_risk_level}")
        logger.info(f"  - System: CPU {snapshot.system_metrics.cpu_usage_percent:.1f}%, "
                   f"Memory {snapshot.system_metrics.memory_usage_percent:.1f}%")
        logger.info(f"  - Trading: {snapshot.trading_metrics.total_trades} trades, "
                   f"Success Rate {snapshot.trading_metrics.execution_success_rate:.1%}")
        logger.info(f"  - Genetic: Gen {snapshot.genetic_metrics.generation}, "
                   f"Fitness {snapshot.genetic_metrics.best_fitness:.3f}")
        logger.info(f"  - Active Alerts: {len(snapshot.active_alerts)}")
    
    def get_current_snapshot(self) -> Optional[MonitoringSnapshot]:
        """Get the most recent monitoring snapshot."""
        if self.snapshot_history:
            return self.snapshot_history[-1]
        return None
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for monitoring dashboard."""
        current_snapshot = self.get_current_snapshot()
        
        if not current_snapshot:
            return {'status': 'no_data', 'message': 'No monitoring data available'}
        
        # Get trend analyses
        genetic_trends = self.performance_analyzer.analyze_genetic_trends()
        trading_trends = self.performance_analyzer.analyze_trading_trends()
        system_trends = self.performance_analyzer.analyze_system_trends()
        
        return {
            'status': current_snapshot.status,
            'timestamp': current_snapshot.timestamp.isoformat(),
            'collection_count': self.collection_count,
            
            # Current metrics
            'current_metrics': {
                'genetic': asdict(current_snapshot.genetic_metrics),
                'trading': asdict(current_snapshot.trading_metrics),
                'system': asdict(current_snapshot.system_metrics),
                'risk_level': current_snapshot.current_risk_level,
                'market_regime': current_snapshot.market_regime
            },
            
            # Trend analyses
            'trends': {
                'genetic': genetic_trends,
                'trading': trading_trends,
                'system': system_trends
            },
            
            # Active alerts
            'alerts': {
                'total': len(current_snapshot.active_alerts),
                'by_level': self._count_alerts_by_level(current_snapshot.active_alerts),
                'recent': [asdict(alert) for alert in current_snapshot.active_alerts[:10]]
            },
            
            # System summary
            'summary': {
                'uptime_hours': current_snapshot.system_metrics.uptime_hours,
                'monitoring_active': self.monitoring_active,
                'snapshot_history_size': len(self.snapshot_history),
                'overall_health': self._calculate_overall_health_score(current_snapshot)
            }
        }
    
    def _count_alerts_by_level(self, alerts: List[MonitoringAlert]) -> Dict[str, int]:
        """Count alerts by severity level."""
        counts = {level.value: 0 for level in AlertLevel}
        
        for alert in alerts:
            counts[alert.level.value] += 1
        
        return counts
    
    def _calculate_overall_health_score(self, snapshot: MonitoringSnapshot) -> float:
        """Calculate overall system health score (0-100)."""
        
        # Base score
        score = 100.0
        
        # Deduct points for system issues
        if snapshot.system_metrics.cpu_usage_percent > 80:
            score -= 20
        elif snapshot.system_metrics.cpu_usage_percent > 60:
            score -= 10
        
        if snapshot.system_metrics.memory_usage_percent > 80:
            score -= 20
        elif snapshot.system_metrics.memory_usage_percent > 60:
            score -= 10
        
        # Deduct points for risk level
        risk_penalties = {
            RiskLevel.LOW: 0,
            RiskLevel.MODERATE: 5,
            RiskLevel.HIGH: 15,
            RiskLevel.CRITICAL: 30,
            RiskLevel.EMERGENCY: 50
        }
        score -= risk_penalties.get(snapshot.current_risk_level, 0)
        
        # Deduct points for active alerts
        alert_penalties = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 2,
            AlertLevel.CRITICAL: 5,
            AlertLevel.EMERGENCY: 10
        }
        
        for alert in snapshot.active_alerts:
            score -= alert_penalties.get(alert.level, 0)
        
        # Deduct points for poor trading performance
        if snapshot.trading_metrics.execution_success_rate < 0.9:
            score -= 15
        elif snapshot.trading_metrics.execution_success_rate < 0.95:
            score -= 5
        
        return max(0.0, min(100.0, score))


# Testing and validation functions
async def test_real_time_monitoring():
    """Test function for real-time monitoring system."""
    
    print("=== Real-time Monitoring System Test ===")
    
    # Initialize monitoring system
    monitoring = RealTimeMonitoringSystem()
    print("✅ Monitoring system initialized")
    
    # Test alert manager
    print("\nTesting alert manager...")
    alert = monitoring.alert_manager.trigger_alert(
        AlertLevel.WARNING,
        AlertCategory.SYSTEM_HEALTH,
        "Test Alert",
        "This is a test alert",
        "TestComponent",
        80.0,
        85.0
    )
    print(f"✅ Alert triggered: {alert.alert_id}")
    
    # Test snapshot collection
    print("\nTesting snapshot collection...")
    snapshot = monitoring.collect_monitoring_snapshot()
    print(f"✅ Snapshot collected: {snapshot.status}")
    print(f"   - System CPU: {snapshot.system_metrics.cpu_usage_percent:.1f}%")
    print(f"   - System Memory: {snapshot.system_metrics.memory_usage_percent:.1f}%")
    print(f"   - Active Alerts: {len(snapshot.active_alerts)}")
    
    # Test dashboard data
    print("\nTesting dashboard data...")
    monitoring.snapshot_history.append(snapshot)  # Add snapshot to history
    dashboard_data = monitoring.get_monitoring_dashboard_data()
    print(f"✅ Dashboard data generated")
    print(f"   - Status: {dashboard_data['status']}")
    print(f"   - Health Score: {dashboard_data['summary']['overall_health']:.1f}")
    print(f"   - Alerts: {dashboard_data['alerts']['total']}")
    
    # Test performance analyzer
    print("\nTesting performance analyzer...")
    monitoring.performance_analyzer.add_system_metrics(snapshot.system_metrics)
    system_trends = monitoring.performance_analyzer.analyze_system_trends()
    print(f"✅ Performance analysis complete")
    print(f"   - Analysis Status: {system_trends.get('status', 'analyzed')}")
    
    # Test brief monitoring run
    print("\nTesting monitoring loop (5 seconds)...")
    monitoring.start_monitoring()
    await asyncio.sleep(5)
    monitoring.stop_monitoring()
    print(f"✅ Monitoring loop tested")
    print(f"   - Collections: {monitoring.collection_count}")
    print(f"   - History Size: {len(monitoring.snapshot_history)}")
    
    print("\n=== Real-time Monitoring Test Complete ===")


if __name__ == "__main__":
    """Real-time monitoring system testing and validation."""
    
    import asyncio
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_real_time_monitoring())