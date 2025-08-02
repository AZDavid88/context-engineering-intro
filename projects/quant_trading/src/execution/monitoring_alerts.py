"""
Monitoring Alert System - Comprehensive alert management and notification dispatch.
Handles alert detection, multi-channel notifications, and escalation management for genetic trading system.
"""

import asyncio
import logging
import time
import uuid
import json
import smtplib
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading

from .monitoring_core import (
    SystemHealthMetrics, GeneticEvolutionMetrics, TradingPerformanceMetrics,
    MonitoringSnapshot, MonitoringStatus, AlertLevel, AlertCategory
)
from src.config.settings import get_settings, Settings
from src.execution.risk_management import RiskLevel, MarketRegime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringAlert:
    """Individual monitoring alert with metadata and context."""
    
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
class NotificationChannel:
    """Configuration for a notification channel."""
    channel_type: str  # 'email', 'webhook', 'slack'
    config: Dict[str, Any]
    enabled: bool = True
    alert_levels: List[AlertLevel] = field(default_factory=lambda: [AlertLevel.CRITICAL, AlertLevel.EMERGENCY])


class AlertManager:
    """Core alert management with deduplication and callback system."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize alert manager."""
        self.settings = settings or get_settings()
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.alert_callbacks: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        
        # Alert suppression to prevent spam
        self.alert_suppression = {}  # alert_key -> last_sent_time
        self.suppression_window = 300  # 5 minutes
        
        logger.info("Alert manager initialized")
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for different categories."""
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
        """Trigger a monitoring alert with deduplication."""
        
        # Create unique alert key for deduplication
        alert_key = f"{category.value}_{title}_{source_component}"
        
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
        
        logger.warning(f"Alert triggered: {level.value} - {title} from {source_component}")
        
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
        """Acknowledge an active alert."""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.context_data['acknowledged_by'] = acknowledged_by
                alert.context_data['acknowledged_at'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve and remove an active alert."""
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
        """Get active alerts with optional filtering."""
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
        """Register callback for alert level."""
        self.alert_callbacks[level].append(callback)
        logger.debug(f"Alert callback registered for level: {level.value}")


class NotificationDispatcher:
    """Multi-channel notification dispatcher for alerts."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize notification dispatcher."""
        self.settings = settings or get_settings()
        self.channels: List[NotificationChannel] = []
        self.dispatch_queue = deque()
        self.rate_limits = {}  # channel -> last_sent_time
        self.rate_limit_window = 60  # 1 minute
        
        self._initialize_channels()
        logger.info("Notification dispatcher initialized")
    
    def _initialize_channels(self):
        """Initialize notification channels from settings."""
        # Email notifications
        if hasattr(self.settings, 'email_notifications') and self.settings.email_notifications.get('enabled'):
            self.channels.append(NotificationChannel(
                channel_type='email',
                config=self.settings.email_notifications,
                alert_levels=[AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
            ))
        
        # Webhook notifications
        if hasattr(self.settings, 'webhook_notifications') and self.settings.webhook_notifications.get('enabled'):
            self.channels.append(NotificationChannel(
                channel_type='webhook',
                config=self.settings.webhook_notifications,
                alert_levels=[AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
            ))
    
    def dispatch_alert(self, alert: MonitoringAlert):
        """Dispatch alert to appropriate notification channels."""
        for channel in self.channels:
            if not channel.enabled:
                continue
                
            if alert.level not in channel.alert_levels:
                continue
            
            if self._is_rate_limited(channel.channel_type):
                logger.debug(f"Rate limit reached for {channel.channel_type}")
                continue
            
            try:
                if channel.channel_type == 'email':
                    self._send_email_notification(alert, channel.config)
                elif channel.channel_type == 'webhook':
                    self._send_webhook_notification(alert, channel.config)
                
                # Update rate limiting
                self.rate_limits[channel.channel_type] = time.time()
                
            except Exception as e:
                logger.error(f"Failed to send {channel.channel_type} notification: {e}")
    
    def _is_rate_limited(self, channel_type: str) -> bool:
        """Check if channel is rate limited."""
        if channel_type not in self.rate_limits:
            return False
        
        time_since_last = time.time() - self.rate_limits[channel_type]
        return time_since_last < self.rate_limit_window
    
    def _send_email_notification(self, alert: MonitoringAlert, config: Dict[str, Any]):
        """Send email notification."""
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
        
        body = f"""
Alert Details:
- Level: {alert.level.value}
- Category: {alert.category.value}
- Source: {alert.source_component}
- Message: {alert.message}
- Triggered: {alert.last_triggered.isoformat()}
- Trigger Count: {alert.trigger_count}

Threshold: {alert.threshold_value}
Current Value: {alert.current_value}

Alert ID: {alert.alert_id}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        if config.get('use_tls'):
            server.starttls()
        if config.get('username'):
            server.login(config['username'], config['password'])
        
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email notification sent for alert: {alert.alert_id}")
    
    def _send_webhook_notification(self, alert: MonitoringAlert, config: Dict[str, Any]):
        """Send webhook notification."""
        payload = {
            'alert_id': alert.alert_id,
            'level': alert.level.value,
            'category': alert.category.value,
            'title': alert.title,
            'message': alert.message,
            'source_component': alert.source_component,
            'threshold_value': alert.threshold_value,
            'current_value': alert.current_value,
            'trigger_count': alert.trigger_count,
            'first_triggered': alert.first_triggered.isoformat(),
            'last_triggered': alert.last_triggered.isoformat(),
            'context_data': alert.context_data
        }
        
        headers = {'Content-Type': 'application/json'}
        if config.get('auth_token'):
            headers['Authorization'] = f"Bearer {config['auth_token']}"
        
        response = requests.post(
            config['webhook_url'],
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Webhook notification sent for alert: {alert.alert_id}")
        else:
            logger.error(f"Webhook notification failed: {response.status_code}")


class EscalationManager:
    """Manages alert escalation policies and workflows."""
    
    def __init__(self, alert_manager: AlertManager, notification_dispatcher: NotificationDispatcher):
        """Initialize escalation manager."""
        self.alert_manager = alert_manager
        self.notification_dispatcher = notification_dispatcher
        self.escalation_policies = self._initialize_escalation_policies()
        self.escalation_timers = {}  # alert_id -> timer_thread
        
        logger.info("Escalation manager initialized")
    
    def _initialize_escalation_policies(self) -> Dict[AlertLevel, Dict[str, Any]]:
        """Initialize escalation policies by alert level."""
        return {
            AlertLevel.WARNING: {
                'escalation_delay': 1800,  # 30 minutes
                'escalate_to': AlertLevel.CRITICAL,
                'max_escalations': 1
            },
            AlertLevel.CRITICAL: {
                'escalation_delay': 900,   # 15 minutes
                'escalate_to': AlertLevel.EMERGENCY,
                'max_escalations': 2
            },
            AlertLevel.EMERGENCY: {
                'escalation_delay': 300,   # 5 minutes
                'escalate_to': None,  # No further escalation
                'max_escalations': 0
            }
        }
    
    def start_escalation_timer(self, alert: MonitoringAlert):
        """Start escalation timer for an alert."""
        if alert.level not in self.escalation_policies:
            return
        
        policy = self.escalation_policies[alert.level]
        if policy['escalate_to'] is None:
            return
        
        # Create escalation timer
        timer = threading.Timer(
            policy['escalation_delay'],
            self._escalate_alert,
            args=[alert.alert_id, policy]
        )
        
        self.escalation_timers[alert.alert_id] = timer
        timer.start()
        
        logger.debug(f"Escalation timer started for alert: {alert.alert_id}")
    
    def stop_escalation_timer(self, alert_id: str):
        """Stop escalation timer for an alert."""
        if alert_id in self.escalation_timers:
            self.escalation_timers[alert_id].cancel()
            del self.escalation_timers[alert_id]
            logger.debug(f"Escalation timer stopped for alert: {alert_id}")
    
    def _escalate_alert(self, alert_id: str, policy: Dict[str, Any]):
        """Escalate an alert to the next level."""
        # Find the alert
        alert = None
        for active_alert in self.alert_manager.active_alerts.values():
            if active_alert.alert_id == alert_id:
                alert = active_alert
                break
        
        if not alert or alert.resolved or alert.acknowledged:
            return
        
        # Escalate the alert
        alert.level = policy['escalate_to']
        alert.context_data['escalated'] = True
        alert.context_data['escalated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Send escalated notification
        self.notification_dispatcher.dispatch_alert(alert)
        
        logger.warning(f"Alert escalated: {alert_id} to {policy['escalate_to'].value}")
        
        # Start new escalation timer if applicable
        if policy['escalate_to'] in self.escalation_policies:
            self.start_escalation_timer(alert)


class AlertChecker:
    """Consolidated alert condition checker for all monitoring categories."""
    
    def __init__(self, alert_manager: AlertManager):
        """Initialize alert checker."""
        self.alert_manager = alert_manager
        
    def check_all_conditions(self, snapshot: MonitoringSnapshot):
        """Check all alert conditions in monitoring snapshot."""
        self._check_system_health_alerts(snapshot.system_metrics)
        self._check_trading_performance_alerts(snapshot.trading_metrics)
        self._check_genetic_evolution_alerts(snapshot.genetic_metrics)
        self._check_risk_management_alerts(snapshot.risk_level, snapshot.active_circuit_breakers)
    
    def _check_system_health_alerts(self, metrics: SystemHealthMetrics):
        """Check system health alert conditions."""
        thresholds = self.alert_manager.alert_thresholds['system_health']
        
        # CPU usage alerts
        if metrics.cpu_usage_percent > thresholds['cpu_usage_critical']:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL, AlertCategory.SYSTEM_HEALTH,
                "Critical CPU Usage", f"CPU usage at {metrics.cpu_usage_percent:.1f}%",
                "SystemHealthMonitor", thresholds['cpu_usage_critical'], metrics.cpu_usage_percent
            )
        elif metrics.cpu_usage_percent > thresholds['cpu_usage_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.SYSTEM_HEALTH,
                "High CPU Usage", f"CPU usage at {metrics.cpu_usage_percent:.1f}%",
                "SystemHealthMonitor", thresholds['cpu_usage_warning'], metrics.cpu_usage_percent
            )
        
        # Memory usage alerts
        if metrics.memory_usage_percent > thresholds['memory_usage_critical']:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL, AlertCategory.SYSTEM_HEALTH,
                "Critical Memory Usage", f"Memory usage at {metrics.memory_usage_percent:.1f}%",
                "SystemHealthMonitor", thresholds['memory_usage_critical'], metrics.memory_usage_percent
            )
        elif metrics.memory_usage_percent > thresholds['memory_usage_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.SYSTEM_HEALTH,
                "High Memory Usage", f"Memory usage at {metrics.memory_usage_percent:.1f}%",
                "SystemHealthMonitor", thresholds['memory_usage_warning'], metrics.memory_usage_percent
            )
    
    def _check_trading_performance_alerts(self, metrics: TradingPerformanceMetrics):
        """Check trading performance alert conditions."""
        thresholds = self.alert_manager.alert_thresholds['trading_performance']
        
        # Execution success rate alerts
        if metrics.execution_success_rate < thresholds['execution_success_threshold']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.TRADING_PERFORMANCE,
                "Low Execution Success Rate", 
                f"Success rate at {metrics.execution_success_rate:.1%}",
                "TradingEngine", thresholds['execution_success_threshold'], metrics.execution_success_rate
            )
        
        # Drawdown alerts
        if metrics.max_drawdown > thresholds['max_drawdown_critical']:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL, AlertCategory.TRADING_PERFORMANCE,
                "Critical Drawdown", f"Max drawdown at {metrics.max_drawdown:.1%}",
                "TradingEngine", thresholds['max_drawdown_critical'], metrics.max_drawdown
            )
        elif metrics.max_drawdown > thresholds['max_drawdown_warning']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.TRADING_PERFORMANCE,
                "High Drawdown", f"Max drawdown at {metrics.max_drawdown:.1%}",
                "TradingEngine", thresholds['max_drawdown_warning'], metrics.max_drawdown
            )
    
    def _check_genetic_evolution_alerts(self, metrics: GeneticEvolutionMetrics):
        """Check genetic evolution alert conditions."""
        thresholds = self.alert_manager.alert_thresholds['genetic_evolution']
        
        # Low diversity alerts
        if metrics.diversity_score < thresholds['low_diversity_threshold']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.GENETIC_EVOLUTION,
                "Low Population Diversity",
                f"Diversity score at {metrics.diversity_score:.3f} - risk of premature convergence",
                "GeneticEngine", thresholds['low_diversity_threshold'], metrics.diversity_score
            )
        
        # Slow evaluation alerts
        if metrics.evaluation_time > thresholds['slow_evaluation_time']:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.GENETIC_EVOLUTION,
                "Slow Genetic Evaluation", f"Evaluation time at {metrics.evaluation_time:.1f}s",
                "GeneticEngine", thresholds['slow_evaluation_time'], metrics.evaluation_time
            )
    
    def _check_risk_management_alerts(self, risk_level: RiskLevel, circuit_breakers: List[str]):
        """Check risk management alert conditions."""
        if risk_level == RiskLevel.EMERGENCY:
            self.alert_manager.trigger_alert(
                AlertLevel.EMERGENCY, AlertCategory.RISK_MANAGEMENT,
                "Emergency Risk Level", "Risk management escalated to emergency level",
                "RiskManager", context_data={'risk_level': risk_level.value}
            )
        elif risk_level == RiskLevel.CRITICAL:
            self.alert_manager.trigger_alert(
                AlertLevel.CRITICAL, AlertCategory.RISK_MANAGEMENT,
                "Critical Risk Level", "Risk management escalated to critical level",
                "RiskManager", context_data={'risk_level': risk_level.value}
            )
        
        if circuit_breakers:
            self.alert_manager.trigger_alert(
                AlertLevel.WARNING, AlertCategory.RISK_MANAGEMENT,
                "Circuit Breakers Active", f"Active circuit breakers: {', '.join(circuit_breakers)}",
                "RiskManager", context_data={'active_breakers': circuit_breakers}
            )