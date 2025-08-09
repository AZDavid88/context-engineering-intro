"""
Smart Alerting System for Human-in-the-Loop Decision Points

Provides intelligent alerting for the 5% of decisions requiring human judgment,
with multiple notification channels and urgency-based routing.

Integration Points:
- AutomatedDecisionEngine for decision alerts
- MonitoringEngine for system health alerts
- ConfigStrategyLoader for strategy lifecycle alerts
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from src.execution.automated_decision_engine import DecisionResult, DecisionUrgency
from src.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    """Available alert channels."""
    CONSOLE = "console"          # Console logging (development)
    EMAIL = "email"              # Email notifications (future)
    DISCORD = "discord"          # Discord webhook (future)  
    SLACK = "slack"              # Slack notifications (future)
    FILE = "file"                # File-based alerts


class AlertPriority(str, Enum):
    """Alert priority levels for routing and throttling."""
    INFORMATIONAL = "informational"
    WARNING = "warning"
    URGENT = "urgent"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configuration for alerting behavior."""
    
    # Channel configuration
    enabled_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.CONSOLE, AlertChannel.FILE])
    
    # Throttling configuration (minutes between same alert types)
    throttle_intervals: Dict[str, int] = field(default_factory=lambda: {
        "informational": 60,    # 1 hour
        "warning": 30,          # 30 minutes
        "urgent": 15,           # 15 minutes
        "critical": 0           # No throttling for critical alerts
    })
    
    # Alert formatting
    include_metadata: bool = True
    include_timestamp: bool = True
    max_message_length: int = 1000
    
    # File alerts configuration
    alerts_file: str = "logs/decision_alerts.log"
    max_alert_history: int = 1000


@dataclass
class AlertRecord:
    """Record of a sent alert."""
    
    timestamp: datetime
    alert_type: str
    priority: AlertPriority
    message: str
    channel: AlertChannel
    sent_successfully: bool
    decision_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertingSystem:
    """Smart alerting for human-in-the-loop decisions."""
    
    def __init__(self, 
                 config: Optional[AlertConfig] = None,
                 settings: Optional[Settings] = None):
        
        self.settings = settings or get_settings()
        self.config = config or AlertConfig()
        
        # Alert history and throttling
        self.alert_history: List[AlertRecord] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Setup alert channels
        self._setup_alert_channels()
        
        logger.info(f"AlertingSystem initialized with channels: {self.config.enabled_channels}")
    
    def _setup_alert_channels(self):
        """Setup and validate alert channels."""
        
        # Ensure alerts directory exists for file-based alerts
        if AlertChannel.FILE in self.config.enabled_channels:
            alerts_file_path = Path(self.config.alerts_file)
            alerts_file_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Future: Setup email, Discord, Slack configurations
        
        logger.debug(f"Alert channels configured: {len(self.config.enabled_channels)} channels active")
    
    async def send_decision_alert(self, decision_result: DecisionResult) -> bool:
        """
        Send alert for decision requiring human review.
        
        Args:
            decision_result: DecisionResult requiring human attention
            
        Returns:
            True if alert sent successfully to at least one channel
        """
        
        if not decision_result.requires_human_review:
            logger.debug(f"Decision {decision_result.decision_type.value} does not require human review - skipping alert")
            return True
        
        # Map decision urgency to alert priority
        priority = self._map_urgency_to_priority(decision_result.urgency)
        
        # Check throttling
        alert_key = f"{decision_result.decision_type.value}_{priority.value}"
        if self._is_throttled(alert_key, priority):
            logger.debug(f"Alert throttled: {alert_key}")
            return False
        
        # Format alert message
        alert_message = self._format_decision_alert(decision_result)
        
        # Send to all enabled channels
        success_count = 0
        total_channels = len(self.config.enabled_channels)
        
        for channel in self.config.enabled_channels:
            try:
                channel_success = await self._send_to_channel(
                    channel, alert_message, priority, decision_result
                )
                if channel_success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
        
        # Record alert attempt
        overall_success = success_count > 0
        self._record_alert(
            alert_type=decision_result.decision_type.value,
            priority=priority,
            message=alert_message,
            channel=AlertChannel.CONSOLE,  # Primary channel for recording
            success=overall_success,
            decision_type=decision_result.decision_type.value,
            metadata={
                "success_channels": success_count,
                "total_channels": total_channels,
                "decision_confidence": decision_result.confidence,
                "urgency": decision_result.urgency.value
            }
        )
        
        # Update throttling timestamp
        if overall_success:
            self.last_alert_times[alert_key] = datetime.now(timezone.utc)
        
        return overall_success
    
    async def send_system_alert(self, 
                              alert_type: str,
                              message: str,
                              priority: AlertPriority = AlertPriority.WARNING,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send system-level alert (not decision-related).
        
        Args:
            alert_type: Type of system alert
            message: Alert message
            priority: Alert priority level
            metadata: Additional alert metadata
            
        Returns:
            True if alert sent successfully
        """
        
        # Check throttling for system alerts
        alert_key = f"system_{alert_type}_{priority.value}"
        if self._is_throttled(alert_key, priority):
            logger.debug(f"System alert throttled: {alert_key}")
            return False
        
        # Format system alert message
        formatted_message = self._format_system_alert(alert_type, message, priority, metadata)
        
        # Send to channels
        success_count = 0
        for channel in self.config.enabled_channels:
            try:
                channel_success = await self._send_to_channel(
                    channel, formatted_message, priority, None, alert_type
                )
                if channel_success:
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to send system alert via {channel}: {e}")
        
        overall_success = success_count > 0
        
        # Record alert
        self._record_alert(
            alert_type=alert_type,
            priority=priority,
            message=formatted_message,
            channel=AlertChannel.CONSOLE,
            success=overall_success,
            metadata=metadata or {}
        )
        
        # Update throttling
        if overall_success:
            self.last_alert_times[alert_key] = datetime.now(timezone.utc)
        
        return overall_success
    
    def _map_urgency_to_priority(self, urgency: DecisionUrgency) -> AlertPriority:
        """Map decision urgency to alert priority."""
        
        mapping = {
            DecisionUrgency.LOW: AlertPriority.INFORMATIONAL,
            DecisionUrgency.MEDIUM: AlertPriority.WARNING,
            DecisionUrgency.HIGH: AlertPriority.URGENT,
            DecisionUrgency.CRITICAL: AlertPriority.CRITICAL
        }
        
        return mapping.get(urgency, AlertPriority.WARNING)
    
    def _is_throttled(self, alert_key: str, priority: AlertPriority) -> bool:
        """Check if alert should be throttled."""
        
        if priority == AlertPriority.CRITICAL:
            return False  # Never throttle critical alerts
        
        throttle_minutes = self.config.throttle_intervals.get(priority.value, 30)
        if throttle_minutes == 0:
            return False
        
        last_alert_time = self.last_alert_times.get(alert_key)
        if not last_alert_time:
            return False
        
        time_since_last = datetime.now(timezone.utc) - last_alert_time
        return time_since_last.total_seconds() < (throttle_minutes * 60)
    
    def _format_decision_alert(self, decision_result: DecisionResult) -> str:
        """Format decision result as alert message."""
        
        urgency_emoji = {
            DecisionUrgency.LOW: "ℹ️",
            DecisionUrgency.MEDIUM: "⚠️", 
            DecisionUrgency.HIGH: "🚨",
            DecisionUrgency.CRITICAL: "🚨🚨🚨"
        }
        
        emoji = urgency_emoji.get(decision_result.urgency, "❓")
        
        # Base message
        message_lines = [
            f"{emoji} {decision_result.urgency.value.upper()} TRADING DECISION ALERT",
            "",
            f"📊 Decision Type: {decision_result.decision_type.value.replace('_', ' ').title()}",
            f"🎯 Decision: {decision_result.decision}",
            f"🔍 Reasoning: {decision_result.reasoning}",
            f"📈 Confidence: {decision_result.confidence:.1%}"
        ]
        
        # Add timestamp if configured
        if self.config.include_timestamp:
            message_lines.append(f"⏰ Time: {decision_result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Add metadata if configured and available
        if self.config.include_metadata and decision_result.metadata:
            message_lines.append("")
            message_lines.append(f"📋 Metadata: {decision_result.metadata}")
        
        message_lines.append("")
        message_lines.append("Action Required: Please review this decision and take appropriate action.")
        
        full_message = "\n".join(message_lines)
        
        # Truncate if too long
        if len(full_message) > self.config.max_message_length:
            full_message = full_message[:self.config.max_message_length - 10] + "...[TRUNCATED]"
        
        return full_message
    
    def _format_system_alert(self, 
                           alert_type: str, 
                           message: str, 
                           priority: AlertPriority,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format system alert message."""
        
        priority_emoji = {
            AlertPriority.INFORMATIONAL: "ℹ️",
            AlertPriority.WARNING: "⚠️",
            AlertPriority.URGENT: "🚨", 
            AlertPriority.CRITICAL: "🚨🚨🚨"
        }
        
        emoji = priority_emoji.get(priority, "❓")
        
        message_lines = [
            f"{emoji} {priority.value.upper()} SYSTEM ALERT",
            "",
            f"🔧 Alert Type: {alert_type}",
            f"📝 Message: {message}"
        ]
        
        if self.config.include_timestamp:
            message_lines.append(f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        if self.config.include_metadata and metadata:
            message_lines.append("")
            message_lines.append(f"📋 Details: {metadata}")
        
        full_message = "\n".join(message_lines)
        
        # Truncate if needed
        if len(full_message) > self.config.max_message_length:
            full_message = full_message[:self.config.max_message_length - 10] + "...[TRUNCATED]"
        
        return full_message
    
    async def _send_to_channel(self, 
                             channel: AlertChannel, 
                             message: str, 
                             priority: AlertPriority,
                             decision_result: Optional[DecisionResult] = None,
                             alert_type: Optional[str] = None) -> bool:
        """Send alert to specific channel."""
        
        try:
            if channel == AlertChannel.CONSOLE:
                return await self._send_console_alert(message, priority)
            elif channel == AlertChannel.FILE:
                return await self._send_file_alert(message, priority, decision_result, alert_type)
            elif channel == AlertChannel.EMAIL:
                return await self._send_email_alert(message, priority)
            elif channel == AlertChannel.DISCORD:
                return await self._send_discord_alert(message, priority)
            elif channel == AlertChannel.SLACK:
                return await self._send_slack_alert(message, priority)
            else:
                logger.warning(f"Unknown alert channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending to channel {channel}: {e}")
            return False
    
    async def _send_console_alert(self, message: str, priority: AlertPriority) -> bool:
        """Send alert to console (development/testing)."""
        
        if priority == AlertPriority.CRITICAL:
            logger.critical(f"URGENT ALERT:\n{message}")
        elif priority == AlertPriority.URGENT:
            logger.error(f"URGENT ALERT:\n{message}")
        elif priority == AlertPriority.WARNING:
            logger.warning(f"DECISION ALERT:\n{message}")
        else:
            logger.info(f"DECISION NOTIFICATION:\n{message}")
            
        return True
    
    async def _send_file_alert(self, 
                             message: str, 
                             priority: AlertPriority,
                             decision_result: Optional[DecisionResult] = None,
                             alert_type: Optional[str] = None) -> bool:
        """Send alert to file system."""
        
        try:
            alerts_file = Path(self.config.alerts_file)
            
            # Prepare log entry
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            log_entry = {
                "timestamp": timestamp,
                "priority": priority.value,
                "alert_type": alert_type or (decision_result.decision_type.value if decision_result else "system"),
                "message": message.replace('\n', ' | ')  # Single line for log file
            }
            
            # Append to alerts file
            with alerts_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False
    
    async def _send_email_alert(self, message: str, priority: AlertPriority) -> bool:
        """Send alert via email (future implementation)."""
        # TODO: Implement email alerting with SMTP
        logger.debug(f"Email alert would be sent (priority: {priority.value}): {message[:100]}...")
        return True
        
    async def _send_discord_alert(self, message: str, priority: AlertPriority) -> bool:
        """Send alert via Discord webhook (future implementation)."""
        # TODO: Implement Discord webhook integration
        logger.debug(f"Discord alert would be sent (priority: {priority.value}): {message[:100]}...")
        return True
        
    async def _send_slack_alert(self, message: str, priority: AlertPriority) -> bool:
        """Send alert via Slack (future implementation)."""
        # TODO: Implement Slack webhook integration
        logger.debug(f"Slack alert would be sent (priority: {priority.value}): {message[:100]}...")
        return True
    
    def _record_alert(self, 
                     alert_type: str,
                     priority: AlertPriority,
                     message: str,
                     channel: AlertChannel,
                     success: bool,
                     decision_type: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record alert in history for analysis and reporting."""
        
        record = AlertRecord(
            timestamp=datetime.now(timezone.utc),
            alert_type=alert_type,
            priority=priority,
            message=message,
            channel=channel,
            sent_successfully=success,
            decision_type=decision_type,
            metadata=metadata or {}
        )
        
        self.alert_history.append(record)
        
        # Trim history if it gets too long
        if len(self.alert_history) > self.config.max_alert_history:
            self.alert_history = self.alert_history[-self.config.max_alert_history:]
    
    def get_alert_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        if not recent_alerts:
            return {"total_alerts": 0, "period_hours": hours_back}
        
        # Calculate summary statistics
        priority_counts = {}
        alert_type_counts = {}
        success_count = 0
        
        for alert in recent_alerts:
            priority = alert.priority.value
            alert_type = alert.alert_type
            
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
            
            if alert.sent_successfully:
                success_count += 1
        
        return {
            "total_alerts": len(recent_alerts),
            "period_hours": hours_back,
            "priority_breakdown": priority_counts,
            "alert_type_breakdown": alert_type_counts,
            "success_rate": success_count / len(recent_alerts) if recent_alerts else 0,
            "critical_alerts": priority_counts.get("critical", 0),
            "urgent_alerts": priority_counts.get("urgent", 0)
        }
    
    def get_throttling_status(self) -> Dict[str, Any]:
        """Get current throttling status for monitoring."""
        
        current_time = datetime.now(timezone.utc)
        throttled_alerts = {}
        
        for alert_key, last_time in self.last_alert_times.items():
            # Parse alert key to get priority
            parts = alert_key.split('_')
            if len(parts) >= 2:
                priority_str = parts[-1]
                throttle_minutes = self.config.throttle_intervals.get(priority_str, 30)
                
                if throttle_minutes > 0:
                    time_since = (current_time - last_time).total_seconds() / 60
                    time_remaining = max(0, throttle_minutes - time_since)
                    
                    if time_remaining > 0:
                        throttled_alerts[alert_key] = {
                            "last_sent": last_time.isoformat(),
                            "throttle_minutes": throttle_minutes,
                            "time_remaining_minutes": round(time_remaining, 1)
                        }
        
        return {
            "currently_throttled": len(throttled_alerts),
            "throttled_alerts": throttled_alerts,
            "throttle_config": self.config.throttle_intervals
        }


# Factory function for easy integration
def get_alerting_system(config: Optional[AlertConfig] = None) -> AlertingSystem:
    """Factory function to get AlertingSystem instance."""
    return AlertingSystem(config=config)