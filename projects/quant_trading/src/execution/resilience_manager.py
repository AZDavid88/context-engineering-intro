"""
Resilience Manager - Advanced Failure Recovery and Circuit Breaker System

Provides comprehensive resilience patterns for the ultra-compressed evolution system,
enhancing the existing AsyncResourceManager with advanced failure detection,
circuit breakers, retry logic, and disaster recovery capabilities.

Integration Architecture:
- Enhances AsyncResourceManager from TradingSystemManager (existing)
- Integrates with GeneticRiskManager for risk-based failure assessment
- Works with AutomatedDecisionEngine for intelligent recovery decisions
- Coordinates with AlertingSystem for failure notifications
- Monitors system health through existing monitoring infrastructure

This resilience manager follows established patterns while adding production-grade
failure handling capabilities for distributed genetic algorithm operations.
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import defaultdict, deque
from functools import wraps

# Verified imports from architecture analysis
from src.config.settings import get_settings, Settings
from src.execution.trading_system_manager import AsyncResourceManager, SessionHealth, SessionStatus
from src.execution.risk_management import GeneticRiskManager, RiskLevel
from src.execution.alerting_system import AlertingSystem, AlertPriority
from src.execution.automated_decision_engine import AutomatedDecisionEngine, DecisionType, DecisionContext

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Types of failures that can occur in the system."""
    NETWORK_FAILURE = "network_failure"
    COMPUTATION_FAILURE = "computation_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT_FAILURE = "timeout_failure"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_FAILURE = "rate_limit_failure"
    SYSTEM_OVERLOAD = "system_overload"
    EXTERNAL_API_FAILURE = "external_api_failure"
    GENETIC_EVOLUTION_FAILURE = "genetic_evolution_failure"
    VALIDATION_FAILURE = "validation_failure"
    DEPLOYMENT_FAILURE = "deployment_failure"


class ResilienceState(str, Enum):
    """Overall system resilience state."""
    HEALTHY = "healthy"                    # All systems operating normally
    DEGRADED = "degraded"                 # Some failures but system functional
    CRITICAL = "critical"                 # Multiple failures, limited functionality
    RECOVERY_MODE = "recovery_mode"       # Actively recovering from failures
    EMERGENCY_SHUTDOWN = "emergency_shutdown"  # System shut down for safety


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit breaker triggered, failing fast
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_exceptions: Tuple[type, ...] = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout_seconds: int = 60  # Time before attempting recovery
    success_threshold: int = 3          # Successes needed to close circuit
    monitoring_window_seconds: int = 300 # Window for failure rate calculation


@dataclass
class FailureRecord:
    """Record of a system failure."""
    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_type: FailureType = FailureType.SYSTEM_OVERLOAD
    component: str = ""
    error_message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    impact_score: float = 0.0  # 0.0 = no impact, 1.0 = critical impact
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResilienceMetrics:
    """System resilience metrics."""
    total_failures: int = 0
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    mean_time_to_recovery: float = 0.0
    system_availability: float = 1.0
    circuit_breaker_trips: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    current_resilience_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_failures": self.total_failures,
            "failures_by_type": dict(self.failures_by_type),
            "mean_time_to_recovery": self.mean_time_to_recovery,
            "system_availability": self.system_availability,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "resilience_score": self.current_resilience_score
        }


class CircuitBreaker:
    """Circuit breaker implementation for service protection."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration parameters
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.failure_history: deque = deque(maxlen=100)
        
        logger.debug(f"Circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")
        
        # Record success in history
        self.failure_history.append({
            "timestamp": datetime.now(timezone.utc),
            "success": True
        })
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        self.success_count = 0
        
        # Record failure in history
        self.failure_history.append({
            "timestamp": datetime.now(timezone.utc),
            "success": False
        })
        
        # Check if should open circuit
        if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout_seconds
    
    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self.failure_history:
            return 0.0
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.config.monitoring_window_seconds)
        recent_events = [
            event for event in self.failure_history
            if event["timestamp"] >= cutoff_time
        ]
        
        if not recent_events:
            return 0.0
        
        failures = len([e for e in recent_events if not e["success"]])
        return failures / len(recent_events)


class ResilienceManager:
    """Comprehensive failure recovery and resilience management system."""
    
    def __init__(self, 
                 settings: Optional[Settings] = None,
                 resource_manager: Optional[AsyncResourceManager] = None):
        """
        Initialize resilience manager.
        
        Args:
            settings: System settings
            resource_manager: Existing resource manager to enhance
        """
        
        self.settings = settings or get_settings()
        
        # Integration with existing systems (verified available)
        self.resource_manager = resource_manager or AsyncResourceManager("ResilienceManager", logger)
        self.risk_manager = GeneticRiskManager()
        self.alerting = AlertingSystem()
        self.decision_engine = AutomatedDecisionEngine()
        
        # Resilience state management
        self.current_state = ResilienceState.HEALTHY
        self.failure_history: List[FailureRecord] = []
        self.recovery_operations: Dict[str, Callable] = {}
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()
        
        # Retry configurations
        self.retry_configs: Dict[FailureType, RetryConfig] = self._initialize_retry_configs()
        
        # Metrics and monitoring
        self.metrics = ResilienceMetrics()
        self.health_check_functions: Dict[str, Callable] = {}
        
        # Recovery coordination
        self.active_recoveries: Set[str] = set()
        self.recovery_lock = asyncio.Lock()
        
        logger.info(f"ResilienceManager initialized with {len(self.circuit_breakers)} circuit breakers")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical components."""
        
        # Circuit breakers for different system components
        circuit_configs = {
            "genetic_evolution": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=120,
                success_threshold=2
            ),
            "validation_pipeline": CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout_seconds=60,
                success_threshold=3
            ),
            "strategy_deployment": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=90,
                success_threshold=2
            ),
            "ray_cluster": CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout_seconds=180,
                success_threshold=1
            ),
            "external_apis": CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout_seconds=30,
                success_threshold=5
            ),
            "database_operations": CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout_seconds=45,
                success_threshold=2
            )
        }
        
        for name, config in circuit_configs.items():
            self.circuit_breakers[name] = CircuitBreaker(name, config)
    
    def _initialize_retry_configs(self) -> Dict[FailureType, RetryConfig]:
        """Initialize retry configurations for different failure types."""
        
        return {
            FailureType.NETWORK_FAILURE: RetryConfig(
                max_attempts=5,
                base_delay_seconds=2.0,
                max_delay_seconds=30.0,
                exponential_backoff=True
            ),
            FailureType.COMPUTATION_FAILURE: RetryConfig(
                max_attempts=3,
                base_delay_seconds=1.0,
                max_delay_seconds=10.0,
                exponential_backoff=True
            ),
            FailureType.RESOURCE_EXHAUSTION: RetryConfig(
                max_attempts=2,
                base_delay_seconds=5.0,
                max_delay_seconds=60.0,
                exponential_backoff=True
            ),
            FailureType.TIMEOUT_FAILURE: RetryConfig(
                max_attempts=3,
                base_delay_seconds=1.0,
                max_delay_seconds=15.0,
                exponential_backoff=True
            ),
            FailureType.RATE_LIMIT_FAILURE: RetryConfig(
                max_attempts=10,
                base_delay_seconds=10.0,
                max_delay_seconds=300.0,
                exponential_backoff=True
            ),
            FailureType.EXTERNAL_API_FAILURE: RetryConfig(
                max_attempts=5,
                base_delay_seconds=3.0,
                max_delay_seconds=60.0,
                exponential_backoff=True
            )
        }
    
    async def execute_with_resilience(self, 
                                    operation: Callable,
                                    operation_name: str,
                                    failure_type: FailureType = FailureType.SYSTEM_OVERLOAD,
                                    circuit_breaker_name: Optional[str] = None,
                                    timeout_seconds: Optional[float] = None) -> Tuple[Any, bool]:
        """
        Execute operation with comprehensive resilience patterns.
        
        Args:
            operation: Operation to execute
            operation_name: Human-readable operation name
            failure_type: Expected failure type for retry configuration
            circuit_breaker_name: Circuit breaker to use (if any)
            timeout_seconds: Operation timeout
            
        Returns:
            Tuple of (result, success_flag)
        """
        
        start_time = time.time()
        retry_config = self.retry_configs.get(failure_type, RetryConfig())
        
        logger.debug(f"🔄 Executing '{operation_name}' with resilience patterns")
        
        for attempt in range(retry_config.max_attempts):
            try:
                # Execute through circuit breaker if specified
                if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                    
                    if timeout_seconds:
                        result = await asyncio.wait_for(
                            circuit_breaker.call(operation),
                            timeout=timeout_seconds
                        )
                    else:
                        result = await circuit_breaker.call(operation)
                else:
                    # Execute directly with timeout
                    if timeout_seconds:
                        if asyncio.iscoroutinefunction(operation):
                            result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
                        else:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(operation),
                                timeout=timeout_seconds
                            )
                    else:
                        result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
                
                # Success - update metrics and return
                execution_time = time.time() - start_time
                logger.debug(f"✅ '{operation_name}' completed successfully in {execution_time:.2f}s (attempt {attempt + 1})")
                
                return result, True
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.warning(f"❌ '{operation_name}' failed on attempt {attempt + 1}/{retry_config.max_attempts}: {e}")
                
                # Record failure
                await self._record_failure(
                    failure_type=failure_type,
                    component=operation_name,
                    error_message=str(e),
                    stack_trace=self._get_stack_trace()
                )
                
                # Check if this is the last attempt
                if attempt == retry_config.max_attempts - 1:
                    logger.error(f"❌ '{operation_name}' failed after {retry_config.max_attempts} attempts")
                    return None, False
                
                # Calculate retry delay
                delay = self._calculate_retry_delay(retry_config, attempt)
                logger.debug(f"⏳ Retrying '{operation_name}' in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        return None, False
    
    async def _record_failure(self, 
                            failure_type: FailureType,
                            component: str,
                            error_message: str,
                            stack_trace: Optional[str] = None):
        """Record system failure for analysis and recovery."""
        
        failure_record = FailureRecord(
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            stack_trace=stack_trace,
            impact_score=self._calculate_impact_score(failure_type, component)
        )
        
        self.failure_history.append(failure_record)
        
        # Update metrics
        self.metrics.total_failures += 1
        self.metrics.failures_by_type[failure_type.value] = self.metrics.failures_by_type.get(failure_type.value, 0) + 1
        
        # Update system state based on failure impact
        await self._update_resilience_state(failure_record)
        
        # Send failure alert if significant
        if failure_record.impact_score > 0.5:  # Significant impact
            await self.alerting.send_system_alert(
                alert_type="system_failure",
                message=f"{failure_type.value} in {component}: {error_message}",
                priority=AlertPriority.URGENT if failure_record.impact_score > 0.8 else AlertPriority.WARNING,
                metadata={
                    "failure_id": failure_record.failure_id,
                    "failure_type": failure_type.value,
                    "component": component,
                    "impact_score": failure_record.impact_score
                }
            )
        
        logger.info(f"📊 Recorded failure: {failure_type.value} in {component} (impact: {failure_record.impact_score:.2f})")
    
    def _calculate_impact_score(self, failure_type: FailureType, component: str) -> float:
        """Calculate impact score for a failure (0.0 = no impact, 1.0 = critical)."""
        
        # Base impact scores by failure type
        type_impacts = {
            FailureType.NETWORK_FAILURE: 0.6,
            FailureType.COMPUTATION_FAILURE: 0.4,
            FailureType.RESOURCE_EXHAUSTION: 0.8,
            FailureType.DATA_CORRUPTION: 0.9,
            FailureType.TIMEOUT_FAILURE: 0.3,
            FailureType.AUTHENTICATION_FAILURE: 0.7,
            FailureType.RATE_LIMIT_FAILURE: 0.2,
            FailureType.SYSTEM_OVERLOAD: 0.8,
            FailureType.EXTERNAL_API_FAILURE: 0.4,
            FailureType.GENETIC_EVOLUTION_FAILURE: 0.6,
            FailureType.VALIDATION_FAILURE: 0.5,
            FailureType.DEPLOYMENT_FAILURE: 0.7
        }
        
        base_impact = type_impacts.get(failure_type, 0.5)
        
        # Adjust based on component criticality
        if "ray" in component.lower() or "cluster" in component.lower():
            base_impact *= 1.2  # Distributed computing is critical
        elif "database" in component.lower():
            base_impact *= 1.1  # Database failures are serious
        elif "deployment" in component.lower():
            base_impact *= 0.9  # Deployment failures are less critical
        
        return min(1.0, base_impact)
    
    async def _update_resilience_state(self, failure_record: FailureRecord):
        """Update overall system resilience state based on recent failures."""
        
        # Analyze recent failures (last hour)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp >= recent_cutoff
        ]
        
        if not recent_failures:
            self.current_state = ResilienceState.HEALTHY
            return
        
        # Calculate recent failure metrics
        recent_count = len(recent_failures)
        avg_impact = sum(f.impact_score for f in recent_failures) / recent_count
        critical_failures = len([f for f in recent_failures if f.impact_score > 0.8])
        
        # Determine resilience state
        old_state = self.current_state
        
        if critical_failures >= 3 or recent_count >= 10:
            self.current_state = ResilienceState.EMERGENCY_SHUTDOWN
        elif critical_failures >= 2 or recent_count >= 8 or avg_impact > 0.7:
            self.current_state = ResilienceState.CRITICAL
        elif critical_failures >= 1 or recent_count >= 5 or avg_impact > 0.4:
            self.current_state = ResilienceState.DEGRADED
        else:
            self.current_state = ResilienceState.HEALTHY
        
        # Alert on state changes
        if old_state != self.current_state:
            priority = AlertPriority.CRITICAL if self.current_state == ResilienceState.EMERGENCY_SHUTDOWN else AlertPriority.WARNING
            
            await self.alerting.send_system_alert(
                alert_type="resilience_state_change",
                message=f"System resilience state changed: {old_state.value} → {self.current_state.value}",
                priority=priority,
                metadata={
                    "old_state": old_state.value,
                    "new_state": self.current_state.value,
                    "recent_failures": recent_count,
                    "critical_failures": critical_failures,
                    "average_impact": avg_impact
                }
            )
            
            logger.warning(f"🔄 Resilience state change: {old_state.value} → {self.current_state.value}")
    
    def _calculate_retry_delay(self, retry_config: RetryConfig, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        
        if retry_config.exponential_backoff:
            delay = retry_config.base_delay_seconds * (2 ** attempt)
        else:
            delay = retry_config.base_delay_seconds
        
        # Apply maximum delay
        delay = min(delay, retry_config.max_delay_seconds)
        
        # Add jitter if configured
        if retry_config.jitter:
            import random
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        return delay
    
    def _get_stack_trace(self) -> Optional[str]:
        """Get current stack trace for error context."""
        
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return None
    
    async def register_recovery_operation(self, 
                                        failure_type: FailureType,
                                        recovery_func: Callable,
                                        component: str = ""):
        """Register a recovery operation for a specific failure type."""
        
        recovery_key = f"{failure_type.value}:{component}" if component else failure_type.value
        self.recovery_operations[recovery_key] = recovery_func
        
        logger.info(f"🔧 Registered recovery operation for {recovery_key}")
    
    async def trigger_recovery(self, failure_record: FailureRecord) -> bool:
        """Attempt automatic recovery for a specific failure."""
        
        if failure_record.failure_id in self.active_recoveries:
            logger.debug(f"Recovery already in progress for failure {failure_record.failure_id}")
            return False
        
        async with self.recovery_lock:
            self.active_recoveries.add(failure_record.failure_id)
            
            try:
                logger.info(f"🔧 Attempting recovery for failure {failure_record.failure_id}")
                failure_record.recovery_attempts += 1
                
                # Find appropriate recovery operation
                recovery_keys = [
                    f"{failure_record.failure_type.value}:{failure_record.component}",
                    failure_record.failure_type.value
                ]
                
                recovery_func = None
                for key in recovery_keys:
                    if key in self.recovery_operations:
                        recovery_func = self.recovery_operations[key]
                        break
                
                if not recovery_func:
                    logger.warning(f"⚠️ No recovery operation found for {failure_record.failure_type.value}")
                    return False
                
                # Execute recovery operation
                recovery_result = await recovery_func(failure_record)
                
                if recovery_result:
                    failure_record.resolved = True
                    failure_record.resolution_time = datetime.now(timezone.utc)
                    self.metrics.successful_recoveries += 1
                    
                    logger.info(f"✅ Successfully recovered from failure {failure_record.failure_id}")
                    
                    await self.alerting.send_system_alert(
                        alert_type="recovery_success",
                        message=f"Successfully recovered from {failure_record.failure_type.value} in {failure_record.component}",
                        priority=AlertPriority.INFORMATIONAL,
                        metadata={
                            "failure_id": failure_record.failure_id,
                            "recovery_attempts": failure_record.recovery_attempts
                        }
                    )
                    
                    return True
                else:
                    self.metrics.failed_recoveries += 1
                    logger.error(f"❌ Recovery failed for failure {failure_record.failure_id}")
                    return False
            
            finally:
                self.active_recoveries.discard(failure_record.failure_id)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        
        logger.info("🏥 Performing system health check")
        
        health_results = {}
        overall_healthy = True
        
        # Check circuit breakers
        circuit_health = {}
        for name, breaker in self.circuit_breakers.items():
            failure_rate = breaker.get_failure_rate()
            is_healthy = breaker.state == CircuitState.CLOSED and failure_rate < 0.1
            
            circuit_health[name] = {
                "state": breaker.state.value,
                "failure_rate": failure_rate,
                "healthy": is_healthy
            }
            
            if not is_healthy:
                overall_healthy = False
        
        health_results["circuit_breakers"] = circuit_health
        
        # Check recent failures
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_failures = [f for f in self.failure_history if f.timestamp >= recent_cutoff]
        
        health_results["recent_failures"] = {
            "count": len(recent_failures),
            "critical_count": len([f for f in recent_failures if f.impact_score > 0.8]),
            "unresolved_count": len([f for f in recent_failures if not f.resolved])
        }
        
        # Check system state
        health_results["resilience_state"] = {
            "current_state": self.current_state.value,
            "healthy": self.current_state == ResilienceState.HEALTHY,
            "active_recoveries": len(self.active_recoveries)
        }
        
        # Execute custom health checks
        custom_health = {}
        for check_name, check_func in self.health_check_functions.items():
            try:
                check_result = await check_func()
                custom_health[check_name] = check_result
                if not check_result.get("healthy", True):
                    overall_healthy = False
            except Exception as e:
                custom_health[check_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        health_results["custom_checks"] = custom_health
        health_results["overall_healthy"] = overall_healthy
        health_results["check_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Update resilience metrics
        self._update_resilience_metrics(health_results)
        
        logger.info(f"🏥 Health check complete: {'✅ HEALTHY' if overall_healthy else '⚠️ ISSUES DETECTED'}")
        
        return health_results
    
    def _update_resilience_metrics(self, health_results: Dict[str, Any]):
        """Update resilience metrics based on health check results."""
        
        # Update availability score
        if health_results["overall_healthy"]:
            self.metrics.system_availability = min(1.0, self.metrics.system_availability + 0.01)
        else:
            self.metrics.system_availability = max(0.0, self.metrics.system_availability - 0.05)
        
        # Update resilience score
        recent_failure_impact = 1.0 - (health_results["recent_failures"]["count"] * 0.1)
        circuit_health_score = sum(
            1.0 if cb["healthy"] else 0.0
            for cb in health_results["circuit_breakers"].values()
        ) / len(health_results["circuit_breakers"])
        
        self.metrics.current_resilience_score = (
            self.metrics.system_availability * 0.4 +
            max(0.0, recent_failure_impact) * 0.3 +
            circuit_health_score * 0.3
        )
        
        # Update mean time to recovery
        resolved_failures = [f for f in self.failure_history if f.resolved and f.resolution_time]
        if resolved_failures:
            recovery_times = [
                (f.resolution_time - f.timestamp).total_seconds()
                for f in resolved_failures
            ]
            self.metrics.mean_time_to_recovery = statistics.mean(recovery_times)
        
        # Count circuit breaker trips
        self.metrics.circuit_breaker_trips = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state != CircuitState.CLOSED
        )
    
    async def register_health_check(self, name: str, check_func: Callable):
        """Register custom health check function."""
        
        self.health_check_functions[name] = check_func
        logger.info(f"🏥 Registered health check: {name}")
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get current resilience metrics."""
        
        return self.metrics.to_dict()
    
    def get_failure_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get failure summary for specified time period."""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        relevant_failures = [
            f for f in self.failure_history
            if f.timestamp >= cutoff_time
        ]
        
        if not relevant_failures:
            return {
                "period_hours": hours_back,
                "total_failures": 0,
                "failures_by_type": {},
                "average_impact": 0.0,
                "resolution_rate": 0.0
            }
        
        failures_by_type = defaultdict(int)
        for failure in relevant_failures:
            failures_by_type[failure.failure_type.value] += 1
        
        resolved_count = len([f for f in relevant_failures if f.resolved])
        average_impact = sum(f.impact_score for f in relevant_failures) / len(relevant_failures)
        
        return {
            "period_hours": hours_back,
            "total_failures": len(relevant_failures),
            "failures_by_type": dict(failures_by_type),
            "resolved_failures": resolved_count,
            "unresolved_failures": len(relevant_failures) - resolved_count,
            "resolution_rate": resolved_count / len(relevant_failures),
            "average_impact": average_impact,
            "critical_failures": len([f for f in relevant_failures if f.impact_score > 0.8])
        }
    
    async def emergency_shutdown(self, reason: str = "System resilience emergency"):
        """Perform emergency system shutdown."""
        
        logger.critical(f"🚨 EMERGENCY SHUTDOWN INITIATED: {reason}")
        
        self.current_state = ResilienceState.EMERGENCY_SHUTDOWN
        
        # Send critical alert
        await self.alerting.send_system_alert(
            alert_type="emergency_shutdown",
            message=f"EMERGENCY SHUTDOWN: {reason}",
            priority=AlertPriority.CRITICAL,
            metadata={
                "shutdown_reason": reason,
                "resilience_state": self.current_state.value,
                "active_recoveries": len(self.active_recoveries)
            }
        )
        
        # Use AutomatedDecisionEngine to coordinate emergency shutdown
        emergency_context = DecisionContext(
            daily_pnl_percentage=-0.15,  # Simulate emergency conditions
            current_drawdown=0.25,
            system_uptime_hours=0.0      # System going down
        )
        
        shutdown_decision = await self.decision_engine.make_decision(
            DecisionType.EMERGENCY_SHUTDOWN,
            emergency_context
        )
        
        logger.critical(f"🚨 Emergency shutdown decision: {shutdown_decision.decision} (confidence: {shutdown_decision.confidence:.3f})")


# Resilience decorators for easy integration
def with_resilience(failure_type: FailureType = FailureType.SYSTEM_OVERLOAD,
                   circuit_breaker: Optional[str] = None,
                   timeout_seconds: Optional[float] = None):
    """Decorator to add resilience patterns to functions."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get resilience manager instance
            manager = ResilienceManager()
            
            # Execute with resilience patterns
            result, success = await manager.execute_with_resilience(
                operation=lambda: func(*args, **kwargs),
                operation_name=func.__name__,
                failure_type=failure_type,
                circuit_breaker_name=circuit_breaker,
                timeout_seconds=timeout_seconds
            )
            
            if not success:
                raise RuntimeError(f"Operation {func.__name__} failed after resilience attempts")
            
            return result
        
        return wrapper
    return decorator


# Factory functions for easy integration
def get_resilience_manager(settings: Optional[Settings] = None,
                         resource_manager: Optional[AsyncResourceManager] = None) -> ResilienceManager:
    """Factory function to get ResilienceManager instance."""
    return ResilienceManager(settings=settings, resource_manager=resource_manager)


if __name__ == "__main__":
    """Test the resilience manager with sample operations."""
    
    async def test_resilience_manager():
        """Test function for development."""
        
        logger.info("🧪 Testing Resilience Manager")
        
        manager = get_resilience_manager()
        logger.info("✅ Resilience manager initialized successfully")
        
        # Test health check
        health_results = await manager.perform_health_check()
        logger.info(f"🏥 Health check results: overall_healthy={health_results['overall_healthy']}")
        
        # Test metrics
        metrics = manager.get_resilience_metrics()
        logger.info(f"📊 Current resilience score: {metrics['resilience_score']:.3f}")
        
        logger.info("✅ Resilience Manager test completed")
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_resilience_manager())