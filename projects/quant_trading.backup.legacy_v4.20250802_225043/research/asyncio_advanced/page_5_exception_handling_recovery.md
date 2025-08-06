# AsyncIO Exception Handling and Error Recovery

**Source**: https://docs.python.org/3/library/asyncio-exceptions.html
**Extraction Method**: Brightdata MCP
**Research Focus**: Exception handling, error recovery patterns, graceful degradation for high-frequency trading systems

## AsyncIO Exception Hierarchy

### Core Exceptions

```python
exception asyncio.TimeoutError
```
A deprecated alias of [`TimeoutError`](exceptions.html#TimeoutError "TimeoutError"), raised when the operation has exceeded the given deadline.

**Changed in version 3.11**: This class was made an alias of [`TimeoutError`](exceptions.html#TimeoutError "TimeoutError").

```python
exception asyncio.CancelledError
```
The operation has been cancelled.

**Critical for Trading Systems**: This exception can be caught to perform custom operations when asyncio Tasks are cancelled. **In almost all situations the exception must be re-raised.**

**Important**: [`CancelledError`](#asyncio.CancelledError "asyncio.CancelledError") is now a subclass of [`BaseException`](exceptions.html#BaseException "BaseException") rather than [`Exception`](exceptions.html#Exception "Exception").

```python
exception asyncio.InvalidStateError
```
Invalid internal state of [`Task`](asyncio-task.html#asyncio.Task "asyncio.Task") or [`Future`](asyncio-future.html#asyncio.Future "asyncio.Future").

Can be raised in situations like setting a result value for a _Future_ object that already has a result value set.

### Network and I/O Exceptions

```python
exception asyncio.SendfileNotAvailableError
```
The "sendfile" syscall is not available for the given socket or file type.

A subclass of [`RuntimeError`](exceptions.html#RuntimeError "RuntimeError").

```python
exception asyncio.IncompleteReadError
```
The requested read operation did not complete fully.

Raised by the [asyncio stream APIs](asyncio-stream.html#asyncio-streams).

**Attributes**:
- `expected`: The total number ([`int`](functions.html#int "int")) of expected bytes
- `partial`: A string of [`bytes`](stdtypes.html#bytes "bytes") read before the end of stream was reached

```python
exception asyncio.LimitOverrunError
```
Reached the buffer size limit while looking for a separator.

Raised by the [asyncio stream APIs](asyncio-stream.html#asyncio-streams).

**Attributes**:
- `consumed`: The total number of to be consumed bytes

## Trading System Exception Handling Patterns

### Comprehensive Exception Handler for Market Data

```python
import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    pass

class ConnectionError(TradingSystemError):
    """Network connection related errors"""
    pass

class DataProcessingError(TradingSystemError):
    """Data processing and validation errors"""
    pass

class RiskManagementError(TradingSystemError):
    """Risk limits and validation errors"""
    pass

@dataclass
class ErrorContext:
    """Context information for error handling"""
    component: str
    operation: str
    timestamp: float
    retry_count: int
    metadata: Dict[str, Any]

class ErrorSeverity(Enum):
    LOW = "low"        # Retry immediately
    MEDIUM = "medium"  # Retry with backoff
    HIGH = "high"      # Requires manual intervention
    CRITICAL = "critical"  # Shutdown system

class TradingSystemExceptionHandler:
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_error_history = 1000
        self.logger = logging.getLogger("trading_system")

    async def handle_exception(self, 
                             exc: Exception, 
                             context: ErrorContext) -> bool:
        """
        Handle exceptions with appropriate recovery strategy
        
        Returns:
            bool: True if operation should be retried, False otherwise
        """
        severity = self._classify_error(exc, context)
        
        # Log error with context
        self._log_error(exc, context, severity)
        
        # Update error statistics
        self._update_error_stats(exc, context)
        
        # Determine recovery strategy
        return await self._execute_recovery_strategy(exc, context, severity)

    def _classify_error(self, exc: Exception, context: ErrorContext) -> ErrorSeverity:
        """Classify error severity based on exception type and context"""
        
        # Network and connection errors
        if isinstance(exc, (ConnectionError, OSError, asyncio.TimeoutError)):
            if context.retry_count < 3:
                return ErrorSeverity.LOW
            elif context.retry_count < 10:
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.HIGH
        
        # Cancellation (usually intentional)
        elif isinstance(exc, asyncio.CancelledError):
            return ErrorSeverity.LOW
        
        # Data processing errors
        elif isinstance(exc, (ValueError, KeyError, TypeError)):
            if "market_data" in context.component:
                return ErrorSeverity.MEDIUM  # Bad data can be skipped
            else:
                return ErrorSeverity.HIGH    # Logic errors are serious
        
        # Stream errors
        elif isinstance(exc, asyncio.IncompleteReadError):
            return ErrorSeverity.MEDIUM
        
        elif isinstance(exc, asyncio.LimitOverrunError):
            return ErrorSeverity.HIGH  # Buffer overflow is serious
        
        # Risk management errors
        elif isinstance(exc, RiskManagementError):
            return ErrorSeverity.CRITICAL
        
        # Unknown errors
        else:
            return ErrorSeverity.HIGH

    async def _execute_recovery_strategy(self, 
                                       exc: Exception, 
                                       context: ErrorContext, 
                                       severity: ErrorSeverity) -> bool:
        """Execute recovery strategy based on error severity"""
        
        if severity == ErrorSeverity.LOW:
            # Immediate retry with minimal delay
            await asyncio.sleep(0.1)
            return True
        
        elif severity == ErrorSeverity.MEDIUM:
            # Exponential backoff retry
            backoff_time = min(2 ** context.retry_count, 30)  # Max 30 seconds
            self.logger.info(f"Backing off for {backoff_time}s before retry")
            await asyncio.sleep(backoff_time)
            return True
        
        elif severity == ErrorSeverity.HIGH:
            # Require manual intervention or circuit breaker
            if context.retry_count < 5:
                await asyncio.sleep(60)  # 1 minute delay
                return True
            else:
                self.logger.critical(f"Too many retries for {context.component}")
                return False
        
        elif severity == ErrorSeverity.CRITICAL:
            # Immediate shutdown
            self.logger.critical(f"Critical error in {context.component}: {exc}")
            await self._trigger_emergency_shutdown()
            return False

    async def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown procedures"""
        self.logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED")
        
        # Close all positions
        # await self.close_all_positions()
        
        # Cancel all orders
        # await self.cancel_all_orders()
        
        # Notify operators
        # await self.send_emergency_alert()

    def _log_error(self, exc: Exception, context: ErrorContext, severity: ErrorSeverity):
        """Log error with appropriate level and context"""
        error_msg = (
            f"{severity.value.upper()} ERROR in {context.component}.{context.operation}: "
            f"{type(exc).__name__}: {exc} (retry #{context.retry_count})"
        )
        
        if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            self.logger.warning(error_msg)
        else:
            self.logger.error(error_msg, exc_info=True)

    def _update_error_stats(self, exc: Exception, context: ErrorContext):
        """Update error statistics for monitoring"""
        error_key = f"{context.component}.{type(exc).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to error history
        self.error_history.append({
            'timestamp': context.timestamp,
            'component': context.component,
            'operation': context.operation,
            'exception': type(exc).__name__,
            'message': str(exc),
            'retry_count': context.retry_count
        })
        
        # Trim history if too long
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]

# Usage in trading components
class RobustMarketDataProcessor:
    def __init__(self):
        self.exception_handler = TradingSystemExceptionHandler()
        self.max_retries = 10

    async def process_websocket_message(self, message: dict) -> bool:
        """Process WebSocket message with robust error handling"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Main processing logic
                await self._process_message_data(message)
                return True
                
            except Exception as exc:
                context = ErrorContext(
                    component="market_data_processor",
                    operation="process_websocket_message",
                    timestamp=asyncio.get_event_loop().time(),
                    retry_count=retry_count,
                    metadata={"message_type": message.get("channel", "unknown")}
                )
                
                should_retry = await self.exception_handler.handle_exception(exc, context)
                
                if not should_retry:
                    return False
                
                retry_count += 1
        
        # Max retries exceeded
        self.exception_handler.logger.error(
            f"Max retries exceeded for message processing: {message}"
        )
        return False

    async def _process_message_data(self, message: dict):
        """Core message processing logic that may raise exceptions"""
        try:
            # Validate message structure
            if not isinstance(message, dict):
                raise DataProcessingError("Invalid message format")
            
            # Check required fields
            if "channel" not in message:
                raise DataProcessingError("Missing channel field")
            
            channel = message["channel"]
            data = message.get("data", {})
            
            # Process based on channel type
            if channel == "allMids":
                await self._process_mid_prices(data)
            elif channel == "l2Book":
                await self._process_order_book(data)
            elif channel == "trades":
                await self._process_trades(data)
            else:
                raise DataProcessingError(f"Unknown channel: {channel}")
                
        except KeyError as e:
            raise DataProcessingError(f"Missing required field: {e}")
        except ValueError as e:
            raise DataProcessingError(f"Invalid data value: {e}")
        except Exception as e:
            # Re-raise as data processing error for proper classification
            raise DataProcessingError(f"Processing failed: {e}") from e

    async def _process_mid_prices(self, data: dict):
        """Process mid price updates with validation"""
        for symbol, price_str in data.items():
            try:
                price = float(price_str)
                if price <= 0:
                    raise ValueError(f"Invalid price for {symbol}: {price}")
                
                # Store price update
                await self._store_price_update(symbol, price)
                
            except ValueError as e:
                raise DataProcessingError(f"Invalid price data for {symbol}: {e}")

    async def _store_price_update(self, symbol: str, price: float):
        """Store price update with error handling"""
        try:
            # Database operation that might fail
            # await self.database.store_price(symbol, price, timestamp)
            pass
        except Exception as e:
            raise DataProcessingError(f"Failed to store price for {symbol}: {e}")
```

### Circuit Breaker Pattern

```python
import asyncio
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    async def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)

    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage with trading components
class RobustTradingComponent:
    def __init__(self):
        # Circuit breakers for different services
        self.websocket_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ConnectionError
        )
        
        self.database_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception
        )

    async def connect_websocket(self):
        """Connect to WebSocket with circuit breaker protection"""
        return await self.websocket_breaker(self._do_websocket_connect)

    async def _do_websocket_connect(self):
        """Actual WebSocket connection logic"""
        # Connection logic that might fail
        pass

    async def store_market_data(self, data):
        """Store data with circuit breaker protection"""
        return await self.database_breaker(self._do_store_data, data)

    async def _do_store_data(self, data):
        """Actual data storage logic"""
        # Storage logic that might fail
        pass
```

### Graceful Degradation Pattern

```python
class GracefulDegradationHandler:
    def __init__(self):
        self.fallback_strategies = {}
        self.service_health = {}

    def register_fallback(self, service: str, fallback_func: Callable):
        """Register fallback strategy for a service"""
        self.fallback_strategies[service] = fallback_func

    async def execute_with_fallback(self, 
                                  service: str, 
                                  primary_func: Callable, 
                                  *args, **kwargs) -> Any:
        """Execute function with graceful fallback"""
        try:
            result = await primary_func(*args, **kwargs)
            self.service_health[service] = True
            return result
            
        except Exception as e:
            self.service_health[service] = False
            
            # Try fallback if available
            if service in self.fallback_strategies:
                print(f"⚠️  Primary {service} failed, using fallback: {e}")
                try:
                    return await self.fallback_strategies[service](*args, **kwargs)
                except Exception as fallback_error:
                    print(f"❌ Fallback for {service} also failed: {fallback_error}")
                    raise
            else:
                print(f"❌ No fallback available for {service}: {e}")
                raise

# Usage example
class TradingSystemWithFallbacks:
    def __init__(self):
        self.degradation_handler = GracefulDegradationHandler()
        
        # Register fallback strategies
        self.degradation_handler.register_fallback(
            "market_data", 
            self._fallback_market_data
        )
        self.degradation_handler.register_fallback(
            "order_execution",
            self._fallback_order_execution
        )

    async def get_market_data(self, symbol: str) -> dict:
        """Get market data with fallback to cached data"""
        return await self.degradation_handler.execute_with_fallback(
            "market_data",
            self._get_live_market_data,
            symbol
        )

    async def _get_live_market_data(self, symbol: str) -> dict:
        """Primary market data source"""
        # Live data retrieval that might fail
        pass

    async def _fallback_market_data(self, symbol: str) -> dict:
        """Fallback to cached/historical data"""
        print(f"📊 Using cached data for {symbol}")
        # Return cached data
        return {"symbol": symbol, "price": 0, "source": "cache"}

    async def _fallback_order_execution(self, order: dict):
        """Fallback order execution strategy"""
        print("⚠️  Using emergency order execution")
        # Emergency order handling
        pass
```

## Best Practices for Trading System Exception Handling

1. **Always Re-raise CancelledError**: Critical for proper task cancellation
2. **Classify Errors by Severity**: Different errors require different recovery strategies
3. **Implement Circuit Breakers**: Prevent cascading failures in distributed systems
4. **Use Graceful Degradation**: Maintain partial functionality when components fail
5. **Monitor Error Patterns**: Track error frequencies and types for system health
6. **Plan Fallback Strategies**: Always have backup plans for critical operations
7. **Log with Context**: Include sufficient context for debugging and monitoring
8. **Test Error Scenarios**: Regularly test error handling and recovery procedures

This comprehensive error handling framework is essential for building robust, fault-tolerant quantitative trading systems that can maintain operations even under adverse conditions.