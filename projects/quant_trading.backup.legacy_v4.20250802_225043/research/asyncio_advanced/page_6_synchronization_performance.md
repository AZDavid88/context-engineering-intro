# AsyncIO Synchronization and Performance Monitoring

**Source**: https://docs.python.org/3/library/asyncio-sync.html & https://docs.python.org/3/library/asyncio-dev.html
**Extraction Method**: Brightdata MCP
**Research Focus**: Synchronization primitives, performance monitoring, debugging patterns for high-frequency trading systems

## Synchronization Primitives Overview

asyncio synchronization primitives are designed to be similar to those of the [`threading`](threading.html#module-threading "threading: Thread-based parallelism.") module with two important caveats:

- **asyncio primitives are not thread-safe**, therefore they should not be used for OS thread synchronization
- **methods of these synchronization primitives do not accept the _timeout_ argument**; use the [`asyncio.wait_for()`](asyncio-task.html#asyncio.wait_for "asyncio.wait_for") function to perform operations with timeouts

## Core Synchronization Primitives

### Lock

```python
class asyncio.Lock
```

Implements a mutex lock for asyncio tasks. **Not thread-safe.**

**Preferred Usage Pattern**:

```python
lock = asyncio.Lock()

# ... later
async with lock:
    # access shared state
```

**Equivalent to**:

```python
lock = asyncio.Lock()

# ... later
await lock.acquire()
try:
    # access shared state
finally:
    lock.release()
```

**Key Methods**:

```python
async def acquire()
```
Acquire the lock. Waits until the lock is _unlocked_, sets it to _locked_ and returns `True`.

**Fairness**: When multiple coroutines are blocked waiting for the lock, the first coroutine that started waiting will proceed.

```python
def release()
```
Release the lock. Raises [`RuntimeError`](exceptions.html#RuntimeError "RuntimeError") if called on an unlocked lock.

```python
def locked()
```
Return `True` if the lock is _locked_.

### Event

```python
class asyncio.Event
```

An event object for notifying multiple asyncio tasks that some event has happened. **Not thread-safe.**

**Usage Example**:

```python
async def waiter(event):
    print('waiting for it ...')
    await event.wait()
    print('... got it!')

async def main():
    # Create an Event object.
    event = asyncio.Event()

    # Spawn a Task to wait until 'event' is set.
    waiter_task = asyncio.create_task(waiter(event))

    # Sleep for 1 second and set the event.
    await asyncio.sleep(1)
    event.set()

    # Wait until the waiter task is finished.
    await waiter_task

asyncio.run(main())
```

**Key Methods**:

```python
async def wait()
```
Wait until the event is set. If the event is already set, return `True` immediately.

```python
def set()
```
Set the event. All tasks waiting for event to be set will be immediately awakened.

```python
def clear()
```
Clear (unset) the event. Tasks awaiting on [`wait()`](#asyncio.Event.wait "asyncio.Event.wait") will now block.

```python
def is_set()
```
Return `True` if the event is set.

### Semaphore

```python
class asyncio.Semaphore(value=1)
```

A semaphore manages an internal counter which is decremented by each [`acquire()`](#asyncio.Semaphore.acquire "asyncio.Semaphore.acquire") call and incremented by each [`release()`](#asyncio.Semaphore.release "asyncio.Semaphore.release") call.

**Usage Pattern**:

```python
sem = asyncio.Semaphore(10)

# ... later
async with sem:
    # work with shared resource
```

**Key Methods**:

```python
async def acquire()
```
Acquire a semaphore. If the internal counter is greater than zero, decrement it by one and return `True` immediately. If it is zero, wait until a [`release()`](#asyncio.Semaphore.release "asyncio.Semaphore.release") is called.

```python
def release()
```
Release a semaphore, incrementing the internal counter by one.

```python
def locked()
```
Returns `True` if semaphore cannot be acquired immediately.

### Condition

```python
class asyncio.Condition(lock=None)
```

A Condition object combines the functionality of an [`Event`](#asyncio.Event "asyncio.Event") and a [`Lock`](#asyncio.Lock "asyncio.Lock"). **Not thread-safe.**

**Usage Pattern**:

```python
cond = asyncio.Condition()

# ... later
async with cond:
    await cond.wait()
```

**Key Methods**:

```python
async def wait()
```
Wait until notified. Releases the underlying lock and blocks until awakened by [`notify()`](#asyncio.Condition.notify "asyncio.Condition.notify") or [`notify_all()`](#asyncio.Condition.notify_all "asyncio.Condition.notify_all").

```python
def notify(n=1)
```
Wake up _n_ tasks (1 by default) waiting on this condition.

```python
def notify_all()
```
Wake up all tasks waiting on this condition.

```python
async def wait_for(predicate)
```
Wait until a predicate becomes _true_. The predicate must be a callable which result will be interpreted as a boolean value.

### Barrier

```python
class asyncio.Barrier(parties)
```

A barrier allows blocking until _parties_ number of tasks are waiting on it. **Not thread-safe.**

**Usage Example**:

```python
async def example_barrier():
   # barrier with 3 parties
   b = asyncio.Barrier(3)

   # create 2 new waiting tasks
   asyncio.create_task(b.wait())
   asyncio.create_task(b.wait())

   await asyncio.sleep(0)
   print(b)

   # The third .wait() call passes the barrier
   await b.wait()
   print(b)
   print("barrier passed")

asyncio.run(example_barrier())
```

## Performance Monitoring and Debugging

### Debug Mode

**Enabling Debug Mode**:

- Setting the [`PYTHONASYNCIODEBUG`](../using/cmdline.html#envvar-PYTHONASYNCIODEBUG) environment variable to `1`
- Using the [Python Development Mode](devmode.html#devmode)
- Passing `debug=True` to [`asyncio.run()`](asyncio-runner.html#asyncio.run "asyncio.run")
- Calling [`loop.set_debug()`](asyncio-eventloop.html#asyncio.loop.set_debug "asyncio.loop.set_debug")

**Debug Mode Features**:

- Many non-threadsafe asyncio APIs raise an exception if called from a wrong thread
- The execution time of the I/O selector is logged if it takes too long to perform an I/O operation
- Callbacks taking longer than 100 milliseconds are logged
- The [`loop.slow_callback_duration`](asyncio-eventloop.html#asyncio.loop.slow_callback_duration "asyncio.loop.slow_callback_duration") attribute can be used to set the minimum execution duration considered "slow"

### Logging Configuration

```python
import logging

# Set asyncio logger level
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Enable debug logging for development
logging.basicConfig(level=logging.DEBUG)
```

**Important**: Network logging can block the event loop. Use a separate thread for handling logs or use non-blocking IO.

### Concurrency and Multithreading Guidelines

**Key Principles**:

- An event loop runs in a thread (typically the main thread) and executes all callbacks and Tasks in its thread
- While a Task is running in the event loop, no other Tasks can run in the same thread
- When a Task executes an `await` expression, the running Task gets suspended, and the event loop executes the next Task

**Thread-Safe Operations**:

```python
# Schedule a callback from another OS thread
loop.call_soon_threadsafe(callback, *args)

# Schedule a coroutine from a different OS thread
future = asyncio.run_coroutine_threadsafe(coro_func(), loop)
result = future.result()
```

## Trading System Implementation Patterns

### Thread-Safe Market Data Coordinator

```python
import asyncio
import threading
import logging
from typing import Dict, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MarketDataUpdate:
    symbol: str
    price: float
    volume: float
    timestamp: float
    source: str

class TradingSystemCoordinator:
    def __init__(self, max_workers: int = 4):
        self.event_loop = None
        self.coordinator_thread = None
        self.max_workers = max_workers
        
        # Synchronization primitives
        self.data_lock = None
        self.processing_semaphore = None
        self.shutdown_event = None
        self.new_data_condition = None
        
        # Data structures
        self.market_data_buffer = []
        self.active_subscriptions = set()
        self.processing_stats = defaultdict(int)
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Logging
        self.logger = logging.getLogger("trading_coordinator")

    async def initialize(self):
        """Initialize all synchronization primitives"""
        self.data_lock = asyncio.Lock()
        self.processing_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent processors
        self.shutdown_event = asyncio.Event()
        self.new_data_condition = asyncio.Condition()
        
        self.logger.info("🚀 Trading system coordinator initialized")

    async def start_market_data_processing(self):
        """Start the main market data processing pipeline"""
        try:
            # Create task group for coordinated execution
            async with asyncio.TaskGroup() as tg:
                # Data ingestion tasks
                tg.create_task(self.websocket_data_ingester(), name="websocket_ingester")
                tg.create_task(self.rest_api_poller(), name="rest_poller")
                
                # Data processing workers
                for i in range(3):
                    tg.create_task(
                        self.market_data_processor(f"processor_{i}"),
                        name=f"processor_{i}"
                    )
                
                # Monitoring and coordination
                tg.create_task(self.performance_monitor(), name="performance_monitor")
                tg.create_task(self.coordination_manager(), name="coordinator")
                
                # Wait for shutdown signal
                await self.shutdown_event.wait()
                
        except Exception as e:
            self.logger.error(f"❌ Processing pipeline failed: {e}")
            raise

    async def websocket_data_ingester(self):
        """Ingest data from WebSocket with proper synchronization"""
        try:
            while not self.shutdown_event.is_set():
                # Simulate WebSocket data reception
                market_update = await self.receive_websocket_data()
                
                if market_update:
                    # Thread-safe data buffering
                    async with self.data_lock:
                        self.market_data_buffer.append(market_update)
                        
                        # Limit buffer size to prevent memory overflow
                        if len(self.market_data_buffer) > 10000:
                            # Remove oldest data
                            self.market_data_buffer = self.market_data_buffer[-5000:]
                            self.logger.warning("📊 Market data buffer trimmed")
                    
                    # Notify processors of new data
                    async with self.new_data_condition:
                        self.new_data_condition.notify_all()
                
                await asyncio.sleep(0.001)  # Yield control
                
        except asyncio.CancelledError:
            self.logger.info("📊 WebSocket ingester cancelled")
            raise
        except Exception as e:
            self.logger.error(f"❌ WebSocket ingester error: {e}")
            raise

    async def market_data_processor(self, processor_name: str):
        """Process market data with semaphore-controlled concurrency"""
        try:
            while not self.shutdown_event.is_set():
                # Wait for new data with timeout
                try:
                    async with asyncio.timeout(5):
                        async with self.new_data_condition:
                            await self.new_data_condition.wait()
                except asyncio.TimeoutError:
                    continue  # Check shutdown and continue
                
                # Process available data
                batch_to_process = []
                
                # Get batch of data to process
                async with self.data_lock:
                    if self.market_data_buffer:
                        batch_size = min(100, len(self.market_data_buffer))
                        batch_to_process = self.market_data_buffer[:batch_size]
                        self.market_data_buffer = self.market_data_buffer[batch_size:]
                
                if batch_to_process:
                    # Process with semaphore control
                    async with self.processing_semaphore:
                        await self.process_market_data_batch(
                            processor_name,
                            batch_to_process
                        )
                
        except asyncio.CancelledError:
            self.logger.info(f"📊 {processor_name} cancelled")
            raise
        except Exception as e:
            self.logger.error(f"❌ {processor_name} error: {e}")
            raise

    async def process_market_data_batch(self, 
                                      processor_name: str, 
                                      batch: List[MarketDataUpdate]):
        """Process a batch of market data updates"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Simulate CPU-intensive processing in thread pool
            processing_tasks = []
            
            for update in batch:
                # Offload CPU-intensive work to thread pool
                task = asyncio.create_task(
                    asyncio.to_thread(self.calculate_technical_indicators, update)
                )
                processing_tasks.append(task)
            
            # Wait for all processing to complete
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Handle results and errors
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(
                        f"⚠️  Processing failed for {batch[i].symbol}: {result}"
                    )
                else:
                    successful_results.append(result)
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.processing_stats[processor_name] += len(successful_results)
            
            self.logger.debug(
                f"📊 {processor_name}: Processed {len(successful_results)}/{len(batch)} "
                f"updates in {processing_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing error in {processor_name}: {e}")
            raise

    def calculate_technical_indicators(self, update: MarketDataUpdate) -> Dict[str, float]:
        """CPU-intensive calculation that runs in thread pool"""
        import time
        
        # Simulate complex calculations
        time.sleep(0.001)  # Simulate 1ms of CPU work
        
        return {
            'symbol': update.symbol,
            'sma_20': update.price * 0.98,  # Simplified calculation
            'rsi': 50.0,
            'bollinger_upper': update.price * 1.02,
            'bollinger_lower': update.price * 0.98
        }

    async def coordination_manager(self):
        """Coordinate between different system components"""
        try:
            while not self.shutdown_event.is_set():
                # Perform coordination tasks
                await self.check_system_health()
                await self.balance_processing_load()
                await self.cleanup_old_data()
                
                # Run coordination every 10 seconds
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            self.logger.info("📊 Coordination manager cancelled")
            raise

    async def check_system_health(self):
        """Monitor system health and performance"""
        try:
            # Check buffer sizes
            async with self.data_lock:
                buffer_size = len(self.market_data_buffer)
            
            if buffer_size > 5000:
                self.logger.warning(f"⚠️  High buffer usage: {buffer_size} items")
            
            # Check processing statistics
            total_processed = sum(self.processing_stats.values())
            self.logger.info(f"📊 Total processed: {total_processed}")
            
            # Check semaphore availability
            available_permits = self.processing_semaphore._value
            if available_permits < 2:
                self.logger.warning(f"⚠️  Low processing capacity: {available_permits} permits")
            
        except Exception as e:
            self.logger.error(f"❌ Health check error: {e}")

    async def balance_processing_load(self):
        """Balance processing load across workers"""
        # Implementation would adjust worker priorities, 
        # semaphore values, or spawn additional workers
        pass

    async def cleanup_old_data(self):
        """Clean up old data and statistics"""
        try:
            # Reset statistics periodically
            current_time = asyncio.get_event_loop().time()
            
            # Reset stats every hour
            if int(current_time) % 3600 == 0:
                self.processing_stats.clear()
                self.logger.info("📊 Processing statistics reset")
                
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")

    async def performance_monitor(self):
        """Monitor performance metrics and resource usage"""
        try:
            while not self.shutdown_event.is_set():
                # Collect performance metrics
                metrics = await self.collect_performance_metrics()
                
                # Log performance summary
                self.logger.info(
                    f"📊 Performance: Buffer={metrics['buffer_size']}, "
                    f"Semaphore={metrics['semaphore_available']}, "
                    f"Processing Rate={metrics['processing_rate']:.1f}/s"
                )
                
                # Check for performance issues
                if metrics['buffer_size'] > 8000:
                    self.logger.warning("🚨 Buffer overflow risk detected")
                
                if metrics['processing_rate'] < 100:
                    self.logger.warning("🚨 Low processing rate detected")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except asyncio.CancelledError:
            self.logger.info("📊 Performance monitor cancelled")
            raise

    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        async with self.data_lock:
            buffer_size = len(self.market_data_buffer)
        
        semaphore_available = self.processing_semaphore._value
        total_processed = sum(self.processing_stats.values())
        
        # Calculate processing rate (simplified)
        processing_rate = total_processed / max(1, asyncio.get_event_loop().time())
        
        return {
            'buffer_size': buffer_size,
            'semaphore_available': semaphore_available,
            'total_processed': total_processed,
            'processing_rate': processing_rate,
            'active_tasks': len([t for t in asyncio.all_tasks() if not t.done()])
        }

    async def receive_websocket_data(self) -> MarketDataUpdate:
        """Simulate receiving WebSocket data"""
        await asyncio.sleep(0.01)  # Simulate network delay
        
        import random
        return MarketDataUpdate(
            symbol=random.choice(['BTC', 'ETH', 'SOL']),
            price=random.uniform(1000, 50000),
            volume=random.uniform(1, 1000),
            timestamp=asyncio.get_event_loop().time(),
            source='websocket'
        )

    async def graceful_shutdown(self):
        """Perform graceful shutdown of the coordinator"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        # Signal shutdown to all components
        self.shutdown_event.set()
        
        # Notify all waiting processors
        async with self.new_data_condition:
            self.new_data_condition.notify_all()
        
        # Wait for current processing to complete
        await asyncio.sleep(2)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("✅ Graceful shutdown completed")

# Usage Example
async def main():
    coordinator = TradingSystemCoordinator(max_workers=4)
    
    try:
        await coordinator.initialize()
        
        # Start processing
        processing_task = asyncio.create_task(
            coordinator.start_market_data_processing()
        )
        
        # Run for 60 seconds
        await asyncio.sleep(60)
        
        # Graceful shutdown
        await coordinator.graceful_shutdown()
        
    except KeyboardInterrupt:
        print("📊 Interrupted by user")
        await coordinator.graceful_shutdown()

# Enable debug mode for development
asyncio.run(main(), debug=True)
```

## Key Benefits for Trading Systems

1. **Coordinated Execution**: Synchronization primitives enable proper coordination between multiple trading components
2. **Resource Management**: Semaphores prevent resource exhaustion during high-frequency processing
3. **Thread Safety**: Proper synchronization patterns for mixing asyncio with thread pools
4. **Performance Monitoring**: Built-in debugging and logging capabilities for performance optimization
5. **Graceful Coordination**: Events and conditions enable clean system state management
6. **Scalable Architecture**: Barrier synchronization for coordinated multi-component operations

This synchronization framework is essential for building robust, high-performance quantitative trading systems that can coordinate multiple components while maintaining thread safety and optimal resource utilization.