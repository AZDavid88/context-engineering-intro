# AsyncIO Task Management and Cancellation

**Source**: https://docs.python.org/3/library/asyncio-task.html
**Extraction Method**: Brightdata MCP
**Research Focus**: Task management, exception handling, graceful shutdown patterns for high-frequency trading data processing

## Coroutines and Tasks Overview

This section outlines high-level asyncio APIs to work with coroutines and Tasks essential for trading system orchestration.

## Creating Tasks

### High-Level Task Creation

```python
asyncio.create_task(coro, *, name=None, context=None)
```

Wrap the _coro_ coroutine into a [`Task`](#asyncio.Task "asyncio.Task") and schedule its execution. Return the Task object.

**Key Parameters**:
- `name`: Optional name for debugging and monitoring
- `context`: Custom [`contextvars.Context`](contextvars.html#contextvars.Context "contextvars.Context") for the coroutine

**Important**: Save a reference to the result of this function, to avoid a task disappearing mid-execution. The event loop only keeps weak references to tasks.

### Task Reference Management Pattern

```python
background_tasks = set()

for i in range(10):
    task = asyncio.create_task(some_coro(param=i))

    # Add task to the set. This creates a strong reference.
    background_tasks.add(task)

    # To prevent keeping references to finished tasks forever,
    # make each task remove its own reference from the set after
    # completion:
    task.add_done_callback(background_tasks.discard)
```

## Task Cancellation

**Core Principle**: Tasks can easily and safely be cancelled. When a task is cancelled, [`asyncio.CancelledError`](asyncio-exceptions.html#asyncio.CancelledError "asyncio.CancelledError") will be raised in the task at the next opportunity.

### Cancellation Best Practices

**Recommended Pattern**: Use `try/finally` blocks to robustly perform clean-up logic:

```python
async def cancel_me():
    print('cancel_me(): before sleep')

    try:
        # Wait for 1 hour
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        print('cancel_me(): cancel sleep')
        raise  # Re-raise is essential
    finally:
        print('cancel_me(): after sleep')

async def main():
    # Create a "cancel_me" Task
    task = asyncio.create_task(cancel_me())

    # Wait for 1 second
    await asyncio.sleep(1)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("main(): cancel_me is cancelled now")

asyncio.run(main())
```

**Important Guidelines**:
- [`asyncio.CancelledError`](asyncio-exceptions.html#asyncio.CancelledError "asyncio.CancelledError") should generally be propagated when clean-up is complete
- [`asyncio.CancelledError`](asyncio-exceptions.html#asyncio.CancelledError "asyncio.CancelledError") directly subclasses [`BaseException`](exceptions.html#BaseException "BaseException")
- Components that enable structured concurrency might misbehave if a coroutine swallows [`asyncio.CancelledError`](asyncio-exceptions.html#asyncio.CancelledError "asyncio.CancelledError")

## Task Groups (Structured Concurrency)

### TaskGroup Class

```python
class asyncio.TaskGroup
```

An [asynchronous context manager](../reference/datamodel.html#async-context-managers) holding a group of tasks. Tasks can be added to the group using [`create_task()`](#asyncio.create_task "asyncio.create_task"). All tasks are awaited when the context manager exits.

**Usage Pattern**:

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(some_coro(...))
        task2 = tg.create_task(another_coro(...))
    print(f"Both tasks have completed now: {task1.result()}, {task2.result()}")
```

### Exception Handling in Task Groups

**Exception Propagation**:
- The first time any task fails with an exception other than [`asyncio.CancelledError`](asyncio-exceptions.html#asyncio.CancelledError "asyncio.CancelledError"), the remaining tasks in the group are cancelled
- Exceptions are combined in an [`ExceptionGroup`](exceptions.html#ExceptionGroup "ExceptionGroup") or [`BaseExceptionGroup`](exceptions.html#BaseExceptionGroup "BaseExceptionGroup")

**Special Cases**:
- [`KeyboardInterrupt`](exceptions.html#KeyboardInterrupt "KeyboardInterrupt") or [`SystemExit`](exceptions.html#SystemExit "SystemExit") are re-raised instead of being grouped

### Task Group Termination Pattern

```python
import asyncio
from asyncio import TaskGroup

class TerminateTaskGroup(Exception):
    """Exception raised to terminate a task group."""

async def force_terminate_task_group():
    """Used to force termination of a task group."""
    raise TerminateTaskGroup()

async def job(task_id, sleep_time):
    print(f'Task {task_id}: start')
    await asyncio.sleep(sleep_time)
    print(f'Task {task_id}: done')

async def main():
    try:
        async with TaskGroup() as group:
            # spawn some tasks
            group.create_task(job(1, 0.5))
            group.create_task(job(2, 1.5))
            # sleep for 1 second
            await asyncio.sleep(1)
            # add an exception-raising task to force the group to terminate
            group.create_task(force_terminate_task_group())
    except* TerminateTaskGroup:
        pass

asyncio.run(main())
```

## Running Tasks Concurrently

### asyncio.gather()

```python
awaitable asyncio.gather(*aws, return_exceptions=False)
```

Run [awaitable objects](#asyncio-awaitables) in the _aws_ sequence _concurrently_.

**Key Features**:
- If all awaitables are completed successfully, the result is an aggregate list of returned values
- Order of result values corresponds to the order of awaitables in _aws_
- If _return_exceptions_ is `False` (default), the first raised exception is immediately propagated
- If _return_exceptions_ is `True`, exceptions are treated the same as successful results

**Example**:

```python
import asyncio

async def factorial(name, number):
    f = 1
    for i in range(2, number + 1):
        print(f"Task {name}: Compute factorial({number}), currently i={i}...")
        await asyncio.sleep(1)
        f *= i
    print(f"Task {name}: factorial({number}) = {f}")
    return f

async def main():
    # Schedule three calls *concurrently*:
    L = await asyncio.gather(
        factorial("A", 2),
        factorial("B", 3),
        factorial("C", 4),
    )
    print(L)

asyncio.run(main())
```

**Note**: [`asyncio.TaskGroup`](#asyncio.TaskGroup "asyncio.TaskGroup") provides stronger safety guarantees than _gather_ for scheduling a nesting of subtasks.

## Timeouts and Waiting

### asyncio.timeout()

```python
asyncio.timeout(delay)
```

Return an [asynchronous context manager](../reference/datamodel.html#async-context-managers) that can be used to limit the amount of time spent waiting on something.

**Usage**:

```python
async def main():
    async with asyncio.timeout(10):
        await long_running_task()
```

**Important**: The [`asyncio.timeout()`](#asyncio.timeout "asyncio.timeout") context manager transforms [`asyncio.CancelledError`](asyncio-exceptions.html#asyncio.CancelledError "asyncio.CancelledError") into a [`TimeoutError`](exceptions.html#TimeoutError "TimeoutError"), which means the [`TimeoutError`](exceptions.html#TimeoutError "TimeoutError") can only be caught _outside_ of the context manager.

### Timeout with Exception Handling

```python
async def main():
    try:
        async with asyncio.timeout(10):
            await long_running_task()
    except TimeoutError:
        print("The long operation timed out, but we've handled it.")

    print("This statement will run regardless.")
```

### asyncio.wait_for()

```python
async asyncio.wait_for(aw, timeout)
```

Wait for the _aw_ [awaitable](#asyncio-awaitables) to complete with a timeout.

**Behavior**:
- If a timeout occurs, it cancels the task and raises [`TimeoutError`](exceptions.html#TimeoutError "TimeoutError")
- To avoid the task [`cancellation`](#asyncio.Task.cancel "asyncio.Task.cancel"), wrap it in [`shield()`](#asyncio.shield "asyncio.shield")
- The function will wait until the future is actually cancelled, so the total wait time may exceed the _timeout_

## Task Object

### Key Task Methods

```python
class asyncio.Task(coro, *, loop=None, name=None, context=None, eager_start=False)
```

**Core Methods**:

```python
def done()
```
Return `True` if the Task is _done_.

```python
def result()
```
Return the result of the Task. Raises exceptions if task failed or was cancelled.

```python
def cancel(msg=None)
```
Request the Task to be cancelled. Returns `True` if cancellation was requested, `False` if already done.

```python
def cancelled()
```
Return `True` if the Task is _cancelled_.

```python
def uncancel()
```
Decrement the count of cancellation requests to this Task. Returns the remaining number of cancellation requests.

```python
def cancelling()
```
Return the number of pending cancellation requests to this Task.

**Task Lifecycle**:

```python
def get_name()
def set_name(value)
```

**Context Management**:

```python
def get_context()
```
Return the [`contextvars.Context`](contextvars.html#contextvars.Context "contextvars.Context") object associated with the task.

## Quant Trading Implementation Patterns

### Market Data Processing Task Management

```python
class TradingSystemTaskManager:
    def __init__(self):
        self.active_tasks = set()
        self.shutdown_event = asyncio.Event()
    
    async def create_market_data_pipeline(self):
        """Create and manage market data processing tasks"""
        async with asyncio.TaskGroup() as tg:
            # WebSocket connection task
            ws_task = tg.create_task(
                self.websocket_handler(),
                name="hyperliquid_websocket"
            )
            
            # Data processing workers
            for i in range(4):
                worker_task = tg.create_task(
                    self.data_processor(f"processor_{i}"),
                    name=f"data_processor_{i}"
                )
                self.active_tasks.add(worker_task)
            
            # Performance monitor
            monitor_task = tg.create_task(
                self.performance_monitor(),
                name="performance_monitor"
            )
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()

    async def websocket_handler(self):
        """Handle WebSocket connection with automatic reconnection"""
        while not self.shutdown_event.is_set():
            try:
                async with asyncio.timeout(30):  # 30-second timeout
                    async with websocket.connect("wss://api.hyperliquid.xyz/ws") as ws:
                        await self.process_websocket_data(ws)
            except TimeoutError:
                print("WebSocket timeout, reconnecting...")
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("WebSocket handler cancelled, cleaning up...")
                raise
            except Exception as e:
                print(f"WebSocket error: {e}, retrying in 5 seconds...")
                await asyncio.sleep(5)

    async def graceful_shutdown(self):
        """Implement graceful shutdown with proper task cancellation"""
        print("Initiating graceful shutdown...")
        
        # Signal all tasks to stop
        self.shutdown_event.set()
        
        # Wait a moment for tasks to finish current work
        await asyncio.sleep(2)
        
        # Cancel remaining tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete cancellation
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        print("Graceful shutdown complete")

    async def data_processor(self, name):
        """Process market data with proper cancellation handling"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Process data with timeout
                    async with asyncio.timeout(5):
                        data = await self.get_market_data()
                        await self.process_and_store(data)
                except TimeoutError:
                    print(f"{name}: Processing timeout, continuing...")
                except asyncio.CancelledError:
                    print(f"{name}: Cancelled, performing cleanup...")
                    await self.cleanup_processor_state()
                    raise
                
                await asyncio.sleep(0.01)  # Yield control
                
        except asyncio.CancelledError:
            print(f"{name}: Clean cancellation completed")
            raise
        finally:
            print(f"{name}: Processor shutdown complete")
```

### Error Recovery with Exponential Backoff

```python
async def resilient_task_with_backoff(operation, max_retries=5):
    """Execute task with exponential backoff on failure"""
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(30):  # 30-second timeout per attempt
                return await operation()
        except asyncio.CancelledError:
            print("Task cancelled, aborting retries...")
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Final attempt failed: {e}")
                raise
            
            backoff_time = 2 ** attempt  # Exponential backoff
            print(f"Attempt {attempt + 1} failed: {e}, retrying in {backoff_time}s...")
            await asyncio.sleep(backoff_time)
```

### Task Performance Monitoring

```python
async def task_performance_monitor():
    """Monitor task performance and health"""
    while True:
        try:
            # Get all running tasks
            all_tasks = asyncio.all_tasks()
            
            for task in all_tasks:
                task_name = task.get_name()
                
                if task.done():
                    if task.cancelled():
                        print(f"Task {task_name}: CANCELLED")
                    else:
                        try:
                            result = task.result()
                            print(f"Task {task_name}: COMPLETED")
                        except Exception as e:
                            print(f"Task {task_name}: FAILED with {e}")
                else:
                    print(f"Task {task_name}: RUNNING")
            
            await asyncio.sleep(10)  # Monitor every 10 seconds
            
        except asyncio.CancelledError:
            print("Performance monitor cancelled")
            break
```

## Key Benefits for Trading Systems

1. **Structured Concurrency**: TaskGroup provides safe concurrent execution with automatic cleanup
2. **Graceful Cancellation**: Proper cancellation handling prevents data corruption and ensures clean shutdown
3. **Timeout Management**: Built-in timeout support prevents hanging operations in market data processing
4. **Error Isolation**: Task groups isolate failures and provide comprehensive exception handling
5. **Resource Management**: Automatic task reference management prevents memory leaks
6. **Performance Monitoring**: Task introspection enables real-time system health monitoring

This task management framework is essential for building robust, high-performance quantitative trading systems that can handle real-time market data processing with proper error recovery and graceful shutdown capabilities.