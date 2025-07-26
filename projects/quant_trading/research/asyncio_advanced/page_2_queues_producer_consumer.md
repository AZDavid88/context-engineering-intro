# AsyncIO Queues - Producer-Consumer Patterns

**Source**: https://docs.python.org/3/library/asyncio-queue.html
**Extraction Method**: Brightdata MCP
**Research Focus**: Producer-consumer patterns, queue management, backpressure handling for high-frequency trading data pipelines

## Overview

asyncio queues are designed to be similar to classes of the [`queue`](queue.html#module-queue "queue: A synchronized queue class.") module. Although asyncio queues are not thread-safe, they are designed to be used specifically in async/await code.

**Important**: Methods of asyncio queues don't have a _timeout_ parameter; use [`asyncio.wait_for()`](asyncio-task.html#asyncio.wait_for "asyncio.wait_for") function to do queue operations with a timeout.

## Queue Class

### Basic Queue (FIFO)

```python
class asyncio.Queue(maxsize=0)
```

A first in, first out (FIFO) queue.

**Backpressure Management**:
- If _maxsize_ is less than or equal to zero, the queue size is infinite
- If it is an integer greater than `0`, then `await put()` blocks when the queue reaches _maxsize_ until an item is removed by [`get()`](#asyncio.Queue.get "asyncio.Queue.get")

**Key Properties**:
- The size of the queue is always known and can be returned by calling the [`qsize()`](#asyncio.Queue.qsize "asyncio.Queue.qsize") method
- This class is **not thread safe**

### Queue Methods

#### Core Operations

```python
async def get()
```
Remove and return an item from the queue. If queue is empty, wait until an item is available.

**Exception Handling**:
- Raises [`QueueShutDown`](#asyncio.QueueShutDown "asyncio.QueueShutDown") if the queue has been shut down and is empty, or if the queue has been shut down immediately

```python
def get_nowait()
```
Return an item if one is immediately available, else raise [`QueueEmpty`](#asyncio.QueueEmpty "asyncio.QueueEmpty").

```python
async def put(item)
```
Put an item into the queue. If the queue is full, wait until a free slot is available before adding the item.

**Exception Handling**:
- Raises [`QueueShutDown`](#asyncio.QueueShutDown "asyncio.QueueShutDown") if the queue has been shut down

```python
def put_nowait(item)
```
Put an item into the queue without blocking.

**Backpressure Handling**:
- If no free slot is immediately available, raise [`QueueFull`](#asyncio.QueueFull "asyncio.QueueFull")

#### Queue Management

```python
def qsize()
```
Return the number of items in the queue.

```python
def empty()
```
Return `True` if the queue is empty, `False` otherwise.

```python
def full()
```
Return `True` if there are [`maxsize`](#asyncio.Queue.maxsize "asyncio.Queue.maxsize") items in the queue.

If the queue was initialized with `maxsize=0` (the default), then [`full()`](#asyncio.Queue.full "asyncio.Queue.full") never returns `True`.

#### Task Coordination

```python
async def join()
```
Block until all items in the queue have been received and processed.

**Workflow Management**:
- The count of unfinished tasks goes up whenever an item is added to the queue
- The count goes down whenever a consumer coroutine calls [`task_done()`](#asyncio.Queue.task_done "asyncio.Queue.task_done")
- When the count of unfinished tasks drops to zero, [`join()`](#asyncio.Queue.join "asyncio.Queue.join") unblocks

```python
def task_done()
```
Indicate that a formerly enqueued work item is complete.

**Usage Pattern**:
- Used by queue consumers
- For each [`get()`](#asyncio.Queue.get "asyncio.Queue.get") used to fetch a work item, a subsequent call to [`task_done()`](#asyncio.Queue.task_done "asyncio.Queue.task_done") tells the queue that the processing on the work item is complete
- Raises [`ValueError`](exceptions.html#ValueError "ValueError") if called more times than there were items placed in the queue

#### Graceful Shutdown

```python
def shutdown(immediate=False)
```
Put a [`Queue`](#asyncio.Queue "asyncio.Queue") instance into a shutdown mode.

**Shutdown Behavior**:
- The queue can no longer grow
- Future calls to [`put()`](#asyncio.Queue.put "asyncio.Queue.put") raise [`QueueShutDown`](#asyncio.QueueShutDown "asyncio.QueueShutDown")
- Currently blocked callers of [`put()`](#asyncio.Queue.put "asyncio.Queue.put") will be unblocked and raise [`QueueShutDown`](#asyncio.QueueShutDown "asyncio.QueueShutDown")

**Graceful vs Immediate Shutdown**:

If _immediate_ is `False` (default):
- The queue can be wound down normally with [`get()`](#asyncio.Queue.get "asyncio.Queue.get") calls to extract tasks that have already been loaded
- If [`task_done()`](#asyncio.Queue.task_done "asyncio.Queue.task_done") is called for each remaining task, a pending [`join()`](#asyncio.Queue.join "asyncio.Queue.join") will be unblocked normally
- Once the queue is empty, future calls to [`get()`](#asyncio.Queue.get "asyncio.Queue.get") will raise [`QueueShutDown`](#asyncio.QueueShutDown "asyncio.QueueShutDown")

If _immediate_ is `True`:
- The queue is terminated immediately
- The queue is drained to be completely empty and the count of unfinished tasks is reduced by the number of tasks drained
- If unfinished tasks is zero, callers of [`join()`](#asyncio.Queue.join "asyncio.Queue.join") are unblocked
- Blocked callers of [`get()`](#asyncio.Queue.get "asyncio.Queue.get") are unblocked and will raise [`QueueShutDown`](#asyncio.QueueShutDown "asyncio.QueueShutDown")

**Warning**: Use caution when using [`join()`](#asyncio.Queue.join "asyncio.Queue.join") with _immediate_ set to true. This unblocks the join even when no work has been done on the tasks, violating the usual invariant for joining a queue.

## Queue Variants

### Priority Queue

```python
class asyncio.PriorityQueue
```
A variant of [`Queue`](#asyncio.Queue "asyncio.Queue"); retrieves entries in priority order (lowest first).

Entries are typically tuples of the form `(priority_number, data)`.

### LIFO Queue

```python
class asyncio.LifoQueue
```
A variant of [`Queue`](#asyncio.Queue "asyncio.Queue") that retrieves most recently added entries first (last in, first out).

## Exceptions

```python
exception asyncio.QueueEmpty
```
This exception is raised when the [`get_nowait()`](#asyncio.Queue.get_nowait "asyncio.Queue.get_nowait") method is called on an empty queue.

```python
exception asyncio.QueueFull
```
Exception raised when the [`put_nowait()`](#asyncio.Queue.put_nowait "asyncio.Queue.put_nowait") method is called on a queue that has reached its _maxsize_.

```python
exception asyncio.QueueShutDown
```
Exception raised when [`put()`](#asyncio.Queue.put "asyncio.Queue.put") or [`get()`](#asyncio.Queue.get "asyncio.Queue.get") is called on a queue which has been shut down.

## Producer-Consumer Example

**Workload Distribution Pattern**:

```python
import asyncio
import random
import time

async def worker(name, queue):
    while True:
        # Get a "work item" out of the queue.
        sleep_for = await queue.get()

        # Sleep for the "sleep_for" seconds.
        await asyncio.sleep(sleep_for)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()

        print(f'{name} has slept for {sleep_for:.2f} seconds')

async def main():
    # Create a queue that we will use to store our "workload".
    queue = asyncio.Queue()

    # Generate random timings and put them into the queue.
    total_sleep_time = 0
    for _ in range(20):
        sleep_for = random.uniform(0.05, 1.0)
        total_sleep_time += sleep_for
        queue.put_nowait(sleep_for)

    # Create three worker tasks to process the queue concurrently.
    tasks = []
    for i in range(3):
        task = asyncio.create_task(worker(f'worker-{i}', queue))
        tasks.append(task)

    # Wait until the queue is fully processed.
    started_at = time.monotonic()
    await queue.join()
    total_slept_for = time.monotonic() - started_at

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)

    print('====')
    print(f'3 workers slept in parallel for {total_slept_for:.2f} seconds')
    print(f'total expected sleep time: {total_sleep_time:.2f} seconds')

asyncio.run(main())
```

## Quant Trading Implementation Patterns

### Market Data Pipeline Architecture

```python
# High-Frequency Data Processing Queue
market_data_queue = asyncio.Queue(maxsize=10000)  # Backpressure at 10k items

# WebSocket Producer
async def market_data_producer(queue):
    async with websocket.connect("wss://api.hyperliquid.xyz/ws") as ws:
        while True:
            try:
                data = await ws.recv()
                await queue.put(orjson.loads(data))  # Blocks if queue full
            except asyncio.QueueShutDown:
                break

# Data Processing Consumer
async def data_processor(name, queue, storage_engine):
    while True:
        try:
            tick_data = await queue.get()
            
            # Process OHLCV aggregation
            processed_data = await process_market_tick(tick_data)
            
            # Store in DuckDB/PyArrow pipeline
            await storage_engine.store(processed_data)
            
            # Signal completion
            queue.task_done()
            
        except asyncio.QueueShutDown:
            break
```

### Error Recovery and Graceful Shutdown

```python
async def graceful_shutdown(queue, workers):
    """Implement graceful shutdown for trading system"""
    print("Initiating graceful shutdown...")
    
    # Stop accepting new data
    queue.shutdown(immediate=False)
    
    # Wait for existing work to complete
    await queue.join()
    
    # Cancel worker tasks
    for worker in workers:
        worker.cancel()
    
    # Wait for workers to finish cleanup
    await asyncio.gather(*workers, return_exceptions=True)
    
    print("Shutdown complete - all market data processed")
```

### Performance Monitoring

```python
async def queue_monitor(queue):
    """Monitor queue depth and performance metrics"""
    while True:
        queue_size = queue.qsize()
        
        if queue_size > 8000:  # 80% of max capacity
            print(f"⚠️  High queue depth: {queue_size}/10000 (backpressure)")
        elif queue_size > 5000:  # 50% of max capacity
            print(f"⚡ Moderate queue depth: {queue_size}/10000")
        
        await asyncio.sleep(1)  # Check every second
```

## Key Benefits for Trading Systems

1. **Backpressure Management**: `maxsize` parameter prevents memory overflow during high-frequency data bursts
2. **Flow Control**: `put()` blocks automatically when queue is full, providing natural throttling
3. **Work Coordination**: `task_done()` and `join()` enable clean shutdown and completion tracking
4. **Exception Safety**: Specific exceptions for different failure modes enable robust error handling
5. **Performance Monitoring**: `qsize()` enables real-time monitoring of pipeline health
6. **Graceful Shutdown**: `shutdown()` method supports clean system termination without data loss

This queue-based architecture is essential for building robust, high-performance quantitative trading systems that can handle real-time market data processing with proper backpressure control and error recovery.