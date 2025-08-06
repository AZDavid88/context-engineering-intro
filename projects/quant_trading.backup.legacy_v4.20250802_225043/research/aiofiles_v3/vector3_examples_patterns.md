# aiofiles Implementation Examples and Patterns (Vector 3)

## Advanced Implementation Patterns

This vector focuses on advanced usage patterns, performance optimization techniques, and production-ready implementation examples extracted from the aiofiles codebase and test suite.

## Core Implementation Architecture

### 1. Thread Pool Delegation Pattern

**Core Mechanism:**
```python
# aiofiles internal pattern
import asyncio
from functools import partial

async def _open(file, mode='r', buffering=-1, encoding=None, 
               errors=None, newline=None, closefd=True, opener=None, 
               *, loop=None, executor=None):
    if loop is None:
        loop = asyncio.get_running_loop()
    
    # Create partial function for synchronous open
    cb = partial(sync_open, file, mode=mode, buffering=buffering,
                encoding=encoding, errors=errors, newline=newline,
                closefd=closefd, opener=opener)
    
    # Delegate to thread pool
    f = await loop.run_in_executor(executor, cb)
    
    # Wrap with async interface
    return wrap(f, loop=loop, executor=executor)
```

**Key Benefits:**
- Non-blocking I/O operations
- Maintains compatibility with standard file interface
- Configurable thread pool management

### 2. Singledispatch Wrapping Pattern

**Implementation:**
```python
from functools import singledispatch
from typing import Union, BinaryIO, TextIO

@singledispatch
def wrap(file, *, loop=None, executor=None):
    raise TypeError(f"Unsupported file type: {type(file)}")

@wrap.register(io.TextIOWrapper)
def _(file, *, loop=None, executor=None):
    return AsyncTextIOWrapper(file, loop=loop, executor=executor)

@wrap.register(io.BufferedReader)
def _(file, *, loop=None, executor=None):
    return AsyncBufferedReader(file, loop=loop, executor=executor)

@wrap.register(io.BufferedWriter)
def _(file, *, loop=None, executor=None):
    return AsyncBufferedWriter(file, loop=loop, executor=executor)
```

**Pattern Advantages:**
- Type-specific async wrapper selection
- Extensible for custom file types
- Clean separation of concerns

## High-Performance Patterns for Trading Systems

### 1. Concurrent File Processing Pipeline

```python
import asyncio
import aiofiles
from typing import List, AsyncGenerator
import time

class HighPerformanceFileProcessor:
    def __init__(self, max_concurrent_files: int = 10, 
                 buffer_size: int = 1048576):  # 1MB buffer
        self.max_concurrent_files = max_concurrent_files
        self.buffer_size = buffer_size
        self.semaphore = asyncio.Semaphore(max_concurrent_files)
    
    async def process_file_batch(self, file_paths: List[str]) -> List[bytes]:
        """Process multiple files concurrently with rate limiting"""
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(
                self._process_single_file(file_path)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    async def _process_single_file(self, file_path: str) -> bytes:
        """Process single file with semaphore rate limiting"""
        async with self.semaphore:
            async with aiofiles.open(file_path, 'rb', 
                                   buffering=self.buffer_size) as f:
                return await f.read()

# Usage for trading data processing
processor = HighPerformanceFileProcessor(max_concurrent_files=20)
market_data_files = ['tick_data_1.bin', 'tick_data_2.bin', 'tick_data_3.bin']
data_chunks = await processor.process_file_batch(market_data_files)
```

### 2. Streaming Data Pipeline with Backpressure Control

```python
import asyncio
import aiofiles
from typing import AsyncGenerator
import logging

class StreamingFileDataPipeline:
    def __init__(self, queue_size: int = 1000, chunk_size: int = 65536):
        self.queue_size = queue_size
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    async def create_data_stream(self, file_path: str) -> AsyncGenerator[bytes, None]:
        """Create async generator for streaming file data"""
        try:
            async with aiofiles.open(file_path, 'rb', 
                                   buffering=self.chunk_size * 16) as f:
                while True:
                    chunk = await f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            self.logger.error(f"Error streaming {file_path}: {e}")
            raise
    
    async def producer_consumer_pipeline(self, file_paths: List[str]) -> None:
        """Producer-consumer pattern with backpressure control"""
        queue = asyncio.Queue(maxsize=self.queue_size)
        
        # Producer task
        async def producer():
            for file_path in file_paths:
                async for chunk in self.create_data_stream(file_path):
                    await queue.put(chunk)
            await queue.put(None)  # Sentinel value
        
        # Consumer task
        async def consumer():
            processed_chunks = 0
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                
                # Process chunk (e.g., parse trading data)
                await self._process_trading_data_chunk(chunk)
                processed_chunks += 1
                queue.task_done()
            
            self.logger.info(f"Processed {processed_chunks} chunks")
        
        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())
    
    async def _process_trading_data_chunk(self, chunk: bytes) -> None:
        """Process trading data chunk - placeholder for actual processing"""
        # Simulate processing time
        await asyncio.sleep(0.001)  # 1ms processing time

# Usage for real-time trading data processing
pipeline = StreamingFileDataPipeline(queue_size=5000, chunk_size=32768)
trading_files = ['market_feed_1.dat', 'market_feed_2.dat']
await pipeline.producer_consumer_pipeline(trading_files)
```

### 3. Async File Writing with Batching and Flush Control

```python
import asyncio
import aiofiles
from typing import List, Optional
import time

class HighThroughputFileWriter:
    def __init__(self, file_path: str, batch_size: int = 1000, 
                 flush_interval: float = 1.0, buffer_size: int = 1048576):
        self.file_path = file_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer_size = buffer_size
        self.write_queue = asyncio.Queue()
        self.batch_buffer = []
        self.last_flush = time.time()
        self._writer_task = None
        self._file = None
    
    async def __aenter__(self):
        self._file = await aiofiles.open(
            self.file_path, 'ab', buffering=self.buffer_size
        )
        self._writer_task = asyncio.create_task(self._batch_writer())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.write_queue.put(None)  # Sentinel to stop writer
        if self._writer_task:
            await self._writer_task
        if self._file:
            await self._file.close()
    
    async def write_data(self, data: bytes) -> None:
        """Queue data for batched writing"""
        await self.write_queue.put(data)
    
    async def _batch_writer(self) -> None:
        """Background task for batched writing with periodic flushes"""
        while True:
            try:
                # Wait for data with timeout for periodic flushes
                data = await asyncio.wait_for(
                    self.write_queue.get(), timeout=self.flush_interval
                )
                
                if data is None:  # Sentinel value
                    await self._flush_remaining()
                    break
                
                self.batch_buffer.append(data)
                
                # Check if batch is full or flush interval exceeded
                current_time = time.time()
                if (len(self.batch_buffer) >= self.batch_size or 
                    current_time - self.last_flush >= self.flush_interval):
                    await self._flush_batch()
                    
            except asyncio.TimeoutError:
                # Periodic flush on timeout
                if self.batch_buffer:
                    await self._flush_batch()
    
    async def _flush_batch(self) -> None:
        """Flush current batch to file"""
        if self.batch_buffer and self._file:
            batch_data = b''.join(self.batch_buffer)
            await self._file.write(batch_data)
            await self._file.flush()
            self.batch_buffer.clear()
            self.last_flush = time.time()
    
    async def _flush_remaining(self) -> None:
        """Flush any remaining data before closing"""
        await self._flush_batch()

# Usage for high-frequency trading data logging
async def log_trading_data(trades: List[bytes]) -> None:
    async with HighThroughputFileWriter(
        'trading_log.dat', 
        batch_size=500, 
        flush_interval=0.5
    ) as writer:
        for trade_data in trades:
            await writer.write_data(trade_data)
```

## Error Handling and Resilience Patterns

### 1. Retry Pattern with Exponential Backoff

```python
import asyncio
import aiofiles
from typing import Optional, Any
import random

class ResilientFileOperations:
    def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def read_with_retry(self, file_path: str, 
                            chunk_size: Optional[int] = None) -> bytes:
        """Read file with exponential backoff retry"""
        for attempt in range(self.max_retries + 1):
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    if chunk_size:
                        return await f.read(chunk_size)
                    else:
                        return await f.read()
            
            except (OSError, IOError) as e:
                if attempt == self.max_retries:
                    raise
                
                # Exponential backoff with jitter
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
        
        raise RuntimeError("Unexpected retry loop exit")
    
    async def write_with_retry(self, file_path: str, data: bytes) -> None:
        """Write file with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(data)
                    await f.flush()
                return
            
            except (OSError, IOError) as e:
                if attempt == self.max_retries:
                    raise
                
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)

# Usage for critical trading data operations
resilient_ops = ResilientFileOperations(max_retries=5, base_delay=0.05)
critical_data = await resilient_ops.read_with_retry('critical_config.json')
```

### 2. Circuit Breaker Pattern for File Operations

```python
import asyncio
import aiofiles
from enum import Enum
from typing import Callable, Any
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class FileOperationCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
    
    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise RuntimeError("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
        
        try:
            result = await operation(*args, **kwargs)
            
            # Success - reset circuit breaker
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            
            raise

# Usage for protecting critical file operations
async def protected_file_read(file_path: str) -> bytes:
    async with aiofiles.open(file_path, 'rb') as f:
        return await f.read()

circuit_breaker = FileOperationCircuitBreaker(failure_threshold=3)
try:
    data = await circuit_breaker.call(protected_file_read, 'trading_data.bin')
except RuntimeError as e:
    print(f"Circuit breaker protection: {e}")
```

## Performance Monitoring and Profiling

### 1. File Operation Performance Metrics

```python
import asyncio
import aiofiles
import time
from typing import Dict, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class FileOperationMetrics:
    operation_type: str
    file_path: str
    duration: float
    bytes_processed: int
    timestamp: float

class FilePerformanceMonitor:
    def __init__(self):
        self.metrics: List[FileOperationMetrics] = []
    
    @asynccontextmanager
    async def monitor_operation(self, operation_type: str, 
                               file_path: str, bytes_count: int = 0):
        """Context manager for monitoring file operations"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            metric = FileOperationMetrics(
                operation_type=operation_type,
                file_path=file_path,
                duration=duration,
                bytes_processed=bytes_count,
                timestamp=time.time()
            )
            self.metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        if not self.metrics:
            return {}
        
        operations = {}
        for metric in self.metrics:
            if metric.operation_type not in operations:
                operations[metric.operation_type] = []
            operations[metric.operation_type].append(metric.duration)
        
        summary = {}
        for op_type, durations in operations.items():
            summary[op_type] = {
                'count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }
        
        return summary

# Usage for monitoring trading system file operations
monitor = FilePerformanceMonitor()

async def monitored_file_operations():
    # Monitor file reading
    async with monitor.monitor_operation('read', 'market_data.bin', 1048576):
        async with aiofiles.open('market_data.bin', 'rb') as f:
            data = await f.read()
    
    # Monitor file writing
    async with monitor.monitor_operation('write', 'processed_data.bin', len(data)):
        async with aiofiles.open('processed_data.bin', 'wb') as f:
            await f.write(data)

await monitored_file_operations()
performance_stats = monitor.get_performance_summary()
print(f"Performance summary: {performance_stats}")
```

## Testing Patterns for aiofiles

### 1. Mock Testing Pattern

```python
import aiofiles
import aiofiles.threadpool
from unittest.mock import mock, patch, MagicMock
import pytest

class TestAiofilesOperations:
    
    @pytest.fixture
    def mock_file_content(self):
        return b"test file content for trading data"
    
    async def test_mocked_file_read(self, mock_file_content):
        """Test aiofiles read operations with mocking"""
        
        # Register mock with aiofiles wrapper
        aiofiles.threadpool.wrap.register(MagicMock)(
            lambda *args, **kwargs: aiofiles.threadpool.AsyncBufferedIOBase(*args, **kwargs)
        )
        
        # Create mock file stream
        mock_file_stream = MagicMock()
        mock_file_stream.read = MagicMock(return_value=mock_file_content)
        mock_file_stream.close = MagicMock()
        
        # Patch sync_open to return mock
        with patch('aiofiles.threadpool.sync_open', return_value=mock_file_stream):
            async with aiofiles.open('test_file.bin', 'rb') as f:
                content = await f.read()
                assert content == mock_file_content
            
            mock_file_stream.read.assert_called_once()
            mock_file_stream.close.assert_called_once()
    
    async def test_concurrent_file_processing(self):
        """Test concurrent file processing patterns"""
        file_contents = [b"data1", b"data2", b"data3"]
        
        async def mock_file_processor(file_path: str, expected_content: bytes):
            # Mock the file reading
            with patch('aiofiles.threadpool.sync_open') as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = expected_content
                mock_open.return_value = mock_file
                
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                    return content
        
        # Test concurrent processing
        tasks = []
        for i, content in enumerate(file_contents):
            task = asyncio.create_task(
                mock_file_processor(f'file_{i}.bin', content)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert results == file_contents
```

## Production Deployment Patterns

### 1. Configuration-Driven File Operations

```python
import asyncio
import aiofiles
from dataclasses import dataclass
from typing import Optional, Dict, Any
import concurrent.futures

@dataclass
class FileOperationConfig:
    max_concurrent_files: int = 10
    buffer_size: int = 1048576  # 1MB
    thread_pool_workers: int = 4
    enable_monitoring: bool = True
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5

class ProductionFileManager:
    def __init__(self, config: FileOperationConfig):
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.thread_pool_workers,
            thread_name_prefix='aiofiles_production'
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent_files)
    
    async def read_file_safely(self, file_path: str) -> Optional[bytes]:
        """Production-safe file reading with all protections"""
        async with self.semaphore:
            try:
                async with aiofiles.open(
                    file_path, 'rb', 
                    buffering=self.config.buffer_size,
                    executor=self.executor
                ) as f:
                    return await f.read()
                    
            except Exception as e:
                if self.config.enable_monitoring:
                    self._log_error(f"Failed to read {file_path}: {e}")
                return None
    
    def _log_error(self, message: str) -> None:
        """Centralized error logging for production monitoring"""
        # Integration point for production logging system
        print(f"[PRODUCTION ERROR] {message}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of file operations"""
        self.executor.shutdown(wait=True)

# Production usage
config = FileOperationConfig(
    max_concurrent_files=20,
    buffer_size=2097152,  # 2MB
    thread_pool_workers=8,
    enable_monitoring=True
)

file_manager = ProductionFileManager(config)
try:
    data = await file_manager.read_file_safely('critical_trading_data.bin')
finally:
    await file_manager.shutdown()
```

These implementation patterns provide comprehensive coverage of advanced aiofiles usage suitable for high-performance trading systems, including concurrency management, error handling, performance monitoring, and production deployment considerations.