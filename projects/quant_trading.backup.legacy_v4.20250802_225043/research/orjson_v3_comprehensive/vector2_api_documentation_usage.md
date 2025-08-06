# Vector 2: orjson API Documentation & Usage Patterns

## Complete API Reference

### Core Functions

#### orjson.dumps()
```python
def dumps(
    __obj: Any,
    default: Optional[Callable[[Any], Any]] = ...,
    option: Optional[int] = ...,
) -> bytes: ...
```

**Input Types Supported**: `str`, `dict`, `list`, `tuple`, `int`, `float`, `bool`, `None`, `dataclasses.dataclass`, `typing.TypedDict`, `datetime.datetime`, `datetime.date`, `datetime.time`, `uuid.UUID`, `numpy.ndarray`, `orjson.Fragment`

**Return Type**: `bytes` (UTF-8 encoded)

**Key Features**:
- **10x faster** than standard `json.dumps()` 
- **Native type serialization** without Python object conversion overhead
- **GIL management**: Predictable lock behavior during serialization
- **Memory efficiency**: Direct memory access patterns

#### orjson.loads()
```python
def loads(__obj: Union[bytes, bytearray, memoryview, str]) -> Any: ...
```

**Input Types**: `bytes`, `bytearray`, `memoryview`, `str`
**Return Types**: `dict`, `list`, `int`, `float`, `str`, `bool`, `None`

**Performance Optimizations**:
- **2x faster** than standard `json.loads()`
- **Zero-copy parsing**: Direct memory access for `memoryview`/`bytearray`
- **Key caching**: Reduces memory usage with 2048-entry cache for keys ≤64 bytes
- **Strict UTF-8 validation**: Prevents data corruption

## Performance Optimization Flags

### Critical Performance Options

#### OPT_APPEND_NEWLINE
```python
# Avoid string copying overhead
orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE)
# Equivalent to dumps(...) + b"\n" but without memory copy
```

#### OPT_NON_STR_KEYS  
```python
# For complex trading data structures
trading_data = {
    datetime.datetime.now(): price_data,
    uuid.uuid4(): order_data,
    42: position_data
}
orjson.dumps(trading_data, option=orjson.OPT_NON_STR_KEYS)
```

#### OPT_SERIALIZE_NUMPY
```python
# High-performance array serialization
import numpy as np
price_array = np.array([[bid, ask, volume], [bid2, ask2, volume2]])
orjson.dumps(price_array, option=orjson.OPT_SERIALIZE_NUMPY)
# Native C array processing - no Python list conversion
```

### Memory Efficiency Options

#### Fragment Integration
```python
# Pre-serialized JSON inclusion (cache optimization)
cached_market_data = orjson.Fragment(b'{"bid": 100.5, "ask": 100.7}')
message = {
    "type": "market_update", 
    "timestamp": datetime.now(),
    "data": cached_market_data
}
result = orjson.dumps(message)
# Avoids deserializing cached JSON
```

## High-Frequency Trading Optimizations

### WebSocket Message Processing Patterns

#### Input Type Optimization
```python
# Optimal for WebSocket frames (avoid string conversion)
async def process_websocket_message(raw_bytes: bytes):
    # FAST: Direct bytes processing
    data = orjson.loads(raw_bytes)
    
    # SLOW: Unnecessary string conversion  
    # data = orjson.loads(raw_bytes.decode('utf-8'))
```

#### Memory-Efficient Parsing
```python
# Use memoryview for zero-copy parsing
def parse_large_message(buffer: bytearray):
    view = memoryview(buffer)
    return orjson.loads(view)  # No memory copy
```

### Serialization Performance Patterns

#### Batch Processing Optimization
```python
# Efficient batch serialization
def serialize_market_updates(updates: List[Dict]):
    # Pre-allocate options for repeated use
    options = orjson.OPT_NON_STR_KEYS | orjson.OPT_NAIVE_UTC
    
    results = []
    for update in updates:
        # Reuse options, avoid object creation
        serialized = orjson.dumps(update, option=options)
        results.append(serialized)
    return results
```

#### Error Handling Patterns
```python
try:
    serialized = orjson.dumps(trading_data)
except orjson.JSONEncodeError as e:
    # Chained exceptions provide context
    if e.__cause__:
        logger.error(f"Serialization failed: {e.__cause__}")
    else:
        logger.error(f"Invalid type: {e}")
```

## Performance Benchmarks from Tests

### Dictionary Performance (Large Structures)
```python
# Test patterns from test_dict.py
test_sizes = [513, 4_097, 65_537]  # Keys per dictionary

# Optimization insights:
# - Handles 65k+ keys efficiently
# - Unicode key support with minimal overhead
# - Nested dictionary structures optimized
```

### JSON Parsing Robustness
```python
# Input type performance (from test_parsing.py)
input_types = [
    bytes,        # Fastest - direct memory access
    bytearray,    # Fast - mutable buffer
    memoryview,   # Zero-copy - most efficient for large data
    str           # Slower - requires UTF-8 validation
]
```

### Benchmark Data Characteristics
```python
# Real-world fixtures used in benchmarks
benchmark_files = {
    "twitter.json": "Social media API response format",
    "github.json": "GitHub API response (52KiB)",
    "citm_catalog.json": "Complex nested structures (489KiB)",
    "canada.json": "Geographic data with arrays"
}
```

## AsyncIO Integration Strategies

### Thread-Safe Usage Patterns
```python
import asyncio
import orjson

class AsyncJSONProcessor:
    def __init__(self):
        # orjson is thread-safe with GIL management
        self.parse_options = None
        
    async def process_stream(self, websocket):
        async for message in websocket:
            # Non-blocking JSON processing
            data = orjson.loads(message)
            
            # Process in thread pool for CPU-intensive work
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._cpu_intensive_processing, 
                data
            )
            
            # Fast serialization back to bytes
            response = orjson.dumps(result)
            await websocket.send(response)
```

### Producer-Consumer Pattern
```python
import asyncio
import orjson
from asyncio import Queue

async def json_producer(queue: Queue, websocket):
    """High-throughput JSON message producer"""
    async for raw_message in websocket:
        # Fast parsing with zero-copy when possible
        parsed = orjson.loads(raw_message)
        await queue.put(parsed)

async def json_consumer(queue: Queue):
    """Batch processing consumer"""
    batch = []
    
    while True:
        try:
            # Collect batch for efficient processing
            item = await asyncio.wait_for(queue.get(), timeout=0.01)
            batch.append(item)
            
            if len(batch) >= 100:  # Process in batches
                # Efficient batch serialization
                serialized_batch = [
                    orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE)
                    for item in batch
                ]
                
                # Process batch...
                batch.clear()
                
        except asyncio.TimeoutError:
            if batch:  # Process remaining items
                # Handle partial batch...
                batch.clear()
```

## Error Handling & Validation

### Comprehensive Error Patterns
```python
# From test suite analysis - robust error handling
def safe_json_processing(data):
    try:
        return orjson.dumps(data)
    except orjson.JSONEncodeError as e:
        error_cases = {
            "Invalid UTF-8": "str is not valid UTF-8",
            "Unsupported type": "Type is not JSON serializable",
            "Integer overflow": "Integer exceeds 53-bit range",
            "Circular reference": "Circular reference detected",
            "Deep recursion": "Recursion limit exceeded"
        }
        
        for error_type, message_pattern in error_cases.items():
            if message_pattern in str(e):
                # Handle specific error type
                return handle_specific_error(error_type, data)
                
        raise  # Re-raise unknown errors
```

### Input Validation Patterns
```python
def validate_json_input(raw_input):
    """Pre-validation for high-frequency processing"""
    
    # Type checking for optimal processing path
    if isinstance(raw_input, (bytes, bytearray)):
        # Fast path - no encoding conversion needed
        return orjson.loads(raw_input)
    elif isinstance(raw_input, memoryview):
        # Zero-copy path - most efficient
        return orjson.loads(raw_input)
    elif isinstance(raw_input, str):
        # Slower path - UTF-8 validation required
        return orjson.loads(raw_input)
    else:
        raise TypeError(f"Unsupported input type: {type(raw_input)}")
```

## Benchmark Dependencies & Tools

### Performance Testing Stack
```python
# From bench/requirements.txt analysis
benchmark_dependencies = [
    "pytest",           # Test framework
    "pytest-benchmark", # Performance measurement
    "numpy",           # Array processing tests
    "pytz",            # Timezone handling
    "python-dateutil", # Date parsing
]
```

### CPU Affinity Optimization
```python
import os

# From bench/util.py - CPU pinning for consistent benchmarks
def optimize_cpu_affinity():
    """Pin process to specific CPU cores for consistent performance"""
    os.sched_setaffinity(0, {0, 1})  # Use cores 0 and 1
```

### Memory Usage Tracking
```python
# Pattern for monitoring memory efficiency
import psutil
import orjson

def benchmark_memory_usage(data):
    """Track memory usage during JSON processing"""
    process = psutil.Process()
    
    # Baseline memory
    baseline = process.memory_info().rss
    
    # JSON processing
    serialized = orjson.dumps(data)
    peak_memory = process.memory_info().rss
    
    parsed = orjson.loads(serialized)
    final_memory = process.memory_info().rss
    
    return {
        "baseline_mb": baseline / 1024 / 1024,
        "peak_mb": peak_memory / 1024 / 1024,
        "final_mb": final_memory / 1024 / 1024,
        "peak_increase_mb": (peak_memory - baseline) / 1024 / 1024
    }
```

## Next Vector Research Requirements

1. **Vector 3**: Detailed performance benchmarks and memory optimization
2. **Vector 4**: Production usage patterns and real-world integration examples  
3. **Vector 5**: AsyncIO-specific patterns and high-frequency trading optimizations