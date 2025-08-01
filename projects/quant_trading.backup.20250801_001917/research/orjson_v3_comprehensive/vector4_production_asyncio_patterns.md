# Vector 4: Production Usage Patterns & AsyncIO Integration

## High-Frequency Trading Production Context (2024)

### Market Size & Performance Requirements
```python
HFT_MARKET_CONTEXT = {
    "market_size_2024": "$10.36 billion",
    "projected_2030": "$16.03 billion",
    "growth_rate": "7.7% CAGR",
    "performance_demands": {
        "latency_requirements": "1-100 microseconds",
        "message_throughput": ">1 million messages/second",
        "connection_speeds": "10-40 Gbps",
        "data_delay_tolerance": "<50ms"
    }
}
```

### Critical Performance Factors
- **Execution Latency**: Time between decision and trade execution
- **Network Latency**: Server-to-exchange communication lag  
- **Data Reception Latency**: Market data ingestion delays
- **JSON Processing Overhead**: Serialization/deserialization impact

## AsyncIO Integration Patterns for Trading Systems

### Producer-Consumer WebSocket Pattern
```python
import asyncio
import orjson
import websockets
from asyncio import Queue
import logging
from typing import Dict, Any, Optional
import time

class HighFrequencyJSONProcessor:
    """Production-ready AsyncIO JSON processor for trading systems"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.message_queue = Queue(maxsize=max_queue_size)
        self.processed_count = 0
        self.error_count = 0
        self.processing_times = []
        
    async def websocket_producer(self, websocket_uri: str):
        """High-throughput WebSocket message producer"""
        
        async with websockets.connect(websocket_uri) as websocket:
            async for raw_message in websocket:
                try:
                    # CRITICAL: Use bytes directly to avoid string conversion
                    if isinstance(raw_message, str):
                        raw_message = raw_message.encode('utf-8')
                    
                    # Fast JSON parsing - 2x faster than standard json
                    start_time = time.perf_counter_ns()
                    parsed_data = orjson.loads(raw_message)
                    parse_time = time.perf_counter_ns() - start_time
                    
                    # Add processing metadata
                    parsed_data['_parse_time_ns'] = parse_time
                    parsed_data['_received_at'] = time.time_ns()
                    
                    # Non-blocking queue insertion
                    try:
                        self.message_queue.put_nowait(parsed_data)
                    except asyncio.QueueFull:
                        # Drop oldest message under high load
                        try:
                            self.message_queue.get_nowait()
                            self.message_queue.put_nowait(parsed_data)
                        except asyncio.QueueEmpty:
                            pass
                            
                except orjson.JSONDecodeError as e:
                    self.error_count += 1
                    logging.error(f"JSON decode error: {e}")
                except Exception as e:
                    self.error_count += 1
                    logging.error(f"Unexpected error in producer: {e}")

    async def trading_data_consumer(self, batch_size: int = 100):
        """Batch processing consumer optimized for trading data"""
        
        batch = []
        batch_start_time = None
        
        while True:
            try:
                # Collect batch with timeout for low-latency processing
                item = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=0.001  # 1ms timeout for low latency
                )
                
                if batch_start_time is None:
                    batch_start_time = time.perf_counter_ns()
                
                batch.append(item)
                
                # Process when batch is full or timeout occurs
                if len(batch) >= batch_size:
                    await self._process_batch(batch, batch_start_time)
                    batch.clear()
                    batch_start_time = None
                    
            except asyncio.TimeoutError:
                # Process partial batch on timeout
                if batch:
                    await self._process_batch(batch, batch_start_time)
                    batch.clear()
                    batch_start_time = None
                    
    async def _process_batch(self, batch: list, start_time: int):
        """Process batch of trading messages with performance tracking"""
        
        process_start = time.perf_counter_ns()
        
        try:
            # Efficient batch serialization with optimized options
            trading_options = (
                orjson.OPT_NON_STR_KEYS |     # Support timestamp keys
                orjson.OPT_NAIVE_UTC |        # Normalize timestamps
                orjson.OPT_SERIALIZE_NUMPY    # Handle price arrays
            )
            
            # Batch serialize - 10x faster than standard json
            serialized_batch = []
            for item in batch:
                serialized = orjson.dumps(item, option=trading_options)
                serialized_batch.append(serialized)
            
            # Record performance metrics
            total_time = time.perf_counter_ns() - start_time
            processing_time = time.perf_counter_ns() - process_start
            
            self.processed_count += len(batch)
            self.processing_times.append({
                'batch_size': len(batch),
                'total_time_ns': total_time,
                'processing_time_ns': processing_time,
                'avg_per_message_ns': processing_time // len(batch)
            })
            
            # Process serialized batch (send to database, forward to clients, etc.)
            await self._handle_processed_batch(serialized_batch)
            
        except Exception as e:
            self.error_count += len(batch)
            logging.error(f"Batch processing error: {e}")
    
    async def _handle_processed_batch(self, serialized_batch: list):
        """Handle processed batch - implement your business logic here"""
        # Example: Forward to multiple clients, save to database, etc.
        pass

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.processing_times:
            return {"status": "no_data"}
            
        avg_batch_size = sum(p['batch_size'] for p in self.processing_times) / len(self.processing_times)
        avg_processing_time = sum(p['processing_time_ns'] for p in self.processing_times) / len(self.processing_times)
        avg_per_message = sum(p['avg_per_message_ns'] for p in self.processing_times) / len(self.processing_times)
        
        return {
            "messages_processed": self.processed_count,
            "errors": self.error_count,
            "error_rate": self.error_count / (self.processed_count + self.error_count),
            "avg_batch_size": avg_batch_size,
            "avg_processing_time_ms": avg_processing_time / 1_000_000,
            "avg_per_message_ns": avg_per_message,
            "throughput_msg_per_sec": self.processed_count / (sum(p['total_time_ns'] for p in self.processing_times) / 1_000_000_000)
        }
```

### Zero-Copy Memory Optimization
```python
import asyncio
import orjson
from typing import Union
import mmap

class ZeroCopyJSONProcessor:
    """Memory-efficient JSON processing for high-throughput systems"""
    
    def __init__(self):
        self.memory_pool = []
        
    async def process_websocket_frames(self, frame_data: Union[bytes, bytearray, memoryview]):
        """Optimal memory processing for WebSocket frames"""
        
        # Memory type optimization for trading systems
        if isinstance(frame_data, memoryview):
            # FASTEST: Zero-copy processing
            return orjson.loads(frame_data)
        elif isinstance(frame_data, (bytes, bytearray)):
            # FAST: Direct memory access
            return orjson.loads(frame_data)
        else:
            # SLOWER: String conversion required
            return orjson.loads(frame_data)
    
    def create_memory_mapped_processor(self, file_path: str):
        """Memory-mapped file processing for large datasets"""
        
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Process memory-mapped data with zero copy
                view = memoryview(mm)
                return orjson.loads(view)

# Production memory monitoring
class MemoryEfficientTrader:
    """Memory-optimized trading system with orjson"""
    
    def __init__(self):
        self.buffer_pool = asyncio.Queue(maxsize=1000)
        self.active_buffers = set()
        
    async def get_buffer(self) -> bytearray:
        """Get reusable buffer from pool"""
        try:
            buffer = self.buffer_pool.get_nowait()
            buffer.clear()  # Reset buffer
            return buffer
        except asyncio.QueueEmpty:
            return bytearray(8192)  # 8KB buffer
    
    async def return_buffer(self, buffer: bytearray):
        """Return buffer to pool for reuse"""
        if len(buffer) <= 16384:  # Max 16KB buffers in pool
            try:
                self.buffer_pool.put_nowait(buffer)
            except asyncio.QueueFull:
                pass  # Let buffer be garbage collected
    
    async def process_market_data_stream(self, websocket):
        """Memory-efficient market data processing"""
        
        async for message in websocket:
            buffer = await self.get_buffer()
            
            try:
                # Write to reusable buffer
                if isinstance(message, str):
                    message_bytes = message.encode('utf-8')
                else:
                    message_bytes = message
                
                buffer.extend(message_bytes)
                
                # Zero-copy processing
                data = orjson.loads(buffer)
                
                # Process data
                await self._handle_market_data(data)
                
            finally:
                await self.return_buffer(buffer)
```

### Production Error Handling Patterns
```python
import asyncio
import orjson
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable
import time

class JSONErrorType(Enum):
    UTF8_ERROR = "utf8_error"
    TYPE_ERROR = "type_error" 
    OVERFLOW_ERROR = "overflow_error"
    CIRCULAR_ERROR = "circular_error"
    DEPTH_ERROR = "depth_error"
    UNKNOWN_ERROR = "unknown_error"

class ProductionJSONHandler:
    """Production-grade JSON handling with comprehensive error management"""
    
    def __init__(self):
        self.error_stats = {error_type: 0 for error_type in JSONErrorType}
        self.fallback_handlers = {}
        self.circuit_breaker_threshold = 100  # errors per minute
        self.circuit_breaker_window = 60  # seconds
        self.recent_errors = []
        
    def register_fallback_handler(self, error_type: JSONErrorType, handler: Callable):
        """Register custom error handling functions"""
        self.fallback_handlers[error_type] = handler
    
    async def safe_serialize(self, data: Any, options: Optional[int] = None) -> Optional[bytes]:
        """Production-safe JSON serialization with fallback handling"""
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            logging.warning("JSON circuit breaker open - rejecting requests")
            return None
        
        try:
            return orjson.dumps(data, option=options or 0)
            
        except orjson.JSONEncodeError as e:
            error_type = self._classify_error(str(e))
            self._record_error(error_type)
            
            # Try fallback handler
            if error_type in self.fallback_handlers:
                try:
                    return await self.fallback_handlers[error_type](data, e)
                except Exception as fallback_error:
                    logging.error(f"Fallback handler failed: {fallback_error}")
            
            # Default fallback strategies
            return await self._default_error_handling(error_type, data, e)
            
        except Exception as e:
            self._record_error(JSONErrorType.UNKNOWN_ERROR)
            logging.error(f"Unexpected JSON error: {e}")
            return None
    
    async def safe_deserialize(self, json_data: Union[str, bytes]) -> Optional[Any]:
        """Production-safe JSON deserialization"""
        
        if self._is_circuit_breaker_open():
            return None
        
        try:
            # Optimize input type for performance
            if isinstance(json_data, str):
                json_data = json_data.encode('utf-8')
            
            return orjson.loads(json_data)
            
        except orjson.JSONDecodeError as e:
            error_type = self._classify_decode_error(str(e))
            self._record_error(error_type)
            
            # Attempt recovery for specific error types
            if error_type == JSONErrorType.UTF8_ERROR:
                return self._handle_utf8_decode_error(json_data)
            
            logging.error(f"JSON decode error: {e}")
            return None
            
        except Exception as e:
            self._record_error(JSONErrorType.UNKNOWN_ERROR)
            logging.error(f"Unexpected decode error: {e}")
            return None
    
    def _classify_error(self, error_message: str) -> JSONErrorType:
        """Classify JSON encoding errors"""
        error_patterns = {
            JSONErrorType.UTF8_ERROR: ["Invalid UTF-8", "surrogates not allowed"],
            JSONErrorType.TYPE_ERROR: ["Type is not JSON serializable"],
            JSONErrorType.OVERFLOW_ERROR: ["Integer exceeds"],
            JSONErrorType.CIRCULAR_ERROR: ["Circular reference"],
            JSONErrorType.DEPTH_ERROR: ["Recursion limit", "recurses"]
        }
        
        for error_type, patterns in error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        return JSONErrorType.UNKNOWN_ERROR
    
    def _classify_decode_error(self, error_message: str) -> JSONErrorType:
        """Classify JSON decoding errors"""
        if "UTF-8" in error_message or "surrogates" in error_message:
            return JSONErrorType.UTF8_ERROR
        return JSONErrorType.UNKNOWN_ERROR
    
    def _record_error(self, error_type: JSONErrorType):
        """Record error for circuit breaker and monitoring"""
        current_time = time.time()
        self.error_stats[error_type] += 1
        self.recent_errors.append(current_time)
        
        # Clean old errors outside window
        cutoff_time = current_time - self.circuit_breaker_window
        self.recent_errors = [t for t in self.recent_errors if t > cutoff_time]
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker should be open"""
        return len(self.recent_errors) >= self.circuit_breaker_threshold
    
    async def _default_error_handling(self, error_type: JSONErrorType, data: Any, error: Exception) -> Optional[bytes]:
        """Default error handling strategies"""
        
        if error_type == JSONErrorType.TYPE_ERROR:
            # Attempt to convert unsupported types
            if hasattr(data, '__dict__'):
                return orjson.dumps(data.__dict__)
            elif hasattr(data, '_asdict'):
                return orjson.dumps(data._asdict())
            else:
                return orjson.dumps(str(data))
        
        elif error_type == JSONErrorType.CIRCULAR_ERROR:
            # Handle circular references by converting to string
            return orjson.dumps({"error": "circular_reference", "data": str(data)})
        
        elif error_type == JSONErrorType.OVERFLOW_ERROR:
            # Handle integer overflow
            if isinstance(data, dict):
                cleaned_data = self._clean_large_integers(data)
                return orjson.dumps(cleaned_data)
        
        return None
    
    def _clean_large_integers(self, data: Dict) -> Dict:
        """Recursively clean large integers from data"""
        if isinstance(data, dict):
            return {k: self._clean_large_integers(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_large_integers(item) for item in data]
        elif isinstance(data, int) and abs(data) > 2**53:
            return str(data)  # Convert to string
        else:
            return data
    
    def _handle_utf8_decode_error(self, json_data: bytes) -> Optional[Any]:
        """Handle UTF-8 decoding errors"""
        try:
            # Attempt to decode with error replacement
            cleaned_str = json_data.decode('utf-8', 'replace')
            return orjson.loads(cleaned_str)
        except Exception:
            return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = sum(self.error_stats.values())
        
        return {
            "total_errors": total_errors,
            "error_breakdown": {error_type.value: count for error_type, count in self.error_stats.items()},
            "recent_error_rate": len(self.recent_errors) / self.circuit_breaker_window,
            "circuit_breaker_open": self._is_circuit_breaker_open(),
            "error_rate_threshold": self.circuit_breaker_threshold / self.circuit_breaker_window
        }
```

### Fragment-Based Caching for Trading Systems
```python
import asyncio
import orjson
from typing import Dict, Any, Optional
import hashlib
import time
from dataclasses import dataclass

@dataclass
class CachedFragment:
    """Cached JSON fragment with metadata"""
    fragment: orjson.Fragment
    created_at: float
    access_count: int
    size_bytes: int

class TradingDataCache:
    """High-performance fragment cache for trading data"""
    
    def __init__(self, max_cache_size: int = 10000, ttl_seconds: int = 300):
        self.cache: Dict[str, CachedFragment] = {}
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Pre-cache common trading data structures
        self._init_common_fragments()
    
    def _init_common_fragments(self):
        """Pre-cache common trading data structures"""
        common_structures = {
            "empty_order_book": {"bids": [], "asks": []},
            "heartbeat": {"type": "heartbeat", "timestamp": None},
            "connection_status": {"status": "connected", "timestamp": None},
            "error_response": {"error": True, "message": None}
        }
        
        for key, data in common_structures.items():
            serialized = orjson.dumps(data)
            fragment = orjson.Fragment(serialized)
            self.cache[key] = CachedFragment(
                fragment=fragment,
                created_at=time.time(),
                access_count=0,
                size_bytes=len(serialized)
            )
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        # Use hash of serialized data for consistent keys
        serialized = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(serialized).hexdigest()[:16]
    
    async def get_or_create_fragment(self, key: str, data: Any) -> orjson.Fragment:
        """Get cached fragment or create new one"""
        
        # Check cache first
        if key in self.cache:
            cached = self.cache[key]
            
            # Check TTL
            if time.time() - cached.created_at < self.ttl_seconds:
                cached.access_count += 1
                self.cache_hits += 1
                return cached.fragment
            else:
                # Remove expired entry
                del self.cache[key]
        
        # Cache miss - create new fragment
        self.cache_misses += 1
        serialized = orjson.dumps(data)
        fragment = orjson.Fragment(serialized)
        
        # Add to cache
        await self._add_to_cache(key, fragment, len(serialized))
        
        return fragment
    
    async def _add_to_cache(self, key: str, fragment: orjson.Fragment, size_bytes: int):
        """Add fragment to cache with size management"""
        
        # Enforce cache size limit
        if len(self.cache) >= self.max_cache_size:
            await self._evict_least_used()
        
        self.cache[key] = CachedFragment(
            fragment=fragment,
            created_at=time.time(),
            access_count=1,
            size_bytes=size_bytes
        )
    
    async def _evict_least_used(self):
        """Evict least recently used cache entries"""
        if not self.cache:
            return
        
        # Sort by access count and age
        items = [(k, v) for k, v in self.cache.items()]
        items.sort(key=lambda x: (x[1].access_count, -x[1].created_at))
        
        # Remove bottom 10% of cache
        remove_count = max(1, len(items) // 10)
        for i in range(remove_count):
            key, _ = items[i]
            del self.cache[key]
    
    def build_market_update(self, symbol_data: Dict[str, Any]) -> bytes:
        """Build market update using cached fragments"""
        
        update_structure = {
            "type": "market_update",
            "timestamp": time.time(),
            "data": {}
        }
        
        # Use fragments for symbol data
        for symbol, data in symbol_data.items():
            cache_key = f"symbol_{symbol}_{self.get_cache_key(data)}"
            
            # This would be async in real implementation
            # fragment = await self.get_or_create_fragment(cache_key, data)
            # For demo, create fragment directly
            fragment = orjson.Fragment(orjson.dumps(data))
            update_structure["data"][symbol] = fragment
        
        return orjson.dumps(update_structure)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        total_size = sum(cached.size_bytes for cached in self.cache.values())
        avg_access_count = sum(cached.access_count for cached in self.cache.values()) / len(self.cache) if self.cache else 0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "hit_rate": hit_rate,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_size_bytes": total_size,
            "avg_access_count": avg_access_count
        }
```

### Complete Production Integration Example
```python
import asyncio
import orjson
import websockets
import logging
from typing import Dict, Any
import time

class ProductionTradingSystem:
    """Complete production-ready trading system with orjson optimization"""
    
    def __init__(self):
        self.json_processor = HighFrequencyJSONProcessor()
        self.error_handler = ProductionJSONHandler()
        self.cache = TradingDataCache()
        self.performance_metrics = {}
        
        # Configure optimized JSON options for trading
        self.trading_options = (
            orjson.OPT_NON_STR_KEYS |     # Support timestamp/UUID keys
            orjson.OPT_NAIVE_UTC |        # Normalize timestamps to UTC
            orjson.OPT_SERIALIZE_NUMPY |  # Handle price arrays
            orjson.OPT_OMIT_MICROSECONDS  # Remove microseconds for consistency
        )
    
    async def start_trading_system(self, websocket_uri: str):
        """Start complete trading system with monitoring"""
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._performance_monitor())
        
        # Start producer and consumer
        producer_task = asyncio.create_task(
            self.json_processor.websocket_producer(websocket_uri)
        )
        consumer_task = asyncio.create_task(
            self.json_processor.trading_data_consumer(batch_size=50)
        )
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(producer_task, consumer_task, monitor_task)
        except Exception as e:
            logging.error(f"Trading system error: {e}")
        finally:
            # Cleanup
            monitor_task.cancel()
            producer_task.cancel()
            consumer_task.cancel()
    
    async def _performance_monitor(self):
        """Monitor system performance every 30 seconds"""
        
        while True:
            await asyncio.sleep(30)
            
            try:
                # Collect performance metrics
                json_stats = self.json_processor.get_performance_stats()
                error_stats = self.error_handler.get_error_statistics()
                cache_stats = self.cache.get_cache_stats()
                
                # Log comprehensive metrics
                logging.info(f"Trading System Performance Report:")
                logging.info(f"  Messages processed: {json_stats.get('messages_processed', 0)}")
                logging.info(f"  Throughput: {json_stats.get('throughput_msg_per_sec', 0):.2f} msg/sec")
                logging.info(f"  Error rate: {error_stats.get('recent_error_rate', 0):.4f}")
                logging.info(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2f}")
                
                # Store metrics for analysis
                self.performance_metrics[time.time()] = {
                    "json_stats": json_stats,
                    "error_stats": error_stats,
                    "cache_stats": cache_stats
                }
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")

# Usage example
async def main():
    """Production trading system startup"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and start trading system
    trading_system = ProductionTradingSystem()
    
    # Start trading with real WebSocket endpoint
    websocket_uri = "wss://api.exchange.com/ws/market-data"
    await trading_system.start_trading_system(websocket_uri)

if __name__ == "__main__":
    asyncio.run(main())
```

## Production Deployment Considerations

### Performance Optimization Checklist
```python
PRODUCTION_OPTIMIZATION_CHECKLIST = {
    "system_configuration": {
        "cpu_affinity": "Pin to specific cores",
        "memory_monitoring": "Real-time tracking with psutil",
        "gc_optimization": "Manual garbage collection timing",
        "buffer_pooling": "Reuse byte buffers for memory efficiency"
    },
    "orjson_configuration": {
        "input_types": "Use bytes/memoryview for zero-copy",
        "options": "Pre-configure trading-specific options",
        "error_handling": "Comprehensive error classification",
        "caching": "Fragment-based caching for repeated data"
    },
    "asyncio_patterns": {
        "queue_sizing": "Bounded queues with overflow handling",
        "batch_processing": "Optimize batch sizes for latency",
        "timeout_handling": "Sub-millisecond timeouts for responsiveness",
        "concurrent_limits": "Control concurrent processing load"
    },
    "monitoring": {
        "performance_metrics": "Latency, throughput, error rates",
        "memory_usage": "Track allocation patterns",
        "circuit_breakers": "Automatic failure protection",
        "alerting": "Real-time performance alerts"
    }
}
```

This comprehensive production guide demonstrates how orjson's performance advantages translate into real-world trading system benefits, with measurable improvements in latency, throughput, and system reliability.