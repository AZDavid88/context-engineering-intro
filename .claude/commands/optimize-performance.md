---
allowed-tools: Bash(python:*), Bash(pip:*), Write, Read, Grep, Glob
description: Find and fix performance bottlenecks with systematic optimization
argument-hint: [component-name] | full system optimization if no component specified
---

# Performance Optimization & Bottleneck Analysis

**Context**: You are using the CodeFarm methodology for systematic performance optimization. This identifies and resolves performance bottlenecks while maintaining code quality and functionality.

## Performance Assessment Setup

### 1. Baseline Measurement
Establish current performance baselines using Claude Code tools:

**System information:**
- Use Bash tool with command: `python -c "import platform, psutil; print(f'Python: {platform.python_version()}, CPU: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total//1024//1024//1024}GB')"` to get system specs

**Target component:** ${ARGUMENTS:-"full-system"}

**Performance tools availability:**
- Use Bash tool with command: `python -c "import sys; modules=['cProfile', 'line_profiler', 'memory_profiler', 'psutil']; [print(f'{m}: Available') if m in sys.modules or __import__(m) else print(f'{m}: Missing') for m in modules]"` to check available profiling tools

### 2. Profiling Environment Setup
Install and configure profiling tools using Bash tool:

```bash
# Install performance profiling tools
pip install line-profiler memory-profiler psutil py-spy

# Create profiling directory
mkdir -p performance_analysis
mkdir -p performance_analysis/profiles
mkdir -p performance_analysis/reports
```

### 3. Test Data Generation
Create realistic test data for profiling:

```python
# performance_test_data.py
import random
import string
from typing import List, Dict, Any

def generate_test_data(size: str = "medium") -> Dict[str, Any]:
    """Generate test data for performance testing"""
    
    sizes = {
        "small": {"records": 100, "string_length": 10},
        "medium": {"records": 1000, "string_length": 50}, 
        "large": {"records": 10000, "string_length": 100},
        "xlarge": {"records": 100000, "string_length": 200}
    }
    
    config = sizes.get(size, sizes["medium"])
    
    # Generate test records
    test_data = {
        "users": [
            {
                "id": i,
                "name": ''.join(random.choices(string.ascii_letters, k=config["string_length"])),
                "email": f"user{i}@example.com",
                "data": {
                    "score": random.randint(1, 100),
                    "items": [random.randint(1, 1000) for _ in range(10)]
                }
            }
            for i in range(config["records"])
        ],
        "metadata": {
            "generated_at": "2025-01-01",
            "size": size,
            "record_count": config["records"]
        }
    }
    
    return test_data

if __name__ == "__main__":
    # Generate test data files
    for size in ["small", "medium", "large"]:
        data = generate_test_data(size)
        with open(f"performance_analysis/test_data_{size}.json", "w") as f:
            import json
            json.dump(data, f, indent=2)
    
    print("Test data generated successfully")
```

## Performance Profiling

### 4. CPU Profiling
Identify CPU-intensive operations:

```python
# cpu_profiler.py
import cProfile
import pstats
import io
from typing import Callable, Any

class CPUProfiler:
    """CPU performance profiler"""
    
    def __init__(self, output_dir: str = "performance_analysis/profiles"):
        self.output_dir = output_dir
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a specific function"""
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile execution
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Generate report
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        report = stats_stream.getvalue()
        
        # Save detailed profile
        func_name = func.__name__
        profiler.dump_stats(f"{self.output_dir}/cpu_{func_name}.prof")
        
        return {
            "result": result,
            "profile_report": report,
            "profile_file": f"cpu_{func_name}.prof"
        }
    
    def profile_module(self, module_path: str):
        """Profile entire module execution"""
        
        profiler = cProfile.Profile()
        
        # Execute module with profiling
        profiler.run(f"exec(open('{module_path}').read())")
        
        # Generate report
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Save results
        module_name = module_path.split('/')[-1].replace('.py', '')
        profiler.dump_stats(f"{self.output_dir}/module_{module_name}.prof")
        
        # Print top functions
        print(f"\n=== CPU Profile: {module_name} ===")
        stats.print_stats(10)

# Usage example
def example_slow_function(data_size: int = 10000):
    """Example function that can be optimized"""
    # Simulate CPU-intensive work
    result = []
    for i in range(data_size):
        # Inefficient string concatenation (example bottleneck)
        temp = ""
        for j in range(100):
            temp += str(j)
        result.append(temp)
    return result

if __name__ == "__main__":
    profiler = CPUProfiler()
    
    # Profile the slow function
    profile_result = profiler.profile_function(example_slow_function, 1000)
    print("CPU profiling completed")
    print(profile_result["profile_report"])
```

### 5. Memory Profiling
Track memory usage and identify leaks:

```python
# memory_profiler.py
import psutil
import gc
import tracemalloc
from typing import Callable, Any, Dict

class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a function"""
        
        # Start memory tracing
        tracemalloc.start()
        gc.collect()  # Clean up before measurement
        
        # Measure memory before
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure memory after
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Get memory trace
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "result": result,
            "memory_used_mb": memory_after - memory_before,
            "peak_memory_mb": peak / 1024 / 1024,
            "current_memory_mb": current / 1024 / 1024,
            "total_memory_mb": memory_after
        }
    
    def find_memory_leaks(self, func: Callable, iterations: int = 10):
        """Detect potential memory leaks"""
        
        memory_usage = []
        
        for i in range(iterations):
            gc.collect()
            memory_before = self.process.memory_info().rss / 1024 / 1024
            
            # Execute function
            func()
            
            gc.collect()
            memory_after = self.process.memory_info().rss / 1024 / 1024
            
            memory_usage.append(memory_after)
            
            if i > 0:
                growth = memory_after - memory_usage[0]
                if growth > 10:  # More than 10MB growth
                    print(f"⚠ Potential memory leak detected: {growth:.2f}MB growth after {i} iterations")
        
        return memory_usage

# Example memory-intensive function
def memory_intensive_function():
    """Example function with memory usage to profile"""
    # Create large data structures
    large_list = [i * 2 for i in range(100000)]
    large_dict = {i: str(i) * 100 for i in range(10000)}
    
    # Simulate some processing
    result = sum(large_list) + len(large_dict)
    
    # Memory should be released when function exits
    return result

if __name__ == "__main__":
    profiler = MemoryProfiler()
    
    # Profile memory usage
    result = profiler.profile_memory_usage(memory_intensive_function)
    print(f"Memory usage: {result['memory_used_mb']:.2f}MB")
    print(f"Peak memory: {result['peak_memory_mb']:.2f}MB")
    
    # Check for memory leaks
    print("\nChecking for memory leaks...")
    profiler.find_memory_leaks(memory_intensive_function)
```

### 6. I/O Performance Analysis
Identify I/O bottlenecks:

```python
# io_profiler.py
import time
import os
from pathlib import Path
from typing import Dict, Any

class IOProfiler:
    """I/O performance profiler"""
    
    def profile_file_operations(self, file_path: str, data_size: int = 1000) -> Dict[str, Any]:
        """Profile file I/O operations"""
        
        results = {}
        test_data = "x" * 1000  # 1KB of data
        
        # Write performance
        start_time = time.time()
        with open(file_path, 'w') as f:
            for _ in range(data_size):
                f.write(test_data)
        write_time = time.time() - start_time
        
        # Read performance
        start_time = time.time()
        with open(file_path, 'r') as f:
            content = f.read()
        read_time = time.time() - start_time
        
        # File size
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        
        # Cleanup
        os.remove(file_path)
        
        return {
            "write_time_s": write_time,
            "read_time_s": read_time,
            "file_size_mb": file_size_mb,
            "write_speed_mbs": file_size_mb / write_time,
            "read_speed_mbs": file_size_mb / read_time
        }
    
    def profile_database_operations(self, connection, query: str, iterations: int = 100):
        """Profile database query performance"""
        
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            # Execute query (adapt to your database)
            cursor = connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "avg_time_ms": (sum(times) / len(times)) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "total_time_s": sum(times)
        }

if __name__ == "__main__":
    profiler = IOProfiler()
    
    # Test file I/O
    results = profiler.profile_file_operations("test_io.txt", 1000)
    print(f"File I/O Performance:")
    print(f"  Write: {results['write_speed_mbs']:.2f} MB/s")
    print(f"  Read: {results['read_speed_mbs']:.2f} MB/s")
```

## Optimization Implementation

### 7. Common Optimization Patterns
Apply systematic optimizations:

#### Algorithm Optimization
```python
# algorithm_optimization.py
from typing import List, Dict, Any
import time

class AlgorithmOptimizer:
    """Common algorithm optimizations"""
    
    @staticmethod
    def optimize_loops(data: List[Any]) -> List[Any]:
        """Optimize loop performance"""
        
        # BEFORE (inefficient)
        def slow_processing(items):
            result = []
            for item in items:
                if item is not None:
                    processed = str(item).upper()
                    if len(processed) > 5:
                        result.append(processed)
            return result
        
        # AFTER (optimized)
        def fast_processing(items):
            # List comprehension with filtering
            return [
                str(item).upper() 
                for item in items 
                if item is not None and len(str(item)) > 5
            ]
        
        return fast_processing(data)
    
    @staticmethod
    def optimize_data_structures(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize data structure usage"""
        
        # BEFORE (inefficient - repeated searches)
        def slow_lookup(items, target_id):
            for item in items:
                if item.get('id') == target_id:
                    return item
            return None
        
        # AFTER (optimized - use dictionary for O(1) lookup)
        def fast_lookup(items, target_id):
            lookup_dict = {item['id']: item for item in items if 'id' in item}
            return lookup_dict.get(target_id)
        
        # Demonstrate performance difference
        test_data = [{"id": i, "value": f"item_{i}"} for i in range(10000)]
        
        # Time slow approach
        start = time.time()
        result1 = slow_lookup(test_data, 9999)
        slow_time = time.time() - start
        
        # Time fast approach  
        start = time.time()
        result2 = fast_lookup(test_data, 9999)
        fast_time = time.time() - start
        
        return {
            "slow_time_ms": slow_time * 1000,
            "fast_time_ms": fast_time * 1000,
            "speedup_factor": slow_time / fast_time if fast_time > 0 else 0
        }

# String optimization
class StringOptimizer:
    """String operation optimizations"""
    
    @staticmethod
    def optimize_string_building(items: List[str]) -> str:
        """Optimize string concatenation"""
        
        # BEFORE (inefficient - creates new string each time)
        def slow_concat(strings):
            result = ""
            for s in strings:
                result += s + " "
            return result.strip()
        
        # AFTER (optimized - use join)
        def fast_concat(strings):
            return " ".join(strings)
        
        return fast_concat(items)
    
    @staticmethod
    def optimize_string_formatting(name: str, age: int, score: float) -> str:
        """Optimize string formatting"""
        
        # BEFORE (slower)
        slow_format = "Name: " + name + ", Age: " + str(age) + ", Score: " + str(score)
        
        # AFTER (faster)
        fast_format = f"Name: {name}, Age: {age}, Score: {score}"
        
        return fast_format

if __name__ == "__main__":
    optimizer = AlgorithmOptimizer()
    
    # Test algorithm optimization
    test_data = list(range(10000))
    result = optimizer.optimize_data_structures([{"id": i, "value": i} for i in test_data])
    print(f"Algorithm optimization: {result['speedup_factor']:.2f}x speedup")
```

### 8. Database Optimization
Optimize database interactions:

```python
# database_optimization.py
import asyncio
from typing import List, Dict, Any

class DatabaseOptimizer:
    """Database performance optimizations"""
    
    @staticmethod
    def optimize_queries():
        """Common database query optimizations"""
        
        optimizations = {
            "Use indexes": {
                "before": "SELECT * FROM users WHERE email = 'user@example.com'",
                "after": "CREATE INDEX idx_users_email ON users(email); SELECT * FROM users WHERE email = 'user@example.com'",
                "improvement": "100-1000x faster lookups"
            },
            
            "Batch operations": {
                "before": "Multiple individual INSERT statements",
                "after": "INSERT INTO users (name, email) VALUES ('John', 'john@ex.com'), ('Jane', 'jane@ex.com')",
                "improvement": "10-100x faster inserts"
            },
            
            "Limit results": {
                "before": "SELECT * FROM large_table",
                "after": "SELECT * FROM large_table LIMIT 100 OFFSET 0",
                "improvement": "Prevents memory overflow"
            },
            
            "Select specific columns": {
                "before": "SELECT * FROM users",
                "after": "SELECT id, name, email FROM users",
                "improvement": "Reduces network transfer"
            }
        }
        
        return optimizations
    
    @staticmethod
    async def optimize_connection_pooling():
        """Connection pooling optimization example"""
        
        # BEFORE (inefficient - new connection each time)
        async def slow_database_calls():
            results = []
            for i in range(10):
                # Create new connection each time (slow)
                connection = await create_connection()
                result = await connection.execute("SELECT * FROM users LIMIT 1")
                results.append(result)
                await connection.close()
            return results
        
        # AFTER (optimized - use connection pool)
        async def fast_database_calls():
            # Use connection pool
            pool = await create_connection_pool(min_size=5, max_size=20)
            
            tasks = []
            for i in range(10):
                async def query():
                    async with pool.acquire() as connection:
                        return await connection.execute("SELECT * FROM users LIMIT 1")
                tasks.append(query())
            
            results = await asyncio.gather(*tasks)
            await pool.close()
            return results
        
        return "Connection pooling configured"

async def create_connection():
    """Mock connection creation"""
    await asyncio.sleep(0.1)  # Simulate connection time
    return MockConnection()

async def create_connection_pool(min_size, max_size):
    """Mock connection pool"""
    return MockConnectionPool()

class MockConnection:
    async def execute(self, query):
        await asyncio.sleep(0.01)  # Simulate query time
        return {"result": "data"}
    
    async def close(self):
        pass

class MockConnectionPool:
    def acquire(self):
        return MockConnection()
    
    async def close(self):
        pass
```

### 9. Caching Implementation
Add strategic caching for performance:

```python
# caching_optimization.py
import time
import functools
from typing import Any, Dict, Optional

class CacheOptimizer:
    """Caching strategies for performance optimization"""
    
    def __init__(self):
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def simple_cache(self, key: str, value: Any = None) -> Any:
        """Simple in-memory cache"""
        
        if value is not None:
            # Store value
            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
            return value
        else:
            # Retrieve value
            if key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[key]["value"]
            else:
                self.cache_stats["misses"] += 1
                return None
    
    def time_based_cache(self, key: str, ttl_seconds: int = 300) -> Optional[Any]:
        """Cache with time-to-live expiration"""
        
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() - cache_entry["timestamp"] < ttl_seconds:
                self.cache_stats["hits"] += 1
                return cache_entry["value"]
            else:
                # Expired - remove from cache
                del self.cache[key]
        
        self.cache_stats["misses"] += 1
        return None
    
    @staticmethod
    def memoize_decorator(ttl_seconds: int = 300):
        """Decorator for function memoization"""
        
        def decorator(func):
            cache = {}
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = str(args) + str(sorted(kwargs.items()))
                
                # Check cache
                if key in cache:
                    cached_result, timestamp = cache[key]
                    if time.time() - timestamp < ttl_seconds:
                        return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache[key] = (result, time.time())
                
                return result
            
            return wrapper
        return decorator

# Example usage
@CacheOptimizer.memoize_decorator(ttl_seconds=600)
def expensive_calculation(n: int) -> int:
    """Simulate expensive calculation"""
    time.sleep(0.1)  # Simulate work
    return sum(i * i for i in range(n))

# Performance comparison
def demonstrate_caching_benefits():
    """Show caching performance improvements"""
    
    print("=== Caching Performance Demo ===")
    
    # Without caching
    start_time = time.time()
    for _ in range(5):
        result = sum(i * i for i in range(10000))  # Expensive operation
    no_cache_time = time.time() - start_time
    
    # With caching
    start_time = time.time()
    for _ in range(5):
        result = expensive_calculation(10000)  # Cached after first call
    cache_time = time.time() - start_time
    
    print(f"Without caching: {no_cache_time:.3f}s")
    print(f"With caching: {cache_time:.3f}s")
    print(f"Speedup: {no_cache_time / cache_time:.2f}x")

if __name__ == "__main__":
    demonstrate_caching_benefits()
```

## Performance Validation

### 10. Benchmark Comparison
Measure optimization improvements:

```python
# performance_benchmark.py
import time
import statistics
from typing import Callable, List, Dict, Any

class PerformanceBenchmark:
    """Benchmark and compare performance improvements"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_function(self, func: Callable, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
        """Benchmark a function's performance"""
        
        times = []
        
        # Warm up
        for _ in range(10):
            func(*args, **kwargs)
        
        # Benchmark
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            "avg_time_ms": statistics.mean(times) * 1000,
            "median_time_ms": statistics.median(times) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_dev_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0
        }
    
    def compare_implementations(self, old_func: Callable, new_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Compare old vs new implementation performance"""
        
        print("Benchmarking old implementation...")
        old_results = self.benchmark_function(old_func, *args, **kwargs)
        
        print("Benchmarking new implementation...")
        new_results = self.benchmark_function(new_func, *args, **kwargs)
        
        # Calculate improvement
        speedup = old_results["avg_time_ms"] / new_results["avg_time_ms"]
        improvement_percent = ((old_results["avg_time_ms"] - new_results["avg_time_ms"]) / old_results["avg_time_ms"]) * 100
        
        return {
            "old_performance": old_results,
            "new_performance": new_results,
            "speedup_factor": speedup,
            "improvement_percent": improvement_percent,
            "verdict": "✅ Improved" if speedup > 1.1 else "⚠ Marginal" if speedup > 0.95 else "❌ Slower"
        }

# Generate optimization report
def generate_optimization_report():
    """Generate comprehensive optimization report"""
    
    report = f"""
# Performance Optimization Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total optimizations applied**: [count]
- **Average performance improvement**: [percentage]%
- **Critical bottlenecks resolved**: [count]

## Key Improvements
### Algorithm Optimizations
- Data structure improvements: [details]
- Loop optimizations: [details]
- String operation improvements: [details]

### I/O Optimizations
- Database query optimization: [details]
- File I/O improvements: [details]
- Network request optimization: [details]

### Memory Optimizations
- Memory usage reduction: [details]
- Garbage collection improvements: [details]
- Memory leak fixes: [details]

## Performance Metrics
### Before Optimization
- Average response time: [time]ms
- Memory usage: [memory]MB
- CPU utilization: [percentage]%

### After Optimization
- Average response time: [time]ms
- Memory usage: [memory]MB
- CPU utilization: [percentage]%

### Improvement Summary
- Response time improvement: [percentage]%
- Memory reduction: [percentage]%
- CPU efficiency gain: [percentage]%

## Recommendations
### Immediate Actions
- [ ] [High-priority optimization]
- [ ] [Performance monitoring setup]

### Future Optimizations
- [ ] [Advanced optimization opportunities]
- [ ] [Architectural improvements]

## Monitoring Setup
- Performance monitoring enabled: [status]
- Alert thresholds configured: [status]
- Regular performance reviews scheduled: [status]
"""
    
    return report

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    
    # Example comparison
    def old_slow_function(n):
        return sum([i * i for i in range(n)])
    
    def new_fast_function(n):
        return sum(i * i for i in range(n))
    
    comparison = benchmark.compare_implementations(old_slow_function, new_fast_function, 10000)
    print(f"Performance comparison: {comparison['verdict']}")
    print(f"Speedup: {comparison['speedup_factor']:.2f}x")
    print(f"Improvement: {comparison['improvement_percent']:.1f}%")
```

### 11. Success Criteria
Optimization complete when:

- [ ] Performance bottlenecks identified and resolved
- [ ] Benchmarks show measurable improvement
- [ ] Memory usage optimized
- [ ] Database queries optimized
- [ ] Caching implemented where beneficial
- [ ] Performance monitoring established
- [ ] Code maintains functionality and quality
- [ ] Optimization documentation updated

---

This systematic performance optimization process ensures measurable improvements while maintaining code quality and system reliability.