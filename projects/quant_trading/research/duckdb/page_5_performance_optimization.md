# DuckDB Performance Optimization Guide

**Source URL**: https://duckdb.org/docs/stable/guides/performance/overview and https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads
**Extraction Date**: 2025-01-26
**Content Quality**: ✅ HIGH - Comprehensive performance optimization strategies

## Performance Philosophy

DuckDB is designed to **automatically achieve high performance** with well-chosen defaults and a forgiving architecture. Key principles:

- **Columnar storage** with vectorized execution
- **Automatic parallelization** across multiple cores  
- **Memory-efficient algorithms** with spilling to disk
- **Intelligent query optimization** with cost-based planning

## Memory Management and Out-of-Core Processing

### Preserve Insertion Order Configuration
```sql
-- For large datasets that exceed memory, disable ordering constraints
SET preserve_insertion_order = false;
```

This allows DuckDB to reorder results without `ORDER BY` clauses, potentially reducing memory usage for large import/export operations.

### Spilling to Disk Configuration
```sql
-- Configure temporary directory for larger-than-memory workloads
SET temp_directory = '/path/to/fast_ssd/temp_dir.tmp/';
```

DuckDB automatically creates:
- **Persistent mode**: `database_file_name.tmp` directory
- **In-memory mode**: `.tmp` directory

### Blocking Operators (Memory-Intensive Operations)
These operators require buffering entire input and are most memory-intensive:

1. **Grouping**: `GROUP BY` operations
2. **Joining**: `JOIN` operations  
3. **Sorting**: `ORDER BY` operations
4. **Windowing**: `OVER ... (PARTITION BY ... ORDER BY ...)` operations

```sql
-- Example of memory-intensive query with multiple blocking operators
SELECT 
    symbol,
    AVG(price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 20 PRECEDING) as sma_20,
    SUM(volume) as total_volume
FROM market_data 
WHERE timestamp >= '2024-01-01'
GROUP BY symbol, timestamp
ORDER BY timestamp DESC;
```

## Parallelism Optimization

### Row Group-Based Parallelism
- DuckDB parallelizes based on **row groups** (max 122,880 rows each)
- For _k_ threads, need at least _k_ * 122,880 rows for full parallelization

```sql
-- Check row group distribution
SELECT 
    count(*) as total_rows,
    count(*) / 122880 as estimated_row_groups
FROM large_table;
```

### Thread Management
```sql
-- Manually limit threads if too many are launched (HyperThreading issues)
SET threads = 8;  -- Set to actual CPU cores, not virtual cores

-- For remote files, increase threads for better I/O parallelism
SET threads = 16;  -- 2-5x CPU cores for network-bound workloads
```

### Parallel Query Patterns
```sql
-- Optimize for parallel execution
-- ✅ Good: Hash joins parallelize well
SELECT a.*, b.price
FROM large_table_a a
JOIN large_table_b b ON a.symbol = b.symbol;

-- ❌ Avoid: Nested loop joins don't parallelize
SELECT a.*, (SELECT price FROM table_b WHERE symbol = a.symbol LIMIT 1)
FROM large_table_a a;
```

## Query Optimization Strategies

### Profiling and Analysis
```sql
-- Analyze query execution without running
EXPLAIN SELECT * FROM market_data WHERE symbol = 'BTC';

-- Profile actual query performance
EXPLAIN ANALYZE 
SELECT 
    symbol,
    AVG(price) as avg_price,
    COUNT(*) as trade_count
FROM trades 
WHERE timestamp >= '2024-01-01' 
GROUP BY symbol;
```

### Join Optimization
```sql
-- ✅ Preferred: Hash joins for large datasets
SELECT a.symbol, a.price, b.volume
FROM prices a
JOIN volumes b ON a.symbol = b.symbol AND a.timestamp = b.timestamp;

-- ✅ Filter pushdown optimization
SELECT symbol, price
FROM large_dataset
WHERE timestamp BETWEEN '2024-01-01' AND '2024-01-31'  -- Pushed to scan
  AND price > 100;  -- Also pushed to scan
```

### Prepared Statements for Small Repeated Queries
```python
import duckdb

con = duckdb.connect()

# Prepare statement for repeated execution
stmt = con.prepare("SELECT * FROM trades WHERE symbol = ? AND timestamp >= ?")

# Execute multiple times with different parameters
for symbol in ['BTC', 'ETH', 'SOL']:
    result = stmt.execute([symbol, '2024-01-01']).fetchdf()
    process_symbol_data(result)
```

## Remote File Optimization

### Thread Configuration for Network I/O
```sql
-- Increase threads for remote file access (synchronous I/O limitation)
SET threads = 20;  -- 2-5x CPU cores for better network parallelism
```

### Minimize Unnecessary Data Transfer
```sql
-- ✅ Column pruning - only select needed columns
SELECT symbol, close_price, timestamp
FROM 's3://bucket/market_data.parquet'
WHERE timestamp >= '2024-01-01';

-- ❌ Avoid SELECT * on remote files
SELECT * FROM 's3://bucket/market_data.parquet';  -- Downloads all columns

-- ✅ Effective filtering with partitioned data
SELECT symbol, price
FROM 's3://bucket/data/year=2024/month=01/*.parquet'
WHERE symbol IN ('BTC', 'ETH');
```

### Avoid Multiple Downloads
```sql
-- ❌ Inefficient: Downloads data twice
SELECT AVG(price) FROM 's3://bucket/data.parquet' WHERE symbol = 'BTC';
SELECT MAX(price) FROM 's3://bucket/data.parquet' WHERE symbol = 'BTC';

-- ✅ Efficient: Single download with local caching
CREATE TABLE btc_data AS 
SELECT * FROM 's3://bucket/data.parquet' WHERE symbol = 'BTC';

SELECT AVG(price) FROM btc_data;
SELECT MAX(price) FROM btc_data;
```

## Connection Management Best Practices

### Connection Reuse Pattern
```python
class OptimizedTradingEngine:
    def __init__(self, db_path="trading.db"):
        # Reuse connection for best performance
        self.con = duckdb.connect(db_path)
        
        # Warm up connection with metadata cache
        self.con.sql("SELECT 1").fetchone()
        
    def get_market_data(self, symbol, start_date):
        # Connection reuse avoids overhead
        return self.con.sql(f"""
            SELECT * FROM market_data 
            WHERE symbol = '{symbol}' AND date >= '{start_date}'
        """).df()
        
    def __del__(self):
        if hasattr(self, 'con'):
            self.con.close()
```

### Connection Pool for Concurrent Reads
```python
import threading
from queue import Queue

class DuckDBConnectionPool:
    def __init__(self, db_path, pool_size=4):
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            con = duckdb.connect(db_path, config={'access_mode': 'READ_ONLY'})
            self.pool.put(con)
            
    def get_connection(self):
        return self.pool.get()
        
    def return_connection(self, con):
        self.pool.put(con)
        
    def execute_query(self, query):
        con = self.get_connection()
        try:
            result = con.sql(query).df()
            return result
        finally:
            self.return_connection(con)
```

## Storage and Compression Optimization

### Persistent vs In-Memory Performance
```sql
-- Persistent databases benefit from compression
-- Create persistent database for better performance on large datasets
.open trading_data.db

CREATE TABLE ohlcv AS 
SELECT * FROM read_parquet('large_dataset.parquet');

-- In-memory databases lack compression
-- May be slower for large datasets despite being "in memory"
```

### Optimal File Formats
```sql
-- ✅ Parquet: Best for analytical queries
COPY (SELECT * FROM trading_data) 
TO 'optimized_data.parquet' (FORMAT parquet, COMPRESSION zstd);

-- ✅ Row group sizing for parallelism
COPY large_dataset TO 'parallel_optimized.parquet' (
    FORMAT parquet,
    COMPRESSION zstd,
    ROW_GROUP_SIZE 1000000  -- Optimize for parallel processing
);
```

## Time-Series Specific Optimizations

### Window Function Optimization
```sql
-- ✅ Efficient window function usage
SELECT 
    symbol,
    timestamp,
    price,
    -- Partition by symbol for parallel processing
    AVG(price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as sma_20
FROM market_data
-- Filter first to reduce window function input
WHERE timestamp >= '2024-01-01'
ORDER BY symbol, timestamp;
```

### Time-Based Partitioning
```sql
-- Partition data by time for better query performance
CREATE TABLE daily_ohlcv (
    symbol VARCHAR,
    trading_date DATE,
    open_price DOUBLE,
    high_price DOUBLE,
    low_price DOUBLE, 
    close_price DOUBLE,
    volume BIGINT
) PARTITION BY (trading_date);

-- Queries with date filters will scan fewer partitions
SELECT * FROM daily_ohlcv 
WHERE trading_date = '2024-01-15' AND symbol = 'BTC';
```

## Memory Usage Monitoring
```sql
-- Monitor memory usage during development
SELECT 
    database_size,
    total_blocks,
    used_blocks,
    free_blocks
FROM pragma_database_size();

-- Check query memory usage
EXPLAIN ANALYZE SELECT * FROM large_computation;
```

## Performance Testing Framework
```python
import time
import duckdb
import pandas as pd

def benchmark_query(con, query, iterations=5):
    """Benchmark query performance with multiple runs."""
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        result = con.sql(query).fetchdf()
        end_time = time.time()
        times.append(end_time - start_time)
        
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'rows_returned': len(result)
    }

# Usage
con = duckdb.connect("trading.db")
perf_stats = benchmark_query(con, """
    SELECT symbol, AVG(price) as avg_price
    FROM market_data 
    WHERE timestamp >= '2024-01-01'
    GROUP BY symbol
""")
print(f"Query completed in avg {perf_stats['avg_time']:.3f}s")
```

This performance optimization guide provides the foundation for building high-performance quant trading systems that can handle large-scale data processing with optimal resource utilization.