# DuckDB Concurrency and Connection Management

**Source URL**: https://duckdb.org/docs/stable/connect/concurrency
**Extraction Date**: 2025-01-26
**Content Quality**: ✅ HIGH - Complete concurrency documentation with implementation patterns

## Concurrency Model Overview

DuckDB uses a **single-writer, multiple-reader** concurrency model with two main configurations:

1. **Single Process Mode**: One process can read and write to the database
2. **Read-Only Mode**: Multiple processes can read, but no writes allowed (`access_mode = 'READ_ONLY'`)

### Key Design Principles
- **MVCC (Multi-Version Concurrency Control)** for transaction isolation
- **Optimistic Concurrency Control** for conflict resolution
- **Memory-centric design** for faster analytical queries
- **Bulk operation optimization** rather than many small transactions

## Concurrency Within a Single Process

### Thread Safety Rules
```python
import duckdb
import threading

# ❌ INCORRECT: Sharing connection across threads
con = duckdb.connect("trading.db")

def worker_thread():
    # This is NOT thread-safe
    con.sql("INSERT INTO trades VALUES (...)")

# ✅ CORRECT: Create cursor per thread
con = duckdb.connect("trading.db")

def worker_thread():
    cursor = con.cursor()  # Thread-safe cursor
    cursor.execute("INSERT INTO trades VALUES (...)")
    cursor.close()
```

### Concurrent Write Patterns
```sql
-- ✅ SAFE: Concurrent appends to different tables
-- Thread 1
INSERT INTO market_data_btc SELECT * FROM live_feed_btc;

-- Thread 2  
INSERT INTO market_data_eth SELECT * FROM live_feed_eth;

-- ✅ SAFE: Concurrent appends to same table (no conflicts)
-- Multiple threads can append simultaneously
INSERT INTO trade_log VALUES (timestamp, symbol, quantity, price);

-- ✅ SAFE: Concurrent updates to different rows
-- Thread 1
UPDATE portfolio SET quantity = quantity + 100 WHERE symbol = 'BTC';

-- Thread 2
UPDATE portfolio SET quantity = quantity - 50 WHERE symbol = 'ETH';
```

### Optimistic Concurrency Control
```python
import duckdb
import time
from threading import Thread

def update_portfolio_position(connection, symbol, quantity_change):
    try:
        with connection.begin():  # Start transaction
            # Read current position
            current_qty = connection.sql(f"""
                SELECT quantity FROM portfolio WHERE symbol = '{symbol}'
            """).fetchone()[0]
            
            # Calculate new position
            new_qty = current_qty + quantity_change
            
            # Update position
            connection.sql(f"""
                UPDATE portfolio 
                SET quantity = {new_qty}, last_updated = now()
                WHERE symbol = '{symbol}'
            """)
            
    except Exception as e:
        if "Transaction conflict" in str(e):
            print(f"Conflict detected for {symbol}, retrying...")
            time.sleep(0.01)  # Brief backoff
            return update_portfolio_position(connection, symbol, quantity_change)
        else:
            raise e
```

## Multi-Process Access Patterns

### Read-Only Multi-Process Pattern
```python
# Process 1: Writer process
writer_con = duckdb.connect("trading_data.db")
writer_con.sql("CREATE TABLE IF NOT EXISTS prices (symbol VARCHAR, price DOUBLE, timestamp TIMESTAMP)")

# Process 2-N: Reader processes  
reader_con = duckdb.connect("trading_data.db", config={'access_mode': 'READ_ONLY'})
current_prices = reader_con.sql("SELECT * FROM prices WHERE timestamp >= now() - INTERVAL 1 HOUR").df()
```

### Mutex-Based Multi-Writer Pattern
```python
import fcntl
import duckdb
import time

class MutexDuckDBWriter:
    def __init__(self, db_path, lock_path):
        self.db_path = db_path
        self.lock_path = lock_path
        
    def execute_with_lock(self, query, retries=3):
        for attempt in range(retries):
            try:
                with open(self.lock_path, 'w') as lock_file:
                    # Acquire exclusive lock
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Open connection and execute
                    with duckdb.connect(self.db_path) as con:
                        result = con.sql(query)
                        con.commit()
                        return result
                        
            except BlockingIOError:
                # Another process has the lock
                wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
                continue
                
        raise Exception(f"Failed to acquire lock after {retries} attempts")

# Usage
writer = MutexDuckDBWriter("trading.db", "trading.db.lock")
writer.execute_with_lock("INSERT INTO trades VALUES (now(), 'BTC', 100, 50000)")
```

### Retry-Based Multi-Writer Pattern
```python
import duckdb
import time
import random

def safe_database_operation(db_path, operation_func, max_retries=5):
    """Execute database operation with automatic retry on connection conflicts."""
    
    for attempt in range(max_retries):
        try:
            with duckdb.connect(db_path) as con:
                return operation_func(con)
                
        except Exception as e:
            if "database is locked" in str(e).lower() or "busy" in str(e).lower():
                # Add jitter to prevent thundering herd
                wait_time = (0.1 * (2 ** attempt)) + random.uniform(0, 0.1)
                time.sleep(wait_time)
                continue
            else:
                raise e
                
    raise Exception(f"Database operation failed after {max_retries} attempts")

# Usage example
def insert_trade_data(con):
    return con.sql("""
        INSERT INTO trades 
        SELECT * FROM read_parquet('new_trades.parquet')
    """)

# Multiple processes can safely call this
result = safe_database_operation("trading.db", insert_trade_data)
```

## Alternative Architectures for High Concurrency

### Web Server Pattern
```python
from fastapi import FastAPI
import duckdb
import asyncio

app = FastAPI()

# Single connection managed by web server
db_connection = duckdb.connect("trading_data.db")

@app.post("/insert_trade")
async def insert_trade(trade_data: dict):
    # All requests go through single connection
    result = db_connection.sql(f"""
        INSERT INTO trades VALUES (
            '{trade_data['timestamp']}',
            '{trade_data['symbol']}', 
            {trade_data['quantity']},
            {trade_data['price']}
        )
    """)
    return {"status": "success"}

@app.get("/get_portfolio")
async def get_portfolio():
    portfolio = db_connection.sql("""
        SELECT symbol, SUM(quantity) as total_quantity
        FROM trades 
        GROUP BY symbol
    """).df()
    return portfolio.to_dict('records')
```

### External Database Pattern
```python
# Use DuckDB for analytics, external DB for transactions
import duckdb
import psycopg2  # PostgreSQL for concurrent writes

def sync_transactional_to_analytical():
    # PostgreSQL for high-concurrency writes
    pg_con = psycopg2.connect("postgresql://user:pass@localhost/trading")
    
    # DuckDB for analytics
    duck_con = duckdb.connect("analytics.db")
    
    # Periodic sync from transactional to analytical store
    duck_con.sql("""
        CREATE OR REPLACE TABLE daily_trades AS
        SELECT * FROM postgres_scan_pushdown(
            'postgresql://user:pass@localhost/trading',
            'trades', 
            'timestamp >= current_date'
        )
    """)
    
    # Run analytics on DuckDB
    analysis = duck_con.sql("""
        SELECT 
            symbol,
            COUNT(*) as trade_count,
            AVG(price) as avg_price,
            SUM(quantity) as total_volume
        FROM daily_trades
        GROUP BY symbol
        ORDER BY total_volume DESC
    """).df()
    
    return analysis
```

## Best Practices for Trading Systems

### 1. Connection Pooling for Read Workloads
```python
class DuckDBReadPool:
    def __init__(self, db_path, pool_size=4):
        self.connections = [
            duckdb.connect(db_path, config={'access_mode': 'READ_ONLY'})
            for _ in range(pool_size)
        ]
        self.current = 0
        
    def get_connection(self):
        con = self.connections[self.current]
        self.current = (self.current + 1) % len(self.connections)
        return con
```

### 2. Write Batching Pattern
```python
import threading
import queue
import time

class BatchedWriter:
    def __init__(self, db_path, batch_size=1000, flush_interval=5):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._batch_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def _batch_worker(self):
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Get item with timeout
                item = self.queue.get(timeout=1.0)
                batch.append(item)
                
                # Flush conditions
                should_flush = (
                    len(batch) >= self.batch_size or
                    time.time() - last_flush >= self.flush_interval
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except queue.Empty:
                # Timeout - flush if we have items
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
    def _flush_batch(self, batch):
        with duckdb.connect(self.db_path) as con:
            # Insert batch efficiently
            con.sql("INSERT INTO trades VALUES (?, ?, ?, ?)", batch)
            
    def write_async(self, trade_record):
        self.queue.put(trade_record)
```

This concurrency documentation provides the foundation for implementing thread-safe and process-safe data access patterns critical for production quant trading systems.