# AsyncPG API Reference - Connection & Core Functions

**Source**: https://magicstack.github.io/asyncpg/current/api/index.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - Neon Connection Management for Ray Workers

## Connection Function

### `async connect(dsn=None, *, host=None, port=None, user=None, password=None, ...)`

A coroutine to establish a connection to a PostgreSQL server.

**Key Parameters for Phase 4:**
- **dsn** – Connection URI in libpq format: `postgres://user:password@host:port/database?option=value`
- **host** – Database host address (IP, domain, or Unix socket path)
- **port** – Port number (default: 5432)
- **user** – Database role for authentication
- **password** – Password (can be callable for dynamic passwords)
- **database** – Database name to connect to
- **timeout** (float) – Connection timeout in seconds (default: 60)
- **statement_cache_size** (int) – Prepared statement LRU cache size (default: 100)
- **max_cached_statement_lifetime** (int) – Max time statements stay cached in seconds (default: 300)
- **command_timeout** (float) – Default timeout for operations (default: None)
- **ssl** – SSL configuration ('prefer', 'require', 'verify-ca', 'verify-full', or SSLContext)
- **server_settings** (dict) – Optional server runtime parameters

**Critical for Neon Integration:**
```python
# Neon connection with SSL and optimized settings
conn = await asyncpg.connect(
    dsn='postgresql://user:password@neon-host/dbname?sslmode=require',
    statement_cache_size=100,
    command_timeout=30,
    server_settings={
        'application_name': 'quant_trading_ray_worker'
    }
)
```

**SSL Configuration for Neon:**
- Default: `'prefer'` - try SSL first, fallback to non-SSL
- For Neon: Use `'require'` or higher for security
- Programmatic SSL context for advanced configuration

## Connection Class

### `class Connection(protocol, transport, loop, addr, config, params)`

A representation of a database session. Connections are created by calling `connect()`.

### Core Methods for Phase 4

#### Connection Management
- `async close(*, timeout=None)` – Close connection gracefully
- `add_termination_listener(callback)` – Add listener for connection close events

#### Query Execution
- `async execute(query: str, *args, timeout: float = None) → str`
  - Execute SQL command(s)
  - Returns status of last SQL command
  - Can execute multiple commands when no arguments provided

- `async executemany(command: str, args, *, timeout: float = None)`
  - Execute SQL command for each sequence of arguments
  - Efficient for bulk operations

#### Data Retrieval
- `async fetch(query, *args, timeout=None, record_class=None)` – Fetch all query results
- `async fetchrow(query, *args, timeout=None, record_class=None)` – Fetch single row
- `async fetchval(query, *args, column=0, timeout=None)` – Fetch single value

#### Transaction Management
- `transaction()` – Create transaction context manager
```python
async with connection.transaction():
    await connection.execute("INSERT INTO table VALUES($1)", value)
```

#### Bulk Operations (Critical for Trading Data)
- `async copy_records_to_table(table_name, *, records, columns=None, schema_name=None, timeout=None, where=None)`
  - Copy list of records using binary COPY
  - Extremely efficient for bulk inserts
  - Supports asynchronous iterables

```python
# Efficient bulk insert for OHLCV data
await conn.copy_records_to_table(
    'ohlcv_bars',
    records=[
        (symbol, timestamp, open_val, high_val, low_val, close_val, volume)
        for symbol, timestamp, open_val, high_val, low_val, close_val, volume in ohlcv_data
    ],
    columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
)
```

#### Cursors (For Large Result Sets)
- `cursor(query, *args, prefetch=None, timeout=None, record_class=None)`
  - Returns cursor factory for memory-efficient iteration
  - Useful for processing large datasets

#### Type Codecs
- `async set_type_codec(typename, *, encoder, decoder, schema='public', format='text')`
- `async set_builtin_type_codec(typename, *, codec_name)`

## Error Handling Patterns

### Connection Errors
```python
import asyncpg.exceptions

try:
    conn = await asyncpg.connect(dsn)
except asyncpg.exceptions.ConnectionError as e:
    # Handle connection failures
    logger.error(f"Failed to connect to database: {e}")
    # Implement retry logic or fallback
```

### Query Errors  
```python
try:
    result = await conn.fetchval("SELECT value FROM table WHERE id = $1", record_id)
except asyncpg.exceptions.PostgresError as e:
    # Handle PostgreSQL errors
    logger.error(f"Query failed: {e}")
```

## Connection String Examples for Neon

### Basic Neon Connection
```python
# Basic Neon connection
dsn = "postgresql://user:password@ep-example-123456.us-east-2.aws.neon.tech/neondb?sslmode=require"
```

### Advanced Neon Configuration
```python
# Advanced configuration for Ray workers
dsn = "postgresql://user:password@ep-example-123456.us-east-2.aws.neon.tech/neondb" \
      "?sslmode=require" \
      "&application_name=quant_trading_ray_worker" \
      "&connect_timeout=30"

conn = await asyncpg.connect(
    dsn,
    statement_cache_size=100,
    max_cached_statement_lifetime=300,
    command_timeout=30
)
```

## Performance Optimization Notes

1. **Statement Caching**: Default cache size of 100 is good for most use cases
2. **Connection Timeout**: Set appropriate timeout for cloud deployments (30-60s)
3. **Command Timeout**: Set for long-running queries (e.g., backtesting operations)
4. **Bulk Operations**: Use `copy_records_to_table` for high-volume inserts
5. **Connection Reuse**: Always use connection pools in production

## Phase 4 Integration Checklist

- ✅ SSL configuration for Neon security requirements
- ✅ Connection pooling for Ray worker efficiency  
- ✅ Bulk copy operations for OHLCV data ingestion
- ✅ Transaction management for data consistency
- ✅ Error handling and connection recovery
- ✅ Performance tuning for cloud database latency