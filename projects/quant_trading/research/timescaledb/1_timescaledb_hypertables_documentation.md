# TimescaleDB/TigerData - Hypertables and Time-Series Best Practices

**Source**: https://docs.timescale.com/use-timescale/latest/hypertables/  
**Extracted**: 2025-07-25  
**Purpose**: Time-series data storage and query optimization for trading system metrics

## Overview

TimescaleDB (now TigerData) is a Postgres extension designed for time-series data that automatically partitions data by time. Tiger Cloud enhances Postgres with real-time analytics, AI capabilities, and hybrid applications support.

## Key Concepts

### Hypertables

Hypertables are PostgreSQL tables in TimescaleDB that automatically partition time-series data by time. They consist of:

- **Child tables called chunks**: Each assigned a specific time range
- **Automatic partitioning**: Data divided by time intervals (default: 7 days)
- **Query optimization**: TimescaleDB identifies correct chunks for queries
- **Full SQL support**: Works with standard PostgreSQL features

### Hypercore Storage Engine

Hypercore provides hybrid row-columnar storage with dynamic optimization:

- **Row-based storage**: For recent data (fast inserts, updates, low-latency queries)
- **Columnar storage**: For analytical performance (up to 95% compression)
- **Automatic transitions**: Data moves from row to columnar based on lifecycle
- **Real-time flexibility**: Insert/modify data at any stage

## Architecture Benefits

### Comparison: Relational Table vs Hypertable

**Traditional Relational Table:**
- Single large table
- All data in one structure
- Full table scans for time-range queries
- Index maintenance overhead

**Hypertable:**
- Automatically partitioned chunks
- Each chunk covers specific time range
- Query only relevant chunks
- Distributed index maintenance

![Hypertable structure shows automatic partitioning by time](architecture_diagram)

## Best Practices for Scaling and Performance

### 1. Chunk Size Optimization

**Critical Rule**: Set `chunk_interval` so that indexes for actively ingested chunks fit within 25% of main memory.

**Example Calculations:**
- **64 GB Memory**: 25% = 16 GB available for chunk indexes
- **2 GB index growth/day**: Use 7-day chunks (14 GB total)
- **10 GB index growth/day**: Use 1-day chunks (10 GB total)

**Configuration:**
```sql
-- Set chunk interval during hypertable creation
SELECT create_hypertable('trading_data', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Modify existing hypertable
SELECT set_chunk_time_interval('trading_data', INTERVAL '6 hours');
```

### 2. Chunk Time Interval Guidelines

| Data Volume | Index Growth | Recommended Interval |
|-------------|-------------|---------------------|
| Low | <1 GB/day | 7 days (default) |
| Medium | 1-5 GB/day | 2-3 days |
| High | 5-10 GB/day | 1 day |
| Very High | >10 GB/day | 6-12 hours |

### 3. Hypertable Limitations

- **Limit hypertables**: Avoid tens of thousands of hypertables per service
- **Memory constraints**: Keep active chunk indexes in memory (25% rule)
- **Query planning**: Too many small chunks slow down planning
- **Compression efficiency**: Sparse chunks affect compression ratios

## Index Strategy

### Default Indexes

TimescaleDB automatically creates:
1. **Time index** (descending): For time-range queries
2. **Space + Time index**: If using space partitioning

### Custom Index Creation

```sql
-- Create index on commonly queried columns
CREATE INDEX ON trading_data (symbol, timestamp DESC);

-- Create composite index for complex queries
CREATE INDEX ON trading_data (strategy_id, timestamp DESC, pnl);

-- Prevent default index creation
SELECT create_hypertable('trading_data', 'timestamp', create_default_indexes => FALSE);
```

### Unique Constraints

**Important**: Unique indexes on hypertables must include ALL partitioning columns.

```sql
-- Correct: includes timestamp (partitioning column)
CREATE UNIQUE INDEX ON trading_data (trade_id, timestamp);

-- Incorrect: missing timestamp
CREATE UNIQUE INDEX ON trading_data (trade_id); -- This will fail
```

## Implementation for Quant Trading System

### 1. Trading Data Schema

```sql
-- Create hypertable for OHLCV data
CREATE TABLE market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume DECIMAL(20, 8),
    source TEXT
);

SELECT create_hypertable('market_data', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for common queries
CREATE INDEX ON market_data (symbol, timestamp DESC);
CREATE INDEX ON market_data (timestamp DESC, volume);
```

### 2. Strategy Performance Metrics

```sql
-- Strategy performance tracking
CREATE TABLE strategy_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    generation INTEGER,
    sharpe_ratio DECIMAL(10, 4),
    total_return DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    trade_count INTEGER,
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(10, 4)
);

SELECT create_hypertable('strategy_metrics', 'timestamp', chunk_time_interval => INTERVAL '6 hours');

-- Optimize for strategy evolution queries
CREATE INDEX ON strategy_metrics (strategy_id, timestamp DESC);
CREATE INDEX ON strategy_metrics (generation, sharpe_ratio DESC);
```

### 3. Real-time Trade Execution

```sql
-- Trade execution log
CREATE TABLE trade_log (
    timestamp TIMESTAMPTZ NOT NULL,
    trade_id UUID NOT NULL,
    strategy_id TEXT,
    symbol TEXT,
    side TEXT, -- 'buy' or 'sell'
    quantity DECIMAL(20, 8),
    price DECIMAL(20, 8),
    fee DECIMAL(20, 8),
    execution_time_ms INTEGER,
    order_type TEXT,
    fill_status TEXT
);

-- High-frequency data requires smaller chunks
SELECT create_hypertable('trade_log', 'timestamp', chunk_time_interval => INTERVAL '2 hours');

-- Indexes for order tracking and analysis
CREATE UNIQUE INDEX ON trade_log (trade_id, timestamp);
CREATE INDEX ON trade_log (strategy_id, timestamp DESC);
CREATE INDEX ON trade_log (symbol, timestamp DESC);
```

### 4. Risk Management Events

```sql
-- Risk management and circuit breaker events
CREATE TABLE risk_events (
    timestamp TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL, -- 'stop_loss', 'position_limit', 'drawdown_limit'
    strategy_id TEXT,
    symbol TEXT,
    triggered_value DECIMAL(20, 8),
    threshold_value DECIMAL(20, 8),
    action_taken TEXT,
    severity TEXT
);

SELECT create_hypertable('risk_events', 'timestamp', chunk_time_interval => INTERVAL '1 day');
CREATE INDEX ON risk_events (event_type, timestamp DESC);
CREATE INDEX ON risk_events (strategy_id, severity, timestamp DESC);
```

## Advanced Features for Trading Systems

### 1. Continuous Aggregates

Pre-compute common aggregations for faster analytics:

```sql
-- 1-minute OHLCV aggregation
CREATE MATERIALIZED VIEW market_data_1min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data
GROUP BY bucket, symbol;

-- Strategy performance hourly aggregation
CREATE MATERIALIZED VIEW strategy_performance_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    strategy_id,
    avg(sharpe_ratio) AS avg_sharpe,
    max(total_return) AS max_return,
    min(max_drawdown) AS worst_drawdown,
    count(*) AS sample_count
FROM strategy_metrics
GROUP BY bucket, strategy_id;
```

### 2. Data Retention Policies

Automatically manage data lifecycle:

```sql
-- Keep detailed trade data for 30 days
SELECT add_retention_policy('trade_log', INTERVAL '30 days');

-- Keep market data for 1 year
SELECT add_retention_policy('market_data', INTERVAL '1 year');

-- Keep aggregated data longer
SELECT add_retention_policy('strategy_performance_hourly', INTERVAL '3 years');
```

### 3. Compression for Historical Data

Enable automatic compression for older chunks:

```sql
-- Enable compression after 7 days
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Compression for strategy metrics
ALTER TABLE strategy_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'strategy_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('strategy_metrics', INTERVAL '2 days');
```

## Query Optimization Patterns

### 1. Time-Range Queries

```sql
-- Efficient: Uses chunk exclusion
SELECT * FROM market_data 
WHERE timestamp >= '2025-07-01' 
  AND timestamp < '2025-07-02'
  AND symbol = 'BTC-USD';

-- Inefficient: Forces full table scan
SELECT * FROM market_data 
WHERE extract(hour from timestamp) = 14; -- Avoid functions on time column
```

### 2. Latest Value Queries

```sql
-- Use time-descending index
SELECT DISTINCT ON (symbol) 
    symbol, timestamp, close
FROM market_data 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY symbol, timestamp DESC;
```

### 3. Aggregation Queries

```sql
-- Use time_bucket for regular intervals
SELECT 
    time_bucket('5 minutes', timestamp) AS bucket,
    symbol,
    avg(close) AS avg_price,
    stddev(close) AS volatility
FROM market_data 
WHERE timestamp >= NOW() - INTERVAL '1 day'
GROUP BY bucket, symbol
ORDER BY bucket DESC;
```

## Performance Monitoring

### 1. Chunk Analysis

```sql
-- View chunk information
SELECT 
    chunk_schema,
    chunk_name,
    range_start,
    range_end,
    pg_size_pretty(compressed_total_size) AS compressed_size,
    pg_size_pretty(uncompressed_total_size) AS uncompressed_size
FROM chunk_detail_view 
WHERE hypertable_name = 'market_data'
ORDER BY range_start DESC;
```

### 2. Query Performance

```sql
-- Analyze query execution
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM market_data 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
  AND symbol = 'BTC-USD';
```

### 3. Index Usage

```sql
-- Monitor index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename LIKE '%market_data%';
```

## Integration with Trading System Architecture

### VPN Zone Considerations

- **TimescaleDB location**: Can be in non-VPN zone (90% of system)
- **Data ingestion**: Receive data from VPN zone via message queue
- **Query interface**: Direct connection for analytics and backtesting
- **Backup strategy**: Regular backups to cloud storage

### Scaling Strategy

1. **Start Simple**: Single TimescaleDB instance with appropriate chunk sizing
2. **Monitor Growth**: Track index sizes and query performance
3. **Scale Vertically**: Increase memory and CPU as data grows
4. **Scale Horizontally**: Consider read replicas for analytical workloads
5. **Data Tiering**: Move old data to cheaper storage (S3 integration)

### Disaster Recovery

```sql
-- Point-in-time recovery setup
SELECT set_config('log_statement', 'all', false);
SELECT set_config('wal_level', 'replica', false);

-- Backup strategy
pg_dump --format=custom --no-owner --no-privileges trading_db > backup.dump
```

This TimescaleDB/TigerData implementation provides a robust foundation for storing and analyzing time-series trading data with optimal performance characteristics for our genetic algorithm trading system.