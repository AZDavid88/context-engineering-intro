# DuckDB Parquet Integration Documentation

**Source URL**: https://duckdb.org/docs/stable/data/parquet/overview
**Extraction Date**: 2025-01-26  
**Content Quality**: ✅ HIGH - Complete Parquet integration reference with performance optimization

## Overview

DuckDB provides comprehensive support for reading and writing Parquet files with advanced features:
- **Columnar compression** for efficient storage and processing
- **Filter and projection pushdown** for optimized queries
- **Metadata extraction** and schema inference
- **Parallel processing** of multiple files
- **Encryption support** for sensitive data

## Reading Parquet Files

### Basic Reading Operations
```sql
-- Single file reading
SELECT * FROM 'test.parquet';
SELECT * FROM read_parquet('test.parquet');

-- Schema inspection
DESCRIBE SELECT * FROM 'test.parquet';

-- Multiple files
SELECT * FROM read_parquet(['file1.parquet', 'file2.parquet', 'file3.parquet']);

-- Glob patterns
SELECT * FROM 'test/*.parquet';
SELECT * FROM read_parquet(['folder1/*.parquet', 'folder2/*.parquet']);
```

### Advanced Reading Features
```sql
-- Include filename information (automatic in DuckDB v1.3.0+)
SELECT *, filename FROM read_parquet('test/*.parquet');

-- Remote file access
SELECT * FROM read_parquet('https://some.url/some_file.parquet');

-- Metadata queries
SELECT * FROM parquet_metadata('test.parquet');
SELECT * FROM parquet_file_metadata('test.parquet');
SELECT * FROM parquet_kv_metadata('test.parquet');
SELECT * FROM parquet_schema('test.parquet');
```

### `read_parquet` Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `binary_as_string` | `BOOL` | `false` | Load binary columns as strings for legacy compatibility |
| `encryption_config` | `STRUCT` | - | Configuration for Parquet encryption |
| `filename` | `BOOL` | `false` | Include filename column (legacy, now automatic) |
| `file_row_number` | `BOOL` | `false` | Include file row number column |
| `hive_partitioning` | `BOOL` | auto-detect | Interpret path as Hive partitioned |
| `union_by_name` | `BOOL` | `false` | Unify schemas by name vs position |

## Writing Parquet Files

### Basic Writing Operations
```sql
-- Default Snappy compression
COPY (SELECT * FROM tbl) TO 'result-snappy.parquet' (FORMAT parquet);

-- Custom compression and row group size
COPY (FROM generate_series(100_000)) TO 'test.parquet' 
(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 100_000);

-- Various compression algorithms
COPY tbl TO 'result-zstd.parquet' (FORMAT parquet, COMPRESSION zstd);
COPY tbl TO 'result-lz4.parquet' (FORMAT parquet, COMPRESSION lz4);
COPY tbl TO 'result-brotli.parquet' (FORMAT parquet, COMPRESSION brotli);
COPY tbl TO 'result-uncompressed.parquet' (FORMAT parquet, COMPRESSION uncompressed);
```

### Advanced Writing Features
```sql
-- Compression levels
COPY tbl TO 'result-zstd.parquet' 
(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 1);

-- Key-value metadata
COPY (SELECT 42 AS number, true AS is_even) TO 'kv_metadata.parquet' (
    FORMAT parquet,
    KV_METADATA {
        number: 'Answer to life, universe, and everything',
        is_even: 'not ''odd''' -- Escaped single quotes
    }
);

-- Dictionary page size configuration
COPY lineitem TO 'lineitem-with-custom-dictionary-size.parquet'
(FORMAT parquet, STRING_DICTIONARY_PAGE_SIZE_LIMIT 100_000);

-- Export entire database
EXPORT DATABASE 'target_directory' (FORMAT parquet);
```

## Performance Optimization Features

### Projection and Filter Pushdown
```sql
-- DuckDB automatically pushes down:
-- 1. Column selection (projection pushdown)
SELECT symbol, close_price FROM 'large_dataset.parquet';

-- 2. Filter conditions (filter pushdown with zonemaps)
SELECT * FROM 'large_dataset.parquet' 
WHERE timestamp >= '2024-01-01' AND symbol = 'BTCUSD';
```

### Parallel Processing
```sql
-- Multiple files processed in parallel automatically
SELECT * FROM 'data_partitions/*.parquet';

-- Union multiple datasets
SELECT * FROM read_parquet(['2023/*.parquet', '2024/*.parquet']);
```

## Data Management Patterns

### Creating Tables and Views
```sql
-- Create table from Parquet
CREATE TABLE historical_prices AS 
SELECT * FROM read_parquet('historical_data.parquet');

-- Insert from Parquet
INSERT INTO existing_table 
SELECT * FROM read_parquet('new_data.parquet');

-- Create view for direct querying
CREATE VIEW live_data AS 
SELECT * FROM read_parquet('live_feed/*.parquet');
```

### File Organization Best Practices
```sql
-- Partitioned writing for better query performance
COPY (
    SELECT *, date_trunc('day', timestamp) as partition_date
    FROM market_data 
) TO 'data/year=2024' (FORMAT parquet, PARTITION_BY partition_date);

-- Time-based partitioning
COPY market_data TO 'data' (
    FORMAT parquet, 
    PARTITION_BY (date_trunc('day', timestamp))
);
```

## Integration with Trading Data Pipeline

### OHLCV Data Storage
```sql
-- Optimized schema for trading data
CREATE TABLE ohlcv_staging AS
SELECT 
    symbol,
    timestamp,
    open_price,
    high_price, 
    low_price,
    close_price,
    volume,
    date_trunc('day', timestamp) as trading_date
FROM read_parquet('raw_market_data/*.parquet');

-- Write with optimal compression for time-series data
COPY ohlcv_staging TO 'processed_data/ohlcv.parquet' (
    FORMAT parquet,
    COMPRESSION zstd,
    COMPRESSION_LEVEL 3,
    ROW_GROUP_SIZE 1000000
);
```

### Incremental Data Updates
```sql
-- Efficient incremental loading pattern
INSERT INTO historical_ohlcv
SELECT * FROM read_parquet('daily_updates/*.parquet')
WHERE timestamp > (SELECT MAX(timestamp) FROM historical_ohlcv);
```

### Query Performance Optimization
```sql
-- Leverage filter pushdown for time-range queries
SELECT 
    symbol,
    timestamp,
    close_price,
    LAG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close
FROM 'time_series_data/*.parquet'
WHERE timestamp BETWEEN '2024-01-01' AND '2024-01-31'
  AND symbol IN ('BTCUSD', 'ETHUSD')
ORDER BY symbol, timestamp;
```

## Key Benefits for Quant Trading Implementation

1. **Storage Efficiency**: Columnar compression reduces storage costs by 5-10x
2. **Query Performance**: Filter/projection pushdown minimizes I/O operations  
3. **Schema Evolution**: Flexible schema handling for evolving data formats
4. **Parallel Processing**: Automatic parallelization across multiple files
5. **Cloud Integration**: Direct access to remote files via HTTPS/S3
6. **Metadata Rich**: Built-in metadata extraction for data lineage and debugging

This Parquet integration provides the foundation for efficient time-series data storage and retrieval essential for high-performance quant trading systems.