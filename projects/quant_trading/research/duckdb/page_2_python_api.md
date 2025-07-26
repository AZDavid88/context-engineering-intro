# DuckDB Python API Documentation

**Source URL**: https://duckdb.org/docs/stable/clients/python/overview  
**Extraction Date**: 2025-01-26
**Content Quality**: ✅ HIGH - Complete Python API reference with examples

## Installation and Requirements

- **Installation**: `pip install duckdb` or `conda install python-duckdb -c conda-forge`
- **Python Version**: Requires Python 3.9 or newer
- **Latest Version**: 1.3.2

## Basic API Usage

### Core Functionality
```python
import duckdb

# Basic query execution using global in-memory database
duckdb.sql("SELECT 42").show()

# Incremental query building with Relations
r1 = duckdb.sql("SELECT 42 AS i")
duckdb.sql("SELECT i * 2 AS k FROM r1").show()
```

### Data Input Methods

#### File-based Input
```python
# CSV, Parquet, JSON file reading
duckdb.read_csv("example.csv")
duckdb.read_parquet("example.parquet") 
duckdb.read_json("example.json")

# Direct SQL file queries
duckdb.sql("SELECT * FROM 'example.csv'")
duckdb.sql("SELECT * FROM 'example.parquet'")
```

#### DataFrame Integration
```python
# Pandas DataFrame support
import pandas as pd
pandas_df = pd.DataFrame({"a": [42]})
duckdb.sql("SELECT * FROM pandas_df")

# Polars DataFrame support  
import polars as pl
polars_df = pl.DataFrame({"a": [42]})
duckdb.sql("SELECT * FROM polars_df")

# PyArrow Table support
import pyarrow as pa
arrow_table = pa.Table.from_pydict({"a": [42]})
duckdb.sql("SELECT * FROM arrow_table")
```

## Result Conversion

### Multiple Output Formats
```python
# Various result formats
duckdb.sql("SELECT 42").fetchall()   # Python objects
duckdb.sql("SELECT 42").df()         # Pandas DataFrame
duckdb.sql("SELECT 42").pl()         # Polars DataFrame  
duckdb.sql("SELECT 42").arrow()      # Arrow Table
duckdb.sql("SELECT 42").fetchnumpy() # NumPy Arrays
```

### Writing Data to Disk
```python
# Export to various formats
duckdb.sql("SELECT 42").write_parquet("out.parquet")
duckdb.sql("SELECT 42").write_csv("out.csv")
duckdb.sql("COPY (SELECT 42) TO 'out.parquet'")
```

## Connection Management

### In-Memory vs Persistent Databases
```python
# In-memory database (default)
con = duckdb.connect()
con.sql("SELECT 42 AS x").show()

# Persistent database
con = duckdb.connect("file.db")
con.sql("CREATE TABLE test (i INTEGER)")
con.sql("INSERT INTO test VALUES (42)")
con.table("test").show()
con.close()

# Context manager for automatic cleanup
with duckdb.connect("file.db") as con:
    con.sql("CREATE TABLE test (i INTEGER)")
    con.sql("INSERT INTO test VALUES (42)")
    con.table("test").show()
```

### Configuration Options
```python
# Custom configuration
con = duckdb.connect(config={'threads': 1})
```

### Thread Safety Considerations
- **Important**: `DuckDBPyConnection` object is NOT thread-safe
- **Solution**: Create separate cursors for each thread using `connection.cursor()`
- **Recommendation**: Create connection objects instead of using global `duckdb` module for packages

## Extensions Management

### Loading Extensions
```python
con = duckdb.connect()
con.install_extension("spatial")
con.load_extension("spatial")

# Community extensions
con.install_extension("h3", repository="community")
con.load_extension("h3")

# Unsigned extensions (requires configuration)
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
```

## Key Implementation Patterns for Quant Trading

### Real-time Data Processing
```python
# Direct DataFrame access for live data feeds
market_data_df = get_live_market_data()  # Your data source
portfolio_metrics = duckdb.sql("""
    SELECT 
        symbol,
        timestamp,
        price,
        volume,
        price - LAG(price) OVER (PARTITION BY symbol ORDER BY timestamp) as price_change
    FROM market_data_df
    ORDER BY timestamp DESC
""").df()
```

### Historical Data Management
```python
# Persistent connection for historical data
with duckdb.connect("trading_database.db") as con:
    # Create OHLCV table
    con.sql("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE, 
            low DOUBLE,
            close DOUBLE,
            volume BIGINT
        )
    """)
    
    # Efficient bulk insert from Parquet
    con.sql("INSERT INTO ohlcv SELECT * FROM 'historical_data/*.parquet'")
```

### Performance-Critical Operations
```python
# Connection reuse for best performance
class TradingDataManager:
    def __init__(self, db_path="trading.db"):
        self.con = duckdb.connect(db_path)
        
    def get_technical_indicators(self, symbol, lookback_days=30):
        return self.con.sql(f"""
            SELECT 
                timestamp,
                close,
                AVG(close) OVER (ORDER BY timestamp ROWS 19 PRECEDING) as sma_20,
                STDDEV(close) OVER (ORDER BY timestamp ROWS 19 PRECEDING) as volatility
            FROM ohlcv 
            WHERE symbol = '{symbol}' 
            AND timestamp >= current_date - INTERVAL {lookback_days} DAYS
            ORDER BY timestamp
        """).df()
```

This Python API provides the foundation for implementing DuckDB as the core analytical engine with efficient data ingestion, processing, and result conversion capabilities essential for quant trading systems.