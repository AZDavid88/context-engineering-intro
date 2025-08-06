# DuckDB Data Ingestion Patterns

**Source URL**: https://duckdb.org/docs/stable/clients/python/data_ingestion
**Extraction Date**: 2025-01-26
**Content Quality**: ✅ HIGH - Complete data ingestion reference with DataFrame integration

## Overview

DuckDB provides comprehensive data ingestion capabilities essential for quant trading systems:
- **File-based ingestion** (CSV, Parquet, JSON)
- **Direct DataFrame access** (Pandas, Polars, Arrow)
- **Real-time data stream integration**
- **Zero-copy data access** for maximum performance

## File-Based Data Ingestion

### CSV File Patterns
```python
import duckdb

# Single file with auto-detection
duckdb.read_csv("trades.csv")

# Multiple files with glob patterns
duckdb.read_csv("daily_data/*.csv")

# Custom formatting options
duckdb.read_csv("market_data.csv", 
                header=False, 
                sep="|",
                dtype=["varchar", "double", "bigint", "timestamp"])

# Direct SQL file access
duckdb.sql("SELECT * FROM 'trades.csv'")
duckdb.sql("SELECT * FROM read_csv('custom_format.csv')")
```

### Parquet File Patterns
```python
# Single and multiple Parquet files
duckdb.read_parquet("ohlcv_data.parquet")
duckdb.read_parquet("historical_data/*.parquet")

# Remote data access
duckdb.read_parquet("https://data.provider.com/market_data.parquet")

# Multiple file lists
duckdb.read_parquet([
    "2023_data.parquet", 
    "2024_data.parquet", 
    "2025_data.parquet"
])

# Direct SQL integration
duckdb.sql("SELECT * FROM 'daily_ohlcv.parquet'")
```

### JSON Data Ingestion
```python
# JSON file reading with auto-detection
duckdb.read_json("trade_events.json")
duckdb.read_json("event_stream/*.json")

# Direct SQL access
duckdb.sql("SELECT * FROM 'market_events.json'")
duckdb.sql("SELECT * FROM read_json_auto('complex_data.json')")
```

## DataFrame Integration Patterns

### Pandas DataFrame Integration
```python
import duckdb
import pandas as pd

# Direct DataFrame querying (zero-copy access)
market_data = pd.DataFrame({
    "symbol": ["BTC", "ETH", "SOL"],
    "price": [50000, 3000, 100],
    "volume": [1000, 2000, 5000],
    "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"])
})

# Query DataFrame directly by variable name
portfolio_summary = duckdb.sql("""
    SELECT 
        symbol,
        price,
        volume,
        price * volume as market_value
    FROM market_data
    WHERE price > 1000
    ORDER BY market_value DESC
""").df()

print(portfolio_summary)
```

### Polars DataFrame Integration
```python
import polars as pl
import duckdb

# Create Polars DataFrame
crypto_prices = pl.DataFrame({
    "symbol": ["BTC", "ETH", "ADA"],
    "price": [45000.0, 2800.0, 0.85],
    "change_24h": [0.05, -0.02, 0.12]
})

# Direct querying of Polars DataFrame
trending_cryptos = duckdb.sql("""
    SELECT 
        symbol,
        price,
        change_24h,
        price * (1 + change_24h) as price_24h_ago
    FROM crypto_prices
    WHERE change_24h > 0.1
""").pl()  # Return as Polars DataFrame
```

### PyArrow Integration
```python
import pyarrow as pa
import duckdb

# Arrow Table creation
arrow_data = pa.Table.from_pydict({
    "trade_id": [1, 2, 3, 4],
    "symbol": ["BTC", "BTC", "ETH", "ETH"],
    "quantity": [0.1, 0.5, 2.0, 1.5],
    "price": [50000, 49500, 3000, 3100]
})

# Query Arrow Table directly
trade_analysis = duckdb.sql("""
    SELECT 
        symbol,
        COUNT(*) as trade_count,
        SUM(quantity * price) as total_value,
        AVG(price) as avg_price
    FROM arrow_data
    GROUP BY symbol
""").arrow()  # Return as Arrow Table
```

## Real-Time Data Stream Integration

### Live Data Processing Pattern
```python
import duckdb
import pandas as pd
from datetime import datetime, timedelta

class RealTimeMarketProcessor:
    def __init__(self, db_path="live_trading.db"):
        self.con = duckdb.connect(db_path)
        self.setup_tables()
        
    def setup_tables(self):
        """Initialize tables for real-time data."""
        self.con.sql("""
            CREATE TABLE IF NOT EXISTS live_prices (
                symbol VARCHAR,
                price DOUBLE,
                volume BIGINT,
                timestamp TIMESTAMP,
                bid DOUBLE,
                ask DOUBLE
            )
        """)
        
    def process_live_feed(self, market_data_df):
        """Process incoming market data DataFrame."""
        # Direct DataFrame access - no copying required
        technical_signals = self.con.sql("""
            SELECT 
                symbol,
                price,
                timestamp,
                AVG(price) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS 19 PRECEDING
                ) as sma_20,
                (price - AVG(price) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS 19 PRECEDING
                )) / STDDEV(price) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS 19 PRECEDING
                ) as price_zscore
            FROM market_data_df
            WHERE timestamp >= now() - INTERVAL 1 HOUR
            ORDER BY timestamp DESC
        """).df()
        
        return technical_signals
        
    def store_historical(self, df):
        """Store processed data for historical analysis."""
        self.con.register("temp_data", df)
        self.con.sql("""
            INSERT INTO live_prices 
            SELECT * FROM temp_data
        """)
```

### Streaming Data Aggregation
```python
def process_tick_data_stream(tick_stream):
    """Process high-frequency tick data into OHLCV bars."""
    
    # Convert tick stream to DataFrame
    tick_df = pd.DataFrame(tick_stream)
    
    # Real-time OHLCV aggregation using DuckDB
    ohlcv_bars = duckdb.sql("""
        SELECT 
            symbol,
            date_trunc('minute', timestamp) as bar_time,
            first(price ORDER BY timestamp) as open,
            max(price) as high,
            min(price) as low,
            last(price ORDER BY timestamp) as close,
            sum(volume) as volume,
            count(*) as tick_count
        FROM tick_df
        GROUP BY symbol, date_trunc('minute', timestamp)
        ORDER BY symbol, bar_time
    """).df()
    
    return ohlcv_bars
```

## Advanced Data Registration Patterns

### Manual DataFrame Registration
```python
import duckdb

# Data stored in complex structures
trading_data = {
    "portfolio": pd.DataFrame({
        "symbol": ["BTC", "ETH", "ADA"],
        "quantity": [1.5, 10.0, 1000.0],
        "avg_cost": [45000, 2500, 0.75]
    }),
    "current_prices": pd.DataFrame({
        "symbol": ["BTC", "ETH", "ADA"], 
        "price": [50000, 3000, 0.80]
    })
}

con = duckdb.connect()

# Register DataFrames as virtual tables
con.register("portfolio_positions", trading_data["portfolio"])
con.register("market_prices", trading_data["current_prices"])

# Query across registered tables
portfolio_pnl = con.sql("""
    SELECT 
        p.symbol,
        p.quantity,
        p.avg_cost,
        m.price as current_price,
        (m.price - p.avg_cost) * p.quantity as unrealized_pnl,
        ((m.price - p.avg_cost) / p.avg_cost) * 100 as pnl_percent
    FROM portfolio_positions p
    JOIN market_prices m ON p.symbol = m.symbol
    ORDER BY unrealized_pnl DESC
""").df()
```

### Creating Persistent Tables from DataFrames
```python
# Convert DataFrame to persistent DuckDB table
con.execute("CREATE TABLE historical_trades AS SELECT * FROM live_trade_df")

# Insert additional data from DataFrame
con.execute("INSERT INTO historical_trades SELECT * FROM new_trades_df")

# Hybrid approach: DataFrame + SQL processing
def update_portfolio_metrics(portfolio_df, price_df):
    con = duckdb.connect("trading.db")
    
    # Register current data
    con.register("current_portfolio", portfolio_df)
    con.register("current_prices", price_df) 
    
    # Update persistent metrics table
    con.sql("""
        INSERT OR REPLACE INTO portfolio_metrics
        SELECT 
            current_timestamp as calculation_time,
            p.symbol,
            p.quantity,
            pr.price,
            p.quantity * pr.price as market_value
        FROM current_portfolio p
        JOIN current_prices pr ON p.symbol = pr.symbol
    """)
    
    return con.sql("SELECT * FROM portfolio_metrics ORDER BY calculation_time DESC").df()
```

## Object Column Handling (Pandas-specific)

### Managing Complex Data Types
```python
# Configure sample size for object column analysis
duckdb.execute("SET GLOBAL pandas_analyze_sample = 100_000")

# Handle mixed-type object columns
complex_df = pd.DataFrame({
    "trade_id": [1, 2, 3],
    "metadata": [
        {"exchange": "binance", "fees": 0.001},
        {"exchange": "coinbase", "fees": 0.005}, 
        {"exchange": "kraken", "fees": 0.002}
    ],
    "tags": [["spot", "btc"], ["futures", "eth"], ["spot", "ada"]]
})

# DuckDB will analyze and convert object columns appropriately
trade_summary = duckdb.sql("""
    SELECT 
        trade_id,
        metadata,
        tags,
        len(tags) as tag_count
    FROM complex_df
""").df()
```

## Performance Optimization for Data Ingestion

### Batch Processing Pattern
```python
def efficient_bulk_insert(file_paths, table_name, con):
    """Efficient pattern for bulk data ingestion."""
    
    # Use DuckDB's native bulk loading
    for file_path in file_paths:
        if file_path.endswith('.parquet'):
            con.sql(f"""
                INSERT INTO {table_name}
                SELECT * FROM read_parquet('{file_path}')
            """)
        elif file_path.endswith('.csv'):
            con.sql(f"""
                INSERT INTO {table_name}
                SELECT * FROM read_csv('{file_path}')
            """)
```

### Zero-Copy Integration Benefits
```python
def analyze_large_dataset_efficiently():
    """Demonstrate zero-copy benefits for large datasets."""
    
    # Large dataset in memory (no copying to DuckDB)
    large_df = pd.read_parquet("massive_trading_data.parquet")
    
    # DuckDB operates directly on DataFrame memory
    # No data copying occurs - just metadata reference
    analysis = duckdb.sql(f"""
        SELECT 
            symbol,
            date_trunc('day', timestamp) as trading_day,
            COUNT(*) as trade_count,
            SUM(volume * price) as total_value
        FROM large_df
        WHERE timestamp >= current_date - INTERVAL 30 DAYS
        GROUP BY symbol, trading_day
        ORDER BY total_value DESC
        LIMIT 100
    """).df()
    
    return analysis
```

This data ingestion documentation provides the foundation for building efficient, high-performance data pipelines that can handle both batch and real-time data processing requirements for quant trading systems.