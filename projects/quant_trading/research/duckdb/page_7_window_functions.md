# DuckDB Window Functions for Time-Series Analysis

**Source URL**: https://duckdb.org/docs/stable/sql/functions/window_functions
**Extraction Date**: 2025-01-26
**Content Quality**: ✅ HIGH - Complete window functions reference with time-series focus

## Overview

Window functions are essential for time-series analysis in quant trading, enabling:
- **Technical indicator calculations** (moving averages, RSI, Bollinger Bands)
- **Price change analysis** (returns, volatility, momentum)
- **Ranking and percentile analysis** (performance ranking, outlier detection)
- **Sequential data processing** (gaps, sequences, pattern detection)

> **Performance Note**: Window functions are **blocking operators** requiring entire input to be buffered, making them memory-intensive. DuckDB supports larger-than-memory processing for all window functions.

## Core Window Function Categories

### 1. Sequential Access Functions
```sql
-- Price change analysis
SELECT 
    symbol,
    timestamp,
    price,
    price - lag(price) OVER (PARTITION BY symbol ORDER BY timestamp) as price_change,
    (price - lag(price) OVER (PARTITION BY symbol ORDER BY timestamp)) / lag(price) OVER (PARTITION BY symbol ORDER BY timestamp) * 100 as return_pct
FROM market_data
ORDER BY symbol, timestamp;

-- Look-ahead analysis
SELECT 
    symbol,
    timestamp, 
    price,
    lead(price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as next_price,
    lead(price, 5) OVER (PARTITION BY symbol ORDER BY timestamp) as price_5_periods_ahead
FROM ohlcv_data;
```

### 2. Ranking and Distribution Functions
```sql
-- Performance ranking
SELECT 
    symbol,
    daily_return,
    rank() OVER (ORDER BY daily_return DESC) as performance_rank,
    dense_rank() OVER (ORDER BY daily_return DESC) as dense_rank,
    percent_rank() OVER (ORDER BY daily_return) as percentile_rank,
    ntile(10) OVER (ORDER BY daily_return) as decile
FROM daily_returns
WHERE trading_date = '2024-01-15';

-- Row numbering for sequential analysis
SELECT 
    symbol,
    timestamp,
    price,
    row_number() OVER (PARTITION BY symbol ORDER BY timestamp) as sequence_number
FROM tick_data
WHERE symbol = 'BTCUSD';
```

### 3. Value Extraction Functions
```sql
-- First and last values in time windows
SELECT 
    symbol,
    trading_date,
    first_value(price) OVER (PARTITION BY symbol, trading_date ORDER BY timestamp) as open_price,
    last_value(price) OVER (PARTITION BY symbol, trading_date ORDER BY timestamp) as close_price,
    nth_value(price, 2) OVER (PARTITION BY symbol, trading_date ORDER BY timestamp) as second_price
FROM intraday_data;
```

## Technical Indicator Implementation

### Moving Averages and Trend Analysis
```sql
-- Multiple timeframe moving averages
SELECT 
    symbol,
    timestamp,
    close_price,
    -- Simple Moving Averages
    AVG(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as sma_10,
    AVG(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as sma_20,
    AVG(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
    ) as sma_50,
    
    -- Volume-weighted average price
    SUM(close_price * volume) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) / SUM(volume) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as vwap_20
FROM ohlcv_data
ORDER BY symbol, timestamp;
```

### Volatility and Risk Metrics
```sql
-- Rolling volatility and statistical measures
SELECT 
    symbol,
    timestamp,
    close_price,
    -- Price volatility (standard deviation)
    STDDEV(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as volatility_20,
    
    -- Rolling correlation with market
    CORR(close_price, market_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as correlation_30,
    
    -- Bollinger Bands
    AVG(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) + 2 * STDDEV(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as bollinger_upper,
    
    AVG(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) - 2 * STDDEV(close_price) OVER (
        PARTITION BY symbol 
        ORDER BY timestamp 
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as bollinger_lower
FROM market_data_with_index m
JOIN market_index i ON m.timestamp = i.timestamp
ORDER BY symbol, timestamp;
```

### Momentum and Oscillator Indicators
```sql
-- RSI and momentum indicators
WITH price_changes AS (
    SELECT 
        symbol,
        timestamp,
        close_price,
        close_price - lag(close_price) OVER (PARTITION BY symbol ORDER BY timestamp) as price_change
    FROM ohlcv_data
),
gains_losses AS (
    SELECT *,
        CASE WHEN price_change > 0 THEN price_change ELSE 0 END as gain,
        CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END as loss
    FROM price_changes
)
SELECT 
    symbol,
    timestamp,
    close_price,
    price_change,
    
    -- RSI calculation
    100 - (100 / (1 + 
        AVG(gain) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 13 PRECEDING) /
        AVG(loss) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 13 PRECEDING)
    )) as rsi_14,
    
    -- Price momentum
    close_price / lag(close_price, 10) OVER (PARTITION BY symbol ORDER BY timestamp) - 1 as momentum_10,
    
    -- Rate of change
    (close_price - lag(close_price, 12) OVER (PARTITION BY symbol ORDER BY timestamp)) /
    lag(close_price, 12) OVER (PARTITION BY symbol ORDER BY timestamp) * 100 as roc_12
FROM gains_losses
ORDER BY symbol, timestamp;
```

## Advanced Framing Techniques

### Time-Based Windows (RANGE Framing)
```sql
-- 7-day moving average using time-based window
SELECT 
    symbol,
    trading_date,
    close_price,
    AVG(close_price) OVER (
        PARTITION BY symbol
        ORDER BY trading_date
        RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
    ) as ma_7_days,
    
    -- 30-day high/low
    MAX(high_price) OVER (
        PARTITION BY symbol
        ORDER BY trading_date
        RANGE BETWEEN INTERVAL 29 DAYS PRECEDING AND CURRENT ROW
    ) as high_30_days,
    
    MIN(low_price) OVER (
        PARTITION BY symbol
        ORDER BY trading_date
        RANGE BETWEEN INTERVAL 29 DAYS PRECEDING AND CURRENT ROW
    ) as low_30_days
FROM daily_ohlcv
ORDER BY symbol, trading_date;
```

### GROUP-Based Windows
```sql
-- Analysis by trading sessions
SELECT 
    symbol,
    trading_session,
    timestamp,
    close_price,
    -- Statistics within each trading session
    AVG(close_price) OVER (
        PARTITION BY symbol
        ORDER BY trading_session
        GROUPS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as avg_last_3_sessions,
    
    MAX(close_price) OVER (
        PARTITION BY symbol
        ORDER BY trading_session
        GROUPS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) as max_last_5_sessions
FROM session_data
ORDER BY symbol, trading_session, timestamp;
```

### EXCLUDE Clause for Comparative Analysis
```sql
-- Compare current performance to peer group
SELECT 
    symbol,
    trading_date,
    daily_return,
    -- Average return of peers (excluding current symbol)
    AVG(daily_return) OVER (
        PARTITION BY sector
        ORDER BY trading_date
        ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING
        EXCLUDE CURRENT ROW
    ) as peer_avg_return,
    
    -- Compare to sector excluding current stock
    daily_return - AVG(daily_return) OVER (
        PARTITION BY sector
        ORDER BY trading_date
        ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING
        EXCLUDE CURRENT ROW
    ) as alpha_vs_peers
FROM stock_returns
ORDER BY symbol, trading_date;
```

## Window Clauses for Complex Analysis

### Named Windows for Efficiency
```sql
-- Multiple technical indicators using shared windows
SELECT 
    symbol,
    timestamp,
    close_price,
    -- Short-term indicators
    AVG(close_price) OVER short_window as sma_10,
    STDDEV(close_price) OVER short_window as volatility_10,
    MAX(high_price) OVER short_window as high_10,
    MIN(low_price) OVER short_window as low_10,
    
    -- Medium-term indicators  
    AVG(close_price) OVER medium_window as sma_20,
    STDDEV(close_price) OVER medium_window as volatility_20,
    
    -- Long-term indicators
    AVG(close_price) OVER long_window as sma_50,
    STDDEV(close_price) OVER long_window as volatility_50
FROM ohlcv_data
WINDOW 
    short_window AS (PARTITION BY symbol ORDER BY timestamp ROWS 9 PRECEDING),
    medium_window AS (PARTITION BY symbol ORDER BY timestamp ROWS 19 PRECEDING),
    long_window AS (PARTITION BY symbol ORDER BY timestamp ROWS 49 PRECEDING)
ORDER BY symbol, timestamp;
```

## QUALIFY Clause for Signal Filtering

### Trading Signal Generation
```sql
-- Generate buy/sell signals using QUALIFY
SELECT 
    symbol,
    timestamp,
    close_price,
    sma_20,
    sma_50,
    rsi_14,
    'BUY' as signal
FROM (
    SELECT 
        symbol,
        timestamp,
        close_price,
        AVG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 19 PRECEDING) as sma_20,
        AVG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 49 PRECEDING) as sma_50,
        -- RSI calculation would go here
        50 as rsi_14  -- Simplified for example
    FROM ohlcv_data
) 
QUALIFY sma_20 > sma_50  -- Golden cross condition
    AND rsi_14 < 70      -- Not overbought
    AND close_price > sma_20  -- Price above short MA

UNION ALL

SELECT 
    symbol,
    timestamp,
    close_price,  
    sma_20,
    sma_50,
    rsi_14,
    'SELL' as signal
FROM (
    SELECT 
        symbol,
        timestamp,
        close_price,
        AVG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 19 PRECEDING) as sma_20,
        AVG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 49 PRECEDING) as sma_50,
        50 as rsi_14  -- Simplified
    FROM ohlcv_data
)
QUALIFY sma_20 < sma_50  -- Death cross condition
    AND rsi_14 > 30      -- Not oversold
    AND close_price < sma_20  -- Price below short MA

ORDER BY symbol, timestamp;
```

## Performance Optimization for Window Functions

### Partitioning Strategy
```sql
-- Optimize for parallel processing
SELECT 
    symbol,
    timestamp,
    close_price,
    -- Partition by symbol for parallel computation
    AVG(close_price) OVER (
        PARTITION BY symbol  -- Each symbol processed in parallel
        ORDER BY timestamp 
        ROWS 19 PRECEDING
    ) as sma_20
FROM large_dataset
-- Pre-filter to reduce window function input
WHERE timestamp >= current_date - INTERVAL 1 YEAR
ORDER BY symbol, timestamp;
```

### Memory Management
```sql
-- Use DISTINCT and ORDER BY arguments for deterministic results
SELECT 
    symbol,
    timestamp,
    -- Count distinct trading venues per time window
    COUNT(DISTINCT exchange) OVER (
        PARTITION BY symbol
        ORDER BY timestamp
        ROWS 50 PRECEDING
    ) as venue_count,
    
    -- Order-sensitive aggregation for consistency
    string_agg(DISTINCT exchange ORDER BY exchange) OVER (
        PARTITION BY symbol
        ORDER BY timestamp
        ROWS 10 PRECEDING
    ) as venues_list
FROM multi_venue_trades
ORDER BY symbol, timestamp;
```

This window functions documentation provides the foundation for implementing sophisticated technical analysis and time-series processing capabilities essential for quantitative trading strategies.