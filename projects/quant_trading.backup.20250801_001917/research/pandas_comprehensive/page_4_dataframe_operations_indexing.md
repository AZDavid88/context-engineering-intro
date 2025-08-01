# pandas DataFrame Operations & Boolean Indexing - Quantitative Trading Guide

## Summary
Comprehensive documentation for pandas DataFrame operations and indexing patterns essential for quantitative trading: boolean indexing, label-based indexing (.loc), integer-based indexing (.iloc), multi-level indexing, conditional selection, and performance optimization techniques for large trading datasets.

## Core DataFrame Architecture for Trading

### DataFrame Structure Overview
```python
import pandas as pd
import numpy as np

# Typical trading DataFrame structure
trading_data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
    'open': np.random.uniform(100, 200, 1000),
    'high': np.random.uniform(100, 200, 1000),
    'low': np.random.uniform(100, 200, 1000),
    'close': np.random.uniform(100, 200, 1000),
    'volume': np.random.randint(1000, 100000, 1000),
    'asset': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 1000)
}).set_index('timestamp')
```

## Boolean Indexing for Signal Filtering

### Basic Boolean Operations
```python
# Filter high-volume trades
high_volume = trading_data[trading_data['volume'] > 50000]

# Combine multiple conditions
strong_signals = trading_data[
    (trading_data['volume'] > 50000) & 
    (trading_data['close'] > trading_data['open']) &
    (trading_data['high'] - trading_data['low'] > 5)
]

# Complex boolean logic
breakout_signals = trading_data[
    ((trading_data['close'] > trading_data['high'].shift(1)) |
     (trading_data['close'] < trading_data['low'].shift(1))) &
    (trading_data['volume'] > trading_data['volume'].rolling(20).mean())
]
```

### Technical Analysis Filtering
```python
# RSI-based filtering
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Apply RSI filter
trading_data['RSI'] = trading_data.groupby('asset')['close'].apply(calculate_rsi)

# Filter oversold conditions
oversold_signals = trading_data[
    (trading_data['RSI'] < 30) & 
    (trading_data['RSI'].shift(1) >= 30)  # Just crossed into oversold
]

# Multiple technical conditions
entry_signals = trading_data[
    (trading_data['RSI'] < 30) &  # Oversold
    (trading_data['close'] > trading_data['close'].rolling(5).mean()) &  # Above short MA
    (trading_data['volume'] > trading_data['volume'].rolling(10).mean() * 1.5)  # High volume
]
```

### Risk Management Filtering
```python
# Position sizing based on volatility
trading_data['volatility'] = trading_data.groupby('asset')['close'].rolling(20).std()
trading_data['position_size'] = trading_data['volatility'].apply(
    lambda x: 1.0 if x < 5 else 0.5 if x < 10 else 0.25
)

# Filter trades by risk criteria
safe_trades = trading_data[
    (trading_data['volatility'] < 10) &
    (trading_data['position_size'] >= 0.5)
]

# Dynamic filtering based on market conditions
market_conditions = trading_data['volatility'].rolling(50).mean()
conservative_filter = trading_data[
    trading_data['volatility'] < market_conditions * 0.8
]
```

## Label-Based Indexing (.loc) for Time Series

### Time-Based Selection
```python
# Select specific time ranges
morning_session = trading_data.loc['2023-01-01 09:00':'2023-01-01 11:00']

# Select multiple assets for specific time
multi_asset_slice = trading_data.loc[
    (trading_data.index >= '2023-01-01') & 
    (trading_data.index <= '2023-01-02') &
    (trading_data['asset'].isin(['AAPL', 'GOOGL']))
]

# Advanced time selection with conditions
intraday_breakouts = trading_data.loc[
    (trading_data.index.hour.between(9, 16)) &  # Market hours
    (trading_data['high'] > trading_data['high'].shift(1) * 1.02),  # 2% breakout
    ['close', 'volume', 'asset']
]
```

### Conditional Column Selection
```python
# Select specific columns based on conditions
ohlc_data = trading_data.loc[:, ['open', 'high', 'low', 'close']]

# Dynamic column selection
price_columns = [col for col in trading_data.columns if col in ['open', 'high', 'low', 'close']]
price_data = trading_data.loc[:, price_columns]

# Conditional row and column selection
high_vol_ohlc = trading_data.loc[
    trading_data['volume'] > 75000, 
    ['open', 'high', 'low', 'close']
]
```

### Multi-Condition Assignment
```python
# Assign values based on complex conditions
trading_data.loc[
    (trading_data['RSI'] < 30) & (trading_data['volume'] > 50000),
    'signal'
] = 'STRONG_BUY'

trading_data.loc[
    (trading_data['RSI'] > 70) & (trading_data['volume'] > 50000),
    'signal'
] = 'STRONG_SELL'

# Multiple column assignment
trading_data.loc[
    trading_data['signal'] == 'STRONG_BUY',
    ['position', 'confidence']
] = [1, 0.8]
```

## Integer-Based Indexing (.iloc) for Positional Access

### Fixed Position Selection
```python
# Select first and last 100 rows
recent_data = trading_data.iloc[-100:]
historical_data = trading_data.iloc[:100]

# Select every nth row for sampling
sampled_data = trading_data.iloc[::10]  # Every 10th row

# Select specific row ranges
mid_session_data = trading_data.iloc[200:400]
```

### Column-Based Selection
```python
# Select first 4 columns (OHLC)
ohlc_iloc = trading_data.iloc[:, :4]

# Select specific column positions
selected_cols = trading_data.iloc[:, [0, 2, 4]]  # Open, Low, Volume

# Mixed selection
recent_ohlc = trading_data.iloc[-50:, :4]  # Last 50 rows, first 4 columns
```

### Rolling Window Operations
```python
# Apply function to rolling windows using iloc
def rolling_max_drawdown(data, window=20):
    """Calculate rolling maximum drawdown using iloc"""
    results = []
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        peak = window_data.max()
        trough = window_data.min()
        drawdown = (trough - peak) / peak
        results.append(drawdown)
    return pd.Series(results, index=data.index[window:])

# Apply to close prices
trading_data['max_drawdown'] = trading_data.groupby('asset')['close'].apply(
    lambda x: rolling_max_drawdown(x, 20)
)
```

## Multi-Level Indexing for Multi-Asset Strategies

### Creating Multi-Level Index
```python
# Create hierarchical index for multiple assets and timeframes
multi_asset_data = pd.DataFrame({
    'open': np.random.uniform(100, 200, 300),
    'high': np.random.uniform(100, 200, 300),
    'low': np.random.uniform(100, 200, 300),
    'close': np.random.uniform(100, 200, 300),
    'volume': np.random.randint(1000, 100000, 300)
})

# Set multi-level index
multi_index = pd.MultiIndex.from_product([
    ['AAPL', 'GOOGL', 'MSFT'],  # Assets
    ['1H', '4H', 'D'],  # Timeframes
    range(33, 34)  # Just enough to complete 300 rows (3*3*33 = 297, need 3 more)
] + [('AAPL', '1H', 34)], names=['asset', 'timeframe', 'period'])

multi_asset_data.index = multi_index[:300]
```

### Multi-Level Selection
```python
# Select all data for specific asset
aapl_all_timeframes = multi_asset_data.loc['AAPL']

# Select specific asset and timeframe
aapl_daily = multi_asset_data.loc[('AAPL', 'D')]

# Cross-section selection
all_daily_data = multi_asset_data.loc[(slice(None), 'D'), :]

# Multiple assets, specific timeframe
tech_stocks_hourly = multi_asset_data.loc[(['AAPL', 'GOOGL'], '1H'), :]
```

### Portfolio-Level Operations
```python
# Calculate portfolio-level statistics
portfolio_returns = multi_asset_data.groupby(level='asset')['close'].pct_change()

# Asset allocation by volatility
asset_volatility = multi_asset_data.groupby(level='asset')['close'].std()
total_volatility = asset_volatility.sum()
allocation_weights = (1 / asset_volatility) / (1 / asset_volatility).sum()

# Portfolio value calculation
def calculate_portfolio_value(data, weights):
    """Calculate portfolio value with multi-asset weights"""
    portfolio_value = pd.Series(index=data.index, dtype=float)
    
    for asset in weights.index:
        asset_data = data.loc[asset]['close']
        portfolio_value.loc[asset] = asset_data * weights[asset]
    
    return portfolio_value.groupby(level=0).sum()
```

## Query Method for Complex Filtering

### Basic Query Operations
```python
# String-based querying (more readable)
high_momentum = trading_data.query(
    'volume > 50000 and close > open * 1.02'
)

# Variable substitution
min_volume = 25000
max_rsi = 70
filtered_data = trading_data.query(
    'volume > @min_volume and RSI < @max_rsi'
)

# Complex expressions
breakout_query = trading_data.query(
    '(high > high.shift(1) * 1.03) and '
    '(volume > volume.rolling(10).mean() * 1.5) and '
    'RSI.between(40, 60)'
)
```

### Dynamic Query Building
```python
def build_trading_query(conditions):
    """Build dynamic query string from conditions dictionary"""
    query_parts = []
    
    if 'min_volume' in conditions:
        query_parts.append(f"volume > {conditions['min_volume']}")
    
    if 'rsi_range' in conditions:
        rsi_min, rsi_max = conditions['rsi_range']
        query_parts.append(f"RSI.between({rsi_min}, {rsi_max})")
    
    if 'price_change' in conditions:
        query_parts.append(f"close > open * {conditions['price_change']}")
    
    return ' and '.join(query_parts)

# Usage
filter_conditions = {
    'min_volume': 30000,
    'rsi_range': (30, 70),
    'price_change': 1.01
}

query_string = build_trading_query(filter_conditions)
filtered_trades = trading_data.query(query_string)
```

## Performance Optimization Techniques

### Vectorized Operations
```python
# Avoid loops - use vectorized operations
# SLOW: Loop-based calculation
def slow_rsi_calculation(data):
    results = []
    for i in range(len(data)):
        if i < 14:
            results.append(np.nan)
        else:
            window_data = data.iloc[i-14:i]
            # RSI calculation logic here
            results.append(calculate_single_rsi(window_data))
    return pd.Series(results, index=data.index)

# FAST: Vectorized calculation
def fast_rsi_calculation(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=window).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### Memory Optimization
```python
# Use categorical data for repeated strings
trading_data['asset'] = trading_data['asset'].astype('category')

# Efficient data types
trading_data['volume'] = trading_data['volume'].astype('int32')  # Instead of int64
trading_data['close'] = trading_data['close'].astype('float32')  # Instead of float64

# Use sparse arrays for sparse signals
signals = pd.Series(0, index=trading_data.index)
signals.iloc[::100] = 1  # Sparse buy signals
sparse_signals = signals.astype(pd.SparseDtype('int64', 0))
```

### Index Optimization
```python
# Sort index for faster queries
trading_data = trading_data.sort_index()

# Use appropriate index types
if trading_data.index.is_monotonic_increasing:
    # Fast time-based slicing available
    fast_slice = trading_data['2023-01-01':'2023-01-02']

# Create secondary indexes for frequent operations
trading_data_by_asset = trading_data.set_index('asset', append=True)
trading_data_by_asset = trading_data_by_asset.sort_index()
```

## Genetic Algorithm Integration Examples

### Strategy Parameter Optimization
```python
def evaluate_strategy_performance(data, params):
    """Evaluate strategy using DataFrame operations"""
    
    # Extract parameters
    rsi_buy = params['rsi_buy_threshold']
    rsi_sell = params['rsi_sell_threshold']
    volume_multiplier = params['volume_multiplier']
    
    # Generate signals using boolean indexing
    avg_volume = data['volume'].rolling(20).mean()
    
    buy_signals = data[
        (data['RSI'] < rsi_buy) &
        (data['volume'] > avg_volume * volume_multiplier)
    ].index
    
    sell_signals = data[
        (data['RSI'] > rsi_sell) &
        (data['volume'] > avg_volume * volume_multiplier)
    ].index
    
    # Calculate returns using .loc indexing
    strategy_returns = pd.Series(0.0, index=data.index)
    
    for buy_time in buy_signals:
        # Find next sell signal
        future_sells = sell_signals[sell_signals > buy_time]
        if len(future_sells) > 0:
            sell_time = future_sells[0]
            buy_price = data.loc[buy_time, 'close']
            sell_price = data.loc[sell_time, 'close']
            returns = (sell_price - buy_price) / buy_price
            strategy_returns.loc[buy_time:sell_time] = returns
    
    # Performance metrics
    total_return = strategy_returns.sum()
    volatility = strategy_returns.std()
    sharpe_ratio = total_return / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'num_trades': len(buy_signals)
    }

# Genetic algorithm fitness function
def fitness_function(individual, data):
    """Calculate fitness using DataFrame operations"""
    params = {
        'rsi_buy_threshold': individual[0],
        'rsi_sell_threshold': individual[1], 
        'volume_multiplier': individual[2]
    }
    
    performance = evaluate_strategy_performance(data, params)
    return performance['sharpe_ratio'] - abs(performance['volatility']) * 0.1
```

### Multi-Asset Strategy Development
```python
def multi_asset_strategy_evaluation(multi_data, genes):
    """Evaluate strategy across multiple assets using advanced indexing"""
    
    results = {}
    
    for asset in multi_data.index.get_level_values('asset').unique():
        # Extract asset-specific data
        asset_data = multi_data.loc[asset]
        
        # Apply genetic parameters
        ma_short = int(genes['ma_short'])
        ma_long = int(genes['ma_long'])
        vol_threshold = genes['vol_threshold']
        
        # Calculate indicators
        asset_data['MA_short'] = asset_data['close'].rolling(ma_short).mean()
        asset_data['MA_long'] = asset_data['close'].rolling(ma_long).mean()
        asset_data['volume_ma'] = asset_data['volume'].rolling(20).mean()
        
        # Generate signals using complex boolean indexing
        entry_condition = (
            (asset_data['MA_short'] > asset_data['MA_long']) &
            (asset_data['MA_short'].shift(1) <= asset_data['MA_long'].shift(1)) &
            (asset_data['volume'] > asset_data['volume_ma'] * vol_threshold)
        )
        
        exit_condition = (
            (asset_data['MA_short'] < asset_data['MA_long']) &
            (asset_data['MA_short'].shift(1) >= asset_data['MA_long'].shift(1))
        )
        
        # Calculate asset performance
        asset_performance = calculate_asset_returns(asset_data, entry_condition, exit_condition)
        results[asset] = asset_performance
    
    # Portfolio-level aggregation
    portfolio_performance = aggregate_portfolio_results(results)
    return portfolio_performance
```

## Best Practices Summary

### Performance Guidelines
1. **Use vectorized operations** instead of loops
2. **Sort indexes** for faster slicing operations
3. **Use appropriate data types** (categorical, sparse, smaller numeric types)
4. **Minimize chained indexing** - use .loc and .iloc explicitly
5. **Leverage query()** for complex readable conditions

### Trading-Specific Patterns
1. **Boolean indexing** for signal generation and filtering
2. **Multi-level indexing** for multi-asset, multi-timeframe strategies
3. **Time-based .loc slicing** for session-specific analysis
4. **.iloc positional access** for fixed window operations
5. **Conditional assignment** for dynamic strategy parameters

### Memory Management
1. **Use categorical data** for repeated string values (assets, signals)
2. **Convert to appropriate numeric types** (float32 instead of float64)
3. **Use sparse structures** for signal series with many zeros
4. **Delete intermediate DataFrames** in memory-constrained environments

This comprehensive DataFrame indexing and operations guide provides the foundation for implementing sophisticated quantitative trading strategies with optimal performance and clean, maintainable code.