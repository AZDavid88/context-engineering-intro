# pandas Series: diff(), where(), fillna(), mean() - Technical Analysis Essentials

## Summary
Comprehensive documentation for core pandas Series methods essential for technical analysis: diff() for price changes, where() for conditional logic, fillna() for data cleaning, and mean() for statistical calculations in quantitative trading systems.

## pandas.Series.diff() - Price Change Analysis

### Method Signature
```python
Series.diff(periods=1)
```

### Parameters
**periods** (int, default 1)
- Periods to shift for calculating difference
- Positive: Compare with previous values
- Negative: Compare with future values

### Technical Analysis Applications

#### Basic Price Changes
```python
# Daily returns
df['daily_return'] = df['close'].diff()
df['daily_return_pct'] = df['close'].pct_change()

# Multi-period returns
df['return_5d'] = df['close'].diff(5)  # 5-day return
df['return_weekly'] = df['close'].diff(7)  # Weekly return

# Log returns for statistical analysis
import numpy as np
df['log_return'] = np.log(df['close']).diff()
```

#### Momentum Indicators
```python
# Rate of Change (ROC)
df['ROC_10'] = (df['close'].diff(10) / df['close'].shift(10)) * 100

# Price momentum
df['momentum_5'] = df['close'].diff(5)
df['momentum_10'] = df['close'].diff(10)

# Acceleration (second derivative)
df['acceleration'] = df['close'].diff().diff()
```

#### Volatility Measures
```python
# Absolute price changes
df['abs_change'] = df['close'].diff().abs()

# Rolling volatility from price changes
df['volatility'] = df['close'].diff().rolling(20).std()

# High-Low volatility
df['hl_volatility'] = (df['high'] - df['low']).diff().abs()
```

#### Forward-Looking Analysis
```python
# Future returns (for backtesting only)
df['future_return_1d'] = df['close'].diff(-1)  # Next day return
df['future_return_5d'] = df['close'].diff(-5)  # 5-day future return

# Warning: Only use for historical analysis, never for live trading
```

## pandas.Series.where() - Conditional Signal Logic

### Method Signature
```python
Series.where(cond, other=nan, *, inplace=False, axis=None, level=None)
```

### Parameters
**cond** (bool Series/DataFrame, array-like, or callable)
- Condition to evaluate
- True: Keep original value
- False: Replace with `other`

**other** (scalar, Series/DataFrame, or callable)
- Replacement value when condition is False

### Technical Analysis Applications

#### Signal Generation
```python
# Buy/sell signals with price levels
df['buy_signal'] = df['close'].where(df['close'] > df['SMA_20'], 0)
df['sell_signal'] = df['close'].where(df['close'] < df['SMA_20'], 0)

# RSI-based position sizing
df['position_size'] = pd.Series(1.0, index=df.index).where(
    (df['RSI'] > 30) & (df['RSI'] < 70), 
    0.5  # Reduce size in overbought/oversold regions
)
```

#### Risk Management
```python
# Stop-loss conditions
df['stop_loss_price'] = df['entry_price'].where(
    df['close'] > df['entry_price'] * 0.95,  # 5% stop loss
    df['close']  # Current price if stop triggered
)

# Position filtering by volatility
df['filtered_signals'] = df['raw_signals'].where(
    df['volatility'] < df['volatility'].rolling(50).quantile(0.8),
    0  # No signals during high volatility
)
```

#### Technical Indicator Enhancement
```python
# RSI with extreme value handling
df['RSI_adjusted'] = df['RSI'].where(
    (df['RSI'] >= 10) & (df['RSI'] <= 90),
    np.where(df['RSI'] < 10, 10, 90)  # Cap extreme values
)

# Bollinger Band squeeze detection
bb_squeeze = (df['BB_upper'] - df['BB_lower']) < df['ATR'] * 1.5
df['BB_signals'] = df['BB_signals'].where(~bb_squeeze, 0)
```

#### Regime-Based Filtering
```python
# Bull market filter
bull_market = df['close'] > df['SMA_200']
df['bull_signals'] = df['signals'].where(bull_market, 0)

# Volatility regime filtering
low_vol_regime = df['volatility'] < df['volatility'].rolling(100).median()
df['low_vol_signals'] = df['signals'].where(low_vol_regime, 0)
```

## pandas.Series.fillna() - Data Cleaning for Trading

### Method Signature
```python
Series.fillna(value=None, *, method=None, axis=None, inplace=False, limit=None, downcast=<no_default>)
```

### Parameters
**value** (scalar, dict, Series, or DataFrame)
- Replacement value for NaN
- Can be constant or dynamic

**method** ({'backfill', 'bfill', 'ffill', None}, default None)
- Forward fill ('ffill') or backward fill ('bfill')
- Deprecated: Use ffill() or bfill() directly

### Technical Analysis Applications

#### Market Data Cleaning
```python
# Forward fill for weekend gaps
df['close_filled'] = df['close'].fillna(method='ffill')

# Fill with market open price
df['clean_close'] = df['close'].fillna(df['open'])

# Fill with previous day's close
df['continuous_close'] = df['close'].fillna(df['close'].shift(1))
```

#### Technical Indicator Initialization
```python
# Initialize indicators with first valid value
df['SMA_20'] = df['close'].rolling(20).mean()
df['SMA_20_filled'] = df['SMA_20'].fillna(df['close'])

# Fill RSI initialization period
df['RSI'] = calculate_rsi(df['close'], 14)  # Custom function
df['RSI_filled'] = df['RSI'].fillna(50)  # Neutral RSI
```

#### Multi-Asset Data Alignment
```python
# Fill missing data across assets
price_data = pd.DataFrame({
    'AAPL': apple_prices,
    'GOOGL': google_prices,
    'MSFT': microsoft_prices
})

# Forward fill missing prices
price_data_clean = price_data.fillna(method='ffill')

# Fill with cross-sectional median
for col in price_data.columns:
    price_data[col] = price_data[col].fillna(
        price_data.median(axis=1)
    )
```

#### Custom Fill Strategies
```python
# Fill with moving average
df['price_filled'] = df['close'].fillna(
    df['close'].rolling(10, min_periods=1).mean()
)

# Fill with volatility-adjusted values
vol_fill = df['close'].shift(1) + df['daily_return'].rolling(5).mean()
df['vol_adjusted_fill'] = df['close'].fillna(vol_fill)
```

## pandas.Series.mean() - Statistical Foundation

### Method Signature
```python
Series.mean(axis=0, skipna=True, numeric_only=False, **kwargs)
```

### Parameters
**skipna** (bool, default True)
- Exclude NA/null values from calculation

**numeric_only** (bool, default False)
- Include only numeric data (not applicable for Series)

### Technical Analysis Applications

#### Basic Statistics
```python
# Average price levels
avg_close = df['close'].mean()
avg_volume = df['volume'].mean()

# Rolling averages (alternative to rolling().mean())
df['avg_return_20d'] = df['daily_return'].rolling(20).apply(lambda x: x.mean())
```

#### Risk Metrics
```python
# Average return calculation
daily_returns = df['close'].pct_change()
avg_daily_return = daily_returns.mean()
annualized_return = avg_daily_return * 252

# Average volatility
avg_volatility = daily_returns.std()

# Average drawdown
rolling_max = df['close'].expanding().max()
drawdown = (df['close'] - rolling_max) / rolling_max
avg_drawdown = drawdown.mean()
```

#### Performance Analysis
```python
# Strategy performance metrics
strategy_returns = df['strategy_pnl'].pct_change()
avg_strategy_return = strategy_returns.mean()

# Win rate calculation
winning_trades = strategy_returns[strategy_returns > 0]
win_rate = len(winning_trades) / len(strategy_returns.dropna())

# Average win/loss
avg_win = winning_trades.mean()
avg_loss = strategy_returns[strategy_returns < 0].mean()
```

### Genetic Algorithm Integration

#### Fitness Function Components
```python
def evaluate_strategy_fitness(signals, prices):
    """Calculate fitness metrics using pandas statistical functions"""
    
    # Generate returns
    strategy_returns = prices.pct_change().where(signals.shift(1), 0)
    
    # Key fitness components
    total_return = (1 + strategy_returns).prod() - 1
    avg_return = strategy_returns.mean()
    volatility = strategy_returns.std()
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Combined fitness score
    fitness = sharpe_ratio - abs(max_drawdown) * 2
    
    return {
        'fitness': fitness,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_return': avg_return,
        'volatility': volatility
    }
```

#### Parameter Optimization
```python
def optimize_indicator_params(price_data, param_ranges):
    """Optimize technical indicator parameters using statistical measures"""
    
    best_params = {}
    best_score = -np.inf
    
    for fast_ma in param_ranges['fast_ma']:
        for slow_ma in param_ranges['slow_ma']:
            # Calculate moving averages
            fast_avg = price_data.rolling(fast_ma).mean()
            slow_avg = price_data.rolling(slow_ma).mean()
            
            # Generate signals
            signals = (fast_avg > slow_avg).astype(int).diff()
            
            # Calculate performance
            returns = price_data.pct_change().where(signals.shift(1) == 1, 0)
            
            # Score based on mean return and consistency
            avg_return = returns.mean()
            return_std = returns.std()
            score = avg_return / return_std if return_std > 0 else 0
            
            if score > best_score:
                best_score = score
                best_params = {'fast_ma': fast_ma, 'slow_ma': slow_ma}
    
    return best_params, best_score
```

### Integration Examples

#### Complete Technical Analysis Pipeline
```python
def technical_analysis_pipeline(df):
    """Complete pipeline using all four methods"""
    
    # 1. Calculate price changes
    df['returns'] = df['close'].diff()
    df['log_returns'] = np.log(df['close']).diff()
    
    # 2. Generate signals with where()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    buy_condition = (df['ma_20'] > df['ma_50']) & (df['returns'] > 0)
    df['buy_signals'] = pd.Series(1, index=df.index).where(buy_condition, 0)
    
    # 3. Clean data with fillna()
    df['clean_signals'] = df['buy_signals'].fillna(0)
    df['clean_returns'] = df['returns'].fillna(0)
    
    # 4. Calculate performance statistics with mean()
    active_returns = df['clean_returns'].where(df['clean_signals'] == 1, 0)
    
    performance_stats = {
        'avg_return': active_returns.mean(),
        'avg_signal_strength': df['clean_signals'].mean(),
        'return_volatility': active_returns.std(),
        'signal_consistency': df['clean_signals'].rolling(50).mean().std()
    }
    
    return df, performance_stats
```

### Best Practices

1. **diff()**: Always check for sufficient data before calculating differences
2. **where()**: Use boolean indexing for better performance when possible
3. **fillna()**: Choose fill method based on market characteristics
4. **mean()**: Consider skipna parameter for datasets with missing values
5. **Integration**: Combine methods for robust technical analysis pipelines

These four methods form the core foundation for technical analysis and quantitative trading strategy development in pandas.