# pandas Time Series Analysis - Comprehensive Trading Guide

## Summary
Comprehensive documentation for pandas time series analysis capabilities essential for quantitative trading: timestamp handling, resampling techniques, frequency conversion, date offset manipulation, and advanced time-based operations for financial market data processing.

## Time Series Foundation for Trading

### Timestamp and DatetimeIndex Creation
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create trading timestamps
trading_hours = pd.date_range(
    start='2023-01-01 09:30:00',
    end='2023-12-31 16:00:00', 
    freq='1H'
)

# Market-specific date ranges
nasdaq_hours = pd.date_range(
    start='2023-01-01 09:30:00',
    end='2023-01-01 16:00:00',
    freq='1min'
)

# Custom business day calendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=pd.tseries.holiday.USFederalHolidayCalendar())
trading_days = pd.date_range(start='2023-01-01', end='2023-12-31', freq=us_bd)
```

### Time Zone Handling
```python
# Create timezone-aware timestamps
utc_timestamps = pd.date_range(
    start='2023-01-01', 
    periods=1000, 
    freq='1H', 
    tz='UTC'
)

# Convert to different timezones
ny_timestamps = utc_timestamps.tz_convert('America/New_York')
london_timestamps = utc_timestamps.tz_convert('Europe/London')
tokyo_timestamps = utc_timestamps.tz_convert('Asia/Tokyo')

# Localize naive timestamps
naive_timestamps = pd.date_range('2023-01-01', periods=100, freq='1H')
aware_timestamps = naive_timestamps.tz_localize('America/New_York')
```

## Indexing and Selection Operations

### Partial String Indexing
```python
# Create sample trading data
dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
trading_data = pd.DataFrame({
    'open': np.random.uniform(100, 200, 1000),
    'high': np.random.uniform(100, 200, 1000),
    'low': np.random.uniform(100, 200, 1000),
    'close': np.random.uniform(100, 200, 1000),
    'volume': np.random.randint(1000, 100000, 1000)
}, index=dates)

# Partial string selection
january_data = trading_data['2023-01']  # All January data
first_week = trading_data['2023-01-01':'2023-01-07']  # Date range
specific_day = trading_data['2023-01-15']  # Specific day

# Time-based filtering
morning_session = trading_data.between_time('09:30', '12:00')
afternoon_session = trading_data.between_time('13:00', '16:00')
closing_hour = trading_data.at_time('15:00')
```

### Advanced Time Selection
```python
# Business day selection
business_days_only = trading_data[trading_data.index.dayofweek < 5]

# Specific weekdays (Monday=0, Friday=4)
monday_data = trading_data[trading_data.index.dayofweek == 0]
friday_data = trading_data[trading_data.index.dayofweek == 4]

# Month-end trading
month_end = trading_data[trading_data.index.is_month_end]

# Quarterly expiration (3rd Friday of March, June, September, December)
def is_quarterly_expiration(date):
    if date.month not in [3, 6, 9, 12]:
        return False
    if date.weekday() != 4:  # Not Friday
        return False
    # Check if it's the 3rd Friday
    first_day = date.replace(day=1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)
    return date.date() == third_friday.date()

expiration_dates = trading_data[
    trading_data.index.to_series().apply(is_quarterly_expiration)
]
```

## Resampling Techniques for Different Timeframes

### Basic Resampling Operations
```python
# Convert hourly to daily data
daily_data = trading_data.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Weekly aggregation (ending Sunday)
weekly_data = trading_data.resample('W').agg({
    'open': 'first',
    'high': 'max', 
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Monthly aggregation
monthly_data = trading_data.resample('M').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

### Custom Resampling Rules
```python
# 4-hour bars
four_hour_bars = trading_data.resample('4H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min', 
    'close': 'last',
    'volume': 'sum'
})

# 15-minute bars with custom alignment
fifteen_min_bars = trading_data.resample(
    '15min', 
    closed='left',  # Include left boundary
    label='left'    # Label with left boundary
).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Business day resampling
business_daily = trading_data.resample('B').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

### Advanced Resampling with Custom Functions
```python
def calculate_vwap(group):
    """Calculate Volume Weighted Average Price"""
    return (group['close'] * group['volume']).sum() / group['volume'].sum()

def calculate_typical_price(group):
    """Calculate typical price (H+L+C)/3"""
    return (group['high'] + group['low'] + group['close']).mean() / 3

# Apply custom functions during resampling
advanced_daily = trading_data.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'vwap': calculate_vwap,
    'typical_price': calculate_typical_price
})

# Multiple statistics per column
detailed_resampling = trading_data.resample('D').agg({
    'close': ['first', 'last', 'mean', 'std'],
    'volume': ['sum', 'mean', 'max'],
    'high': 'max',
    'low': 'min'
})
```

## Date Offset Manipulation

### Standard Offsets
```python
from pandas.tseries.offsets import (
    BusinessDay, Day, Hour, Minute,
    BusinessHour, Week, Month, Quarter
)

# Add business days
next_business_day = trading_data.index + BusinessDay(1)
five_business_days_later = trading_data.index + BusinessDay(5)

# Business hour calculations
market_open = pd.Timestamp('2023-01-01 09:30')
one_hour_later = market_open + BusinessHour(1)
end_of_day = market_open + BusinessHour(6.5)  # 6.5 trading hours

# Monthly and quarterly offsets
month_end_dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
quarter_end_dates = pd.date_range('2023-01-01', '2023-12-31', freq='Q')
```

### Custom Business Calendars
```python
from pandas.tseries.holiday import (
    USFederalHolidayCalendar, 
    HolidayCalendarFactory,
    Holiday
)

# Custom trading calendar
TradingCalendar = HolidayCalendarFactory(
    'TradingCalendar',
    USFederalHolidayCalendar(),
    [
        Holiday('Market Closure', month=7, day=4),  # July 4th
        Holiday('Christmas Eve Early Close', month=12, day=24),
    ]
)

custom_bd = CustomBusinessDay(calendar=TradingCalendar())
trading_days_custom = pd.date_range(
    start='2023-01-01',
    end='2023-12-31', 
    freq=custom_bd
)

# Business hour with custom calendar
custom_bh = BusinessHour(calendar=TradingCalendar())
```

## Advanced Time Series Functions

### Shift and Lag Operations
```python
# Price change calculations
trading_data['price_change'] = trading_data['close'] - trading_data['close'].shift(1)
trading_data['price_change_pct'] = trading_data['close'].pct_change()

# Multi-period shifts
trading_data['close_5d_ago'] = trading_data['close'].shift(5)
trading_data['volume_1h_ahead'] = trading_data['volume'].shift(-1)  # Look ahead

# Shift with frequency (time-aware shift)
trading_data['close_1d_ago'] = trading_data['close'].shift(1, freq='D')
trading_data['close_1w_ago'] = trading_data['close'].shift(1, freq='W')
```

### Rolling Time Windows
```python
# Time-based rolling windows
rolling_1d = trading_data['close'].rolling('1D').mean()
rolling_1w = trading_data['close'].rolling('7D').mean()
rolling_30d = trading_data['close'].rolling('30D').mean()

# Business day rolling windows
rolling_5bd = trading_data['close'].rolling('5B').mean()

# Rolling with minimum periods
rolling_adaptive = trading_data['close'].rolling(
    '10D', 
    min_periods=5
).mean()

# Expanding windows
expanding_mean = trading_data['close'].expanding().mean()
expanding_std = trading_data['close'].expanding().std()
```

### Frequency Conversion
```python
# Upsample with forward fill
upsampled = trading_data.resample('30min').ffill()

# Downsample with interpolation
downsampled = trading_data.resample('2H').interpolate()

# Convert to different frequency with alignment
aligned_daily = trading_data.asfreq('D', method='ffill')

# Business frequency conversion
business_freq = trading_data.asfreq('B', method='bfill')
```

## Trading-Specific Time Series Operations

### Market Session Analysis
```python
def analyze_trading_sessions(data):
    """Analyze different trading sessions"""
    
    # Define session times
    sessions = {
        'pre_market': ('04:00', '09:30'),
        'market_open': ('09:30', '10:30'),
        'mid_morning': ('10:30', '12:00'),
        'lunch': ('12:00', '14:00'), 
        'afternoon': ('14:00', '15:30'),
        'power_hour': ('15:00', '16:00'),
        'after_hours': ('16:00', '20:00')
    }
    
    session_stats = {}
    
    for session_name, (start_time, end_time) in sessions.items():
        session_data = data.between_time(start_time, end_time)
        
        session_stats[session_name] = {
            'avg_volume': session_data['volume'].mean(),
            'avg_volatility': session_data['close'].std(),
            'avg_return': session_data['close'].pct_change().mean(),
            'num_observations': len(session_data)
        }
    
    return pd.DataFrame(session_stats).T

session_analysis = analyze_trading_sessions(trading_data)
```

### Earnings and Event Analysis
```python
# Simulate earnings dates
earnings_dates = pd.date_range('2023-01-15', '2023-12-15', freq='3M')

def analyze_earnings_impact(data, earnings_dates, window=5):
    """Analyze price movements around earnings"""
    
    earnings_analysis = []
    
    for earnings_date in earnings_dates:
        # Get data around earnings
        start_date = earnings_date - pd.Timedelta(days=window)
        end_date = earnings_date + pd.Timedelta(days=window)
        
        try:
            event_data = data[start_date:end_date]
            
            if len(event_data) > 0:
                # Calculate pre/post earnings metrics
                pre_earnings = event_data[event_data.index < earnings_date]
                post_earnings = event_data[event_data.index >= earnings_date]
                
                if len(pre_earnings) > 0 and len(post_earnings) > 0:
                    pre_close = pre_earnings['close'].iloc[-1]
                    post_close = post_earnings['close'].iloc[0] if len(post_earnings) > 0 else pre_close
                    
                    earnings_analysis.append({
                        'earnings_date': earnings_date,
                        'pre_close': pre_close,
                        'post_close': post_close,
                        'price_change': (post_close - pre_close) / pre_close,
                        'pre_volume': pre_earnings['volume'].mean(),
                        'post_volume': post_earnings['volume'].mean()
                    })
        except KeyError:
            continue
    
    return pd.DataFrame(earnings_analysis)

earnings_impact = analyze_earnings_impact(trading_data, earnings_dates)
```

### Volatility Clustering Analysis
```python
def analyze_volatility_clustering(data, window=20):
    """Analyze volatility clustering patterns"""
    
    # Calculate returns and volatility
    returns = data['close'].pct_change()
    rolling_vol = returns.rolling(window).std()
    
    # Identify high/low volatility periods
    vol_median = rolling_vol.median()
    high_vol_periods = rolling_vol > vol_median * 1.5
    low_vol_periods = rolling_vol < vol_median * 0.5
    
    # Calculate clustering metrics
    def calculate_clustering(series):
        """Calculate how often high vol follows high vol"""
        transitions = series.astype(int).diff()
        persistence = (transitions == 0).sum() / len(transitions)
        return persistence
    
    high_vol_clustering = calculate_clustering(high_vol_periods)
    low_vol_clustering = calculate_clustering(low_vol_periods)
    
    return {
        'high_vol_clustering': high_vol_clustering,
        'low_vol_clustering': low_vol_clustering,
        'vol_median': vol_median,
        'high_vol_periods': high_vol_periods.sum(),
        'low_vol_periods': low_vol_periods.sum()
    }

vol_clustering = analyze_volatility_clustering(trading_data)
```

## Genetic Algorithm Integration for Time Series

### Time-Based Strategy Optimization
```python
def evaluate_time_based_strategy(data, genes):
    """Evaluate strategy with time-based parameters"""
    
    # Extract genetic parameters
    entry_hour = int(genes['entry_hour'])  # 9-15 (market hours)
    exit_hour = int(genes['exit_hour'])    # 10-16 (market hours)
    lookback_days = int(genes['lookback_days'])  # 1-30
    vol_threshold = genes['vol_threshold']  # 0.5-2.0
    
    # Calculate rolling volatility
    data['volatility'] = data['close'].rolling(f'{lookback_days}D').std()
    
    # Time-based entry signals
    entry_times = data[data.index.hour == entry_hour]
    exit_times = data[data.index.hour == exit_hour]
    
    # Filter by volatility
    entry_signals = entry_times[
        entry_times['volatility'] > entry_times['volatility'].rolling('30D').mean() * vol_threshold
    ]
    
    # Calculate returns
    strategy_returns = []
    
    for entry_time in entry_signals.index:
        # Find corresponding exit time
        potential_exits = exit_times[exit_times.index > entry_time]
        if len(potential_exits) > 0:
            exit_time = potential_exits.index[0]
            
            entry_price = data.loc[entry_time, 'close']
            exit_price = data.loc[exit_time, 'close']
            
            trade_return = (exit_price - entry_price) / entry_price
            strategy_returns.append(trade_return)
    
    if len(strategy_returns) == 0:
        return 0.0
    
    # Calculate performance metrics
    avg_return = np.mean(strategy_returns)
    return_std = np.std(strategy_returns)
    sharpe_ratio = avg_return / return_std if return_std > 0 else 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'avg_return': avg_return,
        'volatility': return_std,
        'num_trades': len(strategy_returns)
    }

# Genetic algorithm integration
def time_series_fitness_function(individual, data):
    """Fitness function for time-based strategies"""
    genes = {
        'entry_hour': individual[0],
        'exit_hour': individual[1],
        'lookback_days': individual[2],
        'vol_threshold': individual[3]
    }
    
    performance = evaluate_time_based_strategy(data, genes)
    return performance['sharpe_ratio']
```

### Multi-Timeframe Strategy Development
```python
def multi_timeframe_strategy(data, genes):
    """Strategy using multiple timeframes"""
    
    # Resample to different timeframes
    timeframes = {
        '15min': data.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }),
        '1H': data.resample('1H').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }),
        '4H': data.resample('4H').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        })
    }
    
    # Extract genetic parameters
    short_ma = int(genes['short_ma'])
    long_ma = int(genes['long_ma'])
    trend_timeframe = genes['trend_timeframe']  # '1H', '4H'
    entry_timeframe = genes['entry_timeframe']   # '15min', '1H'
    
    # Calculate trend on higher timeframe
    trend_data = timeframes[trend_timeframe]
    trend_data['ma_short'] = trend_data['close'].rolling(short_ma).mean()
    trend_data['ma_long'] = trend_data['close'].rolling(long_ma).mean()
    trend_data['trend'] = trend_data['ma_short'] > trend_data['ma_long']
    
    # Get entry signals on lower timeframe
    entry_data = timeframes[entry_timeframe]
    
    # Align timeframes
    aligned_trend = trend_data['trend'].reindex(entry_data.index, method='ffill')
    
    # Generate signals
    entry_signals = (
        aligned_trend &  # Trend filter
        (entry_data['close'] > entry_data['close'].shift(1)) &  # Price momentum
        (entry_data['volume'] > entry_data['volume'].rolling(10).mean())  # Volume filter
    )
    
    # Calculate performance
    returns = entry_data['close'].pct_change()
    strategy_returns = returns.where(entry_signals.shift(1), 0)
    
    return {
        'total_return': strategy_returns.sum(),
        'sharpe_ratio': strategy_returns.mean() / strategy_returns.std(),
        'max_drawdown': (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
    }
```

## Best Practices for Time Series Trading

### Performance Optimization
```python
# 1. Use vectorized operations
def efficient_technical_indicators(data):
    """Calculate indicators efficiently"""
    
    # Vectorized RSI calculation
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Vectorized Bollinger Bands
    rolling_mean = data['close'].rolling(20).mean()
    rolling_std = data['close'].rolling(20).std()
    data['BB_upper'] = rolling_mean + (rolling_std * 2)
    data['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    return data

# 2. Efficient resampling
def batch_resample_multiple_timeframes(data, timeframes):
    """Resample to multiple timeframes efficiently"""
    
    resampled_data = {}
    
    for tf in timeframes:
        resampled_data[tf] = data.resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    
    return resampled_data
```

### Data Quality Management
```python
def clean_trading_data(data):
    """Clean and validate trading data"""
    
    # Remove outliers
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        q1 = data[col].quantile(0.01)
        q99 = data[col].quantile(0.99)
        data[col] = data[col].clip(lower=q1, upper=q99)
    
    # Fill missing values appropriately
    data = data.fillna(method='ffill')  # Forward fill prices
    data['volume'] = data['volume'].fillna(0)  # Zero fill volume
    
    # Ensure proper ordering
    data = data.sort_index()
    
    # Validate OHLC relationships
    data = data[
        (data['high'] >= data['low']) &
        (data['high'] >= data['open']) &
        (data['high'] >= data['close']) &
        (data['low'] <= data['open']) &
        (data['low'] <= data['close'])
    ]
    
    return data
```

This comprehensive time series analysis guide provides the foundation for implementing sophisticated time-based quantitative trading strategies with proper handling of financial market data characteristics and optimal performance patterns.