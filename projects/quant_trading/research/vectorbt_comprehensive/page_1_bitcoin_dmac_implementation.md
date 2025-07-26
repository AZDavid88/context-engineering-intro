# Vectorbt Bitcoin Dual Moving Average Crossover (DMAC) Implementation

## Source
- **URL**: https://nbviewer.org/format/script/github/polakowo/vectorbt/blob/master/examples/BitcoinDMAC.ipynb
- **Focus**: Dual moving average crossover implementation, universal strategy patterns, genetic algorithm parameter evolution

## Content Summary

This comprehensive example demonstrates how to implement a Dual Moving Average Crossover (DMAC) strategy using vectorbt, with advanced analysis including 2D heatmaps for parameter optimization and 3D performance cubes for time-based analysis.

## Complete Implementation Code

```python
#!/usr/bin/env python
# coding: utf-8

# Dual Moving Average Crossover (DMAC) Strategy Implementation
# Features: Single/Multi-window analysis, Interactive visualizations, Strategy comparison

import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse
import ipywidgets as widgets
from copy import deepcopy
from tqdm import tqdm
import imageio
from IPython import display
import plotly.graph_objects as go
import itertools
import dateparser
import gc

# Configuration Parameters
seed = 42
symbol = 'BTC-USD'
metric = 'total_return'
start_date = datetime(2018, 1, 1, tzinfo=pytz.utc)
end_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
time_buffer = timedelta(days=100)  # Buffer for pre-calculating moving averages
freq = '1D'

# Portfolio Settings
vbt.settings.portfolio['init_cash'] = 100.  # $100
vbt.settings.portfolio['fees'] = 0.0025     # 0.25%
vbt.settings.portfolio['slippage'] = 0.0025 # 0.25%

# Data Download and Preparation
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
ohlcv_wbuf = vbt.YFData.download(symbol, start=start_date-time_buffer, end=end_date).get(cols)
ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)

# Create data without time buffer
wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date)
ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]

# Plot OHLC data
ohlcv_wbuf.vbt.ohlcv.plot().show_svg()
```

## Single Window Combination Analysis

```python
# Single DMAC Strategy Test
fast_window = 30
slow_window = 80

# Pre-calculate moving averages with time buffer
fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)

# Remove time buffer
fast_ma = fast_ma[wobuf_mask]
slow_ma = slow_ma[wobuf_mask]

# Assert no NaNs after buffer removal
assert(~fast_ma.ma.isnull().any())
assert(~slow_ma.ma.isnull().any())

# Generate crossover signals
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

# Visualize strategy
fig = ohlcv['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
fig = fast_ma.ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=fig)
fig = slow_ma.ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=fig)
fig = dmac_entries.vbt.signals.plot_as_entry_markers(ohlcv['Open'], fig=fig)
fig = dmac_exits.vbt.signals.plot_as_exit_markers(ohlcv['Open'], fig=fig)
fig.show_svg()

# Build portfolio from signals
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)
print(dmac_pf.stats())

# Hold strategy comparison
hold_entries = pd.Series.vbt.signals.empty_like(dmac_entries)
hold_entries.iloc[0] = True
hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
hold_exits.iloc[-1] = True
hold_pf = vbt.Portfolio.from_signals(ohlcv['Close'], hold_entries, hold_exits)

# Equity curve comparison
fig = dmac_pf.value().vbt.plot(trace_kwargs=dict(name='Value (DMAC)'))
hold_pf.value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=fig).show_svg()
```

## Interactive Window Optimization

```python
# Interactive Window Slider Implementation
min_window = 2
max_window = 100

perf_metrics = ['total_return', 'positions.win_rate', 'positions.expectancy', 'max_drawdown']
perf_metric_names = ['Total return', 'Win rate', 'Expectancy', 'Max drawdown']

windows_slider = widgets.IntRangeSlider(
    value=[fast_window, slow_window],
    min=min_window,
    max=max_window,
    step=1,
    layout=dict(width='500px'),
    continuous_update=True
)

def on_value_change(value):
    global dmac_fig
    fast_window, slow_window = value['new']
    
    # Calculate portfolio with new parameters
    fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
    slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)
    fast_ma = fast_ma[wobuf_mask]
    slow_ma = slow_ma[wobuf_mask]
    
    dmac_entries = fast_ma.ma_crossed_above(slow_ma)
    dmac_exits = fast_ma.ma_crossed_below(slow_ma)
    dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)
    
    # Update visualization and metrics
    # ... (visualization update code)

windows_slider.observe(on_value_change, names='value')
```

## Multiple Window Combinations (Parameter Optimization)

```python
# Vectorized Parameter Optimization Across All Window Combinations
fast_ma, slow_ma = vbt.MA.run_combs(
    ohlcv_wbuf['Open'], 
    np.arange(min_window, max_window+1), 
    r=2, 
    short_names=['fast_ma', 'slow_ma']
)

# Remove time buffer
fast_ma = fast_ma[wobuf_mask]
slow_ma = slow_ma[wobuf_mask]

# Generate signals for all combinations (4851 columns)
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

# Build portfolio for all combinations
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

# Calculate performance for each combination
dmac_perf = dmac_pf.deep_getattr(metric)
optimal_combination = dmac_perf.idxmax()

# Convert to heatmap matrix
dmac_perf_matrix = dmac_perf.vbt.unstack_to_df(
    symmetric=True, 
    index_levels='fast_ma_window', 
    column_levels='slow_ma_window'
)

# Visualize as heatmap
dmac_perf_matrix.vbt.heatmap(
    xaxis_title='Slow window',
    yaxis_title='Fast window'
).show_svg()
```

## Advanced Time-Based Analysis (3D Performance Cube)

```python
# Rolling Window Analysis for Temporal Performance
ts_window = timedelta(days=365)
ts_window_n = 50  # 50 overlapping time ranges

# Split data into overlapping windows
open_roll_wbuf, split_indexes = ohlcv_wbuf['Open'].vbt.range_split(
    range_len=(ts_window + time_buffer).days, 
    n=ts_window_n
)
close_roll_wbuf, _ = ohlcv_wbuf['Close'].vbt.range_split(
    range_len=(ts_window + time_buffer).days, 
    n=ts_window_n
)

# Calculate moving averages for all date ranges and window combinations
fast_ma_roll, slow_ma_roll = vbt.MA.run_combs(
    open_roll_wbuf, 
    np.arange(min_window, max_window+1), 
    r=2, 
    short_names=['fast_ma', 'slow_ma']
)

# Remove time buffer
close_roll = close_roll_wbuf.iloc[time_buffer.days:]
fast_ma_roll = fast_ma_roll.iloc[time_buffer.days:]
slow_ma_roll = slow_ma_roll.iloc[time_buffer.days:]

# Generate crossover signals
dmac_entries_roll = fast_ma_roll.ma_crossed_above(slow_ma_roll)
dmac_exits_roll = fast_ma_roll.ma_crossed_below(slow_ma_roll)

# Calculate performance across all time periods
dmac_roll_pf = vbt.Portfolio.from_signals(close_roll, dmac_entries_roll, dmac_exits_roll, freq=freq)
dmac_roll_perf = dmac_roll_pf.deep_getattr(metric)

# Create 3D performance cube
dmac_perf_cube = dmac_roll_perf.vbt.unstack_to_array(
    levels=('fast_ma_window', 'slow_ma_window', 'split_idx')
)

# Mean performance across all time periods
heatmap_index = dmac_roll_perf.index.levels[0]
heatmap_columns = dmac_roll_perf.index.levels[1]
heatmap_df = pd.DataFrame(
    np.nanmean(dmac_perf_cube, axis=2), 
    index=heatmap_index, 
    columns=heatmap_columns
)
heatmap_df = heatmap_df.vbt.make_symmetric()

# Visualize mean performance heatmap
heatmap_df.vbt.heatmap(
    xaxis_title='Slow window',
    yaxis_title='Fast window',
    trace_kwargs=dict(zmid=0, colorscale='RdBu')
).show_svg()
```

## Strategy Comparison Framework

```python
# Random Strategy Implementation
rand_entries_roll = dmac_entries_roll.vbt.signals.shuffle(seed=seed)
rand_exits_roll = rand_entries_roll.vbt.signals.generate_random_exits(seed=seed)
rand_roll_pf = vbt.Portfolio.from_signals(close_roll, rand_entries_roll, rand_exits_roll, freq=freq)
rand_roll_perf = rand_roll_pf.deep_getattr(metric)

# Hold Strategy Implementation
hold_entries_roll = pd.DataFrame.vbt.signals.empty_like(dmac_entries_roll)
hold_entries_roll.iloc[0] = True
hold_exits_roll = pd.DataFrame.vbt.signals.empty_like(hold_entries_roll)
hold_exits_roll.iloc[-1] = True
hold_roll_pf = vbt.Portfolio.from_signals(close_roll, hold_entries_roll, hold_exits_roll, freq=freq)
hold_roll_perf = hold_roll_pf.deep_getattr(metric)

# Performance Comparison
print("Mean Performance:")
print(f"DMAC: {dmac_roll_perf.mean():.4f}")
print(f"Hold: {hold_roll_perf.mean():.4f}")
print(f"Random: {rand_roll_perf.mean():.4f}")

# Cumulative distribution comparison
pd.DataFrame({
    'Random Strategy': rand_roll_perf,
    'Hold Strategy': hold_roll_perf,
    'DMAC Strategy': dmac_roll_perf,
}).vbt.histplot(
    xaxis_title=metric,
    yaxis_title='Cumulative # of tests',
    trace_kwargs=dict(cumulative_enabled=True)
).show_svg()

# Time series performance comparison
pd.DataFrame({
    'Random strategy': rand_roll_perf.groupby('split_idx').mean(),
    'Hold strategy': hold_roll_perf.groupby('split_idx').mean(),
    'DMAC strategy': dmac_roll_perf.groupby('split_idx').mean()
}).vbt.plot(
    xaxis_title='Split index',
    yaxis_title=f'Mean {metric}'
).show_svg()
```

## Key Implementation Patterns for Genetic Algorithms

### 1. Universal Parameter Space
```python
# Parameters suitable for genetic evolution
GENETIC_PARAMETER_SPACE = {
    'fast_window': np.arange(2, 100),     # Fast MA period
    'slow_window': np.arange(2, 100),     # Slow MA period
    'ma_type': ['sma', 'ema', 'wma'],     # Moving average type
    'signal_filter': [0.0, 0.01, 0.02],  # Minimum signal strength
    'volatility_filter': [0.5, 1.0, 2.0] # Volatility-based filtering
}

# Genetic encoding example
def create_genetic_dmac_strategy(genome):
    """Convert genetic genome to DMAC strategy parameters"""
    return {
        'fast_window': int(genome[0] * 98) + 2,  # Scale to 2-100
        'slow_window': int(genome[1] * 98) + 2,  # Scale to 2-100
        'ma_type': ['sma', 'ema', 'wma'][int(genome[2] * 3)],
        'signal_filter': genome[3] * 0.02,       # Scale to 0-0.02
        'volatility_filter': genome[4] * 1.5 + 0.5  # Scale to 0.5-2.0
    }
```

### 2. Cross-Asset Application Pattern
```python
def universal_dmac_strategy(price_data, genetic_params):
    """Universal DMAC that works across any asset"""
    
    # Extract genetic parameters
    fast_window = genetic_params['fast_window']
    slow_window = genetic_params['slow_window']
    ma_type = genetic_params['ma_type']
    signal_filter = genetic_params['signal_filter']
    volatility_filter = genetic_params['volatility_filter']
    
    # Calculate moving averages based on genetic type selection
    if ma_type == 'sma':
        fast_ma = vbt.MA.run(price_data, fast_window)
        slow_ma = vbt.MA.run(price_data, slow_window)
    elif ma_type == 'ema':
        fast_ma = vbt.MA.run(price_data, fast_window, ewm=True)
        slow_ma = vbt.MA.run(price_data, slow_window, ewm=True)
    # ... other MA types
    
    # Generate base signals
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # Apply genetic filters
    if signal_filter > 0:
        # Filter weak signals based on genetic threshold
        signal_strength = abs(fast_ma.ma - slow_ma.ma) / slow_ma.ma
        strong_entries = entries & (signal_strength > signal_filter)
        strong_exits = exits & (signal_strength > signal_filter)
        entries, exits = strong_entries, strong_exits
    
    if volatility_filter > 0:
        # Apply volatility-based filtering
        volatility = price_data.rolling(20).std()
        avg_volatility = volatility.rolling(100).mean()
        low_vol_mask = volatility < (avg_volatility * volatility_filter)
        entries = entries & low_vol_mask
        exits = exits & low_vol_mask
    
    return entries, exits

# Multi-asset application
def apply_universal_dmac_across_assets(asset_data_dict, genetic_params):
    """Apply genetic DMAC strategy across multiple assets"""
    results = {}
    
    for asset_name, price_data in asset_data_dict.items():
        entries, exits = universal_dmac_strategy(price_data, genetic_params)
        pf = vbt.Portfolio.from_signals(price_data, entries, exits)
        results[asset_name] = {
            'portfolio': pf,
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'max_drawdown': pf.max_drawdown()
        }
    
    return results
```

### 3. Genetic Fitness Evaluation
```python
def evaluate_genetic_dmac_fitness(genome, asset_data_dict, validation_period_days=365):
    """Evaluate genetic DMAC strategy fitness across multiple assets and time periods"""
    
    # Convert genome to strategy parameters
    genetic_params = create_genetic_dmac_strategy(genome)
    
    # Apply strategy across all assets
    asset_results = apply_universal_dmac_across_assets(asset_data_dict, genetic_params)
    
    # Calculate multi-objective fitness
    total_returns = [result['total_return'] for result in asset_results.values()]
    sharpe_ratios = [result['sharpe_ratio'] for result in asset_results.values()]
    max_drawdowns = [result['max_drawdown'] for result in asset_results.values()]
    
    # Fitness components
    mean_return = np.mean(total_returns)
    mean_sharpe = np.mean(sharpe_ratios)
    mean_drawdown = np.mean(max_drawdowns)
    consistency = np.std(total_returns)  # Lower is better
    
    # Multi-objective fitness (maximize return & Sharpe, minimize drawdown & inconsistency)
    fitness = (
        0.3 * mean_return +           # 30% weight on returns
        0.4 * mean_sharpe +           # 40% weight on risk-adjusted returns
        0.2 * (1 / mean_drawdown) +   # 20% weight on drawdown protection
        0.1 * (1 / consistency)       # 10% weight on consistency across assets
    )
    
    return fitness
```

## Key Insights for Cross-Asset Momentum Strategies

### 1. Universal Signal Generation
- **Moving Average Crossovers work universally** across different asset classes
- **Parameter ranges (2-100 periods)** provide sufficient genetic search space
- **Multiple MA types (SMA, EMA, WMA)** offer genetic diversity for optimization

### 2. Genetic Algorithm Integration Points
- **Parameter encoding**: Direct genome-to-parameter mapping with scaling
- **Multi-asset fitness**: Evaluate strategy performance across entire asset universe
- **Temporal robustness**: Use rolling windows to test genetic strategies across different market regimes

### 3. Performance Optimization Patterns
- **Vectorized computation**: Process all parameter combinations simultaneously
- **Memory management**: Use time buffers and careful indexing to prevent NaN issues
- **3D analysis**: Time × Parameter1 × Parameter2 cubes for comprehensive optimization

### 4. Strategy Evolution Framework
```python
# Complete genetic evolution framework
class GeneticDMACEvolution:
    def __init__(self, asset_data, population_size=100, generations=50):
        self.asset_data = asset_data
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """Create random initial population"""
        return np.random.random((self.population_size, 5))  # 5 genetic parameters
    
    def evolve(self):
        """Main genetic algorithm evolution loop"""
        for generation in range(self.generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for genome in self.population:
                fitness = evaluate_genetic_dmac_fitness(genome, self.asset_data)
                fitness_scores.append(fitness)
            
            # Selection, crossover, mutation
            self.population = self.genetic_operators(self.population, fitness_scores)
            
            # Track best performer
            best_idx = np.argmax(fitness_scores)
            best_genome = self.population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        return best_genome, best_fitness
```

This implementation provides a complete foundation for genetic algorithm-based dual moving average crossover strategies that can be applied universally across crypto assets on Hyperliquid.