# Vectorbt Installation and Usage Guide

**Source URLs**: 
- https://vectorbt.dev/getting-started/installation/
- https://vectorbt.dev/getting-started/usage/
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Complete installation and usage documentation with examples

## Installation Options

### 1. Standard Installation with pip

```bash
pip install -U vectorbt
```

### 2. Full Installation (with optional dependencies)

```bash
pip install -U "vectorbt[full]"
```

### 3. Docker Installation

```bash
# Pull and run Docker image
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work polakowo/vectorbt
```

**Available Docker Images**:
- **polakowo/vectorbt**: vanilla version (default)
- **polakowo/vectorbt-full**: full version (with optional dependencies)

### 4. Development Installation from Git

```bash
git clone git@github.com:polakowo/vectorbt.git vectorbt
pip install -e vectorbt
```

## Quick Usage Examples

### 1. Basic Buy and Hold Strategy

```python
import vectorbt as vbt

# Download Bitcoin price data
price = vbt.YFData.download('BTC-USD').get('Close')

# Simple buy and hold with $100 initial investment
pf = vbt.Portfolio.from_holding(price, init_cash=100)
pf.total_profit()
# Output: 8961.008555963961
```

### 2. Moving Average Crossover Strategy

```python
# Calculate moving averages
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)

# Generate signals
entries = fast_ma.ma_crossed_above(slow_ma)  # Buy signal
exits = fast_ma.ma_crossed_below(slow_ma)    # Sell signal

# Backtest the strategy
pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
pf.total_profit()
# Output: 16423.251963801864
```

### 3. Random Strategy Testing (Monte Carlo)

```python
import numpy as np

# Multi-asset testing
symbols = ["BTC-USD", "ETH-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

# Generate 1,000 random strategies
n = np.random.randint(10, 101, size=1000).tolist()
pf = vbt.Portfolio.from_random_signals(price, n=n, init_cash=100, seed=42)

# Analyze results
mean_expectancy = pf.trades.expectancy().groupby(['randnx_n', 'symbol']).mean()
fig = mean_expectancy.unstack().vbt.scatterplot(
    xaxis_title='randnx_n', 
    yaxis_title='mean_expectancy'
)
fig.show()
```

### 4. Hyperparameter Optimization (10,000 combinations)

```python
# Multi-asset data
symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

# Test all window combinations (2-100)
windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])

# Generate signals
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# Backtest with fees
pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')
pf = vbt.Portfolio.from_signals(price, entries, exits, **pf_kwargs)

# Interactive heatmap visualization
fig = pf.total_return().vbt.heatmap(
    x_level='fast_window', 
    y_level='slow_window', 
    slider_level='symbol', 
    symmetric=True,
    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%'))
)
fig.show()
```

### 5. Individual Strategy Analysis

```python
# Analyze specific strategy configuration
strategy_stats = pf[(10, 20, 'ETH-USD')].stats()
print(strategy_stats)

# Output includes:
# Start                          2015-08-07 00:00:00+00:00
# End                            2021-08-01 00:00:00+00:00
# Period                                2183 days 00:00:00
# Start Value                                        100.0
# End Value                                  620402.791485
# Total Return [%]                           620302.791485
# Win Rate [%]                                   52.830189
# Sharpe Ratio                                    2.041211
# Max Drawdown [%]                               70.734951

# Portfolio visualization
pf[(10, 20, 'ETH-USD')].plot().show()
```

### 6. Advanced Visualization - Bollinger Bands Animation

```python
# Multi-asset Bollinger Bands analysis
symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
price = vbt.YFData.download(symbols, period='6mo', missing_index='drop').get('Close')
bbands = vbt.BBANDS.run(price)

def plot(index, bbands):
    bbands = bbands.loc[index]
    fig = vbt.make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=('%B', 'Bandwidth')
    )
    fig.update_layout(template='vbt_dark', showlegend=False, width=750, height=400)
    
    # %B heatmap
    bbands.percent_b.vbt.ts_heatmap(
        trace_kwargs=dict(zmin=0, zmid=0.5, zmax=1, colorscale='Spectral'),
        add_trace_kwargs=dict(row=1, col=1), fig=fig
    )
    
    # Bandwidth heatmap
    bbands.bandwidth.vbt.ts_heatmap(
        add_trace_kwargs=dict(row=2, col=1), fig=fig
    )
    return fig

# Generate animated GIF
vbt.save_animation('bbands.gif', bbands.wrapper.index, plot, bbands, 
                  delta=90, step=3, fps=3)
```

## Key Features for Genetic Algorithm Integration

### 1. Vectorized Operations
- **Massive Parallelization**: Test thousands of strategies simultaneously
- **Multi-dimensional Analysis**: Assets × Parameters × Time periods
- **Memory Efficient**: NumPy-based data structures

### 2. Data Integration
- **Yahoo Finance**: Built-in `vbt.YFData.download()` functionality
- **Custom Data**: Compatible with any pandas DataFrame
- **Missing Data Handling**: Automatic alignment and filling

### 3. Technical Indicators
- **Moving Averages**: `vbt.MA.run()` with configurable windows
- **Bollinger Bands**: `vbt.BBANDS.run()` with %B and bandwidth
- **RSI, MACD, ATR**: Comprehensive indicator library
- **Custom Indicators**: Build your own using the factory pattern

### 4. Portfolio Management
- **Signal-based Backtesting**: `Portfolio.from_signals()`
- **Order-based Backtesting**: `Portfolio.from_orders()`
- **Random Signal Testing**: `Portfolio.from_random_signals()`
- **Buy and Hold**: `Portfolio.from_holding()`

### 5. Performance Metrics
- **Returns**: Total, annualized, benchmark-relative
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown, drawdown duration
- **Trade Analysis**: Win rate, profit factor, expectancy

### 6. Visualization
- **Interactive Charts**: Plotly-based portfolio plots
- **Heatmaps**: Multi-dimensional parameter analysis
- **Time Series**: Price and indicator plotting
- **Animation**: GIF generation for temporal analysis

## Implementation Advantages for Quant Trading Organism

### Perfect Backtesting Engine
1. **Speed**: Numba-accelerated operations for genetic algorithm fitness evaluation
2. **Scale**: Handle hundreds of evolved strategies simultaneously
3. **Flexibility**: Universal API that works with any strategy configuration
4. **Accuracy**: Professional-grade backtesting with realistic trading costs

### Integration Points
- **Strategy Evaluation**: Rapid fitness calculation for DEAP individuals
- **Parameter Optimization**: Vectorized testing of evolved parameters
- **Multi-timeframe Validation**: Test strategies across different market periods
- **Performance Comparison**: Comprehensive statistics for strategy selection

### Production Readiness
- **Battle-tested**: Used by professional quant traders
- **Documentation**: Comprehensive API reference and examples
- **Community**: Active development and support
- **Extensibility**: Easy to customize and extend functionality

## Dependencies and Troubleshooting

### Core Dependencies
- **pandas**: DataFrame operations
- **numpy**: Numerical computing
- **numba**: JIT compilation for speed
- **plotly**: Interactive visualizations

### Optional Dependencies (full installation)
- **TA-Lib**: Additional technical indicators
- **yfinance**: Yahoo Finance data
- **jupyterlab**: Notebook environment

### Common Issues
- **TA-Lib Installation**: May require system dependencies
- **Apple M1**: Special installation procedures documented
- **Jupyter Support**: Plotly integration configuration

The combination of vectorbt's speed, flexibility, and comprehensive feature set makes it the ideal backtesting engine for genetic algorithm-driven strategy discovery in the Quant Trading Organism.