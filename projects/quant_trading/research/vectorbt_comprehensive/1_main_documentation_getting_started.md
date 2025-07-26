# Vectorbt Getting Started Documentation

**Source URL**: https://vectorbt.dev/
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Complete main documentation with examples and code snippets

## What is vectorbt?

vectorbt is a Python package for quantitative analysis that takes a novel approach to backtesting: it operates entirely on pandas and NumPy objects, and is accelerated by [Numba](https://github.com/numba/numba) to analyze any data at speed and scale. This allows for testing of many thousands of strategies in **seconds**.

In contrast to other backtesters, vectorbt represents complex data as (structured) NumPy arrays. This enables superfast computation using vectorized operations with NumPy and non-vectorized but dynamically compiled operations with Numba. It also integrates [Plotly](https://github.com/plotly/plotly.py) and [Jupyter Widgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin to Tableau right in the Jupyter notebook. Due to high performance, vectorbt can process large amounts of data even without GPU and parallelization and enables the user to interact with data-hungry widgets without significant delays.

## Core Capabilities

With vectorbt, you can:

- **Backtest strategies in a couple of lines of Python code**
- **Enjoy the best of both worlds**: the ecosystem of Python and the speed of C
- **Retain full control over execution and your data** (as opposed to web-based services such as TradingView)
- **Optimize your trading strategy** against many parameters, assets, and periods in one go
- **Uncover hidden patterns in financial markets**
- **Analyze time series and engineer new features for ML models**
- **Supercharge pandas and your favorite tools** to run much faster
- **Visualize strategy performance** using interactive charts and dashboards (both in Jupyter and browser)
- **Fetch and process data periodically**, send Telegram notifications, and more

## Why vectorbt?

While there are many great backtesting packages for Python, vectorbt combines an extremely fast backtester and a data science tool: it excels at processing performance and offers interactive tools to explore complex phenomena in trading. With it, you can traverse a huge number of strategy configurations, time periods, and instruments in little time, to explore where your strategy performs best and to uncover hidden patterns in data.

## How it works

vectorbt was implemented to address common performance shortcomings of backtesting libraries. It builds upon the idea that each instance of a trading strategy can be represented in a vectorized form, so multiple strategy instances can be packed into a single multi-dimensional array, processed in a highly efficient manner, and compared easily.

Thanks to the time-series nature of trading data, most of the aspects related to backtesting can be translated into vectors. Instead of processing one element at a time, vectorization allows us to avoid naive looping and perform the same operation on all elements at the same time. The path-dependency problem related to vectorization is solved by using Numba - it allows both writing iterative code and compiling slow Python loops to be run at the native machine code speed.

## Complete Example: Dual Moving Average Crossover (DMAC)

### Basic Single Strategy

```python
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime
>>> import vectorbt as vbt

>>> # Prepare data
>>> start = '2019-01-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> btc_price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')

>>> btc_price
Date
2019-01-01 00:00:00+00:00    3843.520020
2019-01-02 00:00:00+00:00    3943.409424
2019-01-03 00:00:00+00:00    3836.741211
...
2019-12-30 00:00:00+00:00    7292.995117
2019-12-31 00:00:00+00:00    7193.599121
2020-01-01 00:00:00+00:00    7200.174316
Freq: D, Name: Close, Length: 366, dtype: float64
```

### Generate Trading Signals

```python
>>> fast_ma = vbt.MA.run(btc_price, 10, short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, 20, short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> exits = fast_ma.ma_crossed_below(slow_ma)

>>> pf = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> pf.total_return()
0.636680693047752
```

### Multiple Strategy Instances - Vectorized Processing

```python
>>> # Multiple strategy instances: (10, 30) and (20, 30)
>>> fast_ma = vbt.MA.run(btc_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> exits = fast_ma.ma_crossed_below(slow_ma)

>>> pf = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> pf.total_return()
fast_window  slow_window
10           30             0.848840
20           30             0.543411
Name: total_return, dtype: float64
```

### Multi-Asset Strategy Testing

```python
>>> # Multiple strategy instances and instruments
>>> eth_price = vbt.YFData.download('ETH-USD', start=start, end=end).get('Close')
>>> comb_price = btc_price.vbt.concat(eth_price,
...     keys=pd.Index(['BTC', 'ETH'], name='symbol'))
>>> comb_price.vbt.drop_levels(-1, inplace=True)

>>> fast_ma = vbt.MA.run(comb_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(comb_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> exits = fast_ma.ma_crossed_below(slow_ma)

>>> pf = vbt.Portfolio.from_signals(comb_price, entries, exits)
>>> pf.total_return()
fast_window  slow_window  symbol
10           30           BTC       0.848840
                          ETH       0.244204
20           30           BTC       0.543411
                          ETH      -0.319102
Name: total_return, dtype: float64

>>> mean_return = pf.total_return().groupby('symbol').mean()
>>> mean_return.vbt.barplot(xaxis_title='Symbol', yaxis_title='Mean total return')
```

### Multi-Period Backtesting

```python
>>> # Multiple strategy instances, instruments, and time periods
>>> mult_comb_price, _ = comb_price.vbt.range_split(n=2)

>>> fast_ma = vbt.MA.run(mult_comb_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(mult_comb_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> exits = fast_ma.ma_crossed_below(slow_ma)

>>> pf = vbt.Portfolio.from_signals(mult_comb_price, entries, exits, freq='1D')
>>> pf.total_return()
fast_window  slow_window  split_idx  symbol
10           30           0          BTC       1.632259
                                     ETH       0.946786
                          1          BTC      -0.288720
                                     ETH      -0.308387
20           30           0          BTC       1.721449
                                     ETH       0.343274
                          1          BTC      -0.418280
                                     ETH      -0.257947
Name: total_return, dtype: float64
```

## Key Features for Quant Trading Implementation

### 1. Vectorized Operations
- **Performance**: Test thousands of strategies in seconds
- **Scale**: Handle multiple assets, time periods, and parameters simultaneously
- **Memory Efficient**: Structured NumPy arrays for optimal data representation

### 2. Technical Indicators
- **Moving Averages**: `vbt.MA.run()` with configurable windows
- **Signal Generation**: `ma_crossed_above()`, `ma_crossed_below()` methods
- **Custom Indicators**: Build complex technical analysis tools

### 3. Portfolio Management
- **Signal-based**: `vbt.Portfolio.from_signals()` for entry/exit based strategies
- **Performance Metrics**: Built-in calculation of returns, Sharpe ratio, drawdowns
- **Multi-dimensional Analysis**: Group and analyze by any combination of parameters

### 4. Data Integration
- **Yahoo Finance**: `vbt.YFData.download()` for historical price data
- **Custom Data Sources**: Compatible with any pandas DataFrame
- **Time Series Manipulation**: Built-in splitting, concatenation, and transformation tools

### 5. Visualization
- **Interactive Charts**: Plotly integration for advanced visualizations
- **Jupyter Widgets**: Real-time dashboard capabilities
- **Performance Plots**: Built-in charting for strategy analysis

## Implementation Advantages for Genetic Algorithm Integration

### Perfect Fit for Strategy Evolution
1. **Massive Parallel Evaluation**: Evaluate hundreds of evolved strategies simultaneously
2. **Parameter Optimization**: Test multiple indicator parameters in single run
3. **Multi-objective Analysis**: Compare strategies across multiple metrics (Sharpe, drawdown, win rate)
4. **Rapid Iteration**: Fast enough for genetic algorithm fitness evaluation
5. **Flexible Strategy Representation**: Easy integration with DEAP's genetic programming trees

### Performance Characteristics
- **Speed**: Numba-accelerated operations for real-time strategy evaluation
- **Memory**: Efficient array-based storage for large strategy populations
- **Scalability**: Handles thousands of strategy combinations without performance degradation
- **Flexibility**: Universal pipeline that works with any strategy configuration

## Integration with Quant Trading Organism

Vectorbt provides the **perfect backtesting engine** for the genetic algorithm evolution system:

1. **Strategy Evaluation**: Rapid fitness calculation for evolved strategies
2. **Multi-timeframe Validation**: Test strategies across different market periods
3. **Performance Metrics**: Comprehensive statistics for strategy selection
4. **Parameter Optimization**: Vectorized testing of indicator combinations
5. **Production-Ready**: Battle-tested library with extensive documentation

The combination of vectorbt's speed and DEAP's genetic programming creates an optimal environment for discovering profitable trading strategies through evolutionary computation.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.