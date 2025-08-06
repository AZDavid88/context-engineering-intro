# Vectorbt Signals and Portfolio API Documentation

**Source URLs**: 
- https://vectorbt.dev/api/signals/
- https://vectorbt.dev/api/portfolio/
- https://vectorbt.dev/api/indicators/
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Core API modules for signal generation and portfolio backtesting

## Signals Package Overview

The signals package provides modules for working with entry and exit signals, which are fundamental building blocks for trading strategies.

### Core Signal Modules

#### 1. vectorbt.signals.accessors
- **Purpose**: Pandas accessor methods for signal manipulation
- **Usage**: Extends pandas Series/DataFrame with signal-specific operations
- **Key Methods**: Signal filtering, transformation, and analysis

#### 2. vectorbt.signals.enums  
- **Purpose**: Enumeration constants for signal types and configurations
- **Usage**: Standardized signal states and parameters
- **Key Constants**: Signal directions, modes, and validation states

#### 3. vectorbt.signals.factory
- **Purpose**: Factory pattern for creating custom signal generators
- **Usage**: Build reusable signal generation components
- **Key Features**: Template-based signal creation and parameterization

#### 4. vectorbt.signals.generators
- **Purpose**: Pre-built signal generation functions
- **Usage**: Common trading signal patterns and strategies
- **Key Generators**: Entry/exit signal combinations, pattern-based signals

#### 5. vectorbt.signals.nb
- **Purpose**: Numba-compiled functions for high-performance signal processing
- **Usage**: Low-level signal operations with optimal speed
- **Key Features**: Vectorized signal computation, memory-efficient processing

## Portfolio Package Overview

The portfolio package provides comprehensive backtesting and portfolio analysis capabilities.

### Core Portfolio Modules

#### 1. vectorbt.portfolio.base
- **Purpose**: Core Portfolio class with backtesting functionality
- **Usage**: Main interface for strategy backtesting and analysis
- **Key Methods**: `from_signals()`, `from_orders()`, `from_holding()`

#### 2. vectorbt.portfolio.decorators
- **Purpose**: Method decorators for portfolio operations
- **Usage**: Enhance portfolio methods with caching, validation
- **Key Features**: Performance optimization, parameter validation

#### 3. vectorbt.portfolio.enums
- **Purpose**: Portfolio-specific enumerations and constants
- **Usage**: Order types, direction constants, status codes
- **Key Constants**: BUY/SELL directions, order execution states

#### 4. vectorbt.portfolio.logs
- **Purpose**: Portfolio execution logging and audit trails
- **Usage**: Track order execution, position changes, cash flows
- **Key Features**: Detailed transaction history, performance attribution

#### 5. vectorbt.portfolio.nb
- **Purpose**: High-performance Numba-compiled portfolio functions
- **Usage**: Core backtesting engine with optimal speed
- **Key Features**: Vectorized position tracking, P&L calculation

#### 6. vectorbt.portfolio.orders
- **Purpose**: Order management and execution simulation
- **Usage**: Model realistic order execution with slippage, fees
- **Key Features**: Order sizing, timing, execution modeling

#### 7. vectorbt.portfolio.trades
- **Purpose**: Trade analysis and performance metrics
- **Usage**: Analyze completed trades, win/loss statistics
- **Key Metrics**: Trade duration, P&L, win rate, expectancy

## Indicators Package Overview

Technical indicators are essential for strategy development and signal generation.

### Core Indicator Features

#### 1. Built-in Indicator Libraries
```python
# TA-Lib integration
vbt.talib(func_name, *args, **kwargs)

# pandas-ta integration  
vbt.pandas_ta(func_name, *args, **kwargs)

# Custom TA integration
vbt.ta(func_name, *args, **kwargs)
```

#### 2. Indicator Factory Pattern
- **Purpose**: Create custom indicators with consistent API
- **Usage**: Build reusable technical analysis components
- **Key Features**: Parameter optimization, vectorized computation

#### 3. Basic Indicators Module
- **Moving Averages**: SMA, EMA, WMA with configurable windows
- **Oscillators**: RSI, MACD, Stochastic with standard parameters
- **Volatility**: Bollinger Bands, ATR, standard deviation
- **Volume**: Volume-based indicators and analysis

#### 4. Configuration Management
- **Purpose**: Standardized indicator parameter management
- **Usage**: Define and reuse indicator configurations
- **Key Features**: Parameter validation, optimization support

## Key Implementation Patterns for Genetic Algorithm Integration

### 1. Signal Generation for Evolved Strategies

```python
# Example: GP-evolved signal generation
def generate_evolved_signals(price_data, gp_individual):
    """Convert genetic programming tree to vectorbt signals."""
    
    # Compile GP tree to executable function
    strategy_func = gp.compile(gp_individual, pset)
    
    # Generate entry signals
    entries = pd.Series(False, index=price_data.index)
    for i in range(50, len(price_data)):  # Skip initial period for indicators
        window_data = price_data.iloc[i-50:i+1]
        try:
            signal = strategy_func(window_data)
            entries.iloc[i] = bool(signal)
        except:
            entries.iloc[i] = False
    
    # Generate exit signals (opposite of entries for simplicity)
    exits = entries.shift(1) & ~entries
    
    return entries, exits
```

### 2. Vectorized Strategy Evaluation

```python
# Example: Evaluate multiple evolved strategies simultaneously
def evaluate_strategy_population(price_data, gp_population):
    """Evaluate entire population of GP strategies using vectorbt."""
    
    # Convert all strategies to signal matrices
    all_entries = pd.DataFrame(index=price_data.index)
    all_exits = pd.DataFrame(index=price_data.index)
    
    for i, individual in enumerate(gp_population):
        entries, exits = generate_evolved_signals(price_data, individual)
        all_entries[f'strategy_{i}'] = entries
        all_exits[f'strategy_{i}'] = exits
    
    # Vectorized backtesting
    portfolio = vbt.Portfolio.from_signals(
        price_data, 
        all_entries, 
        all_exits,
        init_cash=10000,
        fees=0.001,
        freq='1D'
    )
    
    # Return fitness metrics for all strategies
    return {
        'total_returns': portfolio.total_return(),
        'sharpe_ratios': portfolio.sharpe_ratio(),
        'max_drawdowns': portfolio.max_drawdown(),
        'win_rates': portfolio.trades.win_rate()
    }
```

### 3. Multi-Objective Fitness Calculation

```python
# Example: DEAP-compatible fitness evaluation
def calculate_fitness_metrics(portfolio_results, strategy_id):
    """Calculate multi-objective fitness for genetic algorithm."""
    
    # Extract metrics for specific strategy
    total_return = portfolio_results['total_returns'][strategy_id]
    sharpe_ratio = portfolio_results['sharpe_ratios'][strategy_id]
    max_drawdown = portfolio_results['max_drawdowns'][strategy_id]
    win_rate = portfolio_results['win_rates'][strategy_id]
    
    # Multi-objective fitness tuple
    # Maximize: total_return, sharpe_ratio, win_rate
    # Minimize: max_drawdown (negative for minimization)
    fitness = (
        total_return,
        sharpe_ratio, 
        -max_drawdown,  # Negative because we want to minimize drawdown
        win_rate
    )
    
    return fitness
```

### 4. Advanced Portfolio Analysis

```python
# Example: Comprehensive strategy analysis
def analyze_evolved_strategy(price_data, gp_individual, detailed=True):
    """Detailed analysis of evolved trading strategy."""
    
    # Generate signals
    entries, exits = generate_evolved_signals(price_data, gp_individual)
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        price_data, entries, exits,
        init_cash=10000, fees=0.001, freq='1D'
    )
    
    if detailed:
        # Comprehensive statistics
        stats = portfolio.stats()
        trades = portfolio.trades.records_readable
        drawdowns = portfolio.drawdowns.records_readable
        
        return {
            'portfolio': portfolio,
            'stats': stats, 
            'trades': trades,
            'drawdowns': drawdowns,
            'entries': entries,
            'exits': exits
        }
    else:
        # Quick fitness metrics only
        return {
            'total_return': portfolio.total_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown()
        }
```

## Integration Benefits for Quant Trading Organism

### 1. Performance Advantages
- **Vectorized Operations**: Evaluate hundreds of strategies simultaneously
- **Numba Acceleration**: C-speed execution for fitness calculations
- **Memory Efficiency**: Optimized data structures for large populations

### 2. Flexibility
- **Signal-based Backtesting**: Perfect match for GP-generated entry/exit logic
- **Multi-timeframe Support**: Test strategies across different time periods
- **Custom Indicators**: Build domain-specific technical analysis tools

### 3. Realism
- **Transaction Costs**: Realistic fee and slippage modeling
- **Order Execution**: Proper order timing and sizing simulation
- **Risk Management**: Built-in position sizing and drawdown controls

### 4. Analysis Depth
- **Trade-level Analysis**: Detailed examination of individual trades
- **Performance Attribution**: Understanding strategy behavior
- **Risk Metrics**: Comprehensive risk and return statistics

### 5. Scalability
- **Multi-asset Testing**: Simultaneous testing across multiple instruments
- **Parameter Optimization**: Efficient hyperparameter space exploration
- **Production Ready**: Professional-grade backtesting infrastructure

## Key Methods for Strategy Evolution

### Signal Generation
- `vbt.Portfolio.from_signals()`: Primary method for signal-based backtesting
- `entries.vbt.signals.*`: Signal manipulation and analysis methods
- `vbt.signals.generators.*`: Pre-built signal generation patterns

### Performance Analysis  
- `portfolio.total_return()`: Overall strategy performance
- `portfolio.sharpe_ratio()`: Risk-adjusted returns
- `portfolio.max_drawdown()`: Maximum peak-to-trough decline
- `portfolio.trades.win_rate()`: Percentage of profitable trades

### Visualization
- `portfolio.plot()`: Interactive portfolio performance charts
- `entries.vbt.signals.plot()`: Signal visualization
- `returns.vbt.heatmap()`: Multi-dimensional parameter analysis

This comprehensive API provides all the tools necessary for implementing sophisticated genetic programming-based trading strategy evolution within the Quant Trading Organism framework.