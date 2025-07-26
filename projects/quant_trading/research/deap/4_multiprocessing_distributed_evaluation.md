# DEAP Multiprocessing and Distributed Evaluation

**Source URL**: https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Production-ready parallel processing implementation guide

## Overview

DEAP supports distributed operations through serialization (pickling) for parallel evaluation. All distributed objects (functions, arguments, individuals, parameters) must be pickleable. Modifications on distant processing units are only available to other units through explicit communication via function arguments and return values.

## SCOOP (Scalable Concurrent Operations in Python)

### What is SCOOP?
- **Distributed Task Module**: Enables concurrent parallel programming on various environments
- **Scalability**: From heterogeneous grids to supercomputers
- **Interface**: Similar to `concurrent.futures` module (Python 3.2+)
- **Core Functions**: `submit()` and `map()` for efficient computation distribution

### SCOOP Implementation

```python
from scoop import futures
from deap import base, creator, tools, algorithms

# Replace default map with SCOOP's distributed map
toolbox.register("map", futures.map)

def main():
    # Your evolutionary algorithm code here
    population = toolbox.population(n=50)
    
    # SCOOP automatically distributes evaluation across processors
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, 
                                  ngen=40, verbose=True)
    
    return pop

# CRITICAL: Must run from main() when using SCOOP
if __name__ == "__main__":
    main()
```

### Running SCOOP Programs

```bash
# Run on all available processors
$ python -m scoop your_program.py

# Specify number of workers
$ python -m scoop -n 4 your_program.py

# Distributed across multiple hosts
$ python -m scoop --hostfile hosts.txt your_program.py
```

**Key Requirements**:
- **Must use main() function**: SCOOP requires program entry point
- **Automatic distribution**: No code changes needed beyond map registration
- **Cross-platform**: Works on local machines and distributed clusters

## Multiprocessing Module

### Standard Python Multiprocessing

```python
import multiprocessing
from deap import base, creator, tools, algorithms

# Create process pool
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

def main():
    population = toolbox.population(n=50)
    
    # Evaluations automatically distributed across pool
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                  ngen=40, verbose=True)
    
    # Clean up pool
    pool.close()
    pool.join()
    
    return pop

# CRITICAL: Windows requires __main__ protection
if __name__ == "__main__":
    main()
```

### Advanced Pool Configuration

```python
# Specify worker count
pool = multiprocessing.Pool(processes=8)

# Use context manager for automatic cleanup
def main():
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        
        population = toolbox.population(n=50)
        pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                      ngen=40, verbose=True)
    return pop
```

## Platform-Specific Considerations

### Windows Requirements
```python
# Windows REQUIRES __main__ protection due to process initialization method
if __name__ == "__main__":
    # All multiprocessing code must be in this block
    main()
```

### Python Version Compatibility
- **Python 2.6**: Multiprocessing module available, but partial function pickling issues
- **Python 2.7+/3.1+**: Full partial function pickling support
- **Lambda Functions**: Cannot be pickled - use named functions instead

### Pickling Limitations

```python
# ❌ This won't work - lambda functions can't be pickled
toolbox.register("evaluate", lambda x: sum(x))

# ✅ This works - named functions can be pickled
def evaluate_individual(individual):
    return sum(individual),

toolbox.register("evaluate", evaluate_individual)
```

## Trading Strategy Parallel Evaluation Implementation

### Backtesting Pool Setup

```python
import multiprocessing as mp
from functools import partial
import pandas as pd

class TradingStrategyEvaluator:
    def __init__(self, historical_data, validation_periods):
        self.historical_data = historical_data
        self.validation_periods = validation_periods
    
    def evaluate_strategy_parallel(self, strategy_population):
        """Evaluate population of trading strategies in parallel."""
        
        # Use all available CPU cores
        with mp.Pool() as pool:
            # Create partial function with fixed data
            eval_func = partial(self.backtest_strategy, 
                              data=self.historical_data,
                              periods=self.validation_periods)
            
            # Distribute evaluation across workers
            fitness_values = pool.map(eval_func, strategy_population)
            
            # Assign fitness to individuals
            for individual, fitness in zip(strategy_population, fitness_values):
                individual.fitness.values = fitness
        
        return strategy_population
    
    @staticmethod
    def backtest_strategy(individual, data, periods):
        """Static method for strategy backtesting - must be pickleable."""
        try:
            # Compile GP tree to executable function
            from deap import gp
            strategy_func = gp.compile(individual, individual.pset)
            
            results = []
            for start, end in periods:
                period_data = data[start:end]
                
                # Generate signals
                signals = []
                for i in range(100, len(period_data)):
                    window = period_data.iloc[i-100:i]
                    signal = strategy_func(window)
                    signals.append(1 if signal else 0)
                
                # Calculate metrics
                returns = calculate_returns(signals, period_data[100:])
                sharpe = calculate_sharpe_ratio(returns)
                drawdown = calculate_max_drawdown(returns)
                win_rate = calculate_win_rate(returns)
                
                results.append((sharpe, drawdown, win_rate))
            
            # Average across validation periods
            avg_sharpe = sum(r[0] for r in results) / len(results)
            avg_drawdown = sum(r[1] for r in results) / len(results)
            avg_win_rate = sum(r[2] for r in results) / len(results)
            
            return avg_sharpe, avg_drawdown, avg_win_rate
            
        except Exception as e:
            # Return poor fitness for invalid strategies
            return -999.0, 1.0, 0.0

# Helper functions (must be module-level for pickling)
def calculate_returns(signals, price_data):
    """Calculate strategy returns based on signals."""
    returns = []
    for i in range(1, len(signals)):
        if signals[i-1] == 1:  # Long position
            ret = (price_data.iloc[i]['close'] - price_data.iloc[i-1]['close']) / price_data.iloc[i-1]['close']
            returns.append(ret)
    return returns

def calculate_sharpe_ratio(returns):
    """Calculate annualized Sharpe ratio."""
    if not returns or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown."""
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return abs(np.min(drawdown))

def calculate_win_rate(returns):
    """Calculate percentage of winning trades."""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)
```

### DEAP Integration

```python
# Register parallel evaluation
evaluator = TradingStrategyEvaluator(market_data, validation_periods)
toolbox.register("map", evaluator.evaluate_strategy_parallel)

# Or use standard multiprocessing
def setup_parallel_evaluation():
    pool = mp.Pool()
    toolbox.register("map", pool.map)
    
    # Register pickleable evaluation function
    toolbox.register("evaluate", TradingStrategyEvaluator.backtest_strategy,
                    data=market_data, periods=validation_periods)
    
    return pool
```

### SCOOP for Distributed Trading Strategy Evolution

```python
from scoop import futures
import scoop

def distributed_trading_evolution():
    """Run trading strategy evolution across distributed workers."""
    
    toolbox.register("map", futures.map)
    
    def main():
        # Load market data on each worker
        market_data = load_market_data()
        
        # Create initial population
        population = toolbox.population(n=200)  # Larger population for distributed
        
        # Evolution with distributed evaluation
        pop, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=0.7, mutpb=0.2, ngen=100,
            stats=stats, halloffame=hof, verbose=True
        )
        
        return pop, logbook
    
    if __name__ == "__main__":
        final_population, evolution_log = main()
        return final_population, evolution_log

# Run with SCOOP
# $ python -m scoop -n 16 trading_evolution.py
```

## Performance Optimization Strategies

### 1. Evaluation Granularity
```python
# Batch evaluation for efficiency
def batch_evaluate(individuals, batch_size=10):
    """Evaluate individuals in batches to reduce overhead."""
    results = []
    for i in range(0, len(individuals), batch_size):
        batch = individuals[i:i+batch_size]
        batch_results = pool.map(evaluate_individual, batch)
        results.extend(batch_results)
    return results
```

### 2. Memory Management
```python
# Use shared memory for large datasets
from multiprocessing import shared_memory

def create_shared_market_data(data):
    """Share market data across processes to reduce memory usage."""
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    return shm.name, data.shape, data.dtype
```

### 3. Process Pool Reuse
```python
# Global pool for reuse across generations
_global_pool = None

def get_evaluation_pool():
    global _global_pool
    if _global_pool is None:
        _global_pool = mp.Pool()
    return _global_pool

def cleanup_pool():
    global _global_pool
    if _global_pool is not None:
        _global_pool.close()
        _global_pool.join()
        _global_pool = None
```

## Key Benefits for Quant Trading Evolution

1. **Massive Parallelization**: Evaluate hundreds of strategies simultaneously across multiple CPU cores
2. **Distributed Computing**: Scale evaluation across multiple machines using SCOOP
3. **Fault Tolerance**: Individual strategy evaluation failures don't crash entire evolution
4. **Resource Efficiency**: Automatic load balancing across available processors
5. **Scalability**: Easy transition from local multiprocessing to distributed clusters

## Integration with Quant Organism Architecture

The parallel evaluation capabilities perfectly support the quant trading organism's needs:

- **Strategy Population**: Evaluate 200+ strategies simultaneously
- **Multi-timeframe Validation**: Parallel backtesting across different market periods
- **Out-of-sample Testing**: Concurrent validation on multiple data splits
- **Real-time Adaptation**: Rapid re-evaluation when market conditions change
- **Cloud Scaling**: SCOOP enables expansion to distributed cloud computing

This distributed evaluation framework ensures the genetic algorithm can efficiently explore the vast strategy space while maintaining real-time responsiveness for live trading decisions.