# DEAP Examples and Implementation Patterns

**Source URL**: https://deap.readthedocs.io/en/master/examples/index.html
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Comprehensive example collection for various evolutionary approaches

## Examples Overview

DEAP provides documented examples of common evolutionary computation problems, covering multiple paradigms relevant to quant trading strategy evolution.

## Genetic Algorithm (GA) Examples

### Core GA Problems
- **One Max Problem**: Basic fitness optimization
- **One Max Problem: Short Version**: Minimal implementation
- **One Max Problem: Using Numpy**: Array-based individuals
- **Knapsack Problem: Inheriting from Set**: Custom data structures
- **Cooperative Coevolution**: Multi-population evolution
- **Non-dominated Sorting Genetic Algorithm III (NSGA-III)**: Multi-objective optimization

**Trading Relevance**: Portfolio optimization, multi-objective strategy selection (Sharpe vs. drawdown vs. consistency)

## Genetic Programming (GP) Examples

### Symbolic Evolution Problems
- **Symbolic Regression Problem: Introduction to GP**: Mathematical function discovery
- **Even-Parity Problem**: Boolean logic evolution
- **Multiplexer 3-8 Problem**: Digital circuit design
- **Artificial Ant Problem**: Behavioral strategy evolution
- **Spambase Problem: Strongly Typed GP**: Classification with type constraints

**Trading Relevance**: These directly apply to trading strategy evolution, especially symbolic regression for price prediction and artificial ant for market navigation strategies.

## Evolution Strategy (ES) Examples

### Continuous Optimization
- **Evolution Strategies Basics**: Real-valued parameter optimization
- **One Fifth Rule**: Adaptive mutation control
- **Covariance Matrix Adaptation Evolution Strategy**: Advanced parameter optimization
- **Controlling the Stopping Criteria: BI-POP CMA-ES**: Multi-restart strategies
- **Plotting Important Data: Visualizing the CMA Strategy**: Performance monitoring

**Trading Relevance**: Parameter optimization for technical indicators, risk management thresholds, position sizing algorithms

## Particle Swarm Optimization (PSO) Examples

### Swarm Intelligence
- **Particle Swarm Optimization Basics**: Social optimization dynamics
- **Moving Peaks Benchmark with Multiswarm PSO**: Dynamic environment adaptation

**Trading Relevance**: Market regime adaptation, dynamic parameter adjustment for changing market conditions

## Estimation of Distribution Algorithms (EDA) Examples

### Probabilistic Models
- **Making Your Own Strategy: A Simple EDA**: Custom algorithm development

**Trading Relevance**: Learning market pattern distributions, probabilistic strategy generation

## Symbolic Regression Example Deep Dive

From the symbolic regression example (most relevant to trading):

### Problem Definition
```python
# Quartic polynomial target: x^4 + x^3 + x^2 + x
target_function = lambda x: x**4 + x**3 + x**2 + x
evaluation_points = [x/10.0 for x in range(-10, 10)]  # 20 points in [-1, 1]
```

### Primitive Set Creation
```python
import operator
import math
from deap import gp

def protectedDiv(left, right):
    """Protected division to prevent crashes."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Create primitive set for mathematical expressions
pset = gp.PrimitiveSet("MAIN", 1)  # One input variable
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)  
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

# Add ephemeral constants
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

# Rename argument
pset.renameArguments(ARG0='x')
```

### Individual and Fitness Creation
```python
from deap import creator, base

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
```

### Toolbox Configuration
```python
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    """Evaluate symbolic regression fitness."""
    # Transform tree expression into callable function
    func = toolbox.compile(expr=individual)
    
    # Calculate mean squared error
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points=evaluation_points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Size limits to prevent bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
```

### Statistics and Evolution
```python
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, 
                              stats=mstats, halloffame=hof, verbose=True)
```

## Trading Strategy Evolution Implementation Pattern

Based on the symbolic regression example, here's a trading strategy evolution pattern:

### Trading Primitive Set
```python
def create_trading_primitives():
    """Create primitive set for trading strategy evolution."""
    
    # Input: OHLCV DataFrame with technical indicators
    pset = gp.PrimitiveSetTyped("trading_strategy", [pd.DataFrame], bool)
    
    # Technical indicators returning float
    pset.addPrimitive(lambda df: df['sma_14'].iloc[-1], [pd.DataFrame], float)
    pset.addPrimitive(lambda df: df['rsi_14'].iloc[-1], [pd.DataFrame], float)
    pset.addPrimitive(lambda df: df['macd'].iloc[-1], [pd.DataFrame], float)
    pset.addPrimitive(lambda df: df['bb_upper'].iloc[-1], [pd.DataFrame], float)
    pset.addPrimitive(lambda df: df['bb_lower'].iloc[-1], [pd.DataFrame], float)
    pset.addPrimitive(lambda df: df['volume'].iloc[-1], [pd.DataFrame], float)
    pset.addPrimitive(lambda df: df['close'].iloc[-1], [pd.DataFrame], float)
    
    # Price relationships
    pset.addPrimitive(lambda df: df['close'].iloc[-1] / df['sma_20'].iloc[-1], 
                     [pd.DataFrame], float)
    pset.addPrimitive(lambda df: (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2], 
                     [pd.DataFrame], float)
    
    # Comparison operators
    pset.addPrimitive(operator.gt, [float, float], bool)
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.ge, [float, float], bool)
    pset.addPrimitive(operator.le, [float, float], bool)
    
    # Logical operators
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)
    
    # Constants for thresholds
    pset.addEphemeralConstant("rsi_threshold", lambda: random.uniform(20, 80), float)
    pset.addEphemeralConstant("price_threshold", lambda: random.uniform(0.95, 1.05), float)
    pset.addEphemeralConstant("volume_multiplier", lambda: random.uniform(0.5, 2.0), float)
    
    pset.renameArguments(ARG0='market_data')
    return pset
```

### Strategy Evaluation Function
```python
def evaluate_trading_strategy(individual, historical_data, validation_periods):
    """Evaluate trading strategy through backtesting."""
    try:
        # Compile strategy to executable function
        strategy_func = gp.compile(individual, trading_pset)
        
        total_sharpe = 0
        total_drawdown = 0
        total_trades = 0
        
        # Test across multiple validation periods
        for start_date, end_date in validation_periods:
            period_data = historical_data[start_date:end_date]
            
            # Generate trading signals
            positions = []
            for i in range(50, len(period_data)):  # Skip initial period for indicators
                window_data = period_data.iloc[i-50:i+1]
                
                try:
                    signal = strategy_func(window_data)
                    positions.append(1 if signal else 0)
                except:
                    positions.append(0)  # Default to no position on error
            
            # Calculate performance metrics
            if len(positions) > 10:  # Minimum trades for valid evaluation
                returns = calculate_strategy_returns(positions, period_data[50:])
                sharpe = calculate_sharpe_ratio(returns)
                drawdown = calculate_max_drawdown(returns)
                
                total_sharpe += sharpe
                total_drawdown += drawdown
                total_trades += len([p for p in positions if p == 1])
        
        # Average metrics across validation periods
        avg_sharpe = total_sharpe / len(validation_periods)
        avg_drawdown = total_drawdown / len(validation_periods)
        trade_frequency = total_trades / len(validation_periods)
        
        # Multi-objective fitness: maximize Sharpe, minimize drawdown, reasonable trade frequency
        fitness_sharpe = avg_sharpe
        fitness_drawdown = -avg_drawdown  # Negative because we want to minimize
        fitness_frequency = 1.0 - abs(trade_frequency - 20) / 20  # Target ~20 trades per period
        
        return fitness_sharpe, fitness_drawdown, fitness_frequency
        
    except Exception as e:
        # Return poor fitness for invalid strategies
        return -10.0, -1.0, 0.0
```

### Complete Trading Strategy Evolution
```python
def evolve_trading_strategies():
    """Complete trading strategy evolution implementation."""
    
    # Create multi-objective fitness
    creator.create("TradingFitness", base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create("TradingStrategy", gp.PrimitiveTree, fitness=creator.TradingFitness)
    
    # Setup toolbox
    trading_pset = create_trading_primitives()
    toolbox = base.Toolbox()
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=trading_pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.TradingStrategy, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_trading_strategy, 
                    historical_data=market_data, validation_periods=validation_splits)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=trading_pset)
    toolbox.register("select", tools.selNSGA2)  # Multi-objective selection
    
    # Prevent bloat
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=50))
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    # Evolution
    pop = toolbox.population(n=200)
    hof = tools.ParetoFront()
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                      stats=stats, halloffame=hof, verbose=True)
    
    return pop, hof, logbook
```

## Key Implementation Insights

1. **Protected Operations**: Always implement protected division and error handling for robustness
2. **Size Limits**: Use decorators to prevent bloat in GP trees
3. **Multi-objective Fitness**: Use NSGA-II for trading strategies with multiple conflicting objectives
4. **Validation Periods**: Test strategies across multiple time periods to prevent overfitting
5. **Ephemeral Constants**: Essential for evolving numerical thresholds
6. **Type Safety**: Strongly typed GP prevents invalid strategy combinations
7. **Statistics Tracking**: Monitor population diversity and fitness progression

The examples demonstrate that DEAP's flexibility allows adaptation to complex real-world problems like automated trading strategy evolution while maintaining robust evolutionary dynamics.