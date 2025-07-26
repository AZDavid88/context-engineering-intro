# DEAP API Reference - Core Modules

**Source URLs**: 
- https://deap.readthedocs.io/en/master/api/index.html
- https://deap.readthedocs.io/en/master/api/gp.html
- https://deap.readthedocs.io/en/master/api/algo.html
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Complete API documentation with function signatures and examples

## Library Reference Overview

DEAP contains several core modules essential for evolutionary computation:

### Module Structure
- **Creator**: Dynamic class creation for fitness and individuals
- **Base**: Toolbox and Fitness foundational classes
- **Evolutionary Tools**: Operators, statistics, logging, constraints
- **Algorithms**: Complete algorithms and variations
- **Genetic Programming**: Tree-based evolution classes and functions
- **Benchmarks**: Standard test problems and evaluation functions

## Genetic Programming Module (deap.gp)

### Core Classes

#### PrimitiveTree
```python
class deap.gp.PrimitiveTree(content)
```
**Purpose**: Tree specifically formatted for GP operations optimization
**Structure**: Depth-first list representation where nodes have `arity` attribute

**Key Methods**:
```python
# Create tree from string expression
@classmethod
PrimitiveTree.from_string(string, pset)

# Tree properties
tree.height          # Maximum depth of tree
tree.root            # Root element (index 0)

# Subtree operations
tree.searchSubtree(begin)  # Returns slice for subtree starting at index
```

**Usage Example**:
```python
from deap.gp import PrimitiveTree, genFull
expr = genFull(pset, min_=1, max_=3)
tree = PrimitiveTree(expr)
print(f"Height: {tree.height}, Root: {tree.root}")
```

#### PrimitiveSet (Loosely Typed)
```python
class deap.gp.PrimitiveSet(name, arity, prefix='ARG')
```
**Purpose**: Container for primitives and terminals without type constraints

**Core Methods**:
```python
# Add function primitive
pset.addPrimitive(primitive, arity, name=None)

# Add constant terminal
pset.addTerminal(terminal, name=None)

# Add ephemeral constant (runtime-generated)
pset.addEphemeralConstant(name, ephemeral)

# Rename function arguments
pset.renameArguments(**kwargs)
```

**Trading Example**:
```python
pset = gp.PrimitiveSet("trading", 1)  # 1 input (market data)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.gt, 2)
pset.addTerminal(50.0)  # RSI threshold
pset.addEphemeralConstant("threshold", lambda: random.uniform(0.01, 0.05))
pset.renameArguments(ARG0="market_data")
```

#### PrimitiveSetTyped (Strongly Typed)
```python
class deap.gp.PrimitiveSetTyped(name, in_types, ret_type, prefix='ARG')
```
**Purpose**: Type-safe primitive set preventing invalid connections

**Advanced Methods**:
```python
# Add typed primitive
pset.addPrimitive(primitive, in_types, ret_type, name=None)

# Add typed terminal
pset.addTerminal(terminal, ret_type, name=None)

# Add typed ephemeral constant
pset.addEphemeralConstant(name, ephemeral, ret_type)

# Add Automatically Defined Function
pset.addADF(adfset)

# Get terminal ratio
pset.terminalRatio  # Property: terminals / all_primitives
```

**Strongly Typed Trading Example**:
```python
# Strategy returns boolean signal
pset = gp.PrimitiveSetTyped("strategy", [pd.DataFrame], bool)

# Technical indicators: DataFrame → float
pset.addPrimitive(lambda df: df['rsi'].iloc[-1], [pd.DataFrame], float)
pset.addPrimitive(lambda df: df['sma'].iloc[-1], [pd.DataFrame], float)

# Comparisons: float, float → bool
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.lt, [float, float], bool)

# Logical operations: bool, bool → bool
pset.addPrimitive(operator.and_, [bool, bool], bool)

# Typed constants
pset.addTerminal(70.0, float)  # RSI overbought
pset.addTerminal(30.0, float)  # RSI oversold
```

### Core Functions

#### compile()
```python
deap.gp.compile(expr, pset)
```
**Purpose**: Convert tree expression to executable Python function
**Returns**: Callable function or direct result (for 0-argument primitives)

**Usage**:
```python
tree = toolbox.individual()
strategy_func = gp.compile(tree, pset)
signal = strategy_func(market_data)  # Execute strategy
```

#### graph()
```python
deap.gp.graph(expr)
```
**Purpose**: Generate graph representation for visualization
**Returns**: (nodes, edges, labels) tuple for plotting libraries

**Visualization Example**:
```python
import networkx as nx
import matplotlib.pyplot as plt

nodes, edges, labels = gp.graph(tree)
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = nx.spring_layout(g)
nx.draw(g, pos, labels=labels, with_labels=True)
plt.show()
```

## Algorithms Module (deap.algorithms)

### Complete Algorithms

#### eaSimple()
```python
deap.algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                        stats=None, halloffame=None, verbose=__debug__)
```
**Purpose**: Standard generational evolutionary algorithm
**Returns**: (final_population, logbook)

**Algorithm Flow**:
1. Evaluate initial population
2. For each generation:
   - Select parents (stochastic selection required)
   - Apply crossover and mutation (varAnd)
   - Evaluate offspring
   - Replace entire population

**Trading Implementation**:
```python
pop, logbook = algorithms.eaSimple(
    population=toolbox.population(n=100),
    toolbox=toolbox,
    cxpb=0.7,           # 70% crossover probability
    mutpb=0.2,          # 20% mutation probability  
    ngen=50,            # 50 generations
    stats=trading_stats,
    halloffame=hof,
    verbose=True
)
```

#### eaMuPlusLambda()
```python
deap.algorithms.eaMuPlusLambda(population, toolbox, mu, lambda_, 
                              cxpb, mutpb, ngen, stats=None, 
                              halloffame=None, verbose=__debug__)
```
**Purpose**: (μ + λ) evolutionary strategy - parents compete with offspring
**Key Difference**: Selection from combined parent + offspring population

**Benefits for Trading**:
- Elitism: Best strategies always survive
- Population stability: Gradual improvement
- Better for expensive evaluations (backtesting)

#### eaMuCommaLambda()
```python
deap.algorithms.eaMuCommaLambda(population, toolbox, mu, lambda_, 
                               cxpb, mutpb, ngen, stats=None,
                               halloffame=None, verbose=__debug__)
```
**Purpose**: (μ, λ) evolutionary strategy - selection only from offspring
**Key Difference**: Parents discarded every generation

**Benefits for Trading**:
- Prevents stagnation in changing markets
- Forces continuous adaptation
- Good for dynamic environments

### Variation Functions

#### varAnd()
```python
deap.algorithms.varAnd(population, toolbox, cxpb, mutpb)
```
**Purpose**: Apply crossover AND mutation to population
**Process**: Clone → Crossover → Mutation → Return offspring

**Usage in Custom Algorithms**:
```python
def custom_trading_algorithm(pop, toolbox, ngen):
    for gen in range(ngen):
        # Custom selection based on market regime
        parents = regime_aware_selection(pop)
        
        # Standard variation
        offspring = algorithms.varAnd(parents, toolbox, 0.7, 0.2)
        
        # Evaluate with current market data
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Custom replacement strategy
        pop = adaptive_replacement(pop, offspring)
    
    return pop
```

#### varOr()
```python
deap.algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
```
**Purpose**: Apply crossover OR mutation OR reproduction
**Process**: Random choice of single operation per offspring

## Key Implementation Patterns for Trading

### 1. Multi-Objective Strategy Evolution
```python
# NSGA-II for conflicting objectives
creator.create("TradingFitness", base.Fitness, weights=(1.0, -1.0, 1.0))  # Sharpe, -Drawdown, WinRate

def evaluate_multi_objective(individual):
    strategy_func = gp.compile(individual, pset)
    returns = backtest_strategy(strategy_func, market_data)
    
    sharpe = calculate_sharpe_ratio(returns)
    drawdown = calculate_max_drawdown(returns)
    win_rate = calculate_win_rate(returns)
    
    return sharpe, drawdown, win_rate

toolbox.register("select", tools.selNSGA2)
```

### 2. Adaptive Parameter Control
```python
class AdaptiveEvolution:
    def __init__(self):
        self.market_volatility = 0.0
        self.performance_trend = 0.0
    
    def adaptive_parameters(self, generation):
        """Adjust EA parameters based on market conditions."""
        base_cxpb = 0.7
        base_mutpb = 0.2
        
        # Increase mutation in volatile markets
        volatility_factor = min(self.market_volatility * 2, 1.0)
        mutpb = base_mutpb + (0.3 * volatility_factor)
        
        # Increase crossover when performance is good
        performance_factor = max(self.performance_trend, 0)
        cxpb = base_cxpb + (0.2 * performance_factor)
        
        return min(cxpb, 0.9), min(mutpb, 0.5)
```

### 3. Strategy Lifecycle Management
```python
def strategy_lifecycle_evolution(population, max_age=20):
    """Implement strategy aging and retirement."""
    for individual in population:
        if not hasattr(individual, 'age'):
            individual.age = 0
        
        individual.age += 1
        
        # Apply age penalty to fitness
        age_penalty = 1.0 - (individual.age / max_age * 0.2)
        if hasattr(individual.fitness, 'values') and individual.fitness.values:
            penalized_fitness = tuple(f * age_penalty for f in individual.fitness.values)
            individual.fitness.values = penalized_fitness
        
        # Retire very old strategies
        if individual.age > max_age:
            individual.fitness.values = (-999.0, -1.0, 0.0)  # Force removal
    
    return population
```

## Critical Implementation Notes

1. **Fitness Tuples**: Always return tuples from evaluation functions, even for single objectives
2. **Type Safety**: Use strongly typed GP for complex trading strategies to prevent invalid combinations
3. **Bloat Control**: Apply size limits to prevent overly complex strategies
4. **Parallel Evaluation**: Register `pool.map` as toolbox map for distributed backtesting
5. **Statistics Collection**: Use `MultiStatistics` for comprehensive performance monitoring
6. **Hall of Fame**: Track best strategies across generations for deployment
7. **Checkpointing**: Save population state for resuming evolution after interruptions

This API reference provides the foundation for implementing robust, scalable evolutionary trading systems using DEAP's comprehensive genetic programming and algorithm capabilities.