# DEAP Genetic Programming - Comprehensive Guide

**Source URL**: https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
**Extraction Date**: 2025-07-25
**Quality Assessment**: ✅ Production-ready GP implementation guide with complete examples

## Genetic Programming Overview

Genetic Programming (GP) is a special field of evolutionary computation that aims at building programs automatically to solve problems independently of their domain. The most common representation is the syntax tree.

### Tree Representation Example
```
max(x + 3 * y, x + x)
```
- **Terminals** (leaves, green): Constants and arguments (x, y, 3)
- **Primitives** (internal nodes, red): Functions (max, +, *)
- **Program Structure**: Tree represents executable mathematical expression

## Primitive Set Types

### 1. Loosely Typed GP

Loosely typed GP doesn't enforce specific types between nodes - any primitive/terminal can connect:

```python
from deap import gp
import operator

# Create primitive set
pset = gp.PrimitiveSet("main", 2)  # name, number of inputs

# Add primitives (functions)
pset.addPrimitive(max, 2)               # max function, arity 2
pset.addPrimitive(operator.add, 2)      # addition, arity 2  
pset.addPrimitive(operator.mul, 2)      # multiplication, arity 2
pset.addPrimitive(operator.neg, 1)      # negation, arity 1

# Add terminal (constant)
pset.addTerminal(3)

# Rename arguments from ARG0, ARG1 to meaningful names
pset.renameArguments(ARG0="x", ARG1="y")
```

**Tree Generation**:
```python
from deap.gp import genFull, PrimitiveTree

# Generate expression with depth 1-3
expr = genFull(pset, min_=1, max_=3)
tree = PrimitiveTree(expr)
```

### 2. Strongly Typed GP

Enforces type constraints - primitives can only connect if types match:

```python
def if_then_else(input, output1, output2):
    return output1 if input else output2

# Create typed primitive set
pset = gp.PrimitiveSetTyped("main", [bool, float], float)

# Add typed primitives
pset.addPrimitive(operator.xor, [bool, bool], bool)           # bool inputs → bool output
pset.addPrimitive(operator.mul, [float, float], float)        # float inputs → float output
pset.addPrimitive(if_then_else, [bool, float, float], float)  # mixed inputs → float output

# Add typed terminals
pset.addTerminal(3.0, float)
pset.addTerminal(True, bool)

pset.renameArguments(ARG0="x", ARG1="y")
```

**Type Safety Benefits**:
- Prevents invalid operations (e.g., multiplying boolean with float)
- Guarantees type correctness throughout evolution
- Enables domain-specific strongly-typed operations

**Type Constraint Warning**: If no terminals can satisfy a primitive's input type requirements, DEAP raises `IndexError` during tree generation.

## Ephemeral Constants

Dynamic constants generated at runtime with different values per individual:

```python
import random

# Random float in [-1, 1)
pset.addEphemeralConstant(lambda: random.uniform(-1, 1))

# Typed ephemeral constant (random integer)
pset.addEphemeralConstant(lambda: random.randint(-10, 10), int)
```

**Key Properties**:
- Value determined when inserted into tree
- Remains constant for that individual
- Different individuals get different ephemeral values
- Essential for evolving diverse numerical constants

## Creating GP Individuals

Combine creator and toolbox for evolutionary-ready individuals:

```python
from deap import creator, base, tools

# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# Register generation functions
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

# Create individual
individual = toolbox.individual()
```

**Critical Addition**: `pset=pset` reference enables GP operators to access primitive set during evolution.

## Tree Evaluation and Compilation

### Converting Trees to Executable Code

```python
# Generate and display tree
expr = genFull(pset, min_=1, max_=3)
tree = PrimitiveTree(expr)
print(str(tree))  # Output: 'mul(add(x, x), max(y, x))'

# Compile to executable function
function = gp.compile(tree, pset)
result = function(1, 2)  # Evaluate with x=1, y=2
print(result)  # Output: 4
```

**Compilation Process**:
1. `str(tree)` converts tree to readable Python code
2. `gp.compile(tree, pset)` creates executable function
3. Resulting function accepts arguments matching primitive set inputs
4. Returns computed result for given inputs

## Tree Size Limits and Bloat Control

### Python Parser Limitations
- Maximum tree depth: ~91-99 levels
- Exceeding limit causes `MemoryError: s_push: parser stack overflow`
- Common cause: **Bloat** - trees grow excessively without improving fitness

### Bloat Prevention
DEAP provides size/height control operators in the GP tools section to prevent uncontrolled growth during evolution.

## Tree Visualization

Generate tree graphs using NetworkX or pygraphviz:

```python
from deap.gp import graph

# Get graph components
nodes, edges, labels = gp.graph(expr)

# Using pygraphviz
import pygraphviz as pgv
g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw("tree.pdf")

# Using NetworkX + matplotlib
import matplotlib.pyplot as plt
import networkx as nx

g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = nx.graphviz_layout(g, prog="dot")

nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, labels)
plt.show()
```

## Trading Strategy GP Implementation

### Primitive Set for Trading Strategies

```python
import pandas_ta as ta

# Create trading primitive set
trading_pset = gp.PrimitiveSetTyped("trading_strategy", 
                                   [pd.DataFrame],  # OHLCV data input
                                   bool)            # Buy/sell signal output

# Technical indicators (DataFrame → float)
trading_pset.addPrimitive(lambda df: ta.sma(df['close'], 14).iloc[-1], [pd.DataFrame], float)
trading_pset.addPrimitive(lambda df: ta.rsi(df['close'], 14).iloc[-1], [pd.DataFrame], float)
trading_pset.addPrimitive(lambda df: ta.macd(df['close']).iloc[-1]['MACD'], [pd.DataFrame], float)

# Comparison operators (float, float → bool)  
trading_pset.addPrimitive(operator.gt, [float, float], bool)  # Greater than
trading_pset.addPrimitive(operator.lt, [float, float], bool)  # Less than

# Logical operators (bool, bool → bool)
trading_pset.addPrimitive(operator.and_, [bool, bool], bool)
trading_pset.addPrimitive(operator.or_, [bool, bool], bool)

# Constants for thresholds
trading_pset.addEphemeralConstant(lambda: random.uniform(0.1, 0.9), float)  # RSI thresholds
trading_pset.addEphemeralConstant(lambda: random.uniform(-0.01, 0.01), float)  # Price changes

trading_pset.renameArguments(ARG0="ohlcv_data")
```

### Trading Strategy Individual Creation

```python
# Multi-objective fitness: (Sharpe ratio, max drawdown, win rate)
creator.create("TradingFitness", base.Fitness, weights=(1.0, -1.0, 1.0))
creator.create("TradingStrategy", gp.PrimitiveTree, 
               fitness=creator.TradingFitness, 
               pset=trading_pset)

# Strategy generation
toolbox.register("expr", gp.genHalfAndHalf, pset=trading_pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.TradingStrategy, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
```

### Strategy Evaluation Example

```python
def evaluate_trading_strategy(individual, historical_data):
    """Evaluate trading strategy performance via backtesting."""
    try:
        # Compile strategy to executable function
        strategy_func = gp.compile(individual, trading_pset)
        
        signals = []
        for i in range(100, len(historical_data)):  # Skip initial period for indicators
            window_data = historical_data.iloc[i-100:i]
            signal = strategy_func(window_data)
            signals.append(1 if signal else 0)
        
        # Calculate performance metrics
        returns = calculate_strategy_returns(signals, historical_data[100:])
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(returns)
        win_rate = calculate_win_rate(returns)
        
        return sharpe_ratio, max_drawdown, win_rate
        
    except Exception as e:
        # Invalid strategy gets worst possible fitness
        return -999.0, 1.0, 0.0

toolbox.register("evaluate", evaluate_trading_strategy, historical_data=market_data)
```

## Key Advantages for Quant Trading

1. **Automatic Strategy Discovery**: GP evolves complete trading logic without manual rule specification
2. **Complex Expression Building**: Can create sophisticated multi-indicator strategies
3. **Type Safety**: Strongly typed GP prevents invalid trading signal combinations  
4. **Interpretability**: Generated strategies are readable mathematical expressions
5. **Bloat Control**: Size limits prevent overly complex, overfit strategies
6. **Visualization**: Tree plots help understand evolved strategy logic

## Integration with Evolution Engine

GP trees provide the perfect representation for trading strategy genes in the quant organism:
- **Crossover**: Exchange subtrees between successful strategies
- **Mutation**: Modify individual nodes or subtrees to explore variations
- **Selection**: Multi-objective selection based on Sharpe ratio, drawdown, consistency
- **Evaluation**: Direct compilation and backtesting on historical data

This comprehensive GP framework enables the creation of self-evolving trading strategies that can discover novel patterns and combinations of technical indicators automatically.