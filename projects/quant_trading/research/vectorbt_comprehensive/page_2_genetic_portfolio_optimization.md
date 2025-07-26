# Genetic Algorithms in Portfolio Optimization - Implementation Guide

**Source**: https://leomercanti.medium.com/genetic-algorithms-in-portfolio-optimization-a-cutting-edge-approach-to-maximizing-returns-ce9225b9bef3  
**Research Date**: 2025-01-26  
**Extraction Method**: Brightdata MCP + Quality Enhancement  
**Focus**: Genetic algorithm implementation patterns, portfolio optimization integration, VectorBT compatibility patterns

## Executive Summary

This article provides a comprehensive implementation guide for genetic algorithms in portfolio optimization, demonstrating how to overcome traditional Modern Portfolio Theory limitations through evolutionary approaches. The patterns shown are directly applicable to VectorBT integration for genetic algorithm-based trading systems.

## Core Genetic Algorithm Components for Portfolio Optimization

### 1. Population Structure

```python
# Portfolio representation as chromosomes
def generate_population(size, num_assets):
    """Generate initial population of portfolios using Dirichlet distribution."""
    population = [np.random.dirichlet(np.ones(num_assets)) for _ in range(size)]
    return population
```

**VectorBT Integration Pattern**: Each portfolio becomes a set of weights that can be used with `Portfolio.from_signals()`:

```python
# VectorBT integration for genetic portfolio weights
def create_vectorbt_portfolio_from_genome(price_data, genome_weights):
    """
    Create VectorBT portfolio using genetic algorithm evolved weights.
    """
    # Generate signals (could be universal strategy from previous research)
    entries, exits = generate_universal_signals(price_data, genome_params)
    
    # Apply genetic weights to multi-asset portfolio
    portfolio = vbt.Portfolio.from_signals(
        price_data,
        entries,
        exits,
        size=genome_weights,  # Use evolved weights as position sizing
        group_by=True,
        cash_sharing=True
    )
    
    return portfolio
```

### 2. Fitness Function Implementation

```python
# Sharpe ratio as fitness function (from article)
def fitness_function(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # Negative because GAs minimize fitness
```

**Enhanced Multi-Objective Fitness for VectorBT**:

```python
def enhanced_vectorbt_fitness(genome, price_data, universal_strategy_params):
    """
    Enhanced fitness function for VectorBT genetic algorithm integration.
    Combines multiple performance metrics for robust strategy evolution.
    """
    # Create portfolio using genetic parameters
    portfolio = create_vectorbt_portfolio_from_genome(price_data, genome)
    
    # Extract comprehensive performance metrics
    stats = portfolio.stats()
    
    # Multi-objective fitness components
    sharpe_ratio = stats['Sharpe Ratio'] if not pd.isna(stats['Sharpe Ratio']) else 0
    calmar_ratio = stats['Calmar Ratio'] if not pd.isna(stats['Calmar Ratio']) else 0
    max_drawdown = abs(stats['Max Drawdown [%]']) / 100  # Convert to positive ratio
    win_rate = stats['Win Rate [%]'] / 100
    profit_factor = stats['Profit Factor'] if not pd.isna(stats['Profit Factor']) else 0
    
    # Weighted fitness calculation (tunable weights)
    fitness = (
        sharpe_ratio * 0.30 +           # Risk-adjusted returns
        calmar_ratio * 0.25 +           # Drawdown-adjusted returns  
        (1 - max_drawdown) * 0.20 +     # Drawdown penalty (lower is better)
        win_rate * 0.15 +               # Consistency reward
        profit_factor * 0.10            # Profitability factor
    )
    
    # Penalty for invalid portfolios
    if portfolio.total_return() < -0.5:  # More than 50% loss
        fitness *= 0.1
    
    return fitness
```

### 3. Genetic Operators for Portfolio Evolution

```python
# Crossover operation (from article)
def crossover(portfolio1, portfolio2):
    crossover_point = random.randint(1, len(portfolio1) - 1)
    return np.concatenate((portfolio1[:crossover_point], portfolio2[crossover_point:]))

# Mutation operation
def mutate(portfolio, mutation_rate=0.1):
    mutation = np.random.normal(0, mutation_rate, len(portfolio))
    return np.clip(portfolio + mutation, 0, 1)
```

**Enhanced Genetic Operators for VectorBT**:

```python
def advanced_portfolio_crossover(parent1_genome, parent2_genome, crossover_rate=0.7):
    """
    Advanced crossover for portfolio genetic algorithms.
    Maintains portfolio constraints while enabling genetic diversity.
    """
    if random.random() > crossover_rate:
        return parent1_genome.copy(), parent2_genome.copy()
    
    # Arithmetic crossover for portfolio weights
    alpha = random.random()
    child1_genome = alpha * parent1_genome + (1 - alpha) * parent2_genome
    child2_genome = (1 - alpha) * parent1_genome + alpha * parent2_genome
    
    # Ensure weights sum to 1 (portfolio constraint)
    child1_genome = child1_genome / np.sum(child1_genome)
    child2_genome = child2_genome / np.sum(child2_genome)
    
    return child1_genome, child2_genome

def portfolio_mutation(genome, mutation_rate=0.1, mutation_strength=0.05):
    """
    Portfolio-specific mutation maintaining weight constraints.
    """
    if random.random() > mutation_rate:
        return genome.copy()
    
    # Select random asset for mutation
    mutation_index = random.randint(0, len(genome) - 1)
    
    # Apply bounded mutation
    mutation_delta = random.gauss(0, mutation_strength)
    genome[mutation_index] = max(0, min(1, genome[mutation_index] + mutation_delta))
    
    # Renormalize to maintain portfolio constraint
    genome = genome / np.sum(genome)
    
    return genome
```

### 4. Complete Genetic Algorithm Implementation

```python
# Enhanced genetic algorithm with VectorBT integration
class VectorBTGeneticPortfolioOptimizer:
    """
    Complete genetic algorithm for portfolio optimization with VectorBT integration.
    Combines genetic evolution with high-performance backtesting.
    """
    
    def __init__(self, price_data, population_size=50, generations=100):
        self.price_data = price_data
        self.population_size = population_size
        self.generations = generations
        self.num_assets = len(price_data.columns) if hasattr(price_data, 'columns') else 1
        
    def initialize_population(self):
        """Initialize population with random portfolio weights."""
        population = []
        for _ in range(self.population_size):
            # Create random weights using Dirichlet distribution
            weights = np.random.dirichlet(np.ones(self.num_assets))
            population.append(weights)
        return population
    
    def evaluate_population(self, population):
        """Evaluate fitness for entire population."""
        fitness_scores = []
        for genome in population:
            try:
                fitness = enhanced_vectorbt_fitness(genome, self.price_data, {})
                fitness_scores.append(fitness)
            except Exception as e:
                # Handle failed evaluations
                fitness_scores.append(-1.0)  # Penalty for invalid portfolios
        
        return fitness_scores
    
    def selection(self, population, fitness_scores, selection_size):
        """Tournament selection with elitism."""
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Elite selection (top 20%)
        elite_size = int(0.2 * selection_size)
        selected = [population[i] for i in sorted_indices[:elite_size]]
        
        # Tournament selection for remaining
        tournament_size = 3
        while len(selected) < selection_size:
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def evolve(self):
        """Main evolution loop."""
        # Initialize population
        population = self.initialize_population()
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = self.evaluate_population(population)
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Selection
            selected = self.selection(population, fitness_scores, self.population_size)
            
            # Create next generation
            next_generation = []
            
            while len(next_generation) < self.population_size:
                # Select parents
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Crossover
                child1, child2 = advanced_portfolio_crossover(parent1, parent2)
                
                # Mutation
                child1 = portfolio_mutation(child1)
                child2 = portfolio_mutation(child2)
                
                next_generation.extend([child1, child2])
            
            # Truncate to population size
            population = next_generation[:self.population_size]
            
            # Progress reporting
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")
        
        # Return best solution
        final_fitness = self.evaluate_population(population)
        best_idx = np.argmax(final_fitness)
        
        return {
            'best_portfolio': population[best_idx],
            'best_fitness': final_fitness[best_idx],
            'fitness_history': best_fitness_history
        }
```

## Advanced Integration Patterns with VectorBT

### 1. Multi-Asset Strategy Evolution

```python
def evolve_multi_asset_strategy(assets_data, strategy_template):
    """
    Evolve portfolio allocation across multiple assets using genetic algorithms.
    Integrates with VectorBT for high-performance backtesting.
    """
    
    class MultiAssetGenome:
        def __init__(self, num_assets, strategy_params_size):
            # Portfolio weights
            self.asset_weights = np.random.dirichlet(np.ones(num_assets))
            
            # Universal strategy parameters
            self.strategy_params = np.random.uniform(0, 1, strategy_params_size)
            
        def to_array(self):
            return np.concatenate([self.asset_weights, self.strategy_params])
        
        @classmethod
        def from_array(cls, array, num_assets):
            genome = cls.__new__(cls)
            genome.asset_weights = array[:num_assets]
            genome.strategy_params = array[num_assets:]
            return genome
    
    def evaluate_multi_asset_genome(genome):
        # Generate signals using evolved strategy parameters
        all_entries = {}
        all_exits = {}
        
        for asset_name, asset_data in assets_data.items():
            entries, exits = strategy_template.generate_signals(
                asset_data, 
                genome.strategy_params
            )
            all_entries[asset_name] = entries
            all_exits[asset_name] = exits
        
        # Create multi-asset portfolio with genetic weights
        portfolio = vbt.Portfolio.from_signals(
            assets_data,
            pd.DataFrame(all_entries),
            pd.DataFrame(all_exits),
            size=genome.asset_weights,
            group_by=True,
            cash_sharing=True,
            fees=0.001
        )
        
        return enhanced_vectorbt_fitness(genome.asset_weights, assets_data, {})
    
    # Run genetic algorithm
    optimizer = VectorBTGeneticPortfolioOptimizer(assets_data)
    result = optimizer.evolve()
    
    return result
```

### 2. Strategy Parameter and Portfolio Weight Co-Evolution

```python
def co_evolve_strategy_and_allocation(price_data):
    """
    Simultaneously evolve strategy parameters and portfolio allocation.
    Demonstrates complete genetic algorithm integration with VectorBT.
    """
    
    class HybridGenome:
        def __init__(self, num_assets):
            self.num_assets = num_assets
            
            # Portfolio allocation weights
            self.allocation_weights = np.random.dirichlet(np.ones(num_assets))
            
            # Universal strategy parameters
            self.rsi_period = random.randint(5, 50)
            self.rsi_bottom = random.uniform(20, 40)
            self.rsi_top = random.uniform(60, 80)
            self.ema_fast = random.randint(5, 20)
            self.ema_slow = random.randint(21, 100)
            
            # Risk management parameters
            self.stop_loss = random.uniform(0.02, 0.10)
            self.take_profit = random.uniform(0.03, 0.15)
    
    def evaluate_hybrid_genome(genome):
        """Evaluate complete strategy + allocation genome."""
        
        # Generate signals using evolved strategy parameters
        RSI = vbt.IndicatorFactory.from_talib('RSI')
        rsi = RSI.run(price_data, timeperiod=[genome.rsi_period])
        
        # Create evolved entry/exit conditions
        entries = rsi.real_crossed_below(genome.rsi_bottom)
        exits = rsi.real_crossed_above(genome.rsi_top)
        
        # Clean signals
        entries, exits = pd.DataFrame.vbt.signals.clean(entries, exits)
        
        # Create portfolio with evolved allocation
        portfolio = vbt.Portfolio.from_signals(
            price_data,
            entries,
            exits,
            size=genome.allocation_weights,
            sl_stop=genome.stop_loss,
            tp_stop=genome.take_profit,
            fees=0.001
        )
        
        # Multi-objective fitness evaluation
        stats = portfolio.stats()
        return calculate_multi_objective_fitness(stats)
    
    # Implement genetic operators for hybrid genome
    def hybrid_crossover(parent1, parent2):
        child = HybridGenome(parent1.num_assets)
        
        # Crossover allocation weights
        alpha = random.random()
        child.allocation_weights = (alpha * parent1.allocation_weights + 
                                  (1-alpha) * parent2.allocation_weights)
        child.allocation_weights /= np.sum(child.allocation_weights)
        
        # Crossover strategy parameters
        child.rsi_period = random.choice([parent1.rsi_period, parent2.rsi_period])
        child.rsi_bottom = (parent1.rsi_bottom + parent2.rsi_bottom) / 2
        child.rsi_top = (parent1.rsi_top + parent2.rsi_top) / 2
        
        return child
    
    return HybridGenome, evaluate_hybrid_genome, hybrid_crossover
```

## Performance Optimization for Genetic Algorithm Backtesting

### 1. Parallel Evaluation

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def parallel_genetic_evaluation(population, price_data, num_workers=None):
    """
    Parallel evaluation of genetic algorithm population using multiprocessing.
    Critical for large populations and complex VectorBT backtests.
    """
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    def evaluate_single_genome(genome):
        try:
            return enhanced_vectorbt_fitness(genome, price_data, {})
        except Exception as e:
            return -1.0  # Penalty for failed evaluation
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        fitness_scores = list(executor.map(evaluate_single_genome, population))
    
    return fitness_scores
```

### 2. Cached Indicator Computation

```python
class CachedIndicatorEngine:
    """
    Pre-compute indicators for genetic algorithm speed optimization.
    Prevents redundant calculations during population evaluation.
    """
    
    def __init__(self, price_data):
        self.price_data = price_data
        self.indicator_cache = {}
        self._precompute_indicators()
    
    def _precompute_indicators(self):
        """Pre-compute common indicators with various parameters."""
        RSI = vbt.IndicatorFactory.from_talib('RSI')
        
        # Pre-compute RSI for different periods
        for period in range(5, 51, 5):
            self.indicator_cache[f'rsi_{period}'] = RSI.run(
                self.price_data, 
                timeperiod=[period]
            )
        
        # Pre-compute EMAs
        for period in range(5, 101, 5):
            self.indicator_cache[f'ema_{period}'] = (
                self.price_data.ewm(span=period).mean()
            )
    
    def get_signals(self, genome_params):
        """Fast signal generation using cached indicators."""
        rsi_period = int(genome_params['rsi_period'] / 5) * 5  # Round to cached value
        rsi = self.indicator_cache[f'rsi_{rsi_period}']
        
        entries = rsi.real_crossed_below(genome_params['rsi_bottom'])
        exits = rsi.real_crossed_above(genome_params['rsi_top'])
        
        return entries, exits
```

## Integration with Existing Research

### Connection to Previous VectorBT Strategy Porting Research

The genetic algorithm patterns shown here integrate perfectly with the strategy porting patterns from the NBViewer documentation:

```python
# Complete integration pipeline
class GeneticVectorBTBridge:
    """
    Bridge genetic algorithms with VectorBT strategy porting patterns.
    Combines best practices from both research sources.
    """
    
    def __init__(self, price_data):
        self.price_data = price_data
        self.cached_indicators = CachedIndicatorEngine(price_data)
    
    def convert_backtrader_strategy_to_genetic(self, bt_strategy_class):
        """
        Convert backtrader strategy to genetic-optimizable format.
        Integrates with previous strategy porting research.
        """
        # Extract parameters from backtrader strategy
        strategy_params = bt_strategy_class.params._getitems()
        
        # Create genetic genome template
        genome_bounds = {}
        for param_name, param_value in strategy_params:
            if isinstance(param_value, (int, float)):
                genome_bounds[param_name] = (param_value * 0.5, param_value * 2.0)
        
        return genome_bounds
    
    def evaluate_genetic_backtrader_port(self, genome):
        """
        Evaluate ported backtrader strategy using genetic parameters.
        Uses VectorBT Portfolio.from_signals() as shown in porting research.
        """
        # Generate signals using genetic parameters
        entries, exits = self.cached_indicators.get_signals({
            'rsi_period': genome[0] * 45 + 5,     # 5-50 range
            'rsi_bottom': genome[1] * 20 + 20,    # 20-40 range
            'rsi_top': genome[2] * 20 + 60        # 60-80 range
        })
        
        # Use Portfolio.from_signals() pattern from porting research
        portfolio = vbt.Portfolio.from_signals(
            self.price_data,
            entries,
            exits,
            price=self.price_data.vbt.fshift(1),  # Realistic execution price
            fees=0.001,
            init_cash=10000
        )
        
        # Extract fitness using VectorBT stats
        return enhanced_vectorbt_fitness(genome, self.price_data, {})
```

## Conclusion and Next Steps

This research demonstrates comprehensive patterns for integrating genetic algorithms with VectorBT-based portfolio optimization:

1. **Portfolio Weight Evolution**: Genetic algorithms can evolve optimal asset allocation weights
2. **Strategy Parameter Co-Evolution**: Simultaneous optimization of strategy logic and allocation
3. **Multi-Asset Scaling**: Natural extension to universal strategies across multiple assets
4. **Performance Optimization**: Parallel evaluation and caching for large-scale genetic evolution
5. **VectorBT Integration**: Seamless connection with high-performance backtesting engine

### Implementation Priority for Quant Trading Project:

1. Start with simple portfolio weight evolution using existing universal strategy
2. Add strategy parameter co-evolution for enhanced genetic diversity
3. Implement parallel evaluation for population-scale backtesting
4. Scale to multi-asset universal strategy with genetic allocation
5. Integrate with live trading pipeline for paper trading validation

The genetic algorithm approach provides a robust foundation for the evolution of trading strategies that can adapt to changing market conditions while maintaining optimal risk-adjusted returns.