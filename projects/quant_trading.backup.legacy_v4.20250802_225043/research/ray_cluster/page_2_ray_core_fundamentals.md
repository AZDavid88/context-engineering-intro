# Ray Core Fundamentals - Distributed Computing Patterns

## Source URL
https://docs.ray.io/en/latest/ray-core/walkthrough.html

## Ray Core Essential Primitives

### 1. **Tasks** - Stateless Distributed Functions
**Perfect for genetic algorithm parameter evaluation**

```python
# Basic task pattern
@ray.remote
def evaluate_strategy_parameters(seed_type: str, parameters: dict, market_data: dict):
    """
    Stateless function for distributed genetic evaluation.
    Each worker gets a complete task to execute independently.
    """
    # Import locally to avoid global state issues
    from strategy.genetic_seeds.seed_registry import create_seed_instance
    
    # Create strategy with genetic parameters
    strategy = create_seed_instance(seed_type, parameters)
    
    # Run backtesting evaluation
    results = strategy.backtest(market_data)
    
    return {
        'fitness': results.sharpe_ratio,
        'total_return': results.total_return,
        'max_drawdown': results.max_drawdown,
        'trade_count': results.total_trades,
        'parameters': parameters
    }

# Distributed execution pattern for genetic population
def evaluate_genetic_population(population: List[dict], market_data: dict):
    """Evaluate entire population using Ray cluster."""
    
    # Submit all evaluations to Ray cluster (parallel execution)
    futures = []
    for individual in population:
        future = evaluate_strategy_parameters.remote(
            individual['seed_type'],
            individual['parameters'], 
            market_data
        )
        futures.append(future)
    
    # Collect results from all workers
    results = ray.get(futures)
    return results
```

### 2. **Actors** - Stateful Distributed Workers
**Useful for maintaining genetic algorithm state**

```python
# Actor pattern for genetic algorithm coordinator
@ray.remote
class GeneticEvolutionCoordinator:
    """
    Stateful actor that maintains genetic algorithm state.
    Coordinates evolution cycles and tracks performance metrics.
    """
    
    def __init__(self, population_size: int, mutation_rate: float):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_fitness_history = []
        self.population = []
    
    def initialize_population(self, seed_types: List[str]):
        """Initialize random population."""
        from strategy.genetic_seeds.parameter_generator import generate_random_parameters
        
        self.population = []
        for _ in range(self.population_size):
            seed_type = random.choice(seed_types)
            parameters = generate_random_parameters(seed_type)
            individual = {
                'seed_type': seed_type,
                'parameters': parameters,
                'fitness': None
            }
            self.population.append(individual)
        return len(self.population)
    
    def get_population_for_evaluation(self):
        """Return current population for distributed evaluation."""
        return self.population.copy()
    
    def update_fitness_scores(self, evaluation_results: List[dict]):
        """Update population with fitness scores from distributed evaluation."""
        for i, result in enumerate(evaluation_results):
            self.population[i]['fitness'] = result['fitness']
            self.population[i]['metrics'] = {
                'total_return': result['total_return'],
                'max_drawdown': result['max_drawdown'],
                'trade_count': result['trade_count']
            }
    
    def evolve_generation(self):
        """Perform genetic operations: selection, crossover, mutation."""
        # Selection: keep top 50%
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        elite = self.population[:self.population_size // 2]
        
        # Track best fitness
        best_fitness = elite[0]['fitness']
        self.best_fitness_history.append(best_fitness)
        
        # Generate new population through crossover and mutation
        new_population = elite.copy()  # Keep elite
        
        while len(new_population) < self.population_size:
            # Crossover
            parent1, parent2 = random.sample(elite, 2)
            child = self._crossover(parent1, parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'population_size': len(self.population)
        }
    
    def get_evolution_stats(self):
        """Return evolution statistics."""
        return {
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'current_best': max(self.best_fitness_history) if self.best_fitness_history else None
        }

# Usage in genetic evolution loop
coordinator = GeneticEvolutionCoordinator.remote(population_size=100, mutation_rate=0.1)
```

### 3. **Objects** - Distributed Data Management
**Critical for sharing market data efficiently across workers**

```python
# Efficient data sharing pattern
def distributed_genetic_evolution(market_data: pd.DataFrame, 
                                num_generations: int = 10,
                                population_size: int = 100):
    """
    Complete distributed genetic evolution with efficient data management.
    """
    
    # Put large market data in Ray object store (shared across all workers)
    market_data_ref = ray.put(market_data)
    
    # Create genetic coordinator actor
    coordinator = GeneticEvolutionCoordinator.remote(
        population_size=population_size, 
        mutation_rate=0.1
    )
    
    # Initialize population
    seed_types = ['ema_crossover', 'rsi_mean_reversion', 'donchian_breakout']
    ray.get(coordinator.initialize_population.remote(seed_types))
    
    # Evolution loop
    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        
        # Get current population for evaluation
        population = ray.get(coordinator.get_population_for_evaluation.remote())
        
        # Distributed evaluation using shared market data
        futures = []
        for individual in population:
            future = evaluate_strategy_parameters.remote(
                individual['seed_type'],
                individual['parameters'],
                market_data_ref  # Reference to shared data
            )
            futures.append(future)
        
        # Collect evaluation results
        results = ray.get(futures)
        
        # Update fitness scores
        ray.get(coordinator.update_fitness_scores.remote(results))
        
        # Evolve to next generation
        evolution_stats = ray.get(coordinator.evolve_generation.remote())
        
        print(f"Best fitness: {evolution_stats['best_fitness']:.4f}")
    
    # Get final statistics
    final_stats = ray.get(coordinator.get_evolution_stats.remote())
    return final_stats
```

## Key Patterns for Genetic Algorithm Distribution

### 1. **Master-Worker Pattern**
- **Master**: Genetic coordinator actor manages evolution state
- **Workers**: Stateless tasks evaluate strategy parameters
- **Data**: Shared market data via Ray object store

### 2. **Lazy Execution & Efficiency**
```python
# CORRECT: Efficient object passing
market_data_ref = ray.put(large_market_data)  # Put once
futures = [evaluate_strategy.remote(params, market_data_ref) for params in population]

# INCORRECT: Inefficient repeated serialization  
futures = [evaluate_strategy.remote(params, large_market_data) for params in population]
```

### 3. **Fault Tolerance Patterns**
```python
# Handle worker failures gracefully
def robust_distributed_evaluation(population, market_data_ref, max_retries=3):
    """Distributed evaluation with retry logic."""
    
    futures = []
    for individual in population:
        future = evaluate_strategy_parameters.remote(
            individual['seed_type'],
            individual['parameters'],
            market_data_ref
        )
        futures.append(future)
    
    # Get results with timeout and retry
    results = []
    for i, future in enumerate(futures):
        for attempt in range(max_retries):
            try:
                result = ray.get(future, timeout=300)  # 5-minute timeout
                results.append(result)
                break
            except ray.exceptions.RayTimeoutError:
                print(f"Timeout for individual {i}, attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    # Use default poor fitness score
                    results.append({
                        'fitness': -999,
                        'total_return': 0,
                        'max_drawdown': -1,
                        'trade_count': 0,
                        'parameters': population[i]['parameters']
                    })
    
    return results
```

## Integration with Existing Architecture

### Compatible Components (Local):
- **SeedRegistry**: Strategy discovery and creation
- **TradingSystemManager**: Live trading coordination  
- **RetailConnectionOptimizer**: Session management
- **OrderManagement**: Live execution

### Distributable Components (Ray Tasks):
- **Strategy Parameter Evaluation**: Pure backtesting calculations
- **Performance Metrics**: Risk/return analysis
- **Market Data Processing**: Historical analysis
- **Genetic Operations**: Population evolution (via actors)

### Resource Optimization:
```python
# Specify resource requirements for tasks
@ray.remote(num_cpus=1, memory=2000*1024*1024)  # 2GB RAM
def evaluate_strategy_parameters(seed_type, parameters, market_data_ref):
    # CPU-intensive backtesting task
    pass

@ray.remote(num_cpus=2, memory=4000*1024*1024)  # 4GB RAM  
class GeneticEvolutionCoordinator:
    # Memory-intensive coordinator
    pass
```

## Performance Characteristics

### Scalability Benefits:
- **Parallel Evaluation**: 10-50x speedup with cluster workers
- **Memory Efficiency**: Market data shared via object store
- **Resource Elasticity**: Workers scale with population size

### Cost Economics:
- **Ray Cluster**: $7-20 per evolution cycle (100-1000 strategies)
- **Break-even**: ~$647 trading capital vs local execution
- **Autoscaling**: Pay only during evolution cycles

This Ray Core foundation enables seamless scaling from local genetic evolution (Phase 5A) to distributed cluster execution (Phase 5B) while maintaining system architecture integrity.