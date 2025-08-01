
import random
import logging
import multiprocessing
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

from src.strategy.genetic_seeds import SeedRegistry, get_registry, BaseSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedFitness, SeedType
from src.config.settings import get_settings, Settings

# Import for multi-timeframe data support
try:
    from src.data.dynamic_asset_data_collector import AssetDataSet
except ImportError:
    # Fallback for systems without advanced data collector
    AssetDataSet = None

# Configure logging

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

"""
Genetic Evolution Engine - Core DEAP Integration
This module implements the core genetic algorithm engine using DEAP framework
for evolving trading strategies. Based on research-backed patterns and 
consultant recommendations for multi-objective optimization.
Key Features:
- DEAP genetic algorithm framework integration
- Multi-objective fitness optimization (Sharpe + Consistency + Drawdown)
- Parallel strategy evaluation with multiprocessing
- Genetic seed-based population initialization
- Production-ready error handling and monitoring
"""
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    # Create mock classes for testing
    class MockBase:
        def __init__(self): pass
    creator = MockBase()
    tools = MockBase()
    algorithms = MockBase()
logger = logging.getLogger(__name__)
class EvolutionStatus(str, Enum):
    """Status of genetic evolution process."""
    INITIALIZING = "initializing"
    RUNNING = "running" 
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution parameters."""
    # Population parameters
    population_size: int = 100
    n_generations: int = 50
    # Genetic operators
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    elite_size: int = 10
    # Selection parameters
    tournament_size: int = 3
    selection_pressure: float = 1.5
    # Multi-objective weights (Compatible with Settings dictionary format)
    fitness_weights: Dict[str, float] = None  # Will be loaded from Settings
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.2
    min_win_rate: float = 0.45
    # Parallel processing
    use_multiprocessing: bool = True
    n_processes: Optional[int] = None  # None = auto-detect
    # Validation
    validation_split: float = 0.3  # 30% for out-of-sample
    min_trades: int = 10
    
    # Advanced features for large-scale genetic evolution (research-backed patterns)
    # Large population support (DEAP research: 100-200 -> 1000+ strategies)
    enable_large_populations: bool = False
    max_population_size: int = 1000
    memory_chunk_size: int = 100  # Process in chunks to manage memory
    
    # Multi-timeframe fitness evaluation (VectorBT research: 0.7/0.3 weighting)
    enable_multi_timeframe: bool = False
    strategic_timeframe_weight: float = 0.7  # 1h strategic weight
    tactical_timeframe_weight: float = 0.3   # 15m tactical weight
    timeframe_priorities: Tuple[str, str] = ("1h", "15m")
    
    # Walk-forward validation (DEAP research: overfitting prevention)
    enable_walk_forward: bool = False
    walk_forward_periods: int = 3  # Number of validation periods
    validation_window_days: int = 60  # Days per validation window
    
    # Memory optimization (VectorBT research: vectorized evaluation)
    enable_vectorized_evaluation: bool = False
    precompute_indicators: bool = True
    batch_evaluation_size: int = 50
@dataclass
class EvolutionResults:
    """Results from genetic evolution process."""
    best_individual: Optional[BaseSeed] = None
    population: List[BaseSeed] = None
    fitness_history: List[List[float]] = None
    generation_stats: List[Dict[str, float]] = None
    convergence_data: Dict[str, Any] = None
    execution_time: float = 0.0
    status: EvolutionStatus = EvolutionStatus.INITIALIZING
class GeneticEngine:
    """Core genetic algorithm engine for trading strategy evolution."""
    def __init__(self, config: Optional[EvolutionConfig] = None, 
                 settings: Optional[Settings] = None):
        """Initialize genetic engine.
        Args:
            config: Evolution configuration parameters
            settings: System settings
        """
        self.config = config or EvolutionConfig()
        self.settings = settings or get_settings()
        
        # Load fitness weights from Settings if not provided in config
        if self.config.fitness_weights is None:
            self.config.fitness_weights = self.settings.genetic_algorithm.fitness_weights
        
        self.seed_registry = get_registry()
        # Evolution state
        self.status = EvolutionStatus.INITIALIZING
        self.current_generation = 0
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        # DEAP framework setup
        self.toolbox = None
        self._setup_deap_framework()
        # Note: multiprocessing pool is created on-demand with context managers
        logger.info("Genetic engine initialized")
    def _setup_deap_framework(self) -> None:
        """Set up DEAP genetic algorithm framework."""
        if not DEAP_AVAILABLE:
            logger.warning("DEAP not available, using fallback implementation")
            return
        try:
            # Create fitness class for multi-objective optimization
            # Convert dictionary weights to tuple format expected by DEAP
            if isinstance(self.config.fitness_weights, dict):
                # Standard order: sharpe_ratio, consistency, max_drawdown (negative), win_rate
                weights_tuple = (
                    self.config.fitness_weights.get("sharpe_ratio", 1.0),
                    self.config.fitness_weights.get("consistency", 0.3), 
                    self.config.fitness_weights.get("max_drawdown", -1.0),  # Minimize drawdown
                    self.config.fitness_weights.get("win_rate", 0.5)
                )
            else:
                # Fallback to default tuple format
                weights_tuple = (1.0, 0.3, -1.0, 0.5)
            
            if not hasattr(creator, "TradingFitness"):
                creator.create("TradingFitness", base.Fitness, 
                             weights=weights_tuple)
            # Create primitive set for trading strategies - strongly typed GP from research
            from deap import gp
            import pandas as pd
            import operator
            # Trading primitive set: market data → trading signals
            self.pset = gp.PrimitiveSetTyped("trading_strategy", [pd.DataFrame], bool)
            # Technical indicator primitives: DataFrame → float
            self.pset.addPrimitive(lambda df: df['close'].iloc[-1] if 'close' in df.columns else 0.0, 
                                 [pd.DataFrame], float, name="current_price")
            self.pset.addPrimitive(lambda df: df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0, 
                                 [pd.DataFrame], float, name="rsi_value")
            # Comparison operators: float, float → bool
            self.pset.addPrimitive(operator.gt, [float, float], bool, name="greater_than")
            self.pset.addPrimitive(operator.lt, [float, float], bool, name="less_than")
            # Logical operators: bool, bool → bool
            self.pset.addPrimitive(operator.and_, [bool, bool], bool, name="and_op")
            self.pset.addPrimitive(operator.or_, [bool, bool], bool, name="or_op")
            # Ephemeral constants for thresholds (research pattern)
            self.pset.addEphemeralConstant("rsi_threshold", lambda: random.uniform(20, 80), float)
            self.pset.addEphemeralConstant("price_threshold", lambda: random.uniform(0.95, 1.05), float)
            # Boolean constants
            self.pset.addTerminal(True, bool)
            self.pset.addTerminal(False, bool)
            # Create individual class - DEAP GP tree-based following research patterns
            if not hasattr(creator, "Individual"):
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.TradingFitness, pset=self.pset)
            # Initialize toolbox
            self.toolbox = base.Toolbox()
            # Register genetic operators
            self._register_genetic_operators()
            logger.info("DEAP framework setup completed")
        except Exception as e:
            logger.error(f"Failed to setup DEAP framework: {e}")
            raise
    def _register_genetic_operators(self) -> None:
        """Register genetic operators with DEAP toolbox."""
        if not self.toolbox:
            return
        # Individual creation
        self.toolbox.register("individual", self._create_random_individual)
        self.toolbox.register("population", tools.initRepeat, list, 
                            self.toolbox.individual)
        # Genetic operators
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.config.tournament_size)
        # Evaluation
        self.toolbox.register("evaluate", self._evaluate_individual)
        # Multi-processing (configured but pool created on-demand)
        if self.config.use_multiprocessing:
            n_processes = self.config.n_processes or multiprocessing.cpu_count()
            # Register map function that will create pool when needed
            self.toolbox.register("map", self._multiprocessing_map)
            logger.info(f"Multiprocessing enabled with {n_processes} processes")
    
    def _multiprocessing_map(self, func, iterable):
        """Multiprocessing map with context manager to avoid serialization issues."""
        n_processes = self.config.n_processes or multiprocessing.cpu_count()
        
        # Use context manager to ensure proper cleanup
        with multiprocessing.Pool(n_processes) as pool:
            return pool.map(func, iterable)
    
    def _create_random_individual(self) -> BaseSeed:
        """Create a random individual from available genetic seeds."""
        # Get available seed classes using the proper registry interface
        try:
            # Import the function that returns actual seed classes
            from src.strategy.genetic_seeds import get_all_genetic_seeds
            seed_classes = get_all_genetic_seeds()
        except ImportError:
            # Fallback: use registry but access the actual classes
            registry_dict = self.seed_registry.list_all_seeds()
            if not registry_dict:
                raise ValueError("No genetic seeds registered")
            # Extract actual seed classes from registry
            seed_classes = []
            for seed_name, seed_info in registry_dict.items():
                # Try to get the actual class from the registry
                registration = self.seed_registry._registry.get(seed_name)
                if registration and registration.status.value == "registered":
                    seed_classes.append(registration.seed_class)
            
        if not seed_classes:
            raise ValueError("No genetic seeds available for evolution")
            
        # Select random seed type
        seed_class = random.choice(seed_classes)
        
        # Create dummy genes to access parameter bounds
        dummy_genes = SeedGenes(
            seed_id="dummy",
            seed_type=SeedType.MOMENTUM,
            generation=0,
            parameters={}
        )
        dummy_instance = seed_class(dummy_genes, self.settings)
        parameter_bounds = dummy_instance.parameter_bounds
        
        # Create random genetic parameters
        parameters = {}
        for param_name, (min_val, max_val) in parameter_bounds.items():
            parameters[param_name] = random.uniform(min_val, max_val)
            
        # Create seed genes
        genes = SeedGenes(
            seed_id=f"gen_{self.current_generation}_{random.randint(1000, 9999)}",
            seed_type=SeedType.MOMENTUM,  # Will be set by seed constructor
            generation=self.current_generation,
            parameters=parameters
        )
        
        # Create and return individual
        individual = seed_class(genes, self.settings)
        
        # Initialize DEAP fitness object (required for DEAP algorithms)
        if DEAP_AVAILABLE and hasattr(creator, "TradingFitness"):
            individual.fitness = creator.TradingFitness()
        else:
            # Fallback fitness object with valid attribute
            class MockFitness:
                def __init__(self):
                    self.valid = False
                    self.values = None
            individual.fitness = MockFitness()
            
        return individual
    def _crossover(self, ind1: BaseSeed, ind2: BaseSeed) -> Tuple[BaseSeed, BaseSeed]:
        """Perform crossover between two individuals.
        Args:
            ind1: First parent individual
            ind2: Second parent individual
        Returns:
            Tuple of two offspring individuals
        """
        # Use seed's built-in crossover method
        try:
            child1, child2 = ind1.crossover(ind2)
            return child1, child2
        except Exception as e:
            logger.warning(f"Crossover failed: {e}, returning parents")
            return ind1, ind2
    def _mutate(self, individual: BaseSeed) -> Tuple[BaseSeed]:
        """Perform mutation on an individual.
        Args:
            individual: Individual to mutate
        Returns:
            Tuple containing mutated individual
        """
        # Use seed's built-in mutation method
        try:
            mutated = individual.mutate(self.config.mutation_prob)
            return (mutated,)
        except Exception as e:
            logger.warning(f"Mutation failed: {e}, returning original")
            return (individual,)
    def _evaluate_individual(self, individual: BaseSeed) -> Tuple[float, float, float, float]:
        """Evaluate fitness of an individual strategy.
        Args:
            individual: Strategy to evaluate
        Returns:
            Tuple of fitness values (sharpe, consistency, drawdown, turnover)
        """
        try:
            # Generate synthetic backtest data for evaluation
            # In production, this would use real market data
            synthetic_data = self._generate_synthetic_market_data()
            # Generate signals using the strategy
            signals = individual.generate_signals(synthetic_data)
            # Calculate basic performance metrics
            returns = self._calculate_strategy_returns(signals, synthetic_data)
            # Multi-objective fitness components
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            consistency = self._calculate_consistency(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            turnover = self._calculate_turnover(signals)
            # Ensure valid fitness values
            sharpe_ratio = max(0.0, min(5.0, sharpe_ratio))
            consistency = max(0.0, min(1.0, consistency))
            max_drawdown = max(0.0, min(1.0, max_drawdown))
            turnover = max(0.0, min(10.0, turnover))
            # Apply constraints
            if sharpe_ratio < self.config.min_sharpe_ratio:
                sharpe_ratio *= 0.5  # Penalty for low Sharpe
            if max_drawdown > self.config.max_drawdown:
                max_drawdown *= 2.0  # Penalty for high drawdown
            # Log evaluation for debugging
            if random.random() < 0.1:  # Log 10% of evaluations
                logger.debug(f"Evaluated {individual.seed_name}: "
                           f"Sharpe={sharpe_ratio:.3f}, Consistency={consistency:.3f}, "
                           f"Drawdown={max_drawdown:.3f}, Turnover={turnover:.3f}")
            return sharpe_ratio, consistency, max_drawdown, turnover
        except Exception as e:
            logger.error(f"Evaluation failed for {individual.seed_name}: {e}")
            # Return poor fitness for failed evaluation
            return 0.0, 0.0, 1.0, 10.0
    def _generate_synthetic_market_data(self, periods: int = 252) -> pd.DataFrame:
        """Generate synthetic market data for backtesting.
        Args:
            periods: Number of periods to generate
        Returns:
            DataFrame with OHLCV data
        """
        # Generate realistic price series with trend and noise
        dates = pd.date_range('2023-01-01', periods=periods, freq='1h')
        # Price evolution with trend and volatility
        returns = np.random.normal(0.0002, 0.02, periods)  # Small positive drift
        prices = 100 * np.exp(np.cumsum(returns))
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(prices[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, periods))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, periods))
        data['volume'] = np.random.uniform(1000, 10000, periods)
        return data
    def _calculate_strategy_returns(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from signals and price data.
        Args:
            signals: Trading signals (-1, 0, 1)
            data: Market data with price information
        Returns:
            Series of strategy returns
        """
        # Calculate price returns
        price_returns = data['close'].pct_change().fillna(0)
        # Strategy returns = signals * price returns (with lag)
        strategy_returns = signals.shift(1) * price_returns
        # Apply transaction costs (0.1% per trade)
        trade_costs = abs(signals.diff()) * 0.001
        strategy_returns = strategy_returns - trade_costs
        return safe_fillna_zero(strategy_returns)
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        # Annualized Sharpe ratio (assuming hourly data)
        mean_return = returns.mean() * 24 * 365  # Annualized
        return_std = returns.std() * np.sqrt(24 * 365)  # Annualized
        sharpe = mean_return / return_std if return_std > 0 else 0.0
        return sharpe
    def _calculate_consistency(self, returns: pd.Series) -> float:
        """Calculate strategy consistency (rolling Sharpe stability)."""
        if len(returns) < 50:
            return 0.0
        # Calculate rolling 30-day Sharpe ratios
        rolling_sharpe = []
        window = 30 * 24  # 30 days in hours
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            if window_returns.std() > 0:
                sharpe = window_returns.mean() / window_returns.std()
                rolling_sharpe.append(sharpe)
        if not rolling_sharpe:
            return 0.0
        # Consistency = 1 - coefficient of variation of rolling Sharpe
        rolling_sharpe = pd.Series(rolling_sharpe)
        cv = rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else float('inf')
        consistency = max(0.0, 1.0 - cv)
        return consistency
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        return min(max_dd, 1.0)  # Cap at 100%
    def _calculate_turnover(self, signals: pd.Series) -> float:
        """Calculate strategy turnover (position changes per period)."""
        if len(signals) == 0:
            return 0.0
        # Count position changes
        position_changes = abs(signals.diff()).sum()
        turnover = position_changes / len(signals)
        return turnover
    def evolve(self, market_data: Optional[pd.DataFrame] = None,
               n_generations: Optional[int] = None,
               asset_dataset: Optional[Any] = None) -> EvolutionResults:
        """Run genetic evolution process with multi-timeframe support.
        
        Args:
            market_data: Historical market data for evaluation (legacy single timeframe)
            n_generations: Number of generations to evolve
            asset_dataset: AssetDataSet with multi-timeframe data for advanced evaluation
            
        Returns:
            Evolution results and statistics
        """
        logger.info("Starting genetic evolution")
        self.status = EvolutionStatus.RUNNING
        start_time = pd.Timestamp.now()
        
        # Multi-timeframe data preparation (research-backed pattern)
        multi_timeframe_data = None
        if asset_dataset and AssetDataSet and hasattr(asset_dataset, 'timeframe_data'):
            multi_timeframe_data = asset_dataset.timeframe_data
            logger.info(f"✅ Multi-timeframe evolution enabled with timeframes: {list(multi_timeframe_data.keys())}")
        elif market_data is not None:
            logger.info("Using legacy single-timeframe market data")
        else:
            logger.warning("No market data provided, using synthetic data")
        
        try:
            # Adjust population size for large population support
            effective_population_size = self.config.population_size
            if (self.config.enable_large_populations and 
                self.config.population_size > 100):
                effective_population_size = min(
                    self.config.population_size, 
                    self.config.max_population_size
                )
                logger.info(f"🚀 Large population mode: {effective_population_size} individuals")
            
            # Initialize population with memory management for large populations
            if (self.config.enable_large_populations and 
                effective_population_size > self.config.memory_chunk_size):
                population = self._initialize_large_population(effective_population_size)
            else:
                population = self._initialize_population()
            # Evolution parameters
            n_gen = n_generations or self.config.n_generations
            # Setup advanced evaluation context for this evolution run
            self._setup_advanced_evaluation_context(
                multi_timeframe_data=multi_timeframe_data,
                legacy_market_data=market_data,
                population_size=effective_population_size
            )
            
            # Track evolution statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            
            # Run evolution with advanced features
            if self.config.enable_walk_forward and multi_timeframe_data:
                population, logbook = self._run_walk_forward_evolution(
                    population, n_gen, stats, multi_timeframe_data
                )
            elif DEAP_AVAILABLE and self.toolbox:
                population, logbook = self._run_deap_evolution(population, n_gen, stats)
            else:
                population, logbook = self._run_fallback_evolution(population, n_gen)
            # Extract results
            best_ind = self._find_best_individual(population)
            # Calculate execution time
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            # Create results
            results = EvolutionResults(
                best_individual=best_ind,
                population=population,
                fitness_history=self.fitness_history,
                generation_stats=[record for record in logbook] if logbook else [],
                execution_time=execution_time,
                status=EvolutionStatus.COMPLETED
            )
            logger.info(f"Evolution completed in {execution_time:.2f}s. "
                       f"Best fitness: {best_ind.fitness.values if best_ind.fitness else 'N/A'}")
            self.status = EvolutionStatus.COMPLETED
            return results
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            self.status = EvolutionStatus.FAILED
            raise
        finally:
            # Multiprocessing pools are managed with context managers - no cleanup needed
            pass
    def _initialize_population(self) -> List[BaseSeed]:
        """Initialize population of genetic individuals."""
        if DEAP_AVAILABLE and self.toolbox:
            population = self.toolbox.population(n=self.config.population_size)
        else:
            population = []
            for _ in range(self.config.population_size):
                individual = self._create_random_individual()
                population.append(individual)
        logger.info(f"Initialized population of {len(population)} individuals")
        return population
    def _run_deap_evolution(self, population: List[BaseSeed], n_generations: int,
                           stats: tools.Statistics) -> Tuple[List[BaseSeed], Any]:
        """Run evolution using DEAP algorithms."""
        # Use NSGA-II for multi-objective optimization
        final_pop, logbook = algorithms.eaMuPlusLambda(
            population, self.toolbox, 
            mu=self.config.population_size,
            lambda_=self.config.population_size,
            cxpb=self.config.crossover_prob,
            mutpb=self.config.mutation_prob,
            ngen=n_generations,
            stats=stats,
            halloffame=None,
            verbose=True
        )
        return final_pop, logbook
    def _run_fallback_evolution(self, population: List[BaseSeed], 
                               n_generations: int) -> Tuple[List[BaseSeed], None]:
        """Run simple evolution when DEAP is not available."""
        logger.warning("Running fallback evolution (DEAP not available)")
        for generation in range(n_generations):
            self.current_generation = generation
            # Evaluate population
            for individual in population:
                if not individual.fitness:
                    fitness_values = self._evaluate_individual(individual)
                    # Create mock fitness object
                    individual.fitness = type('MockFitness', (), {'values': fitness_values})()
            # Simple selection: keep top 50%
            population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
            survivors = population[:self.config.population_size // 2]
            # Create offspring through mutation
            offspring = []
            for individual in survivors:
                child = self._mutate(individual)[0]
                offspring.append(child)
            # Replace population
            population = survivors + offspring
            # Log progress
            if generation % 10 == 0:
                best_fitness = population[0].fitness.values[0]
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.3f}")
        return population, None
    def _find_best_individual(self, population: List[BaseSeed]) -> Optional[BaseSeed]:
        """Find the best individual in the population."""
        if not population:
            return None
        # For multi-objective, use first objective (Sharpe ratio) as primary
        best_individual = None
        best_fitness = float('-inf')
        for individual in population:
            if individual.fitness and individual.fitness.values:
                primary_fitness = individual.fitness.values[0]  # Sharpe ratio
                if primary_fitness > best_fitness:
                    best_fitness = primary_fitness
                    best_individual = individual
        return best_individual
    def get_population_diversity(self, population: List[BaseSeed]) -> Dict[str, float]:
        """Calculate population diversity metrics."""
        if not population:
            return {}
        # Seed type diversity
        seed_types = [ind.genes.seed_type.value for ind in population]
        unique_types = len(set(seed_types))
        type_diversity = unique_types / len(SeedType)
        # Parameter diversity (average std dev of parameters)
        param_stds = []
        for param_name in population[0].genes.parameters.keys():
            param_values = [ind.genes.parameters.get(param_name, 0) for ind in population]
            param_stds.append(np.std(param_values))
        avg_param_diversity = np.mean(param_stds) if param_stds else 0.0
        # Fitness diversity
        fitness_values = [ind.fitness.values[0] if ind.fitness else 0 for ind in population]
        fitness_diversity = np.std(fitness_values)
        return {
            'type_diversity': type_diversity,
            'parameter_diversity': avg_param_diversity,
            'fitness_diversity': fitness_diversity,
            'unique_seed_types': unique_types
        }
    
    # Advanced Features Implementation (Research-Backed Patterns)
    
    def _initialize_large_population(self, population_size: int) -> List[BaseSeed]:
        """Initialize large population with memory-optimized chunked processing.
        
        Based on VectorBT research patterns for 1000+ strategy populations.
        """
        logger.info(f"🚀 Initializing large population: {population_size} individuals")
        
        population = []
        chunk_size = self.config.memory_chunk_size
        
        # Process in chunks to manage memory (research-backed pattern)
        for chunk_start in range(0, population_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, population_size)
            chunk_size_actual = chunk_end - chunk_start
            
            logger.debug(f"Initializing chunk {chunk_start//chunk_size + 1}: "
                        f"{chunk_size_actual} individuals")
            
            # Create chunk using standard initialization
            if DEAP_AVAILABLE and self.toolbox:
                chunk_population = self.toolbox.population(n=chunk_size_actual)
            else:
                chunk_population = []
                for _ in range(chunk_size_actual):
                    individual = self._create_random_individual()
                    chunk_population.append(individual)
            
            population.extend(chunk_population)
            
            # Memory management: force garbage collection between chunks
            import gc
            gc.collect()
        
        logger.info(f"✅ Large population initialized: {len(population)} individuals")
        return population
    
    def _setup_advanced_evaluation_context(self, multi_timeframe_data: Optional[Dict] = None,
                                         legacy_market_data: Optional[pd.DataFrame] = None,
                                         population_size: int = 100) -> None:
        """Setup evaluation context for advanced features.
        
        Prepares the engine for multi-timeframe evaluation, chunked processing,
        and other advanced capabilities based on research patterns.
        """
        # Store evaluation context in instance variables
        self._current_multi_timeframe_data = multi_timeframe_data
        self._current_legacy_data = legacy_market_data
        self._current_population_size = population_size
        
        # Setup multi-timeframe evaluation if enabled
        if self.config.enable_multi_timeframe and multi_timeframe_data:
            self._setup_multi_timeframe_evaluation(multi_timeframe_data)
        
        # Setup chunked evaluation for large populations
        if (self.config.enable_large_populations and 
            population_size > self.config.memory_chunk_size):
            self._setup_chunked_evaluation()
        
        logger.debug(f"Advanced evaluation context setup complete")
    
    def _setup_multi_timeframe_evaluation(self, timeframe_data: Dict[str, pd.DataFrame]) -> None:
        """Setup multi-timeframe evaluation with 0.7/0.3 weighting.
        
        Based on VectorBT research showing optimal strategic/tactical balance.
        """
        strategic_tf, tactical_tf = self.config.timeframe_priorities
        
        if strategic_tf not in timeframe_data or tactical_tf not in timeframe_data:
            logger.warning(f"Missing required timeframes {strategic_tf}/{tactical_tf}, "
                          f"available: {list(timeframe_data.keys())}")
            return
        
        self._strategic_data = timeframe_data[strategic_tf]
        self._tactical_data = timeframe_data[tactical_tf]
        
        logger.info(f"✅ Multi-timeframe setup: {strategic_tf} (weight: {self.config.strategic_timeframe_weight}) "
                   f"+ {tactical_tf} (weight: {self.config.tactical_timeframe_weight})")
    
    def _setup_chunked_evaluation(self) -> None:
        """Setup chunked evaluation for memory-efficient large population processing."""
        # Override the standard evaluation function with chunked version
        if self.toolbox:
            original_evaluate = self.toolbox.evaluate
            self.toolbox.register("evaluate", self._chunked_evaluate_individual)
            self._original_evaluate = original_evaluate
        
        logger.info(f"✅ Chunked evaluation enabled (chunk size: {self.config.memory_chunk_size})")
    
    def _chunked_evaluate_individual(self, individual: BaseSeed) -> Tuple[float, float, float, float]:
        """Evaluate individual using chunked processing for memory efficiency."""
        try:
            # Use multi-timeframe evaluation if available
            if (self.config.enable_multi_timeframe and 
                hasattr(self, '_strategic_data') and hasattr(self, '_tactical_data')):
                return self._evaluate_multi_timeframe_fitness(individual)
            else:
                # Fall back to standard evaluation
                return self._evaluate_individual(individual)
        except Exception as e:
            logger.error(f"Chunked evaluation failed for {individual.seed_name}: {e}")
            return 0.0, 0.0, 1.0, 10.0
    
    def _evaluate_multi_timeframe_fitness(self, individual: BaseSeed) -> Tuple[float, float, float, float]:
        """Evaluate fitness across multiple timeframes with 0.7/0.3 weighting.
        
        Research-backed pattern for balancing strategic and tactical performance.
        """
        try:
            # Strategic timeframe evaluation (typically 1h)
            strategic_signals = individual.generate_signals(self._strategic_data)
            strategic_returns = self._calculate_strategy_returns(strategic_signals, self._strategic_data)
            strategic_sharpe = self._calculate_sharpe_ratio(strategic_returns)
            strategic_consistency = self._calculate_consistency(strategic_returns)
            strategic_drawdown = self._calculate_max_drawdown(strategic_returns)
            strategic_turnover = self._calculate_turnover(strategic_signals)
            
            # Tactical timeframe evaluation (typically 15m)
            tactical_signals = individual.generate_signals(self._tactical_data)
            tactical_returns = self._calculate_strategy_returns(tactical_signals, self._tactical_data)
            tactical_sharpe = self._calculate_sharpe_ratio(tactical_returns)
            tactical_consistency = self._calculate_consistency(tactical_returns)
            tactical_drawdown = self._calculate_max_drawdown(tactical_returns)
            tactical_turnover = self._calculate_turnover(tactical_signals)
            
            # Weighted combination (research-backed 0.7/0.3 weighting)
            strategic_weight = self.config.strategic_timeframe_weight
            tactical_weight = self.config.tactical_timeframe_weight
            
            combined_sharpe = (strategic_weight * strategic_sharpe + 
                             tactical_weight * tactical_sharpe)
            combined_consistency = (strategic_weight * strategic_consistency + 
                                  tactical_weight * tactical_consistency)
            combined_drawdown = (strategic_weight * strategic_drawdown + 
                               tactical_weight * tactical_drawdown)
            combined_turnover = (strategic_weight * strategic_turnover + 
                               tactical_weight * tactical_turnover)
            
            # Apply constraints and bounds
            combined_sharpe = max(0.0, min(5.0, combined_sharpe))
            combined_consistency = max(0.0, min(1.0, combined_consistency))
            combined_drawdown = max(0.0, min(1.0, combined_drawdown))
            combined_turnover = max(0.0, min(10.0, combined_turnover))
            
            # Log evaluation for debugging (10% sample)
            if random.random() < 0.1:
                logger.debug(f"Multi-timeframe eval {individual.seed_name}: "
                           f"Strategic={strategic_sharpe:.3f}, Tactical={tactical_sharpe:.3f}, "
                           f"Combined={combined_sharpe:.3f}")
            
            return combined_sharpe, combined_consistency, combined_drawdown, combined_turnover
            
        except Exception as e:
            logger.error(f"Multi-timeframe evaluation failed for {individual.seed_name}: {e}")
            return 0.0, 0.0, 1.0, 10.0
    
    def _run_walk_forward_evolution(self, population: List[BaseSeed], n_generations: int,
                                   stats: Any, timeframe_data: Dict[str, pd.DataFrame]) -> Tuple[List[BaseSeed], Any]:
        """Run evolution with walk-forward validation for overfitting prevention.
        
        Based on DEAP research emphasizing multiple validation periods.
        """
        logger.info(f"🚶 Starting walk-forward evolution with {self.config.walk_forward_periods} periods")
        
        # Use the primary timeframe for walk-forward validation
        primary_timeframe = self.config.timeframe_priorities[0]
        primary_data = timeframe_data[primary_timeframe]
        
        # Create validation periods
        total_periods = len(primary_data)
        window_size = min(
            len(primary_data) // self.config.walk_forward_periods,
            self.config.validation_window_days * 24  # Assuming hourly data
        )
        
        validation_periods = []
        for i in range(self.config.walk_forward_periods):
            start_idx = i * (total_periods // self.config.walk_forward_periods)
            end_idx = min(start_idx + window_size, total_periods)
            validation_periods.append((start_idx, end_idx))
        
        logger.info(f"Created {len(validation_periods)} validation periods")
        
        # Run evolution for each period and aggregate results
        all_logbooks = []
        final_population = population
        
        for period_idx, (start_idx, end_idx) in enumerate(validation_periods):
            logger.info(f"Walk-forward period {period_idx + 1}/{len(validation_periods)}")
            
            # Extract period data
            period_data = {
                tf: data.iloc[start_idx:end_idx] 
                for tf, data in timeframe_data.items()
            }
            
            # Update evaluation context for this period
            self._setup_multi_timeframe_evaluation(period_data)
            
            # Run evolution for this period
            if DEAP_AVAILABLE and self.toolbox:
                period_population, period_logbook = self._run_deap_evolution(
                    final_population, n_generations // self.config.walk_forward_periods, stats
                )
                all_logbooks.append(period_logbook)
                final_population = period_population
            else:
                period_population, _ = self._run_fallback_evolution(
                    final_population, n_generations // self.config.walk_forward_periods
                )
                final_population = period_population
        
        # Combine logbooks
        combined_logbook = all_logbooks[0] if all_logbooks else None
        if len(all_logbooks) > 1:
            for logbook in all_logbooks[1:]:
                if combined_logbook and logbook:
                    combined_logbook.extend(logbook)
        
        logger.info("✅ Walk-forward evolution completed")
        return final_population, combined_logbook
    
    def __del__(self):
        """Cleanup method - pools are managed with context managers."""
        pass
