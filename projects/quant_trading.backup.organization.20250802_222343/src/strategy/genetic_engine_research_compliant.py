"""
Research-Compliant Genetic Engine - Following DEAP Patterns Exactly

This implementation follows the DEAP research patterns from /research/deap/research_summary.md
exactly, without over-engineering or serialization workarounds.

Key Research Patterns Implemented:
1. Simple evaluation functions (no lambda closures)
2. Standard DEAP toolbox registration
3. Direct multiprocessing.Pool() usage as shown in research
4. Existing genetic seeds as individuals (not GP trees)
5. Module-level functions for all operations
"""

import random
import logging
import multiprocessing
import time
from typing import Dict, List, Tuple, Optional, Any, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# Import existing components
from src.strategy.genetic_seeds import get_all_genetic_seeds, BaseSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedFitness, SeedType
from src.config.settings import get_settings, Settings

# DEAP imports with fallback
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    # Mock classes for testing without DEAP
    class MockBase:
        pass
    creator = MockBase()
    tools = MockBase()
    algorithms = MockBase()

logger = logging.getLogger(__name__)


class EvolutionStatus(str, Enum):
    """Evolution status tracking."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResearchCompliantConfig:
    """Research-compliant configuration matching DEAP patterns."""
    # Population parameters from research
    population_size: int = 200  # Research: 100-200 strategies per generation
    n_generations: int = 50     # Research: 20-50 generations for discovery
    
    # DEAP standard parameters
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 7
    
    # Multi-objective fitness weights (research pattern)
    fitness_weights: Tuple[float, float, float] = (1.0, -1.0, 1.0)  # Sharpe, Drawdown, Consistency
    
    # Multiprocessing (research pattern)
    use_multiprocessing: bool = True
    n_processes: Optional[int] = None  # Auto-detect
    
    # Performance thresholds from research
    min_sharpe_ratio: float = 2.0
    max_drawdown_threshold: float = 0.15
    min_trades: int = 30


@dataclass
class ResearchCompliantResults:
    """Results following research patterns."""
    best_individual: Optional[BaseSeed]
    population: List[BaseSeed]
    fitness_history: List[Dict[str, float]]
    execution_time: float
    status: EvolutionStatus
    generation_stats: List[Dict]


# Module-level conversion function (EXACT DOCUMENTATION PATTERN)
def individual_to_seed(individual, genetic_seeds: List[Type[BaseSeed]]) -> BaseSeed:
    """Convert DEAP individual to BaseSeed instance."""
    # DOCUMENTATION PATTERN: Individual is a list of floats (genes)
    # First gene determines seed class (scaled to valid range)
    seed_class_index = int(individual[0] * len(genetic_seeds)) % len(genetic_seeds)
    seed_class = genetic_seeds[seed_class_index]
    
    # Create dummy instance to get parameter names and bounds
    dummy_genes = SeedGenes(
        seed_id=seed_class.__name__,
        seed_type=getattr(seed_class, '_seed_type', SeedType.MOMENTUM),
        parameters={}
    )
    dummy_instance = seed_class(dummy_genes)
    param_names = list(dummy_instance.parameter_bounds.keys())
    param_bounds = dummy_instance.parameter_bounds
    
    # Convert normalized genes to parameter values within bounds
    parameters = {}
    for i, param_name in enumerate(param_names):
        if i + 1 < len(individual):
            # Scale [0,1] gene to parameter bounds
            min_val, max_val = param_bounds[param_name]
            normalized_value = individual[i + 1]  # Gene value [0,1]
            parameters[param_name] = min_val + normalized_value * (max_val - min_val)
    
    # Create genes and seed instance
    genes = SeedGenes(
        seed_id=seed_class.__name__,
        seed_type=getattr(seed_class, '_seed_type', SeedType.MOMENTUM),
        parameters=parameters
    )
    
    return seed_class(genes)


# Module-level evaluation function (research pattern) - COMPLETELY STATELESS
def evaluate_strategy(individual, market_data: pd.DataFrame, genetic_seeds: List[Type[BaseSeed]]) -> Tuple[float, float, float]:
    """
    Evaluate individual strategy following EXACT research patterns.
    
    CRITICAL: This function MUST be stateless for multiprocessing serialization.
    Research pattern: Simple function that returns tuple of fitness values.
    """
    try:
        # Convert individual to seed instance
        seed_instance = individual_to_seed(individual, genetic_seeds)
        
        # Generate signals using the seed's strategy
        signals = seed_instance.generate_signals(market_data)
        
        if signals is None or len(signals) == 0:
            return (-999.0, 1.0, 0.0)  # Poor fitness for invalid strategies (research pattern)
        
        # Calculate returns
        returns = signals.shift(1) * market_data['close'].pct_change()
        returns = returns.dropna()
        
        if len(returns) < 10:
            return (-999.0, 1.0, 0.0)
        
        # Calculate fitness metrics (research pattern)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(returns)
        consistency = calculate_consistency(returns)
        
        return (sharpe_ratio, max_drawdown, consistency)
        
    except Exception as e:
        logger.debug(f"Evaluation error for individual: {e}")
        return (-999.0, 1.0, 0.0)  # Research pattern: graceful failure


# Module-level helper functions (research pattern)
def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """Calculate Sharpe ratio following research patterns."""
    if len(returns) == 0 or returns.std() == 0:
        return -999.0
    return (returns.mean() / returns.std()) * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown following research patterns."""
    if len(returns) == 0:
        return 1.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return abs(drawdown.min())


def calculate_consistency(returns: pd.Series) -> float:
    """Calculate consistency score following research patterns."""
    if len(returns) < 20:
        return 0.0
    # Rolling Sharpe ratio consistency
    rolling_sharpe = returns.rolling(20).apply(lambda x: x.mean() / x.std() if x.std() > 0 else 0)
    return 1.0 / (1.0 + rolling_sharpe.std()) if rolling_sharpe.std() > 0 else 0.0


# Module-level crossover function (research pattern)
def crossover_strategies(ind1: list, ind2: list) -> Tuple[list, list]:
    """Crossover two strategies following research patterns."""
    # Simple parameter crossover - skip seed class index, swap parameters
    new_ind1 = ind1.copy()
    new_ind2 = ind2.copy()
    
    # Cross over parameters randomly (skip index 0 which is seed class)
    for i in range(1, min(len(ind1), len(ind2))):
        if random.random() < 0.5:
            new_ind1[i], new_ind2[i] = new_ind2[i], new_ind1[i]
    
    return new_ind1, new_ind2


# Module-level mutation function (research pattern)
def mutate_strategy(individual: list) -> Tuple[list]:
    """Mutate strategy following research patterns."""
    # Simple parameter mutation
    new_individual = individual.copy()
    
    # Mutate parameters randomly (skip index 0 which is seed class)
    for i in range(1, len(individual)):
        if random.random() < 0.1:  # 10% mutation rate per parameter
            if isinstance(individual[i], (int, float)):
                new_individual[i] = individual[i] * random.uniform(0.8, 1.2)  # 20% variation
    
    return (new_individual,)


class ResearchCompliantGeneticEngine:
    """
    Research-compliant genetic engine following DEAP patterns exactly.
    
    Based on /research/deap/research_summary.md patterns:
    - Simple module-level evaluation functions
    - Standard DEAP toolbox registration  
    - Direct multiprocessing.Pool() usage
    - No lambda closures or serialization workarounds
    """
    
    def __init__(self, config: Optional[ResearchCompliantConfig] = None, settings: Optional[Settings] = None):
        """Initialize following research patterns."""
        self.config = config or ResearchCompliantConfig()
        self.settings = settings or get_settings()
        
        # Evolution state
        self.status = EvolutionStatus.INITIALIZING
        self.current_generation = 0
        self.population = []
        self.fitness_history = []
        
        # Get available genetic seeds
        self.genetic_seeds = get_all_genetic_seeds()
        if not self.genetic_seeds:
            raise ValueError("No genetic seeds available")
        
        # Setup DEAP framework following research patterns
        self.toolbox = None
        self._setup_deap_framework()
        
        logger.info(f"Research-compliant genetic engine initialized with {len(self.genetic_seeds)} seed types")
    
    def _setup_deap_framework(self):
        """Setup DEAP framework following research patterns exactly."""
        if not DEAP_AVAILABLE:
            logger.warning("DEAP not available, using fallback")
            return
        
        try:
            # Create fitness class (EXACT DOCUMENTATION PATTERN)
            if not hasattr(creator, "TradingFitness"):
                creator.create("TradingFitness", base.Fitness, weights=self.config.fitness_weights)
            
            # Create individual class (EXACT DOCUMENTATION PATTERN - List of Floats)
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.TradingFitness)
            
            # Create toolbox (EXACT DOCUMENTATION PATTERN)
            self.toolbox = base.Toolbox()
            
            # EXACT DOCUMENTATION PATTERN: Register individual creation 
            self.toolbox.register("attr_gene", self._create_gene_value)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                self.toolbox.attr_gene, n=self._get_gene_count())
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            
            # Register genetic operators (EXACT DEAP BUILT-IN OPERATORS)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
            
            # Multiprocessing setup (research pattern) - Pool created in evolve() method
            if self.config.use_multiprocessing:
                n_processes = self.config.n_processes or multiprocessing.cpu_count()
                logger.info(f"Multiprocessing configured for {n_processes} processes")
            else:
                logger.info("Single-threaded evolution configured")
            
            logger.info("DEAP framework setup completed (research-compliant)")
            
        except Exception as e:
            logger.error(f"DEAP setup failed: {e}")
            raise
    
    def _get_gene_count(self):
        """Get total number of genes needed for individual."""
        # Calculate maximum genes needed across all seed types
        max_genes = 1  # For seed class index
        for seed_class in self.genetic_seeds:
            dummy_genes = SeedGenes(
                seed_id=seed_class.__name__,
                seed_type=getattr(seed_class, '_seed_type', SeedType.MOMENTUM),
                parameters={}
            )
            dummy_instance = seed_class(dummy_genes)
            max_genes = max(max_genes, 1 + len(dummy_instance.parameter_bounds))
        return max_genes
    
    def _create_gene_value(self):
        """Create a single gene value following EXACT documentation pattern."""
        # EXACT DOCUMENTATION: toolbox.register("attr_float", random.random)
        return random.random()
    
    def _register_evaluation(self, market_data: pd.DataFrame):
        """Register evaluation function in toolbox with bound data."""
        # EXACT DOCUMENTATION PATTERN: toolbox.register("evaluate", evaluateInd)
        from functools import partial
        eval_func = partial(evaluate_strategy, 
                          market_data=market_data, 
                          genetic_seeds=self.genetic_seeds)
        self.toolbox.register("evaluate", eval_func)
    
    def evolve(self, market_data: pd.DataFrame, n_generations: Optional[int] = None) -> ResearchCompliantResults:
        """
        Run evolution following research patterns exactly.
        
        Research pattern: Context manager for multiprocessing Pool from 
        /research/deap/4_multiprocessing_distributed_evaluation.md lines 98-105
        """
        logger.info("Starting research-compliant genetic evolution")
        self.status = EvolutionStatus.RUNNING
        start_time = time.time()
        
        # Market data will be bound to evaluation function directly
        
        try:
            # Initialize population (research pattern)
            population = self.toolbox.population(n=self.config.population_size)
            logger.info(f"Initialized population of {len(population)} individuals")
            
            # Evolution parameters
            n_gen = n_generations or self.config.n_generations
            
            # Statistics tracking (research pattern)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            
            # EXACT DOCUMENTATION PATTERN: Register evaluation in toolbox
            self._register_evaluation(market_data)
            
            # EXACT DOCUMENTATION PATTERN: algorithms.eaSimple usage
            if DEAP_AVAILABLE and self.toolbox and self.config.use_multiprocessing:
                # RESEARCH PATTERN: Use context manager for Pool
                with multiprocessing.Pool() as pool:
                    # Register pool map in toolbox
                    self.toolbox.register("map", pool.map)
                    
                    population, logbook = algorithms.eaSimple(
                        population, self.toolbox,
                        cxpb=self.config.crossover_prob,
                        mutpb=self.config.mutation_prob,
                        ngen=n_gen,
                        stats=stats,
                        verbose=True
                    )
            elif DEAP_AVAILABLE and self.toolbox:
                # Single-threaded evolution
                population, logbook = algorithms.eaSimple(
                    population, self.toolbox,
                    cxpb=self.config.crossover_prob,
                    mutpb=self.config.mutation_prob,
                    ngen=n_gen,
                    stats=stats,
                    verbose=True
                )
            else:
                # Fallback evolution
                population, logbook = self._fallback_evolution(population, n_gen)
            
            # Find best individual (research pattern)
            best_individual = max(population, key=lambda ind: ind.fitness.values[0])
            
            # Convert best individual back to BaseSeed for results
            best_individual_seed = individual_to_seed(best_individual, self.genetic_seeds)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create results
            results = ResearchCompliantResults(
                best_individual=best_individual_seed,
                population=population,  # Keep as list for DEAP compatibility
                fitness_history=self.fitness_history,
                execution_time=execution_time,
                status=EvolutionStatus.COMPLETED,
                generation_stats=[dict(record) for record in logbook] if logbook else []
            )
            
            logger.info(f"Evolution completed in {execution_time:.2f}s")
            logger.info(f"Best fitness: {best_individual.fitness.values}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            self.status = EvolutionStatus.FAILED
            raise
    
    def _fallback_evolution(self, population: List, n_generations: int) -> Tuple[List, List]:
        """Fallback evolution when DEAP not available."""
        logger.info("Using fallback evolution (DEAP not available)")
        
        for generation in range(n_generations):
            # Evaluate population using toolbox.evaluate
            fitnesses = [self.toolbox.evaluate(ind) for ind in population]
            for ind, fit in zip(population, fitnesses):
                if hasattr(ind, 'fitness'):
                    ind.fitness.values = fit
                else:
                    ind.fitness = type('Fitness', (), {'values': fit})()
            
            # Select next generation
            population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
            population = population[:self.config.population_size]  # Keep best
            
            logger.info(f"Generation {generation}: Best fitness = {population[0].fitness.values[0]:.3f}")
        
        return population, []


# Factory function for easy usage
def create_research_compliant_engine(settings: Optional[Settings] = None) -> ResearchCompliantGeneticEngine:
    """Create research-compliant genetic engine with optimal settings."""
    config = ResearchCompliantConfig(
        population_size=200,  # Research optimal
        n_generations=50,     # Research optimal
        use_multiprocessing=True
    )
    
    return ResearchCompliantGeneticEngine(config=config, settings=settings)