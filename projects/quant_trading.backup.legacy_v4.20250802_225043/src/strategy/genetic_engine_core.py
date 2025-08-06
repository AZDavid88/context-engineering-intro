"""
Genetic Engine Core - DEAP framework integration and core genetic operations.
Handles DEAP setup, genetic operators, and fundamental evolution workflow.
"""

import random  
import logging
import operator  
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna
from src.strategy.genetic_seeds import SeedRegistry, get_registry, BaseSeed
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedFitness, SeedType
from src.config.settings import get_settings, Settings

# DEAP framework imports
try:
    from deap import base, creator, tools, algorithms, gp
    import pandas as pd
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    # Create mock classes for testing
    class MockBase:
        def __init__(self): pass
    creator = MockBase()
    tools = MockBase()
    algorithms = MockBase()
    gp = MockBase()

# Configure logging
logger = logging.getLogger(__name__)


class EvolutionStatus(str, Enum):
    """Evolution process status tracking."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class EvolutionConfig:
    """Configuration for genetic algorithm evolution process."""
    
    # Population parameters
    population_size: int = 50
    elite_size: int = 5  # Number of best individuals to preserve each generation
    
    # Genetic operator parameters
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    tournament_size: int = 3
    
    # Evolution parameters
    max_generations: int = 20
    convergence_threshold: float = 1e-6  # Fitness improvement threshold for early stopping
    stagnation_limit: int = 5  # Generations without improvement before stopping
    
    # Fitness weights for multi-objective optimization (sum should be ~2.0)
    fitness_weights: Optional[Dict[str, float]] = None  # Will use settings default if None
    
    # Performance parameters
    enable_multiprocessing: bool = True
    max_processes: Optional[int] = None  # Will use CPU count if None
    
    # Advanced features
    enable_diversity_maintenance: bool = True
    diversity_threshold: float = 0.1
    adaptive_mutation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.population_size < 10:
            raise ValueError("Population size must be at least 10")
        if self.elite_size >= self.population_size:
            raise ValueError("Elite size must be less than population size")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("Mutation rate must be between 0.0 and 1.0")


@dataclass  
class EvolutionResults:
    """Results from genetic algorithm evolution process."""
    best_individual: Optional[BaseSeed]
    final_population: List[BaseSeed]
    fitness_history: List[List[float]]
    generation_count: int
    evolution_time: float


class GeneticEngineCore:
    """Core genetic algorithm engine with DEAP framework integration."""
    
    def __init__(self, config: Optional[EvolutionConfig] = None, 
                 settings: Optional[Settings] = None):
        """Initialize genetic engine core.
        
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
        self.pset = None
        self._setup_deap_framework()
        
        logger.info("Genetic engine core initialized")
    
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
                creator.create("TradingFitness", base.Fitness, weights=weights_tuple)
            
            # Create primitive set for trading strategies - strongly typed GP
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
            
            # Ephemeral constants for thresholds
            self.pset.addEphemeralConstant("rsi_threshold", lambda: random.uniform(20, 80), float)
            self.pset.addEphemeralConstant("price_threshold", lambda: random.uniform(0.95, 1.05), float)
            
            # Boolean constants
            self.pset.addTerminal(True, bool)
            self.pset.addTerminal(False, bool)
            
            # Create individual class - DEAP GP tree-based
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
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        
        # Note: Evaluation is registered in the evaluation module
        logger.info("Genetic operators registered")
    
    def _create_random_individual(self) -> BaseSeed:
        """Create a random individual from available genetic seeds."""
        try:
            # Get available seed types
            available_seeds = list(self.seed_registry.get_all_seeds().keys())
            
            if not available_seeds:
                logger.warning("No genetic seeds available, creating default seed")
                # Create a minimal fallback seed if registry is empty
                class FallbackSeed(BaseSeed):
                    def __init__(self):
                        super().__init__()
                        self.seed_type = SeedType.TECHNICAL
                        self.genes = SeedGenes()
                        self.fitness = SeedFitness()
                
                return FallbackSeed()
            
            # Select random seed type and create instance
            seed_type = random.choice(available_seeds)
            seed_class = self.seed_registry.get_seed(seed_type)
            
            if seed_class:
                individual = seed_class()
                
                # Apply random mutations to diversify the initial population
                if hasattr(individual, 'mutate'):
                    individual.mutate(rate=0.3)  # Higher mutation rate for initialization
                
                return individual
            else:
                logger.error(f"Failed to create seed of type: {seed_type}")
                raise ValueError(f"Seed creation failed for type: {seed_type}")
                
        except Exception as e:
            logger.error(f"Error creating random individual: {e}")
            # Return a basic fallback individual
            class FallbackSeed(BaseSeed):
                def __init__(self):
                    super().__init__()
                    self.seed_type = SeedType.TECHNICAL
                    self.genes = SeedGenes()
                    self.fitness = SeedFitness()
            
            return FallbackSeed()
    
    def _crossover(self, ind1: BaseSeed, ind2: BaseSeed) -> Tuple[BaseSeed, BaseSeed]:
        """Perform crossover between two individuals."""
        try:
            # Create offspring by copying parents
            child1, child2 = type(ind1)(), type(ind2)()
            
            # Copy genes from parents
            if hasattr(ind1, 'genes') and hasattr(ind2, 'genes'):
                child1.genes = ind1.genes.copy() if hasattr(ind1.genes, 'copy') else ind1.genes
                child2.genes = ind2.genes.copy() if hasattr(ind2.genes, 'copy') else ind2.genes
                
                # Perform crossover if available
                if hasattr(child1, 'crossover') and hasattr(child2, 'crossover'):
                    child1.crossover(child2)
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            return ind1, ind2  # Return originals if crossover fails
    
    def _mutate(self, individual: BaseSeed) -> Tuple[BaseSeed]:
        """Mutate an individual."""
        try:
            mutated = type(individual)()
            
            # Copy genes
            if hasattr(individual, 'genes'):
                mutated.genes = individual.genes.copy() if hasattr(individual.genes, 'copy') else individual.genes
            
            # Apply mutation
            if hasattr(mutated, 'mutate'):
                mutated.mutate(rate=self.config.mutation_rate)
            
            return (mutated,)
            
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            return (individual,)  # Return original if mutation fails
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """Get current fitness weights configuration."""
        if isinstance(self.config.fitness_weights, dict):
            return self.config.fitness_weights.copy()
        else:
            # Return default weights
            return {
                "sharpe_ratio": 1.0,
                "consistency": 0.3,
                "max_drawdown": -1.0,
                "win_rate": 0.5
            }
    
    def update_config(self, **updates) -> None:
        """Update evolution configuration parameters."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config parameter {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        return {
            'status': self.status.value,
            'current_generation': self.current_generation,
            'population_size': len(self.population),
            'best_fitness': self.best_individual.fitness.sharpe_ratio if self.best_individual else None,
            'deap_available': DEAP_AVAILABLE,
            'toolbox_ready': self.toolbox is not None
        }
    
    def reset_evolution(self) -> None:
        """Reset evolution state for new run."""
        self.status = EvolutionStatus.READY
        self.current_generation = 0
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        logger.info("Evolution state reset")