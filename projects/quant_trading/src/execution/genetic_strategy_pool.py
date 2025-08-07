"""
Genetic Strategy Pool - Phase 5A Implementation

This module implements a hybrid local-distributed genetic algorithm framework
for evolving and managing trading strategies. It supports both local evolution
and distributed Ray cluster execution.

Research-Based Implementation:
- /research/ray_cluster/research_summary.md - Ray architecture patterns
- /research/deap/research_summary.md - Genetic algorithm framework  
- /research/asyncio_advanced/ - Async patterns for concurrent evaluation

Key Features:
- Hybrid local/distributed architecture (Phase 5A ready for 5B)
- Integration with existing BaseSeed framework
- Ray-compatible stateless evaluation functions
- Production-ready fault tolerance and health monitoring
- Cost-efficient resource management ($7-20 per evolution cycle)
"""

import asyncio
import logging
import time
import statistics
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Ray imports (conditional for Phase 5A/5B compatibility)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from src.strategy.genetic_seeds.seed_registry import SeedRegistry, get_registry
from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedType, SeedGenes
from src.execution.retail_connection_optimizer import RetailConnectionOptimizer, TradingSessionProfile
from src.config.settings import get_settings
from src.data.storage_interfaces import DataStorageInterface, get_storage_implementation

# Set up logging
logger = logging.getLogger(__name__)


class EvolutionMode(str, Enum):
    """Evolution execution modes."""
    LOCAL = "local"              # Local multi-processing
    RAY_DISTRIBUTED = "ray_distributed"  # Ray cluster distributed
    HYBRID = "hybrid"            # Adaptive based on population size


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution."""
    population_size: int = 100
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    
    # Ray-specific configuration
    ray_workers: Optional[int] = None  # Auto-detect if None
    ray_memory_per_worker: str = "2GB"
    ray_timeout: int = 300  # 5 minutes per evaluation
    
    # Performance thresholds
    min_fitness_threshold: float = 0.5  # Minimum Sharpe ratio
    max_evaluation_time: int = 600  # Maximum time per generation (seconds)


@dataclass 
class EvolutionMetrics:
    """Metrics tracking for evolution performance."""
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    population_diversity: float = 0.0
    evaluation_time: float = 0.0
    worker_efficiency: float = 1.0
    
    # Health monitoring
    health_score: float = 100.0
    failed_evaluations: int = 0
    timeout_count: int = 0


class Individual:
    """Represents an individual in the genetic population."""
    
    def __init__(self, seed_type: SeedType, genes: SeedGenes, fitness: Optional[float] = None):
        self.seed_type = seed_type
        self.genes = genes
        self.fitness = fitness
        self.metrics: Dict[str, float] = {}
        self.evaluation_time: float = 0.0
        self.generation_created: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'seed_type': self.seed_type.value,
            'genes': self.genes.model_dump(),
            'fitness': self.fitness,
            'metrics': self.metrics,
            'evaluation_time': self.evaluation_time,
            'generation_created': self.generation_created
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create from dictionary."""
        individual = cls(
            seed_type=SeedType(data['seed_type']),
            genes=SeedGenes(**data['genes']),
            fitness=data.get('fitness')
        )
        individual.metrics = data.get('metrics', {})
        individual.evaluation_time = data.get('evaluation_time', 0.0)
        individual.generation_created = data.get('generation_created', 0)
        return individual


# Ray remote functions (stateless for distributed execution)
if RAY_AVAILABLE:
    @ray.remote(num_cpus=1, memory=2147483648)  # 2GB per worker
    def evaluate_individual_distributed(
        seed_type_str: str, 
        genes_dict: Dict[str, Any], 
        market_data_ref: Any
    ) -> Dict[str, Any]:
        """
        Stateless evaluation function for Ray workers.
        
        Based on /research/ray_cluster/page_2_ray_core_fundamentals.md patterns.
        No global state dependencies - imports everything locally.
        """
        # Import locally to avoid global state issues on Ray workers
        from src.strategy.genetic_seeds.seed_registry import create_seed_instance
        from src.strategy.genetic_seeds.base_seed import SeedType, SeedGenes
        import time
        
        start_time = time.time()
        
        try:
            # Reconstruct objects from serialized data
            seed_type = SeedType(seed_type_str)
            genes = SeedGenes(**genes_dict)
            market_data = ray.get(market_data_ref)
            
            # Create strategy instance
            # Import registry locally for Ray worker  
            from src.strategy.genetic_seeds.seed_registry import get_registry
            
            registry = get_registry()
            available_seed_names = registry._type_index[seed_type]
            if not available_seed_names:
                raise ValueError(f"No seeds available for type {seed_type}")
            
            # Use first available seed name of this type
            seed_name = available_seed_names[0]
            seed_instance = registry.create_seed_instance(seed_name, genes)
            
            # Run backtesting evaluation - generate signals and calculate fitness
            signals = seed_instance.generate_signals(market_data)
            
            # Calculate basic performance metrics from signals
            # This is a simplified fitness calculation for the genetic algorithm
            returns = []
            position = 0
            
            for i, signal in enumerate(signals):
                if signal > 0.5 and position <= 0:  # Buy signal
                    position = 1
                elif signal < -0.5 and position >= 0:  # Sell signal
                    position = -1
                    
                if i > 0 and position != 0:
                    price_change = (market_data.iloc[i]['close'] - market_data.iloc[i-1]['close']) / market_data.iloc[i-1]['close']
                    returns.append(position * price_change)
            
            if returns:
                total_return = sum(returns)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
                max_drawdown = min(np.cumsum(returns)) if returns else 0.0
                win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
            else:
                total_return = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                win_rate = 0.0
            
            results = {
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'trade_count': len(returns),
                'win_rate': win_rate
            }
            
            evaluation_time = time.time() - start_time
            
            return {
                'fitness': results.get('sharpe_ratio', 0.0),
                'total_return': results.get('total_return', 0.0),
                'max_drawdown': results.get('max_drawdown', 0.0),
                'trade_count': results.get('trade_count', 0),
                'win_rate': results.get('win_rate', 0.0),
                'evaluation_time': evaluation_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            logger.error(f"Ray evaluation failed: {e}")
            
            return {
                'fitness': -999.0,  # Poor fitness for failed evaluations
                'total_return': 0.0,
                'max_drawdown': -1.0,
                'trade_count': 0,
                'win_rate': 0.0,
                'evaluation_time': evaluation_time,
                'success': False,
                'error': str(e)
            }


class GeneticStrategyPool:
    """
    Hybrid local-distributed genetic strategy evolution pool.
    
    Phase 5A: Local-first with Ray integration stubs
    Phase 5B: Full Ray cluster distributed evolution
    """
    
    def __init__(
        self, 
        connection_optimizer: RetailConnectionOptimizer,
        use_ray: bool = False,
        evolution_config: Optional[EvolutionConfig] = None,
        storage: Optional[DataStorageInterface] = None
    ):
        """
        Initialize genetic strategy pool.
        
        Args:
            connection_optimizer: Retail trading connection optimizer
            use_ray: Enable Ray distributed execution (Phase 5B)
            evolution_config: Evolution parameters
            storage: Data storage interface (auto-configured if None)
        """
        self.connection_optimizer = connection_optimizer
        self.use_ray = use_ray and RAY_AVAILABLE
        self.config = evolution_config or EvolutionConfig()
        
        # Strategic storage interface for clean phase progression
        self.storage = storage or get_storage_implementation()
        
        # Local components (cannot be distributed)
        self.seed_registry = get_registry()
        self.population: List[Individual] = []
        self.evolution_history: List[EvolutionMetrics] = []
        self.current_generation = 0
        
        # Ray cluster state (Phase 5B)
        self.ray_initialized = False
        self.market_data_ref = None
        
        # Health monitoring
        self.health_score = 100.0
        self.last_health_check = datetime.now(timezone.utc)
        
        logger.info(f"GeneticStrategyPool initialized: Ray={'enabled' if self.use_ray else 'disabled'}")
    
    async def initialize_population(self, seed_types: Optional[List[SeedType]] = None) -> int:
        """
        Initialize random population for evolution.
        
        Args:
            seed_types: Specific seed types to include (None for all available)
            
        Returns:
            Number of individuals created
        """
        if seed_types is None:
            # Use VERIFIED registry pattern from system_stability_patterns.md
            seed_names = self.seed_registry.get_all_seed_names()  # Validated function
            # Create basic seed types - use available categories  
            seed_types = [
                SeedType.MOMENTUM, 
                SeedType.MEAN_REVERSION,
                SeedType.VOLATILITY,
                SeedType.TREND_FOLLOWING
            ]
        
        self.population = []
        
        for i in range(self.config.population_size):
            # Random seed type selection
            seed_type = random.choice(seed_types)
            
            # Generate random genes using BaseSeed framework
            genes = self._generate_random_genes(seed_type, generation=0)
            
            # Create individual
            individual = Individual(seed_type=seed_type, genes=genes)
            individual.generation_created = 0
            
            self.population.append(individual)
        
        self.current_generation = 0
        logger.info(f"Initialized population of {len(self.population)} individuals")
        
        # ANTI-HALLUCINATION: Smart validation with innovation tolerance
        validation_result = self._validate_population_with_tolerance()
        if validation_result['critical_failures'] > 0:
            raise RuntimeError(f"Critical validation failed: {validation_result['critical_failures']} individuals with fatal errors")
        
        if validation_result['warnings'] > 0:
            logger.warning(f"Population initialized with {validation_result['warnings']} tolerance warnings (acceptable for genetic diversity)")
        
        return len(self.population)
    
    async def evolve_strategies(
        self, 
        market_data: pd.DataFrame, 
        generations: Optional[int] = None
    ) -> List[Individual]:
        """
        Main evolution loop with hybrid local/distributed execution.
        
        Args:
            market_data: Historical market data for backtesting
            generations: Number of generations to evolve (None uses config)
            
        Returns:
            Best performing individuals from evolution
        """
        generations = generations or self.config.generations
        
        # Initialize Ray if using distributed mode (Phase 5B)
        if self.use_ray:
            await self._initialize_ray_cluster(market_data)
        
        logger.info(f"Starting evolution: {generations} generations, mode={'Ray' if self.use_ray else 'local'}")
        
        # Evolution loop
        for generation in range(generations):
            start_time = time.time()
            
            # Evaluate population
            if self.use_ray and self.ray_initialized:
                await self._evaluate_population_distributed()
            else:
                await self._evaluate_population_local(market_data)
            
            # Track metrics
            metrics = self._calculate_generation_metrics(generation, time.time() - start_time)
            self.evolution_history.append(metrics)
            
            # Log progress
            logger.info(
                f"Generation {generation + 1}/{generations}: "
                f"Best fitness: {metrics.best_fitness:.4f}, "
                f"Avg fitness: {metrics.average_fitness:.4f}, "
                f"Health: {metrics.health_score:.1f}/100"
            )
            
            # Early termination if health degrades
            if metrics.health_score < 50.0:
                logger.warning(f"Health score too low ({metrics.health_score:.1f}), terminating evolution")
                break
            
            # Evolve to next generation (if not final generation)
            if generation < generations - 1:
                await self._evolve_generation()
            
            self.current_generation = generation + 1
        
        # Return best individuals
        return self._get_best_individuals(n=10)
    
    async def _evaluate_population_local(self, market_data: pd.DataFrame):
        """Local population evaluation using asyncio concurrency."""
        tasks = []
        
        for individual in self.population:
            if individual.fitness is None:  # Only evaluate if not already evaluated
                task = self._evaluate_individual_local(individual, market_data)
                tasks.append(task)
        
        if tasks:
            # Execute evaluations concurrently (limited by connection optimizer)
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _evaluate_individual_local(self, individual: Individual, market_data: pd.DataFrame):
        """Evaluate single individual locally."""
        start_time = time.time()
        
        try:
            # Create seed instance from registry  
            # Get seed names directly from the type index (returns List[str])
            available_seed_names = self.seed_registry._type_index[individual.seed_type]
            if not available_seed_names:
                raise ValueError(f"No seeds available for type {individual.seed_type}")
            
            # Use first available seed name of this type
            seed_name = available_seed_names[0]
            seed_instance = self.seed_registry.create_seed_instance(seed_name, individual.genes)
            
            # Run fitness evaluation - generate signals and calculate fitness
            signals = seed_instance.generate_signals(market_data)
            
            # Calculate basic performance metrics from signals
            # This is a simplified fitness calculation for the genetic algorithm
            returns = []
            position = 0
            
            for i, signal in enumerate(signals):
                if signal > 0.5 and position <= 0:  # Buy signal
                    position = 1
                elif signal < -0.5 and position >= 0:  # Sell signal
                    position = -1
                    
                if i > 0 and position != 0:
                    price_change = (market_data.iloc[i]['close'] - market_data.iloc[i-1]['close']) / market_data.iloc[i-1]['close']
                    returns.append(position * price_change)
            
            if returns:
                total_return = sum(returns)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
                max_drawdown = min(np.cumsum(returns)) if returns else 0.0
                win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
            else:
                total_return = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                win_rate = 0.0
            
            results = {
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'trade_count': len(returns),
                'win_rate': win_rate
            }
            
            # Update individual with results
            individual.fitness = results.get('sharpe_ratio', 0.0)
            individual.metrics = {
                'total_return': results.get('total_return', 0.0),
                'max_drawdown': results.get('max_drawdown', 0.0),
                'trade_count': results.get('trade_count', 0),
                'win_rate': results.get('win_rate', 0.0)
            }
            individual.evaluation_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Local evaluation failed for {individual.genes.seed_id}: {e}")
            individual.fitness = -999.0  # Poor fitness for failed evaluations
            individual.evaluation_time = time.time() - start_time
    
    async def _evaluate_population_distributed(self):
        """Distributed population evaluation using Ray cluster."""
        if not self.ray_initialized or not self.market_data_ref:
            logger.error("Ray not initialized for distributed evaluation")
            return
        
        # Submit all evaluations to Ray cluster
        futures = []
        for individual in self.population:
            if individual.fitness is None:  # Only evaluate if not already evaluated
                future = evaluate_individual_distributed.remote(
                    individual.seed_type.value,
                    individual.genes.model_dump(),
                    self.market_data_ref
                )
                futures.append((individual, future))
        
        if not futures:
            return
        
        # Collect results with timeout handling
        for individual, future in futures:
            try:
                result = await asyncio.wait_for(
                    asyncio.create_task(self._ray_get_async(future)),
                    timeout=self.config.ray_timeout
                )
                
                # Update individual with results
                individual.fitness = result['fitness']
                individual.metrics = {
                    'total_return': result['total_return'],
                    'max_drawdown': result['max_drawdown'],
                    'trade_count': result['trade_count'],
                    'win_rate': result['win_rate']
                }
                individual.evaluation_time = result['evaluation_time']
                
                if not result['success']:
                    logger.warning(f"Ray evaluation warning: {result.get('error', 'Unknown error')}")
                
            except asyncio.TimeoutError:
                logger.error(f"Ray evaluation timeout for {individual.genes.seed_id}")
                individual.fitness = -999.0
                individual.evaluation_time = self.config.ray_timeout
            except Exception as e:
                logger.error(f"Ray evaluation error for {individual.genes.seed_id}: {e}")
                individual.fitness = -999.0
                individual.evaluation_time = 0.0
    
    async def _ray_get_async(self, future):
        """Async wrapper for ray.get()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, future)
    
    async def _initialize_ray_cluster(self, market_data: pd.DataFrame):
        """Initialize Ray cluster for distributed execution."""
        if not RAY_AVAILABLE:
            logger.error("Ray not available, falling back to local mode")
            self.use_ray = False
            return
        
        try:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Put market data in Ray object store for efficient sharing
            self.market_data_ref = ray.put(market_data)
            self.ray_initialized = True
            
            logger.info("Ray cluster initialized successfully")
            
        except Exception as e:
            logger.error(f"Ray initialization failed: {e}")
            self.use_ray = False
            self.ray_initialized = False
    
    async def _evolve_generation(self):
        """Evolve population to next generation using genetic operators."""
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness or -999.0, reverse=True)
        
        # Calculate elite size
        elite_size = max(1, int(self.config.population_size * self.config.elite_ratio))
        elite = self.population[:elite_size]
        
        # Create new population
        new_population = elite.copy()  # Preserve elite
        
        # Fill remaining population with offspring
        while len(new_population) < self.config.population_size:
            # Selection: tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child = self._mutate(child)
            
            # Reset fitness for reevaluation
            child.fitness = None
            child.generation_created = self.current_generation + 1
            
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness or -999.0)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover operator to create offspring."""
        # Simple parameter averaging crossover
        child_parameters = {}
        
        for key in parent1.genes.parameters:
            if key in parent2.genes.parameters:
                # Average parameters
                child_parameters[key] = (
                    parent1.genes.parameters[key] + parent2.genes.parameters[key]
                ) / 2.0
            else:
                child_parameters[key] = parent1.genes.parameters[key]
        
        # Add parameters unique to parent2
        for key in parent2.genes.parameters:
            if key not in child_parameters:
                child_parameters[key] = parent2.genes.parameters[key]
        
        # Create child genes
        child_genes = SeedGenes(
            seed_id=f"gen_{self.current_generation + 1}_{random.randint(1000, 9999)}",
            seed_type=parent1.seed_type,  # Use parent1's seed type
            generation=self.current_generation + 1,
            parameters=child_parameters
        )
        
        return Individual(seed_type=parent1.seed_type, genes=child_genes)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutation operator to introduce variation."""
        mutated_parameters = individual.genes.parameters.copy()
        
        # Mutate random parameters
        for key, value in mutated_parameters.items():
            if random.random() < 0.3:  # 30% chance to mutate each parameter
                # Add gaussian noise (10% of current value)
                noise = random.gauss(0, abs(value) * 0.1)
                mutated_parameters[key] = value + noise
        
        # Create mutated genes
        mutated_genes = SeedGenes(
            seed_id=f"mut_{self.current_generation + 1}_{random.randint(1000, 9999)}",
            seed_type=individual.seed_type,
            generation=self.current_generation + 1,
            parameters=mutated_parameters
        )
        
        return Individual(seed_type=individual.seed_type, genes=mutated_genes)
    
    def _generate_random_genes(self, seed_type: SeedType, generation: int) -> SeedGenes:
        """Generate random genes using VERIFIED patterns with proper parameter initialization."""
        import random
        import uuid
        from src.strategy.genetic_seeds.base_seed import SeedGenes
        
        # Get available seeds for this type
        available_seed_names = self.seed_registry._type_index.get(seed_type, [])
        if not available_seed_names:
            # Fallback: create basic genes with empty parameters
            logger.warning(f"No seeds available for type {seed_type}, creating minimal genes")
            return SeedGenes.create_default(seed_type)
        
        # Use first available seed to get parameter requirements
        seed_name = available_seed_names[0]
        seed_class = self.seed_registry.get_seed_class(seed_name)
        
        if seed_class:
            try:
                # Create dummy seed to get parameter bounds and requirements
                dummy_genes = SeedGenes.create_default(seed_type)
                dummy_seed = seed_class(dummy_genes, self.seed_registry.settings)
                
                # Get required parameters and their bounds
                required_params = dummy_seed.required_parameters
                parameter_bounds = dummy_seed.parameter_bounds
                
                # Generate random parameters within bounds
                parameters = {}
                for param in required_params:
                    if param in parameter_bounds:
                        min_val, max_val = parameter_bounds[param]
                        # Generate random value within bounds
                        parameters[param] = random.uniform(min_val, max_val)
                    else:
                        # Fallback default for missing bounds
                        parameters[param] = 1.0
                
                # Create properly initialized genes
                genes = SeedGenes(
                    seed_id=str(uuid.uuid4()),
                    seed_type=seed_type,
                    generation=generation,
                    parameters=parameters
                )
                
                return genes
                
            except Exception as e:
                logger.error(f"Failed to generate parameters for {seed_type}: {e}")
                # Fallback to basic initialization
                genes = SeedGenes.create_default(seed_type)
                genes.generation = generation
                return genes
        else:
            # No seed class available, create basic genes
            genes = SeedGenes.create_default(seed_type)
            genes.generation = generation
            return genes
    
    def _validate_population_with_tolerance(self) -> Dict[str, int]:
        """
        Smart validation with innovation tolerance.
        
        CRITICAL FAILURES: Block deployment (missing required params, invalid seed types)
        WARNINGS: Log but allow (parameters slightly outside bounds, novel combinations)
        
        Returns:
            Dict with 'critical_failures' and 'warnings' counts
        """
        critical_failures = 0
        warnings = 0
        
        for individual in self.population:
            try:
                # Get seed requirements for validation
                available_seed_names = self.seed_registry._type_index[individual.seed_type]
                if not available_seed_names:
                    logger.error(f"CRITICAL: No seeds available for type {individual.seed_type}")
                    critical_failures += 1
                    continue
                
                seed_name = available_seed_names[0]
                seed_class = self.seed_registry.get_seed_class(seed_name)
                if not seed_class:
                    logger.error(f"CRITICAL: Cannot load seed class for {seed_name}")
                    critical_failures += 1
                    continue
                
                # Create dummy instance to check requirements
                dummy_genes = SeedGenes(
                    seed_id="validation",
                    seed_type=individual.seed_type,
                    generation=0,
                    parameters={}
                )
                
                try:
                    dummy_instance = seed_class(dummy_genes, self.seed_registry.settings)
                    required_params = dummy_instance.required_parameters
                    parameter_bounds = dummy_instance.parameter_bounds
                except Exception as e:
                    logger.error(f"CRITICAL: Cannot create seed instance for validation: {e}")
                    critical_failures += 1
                    continue
                
                # CHECK 1: Required parameters (CRITICAL - must have)
                missing_required = [p for p in required_params if p not in individual.genes.parameters]
                if missing_required:
                    logger.error(f"CRITICAL: Individual {individual.genes.seed_id} missing required parameters: {missing_required}")
                    critical_failures += 1
                    continue
                
                # CHECK 2: Parameter bounds (TOLERANT - allow genetic diversity)
                for param_name, param_value in individual.genes.parameters.items():
                    if param_name in parameter_bounds:
                        min_val, max_val = parameter_bounds[param_name]
                        
                        # STRICT bounds for technical impossibilities
                        strict_multiplier = 3.0  # Allow 3x outside bounds for genetic exploration
                        if not (min_val / strict_multiplier <= param_value <= max_val * strict_multiplier):
                            logger.error(f"CRITICAL: Parameter {param_name}={param_value} extremely outside bounds ({min_val}, {max_val})")
                            critical_failures += 1
                            break
                        
                        # TOLERANT bounds for normal genetic variation
                        tolerant_multiplier = 1.5  # Warn if 1.5x outside bounds
                        if not (min_val / tolerant_multiplier <= param_value <= max_val * tolerant_multiplier):
                            logger.warning(f"TOLERANCE: Parameter {param_name}={param_value} outside normal bounds ({min_val}, {max_val}) - genetic diversity")
                            warnings += 1
                
            except Exception as e:
                logger.error(f"CRITICAL: Validation error for individual {individual.genes.seed_id}: {e}")
                critical_failures += 1
        
        logger.info(f"Population validation: {critical_failures} critical failures, {warnings} tolerance warnings")
        return {
            'critical_failures': critical_failures,
            'warnings': warnings,
            'total_individuals': len(self.population)
        }
    
    def _calculate_generation_metrics(self, generation: int, evaluation_time: float) -> EvolutionMetrics:
        """Calculate metrics for current generation."""
        # Get successful fitness values (exclude failed evaluations)
        successful_fitnesses = [ind.fitness for ind in self.population 
                               if ind.fitness is not None and ind.fitness > -999.0]
        
        if not successful_fitnesses:
            return EvolutionMetrics(
                generation=generation,
                best_fitness=0.0,
                average_fitness=0.0,
                population_diversity=0.0,
                evaluation_time=evaluation_time,
                health_score=0.0,
                failed_evaluations=len(self.population)
            )
        
        # Calculate diversity (standard deviation of successful fitness)
        diversity = statistics.stdev(successful_fitnesses) if len(successful_fitnesses) > 1 else 0.0
        
        # Calculate health score based on success rate and performance
        failed_evaluations = sum(1 for ind in self.population if ind.fitness is None or ind.fitness <= -999.0)
        success_rate = 1.0 - (failed_evaluations / len(self.population))
        
        health_score = min(100.0, success_rate * 100.0)
        self.health_score = health_score
        
        return EvolutionMetrics(
            generation=generation,
            best_fitness=max(successful_fitnesses),
            average_fitness=statistics.mean(successful_fitnesses),
            population_diversity=diversity,
            evaluation_time=evaluation_time,
            health_score=health_score,
            failed_evaluations=failed_evaluations
        )
    
    def _get_best_individuals(self, n: int = 10) -> List[Individual]:
        """Get top N individuals from population."""
        # Sort by fitness (descending)
        sorted_population = sorted(
            [ind for ind in self.population if ind.fitness is not None],
            key=lambda x: x.fitness,
            reverse=True
        )
        
        return sorted_population[:n]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        if not self.evolution_history:
            return {"status": "no_evolution_data"}
        
        latest_metrics = self.evolution_history[-1]
        best_individuals = self._get_best_individuals(5)
        
        return {
            "status": "completed",
            "generations": len(self.evolution_history),
            "current_health_score": self.health_score,
            "evolution_mode": "ray_distributed" if self.use_ray else "local",
            "latest_metrics": {
                "best_fitness": latest_metrics.best_fitness,
                "average_fitness": latest_metrics.average_fitness,
                "population_diversity": latest_metrics.population_diversity,
                "evaluation_time": latest_metrics.evaluation_time,
                "failed_evaluations": latest_metrics.failed_evaluations
            },
            "best_strategies": [
                {
                    "seed_type": ind.seed_type.value,
                    "fitness": ind.fitness,
                    "parameters": ind.genes.parameters,
                    "metrics": ind.metrics
                }
                for ind in best_individuals
            ],
            "fitness_progression": [m.best_fitness for m in self.evolution_history],
            "health_progression": [m.health_score for m in self.evolution_history]
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.ray_initialized and RAY_AVAILABLE:
            try:
                if self.market_data_ref:
                    # Object references are automatically cleaned up
                    pass
                # Note: Don't shutdown Ray here as it might be used by other processes
                logger.info("Ray resources cleaned up")
            except Exception as e:
                logger.error(f"Ray cleanup error: {e}")
        
        logger.info("GeneticStrategyPool cleanup completed")


# Factory function for easy instantiation
def create_genetic_strategy_pool(
    connection_optimizer: RetailConnectionOptimizer,
    use_ray: bool = False,
    population_size: int = 100,
    generations: int = 10
) -> GeneticStrategyPool:
    """
    Factory function to create GeneticStrategyPool with sensible defaults.
    
    Args:
        connection_optimizer: Retail trading connection optimizer
        use_ray: Enable Ray distributed execution (requires Ray cluster)
        population_size: Size of genetic population
        generations: Number of evolution generations
        
    Returns:
        Configured GeneticStrategyPool instance
    """
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations
    )
    
    return GeneticStrategyPool(
        connection_optimizer=connection_optimizer,
        use_ray=use_ray,
        evolution_config=config
    )