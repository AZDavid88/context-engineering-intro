"""
Cloud Genetic Algorithm Coordinator - Distributed Evolution with Neon State Management

This module coordinates distributed genetic algorithm evolution across Ray workers
using Neon PostgreSQL + TimescaleDB for centralized state management and evolution
tracking. Integrates with existing genetic algorithm framework for seamless scaling.

Research-Based Implementation:
- /research/ray_cluster/research_summary.md - Ray distributed patterns
- /verified_docs/by_module_simplified/strategy/ - Existing GA framework
- /src/execution/genetic_strategy_pool.py - Current GA execution patterns

Key Features:
- Distributed evolution coordination across multiple Ray workers
- Centralized state management using Neon TimescaleDB
- Integration with existing BaseSeed and genetic algorithm framework
- Fault-tolerant evolution with state recovery
- Performance monitoring and optimization
- Ray-compatible stateless evaluation functions
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Ray imports (conditional for graceful degradation)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Create mock ray for type hints when not available
    class MockRay:
        @staticmethod
        def remote(func):
            # Return the function as-is when Ray not available
            return func
        
        @staticmethod
        def is_initialized():
            return False
            
        @staticmethod
        def init(**kwargs):
            pass
            
        @staticmethod
        def get(task):
            return task
    
    ray = MockRay()

from src.data.neon_hybrid_storage import NeonHybridStorage
from src.data.neon_schema_manager import NeonSchemaManager
from src.strategy.genetic_seeds.seed_registry import SeedRegistry, get_registry
from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedType, SeedGenes
from src.config.settings import get_settings


# Set up logging
logger = logging.getLogger(__name__)


class EvolutionStatus(str, Enum):
    """Evolution status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvolutionConfiguration:
    """Configuration for distributed genetic algorithm evolution."""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.2
    selection_method: str = "tournament"
    tournament_size: int = 3
    fitness_objectives: List[str] = field(default_factory=lambda: ["sharpe_ratio", "total_return"])
    convergence_threshold: float = 0.001
    stagnation_limit: int = 10  # Generations without improvement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for storage."""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_rate": self.elitism_rate,
            "selection_method": self.selection_method,
            "tournament_size": self.tournament_size,
            "fitness_objectives": self.fitness_objectives,
            "convergence_threshold": self.convergence_threshold,
            "stagnation_limit": self.stagnation_limit
        }


@dataclass
class GenerationResult:
    """Results from a single generation of evolution."""
    generation: int
    worker_id: str
    population: List[Dict[str, Any]]
    fitness_scores: List[float]
    best_individual: Dict[str, Any]
    generation_stats: Dict[str, Any]
    execution_time_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for storage."""
        return {
            "generation": self.generation,
            "worker_id": self.worker_id,
            "population": self.population,
            "fitness_scores": self.fitness_scores,
            "best_individual": self.best_individual,
            "generation_stats": self.generation_stats,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EvolutionMetrics:
    """Evolution performance and progress metrics."""
    evolution_id: str
    total_generations_completed: int = 0
    best_fitness_achieved: float = 0.0
    average_fitness_trend: List[float] = field(default_factory=list)
    generations_without_improvement: int = 0
    total_execution_time_seconds: float = 0.0
    worker_performance: Dict[str, float] = field(default_factory=dict)
    convergence_rate: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CloudGeneticAlgorithmCoordinator:
    """
    Coordinates distributed genetic algorithm evolution across Ray workers.
    
    Manages evolution state using Neon PostgreSQL + TimescaleDB for centralized
    coordination, fault tolerance, and performance monitoring across distributed workers.
    """
    
    def __init__(self, 
                 hybrid_storage: NeonHybridStorage,
                 evolution_config: Optional[EvolutionConfiguration] = None):
        """
        Initialize cloud GA coordinator.
        
        Args:
            hybrid_storage: NeonHybridStorage instance for data and state management
            evolution_config: Evolution configuration parameters
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed GA coordination. Install with: pip install ray")
        
        self.storage = hybrid_storage
        self.config = evolution_config or EvolutionConfiguration()
        self.evolution_id = str(uuid.uuid4())
        
        # State management
        self.schema_manager: Optional[NeonSchemaManager] = None
        self.current_generation = 0
        self.evolution_status = EvolutionStatus.INITIALIZING
        self.metrics = EvolutionMetrics(evolution_id=self.evolution_id)
        
        # Ray worker management
        self.active_workers: List[str] = []
        self.worker_tasks: Dict[str, Any] = {}
        
        # Genetic algorithm components
        self.seed_registry = get_registry()
        self.population: List[BaseSeed] = []
        self.fitness_history: List[List[float]] = []
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.CloudGA[{self.evolution_id[:8]}]")
        
    async def initialize_evolution(self) -> str:
        """
        Initialize distributed evolution with state management.
        
        Returns:
            Evolution ID for tracking and coordination
        """
        try:
            self.logger.info(f"Initializing cloud genetic algorithm evolution: {self.evolution_id}")
            
            # Ensure Neon storage is available
            if not self.storage.neon_available:
                raise RuntimeError("Neon database required for distributed GA coordination")
            
            self.schema_manager = self.storage.schema_manager
            if not self.schema_manager:
                raise RuntimeError("Schema manager not available")
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
                self.logger.info("Initialized Ray cluster for distributed evolution")
            
            # Create initial population
            await self._create_initial_population()
            
            # Store initial evolution state
            await self._store_evolution_state(
                generation=0,
                status=EvolutionStatus.INITIALIZING,
                population_data=self._serialize_population(),
                evolution_parameters=self.config.to_dict()
            )
            
            self.evolution_status = EvolutionStatus.ACTIVE
            
            self.logger.info(
                f"Evolution initialized: {len(self.population)} individuals, "
                f"{self.config.max_generations} max generations"
            )
            
            return self.evolution_id
            
        except Exception as e:
            self.evolution_status = EvolutionStatus.FAILED
            self.logger.error(f"Evolution initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize evolution: {e}") from e
    
    async def execute_distributed_evolution(self, 
                                          num_workers: int = 4,
                                          checkpoint_interval: int = 5) -> Dict[str, Any]:
        """
        Execute distributed genetic algorithm evolution across Ray workers.
        
        Args:
            num_workers: Number of Ray workers to utilize
            checkpoint_interval: Generations between state checkpoints
            
        Returns:
            Complete evolution results and statistics
        """
        self.logger.info(f"Starting distributed evolution with {num_workers} workers")
        evolution_start_time = time.time()
        
        try:
            # Initialize worker IDs
            self.active_workers = [f"worker_{i}" for i in range(num_workers)]
            
            results = {
                "evolution_id": self.evolution_id,
                "configuration": self.config.to_dict(),
                "generations_completed": 0,
                "best_individuals": [],
                "convergence_data": [],
                "worker_performance": {},
                "total_execution_time": 0.0
            }
            
            # Main evolution loop
            for generation in range(1, self.config.max_generations + 1):
                self.current_generation = generation
                
                self.logger.info(f"Executing generation {generation}/{self.config.max_generations}")
                generation_start_time = time.time()
                
                # Distribute generation work across Ray workers
                generation_results = await self._execute_generation_distributed(generation)
                
                # Aggregate results and select best individuals
                aggregated_results = await self._aggregate_generation_results(generation_results)
                
                # Update population with best individuals
                await self._update_population(aggregated_results)
                
                # Update metrics and check convergence
                generation_time = time.time() - generation_start_time
                convergence_achieved = await self._update_metrics_and_check_convergence(
                    aggregated_results, generation_time
                )
                
                # Store checkpoint
                if generation % checkpoint_interval == 0:
                    await self._store_evolution_checkpoint(generation, aggregated_results)
                
                # Update results
                results["generations_completed"] = generation
                results["best_individuals"].append(aggregated_results["best_individual"])
                results["convergence_data"].append({
                    "generation": generation,
                    "best_fitness": aggregated_results["best_fitness"],
                    "average_fitness": aggregated_results["average_fitness"],
                    "diversity_score": aggregated_results["diversity_score"]
                })
                
                self.logger.info(
                    f"Generation {generation} complete: "
                    f"best_fitness={aggregated_results['best_fitness']:.4f}, "
                    f"avg_fitness={aggregated_results['average_fitness']:.4f}, "
                    f"time={generation_time:.2f}s"
                )
                
                # Check for convergence or early termination
                if convergence_achieved:
                    self.logger.info(f"Evolution converged at generation {generation}")
                    break
                
                if self.metrics.generations_without_improvement >= self.config.stagnation_limit:
                    self.logger.info(f"Evolution stagnated at generation {generation}")
                    break
            
            # Final results
            self.evolution_status = EvolutionStatus.COMPLETED
            total_time = time.time() - evolution_start_time
            
            results["total_execution_time"] = total_time
            results["worker_performance"] = dict(self.metrics.worker_performance)
            results["final_metrics"] = {
                "best_fitness_achieved": self.metrics.best_fitness_achieved,
                "convergence_rate": self.metrics.convergence_rate,
                "generations_without_improvement": self.metrics.generations_without_improvement
            }
            
            # Store final evolution state
            await self._store_evolution_state(
                generation=self.current_generation,
                status=EvolutionStatus.COMPLETED,
                population_data=self._serialize_population(),
                fitness_metrics=results["final_metrics"],
                performance_metrics={"total_execution_time": total_time}
            )
            
            self.logger.info(
                f"Distributed evolution completed: {results['generations_completed']} generations "
                f"in {total_time:.1f}s, best_fitness={self.metrics.best_fitness_achieved:.4f}"
            )
            
            return results
            
        except Exception as e:
            self.evolution_status = EvolutionStatus.FAILED
            self.logger.error(f"Distributed evolution failed: {e}")
            
            # Store failure state
            await self._store_evolution_state(
                generation=self.current_generation,
                status=EvolutionStatus.FAILED,
                population_data=self._serialize_population(),
                performance_metrics={"error": str(e)}
            )
            
            raise RuntimeError(f"Distributed evolution failed: {e}") from e
    
    async def _create_initial_population(self) -> None:
        """Create initial population of genetic algorithm individuals."""
        available_seeds = list(self.seed_registry.get_all_seeds().keys())
        if not available_seeds:
            raise RuntimeError("No genetic seeds available for evolution")
        
        self.population = []
        
        for i in range(self.config.population_size):
            # Select random seed type
            seed_type = np.random.choice(available_seeds)
            
            # Create seed with random parameters
            seed_class = self.seed_registry.get_seed(seed_type)
            seed_instance = seed_class()
            
            # Randomize parameters within bounds
            seed_instance.randomize_parameters()
            
            self.population.append(seed_instance)
        
        self.logger.info(f"Created initial population of {len(self.population)} individuals")
    
    async def _execute_generation_distributed(self, generation: int) -> List[GenerationResult]:
        """Execute generation across distributed Ray workers."""
        # Divide population among workers
        population_chunks = self._divide_population(self.population, len(self.active_workers))
        
        # Create Ray tasks for each worker
        tasks = []
        for i, (worker_id, population_chunk) in enumerate(zip(self.active_workers, population_chunks)):
            task = self._evolve_generation_on_worker.remote(
                worker_id=worker_id,
                generation=generation,
                population_chunk=population_chunk,
                evolution_config=self.config.to_dict(),
                evolution_id=self.evolution_id
            )
            tasks.append(task)
        
        # Wait for all workers to complete
        try:
            results = await asyncio.gather(*[self._ray_task_to_async(task) for task in tasks])
            return results
        except Exception as e:
            self.logger.error(f"Worker execution failed in generation {generation}: {e}")
            raise
    
    @ray.remote
    @staticmethod
    def _evolve_generation_on_worker(worker_id: str,
                                   generation: int,
                                   population_chunk: List[Dict[str, Any]],
                                   evolution_config: Dict[str, Any],
                                   evolution_id: str) -> Dict[str, Any]:
        """
        Execute generation evolution on a specific Ray worker.
        
        This is a stateless Ray remote function that processes a population chunk.
        """
        import time
        import logging
        from src.backtesting.vectorbt_engine import VectorBTEngine
        from src.data.storage_interfaces import get_storage_implementation
        
        # Set up worker logging
        worker_logger = logging.getLogger(f"CloudGA.Worker[{worker_id}]")
        execution_start = time.time()
        
        try:
            # Get storage implementation (will use same backend as coordinator)
            storage = get_storage_implementation()
            
            # Initialize backtesting engine
            backtesting_engine = VectorBTEngine(storage)
            
            # Evaluate fitness for each individual in chunk
            fitness_scores = []
            evaluated_individuals = []
            
            for individual_data in population_chunk:
                # Reconstruct seed instance from data
                seed_instance = CloudGeneticAlgorithmCoordinator._reconstruct_seed(individual_data)
                
                # Evaluate fitness using backtesting engine
                try:
                    fitness_result = backtesting_engine.evaluate_seed_fitness(seed_instance)
                    fitness_score = fitness_result.composite_score
                except Exception as e:
                    worker_logger.warning(f"Fitness evaluation failed for individual: {e}")
                    fitness_score = 0.0
                
                fitness_scores.append(fitness_score)
                evaluated_individuals.append(individual_data)
            
            # Apply genetic operations (mutation, crossover)
            evolved_individuals = CloudGeneticAlgorithmCoordinator._apply_genetic_operations(
                evaluated_individuals, fitness_scores, evolution_config
            )
            
            # Find best individual from this chunk
            best_idx = np.argmax(fitness_scores)
            best_individual = {
                "individual_data": evolved_individuals[best_idx],
                "fitness": fitness_scores[best_idx]
            }
            
            execution_time = time.time() - execution_start
            
            return {
                "generation": generation,
                "worker_id": worker_id,
                "population": evolved_individuals,
                "fitness_scores": fitness_scores,
                "best_individual": best_individual,
                "generation_stats": {
                    "chunk_size": len(population_chunk),
                    "avg_fitness": np.mean(fitness_scores),
                    "max_fitness": np.max(fitness_scores),
                    "min_fitness": np.min(fitness_scores),
                    "fitness_std": np.std(fitness_scores)
                },
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            worker_logger.error(f"Worker {worker_id} failed in generation {generation}: {e}")
            return {
                "generation": generation,
                "worker_id": worker_id,
                "population": [],
                "fitness_scores": [],
                "best_individual": None,
                "generation_stats": {"error": str(e)},
                "execution_time_seconds": time.time() - execution_start,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @staticmethod
    def _reconstruct_seed(individual_data: Dict[str, Any]) -> BaseSeed:
        """Reconstruct seed instance from serialized data."""
        from src.strategy.genetic_seeds.seed_registry import get_registry
        
        registry = get_registry()
        seed_type = individual_data.get("seed_type")
        
        if not seed_type:
            raise ValueError("Seed type not found in individual data")
        
        seed_class = registry.get_seed(seed_type)
        seed_instance = seed_class()
        
        # Restore parameters
        if "parameters" in individual_data:
            seed_instance.genes.update(individual_data["parameters"])
        
        return seed_instance
    
    @staticmethod
    def _apply_genetic_operations(individuals: List[Dict[str, Any]], 
                                fitness_scores: List[float],
                                config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply genetic operations (selection, crossover, mutation) to population chunk."""
        mutation_rate = config.get("mutation_rate", 0.1)
        crossover_rate = config.get("crossover_rate", 0.8)
        
        evolved = []
        
        for individual in individuals:
            # Apply mutation
            if np.random.random() < mutation_rate:
                individual = CloudGeneticAlgorithmCoordinator._mutate_individual(individual)
            
            # Apply crossover (simplified - would need partner selection in full implementation)
            if np.random.random() < crossover_rate and len(individuals) > 1:
                partner = np.random.choice(individuals)
                individual = CloudGeneticAlgorithmCoordinator._crossover_individuals(individual, partner)
            
            evolved.append(individual)
        
        return evolved
    
    @staticmethod
    def _mutate_individual(individual: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to individual parameters."""
        if "parameters" not in individual:
            return individual
        
        mutated = individual.copy()
        parameters = mutated["parameters"].copy()
        
        # Mutate random parameter
        param_keys = list(parameters.keys())
        if param_keys:
            param_to_mutate = np.random.choice(param_keys)
            current_value = parameters[param_to_mutate]
            
            if isinstance(current_value, (int, float)):
                # Add gaussian noise
                mutation_strength = 0.1
                noise = np.random.normal(0, mutation_strength * abs(current_value))
                parameters[param_to_mutate] = current_value + noise
        
        mutated["parameters"] = parameters
        return mutated
    
    @staticmethod
    def _crossover_individuals(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Apply crossover between two individuals."""
        if "parameters" not in parent1 or "parameters" not in parent2:
            return parent1
        
        child = parent1.copy()
        child_params = {}
        
        # Uniform crossover
        for key in parent1["parameters"]:
            if key in parent2["parameters"]:
                if np.random.random() < 0.5:
                    child_params[key] = parent1["parameters"][key]
                else:
                    child_params[key] = parent2["parameters"][key]
            else:
                child_params[key] = parent1["parameters"][key]
        
        child["parameters"] = child_params
        return child
    
    async def _ray_task_to_async(self, ray_task) -> Any:
        """Convert Ray task to async-compatible result."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ray_task)
    
    def _divide_population(self, population: List[BaseSeed], num_chunks: int) -> List[List[Dict[str, Any]]]:
        """Divide population into chunks for distribution to workers."""
        chunks = [[] for _ in range(num_chunks)]
        
        for i, individual in enumerate(population):
            chunk_idx = i % num_chunks
            chunks[chunk_idx].append(self._serialize_individual(individual))
        
        return chunks
    
    def _serialize_individual(self, individual: BaseSeed) -> Dict[str, Any]:
        """Serialize individual for Ray worker processing."""
        return {
            "seed_type": individual.seed_type.value,
            "parameters": dict(individual.genes),
            "fitness": getattr(individual, 'fitness', 0.0),
            "metadata": getattr(individual, 'metadata', {})
        }
    
    def _serialize_population(self) -> Dict[str, Any]:
        """Serialize current population for storage."""
        return {
            "population_size": len(self.population),
            "individuals": [self._serialize_individual(ind) for ind in self.population],
            "generation": self.current_generation,
            "serialization_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _aggregate_generation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all workers for a generation."""
        if not results:
            raise ValueError("No generation results to aggregate")
        
        # Combine all populations and fitness scores
        all_individuals = []
        all_fitness_scores = []
        
        for result in results:
            all_individuals.extend(result.get("population", []))
            all_fitness_scores.extend(result.get("fitness_scores", []))
        
        if not all_fitness_scores:
            raise ValueError("No fitness scores in generation results")
        
        # Find overall best individual
        best_idx = np.argmax(all_fitness_scores)
        best_fitness = all_fitness_scores[best_idx]
        best_individual = all_individuals[best_idx]
        
        # Calculate generation statistics
        avg_fitness = np.mean(all_fitness_scores)
        fitness_std = np.std(all_fitness_scores)
        
        # Calculate diversity score (simplified)
        diversity_score = fitness_std / max(abs(avg_fitness), 1e-6)
        
        return {
            "generation": results[0].get("generation"),
            "all_individuals": all_individuals,
            "all_fitness_scores": all_fitness_scores,
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "average_fitness": avg_fitness,
            "fitness_std": fitness_std,
            "diversity_score": diversity_score,
            "worker_results": results
        }
    
    async def _update_population(self, aggregated_results: Dict[str, Any]) -> None:
        """Update population with best individuals from generation."""
        all_individuals = aggregated_results["all_individuals"]
        all_fitness_scores = aggregated_results["all_fitness_scores"]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(all_fitness_scores)[::-1]
        
        # Select top individuals for next generation
        elite_count = int(self.config.elitism_rate * self.config.population_size)
        selected_individuals = []
        
        # Keep elite individuals
        for i in range(min(elite_count, len(sorted_indices))):
            idx = sorted_indices[i]
            selected_individuals.append(all_individuals[idx])
        
        # Fill remaining population with tournament selection or random selection
        while len(selected_individuals) < self.config.population_size:
            if len(all_individuals) > 0:
                idx = np.random.randint(len(all_individuals))
                selected_individuals.append(all_individuals[idx])
        
        # Reconstruct population from selected individuals
        self.population = []
        for individual_data in selected_individuals[:self.config.population_size]:
            try:
                seed_instance = self._reconstruct_seed(individual_data)
                seed_instance.fitness = individual_data.get("fitness", 0.0)
                self.population.append(seed_instance)
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct individual: {e}")
        
        self.logger.debug(f"Updated population: {len(self.population)} individuals")
    
    async def _update_metrics_and_check_convergence(self, 
                                                   aggregated_results: Dict[str, Any],
                                                   generation_time: float) -> bool:
        """Update evolution metrics and check for convergence."""
        current_best = aggregated_results["best_fitness"]
        current_avg = aggregated_results["average_fitness"]
        
        # Update metrics
        self.metrics.total_generations_completed += 1
        self.metrics.total_execution_time_seconds += generation_time
        self.metrics.average_fitness_trend.append(current_avg)
        
        # Check for improvement
        if current_best > self.metrics.best_fitness_achieved:
            self.metrics.best_fitness_achieved = current_best
            self.metrics.generations_without_improvement = 0
        else:
            self.metrics.generations_without_improvement += 1
        
        # Update worker performance metrics
        for result in aggregated_results.get("worker_results", []):
            worker_id = result.get("worker_id")
            execution_time = result.get("execution_time_seconds", 0)
            if worker_id:
                if worker_id not in self.metrics.worker_performance:
                    self.metrics.worker_performance[worker_id] = 0
                self.metrics.worker_performance[worker_id] = (
                    self.metrics.worker_performance[worker_id] + execution_time
                ) / 2
        
        # Calculate convergence rate
        if len(self.metrics.average_fitness_trend) > 1:
            recent_trend = self.metrics.average_fitness_trend[-5:]  # Last 5 generations
            if len(recent_trend) > 1:
                self.metrics.convergence_rate = abs(recent_trend[-1] - recent_trend[0])
        
        # Check convergence
        convergence_achieved = (
            self.metrics.convergence_rate < self.config.convergence_threshold and
            len(self.metrics.average_fitness_trend) >= 5
        )
        
        self.metrics.last_updated = datetime.now(timezone.utc)
        
        return convergence_achieved
    
    async def _store_evolution_state(self,
                                   generation: int,
                                   status: EvolutionStatus,
                                   population_data: Dict[str, Any],
                                   fitness_metrics: Optional[Dict[str, Any]] = None,
                                   performance_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Store evolution state in Neon database."""
        if not self.schema_manager:
            self.logger.warning("Schema manager not available - cannot store evolution state")
            return
        
        try:
            await self.schema_manager.create_evolution_state_entry(
                evolution_id=self.evolution_id,
                phase=4,  # Phase 4 implementation
                generation=generation,
                population_data=population_data,
                worker_id=None,  # Coordinator entry
                evolution_parameters={
                    "config": self.config.to_dict(),
                    "status": status.value,
                    "fitness_metrics": fitness_metrics or {},
                    "performance_metrics": performance_metrics or {}
                }
            )
            
            self.logger.debug(f"Stored evolution state for generation {generation}")
            
        except Exception as e:
            self.logger.error(f"Failed to store evolution state: {e}")
    
    async def _store_evolution_checkpoint(self, generation: int, results: Dict[str, Any]) -> None:
        """Store evolution checkpoint for recovery."""
        try:
            await self.schema_manager.update_evolution_state(
                evolution_id=self.evolution_id,
                generation=generation,
                status=EvolutionStatus.ACTIVE.value,
                fitness_metrics={
                    "best_fitness": results["best_fitness"],
                    "average_fitness": results["average_fitness"],
                    "diversity_score": results["diversity_score"]
                },
                performance_metrics={
                    "generations_completed": self.metrics.total_generations_completed,
                    "execution_time": self.metrics.total_execution_time_seconds,
                    "convergence_rate": self.metrics.convergence_rate
                }
            )
            
            self.logger.info(f"Stored checkpoint at generation {generation}")
            
        except Exception as e:
            self.logger.error(f"Failed to store checkpoint: {e}")
    
    async def recover_evolution(self, evolution_id: str) -> bool:
        """
        Recover evolution from stored state.
        
        Args:
            evolution_id: Evolution ID to recover
            
        Returns:
            True if recovery successful
        """
        try:
            if not self.schema_manager:
                raise RuntimeError("Schema manager not available")
            
            # Get evolution state from database
            evolution_states = await self.schema_manager.get_evolution_state(evolution_id)
            
            if not evolution_states:
                raise ValueError(f"No evolution state found for {evolution_id}")
            
            # Get latest state
            latest_state = evolution_states[0]
            
            # Restore evolution parameters
            evolution_params = latest_state.get("evolution_parameters", {})
            if "config" in evolution_params:
                self.config = EvolutionConfiguration(**evolution_params["config"])
            
            # Restore population
            population_data = latest_state.get("population_data", {})
            if "individuals" in population_data:
                self.population = []
                for individual_data in population_data["individuals"]:
                    try:
                        seed_instance = self._reconstruct_seed(individual_data)
                        self.population.append(seed_instance)
                    except Exception as e:
                        self.logger.warning(f"Failed to restore individual: {e}")
            
            # Restore generation and status
            self.current_generation = latest_state.get("generation", 0)
            status_str = evolution_params.get("status", EvolutionStatus.ACTIVE.value)
            self.evolution_status = EvolutionStatus(status_str)
            self.evolution_id = evolution_id
            
            self.logger.info(
                f"Recovered evolution {evolution_id}: generation {self.current_generation}, "
                f"status {self.evolution_status}, population {len(self.population)}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Evolution recovery failed: {e}")
            return False
    
    async def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and metrics."""
        return {
            "evolution_id": self.evolution_id,
            "status": self.evolution_status.value,
            "current_generation": self.current_generation,
            "max_generations": self.config.max_generations,
            "population_size": len(self.population),
            "active_workers": len(self.active_workers),
            "metrics": {
                "generations_completed": self.metrics.total_generations_completed,
                "best_fitness_achieved": self.metrics.best_fitness_achieved,
                "generations_without_improvement": self.metrics.generations_without_improvement,
                "convergence_rate": self.metrics.convergence_rate,
                "total_execution_time_seconds": self.metrics.total_execution_time_seconds,
                "worker_performance": dict(self.metrics.worker_performance)
            },
            "configuration": self.config.to_dict(),
            "last_updated": self.metrics.last_updated.isoformat()
        }