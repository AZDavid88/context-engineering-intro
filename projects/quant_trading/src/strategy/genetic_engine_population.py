"""
Genetic Engine Population - Advanced population management and optimization.
Handles population initialization, multi-timeframe evaluation, and walk-forward optimization.
"""

import logging
import multiprocessing
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna
from src.strategy.genetic_seeds import SeedRegistry, get_registry, BaseSeed
from src.strategy.genetic_seeds.base_seed import SeedType

# Import for multi-timeframe data support
try:
    from src.data.dynamic_asset_data_collector import AssetDataSet
except ImportError:
    AssetDataSet = None

# Configure logging
logger = logging.getLogger(__name__)


class PopulationManager:
    """Advanced population management for genetic algorithm optimization."""
    
    def __init__(self, seed_registry: Optional[SeedRegistry] = None,
                 enable_multiprocessing: bool = True,
                 max_processes: Optional[int] = None):
        """Initialize population manager.
        
        Args:
            seed_registry: Registry of available genetic seeds
            enable_multiprocessing: Whether to enable parallel processing
            max_processes: Maximum number of processes (None for CPU count)
        """
        self.seed_registry = seed_registry or get_registry()
        self.enable_multiprocessing = enable_multiprocessing
        self.max_processes = max_processes or multiprocessing.cpu_count()
        
        # Population tracking
        self._population_history = []
        self._diversity_history = []
        
        # Multi-timeframe evaluation
        self.multi_timeframe_data = None
        self.timeframe_weights = {'1d': 0.5, '4h': 0.3, '1h': 0.2}
        
        logger.info(f"Population manager initialized with {self.max_processes} processes")
    
    def initialize_population(self, population_size: int) -> List[BaseSeed]:
        """Initialize a diverse population of trading strategies."""
        try:
            population = []
            available_seeds = list(self.seed_registry.get_all_seeds().keys())
            
            if not available_seeds:
                logger.warning("No genetic seeds available")
                return self._create_fallback_population(population_size)
            
            # Ensure diversity by distributing across seed types
            seeds_per_type = max(1, population_size // len(available_seeds))
            remaining_slots = population_size
            
            for seed_type in available_seeds:
                if remaining_slots <= 0:
                    break
                
                slots_for_this_type = min(seeds_per_type, remaining_slots)
                
                for _ in range(slots_for_this_type):
                    try:
                        seed_class = self.seed_registry.get_seed(seed_type)
                        if seed_class:
                            individual = seed_class()
                            
                            # Apply random mutations for diversity
                            if hasattr(individual, 'mutate'):
                                individual.mutate(rate=0.3)
                            
                            population.append(individual)
                            remaining_slots -= 1
                        
                    except Exception as e:
                        logger.error(f"Failed to create seed of type {seed_type}: {e}")
                        continue
            
            # Fill remaining slots with random seeds
            while len(population) < population_size and available_seeds:
                seed_type = np.random.choice(available_seeds)
                try:
                    seed_class = self.seed_registry.get_seed(seed_type)
                    if seed_class:
                        individual = seed_class()
                        if hasattr(individual, 'mutate'):
                            individual.mutate(rate=0.5)  # Higher mutation for diversity
                        population.append(individual)
                except Exception as e:
                    logger.error(f"Failed to create additional seed: {e}")
                    break
            
            logger.info(f"Initialized population of {len(population)} individuals")
            self._population_history.append(len(population))
            
            return population
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return self._create_fallback_population(population_size)
    
    def _create_fallback_population(self, size: int) -> List[BaseSeed]:
        """Create fallback population when seed registry is unavailable."""
        logger.warning("Creating fallback population")
        
        class FallbackSeed(BaseSeed):
            def __init__(self):
                super().__init__()
                self.seed_type = SeedType.TECHNICAL
                # Add minimal functionality
        
        return [FallbackSeed() for _ in range(size)]
    
    def initialize_large_population(self, population_size: int) -> List[BaseSeed]:
        """Initialize large population with advanced diversity techniques."""
        try:
            if population_size <= 100:
                return self.initialize_population(population_size)
            
            logger.info(f"Initializing large population of {population_size} individuals")
            
            # Use multiple strategies for large populations
            population = []
            
            # Strategy 1: Base diverse population (40%)
            base_size = int(population_size * 0.4)
            base_population = self.initialize_population(base_size)
            population.extend(base_population)
            
            # Strategy 2: Mutated variants of successful seeds (30%)
            if base_population:
                variant_size = int(population_size * 0.3)
                for _ in range(variant_size):
                    parent = np.random.choice(base_population)
                    try:
                        variant = type(parent)()
                        if hasattr(parent, 'genes'):
                            variant.genes = parent.genes.copy() if hasattr(parent.genes, 'copy') else parent.genes
                        if hasattr(variant, 'mutate'):
                            variant.mutate(rate=0.7)  # High mutation for variants
                        population.append(variant)
                    except Exception as e:
                        logger.error(f"Failed to create variant: {e}")
                        continue
            
            # Strategy 3: Random exploration (30%)
            remaining_size = population_size - len(population)
            if remaining_size > 0:
                exploration_pop = self.initialize_population(remaining_size)
                population.extend(exploration_pop)
            
            logger.info(f"Large population initialized: {len(population)} individuals")
            return population[:population_size]  # Ensure exact size
            
        except Exception as e:
            logger.error(f"Error initializing large population: {e}")
            return self.initialize_population(min(population_size, 50))  # Fallback to smaller size
    
    def calculate_population_diversity(self, population: List[BaseSeed]) -> Dict[str, float]:
        """Calculate population diversity metrics."""
        try:
            if not population:
                return {'genetic_diversity': 0.0, 'type_diversity': 0.0, 'fitness_diversity': 0.0}
            
            # Type diversity (Shannon entropy of seed types)
            type_counts = {}
            for individual in population:
                seed_type = getattr(individual, 'seed_type', 'unknown')
                type_counts[seed_type] = type_counts.get(seed_type, 0) + 1
            
            type_diversity = self._calculate_shannon_entropy(list(type_counts.values()))
            
            # Fitness diversity (coefficient of variation)
            fitness_values = []
            for individual in population:
                if hasattr(individual, 'fitness') and hasattr(individual.fitness, 'sharpe_ratio'):
                    fitness_values.append(individual.fitness.sharpe_ratio)
            
            if fitness_values:
                fitness_mean = np.mean(fitness_values)
                fitness_std = np.std(fitness_values)
                fitness_diversity = fitness_std / fitness_mean if fitness_mean != 0 else 0.0
            else:
                fitness_diversity = 0.0
            
            # Genetic diversity (simplified - based on parameter variation)
            genetic_diversity = self._calculate_genetic_diversity(population)
            
            diversity_metrics = {
                'genetic_diversity': float(genetic_diversity),
                'type_diversity': float(type_diversity),
                'fitness_diversity': float(fitness_diversity)
            }
            
            self._diversity_history.append(diversity_metrics)
            
            return diversity_metrics
            
        except Exception as e:
            logger.error(f"Error calculating population diversity: {e}")
            return {'genetic_diversity': 0.0, 'type_diversity': 0.0, 'fitness_diversity': 0.0}
    
    def _calculate_shannon_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy for diversity measurement."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        entropy = 0.0
        
        for count in counts:
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_genetic_diversity(self, population: List[BaseSeed]) -> float:
        """Calculate genetic diversity based on strategy parameters."""
        try:
            if len(population) < 2:
                return 0.0
            
            # Extract parameter vectors from population
            parameter_vectors = []
            for individual in population:
                if hasattr(individual, 'genes') and hasattr(individual.genes, 'to_dict'):
                    params = individual.genes.to_dict()
                    # Convert to numeric vector
                    numeric_params = []
                    for value in params.values():
                        if isinstance(value, (int, float)):
                            numeric_params.append(float(value))
                        elif isinstance(value, bool):
                            numeric_params.append(float(value))
                    
                    if numeric_params:
                        parameter_vectors.append(numeric_params)
            
            if len(parameter_vectors) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(parameter_vectors)):
                for j in range(i + 1, len(parameter_vectors)):
                    vec1 = np.array(parameter_vectors[i])
                    vec2 = np.array(parameter_vectors[j])
                    
                    if len(vec1) == len(vec2):
                        distance = np.linalg.norm(vec1 - vec2)
                        distances.append(distance)
            
            # Average distance as diversity measure
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating genetic diversity: {e}")
            return 0.0
    
    def setup_multi_timeframe_evaluation(self, timeframe_data: Dict[str, pd.DataFrame]) -> None:
        """Setup multi-timeframe evaluation context."""
        try:
            self.multi_timeframe_data = timeframe_data
            
            # Validate timeframe data
            valid_timeframes = {}
            for timeframe, data in timeframe_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    valid_timeframes[timeframe] = data
                    logger.debug(f"Validated timeframe {timeframe}: {len(data)} periods")
                else:
                    logger.warning(f"Invalid data for timeframe {timeframe}")
            
            self.multi_timeframe_data = valid_timeframes
            
            # Adjust weights based on available timeframes
            available_timeframes = set(valid_timeframes.keys())
            default_weights = {'1d': 0.5, '4h': 0.3, '1h': 0.2}
            
            self.timeframe_weights = {}
            total_weight = 0.0
            
            for tf, weight in default_weights.items():
                if tf in available_timeframes:
                    self.timeframe_weights[tf] = weight
                    total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for tf in self.timeframe_weights:
                    self.timeframe_weights[tf] /= total_weight
            
            logger.info(f"Multi-timeframe evaluation setup: {list(valid_timeframes.keys())}")
            
        except Exception as e:
            logger.error(f"Error setting up multi-timeframe evaluation: {e}")
            self.multi_timeframe_data = None
    
    def evaluate_multi_timeframe_fitness(self, individual: BaseSeed, 
                                       fitness_evaluator) -> Tuple[float, float, float, float]:
        """Evaluate individual fitness across multiple timeframes."""
        try:
            if not self.multi_timeframe_data:
                logger.warning("No multi-timeframe data available, using single timeframe")
                return fitness_evaluator.evaluate_individual(individual)
            
            weighted_metrics = {'sharpe': 0.0, 'consistency': 0.0, 'drawdown': 0.0, 'win_rate': 0.0}
            
            for timeframe, data in self.multi_timeframe_data.items():
                weight = self.timeframe_weights.get(timeframe, 0.0)
                if weight == 0.0:
                    continue
                
                # Evaluate on this timeframe
                sharpe, consistency, drawdown, win_rate = fitness_evaluator.evaluate_individual(individual, data)
                
                # Apply weight
                weighted_metrics['sharpe'] += weight * sharpe
                weighted_metrics['consistency'] += weight * consistency
                weighted_metrics['drawdown'] += weight * drawdown
                weighted_metrics['win_rate'] += weight * win_rate
            
            return (
                weighted_metrics['sharpe'],
                weighted_metrics['consistency'],
                weighted_metrics['drawdown'],
                weighted_metrics['win_rate']
            )
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe evaluation: {e}")
            return fitness_evaluator.evaluate_individual(individual)  # Fallback to single timeframe
    
    def run_walk_forward_evolution(self, population: List[BaseSeed], 
                                 fitness_evaluator, n_windows: int = 5,
                                 window_size: int = 252) -> List[BaseSeed]:
        """Run walk-forward optimization to prevent overfitting."""
        try:
            if not population:
                return population
            
            logger.info(f"Starting walk-forward evolution: {n_windows} windows")
            
            # Generate synthetic data for walk-forward if no real data available
            full_data = fitness_evaluator.generate_synthetic_market_data(
                periods=window_size * (n_windows + 1)
            )
            
            walk_forward_results = []
            
            for window_idx in range(n_windows):
                start_idx = window_idx * window_size
                end_idx = start_idx + window_size
                
                if end_idx >= len(full_data):
                    break
                
                window_data = full_data.iloc[start_idx:end_idx]
                
                logger.debug(f"Walk-forward window {window_idx + 1}: "
                           f"{window_data.index[0]} to {window_data.index[-1]}")
                
                # Evaluate population on this window
                window_fitness = []
                for individual in population:
                    fitness = fitness_evaluator.evaluate_individual(individual, window_data)
                    window_fitness.append((individual, fitness))
                
                # Sort by combined fitness score
                window_fitness.sort(key=lambda x: sum(x[1]), reverse=True)
                walk_forward_results.append([ind for ind, _ in window_fitness])
            
            if not walk_forward_results:
                return population
            
            # Combine results across windows (ensemble approach)
            final_population = self._combine_walk_forward_results(walk_forward_results)
            
            logger.info(f"Walk-forward evolution completed: {len(final_population)} individuals")
            return final_population
            
        except Exception as e:
            logger.error(f"Error in walk-forward evolution: {e}")
            return population  # Return original population on error
    
    def _combine_walk_forward_results(self, walk_forward_results: List[List[BaseSeed]]) -> List[BaseSeed]:
        """Combine walk-forward results using ensemble ranking."""
        try:
            if not walk_forward_results:
                return []
            
            # Count appearances of each individual across windows
            individual_scores = {}
            
            for window_results in walk_forward_results:
                for rank, individual in enumerate(window_results):
                    ind_key = id(individual)  # Use object ID as key
                    
                    if ind_key not in individual_scores:
                        individual_scores[ind_key] = {'individual': individual, 'score': 0.0, 'appearances': 0}
                    
                    # Higher score for better ranks (inverse rank)
                    rank_score = (len(window_results) - rank) / len(window_results)
                    individual_scores[ind_key]['score'] += rank_score
                    individual_scores[ind_key]['appearances'] += 1
            
            # Sort by average score across appearances
            scored_individuals = []
            for ind_data in individual_scores.values():
                avg_score = ind_data['score'] / ind_data['appearances']
                scored_individuals.append((ind_data['individual'], avg_score, ind_data['appearances']))
            
            # Sort by average score, then by number of appearances
            scored_individuals.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            # Return sorted individuals
            return [ind for ind, _, _ in scored_individuals]
            
        except Exception as e:
            logger.error(f"Error combining walk-forward results: {e}")
            return walk_forward_results[0] if walk_forward_results else []
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population management statistics."""
        return {
            'population_history': self._population_history.copy(),
            'diversity_history': self._diversity_history.copy(),
            'multi_timeframe_enabled': self.multi_timeframe_data is not None,
            'available_timeframes': list(self.timeframe_weights.keys()) if self.multi_timeframe_data else [],
            'timeframe_weights': self.timeframe_weights.copy(),
            'multiprocessing_enabled': self.enable_multiprocessing,
            'max_processes': self.max_processes
        }
    
    def clear_history(self) -> None:
        """Clear population history for memory management."""
        self._population_history.clear()
        self._diversity_history.clear()
        logger.info("Population history cleared")
    
    def multiprocessing_map(self, func: Callable, iterable) -> List[Any]:
        """Execute function over iterable using multiprocessing if enabled."""
        if not self.enable_multiprocessing or len(iterable) < 2:
            return [func(item) for item in iterable]
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_processes) as executor:
                results = list(executor.map(func, iterable))
            return results
            
        except Exception as e:
            logger.error(f"Multiprocessing failed, falling back to sequential: {e}")
            return [func(item) for item in iterable]