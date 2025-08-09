"""
Config-Driven Strategy Loader - Genetic Algorithm Integration

This module provides JSON-based strategy management for genetic algorithm
evolved trading strategies, enabling seamless serialization, storage,
and instantiation of evolved strategy configurations.

Integration Points:
- SeedGenes (Pydantic model) for JSON serialization  
- SeedRegistry for strategy instantiation
- BaseSeed interface for execution compatibility
- GeneticStrategyPool for distributed execution
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone
import uuid

from pydantic import BaseModel, Field
import pandas as pd

from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes, SeedType, SeedFitness
from src.strategy.genetic_seeds.seed_registry import SeedRegistry, get_registry
from src.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


class StrategyConfig(BaseModel):
    """Strategy configuration model for JSON serialization."""
    
    name: str = Field(..., description="Strategy display name")
    seed_type: SeedType = Field(..., description="Type of genetic seed")
    genes: Dict = Field(..., description="Serialized SeedGenes parameters")
    
    # Performance metrics
    fitness_score: float = Field(default=0.0, description="Genetic algorithm fitness")
    backtest_sharpe: float = Field(default=0.0, description="Backtesting Sharpe ratio")
    paper_trading_days: int = Field(default=0, description="Days in paper trading")
    paper_sharpe: float = Field(default=0.0, description="Paper trading Sharpe ratio")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown observed")
    
    # Metadata
    generation: int = Field(default=0, description="Genetic evolution generation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = Field(default=True, description="Whether strategy is active")
    
    # Deployment status
    deployment_status: str = Field(default="evolved", description="Current deployment status")
    deployment_history: List[str] = Field(default_factory=list, description="Deployment history")


class ConfigStrategyLoader:
    """Manage strategies through JSON configuration files."""
    
    def __init__(self, 
                 config_dir: Union[str, Path] = "evolved_strategies",
                 archive_dir: Union[str, Path] = None,
                 settings: Optional[Settings] = None):
        
        self.settings = settings or get_settings()
        self.config_dir = Path(config_dir)
        
        # Set archive directory relative to config directory if not specified
        if archive_dir is None:
            self.archive_dir = self.config_dir / "archive"
        else:
            self.archive_dir = Path(archive_dir)
        
        self.registry = get_registry()
        
        # Create directories if they don't exist (with parents for nested paths)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConfigStrategyLoader initialized - configs: {self.config_dir}")
    
    def save_evolved_strategies(self, 
                               evolution_results: List[SeedGenes],
                               fitness_scores: Optional[List[float]] = None) -> List[str]:
        """
        Save genetic algorithm evolution results as JSON configurations.
        
        Args:
            evolution_results: List of evolved SeedGenes from genetic algorithm
            fitness_scores: Optional fitness scores for each strategy
            
        Returns:
            List of saved configuration file paths
        """
        saved_configs = []
        
        for i, genes in enumerate(evolution_results):
            
            # Generate strategy name
            strategy_name = f"evolved_{genes.seed_type.value}_{i:03d}_{int(datetime.now().timestamp())}"
            
            # Create strategy config
            config = StrategyConfig(
                name=strategy_name,
                seed_type=genes.seed_type,
                genes={
                    'seed_id': genes.seed_id,
                    'seed_type': genes.seed_type.value,
                    'generation': genes.generation,
                    'parameters': genes.parameters,
                    'fast_period': genes.fast_period,
                    'slow_period': genes.slow_period,
                    'signal_period': genes.signal_period,
                    'entry_threshold': genes.entry_threshold,
                    'exit_threshold': genes.exit_threshold,
                    'filter_threshold': genes.filter_threshold,
                    'stop_loss': genes.stop_loss,
                    'take_profit': genes.take_profit,
                    'position_size': genes.position_size
                },
                fitness_score=fitness_scores[i] if fitness_scores else 0.0,
                generation=genes.generation
            )
            
            # Save to JSON file
            config_file = self.config_dir / f"{strategy_name}.json"
            config_file.write_text(config.model_dump_json(indent=2))
            saved_configs.append(str(config_file))
            
            logger.info(f"Saved strategy config: {config_file}")
        
        logger.info(f"Saved {len(saved_configs)} evolved strategies to {self.config_dir}")
        return saved_configs
    
    def load_strategies(self, 
                       min_fitness: float = 0.5,
                       max_strategies: Optional[int] = None,
                       active_only: bool = True) -> List[BaseSeed]:
        """
        Load strategies from JSON configurations, filtering by fitness.
        
        Args:
            min_fitness: Minimum fitness score threshold
            max_strategies: Maximum number of strategies to load
            active_only: Only load active strategies
            
        Returns:
            List of instantiated BaseSeed strategy objects
        """
        strategies = []
        config_files = list(self.config_dir.glob("*.json"))
        
        # Load and filter configurations
        configs_with_scores = []
        for config_file in config_files:
            try:
                config_data = json.loads(config_file.read_text())
                config = StrategyConfig(**config_data)
                
                # Apply filters
                if active_only and not config.is_active:
                    continue
                if config.fitness_score < min_fitness:
                    continue
                    
                configs_with_scores.append((config, config.fitness_score))
                
            except Exception as e:
                logger.warning(f"Failed to load config {config_file}: {e}")
                continue
        
        # Sort by fitness score (descending)
        configs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of strategies
        if max_strategies:
            configs_with_scores = configs_with_scores[:max_strategies]
        
        # Create strategy instances
        for config, _ in configs_with_scores:
            try:
                strategy = self.create_strategy_from_config(config)
                strategies.append(strategy)
                
            except Exception as e:
                logger.error(f"Failed to instantiate strategy {config.name}: {e}")
                continue
        
        logger.info(f"Loaded {len(strategies)} strategies from {len(config_files)} configs")
        return strategies
    
    def create_strategy_from_config(self, config: StrategyConfig) -> BaseSeed:
        """
        Create strategy instance from configuration.
        
        Args:
            config: StrategyConfig object
            
        Returns:
            Instantiated BaseSeed strategy object
        """
        # Reconstruct SeedGenes from config
        genes = SeedGenes(
            seed_id=config.genes['seed_id'],
            seed_type=SeedType(config.genes['seed_type']),
            generation=config.genes['generation'],
            parameters=config.genes['parameters'],
            fast_period=config.genes['fast_period'],
            slow_period=config.genes['slow_period'],
            signal_period=config.genes['signal_period'],
            entry_threshold=config.genes['entry_threshold'],
            exit_threshold=config.genes['exit_threshold'],
            filter_threshold=config.genes['filter_threshold'],
            stop_loss=config.genes['stop_loss'],
            take_profit=config.genes['take_profit'],
            position_size=config.genes['position_size']
        )
        
        # Create strategy using existing registry
        # Get available seed names for this type from registry type index
        available_seed_names = self.registry._type_index.get(config.seed_type, [])
        if not available_seed_names:
            raise ValueError(f"No seeds available for type {config.seed_type}")
        
        # Use first available seed name of this type
        seed_name = available_seed_names[0]
        strategy = self.registry.create_seed_instance(seed_name, genes)
        
        if strategy is None:
            raise ValueError(f"Failed to create strategy instance for {seed_name} with genes {genes.seed_id}")
        
        # Attach config metadata
        strategy._config_name = config.name
        strategy._fitness_score = config.fitness_score
        strategy._deployment_status = config.deployment_status
        
        return strategy
    
    def update_strategy_performance(self, 
                                  strategy_name: str,
                                  performance_metrics: Dict[str, float]) -> bool:
        """
        Update strategy configuration with performance metrics.
        
        Args:
            strategy_name: Name of strategy to update
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            True if update successful, False otherwise
        """
        config_file = self.config_dir / f"{strategy_name}.json"
        
        if not config_file.exists():
            logger.warning(f"Strategy config not found: {strategy_name}")
            return False
        
        try:
            # Load existing config
            config_data = json.loads(config_file.read_text())
            config = StrategyConfig(**config_data)
            
            # Update performance metrics
            config.paper_trading_days = performance_metrics.get('paper_trading_days', config.paper_trading_days)
            config.paper_sharpe = performance_metrics.get('paper_sharpe', config.paper_sharpe)  
            config.max_drawdown = performance_metrics.get('max_drawdown', config.max_drawdown)
            config.last_updated = datetime.now(timezone.utc)
            
            # Save updated config
            config_file.write_text(config.model_dump_json(indent=2))
            
            logger.info(f"Updated performance metrics for {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy {strategy_name}: {e}")
            return False
    
    def archive_strategy(self, strategy_name: str, reason: str = "retired") -> bool:
        """
        Archive a strategy configuration.
        
        Args:
            strategy_name: Name of strategy to archive
            reason: Reason for archiving
            
        Returns:
            True if archival successful, False otherwise
        """
        config_file = self.config_dir / f"{strategy_name}.json"
        
        if not config_file.exists():
            logger.warning(f"Strategy config not found: {strategy_name}")
            return False
        
        try:
            # Move to archive directory  
            archive_file = self.archive_dir / f"{strategy_name}_{reason}_{int(datetime.now().timestamp())}.json"
            config_file.rename(archive_file)
            
            logger.info(f"Archived strategy {strategy_name} to {archive_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive strategy {strategy_name}: {e}")
            return False
    
    def get_strategy_summary(self) -> Dict[str, any]:
        """Get summary statistics of managed strategies."""
        
        config_files = list(self.config_dir.glob("*.json"))
        active_count = 0
        total_fitness = 0.0
        strategy_types = {}
        
        for config_file in config_files:
            try:
                config_data = json.loads(config_file.read_text())
                config = StrategyConfig(**config_data)
                
                if config.is_active:
                    active_count += 1
                    total_fitness += config.fitness_score
                    
                seed_type = config.seed_type.value
                strategy_types[seed_type] = strategy_types.get(seed_type, 0) + 1
                
            except Exception:
                continue
        
        return {
            'total_strategies': len(config_files),
            'active_strategies': active_count,
            'average_fitness': total_fitness / max(active_count, 1),
            'strategy_types': strategy_types,
            'config_directory': str(self.config_dir),
            'archive_directory': str(self.archive_dir)
        }


# Factory function for easy integration
def get_config_loader(config_dir: str = "evolved_strategies") -> ConfigStrategyLoader:
    """Factory function to get ConfigStrategyLoader instance."""
    return ConfigStrategyLoader(config_dir=config_dir)