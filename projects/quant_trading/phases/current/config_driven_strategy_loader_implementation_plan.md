# Config-Driven Strategy Loader Implementation Plan

**Date**: 2025-08-08  
**Phase**: Production Enhancement - Strategy Automation  
**Priority**: HIGH - Foundation for Automated Trading Pipeline  
**Timeline**: 1 Week  
**Dependencies**: Existing genetic seeds framework, Pydantic models, seed registry

## Executive Summary

**Objective**: Implement a config-driven strategy loader that bridges genetic algorithm evolution output with automated strategy deployment, enabling seamless JSON-based strategy management and eliminating the need for dynamic Python code generation.

**Key Benefits**:
- **Zero Code Bloat**: Reuse existing `SeedGenes` Pydantic models for JSON serialization
- **Seamless Integration**: Works with existing `SeedRegistry` and `BaseSeed` framework
- **High Performance**: JSON loading faster and safer than dynamic imports
- **Scalable Management**: Handle 500+ evolved strategies as configuration files
- **Version Control Friendly**: Track strategy evolution through JSON diffs
- **Production Ready**: Enterprise-grade error handling and validation

**Architecture Integration**: **PERFECT ALIGNMENT** ⭐⭐⭐⭐⭐
- Leverages existing `SeedGenes` Pydantic model (already JSON serializable)
- Integrates with proven `SeedRegistry` for strategy instantiation  
- Maintains full compatibility with `BaseSeed` interface
- Works seamlessly with Ray cluster distributed execution

---

## Technical Architecture & Integration Points

### Current System Integration Points
```python
# EXISTING COMPONENTS (Already Implemented):
src/strategy/genetic_seeds/base_seed.py     # SeedGenes model (Pydantic)
src/strategy/genetic_seeds/seed_registry.py # SeedRegistry for instantiation
src/strategy/genetic_seeds/__init__.py      # BaseSeed interface
src/execution/genetic_strategy_pool.py      # Ray cluster integration
```

### New Component Architecture  
```python
# NEW COMPONENTS (To Be Implemented):
src/strategy/config_strategy_loader.py      # Core config loader (~150 lines)
src/integration/phase_coordinator.py       # ENHANCEMENT: Pipeline orchestration (~100 lines)
evolved_strategies/                         # JSON config directory
├── evolved_momentum_001.json              # Individual strategy configs  
├── evolved_breakout_002.json
├── evolved_ml_classifier_003.json
└── archive/                               # Retired strategy configs
```

### Enhanced Data Flow Integration
```
                    ┌─── EXISTING IntegratedPipelineOrchestrator ───┐
                    │                                                │
Genetic Evolution → SeedGenes → JSON Configs → ConfigStrategyLoader → BaseSeed → Paper Trading
      ↓                ↓              ↓                 ↓               ↓            ↓
   (existing)    (existing)     (new)           (new)        (existing)   (existing)
      ↓                                          ↓                                   ↓
PhaseCoordinator ←─── ENHANCED: Centralized Orchestration ───→ Performance Monitor
   (enhanced)                        (enhanced)                      (enhanced)
```

**ENHANCEMENT INTEGRATION**: The PhaseCoordinator enhances the existing IntegratedPipelineOrchestrator by providing centralized control over the three implementation phases, while leveraging existing infrastructure.

---

## Implementation Specification

### Core Component: ConfigStrategyLoader

**File**: `src/strategy/config_strategy_loader.py` (150 lines)

```python
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
                 archive_dir: Union[str, Path] = "evolved_strategies/archive",
                 settings: Optional[Settings] = None):
        
        self.settings = settings or get_settings()
        self.config_dir = Path(config_dir)
        self.archive_dir = Path(archive_dir)
        self.registry = get_registry()
        
        # Create directories if they don't exist
        self.config_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
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
        strategy = self.registry.create_seed(config.seed_type, genes)
        
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
```

### ENHANCEMENT: PhaseCoordinator Integration

**File**: `src/integration/phase_coordinator.py` (100 lines)

```python
"""
Phase Coordinator - Centralized Pipeline Orchestration Enhancement

This module enhances the existing IntegratedPipelineOrchestrator by providing
centralized coordination across the three implementation phases:
- Phase 1: ConfigStrategyLoader
- Phase 2: AutomatedDecisionEngine  
- Phase 3: UltraCompressedEvolution

Integration Points:
- Leverages existing IntegratedPipelineOrchestrator infrastructure
- Integrates with existing AsyncResourceManager for cleanup
- Uses existing SessionHealth monitoring
- Works with existing GeneticRiskManager for error recovery
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from src.data.dynamic_asset_data_collector import IntegratedPipelineOrchestrator
from src.execution.trading_system_manager import AsyncResourceManager, SessionHealth, SessionStatus
from src.strategy.config_strategy_loader import ConfigStrategyLoader
from src.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


class PhaseStatus(str, Enum):
    """Phase execution status tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseResult:
    """Result from individual phase execution."""
    phase_name: str
    status: PhaseStatus
    execution_time_seconds: float
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    strategies_processed: int = 0


class PhaseCoordinator:
    """
    Centralized coordinator for multi-phase pipeline execution.
    
    Enhances existing IntegratedPipelineOrchestrator with phase-level
    control and monitoring across ConfigLoader, DecisionEngine, and Evolution phases.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        
        # Integration with existing infrastructure
        self.pipeline_orchestrator = IntegratedPipelineOrchestrator(self.settings)
        self.resource_manager = AsyncResourceManager("PhaseCoordinator", logger)
        
        # Phase tracking
        self.phase_results: Dict[str, PhaseResult] = {}
        self.current_phase: Optional[str] = None
        
        # Components (initialized on demand)
        self._config_loader: Optional[ConfigStrategyLoader] = None
        self._decision_engine = None  # Will be imported from Phase 2
        self._evolution_system = None  # Will be imported from Phase 3
        
        logger.info("PhaseCoordinator initialized - enhancing existing pipeline orchestration")
    
    @property
    def config_loader(self) -> ConfigStrategyLoader:
        """Lazy initialization of ConfigStrategyLoader."""
        if self._config_loader is None:
            self._config_loader = ConfigStrategyLoader()
            self.resource_manager.register_resource(
                self._config_loader,
                lambda: self._config_loader.cleanup() if hasattr(self._config_loader, 'cleanup') else None
            )
        return self._config_loader
    
    async def execute_coordinated_phases(self,
                                       phases_to_run: List[str] = None,
                                       phase_config: Dict[str, Any] = None) -> Dict[str, PhaseResult]:
        """
        Execute coordinated multi-phase pipeline with centralized control.
        
        Args:
            phases_to_run: List of phases to execute ["config_loader", "decision_engine", "ultra_evolution"]
            phase_config: Configuration for each phase
            
        Returns:
            Dictionary of phase results
        """
        phases_to_run = phases_to_run or ["config_loader", "decision_engine", "ultra_evolution"]
        phase_config = phase_config or {}
        
        logger.info(f"🎯 Starting coordinated execution of phases: {phases_to_run}")
        
        try:
            # Phase 1: Config Strategy Loader
            if "config_loader" in phases_to_run:
                await self._execute_config_loader_phase(phase_config.get("config_loader", {}))
            
            # Phase 2: Decision Engine (when implemented)
            if "decision_engine" in phases_to_run:
                await self._execute_decision_engine_phase(phase_config.get("decision_engine", {}))
            
            # Phase 3: Ultra Evolution (when implemented)
            if "ultra_evolution" in phases_to_run:
                await self._execute_ultra_evolution_phase(phase_config.get("ultra_evolution", {}))
            
            # Generate coordination summary
            return await self._generate_coordination_summary()
            
        except Exception as e:
            logger.error(f"❌ Coordinated phase execution failed: {e}")
            await self._handle_coordination_failure(e)
            raise
        finally:
            await self.resource_manager.cleanup_all()
    
    async def _execute_config_loader_phase(self, config: Dict[str, Any]) -> PhaseResult:
        """Execute ConfigStrategyLoader phase with coordination."""
        
        self.current_phase = "config_loader"
        start_time = asyncio.get_event_loop().time()
        
        logger.info("📁 Executing Phase 1: Config Strategy Loader")
        
        try:
            # Integration with existing pipeline
            if config.get("use_existing_strategies", True):
                # Load from existing evolution results via IntegratedPipelineOrchestrator
                pipeline_results = await self.pipeline_orchestrator.execute_full_pipeline()
                evolved_strategies = pipeline_results.get("evolved_strategies", [])
            else:
                # Use provided strategies
                evolved_strategies = config.get("strategies", [])
            
            # Save evolved strategies as configs
            saved_configs = []
            if evolved_strategies:
                fitness_scores = [getattr(s, 'fitness', 0.0) for s in evolved_strategies]
                saved_configs = self.config_loader.save_evolved_strategies(evolved_strategies, fitness_scores)
            
            # Load top strategies for next phase
            min_fitness = config.get("min_fitness", 0.5)
            max_strategies = config.get("max_strategies", 50)
            
            loaded_strategies = self.config_loader.load_strategies(
                min_fitness=min_fitness,
                max_strategies=max_strategies
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            result = PhaseResult(
                phase_name="config_loader",
                status=PhaseStatus.COMPLETED,
                execution_time_seconds=execution_time,
                output_data={
                    "evolved_strategies_count": len(evolved_strategies),
                    "saved_configs_count": len(saved_configs),
                    "loaded_strategies_count": len(loaded_strategies),
                    "loaded_strategies": loaded_strategies,
                    "min_fitness_threshold": min_fitness
                },
                strategies_processed=len(evolved_strategies)
            )
            
            self.phase_results["config_loader"] = result
            logger.info(f"✅ Config Loader Phase completed: {len(loaded_strategies)} strategies ready")
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            result = PhaseResult(
                phase_name="config_loader",
                status=PhaseStatus.FAILED,
                execution_time_seconds=execution_time,
                output_data={},
                error_message=str(e),
                strategies_processed=0
            )
            
            self.phase_results["config_loader"] = result
            logger.error(f"❌ Config Loader Phase failed: {e}")
            raise
    
    async def _execute_decision_engine_phase(self, config: Dict[str, Any]) -> PhaseResult:
        """Execute DecisionEngine phase (placeholder for Phase 2 integration)."""
        
        self.current_phase = "decision_engine"
        start_time = asyncio.get_event_loop().time()
        
        logger.info("🤖 Executing Phase 2: Decision Engine (placeholder)")
        
        # Get strategies from previous phase
        config_result = self.phase_results.get("config_loader")
        if not config_result or config_result.status != PhaseStatus.COMPLETED:
            raise ValueError("Config Loader phase must complete successfully before Decision Engine")
        
        strategies = config_result.output_data.get("loaded_strategies", [])
        
        # Placeholder for DecisionEngine integration
        execution_time = asyncio.get_event_loop().time() - start_time
        
        result = PhaseResult(
            phase_name="decision_engine", 
            status=PhaseStatus.COMPLETED,
            execution_time_seconds=execution_time,
            output_data={
                "input_strategies": len(strategies),
                "decisions_made": 0,  # Placeholder
                "approved_strategies": strategies  # Pass through for now
            },
            strategies_processed=len(strategies)
        )
        
        self.phase_results["decision_engine"] = result
        logger.info(f"✅ Decision Engine Phase completed (placeholder)")
        
        return result
    
    async def _execute_ultra_evolution_phase(self, config: Dict[str, Any]) -> PhaseResult:
        """Execute UltraEvolution phase (placeholder for Phase 3 integration)."""
        
        self.current_phase = "ultra_evolution"
        start_time = asyncio.get_event_loop().time()
        
        logger.info("🚀 Executing Phase 3: Ultra Evolution (placeholder)")
        
        # Placeholder for UltraEvolution integration
        execution_time = asyncio.get_event_loop().time() - start_time
        
        result = PhaseResult(
            phase_name="ultra_evolution",
            status=PhaseStatus.COMPLETED,
            execution_time_seconds=execution_time,
            output_data={
                "deployed_strategies": 0  # Placeholder
            },
            strategies_processed=0
        )
        
        self.phase_results["ultra_evolution"] = result
        logger.info(f"✅ Ultra Evolution Phase completed (placeholder)")
        
        return result
    
    async def _generate_coordination_summary(self) -> Dict[str, PhaseResult]:
        """Generate comprehensive coordination summary."""
        
        total_execution_time = sum(r.execution_time_seconds for r in self.phase_results.values())
        total_strategies = sum(r.strategies_processed for r in self.phase_results.values())
        
        completed_phases = [name for name, result in self.phase_results.items() 
                          if result.status == PhaseStatus.COMPLETED]
        failed_phases = [name for name, result in self.phase_results.items() 
                       if result.status == PhaseStatus.FAILED]
        
        logger.info(f"📊 Coordination Summary:")
        logger.info(f"   • Completed Phases: {len(completed_phases)}/{len(self.phase_results)}")
        logger.info(f"   • Total Execution Time: {total_execution_time:.2f}s")
        logger.info(f"   • Total Strategies Processed: {total_strategies}")
        
        if failed_phases:
            logger.warning(f"   • Failed Phases: {failed_phases}")
        
        return self.phase_results.copy()
    
    async def _handle_coordination_failure(self, error: Exception):
        """Handle coordination-level failures."""
        
        logger.error(f"🚨 Phase coordination failure in {self.current_phase}: {error}")
        
        # Integration with existing error handling
        if hasattr(self.pipeline_orchestrator, 'handle_pipeline_error'):
            await self.pipeline_orchestrator.handle_pipeline_error(error)
    
    def get_phase_health(self) -> Dict[str, SessionHealth]:
        """Get health status for all phases."""
        
        health_status = {}
        
        for phase_name, result in self.phase_results.items():
            if result.status == PhaseStatus.COMPLETED:
                status = SessionStatus.CONNECTED
            elif result.status == PhaseStatus.IN_PROGRESS:
                status = SessionStatus.CONNECTING
            elif result.status == PhaseStatus.FAILED:
                status = SessionStatus.ERROR
            else:
                status = SessionStatus.DISCONNECTED
            
            health_status[phase_name] = SessionHealth(
                component_name=f"Phase_{phase_name}",
                status=status,
                error_count=1 if result.status == PhaseStatus.FAILED else 0,
                last_error=result.error_message
            )
        
        return health_status


# Factory function for integration
async def get_phase_coordinator() -> PhaseCoordinator:
    """Factory function to get PhaseCoordinator instance."""
    return PhaseCoordinator()
```

---

## Integration Testing Framework

### Test Suite: `tests/integration/test_config_strategy_loader.py`

```python
"""
Integration tests for ConfigStrategyLoader with existing genetic framework.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone

from src.strategy.config_strategy_loader import ConfigStrategyLoader, StrategyConfig
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType
from src.strategy.genetic_seeds.ema_crossover_seed import MomentumMACDSeed


class TestConfigStrategyLoader:
    """Test ConfigStrategyLoader integration with existing framework."""
    
    def test_save_and_load_evolved_strategies(self):
        """Test round-trip: SeedGenes → JSON → BaseSeed"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigStrategyLoader(config_dir=temp_dir)
            
            # Create sample evolved strategies (simulate GA output)
            evolved_genes = [
                SeedGenes.create_default(SeedType.MOMENTUM, "test_momentum_001"),
                SeedGenes.create_default(SeedType.BREAKOUT, "test_breakout_002")
            ]
            
            # Simulate fitness scores
            fitness_scores = [1.25, 1.08]
            
            # Save strategies as configs
            saved_configs = loader.save_evolved_strategies(evolved_genes, fitness_scores)
            assert len(saved_configs) == 2
            
            # Load strategies back
            loaded_strategies = loader.load_strategies(min_fitness=1.0)
            assert len(loaded_strategies) == 2
            
            # Verify strategy types match
            loaded_types = [s.genes.seed_type for s in loaded_strategies]
            assert SeedType.MOMENTUM in loaded_types
            assert SeedType.BREAKOUT in loaded_types
    
    def test_integration_with_existing_seed_registry(self):
        """Test integration with existing SeedRegistry."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigStrategyLoader(config_dir=temp_dir)
            
            # Create momentum strategy config
            momentum_genes = SeedGenes.create_default(SeedType.MOMENTUM, "test_momentum")
            loader.save_evolved_strategies([momentum_genes], [1.5])
            
            # Load and verify it creates proper BaseSeed instance
            strategies = loader.load_strategies()
            assert len(strategies) == 1
            
            strategy = strategies[0]
            assert hasattr(strategy, 'generate_signals')  # BaseSeed interface
            assert hasattr(strategy, 'genes')  # SeedGenes attached
            assert strategy.genes.seed_type == SeedType.MOMENTUM
    
    def test_performance_metrics_update(self):
        """Test performance metrics updating."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ConfigStrategyLoader(config_dir=temp_dir)
            
            # Create and save strategy
            genes = SeedGenes.create_default(SeedType.MOMENTUM, "perf_test")
            loader.save_evolved_strategies([genes])
            
            # Update performance metrics
            strategy_name = f"evolved_momentum_000_{int(datetime.now().timestamp())}"
            performance_metrics = {
                'paper_trading_days': 7,
                'paper_sharpe': 1.35,
                'max_drawdown': 0.08
            }
            
            success = loader.update_strategy_performance(strategy_name, performance_metrics)
            assert success
            
            # Verify update persisted
            config_file = Path(temp_dir) / f"{strategy_name}.json"
            config_data = json.loads(config_file.read_text())
            assert config_data['paper_sharpe'] == 1.35
            assert config_data['paper_trading_days'] == 7


# Performance benchmarks
def test_config_loader_performance():
    """Test performance with 100+ strategy configs."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        loader = ConfigStrategyLoader(config_dir=temp_dir)
        
        # Generate 100 strategies
        evolved_genes = [
            SeedGenes.create_default(SeedType.MOMENTUM, f"perf_test_{i}")
            for i in range(100)
        ]
        
        # Benchmark save performance
        import time
        start_time = time.time()
        loader.save_evolved_strategies(evolved_genes)
        save_time = time.time() - start_time
        assert save_time < 2.0  # Should save 100 configs in <2 seconds
        
        # Benchmark load performance  
        start_time = time.time()
        strategies = loader.load_strategies()
        load_time = time.time() - start_time
        assert load_time < 1.0  # Should load 100 configs in <1 second
        assert len(strategies) == 100
```

---

## Implementation Timeline & Dependencies

### Week Implementation Schedule

#### Day 1: Core Implementation
- **Morning**: Implement `ConfigStrategyLoader` core class (75 lines)
- **Afternoon**: Implement JSON serialization/deserialization methods (50 lines)
- **Evening**: Basic integration testing with existing `SeedGenes`

#### Day 2: Advanced Features  
- **Morning**: Performance metrics tracking and updating (25 lines)
- **Afternoon**: Strategy archival and management features
- **Evening**: Integration with existing `SeedRegistry`

#### Day 3: Testing & Validation
- **Morning**: Comprehensive integration test suite  
- **Afternoon**: Performance benchmarking (100+ strategies)
- **Evening**: Error handling and edge case testing

#### Day 4: Production Integration
- **Morning**: Integration with `GeneticStrategyPool` for Ray cluster
- **Afternoon**: Directory structure setup and configuration
- **Evening**: Documentation and usage examples

#### Day 5: System Integration Testing
- **Morning**: End-to-end testing with genetic evolution output
- **Afternoon**: Integration testing with paper trading pipeline  
- **Evening**: Performance optimization and final validation

### Dependencies & Prerequisites

**Existing Components (Already Implemented):**
- ✅ `SeedGenes` Pydantic model with JSON serialization capability
- ✅ `SeedRegistry` for strategy type instantiation
- ✅ `BaseSeed` interface for strategy execution
- ✅ Genetic algorithm framework producing `SeedGenes` output

**New Dependencies:**
- ✅ Directory structure: `evolved_strategies/` and `evolved_strategies/archive/`
- ✅ JSON schema validation (handled by Pydantic)
- ✅ File system permissions for config directory management

---

## Success Metrics & Validation Criteria

### Performance Metrics
```python
class ConfigLoaderSuccessMetrics:
    # Core Functionality
    json_serialization_success_rate: float = 100.0  # All SeedGenes serialize correctly
    strategy_instantiation_success_rate: float = 95.0  # 95% of configs create valid strategies
    round_trip_accuracy: float = 100.0  # SeedGenes → JSON → BaseSeed accuracy
    
    # Performance Characteristics
    save_100_strategies_time_seconds: float = 2.0  # <2s to save 100 strategies
    load_100_strategies_time_seconds: float = 1.0  # <1s to load 100 strategies
    memory_usage_per_strategy_kb: float = 5.0  # <5KB per strategy config
    
    # Integration Quality
    seed_registry_integration_success: bool = True  # Works with existing registry
    genetic_algorithm_compatibility: bool = True  # Accepts GA output directly
    paper_trading_integration: bool = True  # Integrates with paper trading
    
    # Management Features
    performance_update_success_rate: float = 100.0  # Performance metrics update correctly
    archival_system_functionality: bool = True  # Strategy archival works
    directory_management_robustness: bool = True  # Handles file system operations
```

### Validation Commands
```bash
# Core functionality testing
python -m pytest tests/integration/test_config_strategy_loader.py -v

# Performance benchmarking  
python -m pytest tests/integration/test_config_strategy_loader.py::test_config_loader_performance

# Integration with genetic algorithm
python scripts/integration/test_genetic_config_integration.py --strategies 50

# End-to-end pipeline testing
python scripts/integration/test_evolution_to_config_pipeline.py
```

### Go/No-Go Criteria for Production Deployment
- ✅ 100% JSON serialization success rate for all `SeedGenes` outputs
- ✅ 95%+ strategy instantiation success rate from JSON configs
- ✅ <2 seconds to save 100+ evolved strategies as configs
- ✅ <1 second to load and instantiate 100+ strategies from configs
- ✅ Perfect integration with existing `SeedRegistry` and `BaseSeed` framework
- ✅ Comprehensive error handling for file system and JSON parsing errors

---

## Production Deployment Guidelines

### Directory Structure Setup
```bash
# Create production directories
mkdir -p evolved_strategies/archive
chmod 755 evolved_strategies/
chmod 755 evolved_strategies/archive/

# Set up configuration
export CONFIG_STRATEGY_DIR="evolved_strategies"
export CONFIG_ARCHIVE_DIR="evolved_strategies/archive"
```

### Integration with Existing Systems
```python
# Usage in genetic algorithm evolution
from src.strategy.config_strategy_loader import get_config_loader

# After genetic evolution completes
evolution_results = genetic_engine.evolve_population(...)
fitness_scores = [individual.fitness for individual in evolution_results]

# Save evolved strategies as configs
config_loader = get_config_loader()
config_loader.save_evolved_strategies(evolution_results, fitness_scores)

# Load top strategies for paper trading
top_strategies = config_loader.load_strategies(min_fitness=1.5, max_strategies=10)

# Deploy to paper trading system
for strategy in top_strategies:
    paper_trader.deploy_strategy(strategy)
```

### Monitoring & Maintenance
```python
# Daily strategy summary
config_loader = get_config_loader()
summary = config_loader.get_strategy_summary()
print(f"Active strategies: {summary['active_strategies']}")
print(f"Average fitness: {summary['average_fitness']:.2f}")

# Weekly archival of underperforming strategies
underperforming = config_loader.load_strategies(min_fitness=0.0, max_strategies=None)
for strategy in underperforming:
    if strategy._fitness_score < 0.5:  # Archive low-fitness strategies
        config_loader.archive_strategy(strategy._config_name, "low_performance")
```

---

## Risk Management & Troubleshooting

### Common Issues & Solutions

**Issue: JSON serialization fails for complex SeedGenes parameters**
```python
# Solution: Ensure all parameters are JSON-serializable
def validate_json_serializable(data):
    try:
        json.dumps(data)
        return True
    except TypeError:
        logger.error(f"Non-serializable data: {data}")
        return False
```

**Issue: Strategy instantiation fails from config**
```python
# Solution: Comprehensive error handling with fallbacks
def safe_strategy_instantiation(config):
    try:
        return self.registry.create_seed(config.seed_type, genes)
    except Exception as e:
        logger.error(f"Strategy instantiation failed: {e}")
        # Fallback to default strategy with same type
        default_genes = SeedGenes.create_default(config.seed_type)
        return self.registry.create_seed(config.seed_type, default_genes)
```

**Issue: Config directory permissions or disk space**
```python
# Solution: Proactive validation and monitoring
def validate_config_directory(config_dir):
    path = Path(config_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    # Check write permissions
    test_file = path / "test_write.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
    except PermissionError:
        raise RuntimeError(f"No write permission to {config_dir}")
    
    # Check disk space (require at least 100MB free)
    import shutil
    free_space = shutil.disk_usage(path).free
    if free_space < 100 * 1024 * 1024:
        raise RuntimeError(f"Insufficient disk space: {free_space / 1024 / 1024:.1f}MB")
```

---

## Phase Completion Deliverables

- ✅ **ConfigStrategyLoader implementation** with full JSON serialization/deserialization
- ✅ **Integration with existing genetic framework** (SeedGenes, SeedRegistry, BaseSeed)
- ✅ **Comprehensive testing suite** with performance benchmarks
- ✅ **Production directory structure** with proper permissions and management
- ✅ **Performance metrics tracking** and strategy lifecycle management
- ✅ **Error handling and validation** for robust production deployment
- ✅ **Documentation and usage examples** for seamless team adoption

**Phase Success Indicator**: ✅ **PRODUCTION READY** - Genetic algorithm evolved strategies seamlessly serialized to JSON configs and deployed to automated trading pipeline with zero code generation complexity.

---

## Next Phase Integration

This ConfigStrategyLoader serves as the foundation for:
- **AutomatedDecisionEngine** (reads strategy performance from configs)  
- **UltraCompressedEvolution** (saves 500+ strategies as configs)
- **GeneticPaperTradingBridge** (loads strategies from configs for deployment)

**Ready for Phase 2**: Automated Decision Engine Implementation (builds on config-driven strategy management)