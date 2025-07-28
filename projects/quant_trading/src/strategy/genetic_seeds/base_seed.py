"""
Base Genetic Seed Framework

This module provides the foundational framework for all genetic seeds in the
trading organism. Each seed represents a fundamental trading primitive that
can be evolved by genetic algorithms.

Key Features:
- Standardized seed interface for genetic evolution
- Multi-objective fitness evaluation
- Type-safe parameter validation
- Production-ready error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from src.config.settings import get_settings, Settings
from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna


class SeedType(str, Enum):
    """Types of genetic seeds."""
    MOMENTUM = "momentum"           # Trend-following strategies
    MEAN_REVERSION = "mean_reversion"  # Contrarian strategies
    BREAKOUT = "breakout"          # Breakout detection
    RISK_MANAGEMENT = "risk_management"  # Stop losses, position sizing
    CARRY = "carry"                # Funding rate exploitation
    ML_CLASSIFIER = "ml_classifier"  # Machine learning based
    VOLATILITY = "volatility"      # Volatility-based strategies
    VOLUME = "volume"              # Volume-based signals
    TREND_FOLLOWING = "trend_following"  # Trend-following strategies


class SeedGenes(BaseModel):
    """Genetic parameters for a trading seed."""
    
    # Core identification
    seed_id: str = Field(..., description="Unique seed identifier")
    seed_type: SeedType = Field(..., description="Type of trading seed")
    generation: int = Field(default=0, ge=0, description="Generation number")
    
    # Genetic parameters (evolved by GA)
    parameters: Dict[str, float] = Field(default_factory=dict, description="Evolved parameters")
    
    # Technical indicator periods
    fast_period: int = Field(default=12, ge=2, le=100, description="Fast moving average period")
    slow_period: int = Field(default=26, ge=5, le=200, description="Slow moving average period")
    signal_period: int = Field(default=9, ge=2, le=50, description="Signal line period")
    
    # Thresholds and filters
    entry_threshold: float = Field(default=0.0, ge=-1.0, le=1.0, description="Entry signal threshold")
    exit_threshold: float = Field(default=0.0, ge=-1.0, le=1.0, description="Exit signal threshold")
    filter_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Signal filter threshold")
    
    # Risk management
    stop_loss: float = Field(default=0.02, ge=0.001, le=0.1, description="Stop loss percentage")
    take_profit: float = Field(default=0.04, ge=0.005, le=0.2, description="Take profit percentage")
    
    # Position sizing
    position_size: float = Field(default=0.1, ge=0.01, le=0.25, description="Position size percentage")
    
    @validator('parameters')
    @classmethod
    def validate_parameters(cls, v):
        """Validate that all parameters are numeric."""
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {key} must be numeric")
        return v


class SeedFitness(BaseModel):
    """Fitness evaluation results for a genetic seed."""
    
    # Primary fitness metrics (multi-objective)
    sharpe_ratio: float = Field(..., description="Sharpe ratio (higher is better)")
    max_drawdown: float = Field(..., description="Maximum drawdown (lower is better)")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate (0-1)")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Consistency score (0-1)")
    
    # Auxiliary metrics
    total_return: float = Field(..., description="Total return percentage")
    volatility: float = Field(..., ge=0.0, description="Return volatility")
    profit_factor: float = Field(..., ge=0.0, description="Profit factor (>1 profitable)")
    
    # Trade statistics
    total_trades: int = Field(..., ge=0, description="Total number of trades")
    avg_trade_duration: float = Field(..., ge=0.0, description="Average trade duration (hours)")
    max_consecutive_losses: int = Field(..., ge=0, description="Maximum consecutive losses")
    
    # Composite fitness score
    composite_fitness: float = Field(..., description="Weighted composite fitness score")
    
    # Validation across periods
    in_sample_fitness: float = Field(..., description="In-sample fitness score")
    out_of_sample_fitness: float = Field(..., description="Out-of-sample fitness score")
    walk_forward_fitness: float = Field(..., description="Walk-forward fitness score")
    
    @validator('composite_fitness', pre=True, always=True)
    @classmethod
    def calculate_composite_fitness(cls, v, values):
        """Calculate weighted composite fitness score."""
        if not all(key in values for key in ['sharpe_ratio', 'max_drawdown', 'win_rate', 'consistency']):
            return 0.0
            
        # Multi-objective fitness weights from consultant recommendations
        weights = {
            'sharpe_ratio': 0.5,     # Primary: Risk-adjusted returns
            'max_drawdown': 0.3,     # Critical: Risk management
            'win_rate': 0.15,        # Secondary: Consistency
            'consistency': 0.05      # Stability over time
        }
        
        # Normalize components to 0-1 scale
        sharpe_component = max(0, min(values['sharpe_ratio'] / 5.0, 1.0))  # Max Sharpe ~5
        drawdown_component = max(0, 1.0 - values['max_drawdown'])  # Lower is better
        win_rate_component = values['win_rate']  # Already 0-1
        consistency_component = values['consistency']  # Already 0-1
        
        # Calculate weighted composite
        composite = (
            weights['sharpe_ratio'] * sharpe_component +
            weights['max_drawdown'] * drawdown_component +
            weights['win_rate'] * win_rate_component +
            weights['consistency'] * consistency_component
        )
        return max(0.0, min(1.0, composite))  # Clamp to 0-1 range


class BaseSeed(ABC):
    """Base class for all genetic trading seeds."""
    
    def __init__(self, genes: SeedGenes, settings: Optional[Settings] = None):
        """Initialize seed with genetic parameters.
        
        Args:
            genes: Genetic parameters for this seed
            settings: Configuration settings
        """
        self.genes = genes
        self.settings = settings or get_settings()
        self.fitness: Optional[SeedFitness] = None
        
        # Validation
        self._validate_seed()
    
    @property
    @abstractmethod
    def seed_name(self) -> str:
        """Return human-readable seed name."""
        pass
    
    @property
    @abstractmethod
    def seed_description(self) -> str:
        """Return detailed seed description."""
        pass
    
    @property
    @abstractmethod
    def required_parameters(self) -> List[str]:
        """Return list of required genetic parameters."""
        pass
    
    @property
    @abstractmethod
    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for genetic parameters (min, max)."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data.
        
        Args:
            data: OHLCV DataFrame with technical indicators
            
        Returns:
            Boolean Series: True for buy signal, False for hold/sell
        """
        pass
    
    @abstractmethod
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate required technical indicators.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of indicator name -> values
        """
        pass
    
    def _validate_seed(self) -> None:
        """Validate seed configuration."""
        # Check required parameters
        for param in self.required_parameters:
            if param not in self.genes.parameters:
                raise ValueError(f"Required parameter '{param}' missing from genes")
        
        # Check parameter bounds
        for param, (min_val, max_val) in self.parameter_bounds.items():
            if param in self.genes.parameters:
                value = self.genes.parameters[param]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Parameter '{param}' value {value} outside bounds [{min_val}, {max_val}]")
    
    def set_fitness(self, fitness: SeedFitness) -> None:
        """Set fitness evaluation results."""
        self.fitness = fitness
    
    def get_parameter(self, name: str, default: Optional[float] = None) -> float:
        """Get parameter value with optional default."""
        if name in self.genes.parameters:
            return self.genes.parameters[name]
        elif default is not None:
            return default
        else:
            raise KeyError(f"Parameter '{name}' not found and no default provided")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert seed to dictionary representation."""
        return {
            'seed_name': self.seed_name,
            'seed_type': self.genes.seed_type.value,
            'seed_id': self.genes.seed_id,
            'generation': self.genes.generation,
            'parameters': self.genes.parameters,
            'fitness': self.fitness.dict() if self.fitness else None
        }
    
    def clone_with_mutations(self, mutations: Dict[str, float]) -> 'BaseSeed':
        """Create a mutated copy of this seed.
        
        Args:
            mutations: Dictionary of parameter -> new_value
            
        Returns:
            New seed instance with mutated parameters
        """
        # Create new genes with mutations
        new_params = self.genes.parameters.copy()
        new_params.update(mutations)
        
        new_genes = SeedGenes(
            seed_id=f"{self.genes.seed_id}_mut_{self.genes.generation + 1}",
            seed_type=self.genes.seed_type,
            generation=self.genes.generation + 1,
            parameters=new_params,
            # Copy other fields
            fast_period=self.genes.fast_period,
            slow_period=self.genes.slow_period,
            signal_period=self.genes.signal_period,
            entry_threshold=self.genes.entry_threshold,
            exit_threshold=self.genes.exit_threshold,
            filter_threshold=self.genes.filter_threshold,
            stop_loss=self.genes.stop_loss,
            take_profit=self.genes.take_profit,
            position_size=self.genes.position_size
        )
        
        # Create new instance of same class
        return self.__class__(new_genes, self.settings)