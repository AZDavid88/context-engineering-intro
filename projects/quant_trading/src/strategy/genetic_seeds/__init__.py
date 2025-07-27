"""
Genetic Seed Library - 12 Core Implementations

This module provides the comprehensive genetic seed library as specified in the PRP
consultant recommendations. Each seed represents a fundamental trading primitive
that can be evolved by the genetic algorithm.

Based on consultant validation requirements:
- 12 core seed implementations with unit tests
- Self-contained, deterministic primitives
- Genetic parameter evolution capabilities
- Production-ready validation patterns
"""

from .seed_registry import SeedRegistry, register_seed, get_registry
from .base_seed import BaseSeed, SeedGenes, SeedFitness, SeedType

# Import all 12 seed implementations (COMPLETED)
from .ema_crossover_seed import EMACrossoverSeed
from .donchian_breakout_seed import DonchianBreakoutSeed  
from .rsi_filter_seed import RSIFilterSeed
from .stochastic_oscillator_seed import StochasticOscillatorSeed
from .sma_trend_filter_seed import SMATrendFilterSeed
from .atr_stop_loss_seed import ATRStopLossSeed
from .ichimoku_cloud_seed import IchimokuCloudSeed
from .vwap_reversion_seed import VWAPReversionSeed

# Complete 12-seed library with advanced ML seeds
from .volatility_scaling_seed import VolatilityScalingSeed
from .funding_rate_carry_seed import FundingRateCarrySeed
from .linear_svc_classifier_seed import LinearSVCClassifierSeed
from .pca_tree_quantile_seed import PCATreeQuantileSeed

__all__ = [
    'SeedRegistry',
    'register_seed',
    'get_registry',
    'BaseSeed',
    'SeedGenes',
    'SeedFitness',
    'SeedType',
    'EMACrossoverSeed',
    'DonchianBreakoutSeed',
    'RSIFilterSeed',
    'StochasticOscillatorSeed',
    'SMATrendFilterSeed',
    'ATRStopLossSeed',
    'IchimokuCloudSeed',
    'VWAPReversionSeed',
    'VolatilityScalingSeed',
    'FundingRateCarrySeed',
    'LinearSVCClassifierSeed',
    'PCATreeQuantileSeed'
]

# Version info
__version__ = "1.0.0"