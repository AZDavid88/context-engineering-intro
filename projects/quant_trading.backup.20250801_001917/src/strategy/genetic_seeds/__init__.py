"""
Genetic Seed Library - 14 Enhanced Implementations

This module provides the comprehensive genetic seed library as specified in the PRP
consultant recommendations. Each seed represents a fundamental trading primitive
that can be evolved by the genetic algorithm.

Based on consultant validation requirements:
- 14 core seed implementations with unit tests
- Self-contained, deterministic primitives
- Genetic parameter evolution capabilities
- Production-ready validation patterns
- Enhanced volatility and trend analysis seeds
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

# Enhanced genetic seed collection (Seeds #13-14)
from .bollinger_bands_seed import BollingerBandsSeed
from .nadaraya_watson_seed import NadarayaWatsonSeed

__all__ = [
    'SeedRegistry',
    'register_seed',
    'get_registry',
    'BaseSeed',
    'SeedGenes',
    'SeedFitness',
    'SeedType',
    'get_all_genetic_seeds',
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
    'PCATreeQuantileSeed',
    'BollingerBandsSeed',
    'NadarayaWatsonSeed'
]

def get_all_genetic_seeds():
    """
    Get all available genetic seed classes for evolution.
    
    Returns:
        List of genetic seed classes ready for evolution
    """
    return [
        EMACrossoverSeed,
        DonchianBreakoutSeed,
        RSIFilterSeed,
        StochasticOscillatorSeed,
        SMATrendFilterSeed,
        ATRStopLossSeed,
        IchimokuCloudSeed,
        VWAPReversionSeed,
        VolatilityScalingSeed,
        FundingRateCarrySeed,
        LinearSVCClassifierSeed,
        PCATreeQuantileSeed,
        BollingerBandsSeed,
        NadarayaWatsonSeed
    ]

# Version info
__version__ = "1.0.0"