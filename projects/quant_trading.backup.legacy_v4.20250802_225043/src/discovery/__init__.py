"""
Discovery Module - Hierarchical Genetic Strategy Discovery

Implements intelligent discovery systems for the genetic trading organism:
- Asset universe filtering and optimization (ENHANCED with rate limiting)
- Progressive timeframe discovery
- Multi-archetype strategy evolution
- Dynamic resource allocation
- Advanced rate limiting with exponential backoff and correlation pre-filtering

Based on validated research from Hyperliquid, DEAP, and Anyscale documentation.
"""

from .asset_universe_filter import (
    ResearchBackedAssetFilter,
    AssetMetrics,
    FilterCriteria
)

from .enhanced_asset_filter import (
    EnhancedAssetFilter,
    EnhancedFilterMetrics
)

from .optimized_rate_limiter import (
    AdvancedRateLimiter,
    RequestPriority,
    RateLimitMetrics,
    BackoffState
)

from .crypto_safe_parameters import (
    get_crypto_safe_parameters,
    validate_trading_safety,
    CryptoSafeParameters,
    MarketRegime
)

from .hierarchical_genetic_engine import (
    HierarchicalGAOrchestrator,
    DailyPatternDiscovery,
    HourlyTimingRefinement,
    MinutePrecisionEvolution,
    StrategyGenome,
    EvolutionStage,
    TimeframeType
)

__all__ = [
    # Base asset filtering (legacy compatibility)
    'ResearchBackedAssetFilter',
    'AssetMetrics', 
    'FilterCriteria',
    
    # Enhanced rate-limited filtering (RECOMMENDED)
    'EnhancedAssetFilter',
    'EnhancedFilterMetrics',
    
    # Rate limiting system
    'AdvancedRateLimiter',
    'RequestPriority',
    'RateLimitMetrics',
    'BackoffState',
    
    # Crypto-safe parameters
    'get_crypto_safe_parameters',
    'validate_trading_safety',
    'CryptoSafeParameters',
    'MarketRegime',
    
    # Hierarchical genetic algorithm
    'HierarchicalGAOrchestrator',
    'DailyPatternDiscovery',
    'HourlyTimingRefinement',
    'MinutePrecisionEvolution',
    'StrategyGenome',
    'EvolutionStage',
    'TimeframeType'
]