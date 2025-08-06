"""
Universal Strategy Engine - Cross-Asset Strategy Coordination

This module coordinates genetic strategies across the entire Hyperliquid asset
universe, eliminating survivorship bias through continuous allocation rather
than binary selection, and managing correlation constraints.

This addresses GAP 2 from the PRP: Missing Cross-Asset Strategy Coordination.

Key Features:
- Coordinate genetic strategies across 50+ Hyperliquid assets
- Eliminate survivorship bias through continuous allocation weights
- Cross-asset correlation management and position scaling
- Genetic weight evolution for optimal asset allocation
- Dynamic rebalancing based on performance and market conditions
- Multi-objective optimization across asset classes
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict

from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedFitness
from src.strategy.genetic_engine import GeneticEngine, EvolutionResults
from src.backtesting.strategy_converter import StrategyConverter, MultiAssetSignals
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.execution.position_sizer import GeneticPositionSizer, PositionSizeResult
from src.data.hyperliquid_client import HyperliquidClient
from src.config.settings import get_settings, Settings


class AssetClass(str, Enum):
    """Asset classification for Hyperliquid universe."""
    MAJOR_CRYPTO = "major_crypto"  # BTC, ETH
    ALT_CRYPTO = "alt_crypto"     # SOL, AVAX, etc.
    DEFI_TOKENS = "defi_tokens"   # UNI, AAVE, etc.
    LAYER_2 = "layer_2"          # ARB, OP, MATIC
    MEME_COINS = "meme_coins"     # DOGE, SHIB, etc.


class AllocationMethod(str, Enum):
    """Methods for asset allocation."""
    EQUAL_WEIGHT = "equal_weight"
    GENETIC_WEIGHT = "genetic_weight"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOMENTUM_BASED = "momentum_based"
    RISK_PARITY = "risk_parity"


@dataclass
class AssetMetadata:
    """Metadata for each asset in the universe."""
    
    symbol: str
    asset_class: AssetClass
    market_cap_rank: int
    avg_daily_volume: float
    volatility_percentile: float
    correlation_cluster: int
    liquidity_score: float
    genetic_performance_score: float = 0.0
    
    # Trading constraints
    min_position_size: float = 0.001
    max_position_size: float = 0.15
    is_tradeable: bool = True


@dataclass
class UniversalStrategyResult:
    """Result of universal strategy coordination."""
    
    timestamp: datetime
    total_assets: int
    active_assets: int
    
    # Allocation results
    asset_allocations: Dict[str, float]
    allocation_method: AllocationMethod
    total_allocation: float
    
    # Genetic evolution results
    best_universal_fitness: float
    population_diversity: float
    convergence_generation: int
    
    # Risk metrics
    portfolio_volatility: float
    max_correlation: float
    concentration_risk: float
    
    # Performance attribution
    performance_by_asset_class: Dict[AssetClass, float]
    correlation_matrix: Optional[pd.DataFrame] = None


class UniversalStrategyEngine:
    """Coordinates genetic strategies across entire Hyperliquid universe."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize universal strategy engine.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.UniversalEngine")
        
        # Core components
        self.genetic_engine = GeneticEngine(settings=settings)
        self.strategy_converter = StrategyConverter(settings)
        self.performance_analyzer = PerformanceAnalyzer(settings)
        self.position_sizer = GeneticPositionSizer(settings)
        self.hyperliquid_client = HyperliquidClient(settings)
        
        # Universe management
        self.asset_universe: Dict[str, AssetMetadata] = {}
        self.active_strategies: Dict[str, BaseSeed] = {}
        self.strategy_performance: Dict[str, List[SeedFitness]] = defaultdict(list)
        
        # Allocation state
        self.current_allocations: Dict[str, float] = {}
        self.allocation_history: List[UniversalStrategyResult] = []
        
        # Configuration
        self.max_assets = 50  # Maximum assets to manage simultaneously
        self.min_allocation = 0.001  # Minimum allocation per asset
        self.max_allocation = 0.15   # Maximum allocation per asset  
        self.correlation_threshold = 0.7  # Maximum correlation between assets
        self.rebalance_frequency = timedelta(hours=4)  # Rebalance every 4 hours
        
        # Performance tracking
        self.evolution_count = 0
        self.last_rebalance = datetime.now(timezone.utc)
        
        self.logger.info("Universal strategy engine initialized")
    
    async def initialize_universe(self) -> None:
        """Initialize the tradeable asset universe from Hyperliquid."""
        try:
            self.logger.info("Initializing asset universe...")
            
            # Get available assets from Hyperliquid
            await self.hyperliquid_client.connect()
            available_assets = await self._fetch_hyperliquid_universe()
            
            # Create asset metadata for each asset
            for asset_info in available_assets:
                metadata = await self._create_asset_metadata(asset_info)
                self.asset_universe[metadata.symbol] = metadata
            
            self.logger.info(f"Universe initialized with {len(self.asset_universe)} assets")
            
            # Log asset distribution by class
            class_counts = defaultdict(int)
            for asset in self.asset_universe.values():
                class_counts[asset.asset_class] += 1
            
            for asset_class, count in class_counts.items():
                self.logger.info(f"  - {asset_class.value}: {count} assets")
                
        except Exception as e:
            self.logger.error(f"Error initializing universe: {e}")
            # Fallback to predefined universe
            await self._initialize_fallback_universe()
    
    async def evolve_universal_strategies(self, market_data: Dict[str, pd.DataFrame],
                                        n_generations: int = 50) -> UniversalStrategyResult:
        """Evolve strategies across the entire asset universe.
        
        Args:
            market_data: Dictionary mapping symbols to OHLCV data
            n_generations: Number of genetic algorithm generations
            
        Returns:
            Universal strategy coordination result
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting universal strategy evolution for {len(market_data)} assets")
        
        try:
            # Select assets for this evolution cycle
            selected_assets = self._select_assets_for_evolution(market_data)
            self.logger.info(f"Selected {len(selected_assets)} assets for evolution")
            
            # Evolve strategies for each selected asset
            evolution_results = {}
            for symbol in selected_assets:
                if symbol in market_data:
                    asset_data = market_data[symbol]
                    
                    self.logger.debug(f"Evolving strategy for {symbol}...")
                    result = await self.genetic_engine.evolve(asset_data, n_generations)
                    evolution_results[symbol] = result
                    
                    # Store best strategy
                    if result.best_individual:
                        self.active_strategies[symbol] = result.best_individual
                        
                        # Store performance
                        fitness = await self._extract_fitness_from_evolution(result, symbol, asset_data)
                        self.strategy_performance[symbol].append(fitness)
            
            # Calculate cross-asset correlation matrix
            correlation_matrix = await self._calculate_cross_asset_correlations(market_data)
            
            # Optimize universal allocation across assets
            allocation_result = await self._optimize_universal_allocation(
                evolution_results, correlation_matrix, market_data
            )
            
            # Update current allocations
            self.current_allocations = allocation_result.asset_allocations
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                allocation_result.asset_allocations, correlation_matrix
            )
            
            # Create result object
            result = UniversalStrategyResult(
                timestamp=start_time,
                total_assets=len(self.asset_universe),
                active_assets=len(selected_assets),
                asset_allocations=allocation_result.asset_allocations,
                allocation_method=allocation_result.allocation_method,
                total_allocation=sum(allocation_result.asset_allocations.values()),
                best_universal_fitness=allocation_result.best_universal_fitness,
                population_diversity=allocation_result.population_diversity,
                convergence_generation=allocation_result.convergence_generation,
                portfolio_volatility=portfolio_metrics['portfolio_volatility'],
                max_correlation=portfolio_metrics['max_correlation'],
                concentration_risk=portfolio_metrics['concentration_risk'],
                performance_by_asset_class=portfolio_metrics['performance_by_asset_class'],
                correlation_matrix=correlation_matrix
            )
            
            # Store result
            self.allocation_history.append(result)
            self.evolution_count += 1
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.info(f"Universal evolution completed in {execution_time:.2f}s. "
                           f"Total allocation: {result.total_allocation:.3f} across "
                           f"{len(result.asset_allocations)} assets")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in universal strategy evolution: {e}")
            raise
    
    async def rebalance_portfolio(self, current_positions: Dict[str, float],
                                market_data: Dict[str, pd.DataFrame]) -> List[PositionSizeResult]:
        """Rebalance portfolio based on current allocations and positions.
        
        Args:
            current_positions: Current position sizes by symbol
            market_data: Current market data for calculations
            
        Returns:
            List of position sizing results for rebalancing
        """
        self.logger.info("Starting portfolio rebalancing...")
        
        try:
            rebalance_results = []
            
            # Update position sizer with current state
            await self.position_sizer.update_portfolio_state(
                current_positions, 
                self.allocation_history[-1].correlation_matrix if self.allocation_history else None
            )
            
            # Calculate required position changes
            for symbol, target_allocation in self.current_allocations.items():
                current_position = current_positions.get(symbol, 0.0)
                
                if symbol in self.active_strategies and symbol in market_data:
                    strategy = self.active_strategies[symbol]
                    asset_data = market_data[symbol]
                    
                    # Calculate position size for target allocation
                    position_result = await self.position_sizer.calculate_position_size(
                        symbol, strategy, asset_data, target_allocation
                    )
                    
                    # Scale to match target allocation
                    position_result.target_size = target_allocation
                    
                    rebalance_results.append(position_result)
            
            # Handle positions that need to be closed (not in current allocations)
            for symbol, current_position in current_positions.items():
                if symbol not in self.current_allocations and current_position != 0:
                    # Create zero-target position result
                    close_result = PositionSizeResult(
                        symbol=symbol,
                        target_size=0.0,
                        max_size=0.0,
                        raw_size=0.0,
                        scaling_factor=0.0,
                        method_used='close_position',
                        risk_metrics={},
                        correlation_adjustment=0.0,
                        volatility_adjustment=0.0,
                        timestamp=datetime.now(timezone.utc)
                    )
                    rebalance_results.append(close_result)
            
            self.last_rebalance = datetime.now(timezone.utc)
            
            self.logger.info(f"Rebalancing complete: {len(rebalance_results)} position adjustments")
            return rebalance_results
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
            return []
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced.
        
        Returns:
            True if rebalancing is needed
        """
        time_since_rebalance = datetime.now(timezone.utc) - self.last_rebalance
        return time_since_rebalance >= self.rebalance_frequency
    
    async def _fetch_hyperliquid_universe(self) -> List[Dict[str, Any]]:
        """Fetch available assets from Hyperliquid.
        
        Returns:
            List of asset information dictionaries
        """
        try:
            # This would call the actual Hyperliquid API
            # For now, return a comprehensive list of major crypto assets
            return [
                {'symbol': 'BTC-USD', 'marketCap': 1, 'volume24h': 10000000000},
                {'symbol': 'ETH-USD', 'marketCap': 2, 'volume24h': 5000000000},
                {'symbol': 'SOL-USD', 'marketCap': 5, 'volume24h': 1000000000},
                {'symbol': 'AVAX-USD', 'marketCap': 10, 'volume24h': 500000000},
                {'symbol': 'ARB-USD', 'marketCap': 15, 'volume24h': 300000000},
                {'symbol': 'OP-USD', 'marketCap': 20, 'volume24h': 200000000},
                {'symbol': 'MATIC-USD', 'marketCap': 12, 'volume24h': 400000000},
                {'symbol': 'UNI-USD', 'marketCap': 25, 'volume24h': 150000000},
                {'symbol': 'AAVE-USD', 'marketCap': 30, 'volume24h': 100000000},
                {'symbol': 'LINK-USD', 'marketCap': 8, 'volume24h': 600000000}
            ]
        except Exception as e:
            self.logger.error(f"Error fetching Hyperliquid universe: {e}")
            return []
    
    async def _create_asset_metadata(self, asset_info: Dict[str, Any]) -> AssetMetadata:
        """Create asset metadata from Hyperliquid asset information.
        
        Args:
            asset_info: Asset information from Hyperliquid
            
        Returns:
            AssetMetadata object
        """
        symbol = asset_info['symbol']
        market_cap_rank = asset_info.get('marketCap', 999)
        volume_24h = asset_info.get('volume24h', 0)
        
        # Classify asset by symbol and market cap
        asset_class = self._classify_asset(symbol, market_cap_rank)
        
        # Calculate derived metrics
        liquidity_score = min(1.0, volume_24h / 1000000000)  # Normalize to billions
        volatility_percentile = np.random.uniform(0.3, 0.9)  # Would calculate from actual data
        correlation_cluster = hash(symbol) % 5  # Simple clustering
        
        return AssetMetadata(
            symbol=symbol,
            asset_class=asset_class,
            market_cap_rank=market_cap_rank,
            avg_daily_volume=volume_24h,
            volatility_percentile=volatility_percentile,
            correlation_cluster=correlation_cluster,
            liquidity_score=liquidity_score
        )
    
    def _classify_asset(self, symbol: str, market_cap_rank: int) -> AssetClass:
        """Classify asset into asset class.
        
        Args:
            symbol: Trading symbol
            market_cap_rank: Market cap ranking
            
        Returns:
            Asset classification
        """
        symbol_lower = symbol.lower()
        
        if 'btc' in symbol_lower or 'eth' in symbol_lower:
            return AssetClass.MAJOR_CRYPTO
        elif any(token in symbol_lower for token in ['uni', 'aave', 'comp', 'mkr']):
            return AssetClass.DEFI_TOKENS
        elif any(token in symbol_lower for token in ['arb', 'op', 'matic']):
            return AssetClass.LAYER_2
        elif any(token in symbol_lower for token in ['doge', 'shib', 'pepe']):
            return AssetClass.MEME_COINS
        else:
            return AssetClass.ALT_CRYPTO
    
    async def _initialize_fallback_universe(self) -> None:
        """Initialize fallback universe with predefined assets."""
        fallback_assets = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'ARB-USD',
            'OP-USD', 'MATIC-USD', 'UNI-USD', 'AAVE-USD', 'LINK-USD'
        ]
        
        for i, symbol in enumerate(fallback_assets):
            metadata = AssetMetadata(
                symbol=symbol,
                asset_class=self._classify_asset(symbol, i + 1),
                market_cap_rank=i + 1,
                avg_daily_volume=1000000000 / (i + 1),
                volatility_percentile=0.5,
                correlation_cluster=i % 3,
                liquidity_score=1.0 / (i + 1)
            )
            self.asset_universe[symbol] = metadata
        
        self.logger.info(f"Fallback universe initialized with {len(self.asset_universe)} assets")
    
    def _select_assets_for_evolution(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Select assets for evolution based on various criteria.
        
        Args:
            market_data: Available market data
            
        Returns:
            List of selected asset symbols
        """
        # Filter assets with sufficient data
        valid_assets = []
        for symbol, data in market_data.items():
            if symbol in self.asset_universe and len(data) >= 100:  # Minimum data requirement
                valid_assets.append(symbol)
        
        # Sort by multiple criteria
        def asset_score(symbol):
            metadata = self.asset_universe[symbol]
            data_quality = len(market_data[symbol]) / 1000  # Normalize data length
            
            score = (
                metadata.liquidity_score * 0.4 +
                (1.0 / max(metadata.market_cap_rank, 1)) * 0.3 +
                data_quality * 0.2 +
                metadata.genetic_performance_score * 0.1
            )
            return score
        
        # Select top assets up to maximum
        valid_assets.sort(key=asset_score, reverse=True)
        selected = valid_assets[:self.max_assets]
        
        return selected
    
    async def _calculate_cross_asset_correlations(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix across all assets.
        
        Args:
            market_data: Market data for correlation calculation
            
        Returns:
            Correlation matrix
        """
        try:
            # Extract returns for all assets
            returns_data = {}
            
            for symbol, data in market_data.items():
                if len(data) > 1:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > 20:  # Minimum periods for correlation
                        returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Align all return series to common dates
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()
    
    async def _optimize_universal_allocation(self, evolution_results: Dict[str, EvolutionResults],
                                           correlation_matrix: pd.DataFrame,
                                           market_data: Dict[str, pd.DataFrame]) -> 'AllocationOptimizationResult':
        """Optimize allocation across all evolved strategies.
        
        Args:
            evolution_results: Results from genetic evolution
            correlation_matrix: Cross-asset correlations
            market_data: Market data for analysis
            
        Returns:
            Optimization result with allocations
        """
        try:
            # Extract fitness scores for each asset
            asset_fitness = {}
            for symbol, result in evolution_results.items():
                if result.best_individual and result.best_individual.fitness:
                    # Use Sharpe ratio as primary fitness metric
                    fitness_values = result.best_individual.fitness.values
                    sharpe_ratio = fitness_values[0] if len(fitness_values) > 0 else 0.0
                    asset_fitness[symbol] = max(0.0, sharpe_ratio)
                else:
                    asset_fitness[symbol] = 0.0
            
            # Apply genetic allocation optimization
            allocations = self._calculate_genetic_allocations(
                asset_fitness, correlation_matrix, market_data
            )
            
            # Calculate optimization metrics
            best_fitness = max(asset_fitness.values()) if asset_fitness else 0.0
            diversity = self._calculate_allocation_diversity(allocations)
            convergence_gen = sum(
                result.generation_stats[-1]['gen'] if result.generation_stats else 0
                for result in evolution_results.values()
            ) / max(len(evolution_results), 1)
            
            return AllocationOptimizationResult(
                asset_allocations=allocations,
                allocation_method=AllocationMethod.GENETIC_WEIGHT,
                best_universal_fitness=best_fitness,
                population_diversity=diversity,
                convergence_generation=int(convergence_gen)
            )
            
        except Exception as e:
            self.logger.error(f"Error in allocation optimization: {e}")
            return AllocationOptimizationResult(
                asset_allocations={},
                allocation_method=AllocationMethod.EQUAL_WEIGHT,
                best_universal_fitness=0.0,
                population_diversity=0.0,
                convergence_generation=0
            )
    
    def _calculate_genetic_allocations(self, asset_fitness: Dict[str, float],
                                     correlation_matrix: pd.DataFrame,
                                     market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate optimal allocations using genetic principles.
        
        Args:
            asset_fitness: Fitness score for each asset
            correlation_matrix: Cross-asset correlations
            market_data: Market data for volatility calculations
            
        Returns:
            Optimal allocation weights
        """
        if not asset_fitness:
            return {}
        
        # Start with fitness-weighted allocation
        total_fitness = sum(asset_fitness.values())
        if total_fitness == 0:
            # Equal weight fallback
            equal_weight = 1.0 / len(asset_fitness)
            return {symbol: equal_weight for symbol in asset_fitness.keys()}
        
        # Raw fitness-based weights
        raw_weights = {
            symbol: fitness / total_fitness 
            for symbol, fitness in asset_fitness.items()
        }
        
        # Apply correlation penalty
        correlation_adjusted_weights = self._apply_correlation_penalty(
            raw_weights, correlation_matrix
        )
        
        # Apply volatility adjustment
        volatility_adjusted_weights = self._apply_volatility_adjustment(
            correlation_adjusted_weights, market_data
        )
        
        # Apply position limits and normalize
        final_weights = self._apply_allocation_limits(volatility_adjusted_weights)
        
        return final_weights
    
    def _apply_correlation_penalty(self, weights: Dict[str, float],
                                 correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Apply correlation penalty to reduce concentration in correlated assets.
        
        Args:
            weights: Initial weights
            correlation_matrix: Correlation matrix
            
        Returns:
            Correlation-adjusted weights
        """
        if correlation_matrix.empty or len(weights) <= 1:
            return weights
        
        adjusted_weights = weights.copy()
        
        for symbol1 in weights:
            if symbol1 not in correlation_matrix.index:
                continue
                
            penalty_factor = 1.0
            
            for symbol2 in weights:
                if symbol1 != symbol2 and symbol2 in correlation_matrix.columns:
                    correlation = abs(correlation_matrix.loc[symbol1, symbol2])
                    
                    if correlation > self.correlation_threshold:
                        # Apply penalty based on correlation strength and other asset weight
                        penalty = correlation * weights[symbol2] * 0.5
                        penalty_factor -= penalty
            
            penalty_factor = max(0.1, penalty_factor)  # Minimum penalty factor
            adjusted_weights[symbol1] *= penalty_factor
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                symbol: weight / total_weight 
                for symbol, weight in adjusted_weights.items()
            }
        
        return adjusted_weights
    
    def _apply_volatility_adjustment(self, weights: Dict[str, float],
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Apply volatility-based position scaling.
        
        Args:
            weights: Current weights
            market_data: Market data for volatility calculation
            
        Returns:
            Volatility-adjusted weights
        """
        volatility_adjusted = weights.copy()
        
        # Calculate volatilities
        volatilities = {}
        for symbol, data in market_data.items():
            if symbol in weights and len(data) > 20:
                returns = data['close'].pct_change().dropna()
                if len(returns) > 0:
                    volatilities[symbol] = returns.std()
        
        if not volatilities:
            return weights
        
        # Calculate average volatility for scaling
        avg_volatility = np.mean(list(volatilities.values()))
        
        # Adjust weights inversely to volatility
        for symbol in weights:
            if symbol in volatilities:
                vol_ratio = avg_volatility / volatilities[symbol]
                vol_ratio = np.clip(vol_ratio, 0.5, 2.0)  # Reasonable bounds
                volatility_adjusted[symbol] *= vol_ratio
        
        # Renormalize
        total_weight = sum(volatility_adjusted.values())
        if total_weight > 0:
            volatility_adjusted = {
                symbol: weight / total_weight 
                for symbol, weight in volatility_adjusted.items()
            }
        
        return volatility_adjusted
    
    def _apply_allocation_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply individual position limits and normalize.
        
        Args:
            weights: Input weights
            
        Returns:
            Limited and normalized weights
        """
        limited_weights = {}
        
        # Apply individual limits
        for symbol, weight in weights.items():
            limited_weight = max(self.min_allocation, min(self.max_allocation, weight))
            
            # Only include if above minimum
            if limited_weight >= self.min_allocation:
                limited_weights[symbol] = limited_weight
        
        # Renormalize to ensure total doesn't exceed 1.0
        total_weight = sum(limited_weights.values())
        if total_weight > 1.0:
            limited_weights = {
                symbol: weight / total_weight 
                for symbol, weight in limited_weights.items()
            }
        
        return limited_weights
    
    def _calculate_allocation_diversity(self, allocations: Dict[str, float]) -> float:
        """Calculate diversity metric for allocations.
        
        Args:
            allocations: Asset allocations
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not allocations:
            return 0.0
        
        weights = list(allocations.values())
        n_assets = len(weights)
        
        if n_assets <= 1:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (inverse concentration)
        hhi = sum(w ** 2 for w in weights)
        max_hhi = 1.0  # All weight in one asset
        min_hhi = 1.0 / n_assets  # Equal weights
        
        # Normalize to 0-1 scale (1 = most diverse)
        if max_hhi > min_hhi:
            diversity = (max_hhi - hhi) / (max_hhi - min_hhi)
        else:
            diversity = 1.0
        
        return diversity
    
    def _calculate_portfolio_metrics(self, allocations: Dict[str, float],
                                   correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio-level risk and performance metrics.
        
        Args:
            allocations: Asset allocations
            correlation_matrix: Correlation matrix
            
        Returns:
            Dictionary of portfolio metrics
        """
        try:
            # Portfolio volatility (simplified)
            if not correlation_matrix.empty and len(allocations) > 1:
                portfolio_variance = 0.0
                
                for symbol1, weight1 in allocations.items():
                    for symbol2, weight2 in allocations.items():
                        if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                            correlation = correlation_matrix.loc[symbol1, symbol2]
                            # Simplified: assume volatility of 0.02 for all assets
                            portfolio_variance += weight1 * weight2 * correlation * 0.02 * 0.02
                
                portfolio_volatility = np.sqrt(portfolio_variance)
            else:
                portfolio_volatility = 0.02  # Default assumption
            
            # Maximum correlation
            max_correlation = 0.0
            if not correlation_matrix.empty:
                for symbol1 in allocations:
                    for symbol2 in allocations:
                        if (symbol1 != symbol2 and 
                            symbol1 in correlation_matrix.index and 
                            symbol2 in correlation_matrix.columns):
                            corr = abs(correlation_matrix.loc[symbol1, symbol2])
                            max_correlation = max(max_correlation, corr)
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(w ** 2 for w in allocations.values())
            
            # Performance by asset class
            performance_by_class = defaultdict(float)
            for symbol, allocation in allocations.items():
                if symbol in self.asset_universe:
                    asset_class = self.asset_universe[symbol].asset_class
                    performance_by_class[asset_class] += allocation
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'max_correlation': max_correlation,
                'concentration_risk': concentration_risk,
                'performance_by_asset_class': dict(performance_by_class)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                'portfolio_volatility': 0.0,
                'max_correlation': 0.0,
                'concentration_risk': 0.0,
                'performance_by_asset_class': {}
            }
    
    async def _extract_fitness_from_evolution(self, evolution_result: EvolutionResults,
                                            symbol: str, market_data: pd.DataFrame) -> SeedFitness:
        """Extract fitness metrics from evolution result.
        
        Args:
            evolution_result: Evolution result
            symbol: Asset symbol
            market_data: Market data
            
        Returns:
            SeedFitness object
        """
        if not evolution_result.best_individual:
            # Return default fitness
            return SeedFitness(
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                consistency=0.0,
                total_return=0.0,
                volatility=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                max_consecutive_losses=100,
                composite_fitness=0.0,
                in_sample_fitness=0.0,
                out_of_sample_fitness=0.0,
                walk_forward_fitness=0.0
            )
        
        # Use performance analyzer to extract detailed fitness
        try:
            # Convert strategy to signals
            signals_result = self.strategy_converter.convert_seed_to_signals(
                evolution_result.best_individual, market_data, symbol
            )
            
            # Create portfolio for analysis
            portfolio = self.strategy_converter.create_vectorbt_portfolio(
                signals_result, market_data
            )
            
            # Extract fitness
            fitness = self.performance_analyzer.extract_genetic_fitness(portfolio, symbol)
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error extracting fitness for {symbol}: {e}")
            return SeedFitness(
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                consistency=0.0,
                total_return=0.0,
                volatility=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                max_consecutive_losses=100,
                composite_fitness=0.0,
                in_sample_fitness=0.0,
                out_of_sample_fitness=0.0,
                walk_forward_fitness=0.0
            )
    
    def get_universe_stats(self) -> Dict[str, Any]:
        """Get statistics about the asset universe.
        
        Returns:
            Dictionary with universe statistics
        """
        if not self.asset_universe:
            return {}
        
        class_counts = defaultdict(int)
        total_volume = 0.0
        total_liquidity = 0.0
        
        for asset in self.asset_universe.values():
            class_counts[asset.asset_class] += 1
            total_volume += asset.avg_daily_volume
            total_liquidity += asset.liquidity_score
        
        return {
            'total_assets': len(self.asset_universe),
            'asset_classes': dict(class_counts),
            'active_strategies': len(self.active_strategies),
            'total_daily_volume': total_volume,
            'avg_liquidity_score': total_liquidity / len(self.asset_universe),
            'evolution_count': self.evolution_count,
            'last_rebalance': self.last_rebalance.isoformat(),
            'current_allocation_count': len(self.current_allocations)
        }


@dataclass
class AllocationOptimizationResult:
    """Result of allocation optimization."""
    asset_allocations: Dict[str, float]
    allocation_method: AllocationMethod
    best_universal_fitness: float
    population_diversity: float
    convergence_generation: int


async def test_universal_strategy_engine():
    """Test function for universal strategy engine."""
    
    print("=== Universal Strategy Engine Test ===")
    
    # Create engine
    engine = UniversalStrategyEngine()
    
    try:
        # Initialize universe
        print("Initializing asset universe...")
        await engine.initialize_universe()
        print(f"✅ Universe initialized with {len(engine.asset_universe)} assets")
        
        # Create test market data
        symbols = list(engine.asset_universe.keys())[:5]  # Test with first 5 assets
        market_data = {}
        
        for symbol in symbols:
            dates = pd.date_range('2023-01-01', periods=200, freq='1H')
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 200)))
            
            market_data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.uniform(1000, 5000, 200)
            }, index=dates)
        
        print(f"Created test data for {len(market_data)} assets")
        
        # Test universal evolution
        print("Running universal strategy evolution...")
        result = await engine.evolve_universal_strategies(market_data, n_generations=5)
        
        print(f"✅ Universal Evolution Complete:")
        print(f"  - Total assets: {result.total_assets}")
        print(f"  - Active assets: {result.active_assets}")
        print(f"  - Total allocation: {result.total_allocation:.3f}")
        print(f"  - Portfolio volatility: {result.portfolio_volatility:.3f}")
        print(f"  - Max correlation: {result.max_correlation:.3f}")
        print(f"  - Best fitness: {result.best_universal_fitness:.3f}")
        
        print(f"\nAsset Allocations:")
        for symbol, allocation in result.asset_allocations.items():
            print(f"  - {symbol}: {allocation:.3f}")
        
        # Test rebalancing
        print("\nTesting portfolio rebalancing...")
        current_positions = {symbol: 0.02 for symbol in symbols}  # Small current positions
        
        rebalance_results = await engine.rebalance_portfolio(current_positions, market_data)
        print(f"✅ Rebalancing complete: {len(rebalance_results)} position adjustments")
        
        for result in rebalance_results[:3]:  # Show first 3
            print(f"  - {result.symbol}: target {result.target_size:.3f}")
        
        # Show universe stats
        stats = engine.get_universe_stats()
        print(f"\nUniverse Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    print(f"\n✅ Universal Strategy Engine test completed successfully!")


if __name__ == "__main__":
    """Test the universal strategy engine."""
    
    import asyncio
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_universal_strategy_engine())