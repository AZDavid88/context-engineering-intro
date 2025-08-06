"""
Position Sizer - Genetic Position Sizing Implementation

This module implements genetic algorithm evolved position sizing using optimal
allocation weights, Kelly Criterion with genetic optimization, and correlation
management for multi-asset portfolio construction.

This addresses GAP from the PRP: Missing Genetic Position Sizing Implementation.

Key Features:
- Genetic algorithm evolved allocation weights
- Kelly Criterion with safety factor and genetic optimization
- Dynamic position sizing based on volatility and momentum
- Correlation-based position scaling
- Risk scaling with genetic parameters
- Multi-asset coordination with max 15% per asset limit
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import asyncio

from src.strategy.genetic_seeds.base_seed import BaseSeed, SeedGenes
from src.config.settings import get_settings, Settings


class PositionSizeMethod(str, Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    GENETIC_EVOLVED = "genetic_evolved"
    RISK_PARITY = "risk_parity"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    
    symbol: str
    target_size: float
    max_size: float
    raw_size: float
    scaling_factor: float
    method_used: PositionSizeMethod
    risk_metrics: Dict[str, float]
    correlation_adjustment: float
    volatility_adjustment: float
    timestamp: datetime


class GeneticPositionSizer:
    """Calculate optimal position sizes using genetic algorithm evolved parameters."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize genetic position sizer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.PositionSizer")
        
        # Position sizing constraints
        self.max_position_per_asset = self.settings.trading.max_position_size
        self.max_total_exposure = 1.0  # 100% maximum total exposure
        self.min_position_size = 0.001  # Minimum meaningful position
        
        # Risk management parameters
        self.max_correlation = 0.7  # Maximum correlation between positions
        self.volatility_lookback = 20  # Days for volatility calculation
        self.kelly_safety_factor = 0.25  # Conservative Kelly implementation
        
        # Current portfolio state
        self.current_positions: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_estimates: Dict[str, float] = {}
        
        self.logger.info("Genetic position sizer initialized")
    
    async def calculate_position_size(self, symbol: str, seed: BaseSeed, 
                                    market_data: pd.DataFrame,
                                    signal_strength: float = 1.0) -> PositionSizeResult:
        """Calculate optimal position size for a genetic strategy.
        
        Args:
            symbol: Trading symbol
            seed: Genetic seed with evolved parameters
            market_data: Historical market data for calculations
            signal_strength: Strength of the trading signal (0-1)
            
        Returns:
            Position sizing result with all calculations
        """
        try:
            # Extract genetic position sizing parameters
            genetic_params = self._extract_genetic_params(seed)
            
            # Calculate base position size using genetic method
            base_size = await self._calculate_genetic_base_size(
                symbol, genetic_params, market_data, signal_strength
            )
            
            # Apply risk scaling factors
            risk_adjusted_size = self._apply_risk_scaling(
                base_size, symbol, genetic_params, market_data
            )
            
            # Apply correlation constraints
            correlation_adjusted_size = self._apply_correlation_constraints(
                risk_adjusted_size, symbol, genetic_params
            )
            
            # Apply final position limits
            final_size = self._apply_position_limits(correlation_adjusted_size, symbol)
            
            # Calculate scaling factors for analysis
            scaling_factor = final_size / base_size if base_size > 0 else 0.0
            
            # Collect risk metrics
            risk_metrics = self._calculate_risk_metrics(
                symbol, final_size, genetic_params, market_data
            )
            
            result = PositionSizeResult(
                symbol=symbol,
                target_size=final_size,
                max_size=self.max_position_per_asset,
                raw_size=base_size,
                scaling_factor=scaling_factor,
                method_used=PositionSizeMethod.GENETIC_EVOLVED,
                risk_metrics=risk_metrics,
                correlation_adjustment=correlation_adjusted_size / risk_adjusted_size if risk_adjusted_size > 0 else 1.0,
                volatility_adjustment=risk_adjusted_size / base_size if base_size > 0 else 1.0,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.logger.debug(f"Position size calculated for {symbol}: {final_size:.4f} "
                            f"(base: {base_size:.4f}, scaling: {scaling_factor:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            # Return minimum safe position
            return self._create_safe_fallback_result(symbol)
    
    def _extract_genetic_params(self, seed: BaseSeed) -> Dict[str, float]:
        """Extract position sizing parameters from genetic seed.
        
        Args:
            seed: Genetic seed with evolved parameters
            
        Returns:
            Dictionary of position sizing parameters
        """
        params = seed.genes.parameters
        
        # Extract or provide defaults for genetic position sizing parameters
        genetic_params = {
            # Base allocation weight (evolved by GA)
            'base_allocation': params.get('position_size', 0.1),
            
            # Kelly Criterion parameters
            'kelly_multiplier': params.get('kelly_multiplier', 0.25),
            'win_rate_estimate': params.get('win_rate_estimate', 0.5),
            'avg_win_loss_ratio': params.get('avg_win_loss_ratio', 1.2),
            
            # Volatility scaling parameters
            'volatility_target': params.get('volatility_target', 0.02),
            'volatility_multiplier': params.get('volatility_multiplier', 1.0),
            
            # Momentum scaling parameters
            'momentum_multiplier': params.get('momentum_multiplier', 1.0),
            'trend_strength_weight': params.get('trend_strength_weight', 0.5),
            
            # Risk management parameters
            'max_drawdown_tolerance': params.get('max_drawdown_tolerance', 0.15),
            'correlation_penalty': params.get('correlation_penalty', 0.8),
            
            # Signal confidence parameters
            'signal_confidence_min': params.get('signal_confidence_min', 0.1),
            'signal_confidence_scaling': params.get('signal_confidence_scaling', 2.0)
        }
        
        return genetic_params
    
    async def _calculate_genetic_base_size(self, symbol: str, genetic_params: Dict[str, float],
                                         market_data: pd.DataFrame, signal_strength: float) -> float:
        """Calculate base position size using genetic parameters.
        
        Args:
            symbol: Trading symbol
            genetic_params: Genetic algorithm evolved parameters
            market_data: Historical market data
            signal_strength: Signal strength (0-1)
            
        Returns:
            Base position size before adjustments
        """
        # Start with genetic base allocation
        base_allocation = genetic_params['base_allocation']
        
        # Apply Kelly Criterion with genetic parameters
        kelly_size = self._calculate_genetic_kelly_size(genetic_params, market_data)
        
        # Apply volatility targeting
        volatility_size = self._calculate_volatility_adjusted_size(
            base_allocation, genetic_params, market_data
        )
        
        # Apply momentum scaling
        momentum_size = self._calculate_momentum_adjusted_size(
            base_allocation, genetic_params, market_data
        )
        
        # Combine methods using genetic weights
        combined_size = (
            kelly_size * 0.4 +
            volatility_size * 0.3 +
            momentum_size * 0.3
        )
        
        # Scale by signal strength
        signal_adjusted_size = combined_size * signal_strength
        
        # Apply genetic signal confidence scaling
        confidence_scaling = max(
            genetic_params['signal_confidence_min'],
            signal_strength ** genetic_params['signal_confidence_scaling']
        )
        
        final_size = signal_adjusted_size * confidence_scaling
        
        return max(0.0, min(final_size, self.max_position_per_asset))
    
    def _calculate_genetic_kelly_size(self, genetic_params: Dict[str, float],
                                    market_data: pd.DataFrame) -> float:
        """Calculate Kelly Criterion size with genetic parameters.
        
        Args:
            genetic_params: Genetic parameters
            market_data: Historical market data
            
        Returns:
            Kelly optimal position size
        """
        try:
            # Use genetic estimates or calculate from data
            win_rate = genetic_params['win_rate_estimate']
            avg_win_loss_ratio = genetic_params['avg_win_loss_ratio']
            kelly_multiplier = genetic_params['kelly_multiplier']
            
            # Kelly formula: f = (bp - q) / b
            # where p = win probability, q = loss probability, b = win/loss ratio
            if avg_win_loss_ratio > 0:
                kelly_fraction = (
                    (avg_win_loss_ratio * win_rate - (1 - win_rate)) / avg_win_loss_ratio
                )
            else:
                kelly_fraction = 0.0
            
            # Apply genetic multiplier and safety factor
            kelly_size = max(0.0, kelly_fraction) * kelly_multiplier * self.kelly_safety_factor
            
            return min(kelly_size, self.max_position_per_asset)
            
        except Exception as e:
            self.logger.warning(f"Error calculating Kelly size: {e}")
            return genetic_params['base_allocation'] * 0.5
    
    def _calculate_volatility_adjusted_size(self, base_size: float, 
                                          genetic_params: Dict[str, float],
                                          market_data: pd.DataFrame) -> float:
        """Calculate volatility-adjusted position size.
        
        Args:
            base_size: Base position size
            genetic_params: Genetic parameters
            market_data: Historical market data
            
        Returns:
            Volatility-adjusted position size
        """
        try:
            if len(market_data) < self.volatility_lookback:
                return base_size
            
            # Calculate recent volatility
            returns = market_data['close'].pct_change().dropna()
            recent_volatility = returns.tail(self.volatility_lookback).std() * np.sqrt(365 * 24)
            
            # Target volatility from genetic parameters
            target_volatility = genetic_params['volatility_target']
            volatility_multiplier = genetic_params['volatility_multiplier']
            
            # Scale position inverse to volatility
            if recent_volatility > 0:
                volatility_scaling = (target_volatility / recent_volatility) * volatility_multiplier
                volatility_scaling = np.clip(volatility_scaling, 0.1, 3.0)  # Reasonable bounds
            else:
                volatility_scaling = 1.0
            
            adjusted_size = base_size * volatility_scaling
            
            return min(adjusted_size, self.max_position_per_asset)
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility adjustment: {e}")
            return base_size
    
    def _calculate_momentum_adjusted_size(self, base_size: float,
                                        genetic_params: Dict[str, float],
                                        market_data: pd.DataFrame) -> float:
        """Calculate momentum-adjusted position size.
        
        Args:
            base_size: Base position size
            genetic_params: Genetic parameters
            market_data: Historical market data
            
        Returns:
            Momentum-adjusted position size
        """
        try:
            if len(market_data) < 20:
                return base_size
            
            # Calculate momentum indicators
            close_prices = market_data['close']
            
            # Short-term momentum (5-day)
            short_momentum = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(close_prices) >= 6 else 0
            
            # Medium-term momentum (20-day)
            medium_momentum = (close_prices.iloc[-1] / close_prices.iloc[-21] - 1) if len(close_prices) >= 21 else 0
            
            # Combine momentum signals
            momentum_strength = (
                short_momentum * 0.6 + 
                medium_momentum * 0.4
            )
            
            # Apply genetic momentum parameters
            momentum_multiplier = genetic_params['momentum_multiplier']
            trend_weight = genetic_params['trend_strength_weight']
            
            # Scale position based on momentum strength
            momentum_scaling = 1.0 + (momentum_strength * momentum_multiplier * trend_weight)
            momentum_scaling = np.clip(momentum_scaling, 0.2, 2.0)  # Reasonable bounds
            
            adjusted_size = base_size * momentum_scaling
            
            return min(adjusted_size, self.max_position_per_asset)
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum adjustment: {e}")
            return base_size
    
    def _apply_risk_scaling(self, base_size: float, symbol: str,
                          genetic_params: Dict[str, float],
                          market_data: pd.DataFrame) -> float:
        """Apply risk-based scaling to position size.
        
        Args:
            base_size: Base position size
            symbol: Trading symbol
            genetic_params: Genetic parameters
            market_data: Historical market data
            
        Returns:
            Risk-scaled position size
        """
        try:
            # Calculate Value at Risk (simplified)
            if len(market_data) < 30:
                return base_size
            
            returns = market_data['close'].pct_change().dropna()
            
            # 95% VaR
            var_95 = returns.quantile(0.05)  # 5th percentile (negative)
            
            # Maximum drawdown tolerance from genetic parameters
            max_dd_tolerance = genetic_params['max_drawdown_tolerance']
            
            # Scale position to keep expected worst-case loss within tolerance
            if var_95 < 0:  # VaR should be negative
                risk_scaling = min(1.0, abs(max_dd_tolerance / var_95))
                risk_scaling = max(0.1, risk_scaling)  # Minimum scaling
            else:
                risk_scaling = 1.0
            
            risk_scaled_size = base_size * risk_scaling
            
            return min(risk_scaled_size, self.max_position_per_asset)
            
        except Exception as e:
            self.logger.warning(f"Error applying risk scaling: {e}")
            return base_size
    
    def _apply_correlation_constraints(self, base_size: float, symbol: str,
                                     genetic_params: Dict[str, float]) -> float:
        """Apply correlation constraints to position size.
        
        Args:
            base_size: Base position size
            symbol: Trading symbol
            genetic_params: Genetic parameters
            
        Returns:
            Correlation-adjusted position size
        """
        try:
            if not self.correlation_matrix or symbol not in self.correlation_matrix.index:
                return base_size
            
            # Calculate correlation penalty
            symbol_correlations = self.correlation_matrix.loc[symbol]
            
            # Find maximum correlation with existing positions
            max_correlation = 0.0
            for existing_symbol, position_size in self.current_positions.items():
                if existing_symbol != symbol and position_size > 0:
                    if existing_symbol in symbol_correlations.index:
                        correlation = abs(symbol_correlations[existing_symbol])
                        max_correlation = max(max_correlation, correlation)
            
            # Apply correlation penalty from genetic parameters
            if max_correlation > self.max_correlation:
                correlation_penalty = genetic_params['correlation_penalty']
                penalty_scaling = 1.0 - ((max_correlation - self.max_correlation) * correlation_penalty)
                penalty_scaling = max(0.1, penalty_scaling)  # Minimum scaling
            else:
                penalty_scaling = 1.0
            
            correlation_adjusted_size = base_size * penalty_scaling
            
            return min(correlation_adjusted_size, self.max_position_per_asset)
            
        except Exception as e:
            self.logger.warning(f"Error applying correlation constraints: {e}")
            return base_size
    
    def _apply_position_limits(self, size: float, symbol: str) -> float:
        """Apply final position limits and portfolio constraints.
        
        Args:
            size: Calculated position size
            symbol: Trading symbol
            
        Returns:
            Final position size within all limits
        """
        # Apply individual position limit
        size = min(size, self.max_position_per_asset)
        
        # Apply minimum position size
        if size < self.min_position_size:
            size = 0.0
        
        # Check total portfolio exposure
        current_total_exposure = sum(self.current_positions.values())
        available_exposure = self.max_total_exposure - current_total_exposure
        
        if available_exposure <= 0:
            return 0.0
        
        # Limit to available exposure
        size = min(size, available_exposure)
        
        return size
    
    def _calculate_risk_metrics(self, symbol: str, position_size: float,
                              genetic_params: Dict[str, float],
                              market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for the position.
        
        Args:
            symbol: Trading symbol
            position_size: Final position size
            genetic_params: Genetic parameters
            market_data: Historical market data
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if len(market_data) < 20:
                return {'insufficient_data': True}
            
            returns = market_data['close'].pct_change().dropna()
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(365 * 24)  # Annualized
            var_95 = returns.quantile(0.05)
            max_expected_loss = abs(var_95) * position_size
            
            # Sharpe estimate (simplified)
            sharpe_estimate = returns.mean() / returns.std() * np.sqrt(365 * 24) if returns.std() > 0 else 0
            
            return {
                'position_size': position_size,
                'annualized_volatility': volatility,
                'var_95_percent': var_95,
                'max_expected_loss': max_expected_loss,
                'sharpe_estimate': sharpe_estimate,
                'kelly_multiplier': genetic_params['kelly_multiplier'],
                'volatility_multiplier': genetic_params['volatility_multiplier']
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def _create_safe_fallback_result(self, symbol: str) -> PositionSizeResult:
        """Create a safe fallback result for error cases.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Safe fallback position sizing result
        """
        return PositionSizeResult(
            symbol=symbol,
            target_size=0.01,  # Small safe position
            max_size=self.max_position_per_asset,
            raw_size=0.01,
            scaling_factor=1.0,
            method_used=PositionSizeMethod.FIXED,
            risk_metrics={'fallback': True},
            correlation_adjustment=1.0,
            volatility_adjustment=1.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def update_portfolio_state(self, positions: Dict[str, float],
                                   correlation_matrix: Optional[pd.DataFrame] = None) -> None:
        """Update current portfolio state for position sizing calculations.
        
        Args:
            positions: Current positions {symbol: size}
            correlation_matrix: Updated correlation matrix
        """
        self.current_positions = positions.copy()
        
        if correlation_matrix is not None:
            self.correlation_matrix = correlation_matrix
        
        self.logger.debug(f"Portfolio state updated: {len(positions)} positions, "
                         f"total exposure: {sum(positions.values()):.3f}")
    
    async def calculate_portfolio_allocation(self, symbols: List[str], 
                                           seeds: Dict[str, BaseSeed],
                                           market_data: Dict[str, pd.DataFrame],
                                           signal_strengths: Dict[str, float]) -> Dict[str, PositionSizeResult]:
        """Calculate position sizes for entire portfolio.
        
        Args:
            symbols: List of trading symbols
            seeds: Dictionary mapping symbols to genetic seeds
            market_data: Dictionary mapping symbols to market data
            signal_strengths: Dictionary mapping symbols to signal strengths
            
        Returns:
            Dictionary mapping symbols to position sizing results
        """
        results = {}
        
        self.logger.info(f"Calculating portfolio allocation for {len(symbols)} symbols")
        
        # Calculate individual position sizes
        for symbol in symbols:
            if symbol in seeds and symbol in market_data:
                seed = seeds[symbol]
                data = market_data[symbol]
                signal_strength = signal_strengths.get(symbol, 1.0)
                
                try:
                    result = await self.calculate_position_size(
                        symbol, seed, data, signal_strength
                    )
                    results[symbol] = result
                    
                except Exception as e:
                    self.logger.error(f"Error calculating position for {symbol}: {e}")
                    results[symbol] = self._create_safe_fallback_result(symbol)
        
        # Normalize to ensure total exposure doesn't exceed limits
        total_exposure = sum(result.target_size for result in results.values())
        
        if total_exposure > self.max_total_exposure:
            scaling_factor = self.max_total_exposure / total_exposure
            self.logger.info(f"Scaling portfolio down by {scaling_factor:.3f} to meet exposure limits")
            
            for result in results.values():
                result.target_size *= scaling_factor
                result.scaling_factor *= scaling_factor
        
        self.logger.info(f"Portfolio allocation complete: {len(results)} positions, "
                        f"total exposure: {sum(r.target_size for r in results.values()):.3f}")
        
        return results
    
    def get_position_sizing_stats(self) -> Dict[str, Any]:
        """Get position sizing statistics.
        
        Returns:
            Dictionary with position sizing statistics
        """
        current_exposure = sum(self.current_positions.values())
        
        return {
            'max_position_per_asset': self.max_position_per_asset,
            'max_total_exposure': self.max_total_exposure,
            'current_total_exposure': current_exposure,
            'available_exposure': self.max_total_exposure - current_exposure,
            'active_positions': len([p for p in self.current_positions.values() if p > 0]),
            'max_correlation_threshold': self.max_correlation,
            'kelly_safety_factor': self.kelly_safety_factor,
            'min_position_size': self.min_position_size
        }


async def test_genetic_position_sizer():
    """Test function for genetic position sizer."""
    
    print("=== Genetic Position Sizer Test ===")
    
    # Create position sizer
    sizer = GeneticPositionSizer()
    
    # Create test genetic seed
    from src.strategy.genetic_seeds.ema_crossover_seed import EMACrossoverSeed
    from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType
    
    genes = SeedGenes(
        seed_id="test_position_seed",
        seed_type=SeedType.MOMENTUM,
        parameters={
            'position_size': 0.15,
            'kelly_multiplier': 0.3,
            'volatility_multiplier': 1.2,
            'momentum_multiplier': 0.8,
            'max_drawdown_tolerance': 0.12
        }
    )
    
    seed = EMACrossoverSeed(genes)
    
    # Create test market data
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    market_data = pd.DataFrame({
        'close': np.random.uniform(50000, 55000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Add some trend to make it realistic
    trend = np.linspace(0, 0.1, 100)
    market_data['close'] = market_data['close'] * (1 + trend)
    
    try:
        # Test single position calculation
        print("Calculating position size for BTC-USD...")
        result = await sizer.calculate_position_size('BTC-USD', seed, market_data, 0.8)
        
        print(f"✅ Position Size Result:")
        print(f"  - Symbol: {result.symbol}")
        print(f"  - Target size: {result.target_size:.4f}")
        print(f"  - Raw size: {result.raw_size:.4f}")
        print(f"  - Scaling factor: {result.scaling_factor:.3f}")
        print(f"  - Method: {result.method_used}")
        print(f"  - Risk metrics: {result.risk_metrics}")
        
        # Test portfolio allocation
        print("\nTesting portfolio allocation...")
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        seeds_dict = {symbol: seed for symbol in symbols}
        market_data_dict = {symbol: market_data for symbol in symbols}
        signal_strengths = {'BTC-USD': 0.8, 'ETH-USD': 0.6, 'SOL-USD': 0.9}
        
        portfolio_results = await sizer.calculate_portfolio_allocation(
            symbols, seeds_dict, market_data_dict, signal_strengths
        )
        
        print(f"✅ Portfolio Allocation:")
        total_exposure = 0
        for symbol, result in portfolio_results.items():
            total_exposure += result.target_size
            print(f"  - {symbol}: {result.target_size:.4f} "
                  f"(signal: {signal_strengths[symbol]:.1f})")
        
        print(f"  - Total Exposure: {total_exposure:.4f}")
        
        # Show position sizing stats
        stats = sizer.get_position_sizing_stats()
        print(f"\nPosition Sizing Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    print(f"\n✅ Genetic Position Sizer test completed successfully!")


if __name__ == "__main__":
    """Test the genetic position sizer."""
    
    import asyncio
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_genetic_position_sizer())