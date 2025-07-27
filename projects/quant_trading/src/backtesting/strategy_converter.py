



from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
import logging
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# VectorBT imports (from comprehensive research)
import vectorbt as vbt

from src.strategy.genetic_seeds.base_seed import BaseSeed
from src.strategy.ast_strategy import TradingStrategy
from src.config.settings import get_settings, Settings



from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna

    
    
    
    


    
    


    
        
        
    
        
            
            
        
            
            
            
            
            
            
            
            
            
        
    
        
            
        
        
        
    
        
            
        
                
                
        
        
        
    
        
            
            
            
            
            
            
            
            
    
        
            
        
        
            
                    
        
    
        
            
        
        
        
        
    
        
            
            
            
            
            
            
            
            
            
    
        
            
        
        
            
            
        
            
            
        
    
        
            
        
        
        
    
        
            
            
            
    
        
            
        
        
                
                    
        
        
    
        
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
"""
Strategy Converter - AST to VectorBT Bridge
This module provides the critical bridge between genetic AST strategies and 
VectorBT backtesting engine, enabling seamless conversion of evolved genetic
strategies into executable trading signals for performance evaluation.
This addresses GAP 4 from the PRP: Missing Strategy Conversion Bridge.
Key Features:
- Convert genetic seed signals to VectorBT-compatible boolean arrays
- Multi-asset signal generation and coordination  
- Performance optimization for 1000+ strategy populations
- Signal integrity validation and error handling
- Position sizing integration with genetic parameters
"""
class SignalConversionResult(BaseModel):
    """Result of signal conversion process."""
    # Signal arrays for VectorBT
    entries: pd.Series = Field(..., description="Entry signals (boolean array)")
    exits: pd.Series = Field(..., description="Exit signals (boolean array)")
    size: pd.Series = Field(..., description="Position sizes")
    # Metadata
    strategy_id: str = Field(..., description="Source strategy identifier")
    asset_symbol: str = Field(..., description="Asset symbol")
    conversion_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Validation metrics
    signal_count: int = Field(..., ge=0, description="Total number of signals")
    entry_count: int = Field(..., ge=0, description="Number of entry signals")
    exit_count: int = Field(..., ge=0, description="Number of exit signals")
    signal_integrity_score: float = Field(..., ge=0.0, le=1.0, description="Signal quality score")
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            pd.Series: lambda v: v.tolist()
        }
class MultiAssetSignals(BaseModel):
    """Container for multi-asset signal conversion results."""
    signals_by_asset: Dict[str, SignalConversionResult] = Field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    total_signals: int = Field(default=0, ge=0)
    portfolio_allocation: Dict[str, float] = Field(default_factory=dict)
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
class StrategyConverter:
    """Converts genetic strategies to VectorBT-compatible signals."""
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize strategy converter.
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.Converter")
        # Performance tracking
        self.conversion_stats = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'avg_conversion_time': 0.0
        }
    def convert_seed_to_signals(self, seed: BaseSeed, data: pd.DataFrame, 
                              asset_symbol: str = "BTC") -> SignalConversionResult:
        """Convert a genetic seed to VectorBT-compatible signals.
        Args:
            seed: Genetic seed instance
            data: OHLCV market data
            asset_symbol: Asset symbol for identification
        Returns:
            Signal conversion result with VectorBT arrays
        Raises:
            ValueError: If conversion fails or signals are invalid
        """
        import time
        start_time = time.time()
        try:
            # Generate raw signals from genetic seed
            raw_signals = seed.generate_signals(data)
            # Validate raw signals
            if not self._validate_raw_signals(raw_signals, data):
                raise ValueError(f"Invalid raw signals from seed {seed.seed_name}")
            # Convert to entry/exit boolean arrays
            entries, exits = self._convert_to_entry_exit_arrays(raw_signals)
            # Calculate position sizes based on genetic parameters
            position_sizes = self._calculate_position_sizes(seed, data, raw_signals)
            # Validate signal integrity
            integrity_score = self._calculate_signal_integrity(entries, exits, position_sizes)
            # Create result object
            result = SignalConversionResult(
                entries=entries,
                exits=exits,
                size=position_sizes,
                strategy_id=seed.genes.seed_id,
                asset_symbol=asset_symbol,
                signal_count=len(raw_signals[raw_signals != 0]),
                entry_count=int(entries.sum()),
                exit_count=int(exits.sum()),
                signal_integrity_score=integrity_score
            )
            # Update stats
            self.conversion_stats['successful_conversions'] += 1
            conversion_time = time.time() - start_time
            self.conversion_stats['avg_conversion_time'] = (
                (self.conversion_stats['avg_conversion_time'] * self.conversion_stats['total_conversions'] + 
                 conversion_time) / (self.conversion_stats['total_conversions'] + 1)
            )
            self.logger.debug(f"Converted {seed.seed_name} signals: {result.entry_count} entries, "
                            f"{result.exit_count} exits, integrity={integrity_score:.3f}")
            return result
        except Exception as e:
            self.conversion_stats['failed_conversions'] += 1
            self.logger.error(f"Failed to convert seed {seed.seed_name}: {e}")
            raise ValueError(f"Signal conversion failed: {e}") from e
        finally:
            self.conversion_stats['total_conversions'] += 1
    def convert_strategy_to_signals(self, strategy: TradingStrategy, data: pd.DataFrame,
                                  asset_symbol: str = "BTC") -> SignalConversionResult:
        """Convert a complete trading strategy to VectorBT signals.
        Args:
            strategy: Trading strategy with AST representation
            data: OHLCV market data  
            asset_symbol: Asset symbol for identification
        Returns:
            Signal conversion result
        """
        # For now, extract signals from strategy's tree representation
        # This would integrate with the actual AST evaluation from ast_strategy.py
        # Placeholder: Generate signals based on strategy indicators
        signals = self._extract_signals_from_strategy(strategy, data)
        # Convert to VectorBT format
        entries, exits = self._convert_to_entry_exit_arrays(signals)
        position_sizes = pd.Series(strategy.genes.position_size, index=data.index)
        return SignalConversionResult(
            entries=entries,
            exits=exits,
            size=position_sizes,
            strategy_id=strategy.strategy_id,
            asset_symbol=asset_symbol,
            signal_count=len(signals[signals != 0]),
            entry_count=int(entries.sum()),
            exit_count=int(exits.sum()),
            signal_integrity_score=self._calculate_signal_integrity(entries, exits, position_sizes)
        )
    def convert_multi_asset_signals(self, seed: BaseSeed, 
                                  data_by_asset: Dict[str, pd.DataFrame]) -> MultiAssetSignals:
        """Convert genetic seed to signals across multiple assets.
        Args:
            seed: Genetic seed instance
            data_by_asset: Dictionary mapping asset symbols to OHLCV data
        Returns:
            Multi-asset signal container
        """
        signals_by_asset = {}
        all_signals = []
        # Convert signals for each asset
        for asset_symbol, asset_data in data_by_asset.items():
            try:
                conversion_result = self.convert_seed_to_signals(seed, asset_data, asset_symbol)
                signals_by_asset[asset_symbol] = conversion_result
                # Collect for correlation analysis
                all_signals.append(conversion_result.entries.astype(int) - conversion_result.exits.astype(int))
            except Exception as e:
                self.logger.warning(f"Failed to convert signals for {asset_symbol}: {e}")
                continue
        # Calculate cross-asset correlation matrix
        correlation_matrix = None
        if len(all_signals) > 1:
            signal_df = pd.DataFrame({asset: signals for asset, signals in 
                                    zip(data_by_asset.keys(), all_signals)})
            correlation_matrix = signal_df.corr()
        # Calculate portfolio allocation based on signal strength
        portfolio_allocation = self._calculate_portfolio_allocation(signals_by_asset)
        return MultiAssetSignals(
            signals_by_asset=signals_by_asset,
            correlation_matrix=correlation_matrix,
            total_signals=sum(result.signal_count for result in signals_by_asset.values()),
            portfolio_allocation=portfolio_allocation
        )
    def _validate_raw_signals(self, signals: pd.Series, data: pd.DataFrame) -> bool:
        """Validate raw signals from genetic seed.
        Args:
            signals: Raw signal series
            data: Source market data
        Returns:
            True if signals are valid
        """
        try:
            # Check basic format
            if not isinstance(signals, pd.Series):
                return False
            # Check length matches data
            if len(signals) != len(data):
                return False
            # Check for valid signal range
            if signals.min() < -1.0 or signals.max() > 1.0:
                return False
            # Check for excessive NaN values
            if signals.isna().sum() > len(signals) * 0.1:  # Max 10% NaN
                return False
            # Check for reasonable signal frequency (not all zeros)
            non_zero_signals = (signals != 0).sum()
            if non_zero_signals < len(signals) * 0.001:  # Min 0.1% signals
                return False
            # Check for signal sanity (not all same value)
            if signals.nunique() <= 1:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return False
    def _convert_to_entry_exit_arrays(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Convert continuous signals to boolean entry/exit arrays.
        Args:
            signals: Continuous signal series (-1 to 1)
        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        # Initialize arrays
        entries = pd.Series(False, index=signals.index)
        exits = pd.Series(False, index=signals.index)
        # Track position state
        position = 0  # 0 = no position, 1 = long, -1 = short
        for i, signal in enumerate(signals):
            if abs(signal) < 0.1:  # No significant signal
                continue
            if signal > 0.1:  # Buy signal
                if position <= 0:  # Can enter long
                    entries.iloc[i] = True
                    if position < 0:  # Exit short first
                        exits.iloc[i] = True
                    position = 1
            elif signal < -0.1:  # Sell signal
                if position >= 0:  # Can enter short
                    entries.iloc[i] = True  # VectorBT uses entries for both long and short
                    if position > 0:  # Exit long first
                        exits.iloc[i] = True
                    position = -1
        return entries, exits
    def _calculate_position_sizes(self, seed: BaseSeed, data: pd.DataFrame, 
                                signals: pd.Series) -> pd.Series:
        """Calculate position sizes based on genetic parameters and market conditions.
        Args:
            seed: Genetic seed with position sizing parameters
            data: Market data for volatility/risk calculations
            signals: Signal series for strength-based sizing
        Returns:
            Position size series
        """
        # Base position size from genetic parameters
        base_size = seed.genes.position_size
        # Initialize position sizes
        position_sizes = pd.Series(0.0, index=data.index)
        # Calculate dynamic position sizes for each signal
        for i, signal in enumerate(signals):
            if abs(signal) > 0.1:  # Significant signal
                # Use seed's position sizing method
                dynamic_size = seed.calculate_position_size(data.iloc[:i+1], signal)
                position_sizes.iloc[i] = dynamic_size
            else:
                position_sizes.iloc[i] = 0.0
        # Ensure sizes are within reasonable bounds
        max_size = self.settings.trading.max_position_size
        position_sizes = position_sizes.clip(0.0, max_size)
        return position_sizes
    def _calculate_signal_integrity(self, entries: pd.Series, exits: pd.Series, 
                                  sizes: pd.Series) -> float:
        """Calculate signal integrity score (0-1).
        Args:
            entries: Entry signal array
            exits: Exit signal array
            sizes: Position size array
        Returns:
            Integrity score (higher is better)
        """
        try:
            # Check for basic signal properties
            total_entries = entries.sum()
            total_exits = exits.sum()
            if total_entries == 0:
                return 0.0
            # Score components
            scores = []
            # 1. Entry/exit balance (should be roughly balanced)
            balance_score = 1.0 - abs(total_entries - total_exits) / max(total_entries, total_exits)
            scores.append(balance_score * 0.3)
            # 2. Signal frequency (not too sparse, not too frequent)
            signal_frequency = total_entries / len(entries)
            optimal_frequency = 0.05  # 5% of bars have signals
            frequency_score = 1.0 - abs(signal_frequency - optimal_frequency) / optimal_frequency
            frequency_score = max(0.0, min(1.0, frequency_score))
            scores.append(frequency_score * 0.3)
            # 3. Position sizing consistency
            non_zero_sizes = sizes[sizes > 0]
            if len(non_zero_sizes) > 0:
                size_cv = non_zero_sizes.std() / non_zero_sizes.mean()  # Coefficient of variation
                size_score = max(0.0, 1.0 - size_cv)  # Lower variation = better
                scores.append(size_score * 0.2)
            else:
                scores.append(0.0)
            # 4. No simultaneous entry/exit
            simultaneous = (entries & exits).sum()
            simultaneous_score = 1.0 - (simultaneous / max(total_entries, 1))
            scores.append(simultaneous_score * 0.2)
            return sum(scores)
        except Exception as e:
            self.logger.warning(f"Error calculating signal integrity: {e}")
            return 0.0
    def _extract_signals_from_strategy(self, strategy: TradingStrategy, 
                                     data: pd.DataFrame) -> pd.Series:
        """Extract signals from trading strategy AST representation.
        Args:
            strategy: Trading strategy with genetic tree
            data: Market data
        Returns:
            Signal series
        """
        # This would integrate with the AST evaluation from ast_strategy.py
        # For now, generate placeholder signals based on indicators used
        indicators_used = strategy.genes.indicators_used
        signals = pd.Series(0.0, index=data.index)
        # Simple signal generation based on strategy indicators
        if 'ema12' in indicators_used and 'ema26' in indicators_used:
            # EMA crossover signals
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            crossover_up = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1))
            crossover_down = (ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))
            signals[crossover_up] = 0.8
            signals[crossover_down] = -0.8
        elif 'rsi' in indicators_used:
            # RSI mean reversion signals
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            oversold = rsi < 30
            overbought = rsi > 70
            signals[oversold] = 0.6
            signals[overbought] = -0.6
        return safe_fillna_zero(signals)
    def _calculate_portfolio_allocation(self, signals_by_asset: Dict[str, SignalConversionResult]) -> Dict[str, float]:
        """Calculate portfolio allocation weights based on signal quality.
        Args:
            signals_by_asset: Signals for each asset
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        if not signals_by_asset:
            return {}
        # Calculate allocation based on signal integrity and frequency
        asset_scores = {}
        for asset, result in signals_by_asset.items():
            # Score based on signal integrity and activity
            score = (
                result.signal_integrity_score * 0.7 +
                min(1.0, result.signal_count / 100) * 0.3  # Normalize signal count
            )
            asset_scores[asset] = score
        # Normalize to sum to 1.0
        total_score = sum(asset_scores.values())
        if total_score > 0:
            return {asset: score / total_score for asset, score in asset_scores.items()}
        else:
            # Equal allocation if no valid scores
            equal_weight = 1.0 / len(signals_by_asset)
            return {asset: equal_weight for asset in signals_by_asset.keys()}
    def create_vectorbt_portfolio(self, conversion_result: SignalConversionResult,
                                data: pd.DataFrame, 
                                initial_cash: float = 10000.0) -> vbt.Portfolio:
        """Create VectorBT portfolio from conversion result.
        Args:
            conversion_result: Signal conversion result
            data: OHLCV market data
            initial_cash: Initial portfolio cash
        Returns:
            VectorBT Portfolio object for backtesting
        """
        try:
            # Create portfolio using VectorBT Portfolio.from_signals
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=conversion_result.entries,
                exits=conversion_result.exits,
                size=conversion_result.size,
                init_cash=initial_cash,
                fees=self.settings.trading.taker_fee,  # Use realistic fees
                slippage=self.settings.trading.slippage,
                freq='1H'  # Assuming hourly data
            )
            return portfolio
        except Exception as e:
            self.logger.error(f"Failed to create VectorBT portfolio: {e}")
            raise ValueError(f"Portfolio creation failed: {e}") from e
    def batch_convert_population(self, seeds: List[BaseSeed], 
                               data: pd.DataFrame) -> List[SignalConversionResult]:
        """Convert an entire population of genetic seeds efficiently.
        Args:
            seeds: List of genetic seeds to convert
            data: Market data
        Returns:
            List of conversion results
        """
        results = []
        self.logger.info(f"Converting population of {len(seeds)} seeds...")
        for i, seed in enumerate(seeds):
            try:
                result = self.convert_seed_to_signals(seed, data, f"ASSET_{i}")
                results.append(result)
                if (i + 1) % 100 == 0:  # Progress logging
                    self.logger.info(f"Converted {i + 1}/{len(seeds)} seeds")
            except Exception as e:
                self.logger.warning(f"Failed to convert seed {i}: {e}")
                continue
        success_rate = len(results) / len(seeds) if seeds else 0
        self.logger.info(f"Population conversion complete: {len(results)}/{len(seeds)} "
                        f"successful ({success_rate:.1%})")
        return results
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get conversion statistics.
        Returns:
            Dictionary with conversion statistics
        """
        success_rate = (self.conversion_stats['successful_conversions'] / 
                       max(self.conversion_stats['total_conversions'], 1))
        return {
            'total_conversions': self.conversion_stats['total_conversions'],
            'successful_conversions': self.conversion_stats['successful_conversions'],
            'failed_conversions': self.conversion_stats['failed_conversions'],
            'success_rate': success_rate,
            'avg_conversion_time_ms': self.conversion_stats['avg_conversion_time'] * 1000
        }
async def test_strategy_converter():
    """Test function for strategy converter."""
    print("=== Strategy Converter Test ===")
    # Import required components
    from src.strategy.genetic_seeds.ema_crossover_seed import EMACrossoverSeed
    from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType
    # Create test data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 5000, 1000)
    }, index=dates)
    # Create test seed
    genes = SeedGenes(
        seed_id="test_ema_seed",
        seed_type=SeedType.MOMENTUM,
        parameters={
            'fast_ema_period': 12.0,
            'slow_ema_period': 26.0,
            'momentum_threshold': 0.01,
            'signal_strength': 0.8,
            'trend_filter': 0.005
        }
    )
    seed = EMACrossoverSeed(genes)
    # Test conversion
    converter = StrategyConverter()
    print(f"Converting seed: {seed}")
    result = converter.convert_seed_to_signals(seed, test_data, "BTC")
    print(f"Conversion successful!")
    print(f"  - Entries: {result.entry_count}")
    print(f"  - Exits: {result.exit_count}")
    print(f"  - Signal integrity: {result.signal_integrity_score:.3f}")
    print(f"  - Total signals: {result.signal_count}")
    # Test VectorBT portfolio creation
    print(f"\nCreating VectorBT portfolio...")
    portfolio = converter.create_vectorbt_portfolio(result, test_data)
    print(f"Portfolio created successfully!")
    print(f"  - Total return: {portfolio.total_return():.2%}")
    print(f"  - Sharpe ratio: {portfolio.sharpe_ratio():.3f}")
    print(f"  - Max drawdown: {portfolio.max_drawdown():.2%}")
    # Test batch conversion
    print(f"\nTesting batch conversion...")
    seeds = [seed] * 5  # Small batch for testing
    batch_results = converter.batch_convert_population(seeds, test_data)
    print(f"Batch conversion complete: {len(batch_results)} results")
    # Show conversion stats
    stats = converter.get_conversion_stats()
    print(f"\nConversion Statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    print(f"\n✅ Strategy Converter test completed successfully!")
if __name__ == "__main__":
    """Test the strategy converter."""
    import asyncio
    import logging
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Run test
    asyncio.run(test_strategy_converter())