"""
Test Market Data Fixtures - Mock Data Generation for Testing

This module contains mock data generation utilities that were previously
contaminating production files. All synthetic data generation is now
isolated to test utilities only.

Integration:
- Unit tests and integration tests
- Development and testing environments only
- Never used in production code paths
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any


def create_test_market_data(days: int = 30, 
                           base_price: float = 30000.0,
                           seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic OHLCV data for testing purposes only.
    
    Args:
        days: Number of days of data to generate
        base_price: Starting price (default BTC-like)
        seed: Random seed for reproducible tests
        
    Returns:
        DataFrame with synthetic OHLCV data for testing
    """
    
    # Reproducible for testing
    np.random.seed(seed)
    dates = pd.date_range(
        end=datetime.now(timezone.utc), 
        periods=days * 24,  # Hourly data
        freq='h'
    )
    
    # Generate realistic price movement
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% mean, 2% std
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Create OHLCV
    data = {
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(prices))),
        'high': prices * (1 + np.random.uniform(0, 0.005, len(prices))),
        'low': prices * (1 + np.random.uniform(-0.005, 0, len(prices))),
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, len(prices))
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def create_test_ohlcv_data(symbol: str = "BTC",
                          days: int = 30,
                          timeframe: str = "1h",
                          seed: int = 42) -> pd.DataFrame:
    """
    Create test OHLCV data with specific symbol and timeframe.
    
    Args:
        symbol: Asset symbol (BTC, ETH, etc.)
        days: Number of days of data
        timeframe: Timeframe (1h, 15m, 1d)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with test OHLCV data
    """
    
    base_prices = {
        "BTC": 30000.0,
        "ETH": 2000.0,
        "SOL": 100.0,
        "DOGE": 0.10
    }
    
    base_price = base_prices.get(symbol, 1000.0)
    
    # Adjust frequency based on timeframe
    freq_map = {
        "1m": "min",
        "5m": "5min", 
        "15m": "15min",
        "1h": "h",
        "1d": "D"
    }
    
    frequency = freq_map.get(timeframe, "h")
    periods = days * 24 if frequency == "h" else days  # Adjust periods
    
    return create_test_market_data(
        days=days if frequency == "D" else days * 24,
        base_price=base_price,
        seed=seed
    )


def create_test_genetic_data(strategy_count: int = 10,
                           fitness_range: tuple = (0.5, 2.5),
                           seed: int = 42) -> List[Dict[str, Any]]:
    """
    Create test data for genetic evolution testing.
    
    Args:
        strategy_count: Number of mock strategies to create
        fitness_range: Min and max fitness scores
        seed: Random seed for reproducibility
        
    Returns:
        List of mock strategy data for testing
    """
    
    np.random.seed(seed)
    
    strategies = []
    for i in range(strategy_count):
        fitness = np.random.uniform(fitness_range[0], fitness_range[1])
        strategies.append({
            "config_name": f"test_strategy_{i}",
            "fitness": fitness,
            "validation_score": fitness * 0.9,
            "strategy_type": np.random.choice(["momentum", "mean_reversion", "breakout"]),
            "parameters": {
                "lookback_period": np.random.randint(5, 50),
                "threshold": np.random.uniform(0.01, 0.1)
            }
        })
    
    return strategies


def create_multi_asset_test_data(symbols: List[str] = ["BTC", "ETH", "SOL"],
                                days: int = 30,
                                seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Create test data for multiple assets.
    
    Args:
        symbols: List of asset symbols
        days: Number of days of data per asset
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    
    data = {}
    for i, symbol in enumerate(symbols):
        # Use different seed for each symbol to create variation
        data[symbol] = create_test_ohlcv_data(
            symbol=symbol,
            days=days,
            seed=seed + i
        )
    
    return data


def create_validation_test_data(validation_periods: int = 3,
                               days_per_period: int = 30,
                               seed: int = 42) -> List[pd.DataFrame]:
    """
    Create test data for validation pipeline testing.
    
    Args:
        validation_periods: Number of validation periods
        days_per_period: Days of data per period
        seed: Random seed for reproducibility
        
    Returns:
        List of DataFrames for different validation periods
    """
    
    periods = []
    for i in range(validation_periods):
        period_data = create_test_market_data(
            days=days_per_period,
            seed=seed + i
        )
        periods.append(period_data)
    
    return periods


# Test data presets for common scenarios
def get_bull_market_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Get test data simulating bull market conditions."""
    np.random.seed(seed)
    
    # Generate upward trending data
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=days * 24, freq='h')
    trend = np.linspace(30000, 35000, len(dates))  # 16% increase
    noise = np.random.normal(0, 200, len(dates))   # Reduced volatility
    prices = trend + noise
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.0005, 0.0005, len(prices))),
        'high': prices * (1 + np.random.uniform(0, 0.003, len(prices))),
        'low': prices * (1 + np.random.uniform(-0.003, 0, len(prices))),
        'close': prices,
        'volume': np.random.uniform(500000, 2000000, len(prices))
    }, index=dates)
    
    # Ensure proper OHLCV constraints
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def get_bear_market_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Get test data simulating bear market conditions."""
    np.random.seed(seed)
    
    # Generate downward trending data
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=days * 24, freq='h')
    trend = np.linspace(30000, 25000, len(dates))  # 16% decrease
    noise = np.random.normal(0, 300, len(dates))   # Higher volatility
    prices = trend + noise
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(prices))),
        'high': prices * (1 + np.random.uniform(0, 0.002, len(prices))),
        'low': prices * (1 + np.random.uniform(-0.008, 0, len(prices))),
        'close': prices,
        'volume': np.random.uniform(800000, 3000000, len(prices))  # Higher volume
    }, index=dates)
    
    # Ensure proper OHLCV constraints
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def get_sideways_market_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Get test data simulating sideways market conditions."""
    np.random.seed(seed)
    
    # Generate sideways trending data
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=days * 24, freq='h')
    base_price = 30000.0
    noise = np.random.normal(0, 400, len(dates))  # Higher volatility, no trend
    prices = np.full(len(dates), base_price) + noise
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(prices))),
        'high': prices * (1 + np.random.uniform(0, 0.005, len(prices))),
        'low': prices * (1 + np.random.uniform(-0.005, 0, len(prices))),
        'close': prices,
        'volume': np.random.uniform(600000, 1500000, len(prices))
    }, index=dates)
    
    # Ensure proper OHLCV constraints
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df