"""
Tests Package for Quantitative Trading System

This package contains comprehensive test suites for the quantitative trading
system, including unit tests, integration tests, and test utilities.

Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- comprehensive/: End-to-end system validation tests
- utils/: Test utilities and fixtures
- system/: System-level and performance tests
"""

# Test utilities exports for easy access
from .utils import (
    create_test_market_data,
    create_test_ohlcv_data,
    create_test_genetic_data,
    create_multi_asset_test_data,
    create_validation_test_data,
    get_bull_market_data,
    get_bear_market_data,
    get_sideways_market_data
)

__version__ = "1.0.0"
__all__ = [
    "create_test_market_data",
    "create_test_ohlcv_data", 
    "create_test_genetic_data",
    "create_multi_asset_test_data",
    "create_validation_test_data",
    "get_bull_market_data",
    "get_bear_market_data",
    "get_sideways_market_data"
]