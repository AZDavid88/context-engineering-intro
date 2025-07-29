"""
Backtesting Engine Package

Provides comprehensive backtesting capabilities using VectorBT:
- Strategy signal conversion and validation
- Performance analysis and metrics calculation
- Multi-asset portfolio backtesting
- Risk-adjusted performance evaluation
"""

from .vectorbt_engine import VectorBTEngine
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .strategy_converter import StrategyConverter, SignalConversionResult

__all__ = [
    'VectorBTEngine',
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'StrategyConverter', 
    'SignalConversionResult'
]