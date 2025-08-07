"""
Analysis Module - Market Structure and Cross-Asset Analysis

This module provides advanced market analysis capabilities including
cross-asset correlation analysis for regime detection and signal enhancement.

Key Components:
- FilteredAssetCorrelationEngine: Cross-asset correlation analysis
"""

from .correlation_engine import FilteredAssetCorrelationEngine, CorrelationMetrics

__all__ = [
    'FilteredAssetCorrelationEngine',
    'CorrelationMetrics'
]

# Version info
__version__ = "1.0.0"