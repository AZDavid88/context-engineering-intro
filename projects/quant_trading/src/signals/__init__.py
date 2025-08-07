"""
Signals Module - Trading Signal Generation Framework

This module provides comprehensive trading signal generation capabilities
for the quantitative trading system, including correlation-based signals
and other market structure analysis.

Key Components:
- CorrelationSignalGenerator: Cross-asset correlation signal generation
- Integration with DataStorageInterface and existing data infrastructure
"""

from .correlation_signals import CorrelationSignalGenerator

__all__ = [
    'CorrelationSignalGenerator'
]

# Version info
__version__ = "1.0.0"