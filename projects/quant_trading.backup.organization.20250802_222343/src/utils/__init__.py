"""
Utilities Package

Provides utility functions and compatibility layers:
- Pandas compatibility utilities for version migration
- Technical analysis fixes and enhancements
- Common helper functions and data processing utilities
"""

from .pandas_compatibility import (
    safe_fillna, safe_fillna_false, safe_fillna_zero,
    safe_forward_fill, safe_backward_fill, check_pandas_compatibility
)
from .technical_analysis_fix import TechnicalAnalysisManager

__all__ = [
    'safe_fillna',
    'safe_fillna_false', 
    'safe_fillna_zero',
    'safe_forward_fill',
    'safe_backward_fill',
    'check_pandas_compatibility',
    'TechnicalAnalysisManager'
]