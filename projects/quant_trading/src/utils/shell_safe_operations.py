"""
Shell-Safe Operations Utility

This module provides safe alternatives to shell operations that cause escaping issues,
particularly the \! problem when using != in Bash commands through Claude Code.

The \! issue occurs when using != comparisons in shell commands passed through tools.
This utility provides safe alternatives.
"""

import pandas as pd
from typing import Any, Union


def safe_not_equal(series: pd.Series, value: Any) -> pd.Series:
    """
    Safe alternative to series != value that avoids shell escaping issues.
    
    Args:
        series: Pandas Series to compare
        value: Value to compare against
        
    Returns:
        Boolean Series indicating where values are not equal
    """
    return series.ne(value)


def safe_not_equal_mask(data: Union[pd.Series, pd.DataFrame], value: Any) -> Union[pd.Series, pd.DataFrame]:
    """
    Create a boolean mask for not-equal comparisons without shell escaping issues.
    
    Args:
        data: Data to compare (Series or DataFrame)  
        value: Value to compare against
        
    Returns:
        Boolean mask where data != value
    """
    return ~(data == value)


def count_non_zero_safe(series: pd.Series) -> int:
    """
    Count non-zero values safely without shell escaping issues.
    
    Args:
        series: Series to count non-zero values in
        
    Returns:
        Number of non-zero values
    """
    return (series.abs() > 1e-10).sum()


def count_non_equal_safe(series: pd.Series, value: Any) -> int:
    """
    Count values not equal to specified value safely.
    
    Args:
        series: Series to analyze
        value: Value to compare against
        
    Returns:
        Count of values not equal to the specified value
    """
    return safe_not_equal(series, value).sum()


def filter_non_zero_safe(series: pd.Series) -> pd.Series:
    """
    Filter out zero values safely without shell escaping issues.
    
    Args:
        series: Series to filter
        
    Returns:
        Series with only non-zero values
    """
    return series[series.abs() > 1e-10]


# Usage examples for shell-safe operations:
"""
# Instead of this (causes \! escaping issues):
signals[signals != 0]

# Use this:
signals[safe_not_equal(signals, 0)]

# Or this:
signals[safe_not_equal_mask(signals, 0)]

# Instead of len(signals[signals != 0]):
count_non_equal_safe(signals, 0)

# Instead of len(signals[signals != 0.0]):
count_non_zero_safe(signals)
"""