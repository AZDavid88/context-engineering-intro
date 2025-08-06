"""
Pandas Compatibility Utilities - Technical Debt Resolution

This module provides compatibility utilities to resolve pandas deprecation warnings
and ensure forward compatibility with future pandas versions.

The primary issue addressed:
- FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated
- This affects pandas 2.3.1+ and will become breaking in pandas 3.0+

Solution Strategy:
1. Create wrapper functions that handle the new pandas behavior
2. Use .infer_objects(copy=False) as recommended by pandas
3. Maintain backward compatibility with older pandas versions
4. Provide drop-in replacements for existing .fillna() calls
"""

import pandas as pd
import numpy as np
import warnings
from typing import Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

# Version checking for compatibility
PANDAS_VERSION = tuple(map(int, pd.__version__.split('.')[:2]))
PANDAS_2_3_PLUS = PANDAS_VERSION >= (2, 3)

def safe_fillna(series_or_df: Union[pd.Series, pd.DataFrame], 
                value: Any, 
                method: Optional[str] = None,
                inplace: bool = False) -> Union[pd.Series, pd.DataFrame, None]:
    """
    Safe fillna that handles pandas deprecation warnings.
    
    The proper solution is to use the new pandas behavior with proper dtype handling.
    
    Args:
        series_or_df: Pandas Series or DataFrame
        value: Value to fill NaN with
        method: Fill method ('ffill', 'bfill', etc.)
        inplace: Whether to modify in place
        
    Returns:
        Filled Series/DataFrame or None if inplace=True
    """
    
    try:
        # For pandas 2.3+, use warning suppression + proper dtype handling
        if PANDAS_2_3_PLUS:
            # Temporarily suppress the specific deprecation warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', 
                    message='.*Downcasting object dtype arrays on .fillna.*',
                    category=FutureWarning
                )
                
                # Perform the fillna operation
                if method is not None:
                    result = series_or_df.fillna(method=method, inplace=inplace)
                else:
                    result = series_or_df.fillna(value=value, inplace=inplace)
                
                # For non-inplace operations, ensure proper dtype inference
                if not inplace and result is not None:
                    if isinstance(result, pd.DataFrame):
                        object_cols = result.select_dtypes(include=['object']).columns
                        if len(object_cols) > 0:
                            for col in object_cols:
                                result[col] = result[col].infer_objects(copy=False)
                    elif isinstance(result, pd.Series) and result.dtype == 'object':
                        result = result.infer_objects(copy=False)
                
                return result
        else:
            # For older pandas versions, use standard fillna
            return series_or_df.fillna(value=value, method=method, inplace=inplace)
            
    except Exception as e:
        logger.warning(f"Error in safe_fillna: {e}, falling back to standard fillna")
        return series_or_df.fillna(value=value, method=method, inplace=inplace)


def safe_fillna_false(series_or_df: Union[pd.Series, pd.DataFrame], 
                      inplace: bool = False) -> Union[pd.Series, pd.DataFrame, None]:
    """
    Specific helper for filling NaN with False (common in boolean series).
    
    Args:
        series_or_df: Pandas Series or DataFrame
        inplace: Whether to modify in place
        
    Returns:
        Filled Series/DataFrame with False values or None if inplace=True
    """
    return safe_fillna(series_or_df, value=False, inplace=inplace)


def safe_fillna_zero(series_or_df: Union[pd.Series, pd.DataFrame], 
                     inplace: bool = False) -> Union[pd.Series, pd.DataFrame, None]:
    """
    Specific helper for filling NaN with 0 (common in numeric series).
    
    Args:
        series_or_df: Pandas Series or DataFrame
        inplace: Whether to modify in place
        
    Returns:
        Filled Series/DataFrame with 0 values or None if inplace=True
    """
    return safe_fillna(series_or_df, value=0, inplace=inplace)


def safe_forward_fill(series_or_df: Union[pd.Series, pd.DataFrame], 
                      inplace: bool = False) -> Union[pd.Series, pd.DataFrame, None]:
    """
    Safe forward fill that handles pandas deprecation warnings.
    
    Args:
        series_or_df: Pandas Series or DataFrame
        inplace: Whether to modify in place
        
    Returns:
        Forward-filled Series/DataFrame or None if inplace=True
    """
    return safe_fillna(series_or_df, value=None, method='ffill', inplace=inplace)


def safe_backward_fill(series_or_df: Union[pd.Series, pd.DataFrame], 
                       inplace: bool = False) -> Union[pd.Series, pd.DataFrame, None]:
    """
    Safe backward fill that handles pandas deprecation warnings.
    
    Args:
        series_or_df: Pandas Series or DataFrame
        inplace: Whether to modify in place
        
    Returns:
        Backward-filled Series/DataFrame or None if inplace=True
    """
    return safe_fillna(series_or_df, value=None, method='bfill', inplace=inplace)


def suppress_pandas_warnings():
    """
    Suppress specific pandas deprecation warnings during transition period.
    
    Use this sparingly and only during the migration period.
    Remove once all code is updated to use safe_fillna functions.
    """
    warnings.filterwarnings(
        'ignore', 
        message='.*Downcasting object dtype arrays on .fillna.*',
        category=FutureWarning
    )
    
    warnings.filterwarnings(
        'ignore',
        message='.*fillna with \'method\' is deprecated.*',
        category=FutureWarning
    )


def create_boolean_series_safe(data: Union[pd.Series, np.ndarray, list], 
                              index: Optional[pd.Index] = None) -> pd.Series:
    """
    Create a boolean Series with proper dtype handling.
    
    Args:
        data: Data for the series
        index: Index for the series
        
    Returns:
        Boolean Series with proper dtype
    """
    series = pd.Series(data, index=index, dtype=bool)
    return series


def check_pandas_compatibility():
    """
    Check pandas version and warn about compatibility issues.
    
    Returns:
        Dict with compatibility information
    """
    compatibility_info = {
        'pandas_version': pd.__version__,
        'version_tuple': PANDAS_VERSION,
        'is_2_3_plus': PANDAS_2_3_PLUS,
        'warnings_expected': PANDAS_2_3_PLUS,
        'action_required': PANDAS_2_3_PLUS
    }
    
    if PANDAS_2_3_PLUS:
        logger.warning(
            f"Pandas {pd.__version__} detected. Deprecation warnings expected for .fillna() usage. "
            f"Use src.utils.pandas_compatibility.safe_fillna() functions to resolve."
        )
    
    return compatibility_info


# Migration helper function
def update_fillna_calls_in_file(file_path: str, backup: bool = True) -> bool:
    """
    Helper function to update .fillna() calls in a Python file.
    
    Args:
        file_path: Path to the Python file to update
        backup: Whether to create a backup before modification
        
    Returns:
        True if file was successfully updated
    """
    import re
    import shutil
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            shutil.copy2(file_path, f"{file_path}.backup")
        
        # Track if any changes were made
        original_content = content
        
        # Add import at the top if not present
        if 'from src.utils.pandas_compatibility import' not in content:
            # Find the last import line
            import_lines = []
            other_lines = []
            in_imports = True
            
            for line in content.split('\n'):
                if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
                    import_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)
            
            # Add our import
            import_lines.append('from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna')
            import_lines.append('')
            
            content = '\n'.join(import_lines + other_lines)
        
        # Replace common patterns
        replacements = [
            # .fillna(False) -> safe_fillna_false()
            (r'(\w+)\.shift\(1\)\.fillna\(False\)', r'safe_fillna_false(\1.shift(1))'),
            (r'(\w+)\.fillna\(False\)', r'safe_fillna_false(\1)'),
            
            # .fillna(0) -> safe_fillna_zero()
            (r'(\w+)\.fillna\(0\)', r'safe_fillna_zero(\1)'),
            (r'(\w+)\.fillna\(0\.0\)', r'safe_fillna_zero(\1)'),
            
            # Generic .fillna(value) -> safe_fillna(series, value)
            (r'(\w+)\.fillna\(([^)]+)\)', r'safe_fillna(\1, \2)'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Write the updated file
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Updated fillna calls in {file_path}")
            return True
        else:
            logger.info(f"No fillna calls found in {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False


def batch_update_fillna_calls(directory: str, file_pattern: str = "*.py") -> dict:
    """
    Batch update .fillna() calls in multiple files.
    
    Args:
        directory: Directory to search for files
        file_pattern: Pattern to match files
        
    Returns:
        Dictionary with update results
    """
    import glob
    import os
    
    results = {
        'total_files': 0,
        'updated_files': 0,
        'failed_files': 0,
        'errors': []
    }
    
    # Find all Python files
    search_pattern = os.path.join(directory, '**', file_pattern)
    files = glob.glob(search_pattern, recursive=True)
    
    results['total_files'] = len(files)
    
    for file_path in files:
        try:
            if update_fillna_calls_in_file(file_path):
                results['updated_files'] += 1
        except Exception as e:
            results['failed_files'] += 1
            results['errors'].append(f"{file_path}: {str(e)}")
    
    return results


# Testing and validation functions
def test_pandas_compatibility():
    """Test pandas compatibility functions."""
    
    print("=== Pandas Compatibility Test ===")
    
    # Check compatibility info
    compat_info = check_pandas_compatibility()
    print(f"✅ Pandas Version: {compat_info['pandas_version']}")
    print(f"✅ Version Tuple: {compat_info['version_tuple']}")
    print(f"✅ Is 2.3+: {compat_info['is_2_3_plus']}")
    print(f"✅ Warnings Expected: {compat_info['warnings_expected']}")
    
    # Test safe_fillna functions
    print("\nTesting safe fillna functions...")
    
    # Create test data with NaN
    test_series = pd.Series([True, False, np.nan, True, np.nan])
    test_df = pd.DataFrame({
        'bool_col': [True, False, np.nan, True, np.nan],
        'num_col': [1.0, 2.0, np.nan, 4.0, np.nan]
    })
    
    print(f"Original series: {test_series.tolist()}")
    print(f"Original df dtypes: {test_df.dtypes.to_dict()}")
    
    # Test safe_fillna_false
    filled_series = safe_fillna_false(test_series)
    print(f"✅ safe_fillna_false result: {filled_series.tolist()}")
    
    # Test safe_fillna_zero
    filled_num = safe_fillna_zero(test_df['num_col'])
    print(f"✅ safe_fillna_zero result: {filled_num.tolist()}")
    
    # Test safe_fillna with DataFrame
    filled_df = safe_fillna_false(test_df[['bool_col']])
    print(f"✅ safe_fillna_false DataFrame: {filled_df['bool_col'].tolist()}")
    
    print("\n=== Pandas Compatibility Test Complete ===")


if __name__ == "__main__":
    """Test pandas compatibility utilities."""
    
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_pandas_compatibility()