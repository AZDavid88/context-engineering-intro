"""
Scripts Package - Common Utilities for All Scripts

This module provides common utilities for all scripts in the project,
including standardized path configuration to fix the "No module named 'src'" error.
"""

import sys
import os
from pathlib import Path

def setup_project_path():
    """
    Setup project path for imports - fixes "No module named 'src'" error.
    
    This function adds the project root directory to sys.path, allowing
    all scripts to import from the src/ directory regardless of where
    they are executed from.
    
    Returns:
        Path: The project root path that was added to sys.path
    """
    # Get the project root (two levels up from scripts directory)
    script_file = Path(__file__).resolve()
    project_root = script_file.parent.parent
    
    # Add to sys.path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root

# Auto-setup when module is imported
PROJECT_ROOT = setup_project_path()