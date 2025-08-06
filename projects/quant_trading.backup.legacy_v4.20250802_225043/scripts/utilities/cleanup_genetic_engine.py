#!/usr/bin/env python3
"""
Genetic Engine Whitespace Cleanup Utility

Adapted from cleanup_seed_whitespace.py to fix the genetic_engine.py file
that has excessive whitespace issues affecting its structural integrity.
"""

import os
import re
from typing import List

def cleanup_whitespace(file_path: str, max_consecutive_empty: int = 2) -> bool:
    """
    Clean up excessive whitespace in a Python file while preserving functionality.
    
    Args:
        file_path: Path to the Python file to clean
        max_consecutive_empty: Maximum consecutive empty lines allowed
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    
    # Read the original file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Create backup
    backup_path = f"{file_path}.cleanup_backup"
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Process lines to remove excessive whitespace
    cleaned_lines = []
    consecutive_empty = 0
    
    for i, line in enumerate(lines):
        # Check if line is effectively empty (whitespace only)
        if line.strip() == '':
            consecutive_empty += 1
            
            # Only add empty line if within limit
            if consecutive_empty <= max_consecutive_empty:
                cleaned_lines.append('\n')  # Normalize to simple newline
        else:
            # Reset counter and add the functional line
            consecutive_empty = 0
            cleaned_lines.append(line)
    
    # Remove trailing empty lines at end of file
    while cleaned_lines and cleaned_lines[-1].strip() == '':
        cleaned_lines.pop()
    
    # Ensure file ends with single newline
    if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
        cleaned_lines[-1] += '\n'
    
    # Write cleaned version
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        # Validate the file can still be imported
        file_name = os.path.basename(file_path)
        if file_name.endswith('.py'):
            try:
                # Try to compile the cleaned file
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, file_path, 'exec')
                print(f"✅ Cleaned {file_path} - syntax validation passed")
                return True
            except SyntaxError as e:
                print(f"❌ Syntax error after cleanup in {file_path}: {e}")
                # Restore backup
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original_lines = f.readlines()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(original_lines)
                print(f"🔄 Restored original file from backup")
                return False
                
    except Exception as e:
        print(f"Error writing cleaned file: {e}")
        return False
    
    return True

def main():
    """Clean up the genetic_engine.py file."""
    engine_file = "src/strategy/genetic_engine.py"
    
    if not os.path.exists(engine_file):
        print(f"File not found: {engine_file}")
        return
    
    print(f"🧹 Cleaning whitespace in {engine_file}")
    
    # Show before stats
    with open(engine_file, 'r') as f:
        lines_before = f.readlines()
    
    empty_lines_before = sum(1 for line in lines_before if line.strip() == '')
    total_lines_before = len(lines_before)
    
    print(f"📊 Before: {total_lines_before} total lines, {empty_lines_before} empty lines")
    
    # Clean the file
    success = cleanup_whitespace(engine_file, max_consecutive_empty=1)
    
    if success:
        # Show after stats
        with open(engine_file, 'r') as f:
            lines_after = f.readlines()
        
        empty_lines_after = sum(1 for line in lines_after if line.strip() == '')
        total_lines_after = len(lines_after)
        
        print(f"📊 After: {total_lines_after} total lines, {empty_lines_after} empty lines")
        print(f"🎯 Removed {empty_lines_before - empty_lines_after} excessive empty lines")
        print(f"✅ Cleanup completed successfully!")
        
        # Test import
        try:
            import sys
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("test_engine", engine_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ Import test passed - file is functional")
            
        except Exception as e:
            print(f"⚠️ Import test failed: {e}")
    
    else:
        print(f"❌ Cleanup failed - check backup file")

if __name__ == "__main__":
    main()