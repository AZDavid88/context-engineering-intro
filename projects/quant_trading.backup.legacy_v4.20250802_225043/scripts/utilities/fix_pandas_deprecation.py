"""
Automated Pandas Deprecation Fix Script

This script automatically fixes .fillna() deprecation warnings across the codebase
by replacing problematic calls with safe alternatives from pandas_compatibility.py

Usage:
    python fix_pandas_deprecation.py --dry-run    # Preview changes
    python fix_pandas_deprecation.py --apply      # Apply fixes
    python fix_pandas_deprecation.py --verify     # Verify fixes work
"""

import os
import re
import argparse
import shutil
from pathlib import Path


def find_fillna_usage(directory: str) -> list:
    """Find all files with .fillna() usage."""
    
    fillna_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '.fillna(' in content:
                            fillna_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return fillna_files


def analyze_fillna_patterns(file_path: str) -> dict:
    """Analyze fillna patterns in a file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    patterns = {
        'fillna_false': len(re.findall(r'\.fillna\(False\)', content)),
        'fillna_zero': len(re.findall(r'\.fillna\(0\)', content)),
        'fillna_method': len(re.findall(r'\.fillna\(method=', content)),
        'fillna_other': len(re.findall(r'\.fillna\([^)]*\)', content))
    }
    
    # Subtract the already counted ones from 'other'
    patterns['fillna_other'] -= (patterns['fillna_false'] + 
                                patterns['fillna_zero'] + 
                                patterns['fillna_method'])
    
    return patterns


def fix_file_fillna(file_path: str, dry_run: bool = True) -> dict:
    """Fix fillna usage in a single file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    content = original_content
    changes = []
    
    # Check if imports already exist
    has_compatibility_import = 'from src.utils.pandas_compatibility import' in content
    
    # Add import if needed
    if not has_compatibility_import:
        # Find the right place to add the import
        import_lines = []
        other_lines = []
        found_import_section = False
        
        for line in content.split('\n'):
            if (line.startswith('import ') or line.startswith('from ') or 
                line.strip() == '' or line.startswith('#')):
                import_lines.append(line)
                if line.startswith('import ') or line.startswith('from '):
                    found_import_section = True
            else:
                if found_import_section and not line.strip().startswith('#'):
                    # Add our import before the first non-import, non-comment line
                    import_lines.append('')
                    import_lines.append('from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna')
                    import_lines.append('')
                    found_import_section = False
                    changes.append("Added pandas_compatibility import")
                other_lines.append(line)
        
        if found_import_section:  # All lines were imports
            import_lines.append('')
            import_lines.append('from src.utils.pandas_compatibility import safe_fillna_false, safe_fillna_zero, safe_fillna')
            import_lines.append('')
            changes.append("Added pandas_compatibility import")
        
        content = '\n'.join(import_lines + other_lines)
    
    # Fix patterns
    replacements = [
        # Most common pattern: .shift(1).fillna(False)
        (r'(\w+)\.shift\(1\)\.fillna\(False\)', 
         r'safe_fillna_false(\1.shift(1))', 
         'shift().fillna(False) -> safe_fillna_false()'),
        
        # Direct .fillna(False)
        (r'(\w+)\.fillna\(False\)', 
         r'safe_fillna_false(\1)', 
         '.fillna(False) -> safe_fillna_false()'),
        
        # .fillna(0) patterns
        (r'(\w+)\.fillna\(0\)', 
         r'safe_fillna_zero(\1)', 
         '.fillna(0) -> safe_fillna_zero()'),
        
        (r'(\w+)\.fillna\(0\.0\)', 
         r'safe_fillna_zero(\1)', 
         '.fillna(0.0) -> safe_fillna_zero()'),
    ]
    
    for pattern, replacement, description in replacements:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes.append(f"{description} - {len(matches)} replacements")
    
    # Count remaining complex fillna patterns that need manual review
    remaining_fillna = re.findall(r'\.fillna\([^)]+\)', content)
    if remaining_fillna:
        changes.append(f"MANUAL REVIEW NEEDED: {len(remaining_fillna)} complex .fillna() patterns remain")
    
    result = {
        'file_path': file_path,
        'changes': changes,
        'content_changed': content != original_content,
        'original_content': original_content,
        'new_content': content
    }
    
    # Apply changes if not dry run
    if not dry_run and result['content_changed']:
        # Create backup
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result['backup_created'] = backup_path
    
    return result


def main():
    """Main script execution."""
    
    parser = argparse.ArgumentParser(description='Fix pandas .fillna() deprecation warnings')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without applying them')
    parser.add_argument('--apply', action='store_true', 
                       help='Apply fixes to files')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify that fixes work by running tests')
    parser.add_argument('--directory', default='src', 
                       help='Directory to search for files (default: src)')
    
    args = parser.parse_args()
    
    if not any([args.dry_run, args.apply, args.verify]):
        print("Please specify --dry-run, --apply, or --verify")
        return
    
    print("=== Pandas Deprecation Fix Script ===")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY FIXES' if args.apply else 'VERIFY'}")
    print(f"Directory: {args.directory}")
    print()
    
    # Find files with fillna usage
    fillna_files = find_fillna_usage(args.directory)
    print(f"Found {len(fillna_files)} files with .fillna() usage:")
    
    for file_path in fillna_files:
        patterns = analyze_fillna_patterns(file_path)
        total_patterns = sum(patterns.values())
        if total_patterns > 0:
            print(f"  {file_path}: {total_patterns} patterns")
            for pattern_type, count in patterns.items():
                if count > 0:
                    print(f"    - {pattern_type}: {count}")
    
    print()
    
    if args.verify:
        print("=== Verification Mode ===")
        # Test the pandas compatibility module
        try:
            import subprocess
            result = subprocess.run(['python', 'src/utils/pandas_compatibility.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ pandas_compatibility.py works correctly")
            else:
                print("❌ pandas_compatibility.py has issues:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error testing pandas_compatibility.py: {e}")
        return
    
    # Process files
    total_files_changed = 0
    total_changes = 0
    
    for file_path in fillna_files:
        # Skip certain files that might be problematic
        if any(skip in file_path for skip in ['__pycache__', '.backup', 'pandas_compatibility.py']):
            continue
        
        result = fix_file_fillna(file_path, dry_run=args.dry_run)
        
        if result['changes']:
            print(f"\n📁 {file_path}:")
            for change in result['changes']:
                print(f"  ✓ {change}")
            
            if result['content_changed']:
                total_files_changed += 1
                total_changes += len(result['changes'])
                
                if args.apply and 'backup_created' in result:
                    print(f"  💾 Backup created: {result['backup_created']}")
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {len(fillna_files)}")
    print(f"Files changed: {total_files_changed}")
    print(f"Total changes: {total_changes}")
    
    if args.dry_run:
        print("\n🔍 This was a dry run. Use --apply to make changes.")
    elif args.apply:
        print("\n✅ Changes applied. Backup files created with .backup extension.")
        print("\n🧪 Run tests to verify everything works:")
        print("  python src/execution/monitoring.py")
        print("  python test_monitoring_integration.py")


if __name__ == "__main__":
    main()