---
allowed-tools: Bash(pip:*), Bash(python:*), Read, Grep, Glob, LS, Edit, MultiEdit
description: Find and fix import/dependency issues, prevent hallucinations, validate all dependencies
argument-hint: [project-path] | audit current directory if no path provided
---

# Dependency Audit & Import Validation

**Context**: You are using the CodeFarm methodology to audit and fix dependency issues. This prevents AI hallucinations, ensures all imports work, and establishes a clean dependency baseline following IndyDevDan's principle that context must be accurate.

## Dependency Discovery & Analysis

### 1. Package Configuration Files
Find all dependency definitions using Claude Code tools:

**Python configuration files:**
- Use Glob tool with patterns "requirements*.txt", "pyproject.toml", "setup.py", "Pipfile" in ${ARGUMENTS:-"."}

**Node.js configuration files:**
- Use Glob tool with patterns "package.json", "yarn.lock", "package-lock.json" in ${ARGUMENTS:-"."}

**Go configuration files:**
- Use Glob tool with patterns "go.mod", "go.sum" in ${ARGUMENTS:-"."}

**Rust configuration files:**
- Use Glob tool with patterns "Cargo.toml", "Cargo.lock" in ${ARGUMENTS:-"."}

**Java configuration files:**
- Use Glob tool with patterns "pom.xml", "build.gradle" in ${ARGUMENTS:-"."}

### 2. Import Analysis (Python Focus)
Comprehensive import scanning using Claude Code tools:

**All imports analysis:**
- Use Grep tool with pattern "^import |^from " and type "py" in ${ARGUMENTS:-"."} to find all Python imports
- Use head_limit parameter to show first 20 matches

**Local imports analysis:**
- Use Grep tool with pattern "from \\." and type "py" in ${ARGUMENTS:-"."} to find local imports

**Relative imports analysis:**
- Use Grep tool with pattern "from \\.\\." and type "py" in ${ARGUMENTS:-"."} to find relative imports

**Sys.path modifications:**
- Use Grep tool with pattern "sys\\.path" and type "py" in ${ARGUMENTS:-"."} to find path manipulations

### 3. Commented/Problematic Dependencies
Find dependency issues using Claude Code tools:

**Commented requirements analysis:**
- Use Read tool to examine requirements.txt file in ${ARGUMENTS:-"."}
- Use Grep tool with pattern "^#" in requirements.txt to find commented dependencies

**Version conflicts analysis:**
- Use Grep tool with pattern ">=|<=|==|~=" in requirements.txt to find version specifications
- Look for potential conflicts in version constraints

**Development vs production separation:**
- Use Glob tool with pattern "*requirements*.txt" to find multiple requirement files

## Import Validation Testing

### 4. Test All Imports
Create comprehensive import test script using Bash tool:

```bash
# Create and run import validation script
python -c "
import sys
import importlib
import subprocess
from pathlib import Path
import re

def test_imports(project_path='.'):
    '''Test all imports found in project files'''
    failed_imports = []
    successful_imports = []
    
    # Find all Python files
    py_files = list(Path(project_path).rglob('*.py'))
    
    # Extract import statements
    imports = set()
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple regex to find imports
                import_lines = re.findall(r'^(?:import|from)\s+([^\s\.]+)', content, re.MULTILINE)
                imports.update(import_lines)
        except Exception as e:
            print(f'Could not read {py_file}: {e}')
    
    # Test each import
    for imp in sorted(imports):
        if imp in ['sys', 'os', 'json', 'datetime', 'time', 'math', 'random']:
            continue  # Skip standard library
            
        try:
            importlib.import_module(imp)
            successful_imports.append(imp)
            print(f'✓ {imp}')
        except ImportError as e:
            failed_imports.append((imp, str(e)))
            print(f'✗ {imp}: {e}')
    
    return failed_imports, successful_imports

# Run the test
failed, successful = test_imports('${ARGUMENTS:-\".\"}')
print(f'\nSummary: {len(successful)} successful, {len(failed)} failed imports')

if failed:
    print('\nFailed imports to investigate:')
    for imp, error in failed:
        print(f'  - {imp}: {error}')
"
```

**Core import validation using Bash tool:**
```bash
# Test critical imports from the project
python -c "
import sys
import importlib
from pathlib import Path

def test_core_imports():
    test_imports = []
    
    # Find main modules
    if Path('src').exists():
        test_imports.append('src')
    
    # Test each import
    for imp in test_imports:
        try:
            if '.' not in imp:
                importlib.import_module(imp)
                print(f'✓ {imp}')
            else:
                # Handle relative imports
                module_path = imp.replace('.', '/')
                if Path(f'{module_path}.py').exists() or Path(f'{module_path}/__init__.py').exists():
                    print(f'✓ {imp} (path exists)')
                else:
                    print(f'✗ {imp} (path missing)')
        except ImportError as e:
            print(f'✗ {imp}: {e}')

test_core_imports()
"
```

## Dependency Resolution Strategy

### 5. Missing Dependencies Identification
For each failed import, determine resolution:

#### Standard Resolution Process:
1. **Check if it's a typo**: Common misspellings
2. **Find correct package name**: Use Bash tool with `pip search` or package documentation
3. **Check if it's a local module**: Should be relative import
4. **Determine if it's optional**: Some imports are conditional

#### Specific Package Issues:
- **talib vs ta-lib**: Common confusion, different package names
- **sklearn vs scikit-learn**: Package vs import name differences
- **PIL vs Pillow**: Legacy vs modern package names

### 6. Requirements File Cleanup
Clean up requirements.txt or pyproject.toml:

#### Remove Problematic Entries:
- Commented out packages that cause issues
- Packages that don't exist or are deprecated
- Conflicting version specifications

#### Add Missing Dependencies:
- Packages that are imported but not in requirements
- Version pins for stability
- Optional dependencies for development

### 7. Virtual Environment Validation
Test in clean environment using Bash tool:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Test core functionality
python -c "import src; print('Core imports work')"

# Deactivate
deactivate
```

## Import Pattern Fixes

### 8. Relative Import Standardization
Fix problematic import patterns:

#### Common Issues:
- **Absolute vs relative**: Standardize approach
- **Circular imports**: Identify and resolve
- **sys.path hacks**: Replace with proper imports

#### Standard Fixes:
```python
# BEFORE (problematic)
import sys
sys.path.append('..')
from src.module import function

# AFTER (proper)
from ..src.module import function
# OR
from src.module import function  # if running from project root
```

### 9. __init__.py File Audit
Ensure proper package structure using Claude Code tools:

**Missing __init__.py files:**
- Use Glob tool with pattern "**/src/**/" to find source directories
- Use Glob tool with pattern "**/lib/**/" to find library directories  
- Use Glob tool with pattern "**/app/**/" to find application directories
- Use Glob tool with pattern "**/__init__.py" to find existing __init__.py files
- Compare directory structure with __init__.py presence to identify missing files

**Package structure validation:**
- Use LS tool to examine directory structure
- Check for empty vs populated __init__.py files
- Identify import errors in __init__.py files

## Dependency Documentation

### 10. Create Dependency Map
Document the dependency structure:

Create `ai_docs/dependencies.md` using Write tool:
```markdown
# Project Dependencies

## Core Dependencies
| Package | Version | Purpose | Critical |
|---------|---------|---------|----------|
| [package] | [version] | [description] | [yes/no] |

## Development Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| [package] | [version] | [description] |

## Optional Dependencies
| Package | Version | Purpose | Install Command |
|---------|---------|---------|-----------------|
| [package] | [version] | [description] | pip install [package] |

## External Services
| Service | Purpose | Authentication | Documentation |
|---------|---------|----------------|---------------|
| [service] | [purpose] | [auth method] | [link] |

## Import Patterns
- **Local modules**: `from src.module import function`
- **Relative imports**: `from .module import function`
- **External packages**: `import package`

## Known Issues
- [Document any package-specific issues]
- [Version compatibility notes]
- [Platform-specific requirements]
```

### 11. Environment Setup Documentation
Create `ai_docs/setup.md` using Write tool:
```markdown
# Development Environment Setup

## Prerequisites
- Python [version]
- [Other system requirements]

## Installation
```bash
# Clone repository
git clone [repository-url]
cd [project-name]

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Setup successful')"
```

## Testing Commands
```bash
# Run tests
[test command]

# Verify imports
python -m pytest tests/ -v
```
```

## Validation & Quality Gates

### 12. Automated Dependency Checking
Set up automated validation using Write tool:

Create `scripts/validate_dependencies.py`:
```python
#!/usr/bin/env python3
"""Automated dependency validation script"""
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Validate all requirements can be installed"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'check'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All dependencies are compatible")
            return True
        else:
            print("✗ Dependency conflicts found:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False

def test_core_imports():
    """Test that core project imports work"""
    try:
        import src
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Core import failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= check_requirements()
    success &= test_core_imports()
    
    if success:
        print("\n✓ All dependency checks passed")
        sys.exit(0)
    else:
        print("\n✗ Dependency issues found")
        sys.exit(1)
```

### 13. Success Criteria
Mark audit complete when:

- [ ] All requirements files are clean and installable
- [ ] No commented-out dependencies due to failures
- [ ] All imports in codebase work correctly
- [ ] Dependencies are documented in ai_docs/
- [ ] Setup instructions are clear and tested
- [ ] Automated validation script passes

## Emergency Fixes

### Common Quick Fixes using Bash tool:
```bash
# Update pip and try again
pip install --upgrade pip

# Install with no-cache to get fresh packages
pip install --no-cache-dir -r requirements.txt

# Check for conflicting installations
pip list --outdated

# Force reinstall problematic package
pip uninstall [package] && pip install [package]
```

---

This dependency audit ensures AI tools have accurate context and prevents hallucination issues caused by unclear or broken dependencies.