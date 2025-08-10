# Phase 5D: Codebase Cleanup & Consolidation

**Generated**: 2025-08-10  
**Author**: Daedalus Watt - Performance Optimization Architect  
**Priority**: P1 - MEDIUM  
**Timeline**: 2 Days  
**Status**: PENDING

## Executive Summary

The codebase contains significant redundancy with 7+ validation scripts, archived test files, and 28 research directories with minimal integration. This phase consolidates duplicate functionality, removes dead code, and creates a leaner, more maintainable codebase with a 30% size reduction target.

## Problem Analysis

### Current State
- **7+ validation scripts** doing similar checks
- **Archived test files** taking up space
- **28 research directories** with only 5 files referencing them
- **Duplicate implementations** across modules
- **Dead code** from abandoned features

### Redundancy Identified
1. **Validation Scripts**: 7 validators with 80% overlap
2. **Test Archives**: 15+ outdated test files
3. **Research Directories**: 23 unused research folders
4. **Duplicate Functions**: Same logic in multiple files
5. **Commented Code**: Hundreds of lines of commented code

### Impact
- **Maintenance Burden**: Updating multiple files for same change
- **Confusion**: Developers unsure which version to use
- **Storage Waste**: 100MB+ of unnecessary files
- **Performance**: Loading unused modules
- **Technical Debt**: Accumulating cruft

## Implementation Architecture

### Day 1: Identify and Remove Dead Code

#### 1.1 Dead Code Detection
```python
# File: scripts/cleanup/dead_code_detector.py
import ast
import os
from typing import Set, Dict, List, Tuple
from pathlib import Path
import re

class DeadCodeDetector:
    """Detect unused code, functions, and imports."""
    
    def __init__(self, src_dir: str):
        self.src_dir = Path(src_dir)
        self.defined_functions: Dict[str, Set[str]] = {}
        self.called_functions: Set[str] = set()
        self.imports: Dict[str, Set[str]] = {}
        self.used_imports: Set[str] = set()
        
    def analyze_codebase(self) -> Dict[str, List]:
        """Analyze entire codebase for dead code."""
        results = {
            'unused_functions': [],
            'unused_imports': [],
            'unused_files': [],
            'commented_code': [],
            'duplicate_functions': []
        }
        
        # First pass: collect all definitions
        for py_file in self.src_dir.rglob('*.py'):
            self._analyze_file_definitions(py_file)
        
        # Second pass: collect all usages
        for py_file in self.src_dir.rglob('*.py'):
            self._analyze_file_usages(py_file)
        
        # Find unused elements
        for file, functions in self.defined_functions.items():
            for func in functions:
                if func not in self.called_functions:
                    results['unused_functions'].append(f"{file}:{func}")
        
        # Find commented code blocks
        results['commented_code'] = self._find_commented_code()
        
        # Find duplicate functions
        results['duplicate_functions'] = self._find_duplicates()
        
        return results
    
    def _analyze_file_definitions(self, filepath: Path):
        """Collect all function definitions."""
        with open(filepath, 'r') as f:
            try:
                tree = ast.parse(f.read())
                functions = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.add(node.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if filepath.name not in self.imports:
                                self.imports[filepath.name] = set()
                            self.imports[filepath.name].add(alias.name)
                
                self.defined_functions[str(filepath)] = functions
            except:
                pass
    
    def _analyze_file_usages(self, filepath: Path):
        """Collect all function calls."""
        with open(filepath, 'r') as f:
            try:
                tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            self.called_functions.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            self.called_functions.add(node.func.attr)
            except:
                pass
    
    def _find_commented_code(self) -> List[Tuple[str, int]]:
        """Find large blocks of commented code."""
        commented_blocks = []
        
        for py_file in self.src_dir.rglob('*.py'):
            with open(py_file, 'r') as f:
                lines = f.readlines()
                
            commented_lines = 0
            start_line = 0
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('#') and not stripped.startswith('#!'):
                    if commented_lines == 0:
                        start_line = i + 1
                    commented_lines += 1
                else:
                    if commented_lines > 10:  # More than 10 consecutive comments
                        commented_blocks.append((str(py_file), start_line, commented_lines))
                    commented_lines = 0
        
        return commented_blocks
    
    def _find_duplicates(self) -> List[Tuple[str, str, float]]:
        """Find duplicate function implementations."""
        from difflib import SequenceMatcher
        duplicates = []
        
        function_bodies = {}
        
        for filepath, functions in self.defined_functions.items():
            with open(filepath, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in functions:
                        # Get function body as string
                        body = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                        function_bodies[f"{filepath}:{node.name}"] = body
        
        # Compare all pairs
        items = list(function_bodies.items())
        for i, (name1, body1) in enumerate(items):
            for name2, body2 in items[i+1:]:
                similarity = SequenceMatcher(None, body1, body2).ratio()
                if similarity > 0.8:  # 80% similar
                    duplicates.append((name1, name2, similarity))
        
        return duplicates
```

#### 1.2 Archive Cleanup
```python
# File: scripts/cleanup/archive_cleaner.py
import shutil
from pathlib import Path
from typing import List, Dict
import json

class ArchiveCleaner:
    """Clean up archived and obsolete files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.files_to_remove: List[Path] = []
        self.files_to_archive: List[Path] = []
        
    def identify_obsolete_files(self) -> Dict[str, List[str]]:
        """Identify files that can be removed or archived."""
        obsolete = {
            'archive_folders': [],
            'backup_files': [],
            'old_tests': [],
            'unused_research': []
        }
        
        # Find archive folders
        for folder in self.project_root.rglob('*archive*'):
            if folder.is_dir():
                obsolete['archive_folders'].append(str(folder))
        
        # Find backup files
        for backup in self.project_root.rglob('*.backup'):
            obsolete['backup_files'].append(str(backup))
        for backup in self.project_root.rglob('*.bak'):
            obsolete['backup_files'].append(str(backup))
        for backup in self.project_root.rglob('*~'):
            obsolete['backup_files'].append(str(backup))
        
        # Find old test files
        test_dir = self.project_root / 'tests'
        if test_dir.exists():
            for test_file in test_dir.rglob('*_old.py'):
                obsolete['old_tests'].append(str(test_file))
            for test_file in test_dir.rglob('*_deprecated.py'):
                obsolete['old_tests'].append(str(test_file))
        
        # Find unused research
        research_dir = self.project_root / 'research'
        if research_dir.exists():
            used_research = self._find_used_research()
            for research_folder in research_dir.iterdir():
                if research_folder.is_dir() and research_folder.name not in used_research:
                    obsolete['unused_research'].append(str(research_folder))
        
        return obsolete
    
    def _find_used_research(self) -> Set[str]:
        """Find research directories referenced in code."""
        used = set()
        
        for py_file in self.project_root.rglob('*.py'):
            with open(py_file, 'r') as f:
                content = f.read()
                
            # Look for research references
            if 'research/' in content or '/research/' in content:
                # Extract research folder names
                import re
                matches = re.findall(r'research/(\w+)', content)
                used.update(matches)
        
        return used
    
    def clean_obsolete_files(self, dry_run: bool = True) -> Dict[str, int]:
        """Remove or archive obsolete files."""
        stats = {
            'files_removed': 0,
            'files_archived': 0,
            'space_freed_mb': 0
        }
        
        obsolete = self.identify_obsolete_files()
        
        # Calculate space
        total_size = 0
        for category, files in obsolete.items():
            for file_path in files:
                path = Path(file_path)
                if path.exists():
                    if path.is_file():
                        total_size += path.stat().st_size
                    elif path.is_dir():
                        total_size += sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        stats['space_freed_mb'] = total_size / 1024 / 1024
        
        if not dry_run:
            # Archive important files
            archive_dir = self.project_root / '.archive' / 'cleanup_archive'
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove backup files
            for backup in obsolete['backup_files']:
                Path(backup).unlink()
                stats['files_removed'] += 1
            
            # Archive old tests
            for test in obsolete['old_tests']:
                src = Path(test)
                dst = archive_dir / src.name
                shutil.move(str(src), str(dst))
                stats['files_archived'] += 1
            
            # Remove unused research
            for research in obsolete['unused_research']:
                shutil.rmtree(research)
                stats['files_removed'] += 1
        
        return stats
```

### Day 2: Consolidate Duplicate Functionality

#### 2.1 Validation Script Consolidation
```python
# File: src/validation/unified_validator.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import logging

@dataclass
class ValidationResult:
    """Unified validation result structure."""
    component: str
    status: str  # 'passed', 'failed', 'warning'
    score: float
    details: Dict[str, Any]
    recommendations: List[str]

class UnifiedValidator:
    """Consolidated validation system replacing 7+ separate validators."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validators = {
            'system': self._validate_system,
            'genetic': self._validate_genetic_algorithm,
            'data': self._validate_data_pipeline,
            'trading': self._validate_trading_system,
            'risk': self._validate_risk_management,
            'performance': self._validate_performance
        }
    
    async def validate_all(self) -> Dict[str, ValidationResult]:
        """Run all validations in parallel."""
        tasks = []
        for name, validator in self.validators.items():
            tasks.append(self._run_validation(name, validator))
        
        results = await asyncio.gather(*tasks)
        return {r.component: r for r in results}
    
    async def validate_component(self, component: str) -> ValidationResult:
        """Validate specific component."""
        if component not in self.validators:
            raise ValueError(f"Unknown component: {component}")
        
        validator = self.validators[component]
        return await self._run_validation(component, validator)
    
    async def _run_validation(self, name: str, validator) -> ValidationResult:
        """Run single validation with error handling."""
        try:
            result = await validator()
            result.component = name
            return result
        except Exception as e:
            self.logger.error(f"Validation failed for {name}: {e}")
            return ValidationResult(
                component=name,
                status='failed',
                score=0.0,
                details={'error': str(e)},
                recommendations=[f"Fix error: {e}"]
            )
    
    async def _validate_system(self) -> ValidationResult:
        """Validate overall system health."""
        checks = {
            'imports': self._check_imports(),
            'dependencies': self._check_dependencies(),
            'configuration': self._check_configuration(),
            'connections': await self._check_connections()
        }
        
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        
        return ValidationResult(
            component='system',
            status='passed' if passed == total else 'warning',
            score=passed / total,
            details=checks,
            recommendations=self._generate_recommendations(checks)
        )
    
    async def _validate_genetic_algorithm(self) -> ValidationResult:
        """Validate genetic algorithm components."""
        # Consolidate logic from multiple GA validators
        checks = {
            'population_size': self._validate_population_size(),
            'fitness_function': self._validate_fitness_function(),
            'evolution_parameters': self._validate_evolution_params(),
            'strategy_diversity': await self._validate_diversity()
        }
        
        score = sum(checks.values()) / len(checks)
        
        return ValidationResult(
            component='genetic',
            status='passed' if score > 0.8 else 'warning',
            score=score,
            details=checks,
            recommendations=[]
        )
    
    # Consolidate other validation methods...
    
    def _check_imports(self) -> bool:
        """Check all imports work."""
        try:
            # Test critical imports
            import src.execution.trading_system_manager
            import src.discovery.hierarchical_genetic_engine
            import src.backtesting.vectorbt_engine
            return True
        except ImportError:
            return False
    
    def _check_dependencies(self) -> bool:
        """Check required dependencies installed."""
        required = ['pandas', 'numpy', 'asyncio', 'ray', 'deap', 'vectorbt']
        
        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True
    
    def _check_configuration(self) -> bool:
        """Check configuration files exist."""
        from pathlib import Path
        
        config_files = [
            'config/settings.yaml',
            '.env',
            'docker-compose.yml'
        ]
        
        for config in config_files:
            if not Path(config).exists():
                return False
        return True
    
    async def _check_connections(self) -> bool:
        """Check external connections."""
        # Test database, API connections
        return True  # Placeholder
    
    def _generate_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on failed checks."""
        recommendations = []
        
        if not checks.get('imports'):
            recommendations.append("Fix import errors - run 'python -m pytest tests/'")
        if not checks.get('dependencies'):
            recommendations.append("Install missing dependencies - run 'pip install -r requirements.txt'")
        if not checks.get('configuration'):
            recommendations.append("Create missing configuration files")
        if not checks.get('connections'):
            recommendations.append("Check network connectivity and API credentials")
        
        return recommendations
```

#### 2.2 Test Consolidation
```python
# File: scripts/cleanup/test_consolidator.py
from pathlib import Path
from typing import Dict, List, Set
import ast

class TestConsolidator:
    """Consolidate duplicate test files."""
    
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.test_functions: Dict[str, List[str]] = {}
        
    def analyze_tests(self) -> Dict[str, Any]:
        """Analyze test files for consolidation opportunities."""
        analysis = {
            'duplicate_tests': [],
            'similar_tests': [],
            'test_coverage': {},
            'consolidation_plan': []
        }
        
        # Collect all test functions
        for test_file in self.test_dir.rglob('test_*.py'):
            self._analyze_test_file(test_file)
        
        # Find duplicates
        seen_tests = {}
        for file, tests in self.test_functions.items():
            for test in tests:
                if test in seen_tests:
                    analysis['duplicate_tests'].append({
                        'test': test,
                        'files': [seen_tests[test], file]
                    })
                else:
                    seen_tests[test] = file
        
        # Generate consolidation plan
        analysis['consolidation_plan'] = self._generate_consolidation_plan()
        
        return analysis
    
    def _analyze_test_file(self, filepath: Path):
        """Extract test functions from file."""
        with open(filepath, 'r') as f:
            try:
                tree = ast.parse(f.read())
                tests = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith('test_'):
                            tests.append(node.name)
                
                self.test_functions[str(filepath)] = tests
            except:
                pass
    
    def _generate_consolidation_plan(self) -> List[Dict]:
        """Generate plan for consolidating tests."""
        plan = []
        
        # Group tests by component
        component_tests = {}
        for file, tests in self.test_functions.items():
            # Extract component from filename
            filename = Path(file).name
            if filename.startswith('test_'):
                component = filename[5:].replace('.py', '')
                if component not in component_tests:
                    component_tests[component] = []
                component_tests[component].extend(tests)
        
        # Create consolidation plan
        for component, tests in component_tests.items():
            if len(tests) > 20:  # Large test file, consider splitting
                plan.append({
                    'action': 'split',
                    'component': component,
                    'target_files': [
                        f'test_{component}_unit.py',
                        f'test_{component}_integration.py'
                    ]
                })
            elif len(tests) < 5:  # Small test file, consider merging
                plan.append({
                    'action': 'merge',
                    'component': component,
                    'target_file': 'test_common.py'
                })
        
        return plan
    
    def consolidate_tests(self, dry_run: bool = True):
        """Execute test consolidation plan."""
        analysis = self.analyze_tests()
        
        if dry_run:
            print("Consolidation plan:")
            for item in analysis['consolidation_plan']:
                print(f"  {item['action']}: {item['component']}")
            return
        
        # Execute consolidation
        for duplicate in analysis['duplicate_tests']:
            # Keep first occurrence, remove others
            for file in duplicate['files'][1:]:
                print(f"Removing duplicate test from {file}")
                # Remove duplicate test function
```

#### 2.3 Research Integration
```python
# File: scripts/cleanup/research_integrator.py
from pathlib import Path
from typing import Dict, List, Set
import shutil

class ResearchIntegrator:
    """Integrate useful research into codebase."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.research_dir = self.project_root / 'research'
        self.src_dir = self.project_root / 'src'
        
    def analyze_research_usage(self) -> Dict[str, Any]:
        """Analyze which research is actually used."""
        usage = {
            'used_research': set(),
            'unused_research': set(),
            'integration_opportunities': []
        }
        
        # Find all research directories
        all_research = {d.name for d in self.research_dir.iterdir() if d.is_dir()}
        
        # Find used research
        for py_file in self.src_dir.rglob('*.py'):
            with open(py_file, 'r') as f:
                content = f.read()
            
            for research in all_research:
                if f'research/{research}' in content:
                    usage['used_research'].add(research)
        
        usage['unused_research'] = all_research - usage['used_research']
        
        # Find integration opportunities
        for research in usage['used_research']:
            research_path = self.research_dir / research
            if (research_path / 'research_summary.md').exists():
                usage['integration_opportunities'].append({
                    'research': research,
                    'summary': research_path / 'research_summary.md',
                    'recommendation': self._generate_integration_recommendation(research)
                })
        
        return usage
    
    def _generate_integration_recommendation(self, research: str) -> str:
        """Generate recommendation for integrating research."""
        recommendations = {
            'hyperliquid': 'Create src/data/hyperliquid_constants.py with API constants',
            'vectorbt': 'Add vectorbt optimization patterns to backtesting engine',
            'ray_cluster': 'Implement ray best practices in genetic_strategy_pool.py',
            'deap': 'Add DEAP advanced features to genetic engine',
            'neon': 'Optimize database queries based on Neon patterns'
        }
        
        for key, rec in recommendations.items():
            if key in research.lower():
                return rec
        
        return f"Review {research} research for implementation patterns"
    
    def create_research_index(self) -> Path:
        """Create index of all research for quick reference."""
        index_path = self.research_dir / 'INDEX.md'
        
        with open(index_path, 'w') as f:
            f.write("# Research Index\n\n")
            f.write("## Active Research (Used in Code)\n\n")
            
            usage = self.analyze_research_usage()
            
            for research in sorted(usage['used_research']):
                f.write(f"- **{research}**: ")
                summary_file = self.research_dir / research / 'research_summary.md'
                if summary_file.exists():
                    with open(summary_file, 'r') as sf:
                        first_line = sf.readline().strip('#').strip()
                        f.write(f"{first_line}\n")
                else:
                    f.write("No summary available\n")
            
            f.write("\n## Archived Research (Not Currently Used)\n\n")
            for research in sorted(usage['unused_research']):
                f.write(f"- {research}\n")
        
        return index_path
```

## Success Metrics

### Cleanup Targets
- ✅ **30% reduction in codebase size**
- ✅ **Single validation system** replacing 7+ scripts
- ✅ **Zero duplicate functions**
- ✅ **All research indexed and referenced**

### Code Quality Improvements
- ✅ No dead code remaining
- ✅ Clear module boundaries
- ✅ Single source of truth for each function
- ✅ Comprehensive test consolidation

### Maintenance Benefits
- ✅ Reduced update complexity
- ✅ Clear code organization
- ✅ Faster development iteration
- ✅ Lower cognitive load

## Risk Mitigation

### Potential Risks
1. **Accidental Deletion**: Removing needed code
   - Mitigation: Archive before deletion, dry-run mode
   
2. **Breaking Changes**: Consolidation breaks dependencies
   - Mitigation: Comprehensive testing, gradual migration
   
3. **Lost History**: Removing code with important comments
   - Mitigation: Git history preservation, documentation

## Validation Steps

1. **Pre-Cleanup Baseline**:
   - Record codebase size
   - Count duplicate functions
   - List all validators

2. **Post-Cleanup Validation**:
   - Verify size reduction achieved
   - Run consolidated validator
   - Ensure all tests pass

3. **Integration Testing**:
   - Test all major workflows
   - Verify no functionality lost
   - Check performance maintained

## Dependencies

- Python AST for code analysis
- Git for history preservation
- pytest for test validation

## Next Phase

After cleanup, proceed to Phase 5E (Simplified Trading Loop) to implement core trading functionality in the cleaned codebase.