---
allowed-tools: Bash(pytest:*), Bash(python:*), Bash(git:*), Bash(find:*), Read, Grep, Glob, LS
description: Comprehensive testing and system validation with quality gates
argument-hint: [test-scope] | full system validation if no scope provided
---

# Comprehensive System Validation

**Context**: You are using the CodeFarm methodology for comprehensive system validation. This implements quality gates that ensure system reliability and maintainability following IndyDevDan's self-validation principles.

## Validation Scope Analysis

### 1. System Discovery
Identify what needs validation using Claude Code tools:

**Project structure analysis:**
- Use Glob tool with pattern "**/*.py" to find all Python files for validation
- Use Glob tool with patterns "**/test*.py", "**/test_*.py" to locate test files
- Use Glob tool with patterns "**/main.py", "**/__main__.py", "**/app.py" to find entry points
- Use Glob tool with patterns "**/*.toml", "**/*.yaml", "**/*.json", "**/requirements.txt" to find configuration files

### 2. Validation Target
**Scope**: ${ARGUMENTS:-"full-system"}

Available scopes:
- `unit` - Unit tests only
- `integration` - Integration tests  
- `api` - API endpoint testing
- `performance` - Performance benchmarks
- `security` - Security validation
- `full-system` - Complete validation suite

## Test Environment Setup

### 3. Environment Validation
Ensure test environment is properly configured using Bash tool:

```bash
# Python environment check
python --version
echo "Virtual env: $VIRTUAL_ENV"

# Dependencies validation
pip check || echo "Dependency conflicts found"

# Test framework availability
python -m pytest --version || echo "pytest not available"

# Git repository status
git status --porcelain | wc -l
```

### 4. Test Discovery
Use Glob and LS tools to discover test structure:

**Test file organization:**
- Use Glob tool with patterns "**/tests/**/*.py", "**/test/**/*.py" to find organized tests
- Use LS tool to examine test directory structure
- Use Glob tool with pattern "**/conftest.py" to find pytest configuration files

## Core System Validation

### 5. Unit Test Execution
Run comprehensive unit tests using Bash tool:

```bash
# Unit test execution
echo "=== Unit Test Execution ==="
python -m pytest tests/unit/ -v --tb=short --cov=src --cov-report=term-missing || echo "Unit tests completed with issues"

# Generate coverage report
python -m pytest tests/unit/ --cov=src --cov-report=html:validation_reports/coverage || echo "Coverage report generated"

# Test summary
python -m pytest tests/unit/ --tb=no -q || echo "Unit test summary completed"
```

### 6. Integration Testing
Execute integration tests using Bash tool:

**Integration test analysis:**
- Use Glob tool with patterns "**/integration/**/*.py", "**/*integration*test*.py" to find integration tests
- Use Bash tool to run: `python -m pytest tests/integration/ -v --tb=short`

### 7. API Endpoint Validation
Test API endpoints if they exist:

**API test discovery:**
- Use Glob tool with patterns "**/api/**/*.py", "**/*api*test*.py" to find API tests
- Use Grep tool to search for "@app.route", "@router." patterns in source code
- Use Bash tool to run API tests if found

## Performance and Load Testing

### 8. Performance Benchmarks
Run performance validation using Bash tool:

```bash
# Performance test execution
echo "=== Performance Testing ==="
python -m pytest tests/performance/ -v --benchmark-only || echo "No performance tests found"

# Memory usage analysis
python -c "
import psutil
import sys
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Python version: {sys.version}')
"
```

### 9. Load Testing
Execute load tests if available:

**Load test analysis:**
- Use Glob tool with pattern "**/*load*test*.py" to find load tests
- Use Bash tool to run load tests with appropriate parameters

## Security and Compliance Validation

### 10. Security Test Suite
Run security validation using Bash tool:

```bash
# Security testing
echo "=== Security Validation ==="
python -m pytest tests/security/ -v || echo "No security tests found"

# Basic security checks
python -c "
import os
import stat
import pathlib

# Check file permissions
sensitive_files = ['.env', 'config.py', 'secrets.json']
for file in sensitive_files:
    if os.path.exists(file):
        perms = oct(stat.S_IMODE(os.lstat(file).st_mode))
        print(f'{file}: {perms}')
"
```

## System Integration Validation

### 11. End-to-End Testing
Execute complete system tests:

**E2E test discovery:**
- Use Glob tool with patterns "**/e2e/**/*.py", "**/*e2e*test*.py" to find end-to-end tests
- Use Bash tool to run: `python -m pytest tests/e2e/ -v --tb=short`

### 12. Configuration Validation
Validate system configuration:

**Configuration analysis:**
- Use Read tool to examine configuration files found earlier
- Use Grep tool to search for environment variable references
- Validate configuration completeness and security

## Quality Gates and Metrics

### 13. Code Quality Validation
Run code quality checks using Bash tool:

```bash
# Code quality validation
echo "=== Code Quality Checks ==="

# Linting
python -m flake8 src/ || echo "Linting completed with issues"

# Type checking
python -m mypy src/ || echo "Type checking completed with issues"

# Security scanning
python -m bandit -r src/ -ll || echo "Security scan completed"
```

### 14. Documentation Validation
Check documentation completeness:

**Documentation analysis:**
- Use Glob tool with patterns "**/*.md", "**/docs/**/*" to find documentation
- Use Read tool to examine README.md and main documentation files
- Use Grep tool to search for TODO, FIXME, or incomplete documentation markers

## Validation Reporting

### 15. Comprehensive Validation Report
Generate validation report using collected data:

```python
# validation_reporter.py
import json
import time
from pathlib import Path

def generate_validation_report(results):
    """Generate comprehensive validation report"""
    
    report = f"""
# System Validation Report
**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Validation Scope**: {results.get('scope', 'full-system')}

## Summary
- **Total tests run**: {results.get('total_tests', 0)}
- **Passed**: {results.get('passed', 0)}
- **Failed**: {results.get('failed', 0)}
- **Coverage**: {results.get('coverage', 'N/A')}%

## Test Results by Category
- **Unit Tests**: {results.get('unit_status', 'Not run')}
- **Integration Tests**: {results.get('integration_status', 'Not run')}
- **API Tests**: {results.get('api_status', 'Not run')}
- **Performance Tests**: {results.get('performance_status', 'Not run')}
- **Security Tests**: {results.get('security_status', 'Not run')}

## Quality Gates Status
- [ ] Unit test coverage > 80%
- [ ] All critical tests passing
- [ ] No high-severity security issues
- [ ] Performance benchmarks met
- [ ] Code quality standards met

## Recommendations
{results.get('recommendations', [])}

## Next Steps  
{results.get('next_steps', [])}
"""
    
    return report

if __name__ == "__main__":
    # Example usage
    results = {
        'scope': 'full-system',
        'total_tests': 45,
        'passed': 42,
        'failed': 3,
        'coverage': 85
    }
    
    report = generate_validation_report(results)
    
    with open("validation_reports/system_validation_report.md", "w") as f:
        f.write(report)
    
    print("Validation report generated")
```

### 16. Success Criteria
System validation complete when:

- [ ] All critical tests passing (>95%)
- [ ] Unit test coverage >80%
- [ ] Integration tests passing
- [ ] No critical security vulnerabilities
- [ ] Performance benchmarks met
- [ ] Code quality gates passed
- [ ] Documentation complete
- [ ] Configuration validated
- [ ] Dependencies secure and compatible

---

This comprehensive validation ensures system reliability and maintainability through systematic quality gates and testing protocols.