---
allowed-tools: Read, Edit, MultiEdit, Bash(git:*), Bash(python:*), Bash(pytest:*), Grep, Glob
description: Improve code quality and structure without breaking existing functionality
argument-hint: [file-or-module-path] | interactive refactoring if no path provided
---

# Safe Refactoring - Improve Without Breaking

**Context**: You are using the CodeFarm methodology for safe refactoring that improves code quality while preserving all existing functionality. This follows IndyDevDan's principle of maintaining working systems while scaling their capabilities.

## Pre-Refactoring Safety Protocol

### 1. Current State Analysis
Before making any changes, establish the current state using Claude Code tools:

**Git status check:**
- Use Bash tool with command: `git status` to ensure clean working directory

**Target analysis:**
- Use Glob tool with pattern "**/*.py" in ${ARGUMENTS:-"."} to find Python files if no specific target
- Use head_limit parameter with value 10 to show first 10 files

**Test baseline:**
- Use Bash tool with command: `python -m pytest --tb=short -q 2>/dev/null || echo "No pytest setup"` to establish test baseline

**Import verification:**
- Test that current imports work using appropriate validation

### 2. Safety Commit
Create restoration point using Bash tool:

```bash
# Create safety commit
git add -A
git commit -m "Pre-refactor safety commit: $(date)"

# Create refactor branch
git checkout -b refactor/$(basename ${ARGUMENTS:-"general"})-$(date +%Y%m%d)
```

### 3. Refactoring Target Assessment
Analyze the code to be refactored:

- **File/module**: @${ARGUMENTS:-"[Please specify file to refactor]"}
- **Dependencies**: Find what imports this code
- **Test coverage**: Identify existing tests for this code
- **Usage patterns**: How this code is currently used

## Refactoring Analysis

### 4. Code Quality Assessment
Identify improvement opportunities:

#### Code Smells Detection
- **Long functions**: Functions >50 lines
- **Complex conditions**: Nested if/elif chains >3 levels
- **Repeated code**: Similar logic patterns
- **Large classes**: Classes >500 lines
- **Poor naming**: Unclear variable/function names
- **Mixed concerns**: Functions doing multiple things

#### Technical Debt Identification
**TODO/FIXME comments:**
- Use Grep tool with pattern "TODO|FIXME|HACK" and type "py" in ${ARGUMENTS:-"."} to find technical debt markers

**Other technical debt patterns:**
- **Commented code**: Large blocks of commented code
- **Magic numbers**: Hardcoded values without explanation
- **Missing error handling**: Functions without try/catch
- **Outdated patterns**: Anti-patterns or deprecated approaches

### 5. Safety Classification
Classify refactoring safety level:

#### **Green (Safe)**
- Rename variables/functions (with IDE support)
- Extract small utility functions
- Add type hints
- Improve documentation
- Remove unused imports

#### **Yellow (Careful)**
- Extract larger functions into classes
- Restructure conditional logic
- Change function signatures (with compatibility wrapper)
- Move functions between modules

#### **Red (High Risk)**
- Change public API contracts
- Modify data structures
- Change inheritance hierarchies
- Alter async/sync patterns

## Safe Refactoring Execution

### 6. Incremental Refactoring Strategy

#### Phase 1: Documentation & Type Hints (Green)
Start with safest improvements:

```python
# BEFORE
def process_data(data):
    # Process some data
    result = []
    for item in data:
        if item:
            result.append(item.upper())
    return result

# AFTER  
def process_data(data: List[str]) -> List[str]:
    """
    Process list of strings by converting to uppercase.
    
    Args:
        data: List of strings to process
        
    Returns:
        List of uppercase strings, filtering out empty values
    """
    result: List[str] = []
    for item in data:
        if item:  # Skip empty strings
            result.append(item.upper())
    return result
```

Validation after Phase 1 using Bash tool:
```bash
# Test imports still work
python -c "from ${module} import ${function}; print('Import OK')"

# Run tests
python -m pytest ${test_file} -v

# Git commit
git add -A && git commit -m "Phase 1: Add documentation and type hints"
```

#### Phase 2: Extract Functions (Green-Yellow)
Break down complex functions:

```python
# BEFORE
def complex_processing(data):
    # Validation
    if not data or len(data) == 0:
        raise ValueError("No data provided")
    
    # Processing
    results = []
    for item in data:
        if item.get('active', False):
            processed = item['value'] * 2 + 10
            if processed > 100:
                processed = 100
            results.append({
                'id': item['id'],
                'processed_value': processed,
                'timestamp': datetime.now()
            })
    
    # Sorting
    results.sort(key=lambda x: x['processed_value'])
    return results

# AFTER
def complex_processing(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process data with validation, transformation, and sorting."""
    _validate_input_data(data)
    processed_items = _process_active_items(data)
    return _sort_by_processed_value(processed_items)

def _validate_input_data(data: List[Dict[str, Any]]) -> None:
    """Validate input data is not empty."""
    if not data or len(data) == 0:
        raise ValueError("No data provided")

def _process_active_items(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process only active items with value transformation."""
    results = []
    for item in data:
        if item.get('active', False):
            processed_value = _calculate_processed_value(item['value'])
            results.append({
                'id': item['id'],
                'processed_value': processed_value,
                'timestamp': datetime.now()
            })
    return results

def _calculate_processed_value(value: float) -> float:
    """Calculate processed value with cap at 100."""
    processed = value * 2 + 10
    return min(processed, 100)

def _sort_by_processed_value(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort items by processed value."""
    return sorted(items, key=lambda x: x['processed_value'])
```

#### Phase 3: Class Extraction (Yellow)
For related functions, create service classes:

```python
# BEFORE: Multiple related functions
def process_user_data(user_data):
    # processing logic
    pass

def validate_user_data(user_data):
    # validation logic  
    pass

def save_user_data(user_data):
    # saving logic
    pass

# AFTER: Cohesive service class
class UserDataService:
    """Service for user data operations."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def process(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user data with validation and saving."""
        self._validate(user_data)
        processed_data = self._transform(user_data)
        self._save(processed_data)
        return processed_data
    
    def _validate(self, user_data: Dict[str, Any]) -> None:
        """Validate user data."""
        # validation logic
        pass
    
    def _transform(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform user data."""
        # processing logic
        pass
    
    def _save(self, user_data: Dict[str, Any]) -> None:
        """Save user data."""
        # saving logic
        pass

# Backward compatibility wrapper
def process_user_data(user_data):
    """Backward compatibility wrapper."""
    service = UserDataService()
    return service.process(user_data)
```

### 7. Advanced Refactoring Patterns

#### Replace Magic Numbers with Constants
```python
# BEFORE
def calculate_fee(amount):
    return amount * 0.025 + 5.00

# AFTER
FEE_PERCENTAGE = 0.025
BASE_FEE = 5.00

def calculate_fee(amount: float) -> float:
    """Calculate transaction fee with base fee and percentage."""
    return amount * FEE_PERCENTAGE + BASE_FEE
```

#### Improve Error Handling
```python
# BEFORE
def risky_operation(data):
    result = external_api.call(data)
    return result.value

# AFTER
def risky_operation(data: Dict[str, Any]) -> Any:
    """
    Perform operation with proper error handling.
    
    Raises:
        ServiceError: When external service fails
        ValidationError: When data is invalid
    """
    try:
        if not data:
            raise ValidationError("Data cannot be empty")
            
        result = external_api.call(data)
        
        if not result or not hasattr(result, 'value'):
            raise ServiceError("Invalid response from external service")
            
        return result.value
        
    except requests.RequestException as e:
        logger.error(f"External API failed: {e}")
        raise ServiceError(f"External service unavailable: {e}")
    except KeyError as e:
        logger.error(f"Missing required data field: {e}")
        raise ValidationError(f"Missing required field: {e}")
```

## Testing & Validation

### 8. Comprehensive Testing Strategy
After each refactoring phase using Bash tool:

#### Functionality Tests
```bash
# Run existing tests
python -m pytest ${target_test_file} -v

# Test specific functionality
python -c "
import ${module}
# Test key functions work as expected
result = ${module}.${function}(test_data)
assert result == expected_result
print('Functionality preserved')
"
```

#### Integration Tests
```bash
# Test imports from other modules
python -c "
from ${other_module} import function_that_uses_refactored_code
result = function_that_uses_refactored_code()
print('Integration preserved')
"
```

#### Performance Validation
```python
# Performance comparison script
import time
import ${module}

def benchmark_function(func, data, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        func(data)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Compare performance
old_time = benchmark_function(old_function, test_data)
new_time = benchmark_function(new_function, test_data)
print(f"Performance change: {((new_time - old_time) / old_time) * 100:.2f}%")
```

### 9. Rollback Procedures
If any validation fails, use Bash tool:

```bash
# Check what changed
git diff HEAD~1

# Rollback specific file
git checkout HEAD~1 -- ${problematic_file}

# Full rollback if needed
git reset --hard HEAD~1

# Analyze what went wrong
echo "Analyze failure and create smaller, safer change"
```

## Quality Metrics & Validation

### 10. Code Quality Improvements
Measure improvement:

#### Before/After Metrics
- **Lines of code**: Function/class size reduction
- **Cyclomatic complexity**: Reduced branching complexity
- **Test coverage**: Maintained or improved coverage
- **Performance**: Response time comparison
- **Maintainability**: Reduced code duplication

#### Documentation Quality
- **Function documentation**: All public functions documented
- **Type hints**: Complete type annotation
- **Code clarity**: Self-documenting variable names
- **Error handling**: Comprehensive exception management

### 11. Success Criteria
Refactoring complete when:

- [ ] All existing tests pass
- [ ] Code quality metrics improved
- [ ] No breaking changes to public APIs
- [ ] Performance maintained or improved
- [ ] Documentation updated and comprehensive
- [ ] Error handling enhanced
- [ ] Code duplication reduced
- [ ] Function/class sizes reasonable
- [ ] Type hints added throughout
- [ ] Git history shows clear progression

## Post-Refactoring Benefits

### Immediate Benefits
- **Readability**: Code easier to understand
- **Maintainability**: Changes easier to make
- **Testability**: Code easier to test
- **Performance**: Potential performance improvements

### Long-term Benefits
- **Technical debt reduction**: Cleaner codebase foundation
- **Development velocity**: Faster feature development
- **Bug reduction**: Clearer code = fewer bugs
- **Team collaboration**: Easier for team members to contribute

---

This safe refactoring process improves code quality systematically while preserving all existing functionality through careful validation and incremental changes.