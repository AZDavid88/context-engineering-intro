# /codebase-validator

---
description: Comprehensive codebase validation through systematic behavioral analysis - validate entire system functionality, integration, and reliability
allowed-tools: Bash, Read, Write, Glob, Grep, LS, Task, TodoWrite
argument-hint: [system-path] - Path to complex codebase for complete validation (defaults to current directory)
---

# Codebase Validator
## Systematic Validation of Complex Systems Through Observable Evidence

**Mission**: Comprehensively validate any complex codebase by systematically measuring what it actually does, testing real integrations, and identifying gaps between claims and reality.

## Dynamic System Context Discovery
The validator will automatically discover system context during execution.

## Your Comprehensive Validation Mission

**Target System**: `$ARGUMENTS` (defaults to current directory)

Execute systematic validation of the entire codebase through evidence-based analysis, breaking down complex validation into manageable, traceable steps that build comprehensive understanding.

---

```bash
# Comprehensive Codebase Validation Through Code-as-Truth Analysis
echo "=== COMPREHENSIVE CODEBASE VALIDATION ==="
echo "Analysis initiated: $(date)"
echo "System path: $(pwd)"
echo ""

# Initialize scoring variables
TOTAL_TESTS=0
PASSED_TESTS=0
VALIDATION_ISSUES=()

# Phase 1: Technology Stack Discovery (Code-Based)
echo "=== TECHNOLOGY STACK DISCOVERY ==="
PYTHON_FILES=$(find . -name "*.py" | wc -l)
JS_FILES=$(find . -name "*.js" -o -name "*.ts" | wc -l)
GO_FILES=$(find . -name "*.go" | wc -l)
JAVA_FILES=$(find . -name "*.java" | wc -l)
RUST_FILES=$(find . -name "*.rs" | wc -l)

echo "Code Distribution:"
[ $PYTHON_FILES -gt 0 ] && echo "  Python: $PYTHON_FILES files"
[ $JS_FILES -gt 0 ] && echo "  JavaScript/TypeScript: $JS_FILES files"
[ $GO_FILES -gt 0 ] && echo "  Go: $GO_FILES files"
[ $JAVA_FILES -gt 0 ] && echo "  Java: $JAVA_FILES files"
[ $RUST_FILES -gt 0 ] && echo "  Rust: $RUST_FILES files"

# Runtime Environment Validation
echo ""
echo "Runtime Environment Validation:"
PYTHON_RUNTIME=0
NODE_RUNTIME=0
GO_RUNTIME=0

if [ $PYTHON_FILES -gt 0 ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if python3 --version >/dev/null 2>&1; then
        echo "  ✅ Python runtime: Available"
        PYTHON_RUNTIME=1
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "  ❌ Python runtime: Unavailable"
        VALIDATION_ISSUES+=("Python files exist but runtime unavailable")
    fi
fi

if [ $JS_FILES -gt 0 ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if node --version >/dev/null 2>&1; then
        echo "  ✅ Node.js runtime: Available"
        NODE_RUNTIME=1
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "  ❌ Node.js runtime: Unavailable"
        VALIDATION_ISSUES+=("JavaScript files exist but Node.js runtime unavailable")
    fi
fi

if [ $GO_FILES -gt 0 ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if go version >/dev/null 2>&1; then
        echo "  ✅ Go runtime: Available"
        GO_RUNTIME=1
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "  ❌ Go runtime: Unavailable"
        VALIDATION_ISSUES+=("Go files exist but Go runtime unavailable")
    fi
fi

# Phase 2: Actual Import Analysis (Code-as-Truth)
echo ""
echo "=== ACTUAL IMPORT ANALYSIS ==="

if [ $PYTHON_FILES -gt 0 ] && [ $PYTHON_RUNTIME -eq 1 ]; then
    echo "Analyzing Python imports from actual source code..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    PYTHON_IMPORT_RESULT=$(python3 -c '
import ast, os, sys
imports = set()
local_modules = set()
failed_files = 0

# Discover local modules first
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            local_modules.add(module_name)

# Analyze imports from actual code
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.add(alias.name.split(".")[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.add(node.module.split(".")[0])
                    except SyntaxError:
                        failed_files += 1
            except:
                failed_files += 1

# Filter out local modules and test actual imports
external_imports = imports - local_modules
successful_imports = 0
failed_imports = []

for imp in sorted(external_imports):
    if imp and imp not in ["__future__"]:
        try:
            __import__(imp)
            successful_imports += 1
        except ImportError:
            failed_imports.append(imp)

print(f"FILES_ANALYZED:{len([f for r,d,files in os.walk(\".\") for f in files if f.endswith(\".py\")])}")
print(f"PARSE_FAILURES:{failed_files}")
print(f"EXTERNAL_IMPORTS:{len(external_imports)}")
print(f"SUCCESSFUL_IMPORTS:{successful_imports}")
print(f"FAILED_IMPORTS:{len(failed_imports)}")
if failed_imports:
    print("MISSING_MODULES:" + ",".join(failed_imports[:5]))
else:
    print("MISSING_MODULES:none")
' 2>/dev/null)
C    
    if [ -n "$PYTHON_IMPORT_RESULT" ]; then
        FILES_ANALYZED=$(echo "$PYTHON_IMPORT_RESULT" | grep "FILES_ANALYZED:" | cut -d: -f2)
        SUCCESSFUL_IMPORTS=$(echo "$PYTHON_IMPORT_RESULT" | grep "SUCCESSFUL_IMPORTS:" | cut -d: -f2)
        FAILED_IMPORTS=$(echo "$PYTHON_IMPORT_RESULT" | grep "FAILED_IMPORTS:" | cut -d: -f2)
        MISSING_MODULES=$(echo "$PYTHON_IMPORT_RESULT" | grep "MISSING_MODULES:" | cut -d: -f2)
        
        echo "  Python files analyzed: $FILES_ANALYZED"
        echo "  Successful imports: $SUCCESSFUL_IMPORTS"
        echo "  Failed imports: $FAILED_IMPORTS"
        
        if [ "$MISSING_MODULES" != "none" ]; then
            echo "  ❌ Missing modules: $MISSING_MODULES"
            VALIDATION_ISSUES+=("Python code requires unavailable modules: $MISSING_MODULES")
        else
            echo "  ✅ All required modules available"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        fi
    else
        echo "  ❌ Python import analysis failed"
        VALIDATION_ISSUES+=("Python import analysis failed to execute")
    fi
fi

if [ $JS_FILES -gt 0 ] && [ $NODE_RUNTIME -eq 1 ]; then
    echo "Analyzing Node.js requires from actual source code..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    NODE_REQUIRE_RESULT=$(node -e "
const fs = require('fs');
const path = require('path');
const requires = new Set();
let filesAnalyzed = 0;

function analyzeFile(filepath) {
    try {
        const content = fs.readFileSync(filepath, 'utf8');
        filesAnalyzed++;
        
        // Extract require() calls
        const requireRegex = /require\\s*\\(\\s*['\\\"]([^'\\\"]+)['\\\"]\\s*\\)/g;
        let match;
        while ((match = requireRegex.exec(content)) !== null) {
            const module = match[1];
            if (!module.startsWith('.') && !module.startsWith('/')) {
                requires.add(module.split('/')[0]); // Get base module name
            }
        }
        
        // Extract ES6 imports
        const importRegex = /import.*?from\\s*['\\\"]([^'\\\"]+)['\\\"]/g;
        while ((match = importRegex.exec(content)) !== null) {
            const module = match[1];
            if (!module.startsWith('.') && !module.startsWith('/')) {
                requires.add(module.split('/')[0]);
            }
        }
    } catch(e) {}
}

function walkDir(dir) {
    try {
        fs.readdirSync(dir).forEach(file => {
            const filepath = path.join(dir, file);
            const stat = fs.statSync(filepath);
            if (stat.isDirectory() && file !== 'node_modules' && !file.startsWith('.')) {
                walkDir(filepath);
            } else if (stat.isFile() && (file.endsWith('.js') || file.endsWith('.ts'))) {
                analyzeFile(filepath);
            }
        });
    } catch(e) {}
}

walkDir('.');

let successfulRequires = 0;
let failedRequires = [];

Array.from(requires).forEach(module => {
    try {
        require(module);
        successfulRequires++;
    } catch(e) {
        failedRequires.push(module);
    }
});

console.log('FILES_ANALYZED:' + filesAnalyzed);
console.log('EXTERNAL_REQUIRES:' + requires.size);
console.log('SUCCESSFUL_REQUIRES:' + successfulRequires);
console.log('FAILED_REQUIRES:' + failedRequires.length);
console.log('MISSING_MODULES:' + (failedRequires.length > 0 ? failedRequires.slice(0,5).join(',') : 'none'));
" 2>/dev/null)
    
    if [ -n "$NODE_REQUIRE_RESULT" ]; then
        FILES_ANALYZED=$(echo "$NODE_REQUIRE_RESULT" | grep "FILES_ANALYZED:" | cut -d: -f2)
        SUCCESSFUL_REQUIRES=$(echo "$NODE_REQUIRE_RESULT" | grep "SUCCESSFUL_REQUIRES:" | cut -d: -f2)
        FAILED_REQUIRES=$(echo "$NODE_REQUIRE_RESULT" | grep "FAILED_REQUIRES:" | cut -d: -f2)
        MISSING_MODULES=$(echo "$NODE_REQUIRE_RESULT" | grep "MISSING_MODULES:" | cut -d: -f2)
        
        echo "  Node.js files analyzed: $FILES_ANALYZED"
        echo "  Successful requires: $SUCCESSFUL_REQUIRES"
        echo "  Failed requires: $FAILED_REQUIRES"
        
        if [ "$MISSING_MODULES" != "none" ]; then
            echo "  ❌ Missing modules: $MISSING_MODULES"
            VALIDATION_ISSUES+=("Node.js code requires unavailable modules: $MISSING_MODULES")
        else
            echo "  ✅ All required modules available"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        fi
    else
        echo "  ❌ Node.js require analysis failed"
        VALIDATION_ISSUES+=("Node.js require analysis failed to execute")
    fi
fi

# Phase 3: Behavioral Validation Testing
echo ""
echo "=== BEHAVIORAL VALIDATION TESTING ==="

if [ $PYTHON_FILES -gt 0 ] && [ $PYTHON_RUNTIME -eq 1 ]; then
    echo "Testing Python code execution behavior..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    PYTHON_BEHAVIOR=$(python3 -c '
import sys, os, time
start_time = time.time()
sys.path.insert(0, ".")

executed_modules = 0
execution_errors = []

# Try to import and test main modules
for root, dirs, files in os.walk("."):
    if "src" in root.lower() or "lib" in root.lower() or root == ".":
        for file in files:
            if file.endswith(".py") and not file.startswith("__") and file not in ["setup.py", "test_", "tests"]:
                module_name = file[:-3]
                try:
                    module = __import__(module_name)
                    executed_modules += 1
                    if executed_modules >= 5:  # Limit testing to avoid long execution
                        break
                except Exception as e:
                    execution_errors.append(f"{module_name}:{str(e)[:50]}")
                    if len(execution_errors) >= 3:  # Limit error collection
                        break
        if executed_modules >= 5 or len(execution_errors) >= 3:
            break

end_time = time.time()
print(f"EXECUTED_MODULES:{executed_modules}")
print(f"EXECUTION_ERRORS:{len(execution_errors)}")
print(f"EXECUTION_TIME:{end_time - start_time:.2f}")
if execution_errors:
    print("ERROR_SAMPLES:" + "|".join(execution_errors))
' 2>/dev/null)
    
    if [ -n "$PYTHON_BEHAVIOR" ]; then
        EXECUTED_MODULES=$(echo "$PYTHON_BEHAVIOR" | grep "EXECUTED_MODULES:" | cut -d: -f2)
        EXECUTION_ERRORS=$(echo "$PYTHON_BEHAVIOR" | grep "EXECUTION_ERRORS:" | cut -d: -f2)
        
        if [ "$EXECUTION_ERRORS" -eq 0 ] && [ "$EXECUTED_MODULES" -gt 0 ]; then
            echo "  ✅ Python modules execute successfully ($EXECUTED_MODULES tested)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        elif [ "$EXECUTED_MODULES" -gt 0 ]; then
            echo "  ⚠️  Python execution partial success: $EXECUTED_MODULES modules, $EXECUTION_ERRORS errors"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "  ❌ Python modules failed to execute"
            VALIDATION_ISSUES+=("Python modules cannot be executed successfully")
        fi
    else
        echo "  ❌ Python behavioral testing failed"
        VALIDATION_ISSUES+=("Python behavioral testing failed to execute")
    fi
fi

if [ $JS_FILES -gt 0 ] && [ $NODE_RUNTIME -eq 1 ]; then
    echo "Testing Node.js code execution behavior..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Test if main entry points can be loaded
    for entry in index.js app.js main.js server.js; do
        if [ -f "$entry" ]; then
            if timeout 5 node -e "require('./$entry')" >/dev/null 2>&1; then
                echo "  ✅ Entry point $entry loads successfully"
                PASSED_TESTS=$((PASSED_TESTS + 1))
                break
            else
                echo "  ❌ Entry point $entry failed to load"
                VALIDATION_ISSUES+=("Node.js entry point $entry fails to load")
                break
            fi
        fi
    done
fi

# Phase 4: Integration Point Validation
echo ""
echo "=== INTEGRATION POINT VALIDATION ==="

# Database Integration (actual usage analysis)
DB_FILES=$(find . -name "*.db" -o -name "*.sqlite" -o -name "*.duckdb" | wc -l)
if [ $DB_FILES -gt 0 ]; then
    echo "Testing database integration..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Check if code actually uses databases
    if [ $PYTHON_FILES -gt 0 ]; then
        DB_USAGE=$(find . -name "*.py" -exec grep -l "sqlite3\|database\|db\.\|\.db\|\.execute\|\.query" {} \; 2>/dev/null | wc -l)
        if [ $DB_USAGE -gt 0 ]; then
            echo "  ✅ Database integration detected in code ($DB_USAGE files)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "  ❌ Database files exist but no usage detected in code"
            VALIDATION_ISSUES+=("Database files exist but not used by code")
        fi
    fi
fi

# Container Integration (actual build testing)
if [ -f "Dockerfile" ]; then
    echo "Testing container integration..."
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if command -v docker >/dev/null 2>&1; then
        echo "  Testing Docker build capability..."
        if timeout 60 docker build -t temp-validation-$(date +%s) . >/dev/null 2>&1; then
            echo "  ✅ Container builds successfully"
            docker rmi temp-validation-$(date +%s) >/dev/null 2>&1
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "  ❌ Container build failed"
            VALIDATION_ISSUES+=("Dockerfile exists but container build fails")
        fi
    else
        echo "  ⚠️  Docker not available for testing"
    fi
fi

# Phase 5: Temporal Integration Analysis
echo ""
echo "=== TEMPORAL INTEGRATION ANALYSIS ==="

LATEST_CODE=$(find . -name "*.py" -o -name "*.js" -o -name "*.go" -o -name "*.java" | xargs ls -t | head -1 2>/dev/null)
LATEST_TEST=$(find . -path "*test*" -name "*.py" -o -path "*test*" -name "*.js" -o -path "*spec*" -name "*.js" | xargs ls -t | head -1 2>/dev/null)

if [ -n "$LATEST_CODE" ] && [ -n "$LATEST_TEST" ]; then
    CODE_TIME=$(stat -f%m "$LATEST_CODE" 2>/dev/null || stat -c%Y "$LATEST_CODE" 2>/dev/null)
    TEST_TIME=$(stat -f%m "$LATEST_TEST" 2>/dev/null || stat -c%Y "$LATEST_TEST" 2>/dev/null)
    
    if [ -n "$CODE_TIME" ] && [ -n "$TEST_TIME" ]; then
        if [ $CODE_TIME -gt $TEST_TIME ]; then
            DAYS_DIFF=$(( (CODE_TIME - TEST_TIME) / 86400 ))
            if [ $DAYS_DIFF -gt 7 ]; then
                echo "  ⚠️  Test lag detected: $DAYS_DIFF days behind implementation"
                VALIDATION_ISSUES+=("Tests are $DAYS_DIFF days behind latest implementation")
            else
                echo "  ✅ Tests are reasonably current ($DAYS_DIFF days lag)"
            fi
        else
            echo "  ✅ Tests are current with implementation"
        fi
    else
        echo "  ℹ️  Cannot determine file modification times"
    fi
else
    echo "  ℹ️  Insufficient data for temporal analysis"
fi

# Phase 6: System Health Assessment
echo ""
echo "=== SYSTEM HEALTH ASSESSMENT ==="

HEALTH_SCORE=0
if [ $TOTAL_TESTS -gt 0 ]; then
    HEALTH_SCORE=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
fi

echo "Validation Results:"
echo "  Total tests: $TOTAL_TESTS"
echo "  Passed tests: $PASSED_TESTS"
echo "  Health score: $HEALTH_SCORE%"
echo "  Issues found: ${#VALIDATION_ISSUES[@]}"

# Determine system status
if [ $HEALTH_SCORE -gt 80 ]; then
    SYSTEM_STATUS="Production Ready"
elif [ $HEALTH_SCORE -gt 60 ]; then
    SYSTEM_STATUS="Advanced Development"
elif [ $HEALTH_SCORE -gt 40 ]; then
    SYSTEM_STATUS="Basic Development"
else
    SYSTEM_STATUS="Needs Attention"
fi

echo "  System status: $SYSTEM_STATUS"

# Primary technology detection
PRIMARY_TECH="Mixed"
if [ $PYTHON_FILES -gt $JS_FILES ] && [ $PYTHON_FILES -gt $GO_FILES ]; then
    PRIMARY_TECH="Python"
elif [ $JS_FILES -gt $PYTHON_FILES ] && [ $JS_FILES -gt $GO_FILES ]; then
    PRIMARY_TECH="JavaScript"
elif [ $GO_FILES -gt 0 ]; then
    PRIMARY_TECH="Go"
fi

# Generate comprehensive report
REPORT_FILE="CODEBASE_VALIDATION_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT_FILE" << EOF
# Codebase Validation Report

**Date**: $(date)
**Path**: $(pwd)
**Method**: Code-as-Truth Behavioral Analysis

## Executive Summary
- **Primary Technology**: $PRIMARY_TECH
- **System Status**: $SYSTEM_STATUS
- **Health Score**: $HEALTH_SCORE%
- **Tests Performed**: $TOTAL_TESTS
- **Tests Passed**: $PASSED_TESTS

## Technology Stack Analysis
- **Python**: $PYTHON_FILES files (Runtime: $([ $PYTHON_RUNTIME -eq 1 ] && echo 'Available' || echo 'Unavailable'))
- **JavaScript/TypeScript**: $JS_FILES files (Runtime: $([ $NODE_RUNTIME -eq 1 ] && echo 'Available' || echo 'Unavailable'))
- **Go**: $GO_FILES files (Runtime: $([ $GO_RUNTIME -eq 1 ] && echo 'Available' || echo 'Unavailable'))
- **Java**: $JAVA_FILES files
- **Rust**: $RUST_FILES files

## Validation Issues
EOF

if [ ${#VALIDATION_ISSUES[@]} -eq 0 ]; then
    echo "No critical issues detected." >> "$REPORT_FILE"
else
    for issue in "${VALIDATION_ISSUES[@]}"; do
        echo "- $issue" >> "$REPORT_FILE"
    done
fi

cat >> "$REPORT_FILE" << EOF

## Recommendations
EOF

if [ $HEALTH_SCORE -lt 70 ]; then
    echo "### Priority Actions Required" >> "$REPORT_FILE"
    echo "- Address validation issues identified above" >> "$REPORT_FILE"
    echo "- Ensure all runtime dependencies are properly installed" >> "$REPORT_FILE"
    echo "- Update tests to match current implementation" >> "$REPORT_FILE"
fi

if [ "$PRIMARY_TECH" = "Python" ]; then
    echo "### Python-Specific Recommendations" >> "$REPORT_FILE"
    echo "- Ensure virtual environment is properly configured" >> "$REPORT_FILE"
    echo "- Consider adding type hints for better reliability" >> "$REPORT_FILE"
elif [ "$PRIMARY_TECH" = "JavaScript" ]; then
    echo "### JavaScript-Specific Recommendations" >> "$REPORT_FILE"
    echo "- Ensure node_modules are properly installed" >> "$REPORT_FILE"
    echo "- Consider migrating to TypeScript for better type safety" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

---
**Generated**: $(date)
**Health**: $HEALTH_SCORE% | **Status**: $SYSTEM_STATUS | **Issues**: ${#VALIDATION_ISSUES[@]}
EOF

# Create validation tracking
mkdir -p .validation
echo "$HEALTH_SCORE" > .validation/health_score
echo "$SYSTEM_STATUS" > .validation/system_status
echo "$PRIMARY_TECH" > .validation/primary_technology
echo "${#VALIDATION_ISSUES[@]}" > .validation/issue_count
echo "$(date)" > .validation/last_validation

echo ""
echo "=== VALIDATION COMPLETE ==="
echo "📊 Health Score: $HEALTH_SCORE%"
echo "📋 Report: $REPORT_FILE"
echo "📁 Tracking: .validation/ directory updated"
echo ""

if [ ${#VALIDATION_ISSUES[@]} -gt 0 ]; then
    echo "⚠️  Issues requiring attention:"
    for issue in "${VALIDATION_ISSUES[@]}"; do
        echo "   - $issue"
    done
else
    echo "✅ No critical issues detected"
fi

echo ""
echo "🎯 Validation complete. System status: $SYSTEM_STATUS"
```