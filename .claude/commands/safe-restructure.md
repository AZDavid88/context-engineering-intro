---
allowed-tools: Bash(mkdir:*), Bash(mv:*), Bash(cp:*), Bash(git:*), Edit, MultiEdit, Write, Read, Glob, Grep
description: Reorganize project structure without breaking existing functionality using safe migration
argument-hint: [project-path] | restructure current directory if no path provided
---

# Safe Project Restructuring - Zero Breakage Migration

**Context**: You are using the CodeFarm methodology to restructure a project safely. This implements IndyDevDan's three-directory system while preserving all existing functionality. Every step includes validation and rollback procedures.

## Pre-Restructure Safety Protocol

### 1. Backup & Git Safety
Before making ANY changes:

- **Git status check**: Use Bash tool with command: `git status` to check current state
- **Create safety commit**: Use Bash tool with command: `git add -A && git commit -m "Pre-restructure safety commit - $(date)"` to create backup
- **Document current working directory**: Use Bash tool with command: `pwd` to confirm location
- **Test current functionality**: Run existing tests to establish baseline

### 2. Import Dependencies Analysis
Critical: Identify what will break if we move files:

- **Find relative imports**: Use Grep tool with pattern "from \\." and type "py" in ${ARGUMENTS:-"."} to find relative imports, use head_limit 10
- **Find sys.path manipulations**: Use Grep tool with pattern "sys\\.path" and type "py" in ${ARGUMENTS:-"."} to find path manipulations
- **Find hardcoded paths**: Use Grep tool with pattern "__file__" and type "py" in ${ARGUMENTS:-"."} to find file path references, use head_limit 5
- **List test imports**: Use Glob tool with pattern "*test*.py" in ${ARGUMENTS:-"."} to find test files, then examine imports

### 3. Critical Files Inventory
Identify files we must NOT move to avoid breaking imports:

- **Core modules**: Use Glob tool with pattern "src/**/*.py" in ${ARGUMENTS:-"."} to find source modules, use head_limit 10
- **Main entry points**: Use Glob tool with patterns "main.py", "__main__.py", "app.py" in ${ARGUMENTS:-"."} to find entry points
- **Init files**: Use Glob tool with pattern "**/__init__.py" in ${ARGUMENTS:-"."} to find package initialization files

## Safe Restructuring Plan

### Phase 1: Create Structure (Safe - No Code Movement)

Create IndyDevDan's three essential directories:

Use Bash tool with the following commands to create directory structure:

```bash
# Create the three-directory system
mkdir -p ai_docs
mkdir -p specs  
mkdir -p scripts/debug
mkdir -p scripts/validation
mkdir -p scripts/utilities
mkdir -p logs/archive
```

Verify structure created: Use LS tool with path ${ARGUMENTS:-"."} to confirm directories were created

### Phase 2: Move Safe Files (No Import Dependencies)

Move files that don't affect imports:

#### 2a. Debug/Validation Scripts (Root Level Clutter)
Identify scattered debug files: Use Glob tool with patterns "debug_*.py", "test_*.py", "validate_*.py", "*_validation.py" in ${ARGUMENTS:-"."} to find debug scripts

For each debug/validation script:
1. **Test the script works**: `python script_name.py` 
2. **Move to scripts/debug/**: `mv script_name.py scripts/debug/`
3. **Test still works**: `python scripts/debug/script_name.py`
4. **Git commit**: `git add -A && git commit -m "Move debug script: script_name.py"`

#### 2b. Documentation Files
Move scattered docs: Use Glob tool with pattern "*.md" in ${ARGUMENTS:-"."} to find documentation files (exclude README.md and PLANNING.md manually)

Safe documentation moves:
- Move analysis/report files to `ai_docs/`
- Keep README.md and core planning docs in root
- Git commit after each successful move

#### 2c. Log Files and Archives
Archive old logs: Use Glob tool with patterns "*.log", "*.log.*" in ${ARGUMENTS:-"."} to find log files

Move logs to `logs/archive/` directory

### Phase 3: Create AI Context Files (Pure Addition)

#### 3a. Project Overview for AI Tools
Create `ai_docs/project_overview.md`:

```markdown
# Project Context for AI Tools

## Quick Start
- **Primary Language**: [Language]
- **Main Framework**: [Framework]
- **Core Business Logic**: [Location]
- **Testing**: [How to run tests]

## Key Directories
- `src/`: [Core application code]
- `tests/`: [Test files]
- `research/`: [Documentation and research]
- `scripts/`: [Development utilities]

## Import Patterns
- **Relative imports**: [How modules import each other]
- **External dependencies**: [Key external libraries]
- **Configuration**: [How settings work]

## Common Tasks
- **Development**: [How to run for development]
- **Testing**: [Test command]
- **Dependencies**: [How to install]
```

#### 3b. Dependencies Documentation
Create `ai_docs/dependencies.md` with:
- Third-party API documentation summaries
- Critical integration details
- Configuration requirements
- Authentication patterns

#### 3c. Project Conventions
Create `ai_docs/patterns.md` with:
- Coding conventions
- Architecture patterns
- Testing patterns
- Deployment procedures

### Phase 4: Advanced Restructuring (High Care Required)

⚠️ **WARNING**: Only proceed if Phase 1-3 completed successfully

#### 4a. Test File Organization
**Current test analysis**: Use Glob tool with pattern "*test*" in ${ARGUMENTS:-"."} to find test files

For misplaced test files:
1. **Verify test runs**: `python -m pytest test_file.py` or appropriate test command
2. **Check test imports**: Analyze what the test imports
3. **Update imports if needed**: Fix any broken import paths
4. **Move test file**: To appropriate `tests/` subdirectory
5. **Verify test still runs**: Confirm functionality preserved
6. **Git commit**: Individual commit per test file moved

#### 4b. Configuration File Organization
Move config files to `config/` directory IF they don't break imports:
- Settings files
- Environment configurations  
- Non-critical configuration files

### Phase 5: Create Specs Structure

#### 5a. Current Feature Documentation
Create `specs/current_features.md`:
- Document existing functionality
- Current architecture decisions
- Known technical debt

#### 5b. Future Plans
Create `specs/planned_features.md`:
- Roadmap items
- Architecture improvements
- Technical debt resolution plans

## Validation Protocol

### After Each Phase
1. **Git status check**: Use Bash tool with command: `git status` to check repository state
2. **Import validation**: Test critical imports still work
3. **Functionality test**: Run key scripts/tests
4. **Git commit**: Commit successful changes with descriptive message

### Full System Validation
After complete restructure, use Bash tool with the following commands:

```bash
# Test imports
python -c "import sys; sys.path.append('.'); import src; print('Core imports: OK')"

# Run test suite if available
python -m pytest tests/ || echo "No pytest setup found"

# Test key scripts
# [Run project-specific validation commands]
```

## Rollback Procedures

### If Any Step Fails
1. **Stop immediately**: Don't continue with failed step
2. **Git reset**: `git reset --hard HEAD~1` to undo last commit
3. **Analyze failure**: Understand what went wrong
4. **Fix import issues**: Update import paths if needed
5. **Retry with smaller change**: Make minimal change and test

### Emergency Full Rollback
Use Bash tool with commands for emergency full rollback:

```bash
# Return to pre-restructure state
git reset --hard [pre-restructure-commit-hash]

# Verify functionality restored
[Run critical tests]
```

## Success Criteria

### Structure Goals Achieved
- [ ] Three-directory system created (ai_docs, specs, scripts)
- [ ] Root directory cleaned (< 5 non-config files)
- [ ] Debug/validation scripts organized
- [ ] Documentation accessible to AI tools

### Functionality Preserved
- [ ] All imports work correctly
- [ ] Tests run successfully
- [ ] Key scripts function properly
- [ ] No broken dependencies

### AI Context Enhanced
- [ ] Project overview available
- [ ] Dependencies documented
- [ ] Patterns and conventions clear
- [ ] Quick context priming possible

## Post-Restructure Benefits

### For AI Tools
- Faster context establishment
- Clear separation of concerns
- Easy access to project knowledge
- Reduced confusion from scattered files

### For Development
- Cleaner development environment
- Organized utilities and scripts
- Better documentation accessibility
- Reduced cognitive overhead

---

This restructuring process transforms chaotic projects into organized, AI-friendly development environments while maintaining complete functional integrity.