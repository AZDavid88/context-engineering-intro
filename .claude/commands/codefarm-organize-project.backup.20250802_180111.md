---
description: Systematic project organization with CODEFARM methodology compliance and scattered file coordination
allowed-tools: LS, Glob, Grep, Read, Write, Edit, MultiEdit, Task
argument-hint: [project-path] - Path to project requiring systematic file organization
---

# CODEFARM Project Organization Command v1.0
**Systematic File Organization & Methodology Compliance**

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Organization-Enhanced)

**CodeFarmer (Strategic Organization Architect):** "I'll analyze project structure patterns and design evidence-based organization strategies that enhance systematic development workflow and reduce cognitive overhead."

**Critibot (Safety & Structure Validator):** "I'll ensure every file movement preserves functionality, validates against systematic methodology requirements, and includes comprehensive rollback procedures."

**Programmatron (Organization Implementation Designer):** "I'll create detailed, executable organization plans with clear directory structures, file categorization logic, and validation checkpoints."

**TestBot (Organization Validation Controller):** "I'll validate every aspect of the organization plan against real project evidence, import dependencies, and systematic methodology compliance."

---

## Phase A: Comprehensive Project Discovery & File Intelligence Gathering

### Step 1: Project Structure & Scale Analysis

**CodeFarmer Strategic Project Assessment:**

**Using LS tool to understand project structure and scale**
Using LS tool with path: $ARGUMENTS to analyze project organization complexity

**Using LS tool to identify root-level scattered files**
Using LS tool with path: $ARGUMENTS to discover files requiring organization

**Using Glob tool to comprehensive file type discovery**
Using Glob tool with path: $ARGUMENTS and pattern: "*.py" to find Python implementation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.md" to find documentation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.txt" to find configuration and requirement files
Using Glob tool with path: $ARGUMENTS and pattern: "*.json" to find configuration and data files
Using Glob tool with path: $ARGUMENTS and pattern: "*.yaml" to find configuration files
Using Glob tool with path: $ARGUMENTS and pattern: "*.yml" to find configuration files
Using Glob tool with path: $ARGUMENTS and pattern: "*.sh" to find shell scripts
Using Glob tool with path: $ARGUMENTS and pattern: "*.sql" to find database files
Using Glob tool with path: $ARGUMENTS and pattern: "*.csv" to find data files

### Step 2: File Usage Pattern & Import Dependency Analysis

**Programmatron Systematic Dependency Intelligence:**

**Using Grep tool to analyze import patterns and dependencies**
Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import", output_mode: "content", and glob: "*.py" to map import relationships
Using Grep tool with path: $ARGUMENTS, pattern: "import.*$ARGUMENTS", output_mode: "files_with_matches" to find files importing project modules

**Using Grep tool to identify file usage patterns**
Using Grep tool with path: $ARGUMENTS, pattern: "def |class ", output_mode: "count", and glob: "*.py" to quantify implementation complexity
Using Grep tool with path: $ARGUMENTS, pattern: "TODO|FIXME|BUG|HACK", output_mode: "count", and glob: "*.py" to assess technical debt indicators

**Using Grep tool to discover configuration and external service patterns**
Using Grep tool with path: $ARGUMENTS, pattern: "config|Config|CONFIG", output_mode: "files_with_matches" to find configuration files
Using Grep tool with path: $ARGUMENTS, pattern: "api|API|endpoint|URL", output_mode: "files_with_matches" to find external service integration files

### Step 3: Current Organization Assessment & Gap Analysis

**Critibot Organization Validation Analysis:**

**Using Read tool to analyze existing directory structure**
Reading existing directory organization to understand current patterns and identify systematic gaps

**File Organization Quality Assessment:**
- Root-level file scatter analysis and impact measurement
- Directory structure methodology compliance evaluation
- Import path complexity and maintainability assessment
- Configuration file organization and accessibility review

---

## Phase B: Evidence-Based Organization Strategy Development

### Step 4: Systematic File Categorization Framework

**CodeFarmer Evidence-Based Categorization Strategy:**

**PRIMARY CATEGORIZATION LOGIC:**

**1. CORE IMPLEMENTATION FILES**
- **Criteria**: Main business logic, core algorithms, primary system functionality
- **Target Directory**: `/src/core/` or `/core/`
- **Evidence**: Files with high import frequency, main classes, core business logic

**2. FEATURE/MODULE IMPLEMENTATION FILES**
- **Criteria**: Specific feature implementations, module-specific logic
- **Target Directory**: `/src/[feature-name]/` or `/[feature-name]/`
- **Evidence**: Files grouped by functional domain, related import patterns

**3. UTILITY & HELPER FILES**
- **Criteria**: Support functions, utilities, common helpers
- **Target Directory**: `/src/utils/` or `/utils/`
- **Evidence**: Files imported across multiple modules, general-purpose functions

**4. CONFIGURATION & SETTINGS FILES**
- **Criteria**: Configuration, environment variables, settings
- **Target Directory**: `/config/` or `/settings/`
- **Evidence**: Files containing configuration patterns, environment setup

**5. DOCUMENTATION FILES**
- **Criteria**: Documentation, specifications, guides
- **Target Directory**: `/docs/`
- **Evidence**: `.md`, `.txt`, `.rst` files with documentation patterns

**6. TESTING FILES**
- **Criteria**: Test files, test utilities, test configurations
- **Target Directory**: `/tests/`
- **Evidence**: Files with test patterns, testing framework usage

**7. SCRIPTS & AUTOMATION FILES**
- **Criteria**: Utility scripts, automation, maintenance tools
- **Target Directory**: `/scripts/`
- **Evidence**: Executable files, automation patterns, utility functions

**8. DATA & RESOURCES FILES**
- **Criteria**: Data files, resources, static assets
- **Target Directory**: `/data/` or `/resources/`
- **Evidence**: Data file extensions, resource patterns

### Step 5: Directory Structure Design & Methodology Compliance

**Programmatron Systematic Directory Architecture:**

**EVIDENCE-BASED DIRECTORY STRUCTURE:**

```
PROJECT_ROOT/
├── src/                    # Core implementation (if not already organized)
│   ├── core/              # Core business logic and algorithms
│   ├── [domain-modules]/  # Feature-specific modules
│   └── utils/             # Shared utilities and helpers
├── docs/                  # All documentation and specifications
├── tests/                 # Complete testing infrastructure
├── config/                # Configuration and environment files
├── scripts/               # Utility and maintenance scripts
├── data/                  # Data files and resources
└── [project-specific]/    # Additional directories based on project needs
```

**METHODOLOGY COMPLIANCE VALIDATION:**
- Each directory serves single, clear responsibility
- File organization enhances rather than complicates import paths
- Structure supports systematic development workflow
- Organization reduces cognitive overhead and improves navigation

---

## Phase C: Safe Organization Implementation Planning

### Step 6: Comprehensive Safety & Backup Strategy

**Critibot Safety-First Organization Protocol:**

**PRE-ORGANIZATION SAFETY SETUP:**

**Complete Project Backup Creation:**
```bash
# Create timestamped backup before any organization
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${PROJECT_PATH}.backup.${BACKUP_TIMESTAMP}"
cp -r $ARGUMENTS "${BACKUP_DIR}"

# Validate backup integrity
cd "${BACKUP_DIR}"
# Run basic validation (if tests exist)
find . -name "*.py" -exec python -m py_compile {} \;
```

**INCREMENTAL ORGANIZATION WITH VALIDATION:**

**Step 6A: Directory Structure Creation**

**Creating organized directory structure based on evidence-based categorization:**

Using Write tool to create `/scripts/debug/` directory with proper organization
Using Write tool to create `/scripts/validation/` directory for testing scripts
Using Write tool to create `/scripts/integration/` directory for pipeline scripts
Using Write tool to create `/scripts/utilities/` directory for maintenance tools
Using Write tool to create `/docs/reports/` directory for analysis documentation
Using Write tool to create `/docs/specifications/` directory for technical specs

**Step 6B: File Movement with Import Validation**

**Moving debug and development scripts to organized structure:**

Using Edit tool to move debug_*.py files to `/scripts/debug/` directory
Using Edit tool to move cleanup_*.py and fix_*.py files to `/scripts/maintenance/` directory
Using Edit tool to move validate_*.py and verify_*.py files to `/scripts/validation/` directory
Using Edit tool to move comprehensive_*_validation.py files to `/scripts/validation/` directory
Using Edit tool to move test_*_integration.py files to `/scripts/integration/` directory
Using Edit tool to move run_*.py files to `/scripts/integration/` directory

**Step 6C: Import Statement Updates & Validation**

**Systematically updating import references to reflect new file locations:**

Using Grep tool with path: $ARGUMENTS, pattern: "from.*debug_|import.*debug_" and output_mode: "files_with_matches" to find import references
Using Grep tool with path: $ARGUMENTS, pattern: "from.*validate_|import.*validate_" and output_mode: "files_with_matches" to find validation imports
Using Grep tool with path: $ARGUMENTS, pattern: "from.*run_|import.*run_" and output_mode: "files_with_matches" to find integration imports

For each file containing imports to moved scripts:
- Using Read tool to analyze current import statements
- Using Edit tool to update import paths to reflect new organized structure
- Using comprehensive validation to ensure functionality preservation

### Step 7: File Movement & Import Update Strategy

**TestBot Validation-Driven Organization Execution:**

**SYSTEMATIC FILE MOVEMENT PROTOCOL:**

**Phase 1: Create Directory Structure & Documentation Organization**

**Creating systematic directory structure:**

Using Write tool to create organized directory structure in $ARGUMENTS/scripts/debug/
Using Write tool to create organized directory structure in $ARGUMENTS/scripts/validation/  
Using Write tool to create organized directory structure in $ARGUMENTS/scripts/integration/
Using Write tool to create organized directory structure in $ARGUMENTS/scripts/utilities/
Using Write tool to create organized directory structure in $ARGUMENTS/docs/reports/
Using Write tool to create organized directory structure in $ARGUMENTS/docs/specifications/

**Moving documentation files to organized structure:**

Using Edit tool to move *.md analysis and report files to $ARGUMENTS/docs/reports/
Using Edit tool to move *.md specification files to $ARGUMENTS/docs/specifications/
Using Edit tool to move planning and methodology files to appropriate docs subdirectories

**Phase 2: Systematic File Movement by Category**

**Moving debug and development scripts (lowest risk):**

Using MultiEdit tool to move all debug_*.py files from $ARGUMENTS/ to $ARGUMENTS/scripts/debug/
Using MultiEdit tool to move cleanup_*.py and fix_*.py files to $ARGUMENTS/scripts/maintenance/

**Moving validation and testing scripts:**  

Using MultiEdit tool to move validate_*.py and verify_*.py files to $ARGUMENTS/scripts/validation/
Using MultiEdit tool to move comprehensive_*_validation.py files to $ARGUMENTS/scripts/validation/
Using MultiEdit tool to move test_*_integration.py files to $ARGUMENTS/scripts/integration/

**Moving integration and pipeline scripts:**

Using MultiEdit tool to move run_*.py files to $ARGUMENTS/scripts/integration/
Using MultiEdit tool to move migrate_*.py files to $ARGUMENTS/scripts/utilities/

**Phase 3: Comprehensive Import Statement Updates**

**Analyzing import dependencies before updates:**

Using Grep tool with path: $ARGUMENTS, pattern: "from.*debug_|import.*debug_|from.*validate_|import.*validate_|from.*run_|import.*run_" and output_mode: "files_with_matches" to identify all files with imports to moved scripts

**Systematically updating import statements:**

For each file containing imports to moved scripts:
- Using Read tool to analyze current import statements and dependencies
- Using Edit tool to update import paths: "from debug_" → "from scripts.debug.debug_"  
- Using Edit tool to update import paths: "from validate_" → "from scripts.validation.validate_"
- Using Edit tool to update import paths: "from run_" → "from scripts.integration.run_"
- Using comprehensive validation after each import update batch

**Phase 4: Comprehensive Integration Validation & Testing**

**Import resolution validation:**

Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import" and output_mode: "content" and head_limit: 50 to validate all import statements resolve correctly

**Functionality preservation testing:**

For each moved script category:
- Using Read tool to validate moved files retain all functionality
- Testing import resolution in organized structure
- Validating no circular dependencies introduced
- Confirming all external integrations maintained

---

## Phase D: Organization Validation & Methodology Compliance

### Step 8: Comprehensive Organization Success Validation

**CODEFARM Multi-Agent Final Organization Validation:**

**CodeFarmer Strategic Organization Validation:**
- [ ] File organization reduces cognitive overhead and improves navigation
- [ ] Directory structure follows systematic methodology principles
- [ ] Import paths are logical and maintainable
- [ ] Organization enhances rather than complicates development workflow

**Critibot Safety & Functionality Validation:**
- [ ] All file movements completed without functionality loss
- [ ] Import statements updated correctly and resolve properly
- [ ] No circular dependencies or import conflicts introduced
- [ ] Backup and rollback procedures tested and functional

**Programmatron Implementation Quality Validation:**
- [ ] Directory structure implemented according to evidence-based design
- [ ] File categorization follows logical and consistent patterns
- [ ] Import updates preserve all existing functionality
- [ ] Organization implementation is complete and systematic

**TestBot Organization Effectiveness Validation:**
- [ ] All imports resolve correctly in new structure
- [ ] Existing functionality preserved through comprehensive testing
- [ ] Performance maintained or improved through better organization
- [ ] Methodology compliance achieved through systematic structure

### Step 9: Post-Organization Documentation & Process Enhancement

**Documentation & Knowledge Transfer:**

**Creating comprehensive documentation of the new organization structure:**

**ORGANIZATION ARTIFACTS GENERATION:**

**1. Organization Summary Report Creation:**

Using Write tool to create $ARGUMENTS/docs/reports/organization_summary.md with:
- Complete analysis of files moved and categorization rationale
- Before/after directory structure comparison with file counts and locations
- Import path changes documentation with specific examples
- Methodology compliance improvements achieved with quantified metrics

**2. Navigation Guide Creation:**

Using Write tool to create $ARGUMENTS/docs/project_navigation_guide.md with:
- Directory structure explanation and purpose for each organized section
- File location guide for common development tasks (debugging, validation, integration)
- Import path conventions and best practices for organized structure
- Quick reference guide for finding specific functionality in new structure

**3. Maintenance Procedures Documentation:**

Using Write tool to create $ARGUMENTS/docs/organization_maintenance.md with:
- Guidelines for maintaining organized structure during ongoing development
- File placement decision framework for new files (debug → scripts/debug/, etc.)
- Import path best practices for ongoing development in organized structure
- Periodic organization health check procedures and compliance validation

**4. Import Path Reference Guide:**

Using Write tool to create $ARGUMENTS/docs/import_path_reference.md with:
- Complete mapping of old import paths to new organized paths
- Code examples showing correct import syntax for organized structure
- Common import patterns and troubleshooting guide
- Integration testing commands for validating import resolution

---

## Anti-Hallucination & Evidence-Based Validation Framework

### Evidence-Based Organization Requirements:
- [ ] No file movements without concrete analysis of usage patterns and dependencies
- [ ] No directory structure decisions without validation against actual project needs
- [ ] No import updates without comprehensive testing and validation procedures
- [ ] No organization confidence without systematic methodology compliance verification

### Safety-First Organization Validation:
- [ ] All organization steps based on evidence from actual project analysis
- [ ] File movement strategies validated against import dependency mapping
- [ ] Organization effectiveness proven through reduced complexity and improved navigation
- [ ] Implementation approach ensures zero functionality loss through comprehensive validation

---

## Success Criteria & Confidence Assessment

### Systematic Organization Effectiveness:
- **Project Discovery Quality**: ___/10 (comprehensive analysis of current structure and files)
- **Categorization Logic Validity**: ___/10 (evidence-based file classification and grouping)
- **Organization Strategy Effectiveness**: ___/10 (systematic approach addressing genuine organizational needs)
- **Implementation Safety**: ___/10 (comprehensive backup, rollback, and validation procedures)
- **Methodology Compliance Enhancement**: ___/10 (improved systematic development workflow capability)
- **Overall Organization Command Quality**: ___/10 (systematic project organization effectiveness)

**Threshold for implementation**: Overall score ≥ 8.5/10 WITH Implementation Safety ≥ 9/10

### Organization Implementation Decision Framework:
**If score ≥ 8.5/10 + Safety ≥ 9/10:**
1. Execute comprehensive project organization with safety-first approach
2. Implement file movements with continuous validation and rollback capability
3. Monitor organization effectiveness against systematic methodology compliance
4. Document organization template for future project use

**If score < 8.5/10 OR Safety < 9/10:**
1. Refine organization strategy with additional evidence gathering and analysis
2. Enhance safety procedures and validation frameworks
3. Validate organization plans against systematic methodology requirements
4. Ensure comprehensive understanding and safety before execution

---

**🎯 CODEFARM Project Organization Complete**: Systematic file organization command with evidence-based categorization, comprehensive safety protocols, and methodology compliance enhancement ready for complex project systematization and scattered file coordination.