---
description: Systematic project organization with comprehensive analysis and optional execution
allowed-tools: LS, Glob, Grep, Read, Write, Edit, MultiEdit, Bash, Task
argument-hint: [project-path] - Path to project requiring systematic organization
execution-flow: analysis-then-optional-execution
---

# CODEFARM Project Organization Command v2.1
**Systematic Analysis with Plan-Driven Execution**

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Enhanced Organization)

**CodeFarmer (Project Organization Analyst):** "I'll systematically analyze your project structure using comprehensive categorization frameworks and generate detailed implementation plans for evidence-based organization that enhances development workflow."

**Critibot (Safety & Planning Validator):** "I'll ensure thorough analysis accuracy, validate organizational strategies against systematic methodology, and guarantee safe execution with mandatory backup protocols and incremental validation."

**Programmatron (Implementation Plan Architect):** "I'll design detailed, executable organization plans with specific file movements, directory structures, categorization logic, and validation checkpoints for systematic implementation."

**TestBot (Execution Validator):** "I'll validate implementation plan feasibility, ensure comprehensive safety protocols, and verify all organizational changes maintain functionality and systematic compliance."

---

## PHASE A: COMPREHENSIVE ANALYSIS & PLANNING (Automatic Execution)

### Step 1: Project Structure Discovery & Analysis

**CodeFarmer Systematic Project Assessment:**

**Using LS tool to understand project scale and current organization**
Using LS tool with path: $ARGUMENTS to analyze project complexity and file distribution

**Using LS tool to identify root-level scattered files**
Using LS tool with path: $ARGUMENTS to discover files requiring organization

**Using comprehensive file type discovery for categorization intelligence**
Using Glob tool with path: $ARGUMENTS and pattern: "*.py" to identify Python implementation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.md" to find documentation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.txt" to locate configuration and requirement files
Using Glob tool with path: $ARGUMENTS and pattern: "*.json" to find configuration and data files
Using Glob tool with path: $ARGUMENTS and pattern: "*.yaml" to find YAML configuration files
Using Glob tool with path: $ARGUMENTS and pattern: "*.yml" to find additional YAML files
Using Glob tool with path: $ARGUMENTS and pattern: "*.sh" to identify shell scripts
Using Glob tool with path: $ARGUMENTS and pattern: "*.sql" to find database files
Using Glob tool with path: $ARGUMENTS and pattern: "*.csv" to locate data files

### Step 2: File Categorization & Dependency Analysis

**Programmatron Systematic Dependency Intelligence:**

**Using import pattern analysis to understand file relationships**
Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import", output_mode: "content", glob: "*.py" to map import dependencies
Using Grep tool with path: $ARGUMENTS, pattern: "import.*$ARGUMENTS", output_mode: "files_with_matches" to find files importing project modules

**Using file complexity assessment for organization priority**
Using Grep tool with path: $ARGUMENTS, pattern: "def |class ", output_mode: "count", glob: "*.py" to quantify implementation density
Using Grep tool with path: $ARGUMENTS, pattern: "TODO|FIXME|BUG|HACK", output_mode: "count", glob: "*.py" to assess technical debt patterns

**Using configuration and service pattern identification**
Using Grep tool with path: $ARGUMENTS, pattern: "config|Config|CONFIG", output_mode: "files_with_matches" to find configuration files
Using Grep tool with path: $ARGUMENTS, pattern: "api|API|endpoint|URL", output_mode: "files_with_matches" to identify external service integration

### Step 3: Current Organization Assessment & Strategy Development

**Critibot Evidence-Based Organization Analysis:**

**Using Read tool to analyze existing directory structure**
Reading existing directory organization to understand current patterns and identify systematic gaps

**File Organization Quality Assessment:**
- Root-level file scatter analysis and cognitive overhead measurement
- Directory structure systematic methodology compliance evaluation
- Import path complexity and maintainability assessment
- Configuration and documentation accessibility review

**Organization Strategy Development:**
- Evidence-based file categorization framework
- Systematic directory structure design
- Import dependency preservation strategy
- Safety protocol and backup procedure specification

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

### Step 6: Implementation Plan Generation

**CodeFarmer Systematic Plan Creation:**

**Creating comprehensive implementation plan document:**

Using Write tool to create detailed implementation plan at $ARGUMENTS/docs/organization_implementation_plan.md

**Implementation Plan Contents:**
1. **Current State Analysis**: Complete assessment of existing organization patterns with file categorization
2. **Proposed Organization Structure**: Evidence-based directory architecture with 8-category framework rationale
3. **File Movement Specifications**: Detailed list of every file to be moved with source, destination, and category justification
4. **Safety Protocols**: Backup procedures, incremental validation, and rollback instructions
5. **Validation Checkpoints**: Quality assurance steps throughout implementation with import dependency testing
6. **Import Dependency Updates**: Required import statement modifications with before/after examples
7. **Expected Benefits**: Quantified improvements in organization and development efficiency

**Plan Validation & Feasibility Assessment:**

**TestBot Implementation Plan Validation:**
- Verify all proposed file movements maintain functionality using categorization framework
- Validate directory structure supports systematic development methodology
- Confirm import dependency preservation throughout reorganization
- Assess implementation safety and rollback capability with incremental validation

---

## PHASE A COMPLETION: USER DECISION POINT

**CodeFarmer Plan Presentation:**

## 📋 ORGANIZATION ANALYSIS COMPLETE

### **Project Assessment Summary:**
- **Files Analyzed**: [Total file count] across [file type count] different types
- **Organization Issues Identified**: [Specific problems found with categorization assessment]
- **Proposed Organization Strategy**: [High-level approach using 8-category framework]

### **Implementation Plan Generated:**
**📁 Plan Location**: `$ARGUMENTS/docs/organization_implementation_plan.md`

### **Key Organizational Improvements:**
- **Root Directory Cleanup**: [Files to be organized by category]
- **Systematic Categorization**: [8-category directory structure proposed]
- **Development Efficiency**: [Expected workflow improvements with evidence]

### **Categorization Summary:**
- **Core Implementation**: [Files identified for core modules]
- **Feature Modules**: [Files for feature-specific organization]
- **Utilities**: [Helper and utility files identified]
- **Configuration**: [Config and settings files]
- **Documentation**: [Documentation files for /docs]
- **Testing**: [Test files for /tests organization]
- **Scripts**: [Automation and utility scripts]
- **Data/Resources**: [Data files and static resources]

---

## 🤔 EXECUTION DECISION

**Implementation plan has been created and is ready for review.**

**Would you like to execute the organization plan?**
- **Review the plan**: Check `$ARGUMENTS/docs/organization_implementation_plan.md`
- **Execute**: Reply 'yes' to proceed with automatic implementation
- **Manual Implementation**: Reply 'no' to implement manually using the plan

**⚠️ Note**: If you choose to execute, a complete project backup will be created automatically before any changes.

---

## PHASE B: PLAN-DRIVEN EXECUTION (User-Approved Only)

### EXECUTION SAFETY PROTOCOL

**Critibot Mandatory Safety Implementation:**

**STEP 1: COMPLETE PROJECT BACKUP**
```bash
# Create timestamped backup before any organization
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${PROJECT_PATH}.backup.organization.${BACKUP_TIMESTAMP}"
cp -r $ARGUMENTS "${BACKUP_DIR}"
echo "✅ Backup created: ${BACKUP_DIR}"
```

**STEP 2: BACKUP INTEGRITY VALIDATION**
```bash
# Validate backup completeness
cd "${BACKUP_DIR}"
ORIGINAL_FILE_COUNT=$(find $ARGUMENTS -type f | wc -l)
BACKUP_FILE_COUNT=$(find . -type f | wc -l)
if [ "$ORIGINAL_FILE_COUNT" -eq "$BACKUP_FILE_COUNT" ]; then
    echo "✅ Backup integrity verified: ${BACKUP_FILE_COUNT} files"
else
    echo "❌ Backup integrity check failed - execution halted"
    exit 1
fi
```

### SYSTEMATIC IMPLEMENTATION EXECUTION

**Programmatron Plan-Driven Implementation:**

**Reading implementation plan for systematic execution**
Using Read tool with file_path: $ARGUMENTS/docs/organization_implementation_plan.md to load execution specifications

**PHASE B1: Directory Structure Creation**

**Creating organized directory structure based on implementation plan:**
Using Bash tool to create systematic directory structure as specified in implementation plan using 8-category framework
Using LS tool to validate directory creation successful

**Specific Directory Creation Based on Categorization Framework:**
Using Bash tool to create `/scripts/debug/` directory for debug utilities
Using Bash tool to create `/scripts/validation/` directory for testing scripts
Using Bash tool to create `/scripts/integration/` directory for pipeline scripts
Using Bash tool to create `/scripts/utilities/` directory for maintenance tools
Using Bash tool to create `/docs/reports/` directory for analysis documentation
Using Bash tool to create `/docs/specifications/` directory for technical specs

**PHASE B2: File Movement by Category Priority**

**Executing file movements according to implementation plan priority:**

**Priority 1: Documentation and Configuration Files (Lowest Risk)**
Using Bash tool to move documentation files to organized structure as specified in plan
Using Bash tool to move configuration files to /config/ directory based on categorization
Using LS tool to validate documentation and configuration organization successful

**Priority 2: Utility and Script Files (Medium Risk)**
Using Bash tool to move debug_*.py files to `/scripts/debug/` directory
Using Bash tool to move validate_*.py and verify_*.py files to `/scripts/validation/` directory
Using Bash tool to move run_*.py files to `/scripts/integration/` directory
Using Bash tool to move test_*.py files to `/scripts/testing/` or `/tests/` directory
Using Bash tool to move setup_*.py and initialization files to `/scripts/utilities/` directory
Using LS tool to validate script organization successful

**Priority 3: Core and Feature Module Organization (Higher Risk)**
Using Bash tool to organize core implementation files based on categorization framework
Using Bash tool to group feature-specific modules according to implementation plan
Using LS tool to validate core and feature organization successful

**Priority 4: Import Statement Updates (If Required)**
Using implementation plan specifications to update import statements for moved files
Using Grep tool to identify files requiring import path updates
Using Edit tool to update import paths systematically based on new organization
Using Grep tool to validate all import statements resolve correctly

**PHASE B3: Comprehensive Validation & Verification**

**TestBot Systematic Validation Protocol:**

**Import Resolution Validation:**
Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import", output_mode: "content", head_limit: 50 to verify import statements resolve

**Functionality Preservation Testing:**
- Validate moved files retain all functionality using categorization framework
- Confirm no circular dependencies introduced through systematic organization
- Verify external integrations maintained after file movements

**Organization Effectiveness Assessment:**
- Confirm directory structure matches implementation plan with 8-category compliance
- Validate file categorization completed as specified in framework
- Assess cognitive overhead reduction achieved through systematic organization

### EXECUTION COMPLETION & DOCUMENTATION

**CodeFarmer Organization Success Documentation:**

**Creating comprehensive organization completion artifacts:**

**1. Organization Success Report**
Using Write tool to create $ARGUMENTS/docs/reports/organization_completion_report.md with:
- Complete implementation summary with before/after analysis using categorization framework
- File movement verification with source/destination confirmation and category justification
- Import statement updates documentation with specific examples
- Organization effectiveness metrics and benefits achieved through systematic structure

**2. Project Navigation Guide Update**
Using Write tool to create/update $ARGUMENTS/docs/project_navigation_guide.md with:
- New directory structure explanation using 8-category framework
- File location guide for common development tasks with category-based navigation
- Import path conventions for organized structure with examples
- Quick reference guide for organized project usage and maintenance

**3. Maintenance Procedures Documentation**
Using Write tool to create $ARGUMENTS/docs/organization_maintenance.md with:
- Guidelines for maintaining organized structure during ongoing development
- File placement decision framework using 8-category system
- Import path best practices for ongoing development in organized structure
- Periodic organization health check procedures and compliance validation

---

## SUCCESS VALIDATION & QUALITY ASSURANCE

### CODEFARM Multi-Agent Final Validation

**CodeFarmer Strategic Organization Validation:**
- [ ] File organization enhances development workflow and reduces cognitive overhead using 8-category framework
- [ ] Directory structure follows systematic methodology principles with clear categorization logic
- [ ] Organization strategy implemented exactly as specified in implementation plan with framework compliance
- [ ] Expected benefits achieved with measurable improvements in project navigation and category-based structure

**Critibot Safety & Execution Validation:**
- [ ] Complete project backup created and integrity verified before any modifications
- [ ] All file movements completed successfully without data loss using incremental validation
- [ ] Import statements resolve correctly and functionality preserved throughout reorganization
- [ ] Rollback capability available and tested for complete system restoration

**Programmatron Implementation Quality Validation:**
- [ ] Directory structure implemented according to evidence-based implementation plan with 8-category system
- [ ] File categorization completed systematically using logical and consistent framework patterns
- [ ] Import dependencies preserved and validated throughout reorganization process
- [ ] Implementation plan executed with 100% fidelity using categorization framework guidance

**TestBot Organization Effectiveness Validation:**
- [ ] All imports resolve correctly in new organized structure with category-based paths
- [ ] Existing functionality preserved through comprehensive testing and validation
- [ ] Performance maintained or improved through systematic organization using framework design
- [ ] Methodology compliance achieved with enhanced systematic development workflow

### Anti-Hallucination Framework

**Evidence-Based Implementation Validation:**
- [ ] Every claimed file movement verified through actual file system validation using categorization evidence
- [ ] Directory creation confirmed through direct file system inspection with framework compliance
- [ ] Import statement updates validated through comprehensive dependency testing
- [ ] Organization success verified through measurable improvement assessment using category metrics

**Implementation Integrity Validation:**
- [ ] Implementation plan execution documented with complete audit trail and categorization justification
- [ ] File movement accuracy verified against implementation plan specifications and framework logic
- [ ] Safety protocols executed exactly as designed with backup verification and incremental validation
- [ ] Success reporting based on concrete evidence of organizational improvement using systematic metrics

---

## CONFIDENCE SCORING & SUCCESS METRICS

### Systematic Organization Implementation Effectiveness:
- **Implementation Plan Quality**: ___/10 (comprehensive analysis with 8-category framework and detailed execution specifications)
- **Execution Safety**: ___/10 (backup creation, integrity validation, incremental validation, and rollback capability)
- **Organization Accuracy**: ___/10 (plan fidelity and systematic file categorization completion using framework)
- **Functionality Preservation**: ___/10 (import resolution and system integrity maintenance throughout reorganization)
- **Development Workflow Enhancement**: ___/10 (measurable improvements using category-based navigation and efficiency)
- **Overall Organization Success**: ___/10 (complete systematic organization with methodology and framework compliance)

**Threshold for successful organization**: Overall score ≥ 9/10 WITH Execution Safety = 10/10

### Organization Success Framework:
**After completing systematic organization:**
1. **Immediate Benefits Validation**: Confirm improved navigation and reduced complexity using category structure
2. **Long-term Sustainability**: Monitor organizational maintenance using 8-category framework guidelines
3. **Team Productivity Enhancement**: Assess workflow improvements and development efficiency with systematic metrics
4. **Methodology Integration**: Apply organizational lessons to future systematic development projects using framework

---

**🎯 CODEFARM Project Organization Complete**: Systematic two-phase organization with comprehensive analysis using 8-category framework, detailed implementation planning, user-controlled execution, mandatory safety protocols, incremental validation, and evidence-based success reporting ready for enhanced systematic development workflow and improved team productivity.