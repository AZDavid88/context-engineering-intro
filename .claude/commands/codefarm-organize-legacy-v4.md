---
description: Comprehensive legacy codebase organization with intelligent categorization and systematic cleanup
allowed-tools: LS, Glob, Grep, Read, Write, Edit, MultiEdit, Bash
argument-hint: [project-path] - Path to legacy project requiring comprehensive organization and cleanup
---

# CODEFARM Legacy Codebase Organization (WORKING VERSION)

**Target Legacy Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Legacy Systematization)

**CodeFarmer (Legacy Strategist):** "I'll analyze chaotic legacy codebase using comprehensive file type coverage and intelligent content-based categorization to design systematic organization strategy that eliminates technical debt and improves development workflow."

**Critibot (Legacy Safety Validator):** "I'll ensure comprehensive validation across all file types, implement rigorous safety protocols, and verify that complex reorganization maintains all functionality through systematic evidence-based verification."

**Programmatron (Legacy Architect):** "I'll execute intelligent categorization using content analysis beyond pattern-matching, implement comprehensive directory structures, and systematically clean up technical debt while preserving all critical functionality."

**TestBot (Legacy Effectiveness Validator):** "I'll validate organizational improvements across all file types, test dependency preservation, and confirm comprehensive workflow enhancement through measurable systematic assessment."

---

## Framework: FPT + HTN + CoT (Hybrid Comprehensive Approach)

### Phase A: Comprehensive Legacy Analysis

**CodeFarmer Strategic Legacy Analysis:**

**CoT Reasoning**: "Legacy codebases require comprehensive analysis beyond simple pattern-matching. I must understand all file types, relationships, and organizational debt before applying systematic cleanup strategies."

**First Principles Investigation:**

**1. Complete File Type Discovery:**
- **Python files**: All .py files regardless of naming patterns
- **Documentation files**: .md, .txt, .rst, .adoc, .pdf files
- **Configuration files**: .json, .yaml, .yml, .toml, .ini, .env, .cfg files
- **Log files**: .log, .out, .err files with archival analysis
- **Data files**: .csv, .parquet, .db, .sqlite, .json data files
- **Temporary files**: *_backup, *_old, *.tmp, *.bak for cleanup assessment
- **Build artifacts**: .pyc, __pycache__, .DS_Store, node_modules patterns
- **Version control artifacts**: .git, .svn assessment for cleanup

**2. Intelligent Content-Based Analysis:**
- **Python script classification**: Main modules, utilities, tests, debug tools by content
- **Import relationship mapping**: Which files depend on others
- **Complexity assessment**: Lines of code, function count, class definitions
- **Age and usage analysis**: Last modified dates, file sizes
- **Purpose identification**: Entry points, libraries, one-off scripts

**3. Legacy Technical Debt Assessment:**
- **Scattered script proliferation**: Root-level chaos measurement
- **Documentation fragmentation**: Multiple planning/analysis documents
- **Configuration duplication**: Multiple config files and formats
- **Log accumulation**: Historical log files requiring archival
- **Backup file pollution**: Old versions and temporary files

### Phase B: Systematic Legacy Cleanup Implementation

**Programmatron Legacy Implementation Strategy:**

**CoT Reasoning**: "Legacy reorganization requires careful prioritization - safest moves first, then complex restructuring, with comprehensive validation at each step to prevent functionality loss."

**HTN Task Decomposition:**

**Goal: Transform Legacy Chaos → Systematic Organization**

**Subgoal A: Safety Foundation & Analysis**
1. **Complete Project Backup**: Timestamped backup with integrity validation
2. **Comprehensive File Census**: Count and categorize ALL file types
3. **Dependency Mapping**: Analyze import relationships and critical paths
4. **Risk Assessment**: Identify high-risk vs low-risk reorganization targets

**Subgoal B: Progressive Systematic Cleanup**
1. **Cleanup Phase**: Remove/archive temporary, backup, and build artifacts
2. **Script Organization**: Comprehensive Python file categorization using content analysis
3. **Documentation Systematization**: Organize all documentation by type and purpose
4. **Configuration Consolidation**: Systematic configuration file organization
5. **Log Management**: Archive historical logs, organize current logging

**Subgoal C: Advanced Structural Organization**
1. **Core Implementation Verification**: Ensure /src/ structure optimization
2. **Testing Infrastructure Validation**: Systematic test organization
3. **Dependency Resolution**: Update import paths and validate functionality
4. **Final Structure Validation**: Comprehensive organizational effectiveness verification

**Enhanced 8-Category Framework with Intelligence:**

**1. Core Implementation (Content Analysis)**
- **Criteria**: Main algorithms, business logic, primary system functionality
- **Detection**: Class definitions, main functions, complex import patterns
- **Target**: `/src/core/` or existing `/src/` optimization

**2. Feature Modules (Domain Analysis)**
- **Criteria**: Feature-specific implementations grouped by functionality
- **Detection**: Related import patterns, domain-specific naming, functional cohesion
- **Target**: `/src/[domain]/` directories with logical grouping

**3. Utility & Helpers (Cross-Reference Analysis)**
- **Criteria**: Support functions used across multiple modules
- **Detection**: High import frequency, utility naming patterns, shared functionality
- **Target**: `/src/utils/` with subcategorization by purpose

**4. Scripts & Automation (Intelligent Classification)**
- **Debug utilities**: Pattern matching + content analysis for debugging purpose
- **Validation tools**: Content analysis for testing/validation functionality  
- **Integration scripts**: Analysis for pipeline/integration purpose
- **Maintenance utilities**: Content analysis for cleanup/migration tools
- **Target**: `/scripts/{debug,validation,integration,utilities,maintenance}/`

**5. Configuration (Comprehensive Management)**
- **Project configuration**: Root-level configs (pyproject.toml, requirements.txt)
- **Application configuration**: Runtime configs, environment settings
- **Infrastructure configuration**: Docker, deployment, monitoring configs
- **Target**: Root level vs `/config/` based on scope and purpose

**6. Documentation (Complete Systematization)**
- **Analysis reports**: *_analysis*, *_report*, audit documents
- **Planning documents**: planning_*, methodology_*, strategy documents  
- **Infrastructure docs**: deployment, architecture, system guides
- **User documentation**: README, usage guides, API docs
- **Target**: `/docs/{reports,planning,infrastructure,guides}/`

**7. Testing (Comprehensive Organization)**
- **Unit tests**: Individual component testing
- **Integration tests**: Cross-component testing
- **System tests**: End-to-end testing
- **Research/archive tests**: Historical or experimental tests
- **Target**: `/tests/{unit,integration,system,archive}/`

**8. Data & Resources (Systematic Management)**
- **Active data**: Current datasets, databases
- **Historical data**: Archived data files  
- **Resources**: Static assets, reference materials
- **Logs**: Current logs vs archived logs
- **Target**: `/data/{active,archive}`, `/logs/{current,archive}/`

### Phase C: Comprehensive Legacy Validation

**TestBot Legacy Effectiveness Validation:**

**CoT Reasoning**: "Legacy reorganization success requires verification across all organizational aspects - file type coverage, dependency preservation, workflow improvement, and comprehensive cleanup effectiveness."

**Comprehensive Validation Protocol:**

**1. File Type Coverage Validation:**
- **Python files**: All .py files properly categorized by intelligent analysis
- **Documentation**: All .md/.txt files systematically organized
- **Configuration**: All config files appropriately placed
- **Logs**: Historical logs archived, current logs organized
- **Cleanup verification**: Temporary/backup files properly handled

**2. Functionality Preservation Testing:**
- **Import resolution**: All import statements resolve correctly
- **Core functionality**: Critical modules load and execute properly
- **Configuration access**: Application configs accessible from new locations
- **Test execution**: Existing tests run from organized structure

**3. Organizational Effectiveness Assessment:**
- **Root-level cleanup**: Dramatic reduction in root directory chaos
- **Navigation efficiency**: Improved file discovery and project navigation
- **Development workflow**: Reduced cognitive overhead and enhanced productivity
- **Maintenance improvement**: Clear organization supports ongoing development

**4. Technical Debt Reduction Validation:**
- **Script proliferation eliminated**: No scattered utility scripts
- **Documentation fragmentation resolved**: Systematic documentation organization
- **Configuration consolidation**: Reduced configuration file chaos
- **Cleanup completion**: Temporary and backup files properly managed

---

## Quality Gates & Success Criteria

### CODEFARM Legacy Organization Validation:

**CodeFarmer Legacy Strategy Validation:**
- [ ] Comprehensive file type analysis completed with intelligent content-based categorization
- [ ] All file types addressed using enhanced 8-category framework with legacy-specific intelligence
- [ ] Technical debt systematically identified and resolved through evidence-based cleanup
- [ ] Organization strategy specifically designed for legacy codebase complexity and chaos

**Critibot Legacy Safety Validation:**
- [ ] Complete project backup created with comprehensive integrity verification across all file types
- [ ] Progressive reorganization maintains functionality through rigorous testing at each phase
- [ ] Complex dependency relationships preserved through systematic import path management
- [ ] Safety protocols comprehensive enough for legacy codebase complexity and interconnection risks

**Programmatron Legacy Implementation Validation:**
- [ ] Intelligent categorization implemented using content analysis beyond simple pattern-matching limitations
- [ ] Comprehensive file type coverage achieved with systematic organization across all discovered file types
- [ ] HTN task decomposition executed systematically with progressive validation and risk management
- [ ] Enhanced 8-category framework applied with legacy-specific intelligence and comprehensive scope

**TestBot Legacy Effectiveness Validation:**
- [ ] All file types validated for proper organization and systematic categorization accuracy
- [ ] Comprehensive functionality preservation verified through extensive testing across reorganized structure
- [ ] Legacy technical debt measurably reduced through systematic cleanup and organizational improvements
- [ ] Development workflow enhancement validated through comprehensive navigation and productivity assessment

### Anti-Hallucination Framework:
- [ ] Every organizational decision backed by comprehensive evidence from actual legacy project analysis
- [ ] No file movements without verification through intelligent categorization logic and content analysis
- [ ] No cleanup actions without systematic backup and validation of file importance and dependencies
- [ ] No success claims without measurable validation across all organizational aspects and file types

### Legacy-Specific Success Metrics:
- **Root Directory Cleanup**: Target >90% reduction in root-level file chaos
- **File Type Coverage**: 100% of discovered file types systematically organized
- **Technical Debt Reduction**: Measurable improvement in project maintainability
- **Development Workflow Enhancement**: Demonstrable improvement in navigation and productivity
- **Comprehensive Validation**: All organizational aspects verified through systematic testing

---

## Cleanup & Maintenance Protocols

### Systematic Cleanup Procedures:

**1. Temporary File Management:**
- **Identification**: Find *_backup, *_old, *.tmp, *.bak files
- **Assessment**: Determine if files are safe for removal or need archival
- **Action**: Remove unnecessary files or archive important temporary files

**2. Build Artifact Cleanup:**
- **Python artifacts**: Remove __pycache__, *.pyc files
- **System artifacts**: Remove .DS_Store, thumbs.db files
- **Development artifacts**: Clean up IDE-specific files if appropriate

**3. Log File Management:**
- **Historical logs**: Archive logs older than specified threshold
- **Current logs**: Organize active log files in systematic structure
- **Log rotation**: Implement systematic log management for ongoing development

**4. Configuration Consolidation:**
- **Duplicate configs**: Identify and consolidate duplicate configuration files
- **Environment separation**: Organize configs by environment (dev/staging/prod)
- **Access validation**: Ensure application can access moved configuration files

---

**🎯 CODEFARM Legacy Systematization Complete**: Comprehensive legacy codebase organization using intelligent categorization, systematic cleanup, enhanced 8-category framework, and validated improvements across all file types with measurable technical debt reduction and development workflow enhancement.