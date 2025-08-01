---
description: WORKING VERSION - Reorganize legacy codebase to systematic methodology standards
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS, Task
argument-hint: [legacy-project-path] - Path to legacy project for systematic reorganization
---

# CODEFARM Legacy Code Systematization & Methodology Integration (WORKING VERSION)

**Target Legacy Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Legacy Systematization Specialists)

**CodeFarmer (Systematization Strategist):** "I'll analyze your legacy codebase at $ARGUMENTS and create a systematic reorganization plan that preserves functionality while implementing methodology standards."

**Critibot (Safety & Validation Controller):** "I'll ensure every reorganization step in $ARGUMENTS preserves existing functionality and prevents breaking changes through rigorous validation protocols."

**Programmatron (Refactoring Architect):** "I'll design safe refactoring strategies for $ARGUMENTS and systematic code organization that aligns with your 7-phase development methodology."

**TestBot (Integration & Safety Validator):** "I'll validate every reorganization step in $ARGUMENTS maintains system integrity and creates comprehensive testing for the systematized codebase."

---

## Phase A: Legacy Codebase Analysis & Systematization Planning

### Step 1: Current Architecture & Organization Assessment

**CodeFarmer Legacy Structure Analysis:**

Let me analyze the current project structure at: $ARGUMENTS

**Directory Structure Analysis:**

Using LS tool to analyze current project directory structure at path: $ARGUMENTS

**File Organization Assessment:**

Using Glob tool to find Python files with path: $ARGUMENTS and pattern: "*.py"
Using Glob tool to find JavaScript files with path: $ARGUMENTS and pattern: "*.js"
Using Glob tool to find TypeScript files with path: $ARGUMENTS and pattern: "*.ts"

**Existing Systematic Elements Discovery:**

Using Glob tool to find README files with path: $ARGUMENTS and pattern: "README*"
Using Glob tool to find planning files with path: $ARGUMENTS and pattern: "planning*"
Using Glob tool to find specs directory with path: $ARGUMENTS and pattern: "specs*"
Using Glob tool to find research directory with path: $ARGUMENTS and pattern: "research*"
Using Glob tool to find docs directory with path: $ARGUMENTS and pattern: "docs*"

**Current State Assessment:**
- **Project Structure**: [Current directory organization analysis]
- **Code Organization**: [Module and file organization patterns]
- **Documentation Status**: [Existing documentation and planning elements]
- **Systematic Elements**: [Any existing methodology compliance]

### Step 2: Dependency & Import Analysis

**Programmatron Dependency Structure Assessment:**

**Internal Dependency Mapping:**

Using Grep tool with path: $ARGUMENTS, pattern: "^import\\.|^from\\s+\\." and output_mode: "content" and head_limit: 20

**External Dependency Analysis:**

Using Grep tool with path: $ARGUMENTS, pattern: "^import\\s+[^.]|^from\\s+[^.]" and output_mode: "content" and head_limit: 15

**Path Dependencies & Risks:**

Using Grep tool with path: $ARGUMENTS, pattern: "os\\.path|__file__|\\./|\\.\\./|sys\\.path" and output_mode: "files_with_matches"

**Dependency Risk Assessment:**
- **Internal Dependencies**: [Current module interdependencies]
- **External Dependencies**: [Third-party library usage patterns] 
- **Path Dependencies**: [Hardcoded paths that could break during reorganization]
- **Import Complexity**: [Complex import patterns requiring careful migration]

### Step 3: Methodology Standards Gap Analysis

**Critibot Systematization Requirements Assessment:**

**7-Phase Methodology Compliance Analysis:**

### Systematic Structure Requirements:

**Phase 1-2 Foundation:**
- [ ] `planning_prp.md` - Master project plan under 200 lines
- [ ] `research/` directory structure with technology-specific subdirectories
- [ ] Project vision and requirements documentation

**Phase 3-4 Architecture:**
- [ ] `specs/` directory with architecture and technical specifications  
- [ ] `specs/system_architecture.md` - System design documentation
- [ ] `specs/technical_stack.md` - Technology choices and validation

**Phase 5-6 Implementation:**
- [ ] Organized source code structure (`src/`, `lib/`, or equivalent)
- [ ] Clear module organization with consistent naming
- [ ] Implementation documentation and decision records

**Phase 7 Validation:**
- [ ] `tests/` directory mirroring source structure
- [ ] Comprehensive testing strategy and coverage
- [ ] Validation and quality assurance documentation

**Gap Analysis Results:**
- **Missing Systematic Elements**: [List of methodology components not present]
- **Partial Implementation**: [Elements partially implemented but not compliant]
- **Reorganization Requirements**: [Structural changes needed for full compliance]

---

## Phase B: Systematic Implementation & Safe Migration

### Step 4: Methodology-Compliant Structure Creation

**CodeFarmer Systematic Structure Implementation:**

**Creating Methodology-Compliant Directory Structure:**

I will create the systematic directory structure at: $ARGUMENTS

**Foundation Directories:**
- Create specs directory at: $ARGUMENTS/specs
- Create research directory at: $ARGUMENTS/research
- Create phases directory at: $ARGUMENTS/phases
- Create phases/current directory at: $ARGUMENTS/phases/current
- Create phases/completed directory at: $ARGUMENTS/phases/completed
- Create tests directory at: $ARGUMENTS/tests (if not existing)
- Create docs directory at: $ARGUMENTS/docs (if not existing)

**Source Code Organization (if needed):**
- Analyze existing src structure and enhance as needed
- Create additional systematic organization within existing structure
- Maintain existing functionality while adding systematic elements

### Step 5: Foundation Documentation Generation

**Programmatron Systematic Documentation Creation:**

**1. Master Planning Document Creation:**

I will analyze the existing codebase structure and create planning_prp.md at: $ARGUMENTS/planning_prp.md

**2. System Architecture Documentation:**

Create system architecture documentation at: $ARGUMENTS/specs/system_architecture.md

**3. Technical Stack Documentation:**

Create technical stack documentation at: $ARGUMENTS/specs/technical_stack.md

**4. Research Directory Population:**

Based on technologies discovered in the codebase, I will create appropriate research subdirectories and populate them with relevant documentation.

### Step 6: Safe Code Migration Strategy

**TestBot Safety-First Migration Protocol:**

**Pre-Migration Safety Validation:**

**Backup Creation:**
Create comprehensive backup of project before any structural changes

**Current Import Documentation:**
Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import" and output_mode: "content" to document all current imports

**Testing Baseline Establishment:**
Using Glob tool to find test files with path: $ARGUMENTS and pattern: "*test*.py"

If tests exist, run current test suite to establish functionality baseline

**Safe Migration Process:**

### Three-Phase Safe Reorganization:

**Phase I: Documentation & Testing Foundation**
1. Create methodology-compliant documentation structure
2. Establish comprehensive testing before any code moves
3. Document all current dependencies and integrations
4. Create rollback strategy and backup procedures

**Phase II: Incremental Structure Migration**
1. Create new systematic directory structure alongside existing
2. Copy (don't move) files to new structure with updated imports
3. Test new structure while maintaining old structure functionality
4. Validate all integrations work with new organization

**Phase III: Legacy Structure Retirement**
1. Gradually redirect imports from old to new structure
2. Remove old structure files only after new structure validation
3. Update all external references and configuration files
4. Final integration testing and validation

### Step 7: Code Organization & Module Systematization

**Programmatron Systematic Code Organization:**

**Core Module Identification & Organization:**

Using Grep tool with path: $ARGUMENTS, pattern: "class |def " and output_mode: "files_with_matches" to identify core modules

For each identified core module:
- Analyze functionality and dependencies
- Plan organized placement in systematic structure
- Create updated import paths maintaining functionality
- Test module integration independently

**Utility & Helper Function Consolidation:**

Using Grep tool with path: $ARGUMENTS, pattern: "def.*helper|def.*util|def.*tool" and output_mode: "content" and head_limit: 20

- Consolidate utility functions into organized modules
- Remove code duplication discovered during reorganization
- Create consistent naming and interface patterns
- Validate all utility integrations

**Configuration & Environment Management:**

Using Grep tool with path: $ARGUMENTS, pattern: "config|environ|settings" and output_mode: "files_with_matches"

- Organize configuration management into consistent structure
- Consolidate environment variable handling
- Create systematic configuration patterns
- Validate all configuration integrations

---

## Phase C: Integration Validation & Quality Assurance

### Step 8: Comprehensive Integration Testing

**TestBot Systematic Validation Protocol:**

**Import Resolution Validation:**

For each Python file in the systematized structure:
- Use Read tool to validate imports resolve correctly
- Test module loading and dependency resolution
- Verify no circular dependencies introduced
- Confirm all external integrations maintained

**Functionality Preservation Testing:**

**API Compatibility Validation:**
- Test all external interfaces maintained after reorganization
- Validate data processing unchanged
- Confirm performance impact minimal
- Test all external service connections

**Integration Health Assessment:**
- Validate all imports resolve correctly after reorganization
- Test all Python modules can be imported successfully
- Check for any broken file paths or configuration references
- Run comprehensive test suite on reorganized code

### Step 9: Process Integration & Methodology Implementation

**CodeFarmer Process Integration Strategy:**

**Systematic Development Workflow Implementation:**

### Methodology Integration Components:

**Development Command Integration:**
- Configure project to work with your 7-phase development commands
- Create project-specific configuration for systematic development  
- Establish systematic workflow for future feature development
- Document how to use development methodology with this project

**Quality Assurance Integration:**
- Implement systematic testing strategies aligned with methodology
- Create code quality gates and validation processes
- Establish continuous integration compatible with systematic approach
- Document quality standards and enforcement procedures

**Documentation & Knowledge Management:**
- Create systematic documentation update procedures
- Establish research directory maintenance protocols
- Implement specification update workflows for future changes
- Create onboarding documentation for methodology adoption

---

## Phase D: Validation & Documentation Generation

### Step 10: Success Validation & Deliverable Creation

**Critibot Final Validation & Report Generation:**

**Systematization Success Criteria Validation:**

### Methodology Compliance Validation:

**Structural Compliance:**
- [ ] All 7-phase methodology directory structure implemented
- [ ] Planning and specification documents complete and under line limits
- [ ] Research directories populated with relevant technology documentation
- [ ] Testing infrastructure organized and comprehensive

**Functional Preservation:**
- [ ] All existing functionality maintained after reorganization
- [ ] External integrations working without changes required
- [ ] Performance impact minimal or improved
- [ ] No breaking changes introduced during systematization

**Process Integration:**
- [ ] Project compatible with systematic development commands
- [ ] Clear workflow for future development using methodology
- [ ] Documentation supports methodology-based development approach
- [ ] Quality gates established for ongoing systematic development

### Step 11: Comprehensive Documentation Generation

**Programmatron Report & Documentation Creation:**

**1. Systematization Implementation Report**

Create systematization report at: $ARGUMENTS/systematization_report.md

**2. Methodology Compliance Checklist**

Create methodology compliance checklist at: $ARGUMENTS/methodology_compliance.md

**3. Systematic Development Workflow Guide**

Create systematic workflow guide at: $ARGUMENTS/systematic_workflow.md  

**4. Migration & Integration Log**

Create detailed migration log at: $ARGUMENTS/migration_log.md

---

## Quality Gates & Validation

### CODEFARM Systematization Validation

**CodeFarmer Strategic Integration Validation:**
- [ ] Legacy project successfully reorganized to methodology standards without functionality loss
- [ ] Systematic structure enhances rather than complicates development workflow
- [ ] Integration with 7-phase development methodology seamless and beneficial
- [ ] Documentation and planning elements support ongoing systematic development

**Critibot Safety & Risk Validation:**
- [ ] Every reorganization step validated to preserve existing functionality
- [ ] No breaking changes introduced during systematization process
- [ ] Rollback procedures tested and available if needed
- [ ] External integrations maintained without requiring partner changes

**Programmatron Implementation Quality Validation:**
- [ ] Code organization follows established patterns and conventions
- [ ] Refactoring improved code quality without changing behavior
- [ ] Module organization enhances maintainability and understanding
- [ ] Technical implementation supports long-term systematic development

**TestBot Functional Integrity Validation:**
- [ ] Comprehensive testing validates no regression in functionality
- [ ] All integrations tested and confirmed working after reorganization
- [ ] Performance impact measured and acceptable
- [ ] Quality assurance processes established for ongoing development

### Anti-Hallucination & Safety Validation
- [ ] No code reorganization without comprehensive backup and rollback strategy
- [ ] No structural changes without thorough integration testing
- [ ] No methodology compliance claims without validation against all 7 phases
- [ ] No process integration without testing systematic development workflow

---

## Confidence Scoring & Success Metrics

### Systematic Confidence Assessment
- **Structural Reorganization Success**: [Methodology compliance without breaking changes]
- **Functionality Preservation**: [All existing capabilities maintained]
- **Process Integration Quality**: [Systematic development workflow implementation]
- **Documentation Completeness**: [All methodology documentation requirements met]
- **Development Enhancement**: [Improved development capabilities and workflow]
- **Overall Systematization Quality**: [Complete legacy code methodology integration]

**Minimum threshold for successful systematization**: Overall score ≥ 8/10

### Success Criteria

**Systematization Effectiveness:**
- **Methodology Compliance**: [Percentage of 7-phase methodology standards implemented]
- **Functionality Preservation**: [All existing features work without changes after reorganization]
- **Development Enhancement**: [Improved development workflow and systematic capability]
- **Process Integration**: [Seamless integration with systematic development commands]

### Next Steps Framework

**After successful systematization:**
1. Begin using systematic development methodology for future changes
2. Train team on new systematic workflow and command usage
3. Monitor development effectiveness improvements over time
4. Consider systematizing additional legacy projects using refined process

---

**🎯 CODEFARM Legacy Code Systematization Complete**: Comprehensive reorganization of legacy codebase to 7-phase methodology standards with functionality preservation, process integration, and systematic development workflow implementation ready for ongoing methodology-based development.