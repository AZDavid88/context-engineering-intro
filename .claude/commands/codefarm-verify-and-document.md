---
description: Function verification with data flow tracing and auto-documentation generation
allowed-tools: LS, Glob, Grep, Read, Write, Edit, MultiEdit, Task
argument-hint: [project-path] - Path to project requiring function verification and documentation
---

# CODEFARM Function Verification & Auto-Documentation Command

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (FPT + HTN + CoT Enhanced)

**CodeFarmer (Function Verification Architect):** "I'll apply First Principle Thinking to analyze actual function behavior vs documentation, systematically verifying what code really does rather than what it claims to do."

**Critibot (Evidence Validation Controller):** "I'll ensure every verification is based on concrete code analysis, challenge documentation assumptions, and validate auto-generated documentation accuracy."

**Programmatron (Integrated Documentation Designer):** "I'll design systematic workflows that trace data flow, verify function behavior, and automatically generate accurate living documentation."

**TestBot (Verification Effectiveness Validator):** "I'll validate function behavior analysis accuracy and ensure auto-generated documentation reflects actual code functionality."

---

## Phase A: First Principles Function Analysis

### Step 1: Scope Detection & Function Mapping

**CodeFarmer Systematic Scope Intelligence:**

**Analysis Scope Detection:**
Using LS tool to analyze target path at: $ARGUMENTS

**Scope Classification Logic:**
- **Module Analysis**: If $ARGUMENTS path contains `/src/[module_name]` (e.g., `/src/discovery`)
- **Full Project Analysis**: If $ARGUMENTS path is project root or contains multiple modules

**Project Structure Analysis:**
Using LS tool to understand analysis scope and target boundaries

**Python Implementation Discovery:**
Using Glob tool with path: $ARGUMENTS and pattern: "*.py" to discover all Python files for function analysis

**Function Declaration Mapping:**
Using Grep tool with path: $ARGUMENTS, pattern: "def ", output_mode: "content", head_limit: 50 to map all function declarations

**Function Documentation Discovery:**
Using Grep tool with path: $ARGUMENTS, pattern: '""".*"""', output_mode: "content", head_limit: 30 to find existing function documentation

### Step 2: Function Behavior vs Documentation Analysis

**Programmatron Function Verification Framework:**

**For each discovered function, systematic verification:**

**FPT-Based Function Analysis:**
1. **Stated Purpose Analysis**: Extract docstring claims and documentation
2. **Actual Implementation Analysis**: Analyze what the function really does
3. **Input/Output Verification**: Map actual data transformations
4. **Side Effects Discovery**: Identify hidden behaviors and dependencies

**Function Verification Process:**
Using Read tool to analyze each Python file containing functions

**Verification Framework:**
```
Function: [function_name] in [file_path:line_number]
├── Documented Purpose: [docstring/comment analysis]
├── Actual Behavior: [implementation analysis]
│   ├── Inputs: [actual parameter usage]
│   ├── Processing: [real data transformations]
│   ├── Outputs: [actual return values/side effects]
│   └── Dependencies: [internal/external function calls]
├── Verification Status:
│   ├── ✅ Verified: Implementation matches documentation
│   ├── ⚠️ Partial: Implementation partially matches
│   ├── ❌ Mismatch: Implementation differs from documentation
│   └── 🔍 Undocumented: No documentation exists
└── Evidence: [specific code examples supporting verification]
```

---

## Phase B: Hierarchical Data Flow Tracing

### Step 3: Data Flow Path Discovery

**TestBot Data Flow Intelligence:**

**Import Dependency Mapping:**
Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import", output_mode: "content" to map actual import dependencies

**Data Transformation Tracing:**
Using Grep tool with path: $ARGUMENTS, pattern: "return |yield |=.*\\(", output_mode: "content", head_limit: 50 to trace data flow patterns

**Integration Point Analysis:**
Using Grep tool with path: $ARGUMENTS, pattern: "open\\(|requests\\.|json\\.|pd\\.|np\\.", output_mode: "files_with_matches" to identify external data sources

**Data Flow Verification Framework:**
```
Data Flow Analysis for [module_name]:
├── Input Sources:
│   ├── Function Parameters: [analyzed parameter usage]
│   ├── File Operations: [file read/write operations]
│   ├── External APIs: [network calls and API usage]
│   └── Database: [database query operations]
├── Processing Stages:
│   ├── Data Transformations: [actual data modifications]
│   ├── Calculations: [mathematical/logical operations]
│   ├── Validations: [data validation and error handling]
│   └── Business Logic: [domain-specific processing]
├── Output Destinations:
│   ├── Return Values: [function return analysis]
│   ├── File Operations: [file write operations]
│   ├── External APIs: [outbound API calls]
│   └── Side Effects: [state changes and modifications]
└── Error Handling: [exception handling and fallback behaviors]
```

### Step 4: Integration Dependency Verification

**Critibot Integration Safety Analysis:**

**Dependency Risk Assessment:**
Using Grep tool with path: $ARGUMENTS, pattern: "try:|except:|raise|assert", output_mode: "content", head_limit: 30 to analyze error handling

**External Service Integration:**
Using Grep tool with path: $ARGUMENTS, pattern: "http|api|url|endpoint", output_mode: "files_with_matches" to identify external service dependencies

**Configuration Dependencies:**
Using Grep tool with path: $ARGUMENTS, pattern: "config|environ|env\\[|getenv", output_mode: "files_with_matches" to identify configuration dependencies

---

## Phase C: Integrated Auto-Documentation Generation

### Step 5: Living Documentation Creation

**CodeFarmer Documentation Intelligence:**

**Auto-Documentation Generation Strategy:**

For each verified function and data flow, generate accurate documentation:

**Function Documentation Template:**
```markdown
# [Module Name] - Function Documentation
**Auto-generated from code verification on [timestamp]**

## Function: `[function_name]`
**Location**: `[file_path:line_number]`
**Verification Status**: [✅ Verified / ⚠️ Partial / ❌ Mismatch / 🔍 Undocumented]

### Actual Functionality
[What the function really does based on implementation analysis]

### Parameters
- `[param_name]` ([type]): [actual usage and constraints]

### Returns
- ([type]): [actual return value analysis]

### Data Flow
├── Inputs: [actual input sources and formats]
├── Processing: [real data transformations performed]
└── Outputs: [actual output destinations and effects]

### Dependencies
├── Internal: [functions called within project]
├── External: [third-party libraries used]
└── System: [file/network/database operations]

### Error Handling
[Actual exception handling and error cases]

### Implementation Notes
[Important implementation details and edge cases]
```

### Step 6: Documentation Integration & Auto-Update System

**Programmatron Auto-Update Framework:**

**Dedicated Verification Documentation Structure:**

**Scope-Aware Documentation Organization:**

**For Module Analysis** (e.g., `/src/discovery`):
Using Write tool to create module-specific verification structure:
- `[PROJECT_ROOT]/verified_docs/by_module/[MODULE_NAME]/functions/` - Function verification
- `[PROJECT_ROOT]/verified_docs/by_module/[MODULE_NAME]/data_flow/` - Data flow analysis
- `[PROJECT_ROOT]/verified_docs/by_module/[MODULE_NAME]/integration/` - Dependencies

**For Full Project Analysis**:
Using Write tool to create comprehensive verification structure:
- `$ARGUMENTS/verified_docs/full_system/functions/` - All functions verification
- `$ARGUMENTS/verified_docs/full_system/data_flow/` - Complete data flow analysis
- `$ARGUMENTS/verified_docs/full_system/integration/` - System-wide dependencies

**Auto-Generated Documentation Files:**

**1. Function Verification Report**
- **Module**: `[PROJECT_ROOT]/verified_docs/by_module/[MODULE_NAME]/function_verification_report.md`
- **Full Project**: `$ARGUMENTS/verified_docs/full_system/function_verification_report.md`

**2. Data Flow Analysis**
- **Module**: `[PROJECT_ROOT]/verified_docs/by_module/[MODULE_NAME]/data_flow_analysis.md`
- **Full Project**: `$ARGUMENTS/verified_docs/full_system/data_flow_analysis.md`

**3. Integration Documentation**
- **Module**: `[PROJECT_ROOT]/verified_docs/by_module/[MODULE_NAME]/dependency_analysis.md`
- **Full Project**: `$ARGUMENTS/verified_docs/full_system/dependency_analysis.md`

**4. Master Verification Index**
Using Write tool to create `$ARGUMENTS/verified_docs/verification_index.md` - Master index of all verified documentation

---

## Phase D: Validation & Continuous Verification

### Step 7: Verification Accuracy Validation

**TestBot Verification Quality Assurance:**

**Documentation Accuracy Testing:**
- Validate generated documentation against actual code implementation
- Confirm function behavior descriptions match real functionality
- Verify data flow documentation reflects actual data transformations
- Test integration documentation against real dependencies

**Verification Completeness Assessment:**
- Confirm all functions in project have been analyzed
- Validate data flow coverage across all modules
- Ensure integration points properly documented
- Assess verification confidence levels

### Step 8: Auto-Update Trigger Framework

**Critibot Change Detection System:**

**Documentation Update Triggers:**
- Function signature changes requiring reverification
- New functions added needing documentation
- Import dependency modifications affecting data flow
- Integration point changes requiring updates

**Change Detection Implementation:**
Using Grep tool to establish baseline patterns for future change detection:
- Function signatures and implementations
- Import statements and dependencies
- External integration patterns
- Configuration and environment usage

---

## Quality Gates & Validation

### CODEFARM Function Verification & Documentation Validation

**CodeFarmer Strategic Verification Validation:**
- [ ] Function behavior analysis based on actual implementation vs assumptions or documentation claims
- [ ] Data flow tracing reflects real data transformations and dependencies across project modules  
- [ ] Auto-generated documentation accurately represents verified function behavior and integration patterns
- [ ] Documentation system maintainable and updates automatically with code changes

**Critibot Evidence & Safety Validation:**
- [ ] Every function verification backed by concrete code analysis and implementation examination
- [ ] Data flow analysis validated against actual import dependencies and data transformation patterns
- [ ] Auto-generated documentation accuracy confirmed through systematic verification procedures
- [ ] Documentation update system prevents drift between code reality and documentation claims

**Programmatron Implementation Quality Validation:**
- [ ] Function verification methodology systematically applied with comprehensive coverage analysis
- [ ] Data flow tracing implementation accurate and reliable for production development workflow
- [ ] Auto-documentation generation creates maintainable and useful documentation from code semantics
- [ ] Integration framework enhances rather than complicates development workflow and code understanding

**TestBot Verification Effectiveness Validation:**
- [ ] All function behavior verification validated against actual implementation code
- [ ] Data flow tracing confirmed accurate through comprehensive dependency and transformation analysis
- [ ] Auto-generated documentation quality measured against manual documentation standards
- [ ] Verification and documentation system effectiveness proven through development workflow improvement

### Anti-Hallucination Framework:
- [ ] No function behavior claims without concrete implementation code analysis
- [ ] No data flow documentation without actual dependency and transformation verification
- [ ] No auto-generated documentation without systematic verification against real code functionality
- [ ] No verification confidence without measurable accuracy assessment and evidence validation

---

## Confidence Scoring & Success Metrics

### Function Verification & Documentation Effectiveness:
- **Function Verification Accuracy**: ___/10 (correct identification of actual vs documented behavior)
- **Data Flow Tracing Completeness**: ___/10 (comprehensive dependency and transformation mapping)
- **Auto-Documentation Quality**: ___/10 (useful, accurate, maintainable documentation generation)
- **Integration Workflow Enhancement**: ___/10 (improved development understanding and efficiency)
- **Verification System Reliability**: ___/10 (consistent and dependable verification results)
- **Overall Verification & Documentation Quality**: ___/10 (complete function verification with auto-documentation capability)

**Threshold for system deployment**: Overall score ≥ 8.5/10 WITH Function Verification Accuracy ≥ 9/10

### Success Criteria Framework:
**After completing function verification and auto-documentation:**
1. **Verification Accuracy**: Confirm system correctly identifies function behavior vs documentation discrepancies
2. **Documentation Usefulness**: Validate auto-generated documentation improves development understanding and workflow
3. **Data Flow Understanding**: Verify data flow tracing enhances system comprehension and debugging capability
4. **Workflow Integration**: Measure development efficiency improvement with verified documentation system

### Next Steps Framework:
**After successful verification and documentation implementation:**
1. Use verified documentation for confident development and debugging
2. Monitor auto-update system effectiveness and refine based on usage patterns
3. Apply verification insights to improve code quality and documentation practices
4. Extend verification approach to additional projects for systematic development improvement

---

**🎯 CODEFARM Function Verification & Auto-Documentation Complete**: Systematic function behavior verification with data flow tracing and integrated auto-documentation generation ready for accurate, maintainable development workflow enhancement with continuous verification and evidence-based documentation accuracy.