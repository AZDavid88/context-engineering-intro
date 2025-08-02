---
description: Research-to-code alignment validation with anti-hallucination documentation generation
allowed-tools: Read, Write, Glob, Grep, LS
argument-hint: [project-path] - Path to project requiring anti-hallucination roadmap
---

# CODEFARM Anti-Hallucination Roadmap Generator (WORKING VERSION)

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Research Alignment Specialists)

**CodeFarmer (Research-Code Mapping Analyst):** "I'll systematically map all implementations against research documentation to identify alignment gaps and create comprehensive project understanding for anti-hallucination development."

**Critibot (Validation Consistency Controller):** "I'll identify every deviation between research and implementation, documenting all potential hallucination sources and ensuring systematic adherence validation."

**Programmatron (Documentation Generation Architect):** "I'll create process-enforcing documentation that prevents agentic coding hallucinations through research-backed constraints and systematic development guidelines."

**TestBot (Anti-Hallucination Effectiveness Validator):** "I'll validate the documentation framework against known hallucination patterns and ensure systematic prevention of implementation deviations."

---

## Framework: FPT + HTN + CoT (Anti-Hallucination Hybrid Approach)

### Phase A: Comprehensive Project Scope Discovery

**CodeFarmer Systematic Project Comprehension:**

**CoT Reasoning**: "Anti-hallucination requires complete project understanding. I must systematically map every functional component, understand relationships, and validate against research context before creating process-enforcing guidelines."

**First Principles Project Analysis:**

**1. Complete Functional Script Discovery:**

**Using LS tool to analyze complete project structure**
Using LS tool with path: $ARGUMENTS to understand project organization and scope

**Using Glob tool to discover all functional script types**
Using Glob tool with path: $ARGUMENTS and pattern: "**/*.py" to find all Python implementation files
Using Glob tool with path: $ARGUMENTS and pattern: "src/**/*.py" to find core implementation modules
Using Glob tool with path: $ARGUMENTS and pattern: "scripts/**/*.py" to find utility and validation scripts

**2. Research Documentation Inventory:**

**Using LS tool to analyze research documentation structure**
Using LS tool with path: $ARGUMENTS/research to understand research organization

**Using Glob tool to discover research documentation coverage**
Using Glob tool with path: $ARGUMENTS/research and pattern: "**/*.md" to find all research documentation
Using Glob tool with path: $ARGUMENTS/research and pattern: "**/research_summary.md" to find research syntheses

**3. Implementation-to-Research Mapping Analysis:**

**Using Grep tool to identify research references in code**
Using Grep tool with path: $ARGUMENTS/src, pattern: "research|documentation|TODO|FIXME", output_mode: "files_with_matches" to find research integration points

**Using Grep tool to identify import patterns and dependencies**
Using Grep tool with path: $ARGUMENTS/src, pattern: "^from|^import", output_mode: "content" to map dependency relationships

### Phase B: Research Alignment Validation & Gap Analysis

**Critibot Systematic Research Alignment Assessment:**

**CoT Reasoning**: "Every implementation must be validated against research documentation. Gaps between research and implementation are hallucination sources that must be systematically identified and documented."

**HTN Task Decomposition - Research Validation:**

**Goal: Validate Complete Research-to-Implementation Alignment**

**Subgoal A: Core Implementation Validation**

**Action A1: Genetic Algorithm Implementation Analysis**
**Using Read tool to analyze genetic engine implementations**
Using Read tool with file_path: $ARGUMENTS/src/strategy/genetic_engine_research_compliant.py to understand core genetic algorithm implementation

**Using Grep tool to find research references in genetic implementation**
Using Grep tool with path: $ARGUMENTS/research, pattern: "genetic|DEAP|population|evolution", output_mode: "files_with_matches" to find relevant research documentation

**Action A2: Data Collection Implementation Analysis**
**Using Read tool to analyze data collection implementations**
Using Read tool with file_path: $ARGUMENTS/src/data/hyperliquid_client.py to understand API integration implementation

**Using Grep tool to validate against Hyperliquid research documentation**
Using Grep tool with path: $ARGUMENTS/research/hyperliquid_documentation, pattern: "API|endpoint|rate.*limit", output_mode: "content" to validate implementation adherence

**Subgoal B: Strategy Implementation Validation**

**Action B1: Trading Strategy Implementation Analysis**
**Using Glob tool to discover all genetic seed implementations**
Using Glob tool with path: $ARGUMENTS/src/strategy/genetic_seeds and pattern: "*_seed.py" to find all strategy implementations

**Using Read tool to analyze strategy implementation patterns**
Using Read tool with file_path: $ARGUMENTS/src/strategy/genetic_seeds/base_seed.py to understand strategy framework

**Action B2: Technical Indicator Implementation Validation**
**Using Grep tool to find technical analysis implementations**
Using Grep tool with path: $ARGUMENTS/src, pattern: "pandas|ta|technical.*analysis", output_mode: "files_with_matches" to identify TA usage

**Using Read tool to validate against pandas research documentation**
Using Read tool with file_path: $ARGUMENTS/research/pandas_comprehensive/research_summary.md to validate pandas usage patterns

**Subgoal C: Infrastructure Implementation Validation**

**Action C1: Monitoring and Execution Systems Analysis**
**Using Read tool to analyze monitoring implementation**
Using Read tool with file_path: $ARGUMENTS/src/execution/monitoring.py to understand monitoring architecture

**Using Grep tool to validate against monitoring research**
Using Grep tool with path: $ARGUMENTS/research, pattern: "monitoring|prometheus|grafana", output_mode: "files_with_matches" to find monitoring research

### Phase C: Anti-Hallucination Documentation Generation

**Programmatron Process-Enforcing Documentation Creation:**

**CoT Reasoning**: "Process-enforcing documentation must create systematic constraints that prevent agentic coding from deviating from validated patterns. Every component needs clear implementation guidelines based on research evidence."

**HTN Documentation Generation Strategy:**

**Goal: Create Comprehensive Anti-Hallucination Development Framework**

**Subgoal A: Implementation Pattern Documentation**

**Action A1: Validated Implementation Patterns Documentation**
**Using Write tool to create implementation pattern documentation**
Generate comprehensive documentation file: $ARGUMENTS/docs/development/ANTI_HALLUCINATION_IMPLEMENTATION_PATTERNS.md

**Content Structure:**
- **Genetic Algorithm Patterns**: Validated DEAP usage patterns from research
- **API Integration Patterns**: Validated Hyperliquid API usage from documentation
- **Data Processing Patterns**: Validated pandas usage patterns from research
- **Strategy Implementation Patterns**: Validated genetic seed implementation framework

**Action A2: Research-to-Code Mapping Documentation**
**Using Write tool to create research mapping documentation**
Generate research alignment documentation: $ARGUMENTS/docs/development/RESEARCH_TO_CODE_ALIGNMENT_MAP.md

**Content Structure:**
- **Research Documentation Index**: Complete mapping of research to implementation
- **Implementation Validation Matrix**: Verification status of each component
- **Gap Analysis Report**: Identified deviations and missing implementations
- **Compliance Checklist**: Step-by-step validation for new implementations

**Subgoal B: Process-Enforcing Development Guidelines**

**Action B1: Anti-Hallucination Development Protocol**
**Using Write tool to create development protocol documentation**
Generate development protocol: $ARGUMENTS/docs/development/ANTI_HALLUCINATION_DEVELOPMENT_PROTOCOL.md

**Content Structure:**
- **Research Validation Requirements**: Mandatory research consultation before coding
- **Implementation Verification Checklist**: Step-by-step validation against documentation
- **Code Review Anti-Hallucination Framework**: Systematic review process
- **Dependency Validation Protocol**: Approved libraries and usage patterns

**Action B2: Agentic Coding Constraint Framework**
**Using Write tool to create agentic coding constraints**
Generate constraint framework: $ARGUMENTS/docs/development/AGENTIC_CODING_CONSTRAINTS.md

**Content Structure:**
- **Approved Implementation Patterns**: Only validated patterns from research
- **Forbidden Implementation Patterns**: Known hallucination sources
- **Research Consultation Requirements**: When to consult specific documentation
- **Validation Gate Requirements**: Mandatory checks before implementation acceptance

**Subgoal C: Interface and API Documentation**

**Action C1: Systematic Interface Documentation**
**Using Write tool to create interface documentation**
Generate interface documentation: $ARGUMENTS/docs/development/SYSTEMATIC_INTERFACE_DOCUMENTATION.md

**Content Structure:**
- **Validated Interface Patterns**: Correct interface implementations from research
- **Interface Validation Checklist**: Step-by-step interface verification
- **Common Interface Mismatches**: Known problematic patterns (like registry.list_all_seeds)
- **Interface Evolution Protocol**: Safe interface change procedures

### Phase D: Roadmap Validation & Implementation

**TestBot Anti-Hallucination Framework Validation:**

**CoT Reasoning**: "The complete roadmap must be validated against known hallucination patterns and proven effective at preventing implementation deviations. Every guideline must be actionable and systematically enforceable."

**Comprehensive Validation Protocol:**

**Validation A: Documentation Completeness Assessment**
- **Coverage Analysis**: Every major component has anti-hallucination guidelines
- **Research Alignment Verification**: All guidelines backed by research documentation
- **Implementation Constraint Validation**: All constraints technically enforceable
- **Agentic Coding Compatibility**: Guidelines compatible with systematic development

**Validation B: Anti-Hallucination Effectiveness Testing**
- **Known Pattern Prevention**: Framework prevents identified hallucination sources
- **Research Deviation Detection**: Framework catches implementation-research misalignment
- **Interface Mismatch Prevention**: Framework prevents interface problems like registry issue
- **Systematic Compliance Enforcement**: Framework enables automated compliance checking

---

## Quality Gates & Success Criteria

### CODEFARM Anti-Hallucination Roadmap Validation:

**CodeFarmer Comprehensive Analysis Validation:**
- [ ] Complete project scope mapped with systematic functional script analysis
- [ ] All implementation components validated against research documentation foundation
- [ ] Project comprehension documented with sufficient detail for anti-hallucination development
- [ ] Strategic roadmap created with evidence-based implementation guidelines

**Critibot Research Alignment Validation:**
- [ ] Every major implementation component validated against research documentation
- [ ] All deviations between research and implementation systematically identified
- [ ] Gap analysis complete with prioritized remediation recommendations
- [ ] Research-to-code alignment framework prevents future implementation drift

**Programmatron Process Documentation Validation:**
- [ ] Comprehensive process-enforcing documentation generated with systematic constraints
- [ ] Implementation patterns documented with research-validated guidelines
- [ ] Development protocols created with anti-hallucination focus and enforcement mechanisms
- [ ] Documentation framework enables systematic agentic coding with hallucination prevention

**TestBot Framework Effectiveness Validation:**
- [ ] Anti-hallucination framework validated against known hallucination patterns
- [ ] Process-enforcing guidelines proven effective through systematic validation testing
- [ ] Implementation constraints verified as technically enforceable and development-compatible
- [ ] Complete roadmap enables confident agentic coding with systematic deviation prevention

### Anti-Hallucination Framework:
- [ ] No implementation guidelines without supporting research documentation evidence
- [ ] No process enforcement without validation against actual project implementation patterns
- [ ] No anti-hallucination claims without systematic testing against known problematic patterns
- [ ] No roadmap completion without comprehensive validation of framework effectiveness

### Success Metrics:
- **Project Comprehension Completeness**: 100% of functional scripts analyzed and documented
- **Research Alignment Accuracy**: All implementations validated against research documentation
- **Anti-Hallucination Prevention Coverage**: Framework addresses all identified hallucination sources
- **Process Enforcement Effectiveness**: Guidelines enable systematic compliance validation
- **Agentic Coding Readiness**: Complete roadmap enables confident systematic development

---

## Deliverables & Implementation Roadmap

### Primary Deliverables:

**1. Project Comprehension Documentation**
- Complete functional script analysis and mapping
- Implementation architecture documentation with research alignment
- Component relationship mapping with dependency validation

**2. Research Alignment Validation Report**
- Comprehensive implementation-to-research mapping
- Gap analysis with prioritized remediation recommendations  
- Compliance matrix with validation status for each component

**3. Process-Enforcing Development Framework**
- Anti-hallucination implementation patterns documentation
- Systematic development protocol with research validation requirements
- Agentic coding constraint framework with enforcement mechanisms

**4. Anti-Hallucination Roadmap**
- Strategic implementation guidelines for systematic development
- Validation frameworks for ongoing compliance maintenance
- Systematic methodology integration with anti-hallucination focus

### Implementation Success Criteria:
**After roadmap completion:**
1. Agentic coding can proceed with systematic confidence and hallucination prevention
2. All new implementations systematically validated against research documentation
3. Process-enforcing guidelines prevent implementation deviations and research drift
4. Comprehensive framework enables ongoing systematic development with anti-hallucination focus

---

**🎯 CODEFARM Anti-Hallucination Roadmap Complete**: Comprehensive project analysis with research alignment validation, process-enforcing documentation generation, and systematic anti-hallucination framework enabling confident agentic coding with systematic deviation prevention and research adherence.