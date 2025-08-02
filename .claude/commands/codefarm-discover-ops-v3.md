---
description: Intelligent scenario resolution with automatic command matching and framework selection
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, Task, Bash, LS
argument-hint: [scenario-description] - Describe what you're trying to achieve or the challenge you're facing
---

# CODEFARM Intelligent Scenario Resolver v3.0

**Scenario:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Intelligent Resolution)

**CodeFarmer (Scenario Intelligence Analyst):** "I'll automatically analyze your scenario and determine the optimal solution path without requiring meta-decisions from you."

**Critibot (Solution Validator):** "I'll validate that recommendations are based on evidence and provide the most efficient path forward."

**Programmatron (Command Intelligence Architect):** "I'll automatically match scenarios to existing commands or intelligently generate new commands when needed."

**TestBot (Resolution Effectiveness Validator):** "I'll ensure all recommendations are immediately actionable and technically sound."

---

## AUTOMATIC SCENARIO RESOLUTION ALGORITHM

### Step 1: Scenario Understanding & Context Analysis

**CodeFarmer Automatic Scenario Intelligence:**

**Analyzing scenario keywords to classify task type:**

**Using pattern matching to detect scenario characteristics:**
- Searching for analysis indicators: "audit", "analyze", "assess", "validate", "systematize", "health", "quality", "debt", "issues", "problems"
- Searching for execution indicators: "deploy", "build", "pipeline", "workflow", "orchestrate", "automate", "setup", "create", "implement"
- Searching for organization indicators: "organize", "files", "scattered", "structure", "cleanup", "arrangement"
- Searching for resolution indicators: "resolve", "fix", "complex", "legacy", "multiple issues", "interconnected"

**Using LS tool to analyze existing command inventory**
Using LS tool with path: /workspaces/context-engineering-intro/.claude/commands

**Using Glob tool to find existing CODEFARM commands**
Using Glob tool with path: /workspaces/context-engineering-intro/.claude/commands and pattern: "codefarm-*.md"

### Step 2: Automatic Command Inventory Matching

**Programmatron Intelligent Command Mapping:**

**Reading existing command capabilities for automatic matching:**

**Using Read tool to analyze core command capabilities:**
Using Read tool with file_path: /workspaces/context-engineering-intro/.claude/commands/codefarm-audit-working.md
Using Read tool with file_path: /workspaces/context-engineering-intro/.claude/commands/codefarm-systematize-working.md  
Using Read tool with file_path: /workspaces/context-engineering-intro/.claude/commands/codefarm-resolve-gordian.md
Using Read tool with file_path: /workspaces/context-engineering-intro/.claude/commands/codefarm-organize-project.md

**AUTOMATIC SCENARIO → COMMAND MATCHING ALGORITHM:**

### EXISTING COMMAND CAPABILITIES MAPPING:

**`/codefarm-audit-working` - Analysis & Validation Scenarios:**
- Keywords: "audit", "analyze", "assess", "validate", "health", "quality", "debt", "issues", "problems", "technical debt", "code quality", "systematic analysis"
- Use Cases: Code health assessment, technical debt discovery, quality validation, systematic analysis

**`/codefarm-systematize-working` - Methodology & Standards Scenarios:**  
- Keywords: "systematize", "methodology", "compliance", "standards", "organize code", "structure", "systematic development", "methodology integration"
- Use Cases: Methodology compliance, systematic code organization, development process integration

**`/codefarm-resolve-gordian` - Complex Resolution Scenarios:**
- Keywords: "resolve", "complex", "legacy", "multiple issues", "interconnected", "systematic resolution", "complex problems", "legacy systems"
- Use Cases: Complex legacy project resolution, multiple interconnected issues, systematic problem solving

**`/codefarm-organize-project` - File & Structure Organization Scenarios:**
- Keywords: "organize", "files", "scattered", "structure", "cleanup", "arrangement", "file organization", "project structure"
- Use Cases: File organization, project structure cleanup, scattered content coordination

### Step 3: Intelligent Solution Recommendation

**TestBot Automatic Decision Algorithm:**

**SOLUTION DECISION TREE:**

### PHASE A: EXISTING COMMAND MATCH CHECK

**IF scenario matches existing command capabilities:**
→ **RECOMMENDATION**: "Use `/[command-name]` - it's designed for [specific capability match]"
→ **RATIONALE**: "[Why this command fits the scenario]"
→ **IMMEDIATE ACTION**: "Run: `/[command-name] [context-path]`"

### PHASE B: COMMAND SEQUENCE ANALYSIS

**IF scenario requires multiple commands in sequence:**
→ **RECOMMENDATION**: "Use command sequence: `/[command-1]` → `/[command-2]` → `/[command-3]`"
→ **RATIONALE**: "[Why this sequence addresses the complete scenario]"
→ **IMMEDIATE ACTION**: "Start with: `/[first-command] [context-path]`"

### PHASE C: GAP ANALYSIS & NEW COMMAND GENERATION

**IF no existing command matches the scenario:**

**Automatic Framework Selection Based on Scenario Classification:**

**FOR ANALYSIS-HEAVY SCENARIOS** (validation, assessment, systematic analysis):
→ **FRAMEWORK**: FPT + CoT (First Principles + Chain of Thought)
→ **RATIONALE**: "Requires evidence-based analysis and systematic validation"
→ **CODEFARM PATTERN**: Sequential agent reasoning with documentation adherence

**FOR EXECUTION-HEAVY SCENARIOS** (deployment, automation, multi-phase workflows):
→ **FRAMEWORK**: HTN + CoT (Hierarchical Task Networks + Chain of Thought)  
→ **RATIONALE**: "Requires systematic multi-phase orchestration and conditional branching"
→ **CODEFARM PATTERN**: Parallel agent coordination with goal decomposition

**FOR HYBRID SCENARIOS** (complex resolution, implementation, transformation):
→ **FRAMEWORK**: FPT + HTN + CoT (Hybrid approach)
→ **RATIONALE**: "Requires both foundational analysis and systematic execution"
→ **CODEFARM PATTERN**: Analysis phase → Execution phase with integrated validation

### Step 4: Immediate Action Path Generation

**Critibot Solution Validation & Action Path:**

**SOLUTION OUTPUT FORMAT:**

```markdown
## 🎯 SCENARIO RESOLUTION: [Scenario Summary]

### RECOMMENDED SOLUTION:
**[EXISTING COMMAND | COMMAND SEQUENCE | NEW COMMAND NEEDED]**

### IMMEDIATE ACTION:
```bash
# Exact command to run right now
/[command-name] [arguments]
```

### RATIONALE:
[Why this solution is optimal for your scenario]

### EXPECTED OUTCOME:
[What this will accomplish for you]

### IF ISSUES ARISE:
[Fallback options or troubleshooting guidance]
```

---

## ENHANCED COMMAND GENERATION (When Needed)

### Step 5: Automatic New Command Specification

**Programmatron Smart Command Generation:**

**IF new command required, automatically generate using selected framework:**

### FOR FPT + CoT COMMANDS (Analysis Scenarios):
```markdown
---
description: [Scenario-specific analysis purpose - max 80 characters]
allowed-tools: Read, Write, Glob, Grep, LS, Edit, MultiEdit, Task
argument-hint: [project-path] - Path to project for systematic analysis
---

# CODEFARM [Scenario-Specific] Analysis Command (WORKING VERSION)

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation ([Scenario]-Enhanced)

**CodeFarmer ([Scenario] Strategic Analyst):** "I'll systematically analyze [specific context] using evidence-based first principles thinking to identify root causes and strategic opportunities."

**Critibot ([Scenario] Validation Controller):** "I'll challenge every analysis assumption, validate evidence quality, and ensure systematic methodology compliance throughout the investigation."

**Programmatron ([Scenario] Implementation Architect):** "I'll design comprehensive analysis frameworks and generate actionable recommendations with clear implementation pathways."

**TestBot ([Scenario] Evidence Validator):** "I'll validate all analysis conclusions against concrete evidence and ensure recommendations are immediately implementable."

---

## Phase A: Comprehensive [Scenario] Discovery

### Step 1: Project Structure & Evidence Analysis

**CodeFarmer Systematic Evidence Collection:**

**Using LS tool to analyze project structure and complexity**
Using LS tool with path: $ARGUMENTS to understand project organization

**Using Glob tool to discover relevant file types and patterns**
Using Glob tool with path: $ARGUMENTS and pattern: "*.py" to find Python implementation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.md" to find documentation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.txt" to find configuration files

**Using Grep tool to identify [scenario-specific] indicators**
Using Grep tool with path: $ARGUMENTS, pattern: "[scenario-specific-pattern]", output_mode: "files_with_matches" to find relevant evidence

### Step 2: [Scenario-Specific] Pattern Investigation

**Programmatron Systematic Analysis Framework:**

[Detailed step-by-step analysis using proven tool patterns]

---

## Phase B: [Scenario] Analysis & Validation

### Step 3: First Principles Investigation
[Root cause analysis using FPT methodology]

### Step 4: Evidence-Based Assessment
[Systematic validation using CoT framework]

---

## Phase C: Actionable Recommendations & Implementation

### Step 5: Systematic Solution Design
[Implementation recommendations with validation]

### Step 6: Comprehensive Documentation Generation
[Report creation and deliverables]

---

## Quality Gates & Validation

### CODEFARM [Scenario] Analysis Validation

**CodeFarmer Strategic Validation:**
- [ ] All analysis backed by concrete evidence from project investigation
- [ ] First principles methodology applied systematically throughout
- [ ] Strategic recommendations align with systematic development methodology
- [ ] Implementation pathways clearly defined and immediately actionable

**Critibot Evidence Challenge Validation:**
- [ ] Every conclusion supported by verifiable project evidence
- [ ] No assumptions made without systematic validation procedures
- [ ] Analysis methodology prevents hallucination through evidence requirements
- [ ] Quality gates ensure reliable and accurate assessment

**Programmatron Implementation Validation:**
- [ ] Recommendations technically feasible with clear implementation steps
- [ ] Analysis framework systematically applied with measurable outcomes
- [ ] Documentation comprehensive and supports ongoing development
- [ ] Integration points clearly defined with existing project structure

**TestBot Effectiveness Validation:**
- [ ] All evidence validated through systematic tool usage and verification
- [ ] Recommendations tested against actual project constraints and requirements
- [ ] Analysis methodology proven effective through measurable assessment criteria
- [ ] Implementation guidance immediately actionable with concrete next steps

### Anti-Hallucination Framework:
- [ ] No conclusions without supporting evidence from actual project analysis
- [ ] No recommendations without validation against project constraints
- [ ] No systematic solutions without proven methodology application
- [ ] No implementation confidence without realistic feasibility assessment

---

## Confidence Scoring & Success Metrics

### Systematic Analysis Effectiveness:
- **Evidence Quality**: ___/10 (concrete project evidence supporting all conclusions)
- **First Principles Application**: ___/10 (systematic methodology compliance)  
- **Implementation Feasibility**: ___/10 (realistic and immediately actionable recommendations)
- **Analysis Depth**: ___/10 (comprehensive investigation of [scenario] aspects)
- **Overall [Scenario] Analysis Quality**: ___/10 (complete systematic analysis capability)

**Threshold for actionable recommendations**: Overall score ≥ 8/10

### Success Criteria Framework:
**After completing systematic analysis:**
1. Execute highest priority recommendations based on evidence analysis
2. Validate implementation effectiveness against analysis predictions
3. Monitor progress using systematic measurement criteria established
4. Apply lessons learned to enhance future [scenario] analysis methodology

---

**🎯 CODEFARM [Scenario] Analysis Complete**: Comprehensive systematic analysis with evidence-based assessment, actionable recommendations, and immediate implementation guidance ready for project enhancement.
```

### FOR HTN + CoT COMMANDS (Execution Scenarios):
```markdown
---
description: [Scenario-specific execution purpose - max 80 characters]
allowed-tools: LS, Glob, Grep, Read, Write, Edit, MultiEdit, Bash, Task
argument-hint: [project-path] - Path to project for systematic orchestration
---

# CODEFARM [Scenario-Specific] Orchestration Command (WORKING VERSION)

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation ([Scenario]-Orchestration)

**CodeFarmer ([Scenario] Orchestration Architect):** "I'll decompose [specific goal] into systematic hierarchical workflow with evidence-based goal prioritization and conditional execution paths."

**Critibot ([Scenario] Workflow Validator):** "I'll validate each workflow phase for safety, dependency management, and systematic error handling with comprehensive rollback procedures."

**Programmatron ([Scenario] Execution Designer):** "I'll implement hierarchical task networks with systematic execution patterns and integration validation at each workflow phase."

**TestBot ([Scenario] Integration Validator):** "I'll validate multi-phase workflow execution, test conditional branching logic, and ensure comprehensive integration testing."

---

## Goal: [Primary Objective for Scenario]

### Subgoal A: [Phase 1 - Foundation/Setup]

**CoT Reasoning:** "[Why this phase is the logical starting point and what foundational elements it establishes]"

**CodeFarmer [Phase 1] Analysis:**

**Using LS tool to analyze current project state for [Phase 1]**
Using LS tool with path: $ARGUMENTS to understand project baseline

**Using Glob tool to identify [Phase 1] relevant files**
Using Glob tool with path: $ARGUMENTS and pattern: "[phase1-specific-pattern]" to find target files

**Phase 1 Actions:**
- **Action 1A**: [Specific implementation step with tool usage]
- **Action 1B**: [Sequential implementation step with validation]
- **Action 1C**: [Completion validation and checkpoint]

**Phase 1 Validation:**
- [ ] [Phase 1] baseline established successfully
- [ ] All dependencies for Phase 2 satisfied
- [ ] No breaking changes introduced during Phase 1

### Subgoal B: [Phase 2 - Implementation/Core Work]

**CoT Reasoning:** "[Why this phase follows Phase 1 and how it builds upon established foundation]"

**Programmatron [Phase 2] Implementation:**

**Using Grep tool to analyze [Phase 2] requirements**
Using Grep tool with path: $ARGUMENTS, pattern: "[phase2-specific-pattern]", output_mode: "files_with_matches" to identify implementation targets

**Conditional Execution Based on Phase 1 Results:**
- **IF Phase 1 Type A**: Execute Implementation Strategy Alpha
- **IF Phase 1 Type B**: Execute Implementation Strategy Beta  
- **IF Phase 1 Issues**: Execute Recovery Protocol Gamma

**Phase 2 Actions:**
- **Action 2A**: [Core implementation with dependency validation]
- **Action 2B**: [Integration step with error handling]
- **Action 2C**: [Performance validation and optimization]

**Phase 2 Validation:**
- [ ] [Phase 2] core functionality implemented and tested
- [ ] Integration with Phase 1 results validated
- [ ] All dependencies for Phase 3 established

### Subgoal C: [Phase 3 - Integration/Completion]

**CoT Reasoning:** "[Why this final phase is necessary and how it completes the systematic workflow]"

**TestBot [Phase 3] Integration Validation:**

**Using comprehensive validation protocols for final integration**

**Phase 3 Actions:**
- **Action 3A**: [System integration with comprehensive testing]
- **Action 3B**: [Final validation and quality assurance]
- **Action 3C**: [Documentation and handoff completion]

**Phase 3 Validation:**
- [ ] Complete [scenario] workflow executed successfully
- [ ] All integration points validated and tested
- [ ] System ready for production use

---

## Workflow Error Handling & Recovery

### Conditional Branching Logic:

**IF Phase 1 Fails:**
→ Execute Phase 1 Recovery Protocol
→ Analyze failure causes using systematic debugging
→ Implement fixes and retry Phase 1 execution

**IF Phase 2 Fails:**
→ Execute Phase 2 Recovery Protocol  
→ Validate Phase 1 integrity maintained
→ Implement targeted fixes and continue workflow

**IF Phase 3 Fails:**
→ Execute Phase 3 Recovery Protocol
→ Ensure Phases 1 & 2 remain stable
→ Complete integration with alternative approaches

---

## Quality Gates & Validation

### CODEFARM [Scenario] Orchestration Validation

**CodeFarmer Workflow Strategy Validation:**
- [ ] Goal decomposition logically structured with clear phase dependencies
- [ ] Hierarchical task network systematically designed for [scenario] execution
- [ ] Conditional branching addresses all major execution paths and error scenarios
- [ ] Workflow orchestration enhances rather than complicates systematic execution

**Critibot Execution Safety Validation:**
- [ ] Every workflow phase includes comprehensive error handling and recovery procedures
- [ ] Dependencies between phases validated and systematically managed
- [ ] No workflow execution without safety protocols and rollback capabilities
- [ ] Integration testing ensures system stability throughout execution

**Programmatron Implementation Quality Validation:**
- [ ] Hierarchical implementation systematically structured with measurable outcomes
- [ ] Multi-phase execution proven effective through systematic validation procedures
- [ ] Tool usage patterns follow established working methodologies
- [ ] Integration points clearly defined with comprehensive testing protocols

**TestBot Workflow Effectiveness Validation:**
- [ ] All workflow phases tested independently and in integrated execution
- [ ] Conditional branching logic validated through comprehensive scenario testing
- [ ] Multi-phase execution reliability proven through systematic validation
- [ ] Integration testing ensures complete workflow success and system stability

### Anti-Hallucination Framework:
- [ ] No workflow phases without concrete implementation steps and validation
- [ ] No conditional branching without systematic testing of all execution paths
- [ ] No integration assumptions without comprehensive validation procedures
- [ ] No workflow completion without systematic success criteria verification

---

## Confidence Scoring & Success Metrics

### Systematic Orchestration Effectiveness:
- **Goal Decomposition Quality**: ___/10 (logical hierarchical structure with clear dependencies)
- **Execution Phase Design**: ___/10 (systematic implementation with comprehensive validation)
- **Conditional Logic Reliability**: ___/10 (robust branching with error handling and recovery)
- **Integration Testing Coverage**: ___/10 (comprehensive multi-phase validation and system testing)
- **Overall [Scenario] Orchestration Quality**: ___/10 (complete systematic workflow execution capability)

**Threshold for workflow execution**: Overall score ≥ 8/10

### Success Criteria Framework:
**After completing systematic orchestration:**
1. Execute complete workflow with systematic validation at each phase
2. Monitor execution effectiveness against orchestration design predictions
3. Validate integration quality using comprehensive testing protocols
4. Apply lessons learned to enhance future [scenario] orchestration methodology

---

**🎯 CODEFARM [Scenario] Orchestration Complete**: Systematic hierarchical workflow execution with conditional branching, comprehensive error handling, and integrated validation ready for complex multi-phase implementation.
```

---

## ANTI-HALLUCINATION & VALIDATION FRAMEWORK

### Evidence-Based Solution Recommendation:
- [ ] All recommendations based on actual command analysis, not assumptions
- [ ] Framework selection validated against scenario characteristics
- [ ] Existing command capabilities verified through file analysis
- [ ] New command generation only when genuine gaps identified

### Intelligent Automation Validation:
- [ ] No meta-decisions required from user (automatic classification)
- [ ] Solution recommendations immediately actionable
- [ ] Command matching based on evidence, not guesswork
- [ ] Framework selection transparent with clear rationale

---

## SUCCESS CRITERIA & CONFIDENCE ASSESSMENT

### Intelligent Resolution Effectiveness:
- **Scenario Classification Accuracy**: ___/10 (correct task type identification)
- **Command Matching Precision**: ___/10 (optimal existing command selection)
- **Framework Selection Intelligence**: ___/10 (appropriate framework for new commands)
- **Action Path Clarity**: ___/10 (immediate, actionable next steps)
- **Overall Resolution Intelligence**: ___/10 (automatic scenario resolution capability)

**Threshold for successful resolution**: Overall score ≥ 8/10

### Next Steps Framework:
**After scenario resolution:**
1. Execute recommended solution immediately
2. Validate effectiveness against expected outcomes
3. Provide feedback for resolution algorithm improvement
4. Apply lessons learned to future scenario classification

---

**🎯 CODEFARM Intelligent Scenario Resolution Complete**: Automatic scenario analysis with command matching, framework selection, and immediate action path generation - no meta-decisions required from user.