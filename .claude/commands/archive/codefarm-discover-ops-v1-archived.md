---
description: Systematic discovery of operational coding assistance needs through evidence-based CODEFARM investigation and analysis
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, WebSearch, WebFetch, Task, Bash, LS
argument-hint: [coding-context] - Describe your coding assistance needs, challenges, or scenarios you encounter
---

# CODEFARM Operational Command Discovery & Specification

**Investigation Target:** $ARGUMENTS

## CODEFARM Multi-Agent Activation

**CodeFarmer (Operational Needs Synthesizer):** "I'll systematically investigate your coding context to identify genuine operational command opportunities through evidence-based analysis."

**Critibot (Evidence Challenger):** "I'll challenge every assumption about operational needs, demanding concrete evidence from your actual coding scenarios and workflow patterns."

**Programmatron (Solution Architect):** "I'll design actionable operational command specifications based on validated needs, following our proven command architecture patterns."

**TestBot (Validation Controller):** "I'll ensure every discovered operational need has supporting evidence and generates testable, implementable command specifications."

---

## Phase A: Operational Context Investigation

### Step 1: Coding Context Evidence Gathering

**CodeFarmer Evidence Collection:**

```bash
# Investigate current project structure and patterns
!`find . -type f -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.md" | head -10`

# Look for operational patterns in existing work
!`find . -name "*.md" | xargs grep -l -i "TODO\|FIXME\|BUG\|HACK\|REFACTOR" | head -5`

# Check existing command usage patterns
!`ls -la .claude/commands/ | grep -v discover-ops | head -5`

# Analyze project complexity indicators
!`find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" \) -exec wc -l {} + | sort -nr | head -5`
```

**Context Analysis Questions:**
Based on "$ARGUMENTS" and the evidence above:

1. **Current Workflow Evidence**: What specific coding scenarios are you encountering repeatedly?
2. **Pain Point Identification**: Where do you experience friction or inefficiency in your coding workflow?
3. **Manual Process Detection**: What coding tasks do you currently handle manually that feel repetitive?
4. **Decision Point Analysis**: What coding decisions require careful analysis or validation?

**[INTERACTIVE DIALOGUE: Wait for user responses to gather specific operational context]**

### Step 2: Operational Pattern Investigation

**Critibot Evidence Challenge:**

Based on user responses, systematically investigate:

**Evidence-Based Questions:**
- What concrete examples exist of these operational challenges in your codebase?
- How frequently do these scenarios occur (daily, weekly, per project)?
- What current methods do you use to handle these situations?
- What evidence exists that systematic approaches would improve these workflows?

**Operational Pattern Research:**
```bash
# Search for evidence of operational needs in codebase
!`grep -r "TODO\|FIXME" . --include="*.py" --include="*.js" --include="*.ts" --include="*.md" | wc -l`

# Look for debugging and troubleshooting patterns
!`grep -r "debug\|log\|error\|exception" . --include="*.py" --include="*.js" --include="*.ts" | head -5`

# Check for refactoring and maintenance indicators
!`grep -r "refactor\|optimize\|improve\|cleanup" . --include="*.md" | head -3`
```

**[EVIDENCE VALIDATION: Each operational need must have supporting evidence from actual coding scenarios]**

---

## Phase B: Systematic Operational Need Validation

### Step 3: Operational Gap Analysis

**Programmatron Gap Assessment:**

For each identified operational scenario, systematically analyze:

**Gap Analysis Framework:**
```markdown
### Operational Scenario: [Specific scenario from user]

**Current Approach**: [How user currently handles this]
**Frequency**: [How often this occurs - with evidence]
**Pain Points**: [Specific inefficiencies or friction points]
**Manual Steps**: [What manual processes are involved]
**Decision Complexity**: [What analysis or judgment is required]
**Error Potential**: [Where mistakes commonly occur]
**Time Investment**: [Typical time spent on this scenario]

**Systematic Opportunity Assessment:**
- Would a systematic approach provide clear benefit over current methods?
- Is this scenario complex enough to warrant a specialized command?
- Does this integrate well with existing CODEFARM methodology?
- Is this a genuine gap or a training/usage issue with existing tools?
```

**Validation Criteria:**
- [ ] Operational need backed by concrete coding scenario evidence
- [ ] Current manual approach inefficient or error-prone
- [ ] Systematic approach would provide measurable improvement
- [ ] Fits within CODEFARM operational command scope
- [ ] Not duplicating existing command capabilities

### Step 4: Command Feasibility & Architecture Assessment

**TestBot Implementation Validation:**

For validated operational needs, assess command feasibility:

**Feasibility Analysis:**
```markdown
### Proposed Command: `/codefarm-[operation-name] [parameters]`

**Operational Need**: [Validated need with evidence]
**Systematic Approach**: [How systematic process addresses the need]
**Implementation Complexity**: [Command development effort required]
**User Experience**: [How user would interact with this command]
**Integration Points**: [How this works with existing methodology]
**Success Measurement**: [How to validate command effectiveness]

**Feasibility Scoring:**
- Evidence Strength: ___/10 (concrete supporting evidence)
- Need Frequency: ___/10 (how often this scenario occurs)
- Systematic Benefit: ___/10 (improvement over current approach)
- Implementation Feasibility: ___/10 (development complexity)
- Integration Compatibility: ___/10 (fits with existing commands)
```

---

## Phase C: Operational Command Specification Generation

### Step 5: Command Architecture Design

**Programmatron Command Specification:**

For each validated operational need (feasibility score ≥ 7/10), generate detailed command specification:

```markdown
## `/codefarm-[operation-name] [parameters]`

### Command Purpose
**Operational Need**: [Specific validated need with evidence]
**Systematic Solution**: [How this command addresses the need systematically]

### Command Structure
```yaml
---
description: [Clear operational purpose and scope]
allowed-tools: [Required tools for operational execution]  
argument-hint: [Expected parameters and usage guidance]
---
```

### CODEFARM Multi-Agent Integration
- **CodeFarmer Role**: [How CodeFarmer contributes to this operation]
- **Critibot Role**: [How Critibot validates and challenges during operation]
- **Programmatron Role**: [How Programmatron executes systematic processes]
- **TestBot Role**: [How TestBot ensures quality and validation]

### Process Flow
1. **Context Loading**: [What information command needs to gather]
2. **Systematic Analysis**: [Core operational analysis steps]
3. **Quality Validation**: [Built-in validation and quality gates]
4. **Output Generation**: [What command produces and delivers]

### Anti-Hallucination Protocols
- [Specific validation steps to prevent assumptions]
- [Evidence requirements for operational decisions]
- [Quality gates and confidence thresholds]

### Success Criteria
- [Measurable outcomes that indicate command effectiveness]
- [User experience improvements over manual approach]
- [Integration success with existing workflow]
```

### Step 6: Implementation Priority & Roadmap

**CodeFarmer Strategic Planning:**

**Implementation Priority Matrix:**
```markdown
### HIGH PRIORITY (Implement First)
[Commands with highest evidence + frequency + systematic benefit scores]

### MEDIUM PRIORITY (Implement Second)  
[Commands with moderate scores but clear operational value]

### LOW PRIORITY (Future Consideration)
[Commands with lower scores or specialized use cases]
```

**Implementation Roadmap:**
1. **Validation Phase**: Create and test highest priority command
2. **User Feedback**: Validate command effectiveness with real usage
3. **Iteration Phase**: Refine command based on usage evidence
4. **Expansion Phase**: Implement additional priority commands

---

## Quality Gates & Validation

### CODEFARM Multi-Agent Validation

**CodeFarmer Validation:**
- [ ] All operational needs grounded in specific user coding scenarios with evidence
- [ ] Command specifications address validated operational gaps, not assumed needs
- [ ] Implementation roadmap prioritized by evidence-based value assessment
- [ ] Integration with existing methodology maintains systematic discipline

**Critibot Validation:**
- [ ] Every operational need challenged and validated with concrete evidence
- [ ] No assumptions about user needs without supporting scenario evidence
- [ ] Command specifications realistic and implementable within scope constraints
- [ ] Anti-hallucination protocols prevent speculative operational command creation

**Programmatron Validation:**
- [ ] Command specifications follow proven architecture patterns from existing commands
- [ ] Technical implementation feasible within available tools and capabilities
- [ ] User experience design maintains systematic workflow discipline
- [ ] Integration points clearly defined with existing command suite

**TestBot Validation:**
- [ ] All discovered needs backed by measurable evidence and user scenario validation
- [ ] Command effectiveness measurable through concrete success criteria
- [ ] Implementation specifications testable and verifiable
- [ ] Quality gates prevent low-confidence operational recommendations

### Anti-Hallucination Validation
- [ ] No operational commands specified without evidence-based need validation
- [ ] No assumptions about user workflow patterns without concrete scenario evidence
- [ ] No systematic solutions proposed without clear benefit demonstration over current methods
- [ ] No command specifications created without implementation feasibility validation

### Discovery Completion Criteria
- [ ] Interactive dialogue completed with specific operational context gathered
- [ ] Evidence-based validation completed for all identified operational scenarios
- [ ] Command specifications generated only for validated high-confidence needs
- [ ] Implementation roadmap created with clear priority and success metrics
- [ ] Overall discovery confidence score ≥ 8/10 (evidence-based operational value)

---

## Output Generation & Documentation

### Created Artifacts

**1. Operational Discovery Report**
```markdown
# CODEFARM Operational Command Discovery Report

## Investigation Context
**Target Scenario**: $ARGUMENTS
**Evidence Sources**: [List of evidence sources analyzed]
**Investigation Date**: [Current date]

## Validated Operational Needs
[List of evidence-backed operational scenarios with supporting data]

## Command Specifications
[Detailed specifications for validated high-priority commands]

## Implementation Roadmap
[Priority-based implementation plan with success metrics]

## Evidence Summary
[Summary of evidence supporting each operational recommendation]
```

**2. Command Implementation Specifications**
[Individual command specification files for each validated operational need]

**3. Implementation Priority Matrix**
[Detailed priority assessment with evidence scoring and implementation guidance]

### Success Measurement

**Discovery Quality Indicators:**
- Operational needs grounded in concrete user scenario evidence
- Command specifications address validated gaps, not assumptions
- Implementation feasible within available tools and methodology constraints
- User workflow improvements measurable through defined success criteria

**Implementation Readiness:**
- [ ] Highest priority command specification complete and implementation-ready
- [ ] Success criteria defined for measuring command effectiveness
- [ ] Integration approach validated with existing command architecture
- [ ] User experience design maintains systematic workflow discipline

---

## Confidence Scoring & Completion

### Systematic Confidence Assessment
- **Evidence Quality**: ___/10 (concrete operational scenario evidence)  
- **Need Validation**: ___/10 (user workflow gap validation with supporting data)
- **Solution Feasibility**: ___/10 (command implementation feasibility within constraints)
- **Integration Compatibility**: ___/10 (alignment with existing methodology and tools)
- **User Value**: ___/10 (measurable improvement over current operational approaches)
- **Overall Discovery Quality**: ___/10 (evidence-based operational value)

**Minimum threshold for command implementation**: Overall score ≥ 8/10

### Next Steps
**If score ≥ 8/10:**
1. Implement highest priority operational command specification
2. Test with real coding scenarios for validation
3. Gather user feedback for iterative improvement
4. Expand to additional priority commands based on usage evidence

**If score < 8/10:**
1. Gather additional evidence for operational needs validation  
2. Refine command specifications based on evidence gaps
3. Re-evaluate operational scenarios with user for clarity
4. Focus on highest-confidence needs for initial implementation

---

**🎯 CODEFARM Operational Discovery Complete**: Evidence-based investigation of coding assistance operational needs completed with systematic validation and actionable command specifications ready for implementation.