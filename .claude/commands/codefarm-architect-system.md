---
allowed-tools: Read, Write, Grep, Glob, WebFetch, Bash
description: Systematic architecture design with CODEFARM multi-agent analysis and FPT validation
argument-hint: [project-path] | Path to project directory from Phase 2
pre-conditions: Phase 2 discovery complete with comprehensive research and competitive analysis
post-conditions: Complete system architecture ready for implementation specification
rollback-strategy: Preserve discovery research, flag architectural gaps for re-analysis
---

# Phase 3: System Architecture with CODEFARM + FPT Analysis

**Context**: You are using the CODEFARM methodology for systematic architecture design through First Principles analysis. This phase transforms Phase 2 discovery intelligence into structured, scalable system design.

## Fundamental Operation Definition
**Core Operation**: Transform comprehensive discovery research into structured system architecture by identifying essential components, relationships, and interfaces from first principles.

**FPT Question**: What are the irreducible structural elements needed to solve the fundamental problem identified in Phase 1 and validated in Phase 2?

## Pre-Condition Validation
**Systematic Prerequisites Check:**

### Phase 2 Output Validation
```bash
# Validate Phase 2 completion
PROJECT_PATH="${ARGUMENTS}"
if [[ ! -f "$PROJECT_PATH/phases/completed/phase_2_discovery.md" ]]; then
    echo "ERROR: Phase 2 incomplete - discovery documentation not found"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/competitive_analysis.md" ]]; then
    echo "ERROR: Competitive analysis missing from Phase 2"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/risk_assessment.md" ]]; then
    echo "ERROR: Risk assessment missing from Phase 2"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/user_research.md" ]]; then
    echo "ERROR: User research missing from Phase 2"
    exit 2
fi
```

**Pre-Condition Requirements:**
- [ ] Phase 2 discovery documentation complete and comprehensive
- [ ] Competitive analysis with fundamental approach mapping
- [ ] Risk assessment with fundamental failure modes identified
- [ ] User research with evidence-based insights
- [ ] Research directory populated with official sources
- [ ] Phase 2 confidence score ≥ 7.0/10

## CODEFARM Multi-Agent Activation with FPT Enhancement

**CodeFarmer (Architecture Visionary with FPT):** 
"I'll apply First Principles Thinking to design system architecture from fundamental requirements. We'll identify essential components and their optimal relationships."

**Critibot (Design Challenger with FPT):**
"I'll challenge every architectural decision against discovery research and competitive analysis. No design choice proceeds without fundamental justification."

**Programmatron (Technical Architect with FPT):**
"I'll design concrete system architecture based on proven patterns and fundamental technical requirements, with detailed component specifications."

**TestBot (Architecture Validator with FPT):**
"I'll validate architecture against discovery research, competitive patterns, and fundamental scalability requirements."

---

## Phase 3 Core Process with FPT-Enhanced Architecture

### Step 1: Discovery Intelligence Integration
**CodeFarmer Systematic Architecture Foundation:**

**1A. Load Complete Discovery Context**
```markdown
**Discovery Research Integration:**
- Updated Project Vision: @${PROJECT_PATH}/planning_prp.md
- Discovery Research: @${PROJECT_PATH}/phases/completed/phase_2_discovery.md
- Competitive Analysis: @${PROJECT_PATH}/specs/competitive_analysis.md
- Risk Assessment: @${PROJECT_PATH}/specs/risk_assessment.md
- User Research: @${PROJECT_PATH}/specs/user_research.md
```

**1B. FPT Problem Decomposition**
**Fundamental Question**: Based on discovery research, what are the irreducible functional requirements?

Systematic requirements extraction:
- **Core User Needs**: What fundamental user problems must the system solve?
- **Essential Data Flows**: What information must flow through the system?
- **Critical Business Logic**: What fundamental processes must the system execute?
- **Integration Requirements**: What external systems must fundamentally connect?
- **Performance Requirements**: What fundamental performance characteristics are needed?

**1C. Competitive Pattern Analysis**
From Phase 2 competitive analysis, identify:
- **Proven Architectural Patterns**: What fundamental approaches work?
- **Common Failure Points**: What architectural decisions lead to problems?
- **Scalability Patterns**: How do successful systems handle growth?
- **Integration Patterns**: How do similar systems connect with external services?

### Step 2: FPT Component Identification & Design
**Programmatron Technical Architecture with FPT:**

**2A. Fundamental Component Identification**
**FPT Question**: What are the minimal essential functional units needed?

Systematic component analysis:
```markdown
**Core Component Categories:**
1. **Data Components**: What entities and their relationships are fundamental?
2. **Process Components**: What business logic operations are irreducible?
3. **Interface Components**: What user interaction patterns are essential?
4. **Integration Components**: What external connections are necessary?
5. **Infrastructure Components**: What technical foundations are required?
```

**2B. Component Relationship Mapping**
**FPT Approach**: Design relationships based on fundamental dependencies:
- **Data Dependencies**: Which components require data from others?
- **Process Dependencies**: Which operations must complete before others?
- **User Flow Dependencies**: What user interactions must be sequential?
- **Technical Dependencies**: What infrastructure components depend on others?

**2C. Interface Definition with FPT**
**Fundamental Question**: What are the minimal necessary connection points?

Systematic interface design:
- **API Interfaces**: What data contracts are essential between components?
- **User Interfaces**: What interaction patterns serve fundamental user needs?
- **System Interfaces**: What integration points are technically required?
- **Data Interfaces**: What information exchanges are structurally necessary?

### Step 3: Architecture Pattern Selection & Validation
**Critibot Pattern Challenge with FPT:**

**3A. Fundamental Architecture Pattern Analysis**
**FPT Question**: What architectural approach best serves the irreducible requirements?

Pattern evaluation against discovery research:
- **Monolithic vs Microservices**: Based on complexity and scaling requirements
- **Event-Driven vs Request-Response**: Based on user interaction patterns
- **Centralized vs Distributed**: Based on performance and reliability requirements
- **Layered vs Hexagonal**: Based on integration and testing requirements

**3B. Competitive Architecture Validation**
Validate chosen patterns against competitive analysis:
- **Proven Success**: Do similar successful systems use this approach?
- **Failure Avoidance**: Does this approach avoid common failure patterns?
- **Scalability Alignment**: Does this support growth patterns seen in competition?
- **Integration Compatibility**: Does this work with identified integration requirements?

**3C. Risk Mitigation Architecture**
Design architecture to address Phase 2 risk assessment:
- **Technical Risk Mitigation**: How does architecture address technical risks?
- **Scalability Risk Mitigation**: How does design handle growth challenges?
- **Integration Risk Mitigation**: How does architecture manage external dependencies?
- **User Adoption Risk Mitigation**: How does design support user success patterns?

### Step 4: Detailed Architecture Specification
**TestBot Architecture Documentation with FPT:**

**4A. System Architecture Diagram**
Create comprehensive architecture visualization:
- **Component Diagram**: All essential components and their relationships
- **Data Flow Diagram**: How information moves through the system
- **User Journey Mapping**: How architecture supports user workflows
- **Integration Architecture**: How system connects with external services

**4B. Component Specifications**
**FPT-Based Component Definition**:
```markdown
**For Each Component:**
- **Purpose**: What fundamental problem does this component solve?
- **Responsibilities**: What are its irreducible functions?
- **Interfaces**: What are its minimal necessary connection points?
- **Dependencies**: What other components does it fundamentally require?
- **Data Model**: What information does it manage?
- **Scalability Plan**: How does it handle growth?
```

**4C. Technical Stack Architecture**
Based on Phase 2 research and architectural requirements:
- **Database Architecture**: What data storage approach serves fundamental needs?
- **Application Layer**: What runtime architecture serves user and business requirements?
- **Integration Layer**: What connectivity approach serves external requirements?
- **Infrastructure Architecture**: What deployment approach serves scalability needs?

---

## Quality Gates & Validation with FPT Enhancement

### CODEFARM Multi-Agent Validation with FPT

**CodeFarmer Validation (FPT-Enhanced):**
- [ ] Architecture directly addresses fundamental user needs from discovery research
- [ ] System design aligns with validated business requirements and competitive insights
- [ ] Component identification based on irreducible functional requirements
- [ ] Architecture supports long-term vision while solving immediate problems

**Critibot Validation (FPT-Enhanced):**
- [ ] Every architectural decision justified against discovery research evidence
- [ ] Competitive analysis insights incorporated into design choices
- [ ] Risk assessment findings addressed through architectural mitigation
- [ ] No architectural complexity without fundamental requirement justification

**Programmatron Validation (FPT-Enhanced):**
- [ ] Technical architecture based on proven patterns from competitive research
- [ ] Component design follows minimal viable complexity principles
- [ ] Interface design serves essential integration requirements only
- [ ] Scalability approach aligned with growth patterns from market research

**TestBot Validation (FPT-Enhanced):**
- [ ] Architecture testable with clear component boundaries and interfaces
- [ ] Design supports user research findings with measurable interaction patterns
- [ ] Technical specifications complete enough for implementation planning
- [ ] Architecture documentation provides clear implementation guidance

### Anti-Hallucination Validation (FPT-Enhanced)
- [ ] No architectural patterns used without research validation or competitive precedent
- [ ] No technical stack choices without official documentation and capability validation
- [ ] No scalability claims without performance precedent research
- [ ] No integration assumptions without external system capability validation

### Architecture Completeness Criteria
- [ ] All fundamental user needs addressed through specific architectural components
- [ ] All competitive insights incorporated into design decisions
- [ ] All identified risks mitigated through architectural approaches
- [ ] All technical requirements from discovery research accommodated
- [ ] Clear component specifications ready for detailed implementation planning
- [ ] Architecture documentation comprehensive and implementation-ready
- [ ] Confidence scoring completed with justification (minimum 7.5/10 overall)

## Post-Condition Guarantee

### Systematic Output Validation
**Guaranteed Deliverables:**
1. **`phases/completed/phase_3_architecture.md`** - Complete system architecture documentation
2. **`specs/system_architecture.md`** - Detailed architecture specification with diagrams
3. **`specs/component_specifications.md`** - Individual component design and interfaces
4. **`specs/technical_stack.md`** - Complete technology stack with justification
5. **`specs/integration_architecture.md`** - External system integration design
6. **Updated `planning_prp.md`** - Architecture decisions integrated into project plan

### Context Preservation
**Architecture Intelligence Integration:**
- Discovery research → architectural decisions traceability maintained
- Competitive analysis → design pattern selection rationale documented
- Risk assessment → mitigation architecture mapping preserved
- User research → user experience architecture alignment recorded

### Tool Usage Justification (FPT-Enhanced)
**Optimal Tool Selection for Fundamental Architecture Operations:**
- **Read**: Discovery research integration and context loading
- **Write**: Systematic architecture documentation creation
- **Grep/Glob**: Research cross-reference and pattern validation
- **WebFetch**: Additional pattern research and validation when needed
- **Bash**: Directory structure management and validation automation

## Error Recovery & Rollback Strategy

### Architecture Invalidation Handling
**If Architecture Conflicts with Discovery Research:**
1. **Systematic Analysis**: Document specific conflicts with evidence
2. **Priority Assessment**: Determine if architecture or discovery needs updating
3. **Stakeholder Impact**: Evaluate effect of architectural changes on project timeline
4. **Evidence-Based Resolution**: Choose approach with strongest research backing
5. **Documentation Update**: Update all affected specifications with rationale

### Complexity Management
**If Architecture Becomes Over-Complex:**
1. **FPT Simplification**: Return to fundamental requirements and eliminate non-essential components
2. **Component Consolidation**: Merge components that don't have distinct fundamental purposes
3. **Interface Minimization**: Reduce connection points to essential interactions only
4. **Pattern Simplification**: Choose simpler proven patterns if complexity doesn't serve fundamental needs

## Confidence Scoring & Transition (FPT-Enhanced)

### Systematic Confidence Assessment
- **Requirements Alignment**: ___/10 (Architecture serves all fundamental user and business needs)
- **Technical Feasibility**: ___/10 (Design based on proven patterns and validated technologies)
- **Competitive Validation**: ___/10 (Approach aligned with successful competitive precedents)
- **Risk Mitigation**: ___/10 (Architecture addresses fundamental failure modes from research)
- **Implementation Readiness**: ___/10 (Specifications complete enough for detailed planning)
- **Overall Architecture Quality**: ___/10 (Comprehensive, justified, implementable system design)

**Minimum threshold for Phase 4**: Overall score ≥7.5/10 (highest threshold due to architectural criticality)

### Transition to Phase 4 (Research & Validation)
**Prerequisites Validated:**
- [ ] Complete system architecture designed with fundamental justification
- [ ] All discovery research integrated into architectural decisions
- [ ] Component specifications ready for detailed technology research
- [ ] Technical stack identified and justified through competitive analysis
- [ ] Integration requirements specified with external system understanding
- [ ] Architecture validated against user research and risk assessment

**Next Command:**
```
/codefarm-research-stack ${PROJECT_PATH} [identified-technologies]
```

---

## Output Summary

### Created Artifacts:
1. **`phases/completed/phase_3_architecture.md`** - Complete systematic architecture documentation
2. **`specs/system_architecture.md`** - Detailed architecture with FPT-based component design
3. **`specs/component_specifications.md`** - Individual component design with fundamental justification
4. **`specs/technical_stack.md`** - Technology stack with competitive validation
5. **`specs/integration_architecture.md`** - External system integration with research backing
6. **Updated `planning_prp.md`** - Architecture-enhanced project understanding

### FPT Enhancement Benefits:
- ✅ **Fundamental Architecture**: Design based on irreducible requirements
- ✅ **Research Integration**: Discovery intelligence drives architectural decisions
- ✅ **Competitive Validation**: Patterns proven successful in similar systems
- ✅ **Risk Mitigation**: Architecture addresses fundamental failure modes
- ✅ **Implementation Readiness**: Specifications support systematic development
- ✅ **Complexity Optimization**: Minimal viable architecture serving essential needs

---

**🎯 Phase 3 Complete**: System architecture designed with FPT-enhanced CODEFARM validation. Discovery-driven, competitively-validated, risk-mitigated architecture ready for technology stack research and validation.