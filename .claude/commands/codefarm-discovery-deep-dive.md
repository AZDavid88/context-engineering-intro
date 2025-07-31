---
allowed-tools: WebFetch, WebSearch, Read, Write, Grep, Glob, Bash
description: Deep discovery research and competitive analysis with CODEFARM systematic validation
argument-hint: [project-path] | Path to project directory from Phase 1
pre-conditions: Phase 1 complete with planning_prp.md and research manifest
post-conditions: Comprehensive discovery foundation ready for architecture phase
rollback-strategy: Preserve Phase 1 state, flag discovery gaps for re-analysis
---

# Phase 2: Deep Discovery with CODEFARM + FPT Analysis

**Context**: You are using the CODEFARM methodology for systematic project discovery through comprehensive research and competitive analysis. This phase builds on Phase 1 foundation with FPT-enhanced validation.

## Fundamental Operation Definition
**Core Operation**: Transform initial project vision into comprehensive domain understanding through systematic research, competitive analysis, and technical deep-dive.

**FPT Question**: What are the irreducible components needed to fully understand a project's context and feasibility?

## Pre-Condition Validation
**Systematic Prerequisites Check:**

### Phase 1 Output Validation
```bash
# Validate Phase 1 completion
PROJECT_PATH="${ARGUMENTS}"
if [[ ! -f "$PROJECT_PATH/planning_prp.md" ]]; then
    echo "ERROR: Phase 1 incomplete - planning_prp.md not found"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/research/research_manifest.md" ]]; then
    echo "ERROR: Research manifest missing from Phase 1"
    exit 2
fi

if [[ ! -d "$PROJECT_PATH/phases/current" ]]; then
    echo "ERROR: Phase structure incomplete"
    exit 2
fi
```

**Pre-Condition Requirements:**
- [ ] planning_prp.md exists with complete Phase 1 analysis
- [ ] Research manifest established with initial technology URLs
- [ ] Project directory structure properly created
- [ ] Confidence score from Phase 1 ≥ 6.0/10

## CODEFARM Multi-Agent Activation with FPT Enhancement

**CodeFarmer (Discovery Lead with FPT):** 
"I'll apply First Principles Thinking to expand your project understanding systematically. We'll break down the domain into fundamental components and rebuild comprehensive knowledge."

**Critibot (Quality Enforcer with FPT):**
"I'll challenge every assumption from Phase 1 and identify gaps in understanding. No discovery claim proceeds without evidence validation."

**Programmatron (Technical Researcher with FPT):**
"I'll research implementation patterns and architectural precedents, focusing on fundamental technical requirements rather than surface-level solutions."

**TestBot (Evidence Validator with FPT):**
"I'll validate all research findings and ensure our discovery methodology produces reliable, actionable intelligence."

---

## Phase 2 Core Process with Error Recovery

### Step 1: Context Expansion & Domain Analysis
**CodeFarmer Discovery Systematic Process:**

**1A. Load Phase 1 Foundation**
```markdown
**Phase 1 Analysis Review:**
- Project Vision: !`head -20 ${PROJECT_PATH}/planning_prp.md | grep -A 5 "Project Vision"`
- Technology Stack: !`grep -A 10 "Technology Stack" ${PROJECT_PATH}/planning_prp.md`
- Initial Research Targets: @${PROJECT_PATH}/research/research_manifest.md
```

**1B. FPT Domain Breakdown**
**Fundamental Question**: What are the core domains this project touches?

Systematic domain mapping:
- **User Domain**: Who are the fundamental user types and their core needs?
- **Technical Domain**: What are the irreducible technical requirements?
- **Business Domain**: What are the fundamental value creation mechanisms?
- **Competitive Domain**: What existing solutions address similar fundamental problems?
- **Integration Domain**: What systems must fundamentally interact with this project?

**1C. Research Expansion Strategy**
Based on domain analysis, expand research into:
- User behavior studies and pain point documentation
- Technical implementation patterns and architectural precedents
- Market analysis and competitive landscape research
- Integration requirements and dependency analysis

### Step 2: Competitive Intelligence with FPT Analysis
**Programmatron Technical Deep-Dive:**

**2A. Fundamental Competitive Analysis**
**FPT Question**: What are the irreducible approaches to solving this problem?

Systematic competitive research:
```markdown
**Research Process:**
1. Identify 3-5 direct competitors or similar solutions
2. Analyze fundamental approaches (not just features)
3. Research technical architectures and implementation patterns
4. Document fundamental trade-offs and design decisions
5. Identify gaps in existing solutions
```

**2B. Technical Pattern Research**
Research implementation patterns:
- Architectural approaches used by successful similar projects
- Technology stack patterns and their fundamental trade-offs
- Performance, scalability, and security considerations
- Integration patterns and data flow architectures

**2C. Anti-Hallucination Research Validation**
**Critical**: All competitive analysis must be supported by:
- Official documentation from researched solutions
- Public architecture blog posts or technical papers
- Open source code analysis where available
- Market research reports or user studies

### Step 3: Risk Deep-Dive & Mitigation Research
**Critibot Risk Challenge with FPT:**

**3A. Fundamental Risk Categories**
**FPT Question**: What are the irreducible failure modes for this type of project?

Systematic risk analysis:
- **Technical Risks**: What fundamental technical challenges could derail the project?
- **Market Risks**: What fundamental market changes could invalidate the premise?
- **Resource Risks**: What fundamental resource constraints could prevent completion?
- **Integration Risks**: What fundamental integration challenges could cause failure?
- **User Adoption Risks**: What fundamental user behavior assumptions could be wrong?

**3B. Mitigation Strategy Research**
For each fundamental risk, research:
- How similar projects have handled these challenges
- Proven mitigation strategies and their effectiveness
- Early warning indicators and monitoring approaches
- Fallback options and contingency planning

### Step 4: Stakeholder & User Research Deep-Dive
**TestBot Evidence Validation:**

**4A. User Research Validation**
**FPT Question**: Who are the fundamental users and what are their irreducible needs?

Research validation requirements:
- User persona documentation with evidence backing
- User journey mapping with pain point validation
- Usage pattern research from similar solutions
- User feedback and validation data from comparable projects

**4B. Stakeholder Impact Analysis**
Comprehensive stakeholder research:
- Internal stakeholders and their fundamental concerns
- External stakeholders and integration requirements
- Regulatory or compliance stakeholders and requirements
- Technical stakeholders and architectural constraints

---

## Quality Gates & Validation with FPT Enhancement

### CODEFARM Multi-Agent Validation with FPT

**CodeFarmer Validation (FPT-Enhanced):**
- [ ] Domain understanding expanded systematically from fundamental principles
- [ ] All core domains identified and researched comprehensively
- [ ] Research findings challenge and validate Phase 1 assumptions
- [ ] User understanding deepened with evidence-based insights

**Critibot Validation (FPT-Enhanced):**
- [ ] Every research claim backed by official documentation or evidence
- [ ] All Phase 1 assumptions validated or updated based on discovery
- [ ] Risk analysis covers fundamental failure modes, not just surface issues
- [ ] Competitive analysis based on fundamental approaches, not just features

**Programmatron Validation (FPT-Enhanced):**
- [ ] Technical research covers implementation patterns, not just technology lists
- [ ] Architecture understanding based on proven patterns and trade-offs
- [ ] Integration requirements researched with actual implementation examples
- [ ] Technology stack validated against fundamental project requirements

**TestBot Validation (FPT-Enhanced):**
- [ ] All research sources are official, current, and credible
- [ ] User research backed by data, not assumptions
- [ ] Competitive analysis includes actual usage and performance data
- [ ] Risk mitigation strategies proven effective in similar contexts

### Anti-Hallucination Validation (FPT-Enhanced)
- [ ] No technology claims made without official documentation research
- [ ] No competitive analysis without verified source material
- [ ] No user behavior assumptions without research backing
- [ ] No architectural decisions without pattern precedent research

### Phase Completion Criteria (FPT-Enhanced)
- [ ] Comprehensive research documentation created and organized
- [ ] Phase 1 assumptions validated or updated based on discovery
- [ ] Competitive landscape mapped with fundamental approach analysis
- [ ] Risk assessment includes fundamental failure modes and proven mitigations
- [ ] User and stakeholder research provides actionable insights
- [ ] Technology and architecture research ready for Phase 3 planning
- [ ] Confidence scoring completed with evidence backing (minimum 7/10 overall)

## Post-Condition Guarantee

### Systematic Output Validation
**Guaranteed Deliverables:**
1. **`phases/completed/phase_2_discovery.md`** - Complete discovery research documentation
2. **`research/` directory expansion** - Organized research by domain with official sources
3. **`specs/competitive_analysis.md`** - Fundamental competitive approach analysis
4. **`specs/risk_assessment.md`** - Comprehensive risk analysis with mitigation strategies
5. **`specs/user_research.md`** - Evidence-based user and stakeholder analysis
6. **Updated `planning_prp.md`** - Phase 1 assumptions validated or updated

### Context Preservation
**Discovery Intelligence Integration:**
- All research organized in searchable, referenceable format
- Phase 1 → Phase 2 evolution clearly documented
- Assumption changes tracked with evidence rationale
- Research sources maintained for future validation

### Tool Usage Justification (FPT-Enhanced)
**Optimal Tool Selection for Fundamental Operations:**
- **WebFetch/WebSearch**: Official documentation and research source gathering
- **Read/Write**: Context loading and systematic documentation creation
- **Grep/Glob**: Research organization and cross-reference validation
- **Bash**: Automated validation and directory structure management

## Error Recovery & Rollback Strategy

### Discovery Contradiction Handling
**If Discovery Contradicts Phase 1:**
1. **Systematic Analysis**: Document specific contradictions with evidence
2. **Impact Assessment**: Evaluate which Phase 1 assumptions need updating
3. **Stakeholder Communication**: Prepare evidence-based recommendation for changes
4. **Controlled Update**: Update planning_prp.md with discovery-driven insights
5. **Validation Loop**: Re-run confidence scoring with updated understanding

### Research Quality Control
**If Research Sources Prove Inadequate:**
1. **Source Validation**: Verify all sources are official and current
2. **Alternative Research**: Find additional authoritative sources
3. **Evidence Cross-Check**: Validate findings across multiple credible sources
4. **Documentation Update**: Ensure all claims have supporting evidence

## Confidence Scoring & Transition (FPT-Enhanced)

### Systematic Confidence Assessment
- **Domain Understanding**: ___/10 (Evidence-based comprehension of all relevant domains)
- **Competitive Intelligence**: ___/10 (Fundamental approach analysis with credible research)
- **Technical Feasibility**: ___/10 (Implementation patterns researched and validated)
- **Risk Assessment**: ___/10 (Fundamental failure modes identified with mitigation)
- **User Validation**: ___/10 (Evidence-based user understanding with research backing)
- **Overall Discovery Quality**: ___/10 (Comprehensive, evidence-based project understanding)

**Minimum threshold for Phase 3**: Overall score ≥7/10 (higher than Phase 1 due to expanded requirements)

### Transition to Phase 3 (Architecture)
**Prerequisites Validated:**
- [ ] Comprehensive domain understanding established with evidence
- [ ] Competitive landscape mapped with fundamental approaches
- [ ] Technical patterns and implementation precedents researched
- [ ] Risk assessment covers fundamental failure modes
- [ ] User and stakeholder understanding validated with research
- [ ] Phase 1 assumptions validated or systematically updated

**Next Command:**
```
/codefarm-architect-system ${PROJECT_PATH}
```

---

## Output Summary

### Created Artifacts:
1. **`phases/completed/phase_2_discovery.md`** - Complete systematic discovery documentation
2. **Expanded `research/` directory** - Domain-organized research with official sources
3. **`specs/competitive_analysis.md`** - Fundamental competitive approach analysis
4. **`specs/risk_assessment.md`** - Evidence-based risk analysis with proven mitigations
5. **`specs/user_research.md`** - Research-validated user and stakeholder insights
6. **Updated `planning_prp.md`** - Discovery-enhanced project understanding

### FPT Enhancement Benefits:
- ✅ **Fundamental Thinking**: Domain breakdown from first principles
- ✅ **Evidence-Based**: All research backed by official sources
- ✅ **Systematic Validation**: Pre/post conditions with error recovery
- ✅ **Quality Gates**: Multi-agent validation with FPT enhancement
- ✅ **Context Preservation**: Cumulative intelligence building across phases
- ✅ **Tool Optimization**: Minimal effective tool set for fundamental operations

---

**🎯 Phase 2 Complete**: Deep discovery foundation established with FPT-enhanced CODEFARM validation. Evidence-based domain understanding ready for systematic architecture planning.