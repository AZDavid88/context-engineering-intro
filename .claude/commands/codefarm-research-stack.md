---
allowed-tools: WebFetch, WebSearch, Read, Write, Grep, Glob, Bash
description: Comprehensive technology stack research and validation with CODEFARM anti-hallucination protocols
argument-hint: [project-path] [technologies] | Project path and optional specific technologies to research
pre-conditions: Phase 3 architecture complete with technical stack identified
post-conditions: Technology research validated and implementation-ready
rollback-strategy: Preserve architecture decisions, flag technology gaps for alternative research
---

# Phase 4: Technology Stack Research with CODEFARM + FPT Analysis

**Context**: You are using the CODEFARM methodology for systematic technology stack research and validation. This phase validates architectural technology decisions through comprehensive official documentation research and anti-hallucination protocols.

## Fundamental Operation Definition
**Core Operation**: Transform architectural technology stack decisions into validated, implementation-ready technology foundation through systematic research of official documentation, capabilities, and integration patterns.

**FPT Question**: What are the irreducible validation requirements to confidently implement with chosen technologies?

## Pre-Condition Validation
**Systematic Prerequisites Check:**

### Phase 3 Output Validation
```bash
# Validate Phase 3 completion
PROJECT_PATH="${ARGUMENTS%% *}"  # First argument is project path
TECHNOLOGIES="${ARGUMENTS#* }"   # Optional second argument is technologies

if [[ ! -f "$PROJECT_PATH/phases/completed/phase_3_architecture.md" ]]; then
    echo "ERROR: Phase 3 incomplete - architecture documentation not found"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/system_architecture.md" ]]; then
    echo "ERROR: System architecture specification missing"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/technical_stack.md" ]]; then
    echo "ERROR: Technical stack specification missing from Phase 3"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/component_specifications.md" ]]; then
    echo "ERROR: Component specifications missing from Phase 3"
    exit 2
fi
```

**Pre-Condition Requirements:**
- [ ] Phase 3 architecture documentation complete and validated
- [ ] System architecture with component design finalized
- [ ] Technical stack identified and justified through competitive analysis
- [ ] Component specifications ready for technology validation
- [ ] Integration architecture defined with external system requirements
- [ ] Phase 3 confidence score ≥ 7.5/10

## CODEFARM Multi-Agent Activation with FPT Enhancement

**CodeFarmer (Research Strategist with FPT):** 
"I'll apply First Principles Thinking to research technology validation systematically. We'll verify every technology choice against fundamental project requirements and implementation realities."

**Critibot (Technology Challenger with FPT):**
"I'll challenge every technology assumption with official documentation evidence. No technology proceeds without proven capability validation and integration compatibility."

**Programmatron (Implementation Researcher with FPT):**
"I'll research implementation patterns, performance characteristics, and integration requirements using only official sources and proven precedents."

**TestBot (Technology Validator with FPT):**
"I'll validate all technology research against project requirements and ensure our anti-hallucination protocols prevent unverified technology claims."

---

## Phase 4 Core Process with FPT-Enhanced Technology Research

### Step 1: Architecture Technology Extraction
**CodeFarmer Technology Analysis:**

**1A. Load Complete Architecture Context**
```markdown
**Architecture Research Integration:**
- System Architecture: @${PROJECT_PATH}/specs/system_architecture.md
- Technical Stack: @${PROJECT_PATH}/specs/technical_stack.md  
- Component Specifications: @${PROJECT_PATH}/specs/component_specifications.md
- Integration Architecture: @${PROJECT_PATH}/specs/integration_architecture.md
- Discovery Research: @${PROJECT_PATH}/phases/completed/phase_2_discovery.md
```

**1B. FPT Technology Requirements Extraction**
**Fundamental Question**: What are the irreducible technology capabilities needed for this architecture?

Systematic technology requirement mapping:
- **Data Technologies**: What storage, retrieval, and management capabilities are essential?
- **Compute Technologies**: What processing, logic, and algorithmic capabilities are required?
- **Interface Technologies**: What user interaction and API capabilities are fundamental?
- **Integration Technologies**: What connectivity and communication capabilities are necessary?
- **Infrastructure Technologies**: What deployment, scaling, and operational capabilities are critical?

**1C. Research Priority Matrix**
Based on architecture criticality and implementation risk:
```markdown
**Technology Research Priorities:**
1. **Critical Path Technologies**: Technologies that block implementation if inadequate
2. **Integration Risk Technologies**: Technologies with complex external dependencies
3. **Performance Critical Technologies**: Technologies affecting fundamental user experience
4. **Security Critical Technologies**: Technologies handling sensitive operations
5. **Supporting Technologies**: Technologies that enhance but don't block implementation
```

### Step 2: Official Documentation Research & Validation
**Programmatron Systematic Documentation Research:**

**2A. Fundamental Capability Research**
**FPT Question**: What are the actual documented capabilities vs our architectural assumptions?

Systematic official documentation research:
```markdown
**For Each Technology in Stack:**
1. **Official Documentation Sources**: Find authoritative documentation URLs
2. **Capability Matrix**: Document actual features vs architectural requirements
3. **Implementation Patterns**: Research recommended usage patterns and best practices
4. **Integration Documentation**: Official guides for connecting with other stack components
5. **Performance Characteristics**: Official benchmarks and scalability documentation
6. **Security Documentation**: Official security guidelines and best practices
```

**2B. Anti-Hallucination Research Protocol**
**Critical**: All technology research must follow strict anti-hallucination protocols:

Research validation requirements:
- **Primary Sources Only**: Official documentation, API references, and authorized guides
- **Version Specific**: Research current stable versions and compatibility matrices
- **Implementation Evidence**: Code examples from official sources or verified repositories
- **Performance Data**: Official benchmarks or peer-reviewed performance studies
- **Community Validation**: Official community forums, issues, and support channels

**2C. Technology Compatibility Matrix**
Research technology integration patterns:
- **Stack Compatibility**: How do chosen technologies work together officially?
- **Version Compatibility**: What version combinations are officially supported?
- **Integration Patterns**: What are the documented methods for connecting technologies?
- **Configuration Requirements**: What setup and configuration is officially required?

### Step 3: Implementation Pattern & Performance Research
**Critibot Implementation Challenge with FPT:**

**3A. Fundamental Implementation Requirements**
**FPT Question**: What are the minimal implementation requirements for each technology?

Systematic implementation research:
- **Setup Requirements**: What are the fundamental setup and configuration needs?
- **Development Requirements**: What tools, SDKs, and development environments are needed?
- **Deployment Requirements**: What infrastructure and operational requirements exist?
- **Maintenance Requirements**: What ongoing operational and update requirements exist?

**3B. Performance & Scalability Validation**
Research actual performance against architectural requirements:
```markdown
**Performance Research Protocol:**
1. **Official Benchmarks**: Find authoritative performance documentation
2. **Scalability Patterns**: Research official scaling approaches and limitations
3. **Resource Requirements**: Document CPU, memory, storage, and network needs
4. **Performance Optimization**: Official guidance for performance tuning
5. **Bottleneck Identification**: Known performance limitations and mitigation strategies
```

**3C. Security & Compliance Research**
Validate security capabilities against project requirements:
- **Security Features**: Official security capabilities and implementation guides
- **Compliance Support**: Documentation for regulatory compliance requirements
- **Vulnerability Management**: Official security update and patch processes
- **Authentication/Authorization**: Official identity and access management features

### Step 4: Technology Risk Assessment & Mitigation
**TestBot Risk Validation with FPT:**

**4A. Technology Risk Matrix**
**FPT Question**: What are the fundamental risks of each technology choice?

Systematic risk assessment:
```markdown
**Risk Categories:**
- **Maturity Risks**: How stable and mature is each technology?
- **Support Risks**: What is the official support and community ecosystem?
- **Integration Risks**: What are documented integration challenges and solutions?
- **Performance Risks**: What are known performance limitations and bottlenecks?
- **Security Risks**: What are documented security vulnerabilities and mitigations?
- **Evolution Risks**: What is the technology roadmap and long-term viability?
```

**4B. Alternative Technology Research**
For high-risk technologies, research validated alternatives:
- **Alternative Technologies**: What other technologies could serve the same fundamental purpose?
- **Migration Paths**: If technology choice proves inadequate, what are the migration options?
- **Hybrid Approaches**: Can risk be mitigated through technology combinations?
- **Fallback Strategies**: What are the contingency plans if technology fails to meet requirements?

**4C. Implementation Risk Mitigation**
Research risk mitigation strategies:
- **Proof of Concept Requirements**: What minimal implementations can validate technology fit?
- **Testing Strategies**: How can technology capabilities be validated before full implementation?
- **Monitoring Requirements**: What operational monitoring is needed to track technology performance?
- **Support Strategies**: What support resources and expertise are available?

---

## Quality Gates & Validation with FPT Enhancement

### CODEFARM Multi-Agent Validation with FPT

**CodeFarmer Validation (FPT-Enhanced):**
- [ ] All architectural technology requirements researched with official documentation
- [ ] Technology research directly addresses fundamental capability needs from architecture
- [ ] Research findings validate or challenge architectural technology assumptions
- [ ] Technology choices support project vision and user requirements from previous phases

**Critibot Validation (FPT-Enhanced):**
- [ ] Every technology claim backed by official documentation and evidence
- [ ] All technology assumptions from architecture phase validated or updated
- [ ] Risk assessment covers fundamental technology failure modes and limitations
- [ ] No technology choices proceed without comprehensive capability validation

**Programmatron Validation (FPT-Enhanced):**
- [ ] Implementation patterns researched with official examples and best practices
- [ ] Performance characteristics validated against architectural requirements
- [ ] Integration requirements researched with documented connection patterns
- [ ] Security capabilities validated against project security requirements

**TestBot Validation (FPT-Enhanced):**
- [ ] All research sources are official, current, and authoritative
- [ ] Technology capabilities claims backed by documented evidence
- [ ] Performance assumptions validated with official benchmarks or studies
- [ ] Risk mitigation strategies proven effective through research precedent

### Anti-Hallucination Validation (FPT-Enhanced)
- [ ] No technology capability claims without official documentation validation
- [ ] No performance assumptions without verified benchmark or study backing
- [ ] No integration assumptions without documented compatibility research
- [ ] No security claims without official security documentation validation

### Technology Research Completeness Criteria
- [ ] All architectural technologies researched with comprehensive official documentation
- [ ] Technology capability matrix completed with evidence-based validation
- [ ] Integration compatibility researched and validated between all stack components
- [ ] Performance requirements validated against official benchmarks and limitations
- [ ] Security requirements validated against official security documentation
- [ ] Risk assessment completed with mitigation strategies and alternative options
- [ ] Implementation readiness validated with setup, development, and deployment requirements
- [ ] Confidence scoring completed with evidence backing (minimum 8/10 overall)

## Post-Condition Guarantee

### Systematic Output Validation
**Guaranteed Deliverables:**
1. **`phases/completed/phase_4_technology_research.md`** - Complete technology research documentation
2. **`research/[technology]/`** - Individual technology research directories with official sources
3. **`specs/technology_validation.md`** - Technology capability validation against requirements
4. **`specs/implementation_requirements.md`** - Detailed implementation setup and configuration requirements
5. **`specs/technology_risk_assessment.md`** - Risk analysis with mitigation strategies
6. **`specs/integration_validation.md`** - Technology compatibility and integration research
7. **Updated `planning_prp.md`** - Technology research integrated into project plan

### Context Preservation
**Technology Intelligence Integration:**
- Architecture decisions → technology validation traceability maintained
- Research organized by technology with cross-references to architectural requirements
- Implementation patterns documented with official source attribution
- Risk assessments linked to specific technology choices and mitigation strategies

### Tool Usage Justification (FPT-Enhanced)
**Optimal Tool Selection for Fundamental Technology Research:**
- **WebFetch/WebSearch**: Official documentation research and validation
- **Read/Write**: Architecture context loading and research documentation creation
- **Grep/Glob**: Cross-reference validation and research organization
- **Bash**: Automated research directory management and validation

## Error Recovery & Rollback Strategy

### Technology Validation Failure Handling
**If Technology Research Shows Architectural Assumptions Are Wrong:**
1. **Evidence Documentation**: Document specific capability gaps with official source evidence
2. **Impact Assessment**: Evaluate which architectural decisions need updating
3. **Alternative Research**: Research validated alternative technologies with similar capabilities
4. **Architecture Consultation**: Recommend architecture updates based on technology research
5. **Stakeholder Communication**: Prepare evidence-based recommendations for technology changes

### Research Quality Control
**If Official Documentation Is Inadequate or Conflicting:**
1. **Source Validation**: Verify documentation is current and authoritative
2. **Multi-Source Validation**: Cross-reference multiple official sources
3. **Community Research**: Check official forums and issue trackers for clarification
4. **Expert Consultation**: Recommend reaching out to technology maintainers or experts
5. **Risk Documentation**: Flag research gaps as implementation risks requiring validation

## Confidence Scoring & Transition (FPT-Enhanced)

### Systematic Confidence Assessment
- **Technology Capability Validation**: ___/10 (All technologies validated against requirements with evidence)
- **Integration Compatibility**: ___/10 (Technology stack compatibility researched and validated)
- **Implementation Readiness**: ___/10 (Setup, development, deployment requirements researched)
- **Performance Validation**: ___/10 (Performance characteristics validated against requirements)
- **Security Validation**: ___/10 (Security capabilities researched and validated)
- **Risk Management**: ___/10 (Technology risks identified with proven mitigation strategies)
- **Overall Technology Research Quality**: ___/10 (Comprehensive, evidence-based validation)

**Minimum threshold for Phase 5**: Overall score ≥8/10 (high threshold due to implementation dependency)

### Transition to Phase 5 (Complete System Specification)
**Prerequisites Validated:**
- [ ] All architectural technologies researched with comprehensive official documentation
- [ ] Technology capabilities validated against all architectural requirements
- [ ] Integration compatibility validated between all stack components
- [ ] Implementation requirements documented with official setup guides
- [ ] Performance characteristics validated against project requirements
- [ ] Security capabilities validated against project security needs
- [ ] Risk assessment complete with alternative technologies and mitigation strategies

**Next Command:**
```
/codefarm-spec-complete-system ${PROJECT_PATH}
```

---

## Output Summary

### Created Artifacts:
1. **`phases/completed/phase_4_technology_research.md`** - Complete systematic technology research
2. **Expanded `research/` directories** - Technology-specific research with official documentation
3. **`specs/technology_validation.md`** - Evidence-based technology capability validation
4. **`specs/implementation_requirements.md`** - Official implementation setup and configuration guide
5. **`specs/technology_risk_assessment.md`** - Risk analysis with alternative options
6. **`specs/integration_validation.md`** - Technology compatibility research and validation
7. **Updated `planning_prp.md`** - Technology-validated project understanding

### FPT Enhancement Benefits:
- ✅ **Anti-Hallucination Research**: All technology claims backed by official documentation
- ✅ **Capability Validation**: Architecture assumptions validated against actual technology capabilities
- ✅ **Integration Assurance**: Technology stack compatibility researched and validated
- ✅ **Implementation Readiness**: Official setup and configuration requirements documented
- ✅ **Risk Mitigation**: Technology risks identified with proven alternative strategies
- ✅ **Evidence-Based**: All research decisions backed by authoritative sources

---

**🎯 Phase 4 Complete**: Technology stack comprehensively researched and validated with FPT-enhanced CODEFARM anti-hallucination protocols. Official documentation validated technology foundation ready for complete system specification.