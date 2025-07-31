---
allowed-tools: Write, Read, Glob, Grep, WebFetch, Bash
description: Interactive project initiation with CODEFARM systematic analysis and validation
argument-hint: [project-description] | Optional initial project description to start dialogue
---

# Phase 1: Interactive Project Initiation with CODEFARM Analysis

**Context**: You are using the CODEFARM methodology for systematic project initiation through structured dialogue. This combines interactive user experience with comprehensive multi-agent analysis for error-free project foundation.

## CODEFARM Multi-Agent Activation

**CodeFarmer:** Leading conversational requirements synthesis and project vision
**Critibot:** Challenging assumptions and identifying risks throughout dialogue
**Programmatron:** Analyzing technical feasibility and architecture options
**TestBot:** Defining validation criteria and success metrics

## Interactive Discovery Process

### Initial Project Context
**Starting Point:** ${ARGUMENTS:-"Interactive dialogue - no initial description provided"}

---

## Phase 1A: Interactive Requirements Gathering

### Step 1: Project Vision & Goals Discovery
**CodeFarmer Leads Dialogue:**

"Let's establish your project vision systematically. I'll guide you through a structured discovery process.

**Primary Question:** What is the high-level vision for your project? What problem are you trying to solve?

Please provide a 1-3 sentence description of your project's purpose."

**[WAIT FOR USER RESPONSE]**

**CodeFarmer Confirmation & Analysis:**
- Summarize understanding of the vision
- Extract core value proposition and objectives
- Identify target users and stakeholders
- Ask clarifying questions if vision is unclear

### Step 2: Core Features & User Stories Extraction
**CodeFarmer Continues:**

"Based on your vision, let's define the essential features.

**Key Question:** What are the 1-3 most critical features this project needs to have? Please describe them as user stories if possible (e.g., 'As a user, I want to...')."

**[WAIT FOR USER RESPONSE]**

**CodeFarmer Feature Analysis:**
- List captured features for confirmation
- Identify any missing critical functionality
- Prioritize features by importance and complexity
- Ask for confirmation before proceeding

### Step 3: Technology Stack Recommendation & Analysis
**Programmatron Technical Assessment:**

"Now I'll analyze the optimal technology stack for your requirements.

**Based on your features, I recommend analyzing the following stack options:**

[Programmatron provides 1-2 technology stack options with:]
- Technical feasibility assessment
- Pros and cons for each option
- Resource requirements and complexity analysis
- Integration capabilities and scalability

**Decision Point:** Which technology stack aligns with your preferences and constraints?"

**[WAIT FOR USER RESPONSE]**

**Programmatron Confirmation:**
- Restate chosen technology stack
- Validate technical feasibility (confidence score)
- Identify potential technical risks
- Confirm architectural approach

### Step 4: Research Requirements & Anti-Hallucination Setup
**Research Foundation Establishment:**

"To ensure error-free development, we need comprehensive research foundation.

**Programmatron Research Analysis:**
Based on [Chosen Stack], I recommend researching these official documentation sources:
- [List of essential official documentation URLs]
- [API references and integration guides]
- [Best practices and implementation patterns]

**Additional Research Question:** Are there any specific documentation pages, tutorials, or repositories you consider essential for this project?"

**[WAIT FOR USER RESPONSE]**

**Research Validation:**
- Compile complete research URL list
- Validate all URLs are official sources
- Create research directory structure plan
- Establish anti-hallucination protocols

### Step 5: Risk Assessment & Final Review
**Critibot Risk Challenge:**

"Before finalizing, let me challenge this plan systematically:

**Risk Analysis Questions:**
- What are the highest technical risks in this approach?
- Where might we encounter integration challenges?
- What dependencies could become bottlenecks?
- Are there any scalability concerns?
- What testing challenges should we anticipate?

**Final Plan Summary:**
- **Project Name:** [Derived or requested]
- **Core Vision:** [User's vision statement]
- **Essential Features:** [Confirmed feature list]
- **Technology Stack:** [Chosen stack with rationale]
- **Research Targets:** [Complete URL list]
- **Risk Mitigation:** [Key risks and strategies]

**Final Confirmation:** If this comprehensive plan is correct, I will generate the project foundation with systematic validation. Shall I proceed?"

**[WAIT FOR USER APPROVAL]**

---

## Phase 1B: Systematic Project Foundation Creation

### Foundation Setup Execution
**Programmatron Implementation:**

```bash
# Create systematic project structure in projects directory
mkdir -p projects/${PROJECT_NAME:-"new_project"}/{src,tests,docs,research,specs,config}
mkdir -p projects/${PROJECT_NAME:-"new_project"}/phases/{current,completed,upcoming}
```

### Document Generation

**1. Master Planning Document (`planning_prp.md`)**
```markdown
# ${PROJECT_NAME} - Master Project Plan

## Project Vision
[User's confirmed vision]

## Core Features
[Confirmed feature list with priorities]

## Technology Stack
[Chosen stack with technical rationale]

## Research Foundation
[Complete list of research sources]

## Risk Assessment
[Identified risks with mitigation strategies]

## Success Criteria
[Measurable outcomes and validation metrics]

## Phase Progression
- [x] Phase 1: Interactive Initiation (COMPLETE)
- [ ] Phase 2: Deep Discovery Research
- [ ] Phase 3: System Architecture Planning
[... remaining phases]

## Next Steps
Execute: /codefarm-discovery-deep-dive projects/${PROJECT_NAME}
```

**2. Current Phase Document (`phases/current/phase_1_initiation.md`)**
[Detailed documentation of all dialogue responses and analysis]

**3. Research Directory Structure**
```
research/
├── [technology_1]/
├── [technology_2]/
└── research_manifest.md
```

## Quality Gates & Validation

### CODEFARM Multi-Agent Validation

**CodeFarmer Validation:**
- [ ] Project vision is clear, measurable, and achievable
- [ ] User requirements captured comprehensively
- [ ] Features align with stated vision and goals

**Critibot Validation:**
- [ ] All major risks identified with mitigation strategies
- [ ] No critical assumptions left unvalidated
- [ ] Technical approach realistic and achievable

**Programmatron Validation:**
- [ ] Technology stack technically sound and feasible
- [ ] Architecture approach scalable and maintainable
- [ ] Integration challenges identified and addressable

**TestBot Validation:**
- [ ] Success criteria are quantifiable and measurable
- [ ] Testing strategy defined for each development phase
- [ ] Quality assurance checkpoints established

### Anti-Hallucination Validation
- [ ] No technology assumptions made without research validation
- [ ] All recommendations based on documented capabilities
- [ ] Research sources are official and up-to-date
- [ ] Technical feasibility claims backed by evidence

### Phase Completion Criteria
- [ ] Interactive dialogue completed through all 5 steps
- [ ] User approval received for final plan
- [ ] Project foundation structure created
- [ ] planning_prp.md generated with complete information
- [ ] Research requirements identified and documented
- [ ] Confidence scoring completed (minimum 6/10 overall)

## Confidence Scoring & Transition

### Systematic Confidence Assessment
- **Project Clarity**: ___/10
- **Technical Feasibility**: ___/10
- **User Alignment**: ___/10
- **Research Foundation**: ___/10
- **Risk Management**: ___/10
- **Overall Project Viability**: ___/10

**Minimum threshold for Phase 2**: Overall score ≥6/10

### Transition to Phase 2 (Discovery)
**Prerequisites Validated:**
- [ ] Project vision confirmed and documented
- [ ] Technical approach validated with confidence
- [ ] Research requirements comprehensively identified
- [ ] User alignment confirmed throughout process

**Next Command:**
```
/codefarm-discovery-deep-dive projects/${PROJECT_NAME}
```

---

## Output Summary

### Created Artifacts:
1. **`planning_prp.md`** - Master project plan (under 200 lines)
2. **`phases/current/phase_1_initiation.md`** - Complete dialogue documentation
3. **Project directory structure** - Systematic organization
4. **Research manifest** - Anti-hallucination foundation

### User Experience:
- ✅ **Interactive**: Guided through structured dialogue
- ✅ **Systematic**: CODEFARM multi-agent analysis
- ✅ **Validated**: Quality gates and confidence scoring
- ✅ **Anti-Hallucination**: Research-driven approach

---

**🎯 Phase 1 Complete**: Interactive project foundation established with systematic CODEFARM validation. User-guided dialogue combined with comprehensive multi-agent analysis. Ready for deep discovery research phase.