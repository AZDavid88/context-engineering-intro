---
allowed-tools: Read, Write, Glob, Grep, Bash(wc:*)
description: Break down large specification files into manageable, focused components
argument-hint: [large-spec-file] | will scan for large spec files if none provided
---

# Specification Breakdown & Modularization

**Context**: You are using the CodeFarm methodology to break down large, unwieldy specification files into focused, manageable components that improve AI context and development workflow.

## Large Specification Analysis

### 1. Target File Assessment
**File to break down**: ${ARGUMENTS:-"[Will scan for large spec files]"}

**File size and discovery using Claude Code tools:**
- If specific file provided: Use Bash tool with command `wc -l ${ARGUMENTS}` to check line count
- If no file provided: Use Glob tool with patterns "*.md" in specs directories, "*planning*.md", "*prp*.md" to find specification files
- Use head_limit parameter with value 5 to show first 5 matches

**Analysis steps:**
- **Content analysis**: Identify major sections and themes
- **Context complexity**: Assess cognitive load and AI token usage

### 2. Content Structure Analysis
Read and analyze the large specification file:

#### Main Sections Identification
- Extract major headings and themes
- Identify distinct functional areas
- Find independent components that can be separated
- Locate dependencies between sections

#### Size Metrics
- **Total lines**: Count lines in specification
- **Major sections**: Number of top-level sections
- **Complexity indicators**: Look for nested structures, code blocks, detailed requirements

### 3. Breakdown Strategy Planning
Design the optimal breakdown approach:

#### Separation Criteria
- **Functional boundaries**: Group related functionality
- **Independence level**: Separate components that can work standalone
- **Implementation phases**: Break by development timeline
- **Team ownership**: Separate by team responsibilities

## Modular Specification Creation

### 4. Create Specification Index
Generate a master index that ties everything together:

```markdown
# [Project] Specification Index

## Overview
Brief project description and current status.

## Specification Modules

### Core System
- [`specs/core/architecture.md`](specs/core/architecture.md) - System architecture and design principles
- [`specs/core/data_models.md`](specs/core/data_models.md) - Core data structures and models
- [`specs/core/api_design.md`](specs/core/api_design.md) - API specifications and contracts

### Features
- [`specs/features/user_management.md`](specs/features/user_management.md) - User authentication and management
- [`specs/features/data_processing.md`](specs/features/data_processing.md) - Data processing workflows
- [`specs/features/analytics.md`](specs/features/analytics.md) - Analytics and reporting features

### Infrastructure
- [`specs/infrastructure/deployment.md`](specs/infrastructure/deployment.md) - Deployment and DevOps
- [`specs/infrastructure/monitoring.md`](specs/infrastructure/monitoring.md) - Monitoring and observability
- [`specs/infrastructure/security.md`](specs/infrastructure/security.md) - Security requirements and implementation

### Implementation
- [`specs/implementation/phase_1.md`](specs/implementation/phase_1.md) - Phase 1 implementation plan
- [`specs/implementation/phase_2.md`](specs/implementation/phase_2.md) - Phase 2 implementation plan
- [`specs/implementation/integration_testing.md`](specs/implementation/integration_testing.md) - Testing strategy

## Quick Reference
- **Current Phase**: [Current development phase]
- **Priority Features**: [High-priority items]
- **Dependencies**: [External dependencies and constraints]
- **Team Assignments**: [Who owns what]

## Navigation Tips
- Start with `specs/core/architecture.md` for system understanding
- Check `specs/implementation/phase_X.md` for current development focus
- Refer to feature specs for detailed requirements
- Use infrastructure specs for deployment and operations
```

### 5. Extract Core Architecture
Create `specs/core/architecture.md`:

```markdown
# System Architecture Specification

## Overview
[High-level system purpose and design philosophy]

## Architecture Principles
### Design Philosophy
- [Core design principles]
- [Architectural patterns used]
- [Technology choices rationale]

### System Boundaries
- [What the system does and doesn't do]
- [Integration points with external systems]
- [Data flow boundaries]

## Component Architecture
### High-Level Components
- **[Component 1]**: [Purpose and responsibilities]
- **[Component 2]**: [Purpose and responsibilities]
- **[Component 3]**: [Purpose and responsibilities]

### Component Interactions
[How components communicate and depend on each other]

## Technology Stack
### Core Technologies
- **Backend**: [Technology choices and rationale]
- **Database**: [Database design and rationale]
- **API Layer**: [API technology and design]
- **Frontend**: [If applicable]

### Infrastructure
- **Deployment**: [Deployment strategy]
- **Monitoring**: [Monitoring approach]
- **Security**: [Security architecture]

## Data Architecture
### Data Models
- [Core data entities and relationships]
- [Data flow patterns]
- [Storage strategies]

### Integration Patterns
- [How data moves between components]
- [External data sources]
- [Data transformation patterns]

## Decision Records
### Major Architectural Decisions
- **[Decision 1]**: [Rationale and implications]
- **[Decision 2]**: [Rationale and implications]

### Trade-offs Made
- [Performance vs. simplicity trade-offs]
- [Security vs. usability considerations]
- [Scalability decisions]

## Implementation Guidelines
### Development Patterns
- [Coding standards and patterns]
- [Testing strategies]
- [Deployment procedures]

### Quality Gates
- [Performance requirements]
- [Security requirements]
- [Reliability standards]
```

### 6. Create Feature Specifications
For each major feature, create focused specifications:

#### Template: `specs/features/[feature_name].md`
```markdown
# [Feature Name] Specification

## Purpose
[What this feature does and why it's needed]

## Requirements
### Functional Requirements
- [Specific functionality the feature must provide]
- [User interactions and workflows]
- [Data processing requirements]

### Non-Functional Requirements
- [Performance requirements]
- [Security requirements]
- [Scalability considerations]

## Design
### Component Design
- [How this feature integrates with system architecture]
- [New components needed]
- [Modified existing components]

### Data Model
- [Data structures specific to this feature]
- [Database schema changes]
- [API contracts]

### User Interface
- [If applicable - UI/UX requirements]
- [API endpoints exposed]

## Implementation Plan
### Development Phases
1. **Phase 1**: [Foundation work]
2. **Phase 2**: [Core functionality]
3. **Phase 3**: [Integration and testing]

### Dependencies
- [Other features or components this depends on]
- [External services or APIs needed]
- [Infrastructure requirements]

### Testing Strategy
- [Unit testing approach]
- [Integration testing requirements]
- [Performance testing needs]

## Success Criteria
### Definition of Done
- [ ] [Specific completion criteria]
- [ ] [Performance benchmarks met]
- [ ] [Security requirements satisfied]

### Acceptance Tests
- [Specific tests that must pass]
- [User acceptance criteria]
- [Performance validation]

## Risk Assessment
### Technical Risks
- [Potential technical challenges]
- [Mitigation strategies]

### Dependencies Risks
- [External dependency risks]
- [Timeline risks]
```

### 7. Implementation Phase Planning
Create phase-specific implementation plans:

#### Template: `specs/implementation/phase_[X].md`
```markdown
# Phase [X] Implementation Plan

## Phase Overview
### Goals
[What this phase aims to achieve]

### Scope
[What's included and excluded from this phase]

### Timeline
[Estimated duration and milestones]

## Features in This Phase
### Primary Features
- **[Feature 1]**: [Implementation details and priority]
- **[Feature 2]**: [Implementation details and priority]

### Supporting Work
- [Infrastructure work needed]
- [Testing and quality assurance]
- [Documentation updates]

## Implementation Strategy
### Development Sequence
1. [First implementation step]
2. [Second implementation step]
3. [Integration and testing phase]

### Resource Requirements
- [Development resources needed]
- [Infrastructure requirements]
- [External dependencies]

## Quality Gates
### Exit Criteria
- [ ] [All features implemented and tested]
- [ ] [Performance benchmarks met]
- [ ] [Security review completed]
- [ ] [Documentation updated]

### Validation Process
- [How phase completion will be validated]
- [Testing requirements]
- [Review processes]

## Risk Management
### Phase-Specific Risks
- [Risks specific to this phase]
- [Mitigation strategies]

### Rollback Plan
- [How to rollback if phase fails]
- [Recovery procedures]
```

## Automated Documentation Updates

### 8. Create Documentation Sync Workflow
Add automated documentation updates to your implementation prompts:

#### Enhanced Implementation Prompt Pattern
```markdown
## Documentation Synchronization

### During Implementation
- [ ] Update relevant specification files with actual implementation details
- [ ] Update architecture decisions if design changes during implementation  
- [ ] Record any deviations from original spec with rationale
- [ ] Update API documentation with actual endpoints and contracts

### Post-Implementation
- [ ] Update project README with new capabilities
- [ ] Update deployment documentation if infrastructure changes
- [ ] Update team knowledge base with lessons learned
- [ ] Create or update troubleshooting guides

### Specification Maintenance
- [ ] Mark completed features in specification index
- [ ] Update dependency documentation
- [ ] Refresh performance benchmarks
- [ ] Update security considerations
```

## CONTEXT-FIRST WORKFLOW

**CodeFarmer:** Here's the **hallucination-proof workflow** for your quant trading project:

### **Step 1: Context Establishment**
```bash
/context-prime projects/quant_trading
# This gives me complete understanding of your actual tech stack, dependencies, and structure
```

### **Step 2: Break Down Large Specs**
```bash
/break-down-specs projects/quant_trading/planning_prp.md
# This modularizes your massive planning file into manageable components
```

### **Step 3: Targeted Implementation**
```bash
/plan-feature [specific-component-from-broken-down-specs]
/implement-spec specs/features/[specific-component].md
```

## WHY THIS SEQUENCE PREVENTS HALLUCINATIONS

1. **Context-first**: I understand your actual environment before making suggestions
2. **Modular focus**: Small, focused specifications prevent context overload
3. **Validation loops**: Each step validates against actual project state
4. **Incremental approach**: Build understanding progressively rather than assuming

**Would you like me to start with `/context-prime` on your quant trading project right now?** This will establish the foundation for everything else and show you exactly how this context-first approach works.

---

***Next: Execute context-prime to establish solid foundation, then break down that massive planning file into manageable, focused specifications...***