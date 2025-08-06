---
description: Discovery-first implementation planning with intelligent project analysis
allowed-tools: Read, Write, Edit, Glob, Grep, LS, Bash, mcp__filesystem__read_multiple_files, mcp__filesystem__directory_tree
argument-hint: [implementation-description or project-path] - Describe what to plan or specify project directory
---

# Context-Aware Implementation Planning

## Project Context Discovery

**Current Working Directory:**
!`pwd`

**Smart Project Root Discovery:**
!`PROJECT_PATH=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); if [ -n "$PROJECT_PATH" ] && [ -d "$PROJECT_PATH" ]; then echo "Specified Project: $PROJECT_PATH"; else echo "Project Root: $(pwd)"; fi`

**Target Directory Analysis:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then echo "Analyzing: $TARGET_DIR"; find "$TARGET_DIR" -maxdepth 3 -type d -name "phases" -o -name "research" -o -name "verified_docs" -o -name "docs" -o -name "planning" -o -name "src" -o -name "scripts" 2>/dev/null | head -15; else find "$(pwd)" -maxdepth 3 -type d -name "phases" -o -name "research" -o -name "verified_docs" -o -name "docs" -o -name "planning" 2>/dev/null | head -10; fi`

**Existing Planning Documents:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*plan*.md" -o -name "*planning*.md" -o -name "PLANNING.md" -o -name "planning_prp.md" 2>/dev/null | head -8`

**Research & Documentation Inventory:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -type d -name "research" -exec ls -la {} \; 2>/dev/null | head -20`

**System Architecture Evidence:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" | grep -E "(config|setup|requirements|architecture|system)" | head -10`

**Implementation Context (Scripts & Code):**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*.py" | head -15`

**Methodology & Specifications:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*METHODOLOGY*.md" -o -name "*methodology*.md" -o -path "*/.claude/*" -name "*.md" 2>/dev/null | head -8`

## Your Task - Discovery-First Planning

**Primary Objective:** Create comprehensive phase-based implementation plans through intelligent discovery and analysis.

**Request Analysis:** **$ARGUMENTS**

### Discovery-First Approach:

**If the request includes a specific directory path:**
1. **Analyze the specified project directory comprehensively**
2. **Conduct intelligent discovery based on existing project state and architecture**
3. **Ask targeted questions about implementation goals, integration points, and constraints**
4. **Synthesize findings into evidence-based implementation plans**

**If the request is general or lacks specific technical details:**
1. **Assess project context from discovered documentation**
2. **Ask clarifying questions to understand scope, priorities, and constraints**
3. **Gather technical requirements and success criteria**
4. **Create comprehensive plans based on discovered context**

### Discovery Questions Framework:

**Technical Scope & Integration:**
- What specific components or systems need modification or enhancement?
- How does this integrate with existing architecture and data flows?
- What are the performance targets and validation criteria?
- Are there existing implementation patterns or constraints to follow?

**Implementation Priorities:**
- What are the critical success factors and measurable outcomes?
- What is the risk tolerance and deployment approach?
- What are the timeline constraints and resource considerations?
- How will success be measured and validated?

**System Context:**
- What existing research, documentation, or prior work should inform this?
- What are the integration points with databases, APIs, or external systems?
- What safety protocols and rollback procedures are required?
- How does this align with broader system architecture and goals?

## Approach

**Apply systematic planning naturally:**
- **Hierarchical Task Networks**: Break complex implementations into logical phases with clear dependencies and measurable outcomes
- **First Principles Planning**: Question assumptions about system architecture and validate each phase against existing documentation patterns
- **Evidence-Based Progression**: Build validation criteria that prevent progression without demonstrable success and system safety

## Step-by-Step Process

### Step 1: Implementation Architecture Analysis

Use existing documentation to understand system context and integration points:

**System Architecture Mapping:**
- Analyze current system architecture from verified_docs to understand affected components
- Map dependencies and integration requirements using existing patterns
- Identify potential failure points and system impact areas
- Review research documentation for technical implementation requirements

**Phase Structure Design:**
- Break implementation into logical phases with minimal inter-dependencies
- Define clear success criteria for each phase with quantitative metrics
- Design validation checkpoints that prevent unsafe progression
- Create rollback procedures for each integration point

### Step 2: Phase Definition with Measurable Criteria

For each implementation phase:

**Technical Deliverables:**
- Define specific, testable technical outcomes for phase completion
- Create quantitative success metrics (performance, accuracy, completeness percentages)
- Specify integration points and their validation requirements
- Document expected system behavior changes

**Safety Validation Framework:**
- Design comprehensive testing procedures for each phase
- Create automated validation scripts where possible
- Define rollback triggers and procedures
- Specify system monitoring and alerting requirements

**Decision Points:**
- Create clear go/no-go criteria for phase progression
- Define minimum success thresholds based on evidence
- Specify stakeholder review and approval processes
- Document escalation procedures for validation failures

### Step 3: Risk Assessment and Mitigation

**Systematic Risk Analysis:**
- Identify potential failure points using first principles analysis
- Assess system impact and recovery requirements for each risk
- Design mitigation strategies with measurable effectiveness
- Create contingency plans for high-impact scenarios

**Progressive Integration Strategy:**
- Design incremental integration approaches to minimize system risk
- Create isolated testing environments for validation
- Plan parallel system operation during transition periods
- Specify monitoring and alerting during integration phases

### Step 4: Resource and Infrastructure Planning

**Technical Infrastructure Requirements:**
- Analyze computational, storage, and network requirements
- Plan cloud services, databases, and architectural components based on research documentation
- Design cost-effective implementation approaches with scaling considerations
- Create deployment automation and monitoring strategies

**Timeline and Resource Allocation:**
- Estimate implementation timelines based on complexity analysis
- Identify critical path dependencies and resource bottlenecks
- Plan human resource allocation and expertise requirements
- Create milestone tracking and progress measurement systems

## Output Requirements

Create comprehensive implementation plan with:

**Phase Structure Document:**
- Clear phase definitions with dependencies and measurable success criteria
- Quantitative validation requirements for each phase transition
- Comprehensive safety protocols and rollback procedures
- Resource allocation and timeline estimates

**Technical Specifications:**
- Infrastructure requirements based on research documentation accuracy
- Integration patterns following existing architectural principles
- Performance and scalability requirements with measurement criteria
- Cost analysis and optimization strategies

**Decision Framework:**
- Measurable criteria for phase progression with specific thresholds
- Risk assessment matrix with mitigation strategies
- Stakeholder review and approval processes
- Escalation procedures for validation failures

**Implementation Guidance:**
- Step-by-step technical implementation instructions
- Testing and validation procedures for each phase
- Monitoring and alerting setup requirements
- Post-implementation performance measurement systems

## Quality Standards

- **Measurable Success Criteria**: Every phase must have quantitative validation thresholds based on evidence
- **Comprehensive Safety Protocols**: Rollback procedures and system protection preventing implementation failures
- **Evidence-Based Technical Recommendations**: All infrastructure and architecture decisions backed by research documentation
- **Clear Implementation Timeline**: Realistic milestones with dependency tracking and resource allocation
- **System Integrity Preservation**: Validation that implementation maintains existing system functionality and performance

## Documentation Placement Strategy

**Dynamic Planning Document Location:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; if [ -d "$SEARCH_DIR/phases/current" ]; then echo "Primary: $SEARCH_DIR/phases/current/[implementation_name]_plan.md"; elif [ -d "$SEARCH_DIR/planning" ]; then echo "Primary: $SEARCH_DIR/planning/[implementation_name]_plan.md"; else echo "Primary: $SEARCH_DIR/[implementation_name]_plan.md"; fi`

**Supporting Documentation Structure:**
- Technical specifications in relevant module directories (if verified_docs/ exists)
- Validation scripts in project testing framework (if tests/ exists)  
- Progress tracking with lifecycle management (phases/ or planning/ structure)
- Research references maintained in research/ directory (if exists)

**Integration with Existing Documentation:**
- Link to relevant architectural documentation patterns
- Reference appropriate research documentation for technical accuracy
- Update master planning documents if they exist
- Maintain consistency with discovered project organization patterns

Focus on creating production-ready implementation plans that maintain system integrity through systematic validation and evidence-based progression criteria.