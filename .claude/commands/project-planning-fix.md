---
description: Complete project planning workflow - discovery, synthesis, planning, and documentation
allowed-tools: Read, Write, Edit, Glob, Grep, LS, Bash, mcp__filesystem__read_multiple_files, mcp__filesystem__directory_tree, mcp__filesystem__create_directory
argument-hint: [implementation-description or project-path] - Describe what to plan or specify project directory
---

# Complete Project Implementation Planning

## Project Context Discovery

**Current Working Directory:**
!`pwd`

**Target Project Path:**
!`PROJECT_PATH=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); if [ -n "$PROJECT_PATH" ] && [ -d "$PROJECT_PATH" ]; then echo "Specified Project: $PROJECT_PATH"; else echo "Project Root: $(pwd)"; fi`

**Project Structure Analysis:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then echo "Analyzing: $TARGET_DIR"; find "$TARGET_DIR" -maxdepth 3 -type d -name "phases" -o -name "research" -o -name "verified_docs" -o -name "docs" -o -name "planning" -o -name "src" -o -name "scripts" -o -name "tests" 2>/dev/null | head -15; else find "$(pwd)" -maxdepth 3 -type d -name "phases" -o -name "research" -o -name "verified_docs" -o -name "docs" -o -name "planning" -o -name "src" -o -name "scripts" -o -name "tests" 2>/dev/null | head -10; fi`

**Existing Planning Documents:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*plan*.md" -o -name "*planning*.md" -o -name "PLANNING.md" -o -name "planning_prp.md" 2>/dev/null | head -8`

**Research & Documentation Inventory:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -type d -name "research" -exec ls -la {} \; 2>/dev/null | head -20`

**Code Architecture Overview:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.java" -o -name "*.cpp" -o -name "*.rs" | head -15`

**Configuration & Infrastructure:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; find "$SEARCH_DIR" -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" -o -name "requirements.txt" -o -name "package.json" -o -name "Cargo.toml" -o -name "pom.xml" | head -10`

## Your Task - Complete Implementation Planning

**Objective:** Create comprehensive implementation plans through systematic project analysis, requirements synthesis, and detailed planning.

**Request:** **$ARGUMENTS**

## Phase 1: Discovery & Requirements Analysis

Based on the project structure analysis above, I can see your project's architecture and existing documentation. Now I need to understand your specific implementation requirements.

### **Discovery Questions Framework:**

**Technical Scope & Integration:**
1. **What specific feature or enhancement** do you want to implement?
   - New functionality, optimization, integration, or infrastructure improvement?
   - Which existing components will be affected or enhanced?

2. **Integration Requirements:**
   - How should this integrate with your existing system architecture?
   - What external APIs, databases, or services need to be involved?
   - Are there existing patterns or conventions I should follow?

**Implementation Priorities:**
3. **Success Criteria:**
   - How will you measure success? (Performance metrics, functionality, user experience?)
   - What are the critical must-have features vs. nice-to-have enhancements?
   - Are there specific constraints (timeline, resources, compatibility)?

4. **Risk & Quality Requirements:**
   - What level of testing and validation is required?
   - Are there security, performance, or reliability concerns?
   - What deployment and rollback procedures should be planned?

**System Context:**
5. **Project-Specific Considerations:**
   - Should I leverage any existing research or documentation patterns from your project?
   - Are there specific architectural principles or design patterns to maintain?
   - What monitoring, logging, or observability requirements exist?

**Please provide your responses to these questions, and I'll proceed to synthesize your requirements into a comprehensive implementation plan.**

---

## Phase 2: Requirements Synthesis & Specification

*[This section will be populated after receiving your responses]*

**Based on your responses, I will:**
1. **Consolidate Requirements**: Transform your answers into coherent technical specifications
2. **Validate Scope**: Ensure alignment with your project architecture and constraints  
3. **Define Success Metrics**: Establish measurable criteria for implementation success
4. **Identify Dependencies**: Map integration points and prerequisite components

---

## Phase 3: Implementation Planning & Architecture

*[This section will be populated during synthesis]*

**Implementation Strategy:**
- **Phase Breakdown**: Logical development phases with clear milestones
- **Technical Architecture**: Detailed design patterns and integration approaches
- **Risk Assessment**: Potential issues and mitigation strategies
- **Resource Planning**: Timeline estimates and infrastructure requirements

---

## Phase 4: Documentation Generation & Placement

*[Final phase - comprehensive documentation will be generated]*

**Documentation Output Strategy:**
!`TARGET_DIR=$(echo '$ARGUMENTS' | grep -o '/[^" ]*' | grep -v '^/$' | head -1); SEARCH_DIR="${TARGET_DIR:-$(pwd)}"; if [ -d "$SEARCH_DIR/phases/current" ]; then echo "📁 Implementation Plan: $SEARCH_DIR/phases/current/[feature_name]_implementation_plan.md"; elif [ -d "$SEARCH_DIR/planning" ]; then echo "📁 Implementation Plan: $SEARCH_DIR/planning/[feature_name]_implementation_plan.md"; elif [ -d "$SEARCH_DIR/docs" ]; then echo "📁 Implementation Plan: $SEARCH_DIR/docs/[feature_name]_implementation_plan.md"; else echo "📁 Implementation Plan: $SEARCH_DIR/[feature_name]_implementation_plan.md"; fi`

**Complete Documentation Package Will Include:**
- **Main Implementation Plan**: Comprehensive phase-by-phase implementation guide
- **Technical Specifications**: Architecture diagrams, API designs, data models
- **Testing Strategy**: Unit, integration, and system testing approaches  
- **Deployment Guide**: Step-by-step deployment and validation procedures
- **Progress Tracking**: Milestone checkpoints and success validation criteria

---

## Workflow Instructions

**This command operates in a complete workflow:**

1. **Initial Execution**: Provides project discovery and asks targeted questions
2. **Requirements Input**: You provide responses to the discovery questions
3. **Automatic Synthesis**: I consolidate your requirements into technical specifications
4. **Implementation Planning**: I create detailed phase-by-phase implementation plans
5. **Documentation Generation**: I create comprehensive implementation documentation in your project's appropriate directory structure

**Cross-Project Compatibility:**
- **Dynamic Path Detection**: Works with any project directory structure
- **Adaptive Documentation Placement**: Uses your project's existing organization patterns
- **Technology Agnostic**: Supports Python, JavaScript, Java, Rust, C++, and other languages
- **Flexible Architecture**: Adapts to different project sizes and complexity levels

**Quality Standards:**
- **Evidence-Based Planning**: All recommendations based on actual project analysis
- **Measurable Success Criteria**: Quantitative validation thresholds for each phase
- **Risk-Aware Design**: Comprehensive safety protocols and rollback procedures
- **Production-Ready Output**: Implementation plans ready for immediate development use

---

## Next Steps

**Immediate Action Required:**
Please answer the discovery questions above so I can proceed with:
- Requirements synthesis and technical specification
- Detailed implementation planning with phase breakdowns
- Comprehensive documentation generation in your project structure

**Expected Outcome:**
A complete implementation plan document placed in your project's planning directory, ready to guide your development process from conception to deployment.