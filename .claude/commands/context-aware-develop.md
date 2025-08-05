---
description: Context-aware feature development with research, validation, and auto-documentation
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, Bash
argument-hint: [feature-description] - Describe what you want to build
---

# Context-Aware Feature Development

## Your Task
Develop features systematically using verified documentation and research to prevent hallucination, with automatic validation and documentation updates.

## Approach
**Apply systematic development naturally:**
- Break feature requirements into logical implementation components
- Question assumptions by validating against existing patterns and research
- Build evidence progressively through systematic verification at each step

## Step-by-Step Process

### Step 1: Context Discovery and Architecture Mapping
**Understand the feature within existing system architecture**

Use Glob to identify relevant modules and existing implementations:
```
**/*.py pattern matching to find related functionality
```

Use Read to analyze living documentation for architectural patterns:
- Check `/verified_docs/by_module_simplified/` for interface patterns
- Identify module boundaries and data flow requirements
- Map dependencies and integration points

Use Grep to discover existing implementation patterns:
- Search for similar functionality across codebase
- Identify naming conventions and architectural approaches
- Validate against documented interfaces

**Evidence Validation**: Ensure all architectural assumptions verified against living docs

### Step 2: Research-Driven Implementation Design
**Design implementation using technology-accurate patterns**

Use Read to reference appropriate research documentation:
- Identify technology domains needed (e.g., `/research/hyperliquid_documentation/` for API calls)
- Extract implementation patterns from research summaries
- Validate API usage against official documentation patterns

Map feature requirements to research-backed implementation:
- Break down feature into technology-specific components
- Design interfaces following documented patterns
- Plan validation strategy for each component

Create implementation specification:
- Document planned interfaces before coding
- Specify integration points with existing modules
- Define success criteria and validation tests

**Evidence Validation**: All implementation approaches backed by research documentation

### Step 3: Progressive Implementation with Pattern Validation
**Build following verified patterns with continuous validation**

For each implementation component:

Use existing patterns from living docs:
- Follow interface conventions from verified documentation
- Implement using established architectural patterns
- Maintain consistency with module boundaries

Implement with technology accuracy:
- Reference research docs for correct API usage
- Follow documented integration patterns
- Validate external dependency usage against research

Progressive validation during implementation:
- Test interfaces against documented patterns
- Validate integration points incrementally
- Verify no architectural drift from living docs

**Evidence Validation**: Each component follows documented patterns and research accuracy

### Step 4: Integrated Testing and Architectural Validation
**Validate complete feature integration and system integrity**

Functional testing:
- Test feature functionality against specification
- Validate integration with existing modules
- Verify external dependencies work as documented

Use Bash to run relevant test suites:
- Execute module-specific tests for affected areas
- Run integration tests for modified interfaces
- Validate no regression in existing functionality

Architectural integrity validation:
- Compare implementation against living documentation patterns
- Verify module boundaries maintained
- Confirm no unintended coupling introduced

**Evidence Validation**: Complete feature works and maintains system architectural integrity

### Step 5: Living Documentation Update and Drift Prevention
**Update documentation to reflect changes and prevent future drift**

Analyze implementation impact:
- Identify new interfaces or modified patterns
- Document integration points and dependencies
- Map changes to existing architectural documentation

Update relevant living documentation:
- Modify affected module documentation for new interfaces
- Update dependency mappings for integration changes
- Document new patterns for future reference

Validation against architectural drift:
- Compare updated system against original architectural principles
- Verify documentation accuracy through code verification
- Ensure living docs reflect actual implementation

**Evidence Validation**: Documentation accurately reflects implementation and no architectural drift occurred

## Output Requirements
Create comprehensive implementation with:
- **Working Feature**: Fully functional implementation following documented patterns
- **Integration Validation**: Confirmed compatibility with existing system architecture
- **Updated Documentation**: Living docs reflecting actual implementation state
- **Drift Prevention**: Architectural integrity maintained and documented

## Quality Standards
- High confidence through systematic verification against living docs and research
- Evidence-based implementation only - no assumptions without research validation
- Clear integration with existing architectural patterns
- Natural complexity appropriate for feature scope and system requirements
- Progressive validation documented throughout implementation

**Focus**: Production-ready feature development that maintains system integrity through systematic context preservation and validation.