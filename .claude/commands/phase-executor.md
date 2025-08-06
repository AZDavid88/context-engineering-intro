---
description: Execute implementation phases from planning docs with full architecture integration
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, Bash, Task, TodoWrite
argument-hint: [phase-doc-path] - Path to phase implementation plan (e.g., phases/current/feature_plan.md)
---

# Phase Implementation Executor - Complete Build Workflow

## Dynamic Project Context Discovery
!`find "$(dirname "$ARGUMENTS")" -name "*.md" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo .`
!`pwd`
!`ls -la`

## Your Task - Complete Phase Implementation Execution
**Objective:** Transform detailed phase planning documentation into production-ready implementation following Plan → Spec → Build → Validate → Update methodology

**Critical Documentation Update Protocol:** All documentation updates occur AFTER successful build and validation to ensure consistency with actual implementation, preventing hallucination and version mismatches.

**Workflow Overview:**
1. **Plan Analysis**: Inventory existing architecture against phase requirements
2. **Specification Development**: Create detailed implementation specs based on gaps  
3. **Progressive Build**: Implement all components with systematic integration
4. **Comprehensive Validation**: Test functionality and system integration
5. **Documentation Update**: Update phase status and architectural documentation ONLY after validated implementation

## Phase 1: Plan Analysis & Architecture Inventory
**Systematic discovery of current project state and requirements mapping**

### 1.1 Phase Document Analysis
**Parse and understand the implementation requirements**

Read and analyze the specified phase implementation document: $ARGUMENTS
- Extract all planned features, components, and success criteria
- Identify implementation timeline and dependencies
- Map requirements to specific technical components
- Document exact specifications and acceptance criteria

Create comprehensive requirements breakdown:
- List all deliverables with technical specifications
- Identify integration points with existing system
- Map dependencies between components
- Extract success metrics and validation criteria

### 1.2 Comprehensive Architecture Inventory
**Use systematic discovery to understand existing codebase without making assumptions**

Project Structure Analysis:
- Use Glob `**/*.py`, `**/*.js`, `**/*.ts`, `**/*.go`, `**/*.rs`, `**/*.java` patterns to find all implementation files
- Use Read to analyze key architectural files (docker-compose.yml, package.json, requirements.txt, Cargo.toml, go.mod, pom.xml)
- Map existing module structure and dependencies
- Identify current architectural patterns and conventions

Research Documentation Discovery and Analysis:
- Check if `/research/` directory exists using Glob `research/**/*.md`, `research/**/*.txt`
- Read all research documentation to understand technology constraints and API patterns
- Extract implementation examples, integration approaches, and best practices
- Validate external dependencies and technology requirements against research docs

Living Documentation Analysis:
- Use Glob `**/verified_docs/**/*.md`, `**/docs/**/*.md` to find all architectural documentation
- Read all files in verified documentation directories for architectural patterns
- Understand existing interfaces, data flow, and module boundaries
- Identify integration points and architectural conventions
- Map current system capabilities and constraints

Existing Implementation Discovery:
- Use Grep to search for similar functionality across codebase:
  - Search for `class`, `function`, `def`, `interface`, `struct` patterns
  - Find existing implementations that new features should integrate with
  - Discover naming conventions and coding patterns
  - Identify existing test patterns and validation approaches
- Use Grep to find configuration patterns and environment setup approaches
- Search for existing deployment and build procedures

### 1.3 Gap Analysis & Requirements Mapping
**Compare phase requirements against existing architecture capabilities**

Systematic Gap Identification:
- Compare each phase requirement against discovered existing implementations
- Identify missing components, interfaces, and integration points
- Document architectural changes needed for implementation
- Map existing capabilities that can be leveraged vs. new components needed

Implementation Task Breakdown:
- Create detailed task list with specific technical requirements
- Prioritize tasks based on dependencies and integration complexity  
- Identify critical path components and potential blocking issues
- Plan incremental implementation strategy with validation checkpoints

**Evidence Validation**: All architectural assumptions verified against actual codebase analysis, research documentation, and living docs

## Phase 2: Detailed Specification Development  
**Create comprehensive implementation specifications based on verified gap analysis**

### 2.1 Technical Architecture Specification
**Design implementation approach using discovered patterns and verified research**

Interface and Integration Design:
- Design specific interfaces following existing architectural patterns from living docs
- Specify data flow and module interactions based on current system architecture
- Plan configuration management using existing environment setup patterns
- Define testing strategy following discovered test patterns and frameworks

Technology Integration Specification:
- Reference research documentation for correct API usage and integration patterns
- Specify external dependency usage following documented best practices
- Plan error handling and logging following existing system approaches
- Design monitoring and observability integration using current system patterns

### 2.2 Implementation Component Specifications
**For each identified gap, create detailed component specifications**

Component Interface Design:
- Design component interfaces following existing architectural patterns
- Specify implementation approach with technology accuracy using /research/ documentation
- Plan integration strategy with existing modules based on discovered patterns
- Define comprehensive success criteria and validation tests

Data Flow and State Management:
- Specify data models and state management following existing patterns
- Design persistence and caching strategies using current system approaches
- Plan transaction handling and data consistency using established patterns
- Define security and access control following current system practices

### 2.3 Progressive Implementation Strategy
**Plan systematic build approach with validation checkpoints**

Implementation Sequence Planning:
- Create ordered implementation sequence based on component dependencies
- Design incremental validation checkpoints after each component
- Plan integration testing strategy for each implementation phase
- Specify rollback strategies and error recovery procedures

Quality Assurance Integration:
- Plan code review checkpoints and quality gates
- Design automated testing integration with existing test suites
- Specify performance validation and monitoring integration
- Plan documentation update procedures post-implementation

**Evidence Validation**: All implementation approaches backed by research documentation, existing patterns, and architectural constraints

## Phase 3: Progressive Build Execution
**Implement all components following verified patterns with continuous validation**

### 3.1 Systematic Component Implementation
**Build components in dependency order with continuous integration validation**

Implementation with Pattern Consistency:
- Follow interface conventions from verified documentation analysis
- Use established architectural patterns discovered in living docs analysis
- Maintain consistency with module boundaries and naming conventions found in codebase
- Reference research documentation for technology-accurate implementations
- Follow existing error handling and logging patterns discovered in gap analysis

Progressive Integration and Validation:
- Implement components in the planned dependency order
- Test integration points immediately after each component implementation
- Validate against architectural patterns continuously during development
- Run existing test suites after each component to ensure no regression
- Maintain system stability throughout build process

Code Quality and Consistency:
- Follow existing code style and conventions discovered in codebase analysis
- Add comprehensive docstrings following project patterns found in existing code
- Implement error handling consistent with approaches found in system analysis
- Maintain modularity and separation of concerns per existing architecture

### 3.2 Configuration and Environment Integration
**Update system configuration following existing patterns**

Configuration Management:
- Update configuration files (docker-compose.yml, requirements.txt, package.json, etc.) following existing patterns
- Ensure environment compatibility with current setup discovered in architecture analysis
- Add necessary environment variables using existing configuration approaches
- Update deployment and build processes following current system procedures

Testing and Validation Integration:
- Integrate new components with existing test frameworks discovered in analysis
- Add component tests following existing test patterns and conventions
- Update CI/CD pipelines if applicable using current system approaches
- Validate configuration changes against existing deployment procedures

### 3.3 Integration Testing During Build
**Continuous validation throughout implementation process**

Component-Level Testing:
- Test each component against its specification immediately after implementation
- Validate all interfaces and integration points with existing system components
- Test error handling and edge cases using existing system testing approaches
- Verify external dependencies work as documented in research analysis

System Integration Validation:
- Test integration with existing system components after each implementation phase
- Validate no regression in existing functionality using discovered test suites
- Test complete feature workflows as components are integrated
- Monitor system performance and stability during implementation

**Evidence Validation**: Each component follows documented patterns, maintains architectural integrity, and passes integration testing

## Phase 4: Comprehensive Validation & System Integration Testing
**Validate complete implementation and ensure system integrity before documentation updates**

### 4.1 Complete Functional Testing
**Comprehensive testing of all implemented components and system integration**

End-to-End Feature Testing:
- Test complete feature functionality against original phase requirements
- Validate all user workflows and system interactions work correctly
- Test feature performance against specified success criteria
- Verify feature integrates correctly with existing system workflows

System Stability and Regression Testing:
- Run complete existing test suite to ensure no system regression
- Test system stability under various load and error conditions
- Validate existing functionality continues to work correctly
- Test system recovery and error handling with new components integrated

External Integration Testing:
- Test all external API integrations using research documentation patterns
- Validate external dependency compatibility and error handling
- Test authentication, authorization, and security integrations
- Verify monitoring, logging, and observability integration works correctly

### 4.2 Production Readiness Validation
**Ensure implementation is ready for production deployment**

Deployment and Configuration Testing:
- Test deployment procedures with new components and configurations
- Validate environment variable and configuration management works correctly
- Test scaling, monitoring, and operational procedures with new implementation
- Verify backup, recovery, and maintenance procedures include new components

Performance and Security Validation:
- Run performance tests to ensure implementation meets requirements
- Test security measures and validate no vulnerabilities introduced
- Test resource usage and system impact of new implementation
- Validate logging, monitoring, and debugging capabilities work correctly

### 4.3 Automated Test Suite Execution
**Run all project test suites to ensure system integrity**

Use Bash to execute comprehensive test validation:
- Run unit tests for all affected components and validate 100% pass rate
- Execute integration tests for all modified interfaces and system boundaries
- Run end-to-end tests for complete feature workflows and system integration
- Execute performance tests if applicable to implementation requirements
- Run security tests and validation if applicable to implementation

Test Results Analysis and Validation:
- Analyze all test results and ensure all success criteria are met
- Validate no test regression or system instability introduced
- Document any test updates or additions made during implementation
- Ensure test coverage adequately validates new implementation

**Evidence Validation**: Complete implementation passes all testing requirements and maintains system integrity

## Phase 5: Post-Validation Documentation Update & Phase Progression
**CRITICAL: Update all documentation ONLY after successful build and validation to ensure accuracy**

### 5.1 Implementation-Accurate Phase Documentation Update
**Update phase documents with actual implementation details and results**

Phase Status and Results Documentation:
- Update the original phase document ($ARGUMENTS) with complete implementation status
- Document actual implementation approach vs. originally planned approach
- Record all architectural decisions made during implementation with rationales
- Update timeline and dependencies based on actual implementation experience
- Document lessons learned and recommendations for future phases

Success Criteria and Metrics Documentation:
- Document achievement of all specified success criteria with evidence
- Record performance metrics and system impact measurements
- Document test results and validation evidence for all requirements
- Update risk assessments based on actual implementation outcomes

### 5.2 Living Documentation Synchronization
**Update architectural documentation to reflect actual implemented system**

Module Documentation Updates:
- Update affected module documentation in verified docs directories to reflect actual implementation
- Document new interfaces and integration patterns as actually implemented
- Update architectural diagrams and dependency mappings with real system state
- Ensure all interface specifications match actual implemented interfaces

System Architecture Documentation:
- Update system architecture documentation to reflect new capabilities and integrations
- Document actual data flow patterns and system interactions
- Update configuration and deployment documentation with actual procedures
- Ensure security and operational documentation reflects actual implementation

### 5.3 Research Documentation Enhancement with Implementation Examples
**Add real implementation examples to research documentation**

Implementation Pattern Documentation:
- Add actual implementation examples to relevant research directories
- Document technology integration patterns discovered and validated during implementation
- Update API usage examples with real working code from implementation
- Record performance and integration insights gained during actual development

Technology Best Practices Updates:
- Document best practices discovered during actual implementation
- Update research documentation with real-world integration challenges and solutions
- Add troubleshooting and operational insights from implementation experience
- Document recommended approaches based on actual implementation outcomes

### 5.4 Next Phase Preparation and System State Documentation
**Prepare accurate foundation for subsequent phase planning and implementation**

System Capability Documentation:
- Document new architectural capabilities available for future phases based on actual implementation
- Update system dependency and integration documentation with real implementation details
- Document configuration and operational changes that affect future development
- Update system performance and capacity documentation with actual measurements

Next Phase Prerequisites:
- Analyze impact on subsequent phases in the implementation plan based on actual outcomes
- Update dependencies and prerequisites for next phases with real system requirements
- Document any architectural changes that affect subsequent implementation planning
- Prepare comprehensive summary of completed capabilities for next phase planning

Implementation Handoff Documentation:
- Create comprehensive implementation summary for development team continuity
- Document all configuration changes and deployment updates made during implementation
- Update troubleshooting and operational guides with actual implementation details
- Document recommended maintenance and monitoring procedures for implemented features

### 5.5 Documentation Consistency Validation
**Ensure all documentation accurately reflects implemented system state**

Cross-Reference Validation:
- Validate all updated documentation is consistent with actual implementation
- Ensure no documentation contradicts actual implemented interfaces or behaviors
- Validate all code examples and configuration samples work with actual implementation
- Check all external references and links are current and accessible

Documentation Quality Assurance:
- Review all updated documentation for accuracy against actual implementation
- Ensure documentation provides complete information for system operation and maintenance
- Validate documentation enables successful reproduction of implementation procedures
- Ensure documentation is sufficient for team knowledge transfer and system continuity

**Evidence Validation**: All documentation accurately reflects actual implementation state and enables successful system operation and future development

## Output Requirements & Quality Standards
**Deliver production-ready implementation with implementation-accurate documentation**

### Implementation Deliverables:
- **Complete Feature Implementation**: All phase components fully functional and tested
- **System Integration**: Seamless integration with existing architecture validated through testing
- **Comprehensive Validation**: All success criteria met with documented test evidence
- **Accurate Documentation**: All documentation updated to reflect actual implementation state
- **Next Phase Readiness**: Clear progression path with accurate prerequisites and system state

### Quality Standards:
- **Architectural Consistency**: Maintains existing patterns and module boundaries discovered in analysis
- **Technology Accuracy**: All external integrations follow research documentation and work correctly
- **Production Readiness**: Complete implementation ready for deployment with validation evidence
- **Documentation Integrity**: All documentation accurately reflects actual implemented system state
- **Phase Progression**: Accurate status and realistic preparation for subsequent implementation phases

### Anti-Hallucination Safeguards:
- **Analysis-Driven Development**: All implementation based on actual codebase analysis, not assumptions
- **Research-Validated Integration**: All external integrations validated against research documentation
- **Post-Implementation Documentation**: All documentation updates occur after validated implementation
- **Evidence-Based Updates**: All documentation changes backed by actual implementation evidence
- **Consistency Validation**: All documentation cross-referenced against actual system state

**Focus**: Production-ready phase implementation that maintains system integrity, prevents documentation drift, and enables accurate progression through complex project development cycles with zero hallucination risk through post-implementation documentation updates.