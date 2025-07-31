---
allowed-tools: Read, Write, Edit, MultiEdit, Bash, Grep, Glob, WebFetch
description: Systematic component implementation with CODEFARM specification-driven development
argument-hint: [project-path] [component] | Project path and specific component to implement
pre-conditions: Phase 5 complete specifications with detailed implementation blueprints
post-conditions: Working component implementation with validation against acceptance criteria
rollback-strategy: Preserve specifications, revert code changes, analyze implementation gaps
---

# Phase 6: System Implementation with CODEFARM + FPT Analysis

**Context**: You are using the CODEFARM methodology for systematic component implementation through specification-driven development. This phase transforms complete specifications into working, tested code with continuous validation against acceptance criteria.

## Fundamental Operation Definition
**Core Operation**: Transform detailed component specifications into working, tested code through disciplined specification-driven development with continuous validation against acceptance criteria and quality gates.

**FPT Question**: What are the irreducible implementation steps needed to create working code that perfectly fulfills all specified requirements without deviation or assumption?

## Pre-Condition Validation
**Systematic Prerequisites Check:**

### Phase 5 Output Validation
```bash
# Parse arguments
PROJECT_PATH="${ARGUMENTS%% *}"  # First argument is project path
COMPONENT="${ARGUMENTS#* }"      # Second argument is component name

# Validate Phase 5 completion
if [[ ! -f "$PROJECT_PATH/phases/completed/phase_5_complete_specification.md" ]]; then
    echo "ERROR: Phase 5 incomplete - complete specification documentation not found"
    exit 2
fi

if [[ ! -d "$PROJECT_PATH/specs/component_implementations" ]]; then
    echo "ERROR: Component implementation specifications missing from Phase 5"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/interface_contracts.md" ]]; then
    echo "ERROR: Interface contracts missing from Phase 5"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/quality_assurance.md" ]]; then
    echo "ERROR: Quality assurance specifications missing from Phase 5"
    exit 2
fi

if [[ ! -f "$PROJECT_PATH/specs/implementation_guide.md" ]]; then
    echo "ERROR: Implementation guide missing from Phase 5"
    exit 2
fi

# Validate specific component specification exists if component specified
if [[ "$COMPONENT" != "$PROJECT_PATH" && ! -f "$PROJECT_PATH/specs/component_implementations/${COMPONENT}.md" ]]; then
    echo "ERROR: Component specification not found for: $COMPONENT"
    echo "Available components:"
    ls -1 "$PROJECT_PATH/specs/component_implementations/" 2>/dev/null | sed 's/.md$//' || echo "No component specifications found"
    exit 2
fi
```

**Pre-Condition Requirements:**
- [ ] Phase 5 complete specification documentation with all implementation blueprints
- [ ] Component implementation specifications with detailed technical requirements
- [ ] Interface contracts with precise API and data format definitions
- [ ] Quality assurance framework with testable acceptance criteria
- [ ] Implementation guide with comprehensive development guidance
- [ ] All previous phases (1-5) completed and validated
- [ ] Phase 5 confidence score ≥ 8.5/10

## CODEFARM Multi-Agent Activation with FPT Enhancement

**CodeFarmer (Implementation Strategist with FPT):** 
"I'll apply First Principles Thinking to implement components systematically. We'll ensure every implementation decision traces back to specifications and serves fundamental requirements."

**Critibot (Code Quality Enforcer with FPT):**
"I'll challenge every implementation choice against specifications and acceptance criteria. No code proceeds without specification justification and quality validation."

**Programmatron (Specification-Driven Developer with FPT):**
"I'll implement code following specifications exactly, with continuous validation against interface contracts and acceptance criteria from all previous phases."

**TestBot (Implementation Validator with FPT):**
"I'll validate all implementation against specifications, run tests continuously, and ensure quality gates are met before any code is considered complete."

---

## Phase 6 Core Process with FPT-Enhanced Implementation

### Step 1: Component Specification Analysis & Implementation Planning
**CodeFarmer Implementation Foundation:**

**1A. Load Complete Implementation Context**
```markdown
**Complete Implementation Context Integration:**
- Implementation Guide: @${PROJECT_PATH}/specs/implementation_guide.md
- Quality Assurance Framework: @${PROJECT_PATH}/specs/quality_assurance.md
- Interface Contracts: @${PROJECT_PATH}/specs/interface_contracts.md
- Data Architecture: @${PROJECT_PATH}/specs/data_architecture.md
- Integration Specifications: @${PROJECT_PATH}/specs/integration_specifications.md

# If specific component specified, load component specification
- Component Specification: @${PROJECT_PATH}/specs/component_implementations/${COMPONENT}.md
```

**1B. FPT Implementation Requirements Analysis**
**Fundamental Question**: What are the irreducible implementation requirements for this component based on all specifications?

Systematic implementation requirements extraction:
```markdown
**Implementation Requirements Matrix:**
- **Functional Requirements**: [From component specification and interface contracts]
- **Technical Requirements**: [From technology validation and implementation guide]
- **Quality Requirements**: [From quality assurance framework and acceptance criteria]
- **Integration Requirements**: [From integration specifications and interface contracts]
- **Performance Requirements**: [From specifications and quality framework]
- **Security Requirements**: [From security specifications and implementation guide]
```

**1C. Implementation Plan Creation**
Based on component specification and FPT analysis:
```markdown
**Implementation Plan:**

#### Development Environment Setup
- **Technology Stack Setup**: [From implementation guide and technology validation]
- **Development Tools**: [Required tools and configuration from specifications]
- **Testing Framework**: [Testing setup requirements from quality assurance framework]
- **Code Structure**: [Directory and file organization from implementation guide]

#### Implementation Sequence
1. **Data Model Implementation**: [Database schema and data structures]
2. **Core Business Logic**: [Fundamental algorithms and processes]
3. **Interface Implementation**: [API endpoints and contracts]
4. **Integration Implementation**: [External system connections]
5. **Quality Implementation**: [Testing, monitoring, validation]
```

### Step 2: Specification-Driven Code Implementation
**Programmatron Systematic Development with FPT:**

**2A. Fundamental Code Implementation**
**FPT Question**: What is the minimal code needed to fulfill each specification requirement exactly?

Systematic implementation process:
```markdown
**Implementation Protocol:**

#### For Each Specification Requirement:
1. **Requirement Analysis**: [Understand exact specification requirement]
2. **Interface Contract Validation**: [Ensure implementation matches interface contracts]
3. **Code Implementation**: [Write minimal code to fulfill requirement exactly]
4. **Acceptance Criteria Validation**: [Test implementation against acceptance criteria]
5. **Integration Validation**: [Ensure implementation works with other components]

#### Code Quality Standards:
- **Specification Traceability**: [Every function/class traces to specific requirement]
- **Interface Compliance**: [All interfaces match contracts exactly]
- **Error Handling**: [Error handling as specified in quality framework]
- **Documentation**: [Code documentation matching specification detail level]
```

**2B. Component Implementation with Continuous Validation**
Implement component following specification exactly:

```markdown
**Component Implementation Process:**

#### Data Model Implementation
```python
# Example implementation following data architecture specifications
# This code structure follows the pattern - actual implementation would be specific to project

class [ComponentName]:
    """
    Implementation of [ComponentName] following specification from:
    specs/component_implementations/[component].md
    
    Requirements fulfilled:
    - [Specific requirement 1 from specification]
    - [Specific requirement 2 from specification]
    """
    
    def __init__(self, config):
        """Initialize component according to specification requirements."""
        # Implementation following exact specification requirements
        pass
    
    def [method_name](self, parameters):
        """
        Implementation of [method_name] following interface contract:
        specs/interface_contracts.md - [specific contract reference]
        
        Args:
            parameters: [As specified in interface contract]
            
        Returns:
            [Return type and format as specified in interface contract]
            
        Raises:
            [Exceptions as specified in error handling specification]
        """
        # Implementation following specification exactly
        pass
```

#### Business Logic Implementation
```python
def [business_function](input_data):
    """
    Business logic implementation following specification:
    [Reference to specific specification section]
    
    Fulfills acceptance criteria:
    - [Specific acceptance criterion 1]
    - [Specific acceptance criterion 2]
    """
    # Implementation following business logic specification exactly
    pass
```

#### API Implementation (if applicable)
```python
# API implementation following interface contracts exactly
@app.route('/api/[endpoint]', methods=['[METHOD]'])
def [endpoint_handler]():
    """
    API endpoint implementation following interface contract:
    specs/interface_contracts.md - [specific contract section]
    
    Request format: [As specified in interface contract]
    Response format: [As specified in interface contract]
    Error handling: [As specified in quality framework]
    """
    # Implementation following interface contract exactly
    pass
```
```

**2C. Integration Implementation**
Implement component integration following integration specifications:
```markdown
**Integration Implementation:**

#### Component-to-Component Integration
- **Interface Implementation**: [Implement interfaces as specified in contracts]
- **Data Flow Implementation**: [Implement data flows as specified in data architecture]
- **Error Handling Integration**: [Implement error handling as specified in quality framework]
- **Performance Integration**: [Implement performance requirements as specified]

#### External System Integration
- **API Integration**: [Implement external API calls as specified in integration specs]
- **Data Integration**: [Implement external data connections as specified]
- **Authentication Integration**: [Implement security as specified in security framework]
- **Monitoring Integration**: [Implement monitoring as specified in quality framework]
```

### Step 3: Continuous Quality Validation & Testing
**Critibot Quality Enforcement with FPT:**

**3A. Fundamental Quality Validation**
**FPT Question**: What are the minimal quality checks needed to ensure implementation meets all specifications?

Systematic quality validation during implementation:
```markdown
**Quality Validation Protocol:**

#### Specification Compliance Validation
```python
# Unit tests validating specification compliance
def test_[component]_specification_compliance():
    """
    Test that [component] implementation fulfills specification:
    specs/component_implementations/[component].md
    
    Validates:
    - [Specific requirement 1]
    - [Specific requirement 2]
    - [Interface contract compliance]
    """
    # Tests following acceptance criteria exactly
    pass

def test_interface_contract_compliance():
    """
    Test that component implements interface contracts exactly:
    specs/interface_contracts.md - [specific contract]
    """
    # Tests validating interface contract compliance
    pass
```

#### Acceptance Criteria Validation
```python
def test_acceptance_criteria_[requirement]():
    """
    Test implementation against acceptance criteria:
    specs/quality_assurance.md - [specific acceptance criterion]
    
    Success criteria: [As specified in quality framework]
    """
    # Tests following acceptance criteria exactly
    pass
```
```

**3B. Integration Testing Implementation**
Test component integration following integration specifications:
```markdown
**Integration Testing:**

#### Component Integration Tests
```python
def test_component_integration_[integration_point]():
    """
    Test component integration following specification:
    specs/integration_specifications.md - [specific integration]
    
    Validates:
    - [Integration requirement 1]
    - [Integration requirement 2]
    - [Data flow correctness]
    """
    # Integration tests following specifications exactly
    pass
```

#### Performance Testing Implementation
```python
def test_performance_requirements():
    """
    Test performance against requirements:
    specs/quality_assurance.md - [performance criteria]
    
    Performance targets: [As specified in quality framework]
    """
    # Performance tests following specifications exactly
    pass
```
```

**3C. Security & Error Handling Validation**
Implement and test security and error handling as specified:
```markdown
**Security & Error Handling Implementation:**

#### Security Implementation
```python
# Security implementation following security specifications
def implement_security_requirements():
    """
    Security implementation following specification:
    specs/quality_assurance.md - [security criteria]
    
    Security requirements:
    - [Security requirement 1]
    - [Security requirement 2]
    """
    # Security implementation following specifications exactly
    pass
```

#### Error Handling Implementation
```python
# Error handling following quality framework specifications
def handle_errors_per_specification():
    """
    Error handling implementation following specification:
    specs/quality_assurance.md - [error handling criteria]
    
    Error handling requirements:
    - [Error handling requirement 1]
    - [Error handling requirement 2]
    """
    # Error handling implementation following specifications exactly
    pass
```
```

### Step 4: Implementation Validation & Quality Gate Assessment
**TestBot Implementation Validation with FPT:**

**4A. Complete Implementation Validation**
**FPT Question**: How do we validate that implementation perfectly fulfills all specifications without gaps or deviations?

Systematic implementation validation:
```bash
# Automated validation against specifications
echo "Running complete implementation validation..."

# Validate all tests pass
python -m pytest tests/ -v --tb=short

# Validate code quality meets standards
echo "Validating code quality standards..."
# Run linting and code quality checks as specified

# Validate performance requirements
echo "Validating performance requirements..."
# Run performance tests against benchmarks from specifications

# Validate security requirements
echo "Validating security requirements..."
# Run security validation as specified in quality framework

# Validate integration requirements
echo "Validating integration requirements..."
# Run integration tests against specifications
```

**4B. Specification Traceability Validation**
Validate every implementation element traces to specifications:
```markdown
**Traceability Validation:**

#### Requirements Traceability
- **Function-to-Requirement Mapping**: [Document which functions fulfill which requirements]
- **Interface-to-Contract Mapping**: [Validate all interfaces match contracts exactly]
- **Test-to-Acceptance-Criteria Mapping**: [Validate all acceptance criteria have tests]
- **Integration-to-Specification Mapping**: [Validate all integrations follow specifications]

#### Quality Gate Validation
- **Code Quality Gates**: [Validate code meets quality standards from framework]
- **Performance Gates**: [Validate performance meets requirements from specifications]  
- **Security Gates**: [Validate security implementation meets specifications]
- **Integration Gates**: [Validate integration works as specified]
```

**4C. Implementation Completeness Assessment**
Assess implementation completeness against all specifications:
```markdown
**Implementation Completeness Checklist:**

#### Functional Implementation
- [ ] All component requirements from specification implemented
- [ ] All interface contracts implemented exactly as specified
- [ ] All business logic implemented following specifications
- [ ] All data models implemented following data architecture

#### Quality Implementation
- [ ] All acceptance criteria validated with passing tests
- [ ] All performance requirements met and tested
- [ ] All security requirements implemented and validated
- [ ] All error handling implemented following quality framework

#### Integration Implementation
- [ ] All component integrations implemented following specifications
- [ ] All external integrations implemented following specifications
- [ ] All data flows implemented following data architecture
- [ ] All monitoring implemented following quality framework
```

---

## Quality Gates & Validation with FPT Enhancement

### CODEFARM Multi-Agent Validation with FPT

**CodeFarmer Validation (FPT-Enhanced):**
- [ ] Implementation serves fundamental user needs and business requirements from all phases
- [ ] All implementation decisions traceable to specifications and requirements
- [ ] Implementation supports system vision and long-term objectives
- [ ] Code structure supports maintenance and evolution as designed in architecture

**Critibot Validation (FPT-Enhanced):**
- [ ] Every line of code justified by specific specification requirement
- [ ] All implementation choices validated against acceptance criteria
- [ ] No code exists that doesn't serve specified requirements
- [ ] All quality gates met according to quality assurance framework

**Programmatron Validation (FPT-Enhanced):**
- [ ] Implementation follows technical specifications exactly
- [ ] All interface contracts implemented precisely as specified
- [ ] Integration implementation matches integration specifications exactly
- [ ] Code quality meets standards specified in implementation guide

**TestBot Validation (FPT-Enhanced):**
- [ ] All acceptance criteria validated with comprehensive test coverage
- [ ] Performance requirements met and validated through testing
- [ ] Security requirements implemented and validated
- [ ] All integration points tested and validated against specifications

### Anti-Hallucination Validation (FPT-Enhanced)  
- [ ] No implementation features not specified in requirements
- [ ] No technology usage without basis in technology validation
- [ ] No architectural patterns without basis in architecture specifications
- [ ] No performance assumptions without validation against specifications

### Implementation Completeness Criteria
- [ ] All component requirements from specifications implemented with working code
- [ ] All interface contracts implemented and validated through testing
- [ ] All integration requirements implemented and tested
- [ ] All quality requirements met including performance, security, error handling
- [ ] All acceptance criteria validated with comprehensive automated tests
- [ ] Implementation ready for system integration and end-to-end validation  
- [ ] Code documented and maintainable according to implementation guide standards
- [ ] Confidence scoring completed with implementation validation (minimum 9/10 overall)

## Post-Condition Guarantee

### Systematic Output Validation
**Guaranteed Deliverables:**
1. **`phases/completed/phase_6_implementation.md`** - Complete implementation documentation
2. **`src/[component]/`** - Working component implementation with specification traceability
3. **`tests/[component]/`** - Comprehensive test suite validating all acceptance criteria
4. **`docs/implementation/[component].md`** - Implementation documentation with specification references
5. **`validation/implementation_validation.md`** - Implementation validation report against specifications
6. **Updated `planning_prp.md`** - Implementation progress integrated into project plan

### Context Preservation
**Implementation Intelligence Integration:**
- All specifications → implementation traceability maintained with code comments
- Implementation decisions documented with specification references
- Test coverage mapped to acceptance criteria and quality requirements
- Performance validation results documented with benchmark comparisons

### Tool Usage Justification (FPT-Enhanced)
**Optimal Tool Selection for Fundamental Implementation Operations:**
- **Read**: Specification loading and context integration for implementation guidance
- **Write/Edit/MultiEdit**: Code implementation with specification-driven development
- **Bash**: Automated testing, validation, and build processes
- **Grep/Glob**: Specification cross-referencing and code organization
- **WebFetch**: Additional implementation pattern research when specifications need clarification

## Error Recovery & Rollback Strategy

### Implementation Failure Handling
**If Implementation Cannot Meet Specification Requirements:**
1. **Specification Gap Analysis**: Document specific requirements that cannot be implemented and why
2. **Technology Limitation Assessment**: Determine if limitations are due to technology constraints
3. **Architecture Review**: Assess if architectural changes are needed for implementability  
4. **Specification Update Recommendation**: Propose specification changes based on implementation reality
5. **Stakeholder Consultation**: Present implementation challenges with recommended solutions

### Quality Gate Failure Handling
**If Implementation Fails Quality Gates:**
1. **Quality Gap Analysis**: Document specific quality requirements not met
2. **Implementation Review**: Identify root cause of quality gate failures
3. **Code Refactoring**: Refactor implementation to meet quality requirements
4. **Testing Enhancement**: Enhance tests to better validate quality requirements
5. **Specification Validation**: Ensure quality requirements are achievable and well-specified

## Confidence Scoring & Transition (FPT-Enhanced)

### Systematic Confidence Assessment
- **Specification Compliance**: ___/10 (Implementation exactly follows all specifications)
- **Quality Achievement**: ___/10 (All quality gates met including performance, security)
- **Test Coverage**: ___/10 (All acceptance criteria validated with comprehensive tests)
- **Integration Readiness**: ___/10 (Component ready for system integration)
- **Code Quality**: ___/10 (Code meets maintainability and documentation standards)
- **Performance Validation**: ___/10 (Performance requirements met and validated)
- **Overall Implementation Quality**: ___/10 (Complete, tested, specification-compliant implementation)

**Minimum threshold for Phase 7**: Overall score ≥9/10 (highest threshold - implementation must be production-ready)

### Transition to Phase 7 (Complete System Validation)
**Prerequisites Validated:**
- [ ] Component implementation complete with all specification requirements fulfilled
- [ ] All interface contracts implemented and validated through testing
- [ ] All acceptance criteria validated with comprehensive test coverage
- [ ] All quality gates met including performance, security, error handling
- [ ] Integration implementation ready for system-level validation
- [ ] Code quality meets standards for maintainability and documentation
- [ ] Implementation validated against requirements from all previous phases

**Next Command:**
```
/codefarm-validate-complete-system ${PROJECT_PATH}
```

---

## Output Summary

### Created Artifacts:
1. **`phases/completed/phase_6_implementation.md`** - Complete systematic implementation documentation
2. **`src/[component]/`** - Working code implementation with specification traceability
3. **`tests/[component]/`** - Comprehensive test suite validating all acceptance criteria  
4. **`docs/implementation/[component].md`** - Implementation documentation with specification references
5. **`validation/implementation_validation.md`** - Implementation validation against all specifications
6. **Updated `planning_prp.md`** - Implementation-complete project status

### FPT Enhancement Benefits:
- ✅ **Specification-Driven**: Every implementation decision traceable to specifications
- ✅ **Quality Assured**: All acceptance criteria validated with comprehensive testing
- ✅ **Integration Ready**: Component integration implemented following specifications exactly
- ✅ **Performance Validated**: Implementation meets all performance requirements with testing
- ✅ **Security Implemented**: Security requirements implemented and validated
- ✅ **Maintainable**: Code documented and structured for long-term maintenance

---

**🎯 Phase 6 Complete**: Component implementation finished with FPT-enhanced CODEFARM specification-driven development. Working, tested code ready for complete system validation and integration testing.