---
description: WORKING VERSION - Systematic discovery of hidden issues and health assessment of legacy code
allowed-tools: Read, Write, Glob, Grep, LS, Edit, MultiEdit, Task
argument-hint: [project-path] - Path to legacy project for comprehensive health audit
---

# CODEFARM Legacy Code Health Audit & Unknown Issue Discovery (WORKING VERSION)

**Target Project:** $ARGUMENTS

## CODEFARM Multi-Agent Activation (Legacy Code Specialists)

**CodeFarmer (Legacy Code Analyst):** "I'll systematically analyze your legacy codebase at $ARGUMENTS to discover hidden issues, technical debt, and 'unknown unknowns' that could cause production failures."

**Critibot (Risk Assessment Specialist):** "I'll challenge every assumption about code health in $ARGUMENTS and identify potential failure modes you haven't considered."

**Programmatron (Code Health Architect):** "I'll analyze code structure, dependencies, and patterns in $ARGUMENTS to generate comprehensive health assessment with prioritized risk mitigation."

**TestBot (Validation & Testing Analyst):** "I'll evaluate testing coverage in $ARGUMENTS, identify untested code paths, and assess reliability under real usage conditions."

---

## Phase A: Comprehensive Legacy Code Discovery

### Step 1: Project Structure & Complexity Analysis

**CodeFarmer Legacy Code Investigation:**

Let me begin systematic analysis of the project structure at: $ARGUMENTS

**PROJECT PATH VALIDATION:**

Using LS tool to analyze current project directory structure at path: $ARGUMENTS

### Step 2: Python File Discovery & Analysis

**Programmatron Code Discovery:**

**Finding all Python files in the project:**

Using Glob tool to find Python files with path: $ARGUMENTS and pattern: "*.py"

**Finding configuration files:**

Using Glob tool to find requirements files with path: $ARGUMENTS and pattern: "requirements*.txt"
Using Glob tool to find package files with path: $ARGUMENTS and pattern: "package*.json"  
Using Glob tool to find Pipfile with path: $ARGUMENTS and pattern: "Pipfile*"
Using Glob tool to find pyproject.toml with path: $ARGUMENTS and pattern: "pyproject.toml"

### Step 3: Dependency & Import Analysis

**Programmatron Dependency Health Check:**

**Analyzing import patterns and dependencies:**

Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import" and output_mode: "files_with_matches"

**Searching for conditional imports and complex dependency patterns:**

Using Grep tool with path: $ARGUMENTS, pattern: "try.*import|except.*Import" and output_mode: "files_with_matches"

**Identifying external API usage and integration points:**

Using Grep tool with path: $ARGUMENTS, pattern: "requests|urllib|http|api" and output_mode: "files_with_matches"

**Dependency Risk Assessment:**
- **External Dependencies**: [Third-party library usage and versions]
- **Import Complexity**: [Conditional imports, circular dependencies]
- **API Integrations**: [External service dependencies and failure points]
- **Version Management**: [Dependency version constraints and conflicts]

### Step 4: Code Quality & Technical Debt Analysis

**Critibot Risk & Quality Assessment:**

**Searching for code smells and technical debt indicators:**

Using Grep tool with path: $ARGUMENTS, pattern: "TODO|FIXME|HACK|XXX" and output_mode: "files_with_matches"

**Looking for error handling patterns:**

Using Grep tool with path: $ARGUMENTS, pattern: "try|except|raise|assert" and output_mode: "content" and head_limit: 20

**Checking for hardcoded values and configuration issues:**

Using Grep tool with path: $ARGUMENTS, pattern: "localhost|127.0.0.1|password|secret|key.*=" and output_mode: "content" and head_limit: 10

**Analyzing logging and debugging patterns:**

Using Grep tool with path: $ARGUMENTS, pattern: "print|logging|debug" and output_mode: "files_with_matches"

**Quality Risk Matrix:**
- **Technical Debt**: [TODO/FIXME count and complexity]
- **Error Handling**: [Exception handling coverage and patterns]
- **Security Risks**: [Hardcoded credentials, unsafe patterns]
- **Debugging Infrastructure**: [Logging quality and debugging capabilities]

---

## Phase B: "Unknown Unknowns" Discovery

### Step 5: Hidden Risk Pattern Analysis

**TestBot Unknown Issue Discovery:**

**Testing Coverage Gap Analysis:**

**Finding test files in the project:**

Using Glob tool to find test files with path: $ARGUMENTS and pattern: "*test*.py"
Using Glob tool to find test files with path: $ARGUMENTS and pattern: "test_*.py"
Using Glob tool to find tests directory with path: $ARGUMENTS and pattern: "tests/*.py"

**Looking for complex functions that may need testing:**

Using Grep tool with path: $ARGUMENTS, pattern: "def " and output_mode: "content" and head_limit: 15

**Integration Risk Assessment:**

**Finding database and file system interactions:**

Using Grep tool with path: $ARGUMENTS, pattern: "open\\(|read\\(|write\\(|sql|database|db\\." and output_mode: "files_with_matches"

**Identifying network and external service calls:**

Using Grep tool with path: $ARGUMENTS, pattern: "request|socket|urllib|api|http" and output_mode: "files_with_matches"

**Configuration and Environment Risks:**

**Looking for environment variable dependencies:**

Using Grep tool with path: $ARGUMENTS, pattern: "os.environ|getenv|env\\[" and output_mode: "files_with_matches"

**Checking for file path and system dependencies:**

Using Grep tool with path: $ARGUMENTS, pattern: "os.path|sys.path|__file__" and output_mode: "files_with_matches"

### Step 6: File Size & Methodology Compliance Analysis

**CodeFarmer Architecture Health Assessment:**

**Analyzing individual file sizes to identify methodology violations:**

Using Grep tool with path: $ARGUMENTS, pattern: "." and output_mode: "files_with_matches" to get complete file list

For each Python file found, I'll check line count to identify files exceeding 500-line methodology limit using Read tool.

**Design Pattern Recognition:**
- **Architecture Patterns**: [MVC, microservices, monolith analysis]
- **Code Organization**: [Module coupling and cohesion assessment]
- **Interface Design**: [API consistency and design quality]
- **Data Flow**: [Information flow and state management patterns]

**Scalability Risk Analysis:**
- **Performance Bottlenecks**: [Potential performance issues identification]
- **Resource Usage**: [Memory, CPU, I/O intensive operations]
- **Concurrency Issues**: [Thread safety and async patterns]
- **Database Interactions**: [Query efficiency and connection management]

---

## Phase C: Comprehensive Health Report Generation

### Step 7: Risk Prioritization & File Size Analysis

**Programmatron Health Report Architecture:**

**Critical Risk Assessment (Priority 1 - Immediate Action Required):**

### CRITICAL RISKS - Production Failure Potential

**1. Methodology Violations**: [Files exceeding 500-line limit]
   - **Evidence**: [Specific files and line counts]
   - **Failure Scenario**: [Maintainability breakdown, bug introduction risk]
   - **Impact**: [Development velocity reduction, increased defect rate]
   - **Mitigation**: [Systematic refactoring using 7-phase methodology]
   - **Testing Strategy**: [Comprehensive testing before and after refactoring]

**High Risk Issues (Priority 2 - Address Soon):**
- **Security Vulnerabilities**: [Authentication, authorization, data exposure risks]
- **Integration Failures**: [External service dependencies and failure modes]
- **Performance Issues**: [Scalability bottlenecks and resource constraints]
- **Data Integrity**: [Database consistency and backup concerns]

**Medium Risk Issues (Priority 3 - Technical Debt):**
- **Code Quality**: [Maintainability and readability issues]
- **Documentation Gaps**: [Missing or outdated documentation]
- **Testing Coverage**: [Untested code paths and scenarios]
- **Configuration Management**: [Environment-specific issues]

**Low Risk Issues (Priority 4 - Future Improvements):**
- **Code Style**: [Formatting and convention inconsistencies]
- **Optimization Opportunities**: [Performance improvements]
- **Modernization**: [Technology stack updates and improvements]

### Step 8: Actionable Health Improvement Roadmap

**TestBot Implementation Roadmap:**

**Immediate Actions (Week 1):**
1. **Critical Risk Mitigation**: [Address methodology violations and security issues]
2. **Basic Testing Setup**: [Essential test coverage for critical paths]
3. **Security Hardening**: [Address immediate security vulnerabilities]

**Short-term Actions (Month 1):**
1. **Integration Testing**: [Validate external service interactions]
2. **Error Handling Enhancement**: [Improve exception handling and logging]
3. **Documentation Updates**: [Essential documentation for critical components]

**Medium-term Actions (Month 2-3):**
1. **Test Coverage Expansion**: [Comprehensive test suite development]
2. **Code Quality Improvements**: [Refactoring and technical debt reduction]
3. **Performance Optimization**: [Address identified bottlenecks]

**Long-term Actions (Month 4+):**
1. **Architecture Modernization**: [Systematic architecture improvements]
2. **Continuous Integration**: [Automated testing and deployment pipeline]
3. **Monitoring & Observability**: [Production monitoring and alerting setup]

---

## Phase D: Report Generation & Documentation

### Step 9: Generate Comprehensive Documentation

**Programmatron Report Generation:**

I will create comprehensive health audit documentation at: $ARGUMENTS

**1. Legacy Code Health Report**

Create health audit report file at: $ARGUMENTS/legacy_health_audit_report.md

**2. Risk Mitigation Checklist**

Create risk mitigation checklist at: $ARGUMENTS/risk_mitigation_checklist.md

**3. Testing Strategy Document**

Create testing strategy document at: $ARGUMENTS/legacy_testing_strategy.md

---

## Quality Gates & Validation

### CODEFARM Health Audit Validation

**CodeFarmer Strategic Validation:**
- [ ] All major code components analyzed for hidden risks and technical debt
- [ ] Architecture patterns identified with scalability and maintainability assessment
- [ ] Integration points mapped with failure mode analysis
- [ ] Actionable roadmap created with realistic timelines and priorities

**Critibot Risk Challenge Validation:**
- [ ] Every risk assessment backed by concrete code evidence
- [ ] "Unknown unknowns" systematically investigated through multiple analysis vectors
- [ ] Risk prioritization based on actual impact potential, not assumptions
- [ ] Mitigation strategies proven effective through similar codebase experience

**Programmatron Implementation Validation:**
- [ ] Health assessment technically accurate with specific code references
- [ ] Improvement recommendations implementable within resource constraints
- [ ] Risk mitigation strategies mapped to specific code changes
- [ ] Roadmap realistic with measurable success criteria

**TestBot Reality Validation:**
- [ ] All identified risks validated through code analysis evidence
- [ ] Testing recommendations address actual coverage gaps
- [ ] Performance assessments based on code complexity analysis
- [ ] Security assessments identify genuine vulnerability patterns

### Anti-Hallucination Validation
- [ ] No risk assessments without supporting code evidence
- [ ] No architectural assumptions without structural analysis
- [ ] No performance claims without complexity assessment
- [ ] No security warnings without vulnerability evidence

---

## Confidence Scoring & Success Metrics

### Systematic Confidence Assessment
- **Code Coverage Analysis**: [Percentage of codebase systematically analyzed]
- **Risk Identification Accuracy**: [Confidence in identified risks and evidence quality]
- **Mitigation Strategy Validity**: [Confidence in recommended solutions]
- **Implementation Feasibility**: [Realistic assessment of remediation effort]
- **Unknown Unknown Discovery**: [Confidence in discovering hidden issues]
- **Overall Audit Quality**: [Comprehensive legacy code health assessment]

**Minimum threshold for actionable recommendations**: Overall score ≥ 8/10

### Success Criteria

**Health Audit Effectiveness:**
- **Risk Discovery**: [Number and severity of risks identified through systematic analysis]
- **Unknown Unknown Resolution**: [Hidden issues discovered that manual review missed]
- **Actionable Intelligence**: [Percentage of recommendations that are immediately implementable]
- **Risk Prevention**: [Potential production failures avoided through proactive identification]

### Next Steps Framework

**After completing systematic analysis:**
1. Begin critical risk mitigation based on prioritized roadmap
2. Implement essential testing for highest-risk code paths
3. Monitor improvements and validate risk reduction
4. Schedule follow-up health assessment to measure progress

---

**🎯 CODEFARM Legacy Code Health Audit Complete**: Comprehensive systematic analysis of legacy codebase health with evidence-based risk assessment, prioritized mitigation roadmap, and actionable improvement strategy ready for implementation.