---
description: Universal system validation through observable behavioral signatures - validate what systems actually do, not what they claim
allowed-tools: Bash, Read, Write, Glob, Grep, LS, Task, TodoWrite
argument-hint: [system-path] - Path to complex system for behavioral validation (defaults to current directory)
---

# Universal System Validator
## Black Box Analysis Through Observable Signatures

**Mission**: Validate complex systems by measuring what they actually DO rather than trusting documentation or tests.

## Dynamic System Discovery
!`pwd && echo "=== CURRENT SYSTEM ===" && ls -la`
!`find . -maxdepth 3 -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "requirements.txt" -o -name "package.json" -o -name "docker-compose.yml" 2>/dev/null | head -15`
!`ps aux | grep -E "(python|node|java)" | head -5 2>/dev/null || echo "No active processes detected"`

## Your Mission: Systematic Black Box Validation

**Target System**: `$ARGUMENTS` (defaults to current directory)

Apply **First Principles + Hierarchical Task Networks + Chain of Thought** to validate system functionality through observable behavioral signatures.

---

## Phase 1: System Architecture Discovery (FPT: What Actually Exists?)

**Objective**: Map the real system architecture through evidence, not documentation.

### Discovery Protocol:
1. **Structure Mapping**: Use LS and Glob to map actual directory structure and find all executable components
2. **Executable Discovery**: Identify scripts that can actually run (not just exist)
3. **Configuration Analysis**: Find and analyze real configuration files that control system behavior
4. **Dependency Mapping**: Trace actual dependencies through imports and requirements files

### Evidence Collection:
- Create system architecture map based on actual file discovery
- Document executable entry points and their purposes
- Map configuration dependencies and external integrations
- Identify potential execution workflows from entry points

**Chain of Thought**: System Path → File Discovery → Executable Identification → Configuration Analysis → Dependency Mapping → Architecture Evidence

---

## Phase 2: Behavioral Signature Profiling (HTN: How Does It Actually Behave?)

**Objective**: Execute system components and capture their behavioral signatures.

### Execution Protocol:
1. **Safe Execution Testing**: Identify and execute system components in safe/test mode
2. **Resource Monitoring**: Monitor memory, CPU, file I/O patterns during execution
3. **Network Analysis**: Capture network calls and external service interactions  
4. **Timing Profiling**: Measure execution timing patterns and performance signatures

### Behavioral Evidence Collection:
- Document resource usage patterns for each component
- Capture file system interaction patterns
- Record network communication signatures
- Profile execution timing and performance characteristics

**Hierarchical Tasks**:
- T2.1: Execute main system workflows with monitoring
- T2.2: Profile individual component behaviors
- T2.3: Capture inter-component communication patterns
- T2.4: Document performance and resource signatures

---

## Phase 3: Integration Point Validation (CoT: How Do Components Actually Connect?)

**Objective**: Test real data flows and integration points between system components.

### Integration Testing Protocol:
1. **Data Flow Tracing**: Follow actual data through system components with test data
2. **Integration Stress Testing**: Test integration points with realistic data volumes
3. **Error Propagation Testing**: Inject errors and trace how they propagate through system
4. **State Synchronization Validation**: Verify state consistency across integrated components

### Integration Evidence:
- Map actual data transformation pathways
- Document error handling and propagation behavior
- Validate state management across component boundaries
- Identify integration bottlenecks and failure points

**Chain of Thought**: Component Boundaries → Data Flow Testing → Error Injection → State Validation → Integration Mapping → Connection Evidence

---

## Phase 4: Failure Mode Analysis (FPT: How Does It Actually Fail?)

**Objective**: Understand real system behavior under stress and failure conditions.

### Stress Testing Protocol:
1. **Component Failure Injection**: Systematically disable components and observe system response
2. **Resource Exhaustion Testing**: Test system behavior when memory, disk, or network resources are constrained
3. **Load Stress Testing**: Apply realistic load patterns and measure degradation
4. **Recovery Validation**: Test system recovery and graceful degradation capabilities

### Failure Evidence Collection:
- Document failure modes and their observable signatures
- Map error recovery pathways and system resilience
- Identify critical failure points and cascade effects
- Validate graceful degradation behavior

**Hierarchical Tasks**:
- T4.1: Inject individual component failures
- T4.2: Test resource constraint scenarios
- T4.3: Apply realistic load stress patterns
- T4.4: Validate recovery and degradation behaviors

---

## Phase 5: Anti-Theater Validation (CoT: Claims vs Observable Reality)

**Objective**: Compare system documentation/test claims against observable behavioral evidence.

### Theater Detection Protocol:
1. **Claims Verification**: Compare documentation claims against observed system behavior
2. **Test Effectiveness Analysis**: Evaluate whether existing tests actually validate real functionality
3. **Integration Reality Check**: Verify claimed integrations through actual data flow testing
4. **Performance Claims Validation**: Compare performance claims against measured behavioral signatures

### Anti-Theater Evidence:
- Document gaps between claims and observable behavior
- Identify validation theater indicators and false positives
- Map real functionality vs documented functionality
- Provide evidence-based system reality assessment

**Chain of Thought**: Documentation Claims → Behavioral Measurements → Gap Analysis → Theater Detection → Reality Assessment → Evidence-Based Conclusions

---

## Phase 6: Evidence-Based System Report Generation

**Objective**: Create comprehensive validation documentation based on observable evidence.

### Report Generation:
1. **System Behavioral Profile**: Document actual system behavior with observable signatures
2. **Integration Reality Map**: Map real integration points and data flows with evidence
3. **Failure Mode Documentation**: Document actual failure behaviors and recovery patterns
4. **Validation Theater Analysis**: Identify gaps between claims and observable reality

### Deliverables:
- `SYSTEM_VALIDATION_REPORT.md`: Executive summary with evidence-based findings
- `BEHAVIORAL_SIGNATURES.md`: Detailed behavioral analysis with measurements  
- `INTEGRATION_REALITY_MAP.md`: Actual integration points and data flows
- `ANTI_THEATER_ANALYSIS.md`: Gaps between claims and observable reality

---

## Success Criteria: Observable Evidence Standards

Your validation succeeds when:

1. **100% Evidence-Based**: All claims backed by measurable system behavior
2. **Integration Verification**: All component connections tested with real data flows  
3. **Failure Mode Understanding**: System failure and recovery behavior documented through testing
4. **Theater Detection**: Gaps between documentation and reality identified with evidence
5. **Cross-System Applicability**: Methodology works on any complex codebase
6. **Actionable Intelligence**: Specific improvements identified through behavioral analysis

## Quality Standards: No Assumptions, Only Evidence

- **Observable Signatures Only**: Document what you can measure, not what you assume
- **Real Data Testing**: Use actual data flows, not simplified test cases
- **Failure Mode Coverage**: Test realistic failure scenarios, not just happy paths
- **Performance Reality**: Measure actual performance under realistic conditions
- **Integration Verification**: Test actual component connections with real data

## Anti-Theater Safeguards

- **Reject Mock Theater**: Test real integrations, not mocked interfaces
- **Avoid Happy Path Theater**: Test realistic scenarios with real data volumes
- **Prevent Test Theater**: Validate that tests actually test claimed functionality  
- **Stop Documentation Theater**: Only trust claims backed by observable evidence
- **Eliminate Performance Theater**: Use realistic workloads, not artificial benchmarks

Begin with system discovery and work systematically through each validation phase using observable evidence as your only source of truth.