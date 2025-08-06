---
description: Interactive project planning with critical issue detection and mandatory resolution
allowed-tools: Bash(find:*), Bash(grep:*), Bash(rg:*), Bash(git:*), Read, Glob, Task, WebFetch
argument-hint: [focus_area] - Project area to analyze and plan (e.g., "api integration", "data pipeline", "authentication system")
---

# Interactive Project Planning with Critical Validation

## Your Task - Complete Interactive Planning Workflow

**Objective:** Generate a comprehensive, validated project implementation plan through systematic critical issue detection and mandatory interactive resolution.

**Critical Success Factors:**
- Detect system-breaking issues BEFORE implementation planning
- Force resolution of critical problems through user interaction
- Validate architecture integration at multiple levels
- Provide multiple solution alternatives with clear trade-offs
- Generate bulletproof implementation plans only after full validation

## Phase 1: Dynamic Project Discovery & Context Analysis

### Project Structure Discovery
!`find $CLAUDE_PROJECT_DIR -type f -name "*.md" -o -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.go" -o -name "*.rs" -o -name "*.java" | head -20`

### Research Documentation Analysis
!`find $CLAUDE_PROJECT_DIR -path "*/research/*" -name "*.md" -o -path "*/docs/*" -name "*.md" 2>/dev/null | head -10`

### Existing Architecture Pattern Detection
!`find $CLAUDE_PROJECT_DIR -name "*.py" -exec grep -l "class.*\(Engine\|Manager\|Service\|Handler\|Controller\|Processor\)" {} \; 2>/dev/null | head -8`

### Technology Stack & Dependencies
!`find $CLAUDE_PROJECT_DIR -name "requirements.txt" -o -name "package.json" -o -name "Cargo.toml" -o -name "go.mod" -o -name "pom.xml" 2>/dev/null`

### Current System Timeframes & Data Patterns
!`rg -i "timeframe|interval|frequency|schedule|rate|update.*time" $CLAUDE_PROJECT_DIR --type md -A 1 -B 1 2>/dev/null | head -20`

**Based on the above discovery, analyze:**

1. **Project Technology Stack**: Identify primary languages, frameworks, and architecture patterns
2. **Existing Data Patterns**: Understand current data processing frequencies, timeframes, and update patterns
3. **External Dependencies**: Catalog external APIs, services, and data sources currently in use
4. **Architecture Philosophy**: Determine the project's architectural approach (microservices, monolithic, event-driven, etc.)
5. **Focus Area Context**: Understand how the requested focus area fits within the existing system

**Your Task for Focus Area: "$ARGUMENTS"**

Analyze the requested focus area within the context of the discovered project architecture and prepare for critical issue detection.

## Phase 2: CRITICAL ISSUE DETECTION & ANALYSIS

### External API & Service Validation
!`rg -i "api|endpoint|service.*url|base.*url|client" $CLAUDE_PROJECT_DIR/research/ -A 2 -B 2 2>/dev/null || echo "No research documentation found for external services"`

### Data Frequency & Timeframe Analysis
!`rg -i "frequency|interval|update.*hour|daily|minute|realtime|static|dynamic" $CLAUDE_PROJECT_DIR -A 3 -B 1 2>/dev/null | head -15`

### Architecture Integration Points
!`rg -i "integration|pipeline|workflow|process" $CLAUDE_PROJECT_DIR --type py --type js --type ts -A 2 2>/dev/null | head -10`

### Dependency Chain Analysis
!`rg -i "import|require|use.*external|third.*party" $CLAUDE_PROJECT_DIR -A 1 2>/dev/null | head -10`

**SYSTEMATIC CRITICAL ISSUE DETECTION:**

Analyze the focus area implementation for these **high-risk failure patterns**:

### 1. External API Reality Validation
**Check for these critical mismatches:**
- **Data Update Frequency Assumptions**: Does the implementation assume external data updates at frequencies different from reality?
- **API Behavior Assumptions**: Are there assumptions about external API behavior that may not match reality?
- **Rate Limiting & Constraints**: Are there external service constraints not accounted for?
- **Data Format & Structure**: Are there assumptions about external data formats that need validation?

### 2. Timeframe & Data Alignment Issues
**Critical timeframe analysis:**
- **Multi-Timeframe Systems**: If the system processes data at multiple timeframes, do all external data sources align?
- **Static vs Dynamic Data**: Are there static external data sources being treated as dynamic or vice versa?
- **Update Cycle Mismatches**: Do external data update cycles align with system processing expectations?
- **Historical Data Assumptions**: Are there assumptions about historical data availability or format?

### 3. Architecture Integration Risks
**System integration analysis:**
- **Breaking Changes**: Would the implementation break existing system components?
- **Data Flow Disruption**: Would the changes disrupt existing data processing pipelines?
- **Interface Compatibility**: Are there interface or contract changes that affect other components?
- **Performance Impact**: Could the implementation significantly impact system performance?

### 4. Dependency & Infrastructure Risks
**Infrastructure analysis:**
- **New Dependencies**: Are new external dependencies properly validated and researched?
- **Authentication & Security**: Are there security implications not fully addressed?
- **Scalability Concerns**: Could the implementation create scaling bottlenecks?
- **Testing Complexity**: Are there testing challenges that make validation difficult?

**CRITICAL ISSUE ASSESSMENT PROTOCOL:**

For each detected issue, provide:
1. **Issue Type**: Classification of the critical issue
2. **Impact Analysis**: How this would break or degrade the system
3. **Risk Level**: HIGH/MEDIUM/LOW with justification
4. **Affected Components**: Which parts of the system would be impacted
5. **Detection Evidence**: Specific evidence that led to issue identification

## Phase 3: INTERACTIVE RESOLUTION PROTOCOL

**IF CRITICAL ISSUES DETECTED:**

### 🚨 CRITICAL PLANNING ISSUES DETECTED - RESOLUTION REQUIRED

**Issue #1: [Issue Type]**
- **Description**: [Detailed issue description]
- **System Impact**: [How this would break/degrade the system]
- **Risk Level**: [HIGH/MEDIUM/LOW]
- **Affected Components**: [Specific system components at risk]
- **Evidence**: [What led to detecting this issue]

**Resolution Options:**
1. **Option A - [Approach Name]**
   - **Strategy**: [How this approach solves the issue]
   - **Trade-offs**: [What you gain and lose with this approach]
   - **Implementation Complexity**: [HIGH/MEDIUM/LOW]
   - **Risk Mitigation**: [How this reduces the identified risks]

2. **Option B - [Alternative Approach]**
   - **Strategy**: [How this approach solves the issue]
   - **Trade-offs**: [What you gain and lose with this approach]
   - **Implementation Complexity**: [HIGH/MEDIUM/LOW]
   - **Risk Mitigation**: [How this reduces the identified risks]

3. **Option C - [Third Alternative]**
   - **Strategy**: [How this approach solves the issue]
   - **Trade-offs**: [What you gain and lose with this approach]
   - **Implementation Complexity**: [HIGH/MEDIUM/LOW]
   - **Risk Mitigation**: [How this reduces the identified risks]

**[Repeat for each critical issue detected]**

### USER DECISION REQUIRED 🛑

**COMMAND EXECUTION PAUSED**

Critical issues must be resolved before proceeding with implementation planning. Please choose your preferred resolution approach for each issue:

**Issue #1 Resolution**: Select Option A, B, C, or request additional analysis
**Issue #2 Resolution**: [If multiple issues detected]

**Response Format:**
```
/project-planning-interactive continue issue1:optionA issue2:optionB
```

Or request more analysis:
```
/project-planning-interactive analyze-more [specific area to analyze further]
```

**⚠️ IMPLEMENTATION PLANNING WILL NOT PROCEED UNTIL ALL CRITICAL ISSUES ARE RESOLVED ⚠️**

## Phase 4: ARCHITECTURE INTEGRATION VALIDATION

**POST-RESOLUTION VALIDATION PROTOCOL:**

### Integration Compatibility Assessment
!`find $CLAUDE_PROJECT_DIR -name "*.py" -exec grep -l "class.*\(Base\|Abstract\|Interface\)" {} \; 2>/dev/null | head -5`

### Database Schema Impact Analysis
!`find $CLAUDE_PROJECT_DIR -name "*.py" -exec grep -l "model\|schema\|table\|migration" {} \; 2>/dev/null | head -5`

### API Contract Validation
!`find $CLAUDE_PROJECT_DIR -name "*.py" -exec grep -l "endpoint\|route\|api\|response" {} \; 2>/dev/null | head -5`

**INTEGRATION VALIDATION CHECKLIST:**

✅ **Architecture Pattern Compliance**
- Does the chosen resolution approach align with existing architecture patterns?
- Are new components following established naming and structure conventions?
- Will the implementation integrate cleanly with existing data flows?

✅ **Database & Storage Integration**
- Are there database schema changes required?
- Do new data models align with existing patterns?
- Are there migration or backward compatibility concerns?

✅ **API & Interface Compatibility**
- Will existing API contracts remain intact?
- Are there new API endpoints that follow established patterns?
- Will the changes affect existing client integrations?

✅ **Testing & Validation Framework**
- Can the implementation be properly tested with existing frameworks?
- Are there new testing requirements or challenges?
- Will existing tests continue to pass?

✅ **Performance & Scalability**
- Will the implementation maintain existing performance characteristics?
- Are there new scalability considerations?
- Will the changes affect system resource usage?

**VALIDATION RESULT:**
- **✅ INTEGRATION APPROVED**: Implementation approach is architecturally sound
- **⚠️ INTEGRATION CONCERNS**: Minor issues identified, proceed with caution
- **❌ INTEGRATION BLOCKED**: Major compatibility issues, resolution required

## Phase 5: IMPLEMENTATION SAFETY REVIEW & FINAL VALIDATION

### Pre-Implementation Safety Protocol
!`find $CLAUDE_PROJECT_DIR -name "*.md" -path "*/test*" -o -name "*test*.py" 2>/dev/null | head -5`

**FINAL SAFETY CHECKLIST:**

🔒 **Breaking Change Analysis**
- [ ] No existing functionality will be broken
- [ ] All external interfaces remain compatible
- [ ] Database migrations are backward compatible
- [ ] Existing tests will continue to pass

🔒 **External Dependency Validation**
- [ ] All external services properly researched and documented
- [ ] API rate limits and constraints understood
- [ ] Data format and update frequencies validated
- [ ] Authentication and security requirements addressed

🔒 **System Integration Safety**
- [ ] Implementation aligns with existing architecture patterns
- [ ] New components follow established conventions
- [ ] Data flows integrate cleanly with existing pipelines
- [ ] Performance impact is acceptable

🔒 **Testing & Validation Coverage**
- [ ] Implementation can be properly tested
- [ ] Test coverage plan is comprehensive
- [ ] Integration tests address critical paths
- [ ] Rollback strategy is defined

🔒 **Documentation & Knowledge Transfer**
- [ ] Implementation approach is well-documented
- [ ] Critical decisions and trade-offs are recorded
- [ ] Maintenance and operational concerns addressed
- [ ] Team knowledge transfer plan exists

**FINAL APPROVAL STATUS:**
- **✅ APPROVED FOR IMPLEMENTATION**: All safety checks passed
- **⚠️ CONDITIONAL APPROVAL**: Minor concerns noted, proceed with monitoring
- **❌ IMPLEMENTATION BLOCKED**: Critical safety issues remain unresolved

## Phase 6: VALIDATED IMPLEMENTATION PLAN GENERATION

**ONLY EXECUTE THIS PHASE AFTER:**
- ✅ All critical issues resolved through user interaction
- ✅ Architecture integration validated
- ✅ Safety review completed and approved

### Comprehensive Implementation Plan

Generate a detailed, phase-by-phase implementation plan that incorporates:

1. **Resolved Critical Issues**: How the chosen resolution approaches are integrated
2. **Architecture Integration**: Specific integration points and compatibility measures
3. **Implementation Phases**: Step-by-step implementation with validation checkpoints
4. **Risk Mitigation**: Specific measures to address identified risks
5. **Testing Strategy**: Comprehensive testing approach covering all critical paths
6. **Rollback Plan**: Clear rollback strategy if issues arise during implementation
7. **Performance Monitoring**: Metrics and monitoring for the new implementation
8. **Documentation Requirements**: What documentation needs to be created or updated

### Technical Specification Generation

Create detailed technical specifications including:

1. **Component Architecture**: Detailed design of new components
2. **Data Models**: Database schemas, API contracts, data structures
3. **Integration Interfaces**: How new components integrate with existing system
4. **Configuration Management**: Environment variables, settings, feature flags
5. **Error Handling**: Comprehensive error handling and recovery strategies
6. **Security Considerations**: Authentication, authorization, data protection
7. **Performance Optimization**: Caching, indexing, query optimization strategies
8. **Monitoring & Observability**: Logging, metrics, alerting, debugging support

### Success Metrics & Validation Criteria

Define clear success metrics:

1. **Functional Metrics**: What functionality must work correctly
2. **Performance Metrics**: Acceptable performance characteristics
3. **Integration Metrics**: How integration success will be measured
4. **User Experience Metrics**: Impact on user experience and workflows
5. **Operational Metrics**: System reliability and maintainability measures

## Workflow Instructions

**EXECUTION PROTOCOL:**

1. **Phase 1-2**: Always execute discovery and critical issue detection
2. **Phase 3**: If critical issues detected, PAUSE and require user resolution
3. **Phase 4-5**: Execute validation only after issue resolution
4. **Phase 6**: Generate final plans only after complete validation

**USER INTERACTION REQUIREMENTS:**
- Command must pause execution when critical issues are detected
- User must provide explicit resolution choices before proceeding
- No implementation planning without resolved critical issues
- Clear presentation of trade-offs and alternatives

**CROSS-PROJECT GENERALIZATION:**
- Dynamic discovery adapts to any project structure
- Issue detection patterns work across different technology stacks
- Validation frameworks adapt to project-specific architectures
- Implementation planning scales from simple scripts to complex systems

**VALIDATION SUCCESS CRITERIA:**
- Critical issues detected before implementation planning
- User forced to make informed decisions on issue resolution
- Architecture integration thoroughly validated
- Implementation plans incorporate risk mitigation strategies
- Cross-project usability maintained across different technology stacks