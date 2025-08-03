---
description: Forensic reconstruction with documentation decontamination implementation
allowed-tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS, Bash
argument-hint: [project-path] OR --implement [forensic-report-path] - Project path for full analysis, or implement existing forensic report
---

# CODEFARM Forensic Code Reconstruction & Implementation Command
**Framework**: FPT + HTN + CoT (Hybrid Complex)  
**Purpose**: Establish ground truth from code evidence and implement systematic decontamination  
**Anti-Contamination**: Code-first analysis with systematic documentation validation and implementation

## 🔄 EXECUTION MODES

### **Mode 1: Full Analysis + Implementation (Default)**
```bash
/codefarm-forensic-reconstruction [project-path]
```
- Executes complete forensic analysis (Phases A-C)
- **PAUSE POINT**: User reviews forensic findings
- User confirms implementation → Executes decontamination (Phase D)

### **Mode 2: Implementation Only (Existing Report)**
```bash
/codefarm-forensic-reconstruction --implement [forensic-report-path]
```
- Skips analysis phases, uses existing forensic report
- Directly executes decontamination implementation based on existing findings
- For use when forensic analysis already completed

## 🔍 CODEFARM Multi-Agent Activation (Forensic Implementation Specialists)

**CodeFarmer (Forensic Investigation & Strategy Lead):** "I'll systematically reconstruct project reality from code evidence through First Principles analysis, then design systematic implementation of decontamination protocols based on forensic findings."

**Critibot (Contamination Detection & Safety Controller):** "I'll identify documentation contamination patterns and validation theater indicators, then implement systematic safety protocols ensuring decontamination preserves all functional capabilities."

**Programmatron (Ground Truth & Implementation Architect):** "I'll build accurate project understanding from code analysis, then execute systematic implementation of anti-contamination processes and development workflows based on code reality."

**TestBot (Evidence Validation & Implementation Tester):** "I'll validate all forensic findings against concrete code evidence, then test and verify that decontamination implementation maintains system functionality while preventing future contamination."

---

## 🔬 SYSTEMATIC FORENSIC RECONSTRUCTION METHODOLOGY

### **EXECUTION MODE DETERMINATION**

**CodeFarmer Argument Analysis:**

**Conditional Execution Logic:**

**IF $ARGUMENTS contains "--implement":**
- **Implementation Mode**: Extract forensic report path from arguments
- **Skip to Phase D**: Use existing forensic report as input for implementation
- **Validate Report**: Ensure forensic report exists and contains required analysis

**ELSE (Default Mode):**
- **Full Analysis Mode**: Execute complete forensic analysis (Phases A-C)  
- **Pause Point**: Present findings to user for review and implementation approval
- **Conditional Implementation**: Execute Phase D only with user confirmation

**Argument Processing:**
```bash
# Implementation Mode Example:
$ARGUMENTS = "--implement /path/to/forensic_reconstruction_report.md"
→ MODE: Implementation Only
→ REPORT_PATH: /path/to/forensic_reconstruction_report.md

# Default Mode Example:  
$ARGUMENTS = "/workspaces/context-engineering-intro/projects/quant_trading"
→ MODE: Full Analysis + Implementation (with pause)
→ PROJECT_PATH: /workspaces/context-engineering-intro/projects/quant_trading
```

### **PHASE A: First Principles Code Evidence Collection**

**CodeFarmer Evidence-Based Investigation:**

#### **Step 1: Code-First Project Structure Analysis**

**FPT Principle**: Start with what exists, not what documentation claims exists

**Using LS tool to establish actual project structure**
Using LS tool with path: $ARGUMENTS to discover real project organization

**CoT Reasoning**: Directory structure reveals actual vs. claimed organization patterns

#### **Step 2: Implementation File Discovery (Ignore Documentation)**

**FPT Principle**: Analyze only executable code, configuration, and dependency files

**Using Glob tool to find actual implementation files**
Using Glob tool with path: $ARGUMENTS and pattern: "*.py" to discover Python implementations
Using Glob tool with path: $ARGUMENTS and pattern: "requirements*.txt" to find real dependencies
Using Glob tool with path: $ARGUMENTS and pattern: "*.json" to discover configuration files
Using Glob tool with path: $ARGUMENTS and pattern: "*.yml" to find orchestration files
Using Glob tool with path: $ARGUMENTS and pattern: "*.yaml" to discover YAML configurations

**CoT Reasoning**: Implementation files contain ground truth, documentation files may contain contamination

#### **Step 3: Actual Functionality Discovery**

**FPT Principle**: Functions, classes, and imports reveal real capabilities

**Using Grep tool to discover actual implementation patterns**
Using Grep tool with path: $ARGUMENTS, pattern: "^class |^def |^async def ", output_mode: "content", head_limit: 50

**Using Grep tool to find real import dependencies**
Using Grep tool with path: $ARGUMENTS, pattern: "^import |^from .* import", output_mode: "files_with_matches"

**CoT Reasoning**: Code patterns reveal actual functionality regardless of documentation claims

---

### **PHASE B: Documentation Contamination Detection**

**Critibot Systematic Contamination Analysis:**

#### **Step 4: Documentation vs. Code Reality Comparison**

**FPT Principle**: Every documentation claim must be validated against code evidence

**Using Glob tool to find documentation files**
Using Glob tool with path: $ARGUMENTS and pattern: "*.md" to identify documentation files
Using Glob tool with path: $ARGUMENTS and pattern: "*.rst" to find additional documentation
Using Glob tool with path: $ARGUMENTS and pattern: "README*" to locate readme files

**For each documentation file found, systematic validation:**

**CoT Reasoning**: Compare documentation claims against actual code implementation to identify contamination

#### **Step 5: Validation Theater Detection**

**FPT Principle**: Test files should validate actual functionality, not phantom features

**Using Glob tool to find test files**
Using Glob tool with path: $ARGUMENTS and pattern: "*test*.py" to discover test implementations
Using Glob tool with path: $ARGUMENTS and pattern: "test_*.py" to find test files
Using Glob tool with path: $ARGUMENTS and pattern: "tests/*.py" to locate test directories

**Using Grep tool to analyze test patterns**
Using Grep tool with path: $ARGUMENTS, pattern: "def test_|assert |pytest", output_mode: "content", head_limit: 30

**CoT Reasoning**: Real tests validate actual code behavior; validation theater creates false confidence

#### **Step 6: Configuration Reality Assessment**

**FPT Principle**: Configuration files reveal actual system dependencies and behavior

**Using Read tool to examine dependency files**
Read tool for each requirements.txt, package.json, or similar configuration file discovered

**Using Grep tool to find environment dependencies**
Using Grep tool with path: $ARGUMENTS, pattern: "os.environ|getenv|ENV", output_mode: "files_with_matches"

**CoT Reasoning**: Configuration files cannot lie about actual dependencies and environmental requirements

---

### **PHASE C: Ground Truth Reconstruction**

**Programmatron Evidence-Based Architecture:**

#### **Step 7: Actual System Capabilities Mapping**

**FPT Principle**: Build understanding from verified code evidence only

**For each implementation file discovered:**
- **File Purpose Analysis**: What does this code actually do?
- **Dependency Verification**: What external systems does it actually use?
- **Interface Documentation**: What are the real input/output patterns?
- **Integration Points**: How does it actually connect to other components?

**CoT Reasoning**: Systematic file-by-file analysis builds accurate system understanding

#### **Step 8: Real vs. Claimed Feature Analysis**

**FPT Principle**: Features exist only if implemented and functional

**Implementation Pattern Analysis:**
- **Completed Features**: Code that actually executes and provides functionality
- **Partial Features**: Code that exists but may not be fully functional
- **Missing Features**: Claims in documentation without corresponding implementation
- **Phantom Features**: Documentation claims about non-existent functionality

**CoT Reasoning**: Feature analysis must be evidence-based, not documentation-based

#### **Step 9: Systematic Architecture Reconstruction**

**FPT Principle**: System architecture emerges from actual code relationships and data flow

**Actual Architecture Discovery:**
- **Module Dependencies**: Real import relationships and dependencies
- **Data Flow Patterns**: How information actually moves through the system
- **Integration Architecture**: Real external service connections and APIs
- **Configuration Management**: Actual environment and deployment patterns

**CoT Reasoning**: Architecture documentation must reflect implementation reality

---

## 📊 FORENSIC RECONSTRUCTION OUTPUT

### **Step 10: Comprehensive Forensic Report Generation**

**TestBot Evidence Validation & Documentation:**

**Creating Forensic Reconstruction Report at: $ARGUMENTS/forensic_reconstruction_report.md**

**Required Report Sections:**

#### **1. GROUND TRUTH PROJECT STATE**
- **Actual Implementation**: What the code really does (evidence-based)
- **Real Dependencies**: Verified external requirements and integrations  
- **Functional Features**: Confirmed working capabilities with code evidence
- **System Architecture**: Actual module relationships and data flow patterns

#### **2. DOCUMENTATION CONTAMINATION ANALYSIS**
- **Contamination Patterns**: Where documentation diverges from code reality
- **Validation Theater Detection**: Tests that don't validate real functionality
- **Phantom Claims**: Documentation features without corresponding implementation
- **Accuracy Assessment**: Documentation reliability scoring by section

#### **3. EVIDENCE-BASED DECONTAMINATION PLAN**
- **Documentation Cleanup**: Priority order for aligning docs with code reality
- **Validation Improvement**: Real testing strategies for actual functionality
- **Process Prevention**: Methods to prevent future contamination accumulation
- **Quality Gates**: Code-truth validation checkpoints for ongoing development

#### **4. SYSTEMATIC DEVELOPMENT PATHWAY**
- **Current State Summary**: Accurate baseline from forensic analysis
- **Development Priorities**: Evidence-based next steps for actual improvement
- **Anti-Contamination Protocols**: Systematic development process recommendations
- **Success Metrics**: Measurable indicators tied to code reality, not documentation claims

---

## ⏸️ IMPLEMENTATION DECISION POINT

### **DEFAULT MODE: User Review & Confirmation Required**

**CodeFarmer Implementation Decision Protocol:**

**IF executing in DEFAULT MODE (project-path provided):**

**PAUSE POINT REACHED**: Forensic analysis complete, implementation requires user approval

**Forensic Analysis Summary:**
- **Forensic reconstruction report**: Created at `$ARGUMENTS/forensic_reconstruction_report.md`
- **Ground truth established**: Code reality vs. documentation claims analyzed
- **Contamination patterns identified**: Specific issues documented with evidence
- **Implementation plan ready**: Systematic decontamination strategy prepared

**COMMAND EXECUTION HALT**: Analysis phase complete, awaiting user decision

**USER CONFIRMATION REQUIRED:**

The forensic analysis is complete. Please review the forensic reconstruction report before proceeding.

**Implementation will:**
1. **Quarantine contaminated documentation** in timestamped archive directory
2. **Replace with code-reality documentation** based on forensic findings  
3. **Implement anti-contamination development processes** 
4. **Enable immediate development capability** using functional codebase

**To proceed with implementation:**
- **Option A**: Run `/codefarm-forensic-reconstruction --implement $ARGUMENTS/forensic_reconstruction_report.md`
- **Option B**: Respond with "y" to continue implementation in current session

**EXECUTION STOPS HERE** until user provides explicit approval for implementation phase

**User Input Validation:**
- **IF user responds "y" or "Y"** → Continue to Phase D implementation
- **IF user responds "n" or "N"** → End execution with analysis complete
- **IF user runs --implement command** → Execute Phase D using existing report
- **ANY OTHER RESPONSE** → Request clarification and re-prompt for y/n decision

### **IMPLEMENTATION MODE: Direct Execution**

**IF executing in IMPLEMENTATION MODE (--implement flag provided):**

**SKIP PAUSE**: Using existing forensic report for direct implementation

**Using Read tool to load existing forensic report**
Using Read tool with file_path: $REPORT_PATH (extracted from --implement argument)

**Validating forensic report contains required analysis sections:**
- [ ] Ground truth project state
- [ ] Documentation contamination analysis  
- [ ] Evidence-based decontamination plan
- [ ] Systematic development pathway

**IF validation successful → Proceed directly to Phase D (Implementation)**
**IF validation fails → Error: Invalid or incomplete forensic report**

---

## 🔧 PHASE D: SYSTEMATIC DECONTAMINATION IMPLEMENTATION (HTN + CoT)

**CodeFarmer Implementation Strategy Based on Forensic Findings:**

**CoT Reasoning**: Forensic analysis has established ground truth and identified contamination patterns. Implementation Phase D ONLY executes when:
1. **--implement flag provided** with valid forensic report path, OR  
2. **User explicitly confirms** with "y" response after reviewing analysis

**CRITICAL**: Phase D does NOT auto-execute in DEFAULT mode - requires explicit user approval.

### **HTN Goal: Transform Forensic Findings → Immediate Development Capability**

#### **Subgoal A: Contamination Removal with Minimal Reality Replacement**

**CodeFarmer Balanced Decontamination Execution:**

**Step 11: Contaminated Documentation Quarantine**

**HTN Conditional Logic**: If forensic analysis identified contaminated documentation → Quarantine and replace with minimal code-reality versions

**Using Bash tool to create contamination quarantine structure**
Using Bash tool with command: "mkdir -p $ARGUMENTS/docs/archive/contaminated/$(date +%Y%m%d_%H%M%S)" to create timestamped quarantine directory

**Using Bash tool to move contaminated documentation**
For each contaminated document identified in forensic analysis:
- Move to quarantine directory with contamination tags
- Create minimal code-reality replacement (max 50 lines) addressing specific false claims
- Remove cross-references to contaminated documents

**Using Write tool to create minimal reality-based replacements**
For quarantined documents, create minimal replacements focusing on:
- Actual file sizes and metrics (not inflated claims)
- Real system status (not false crisis narratives)
- Verified capabilities (not phantom problems)

**CoT Reasoning**: Physical separation prevents contaminated documentation from influencing decisions. Minimal reality-based replacements address specific false claims without recreating process bloat.

#### **Subgoal B: Code Interface Correction (Actual Functionality Fixes)**

**Programmatron Direct Code Enhancement:**

**Step 12: Test Interface Method Correction**

**HTN Task Network**: Fix actual code issues identified during system validation

**Using Edit tool to fix method interface evolution issues**
Based on test results analysis:
- Locate and correct method name misalignments (e.g., list_all_seeds vs actual interface)
- Fix monitoring system interface methods (e.g., inject_components method names)
- Update method calls to match current interface expectations

**Using Bash tool to validate fixes**
Using Bash tool with command: "cd $ARGUMENTS && python -m pytest tests/integration/ -v" to confirm interface corrections work

**CoT Reasoning**: Fix actual code functionality issues rather than creating process documentation about problems.

#### **Subgoal C: Development Readiness Confirmation (Minimal Output)**

**TestBot Validation & Development Enablement:**

**Step 13: Development Capability Confirmation**

**HTN Execution Path**: Confirm immediate development readiness using functional codebase

**Using Bash tool to validate current system state**
Using Bash tool with command: "cd $ARGUMENTS && python -m pytest tests/ --tb=short -x" to validate system functionality

**Using Write tool to create minimal development status and anti-contamination guidance**
Create minimal status file at: $ARGUMENTS/DEVELOPMENT_STATUS.md

**Content (maximum 50 lines):**
```markdown
# Development Status
**Date**: $(date +%Y-%m-%d)
**System**: Production-ready quantitative trading system
**Test Status**: [Results from pytest]
**Ready for**: Immediate feature development

## Next Actions:
1. Choose enhancement: strategy expansion, monitoring, or backtesting
2. Build on existing research foundation in /research/
3. Use systematic code organization in /src/

## Code Reality:
- 64 Python files in organized /src/ structure
- 25 test files with comprehensive coverage
- 30+ research domains available
- System functional and ready for development

## Simple Anti-Contamination Protocol:
1. Always verify claims against actual code before accepting
2. Use `wc -l filename` to check real file sizes vs. claims
3. Run `python -m pytest tests/` to verify actual system status
4. Focus development on code functionality, not process documentation
```

**Using Write tool to create basic contamination prevention guideline**
Create simple guideline at: $ARGUMENTS/docs/CONTAMINATION_PREVENTION.md (maximum 30 lines)

**Content focuses on:**
- Simple verification commands for claims vs. reality
- Basic principles to prevent future contamination
- Focus on code-first development approach

**CoT Reasoning**: Minimal guidance addresses forensic report requirements for anti-contamination measures without creating extensive process overhead.

---

## ✅ IMPLEMENTATION VALIDATION FRAMEWORK

### **CODEFARM Implementation Effectiveness Validation**

**CodeFarmer Implementation Strategy Validation:**
- [ ] Decontamination implementation based directly on forensic findings, not theoretical approaches
- [ ] Development processes anchored to code reality rather than documentation artifacts
- [ ] Anti-contamination protocols prevent future documentation drift
- [ ] Implementation enables immediate development capability using functional system

**Critibot Implementation Safety Validation:**
- [ ] All decontamination preserves existing system functionality without disruption
- [ ] No documentation removal without code-reality verification of safety
- [ ] Implementation process prevents introduction of new contamination patterns
- [ ] Validation theater elimination maintains real testing capability

**Programmatron Implementation Architecture Validation:**
- [ ] Development workflow implementation technically sound and immediately usable
- [ ] Anti-contamination processes systematically designed with measurable outcomes
- [ ] Documentation replacement reflects actual code capabilities and constraints
- [ ] Process integration enhances rather than complicates development workflow

**TestBot Implementation Testing Validation:**
- [ ] All implementation changes tested against actual system functionality
- [ ] Decontamination implementation proven effective through systematic verification
- [ ] Development process implementation enables measurable workflow improvement
- [ ] Anti-contamination monitoring system operational and effective

### **Implementation Success Metrics**
- **Contamination Elimination**: % of contaminated documentation successfully quarantined (without replacement bloat)
- **Code Functionality Restoration**: Number of actual code issues fixed (interface methods, test failures)
- **Development Capability Enablement**: System ready for immediate feature development
- **Process Overhead Reduction**: No new documentation files created beyond minimal status confirmation

**Threshold for successful implementation**: Overall implementation score ≥ 8/10

---

## ✅ FORENSIC VALIDATION FRAMEWORK

### **CODEFARM Forensic Reconstruction Validation**

**CodeFarmer Investigation Quality Validation:**
- [ ] All project analysis based on actual code evidence, not documentation assumptions
- [ ] First Principles methodology applied systematically throughout investigation
- [ ] Ground truth reconstruction accurate and verifiable against implementation
- [ ] System understanding built from code reality rather than claimed functionality

**Critibot Contamination Detection Validation:**
- [ ] Every documentation claim validated against corresponding code implementation
- [ ] Validation theater patterns systematically identified and documented
- [ ] No forensic conclusions without supporting code evidence
- [ ] Contamination detection methodology prevents assumption-based analysis

**Programmatron Reconstruction Quality Validation:**
- [ ] Architecture analysis technically accurate with specific code references
- [ ] Feature analysis distinguishes between implemented vs. claimed functionality
- [ ] Integration understanding based on actual code relationships, not documentation
- [ ] Decontamination plan implementable with concrete code-based improvements

**TestBot Evidence-Based Validation:**
- [ ] All forensic findings validated through systematic code examination
- [ ] Documentation contamination identified through evidence comparison
- [ ] Ground truth reconstruction proven accurate against implementation reality
- [ ] Forensic methodology reliable and reproducible for future investigations

### **Anti-Hallucination Framework**
- [ ] No conclusions about system capabilities without code evidence
- [ ] No recommendations without validation against actual implementation constraints
- [ ] No documentation trust without systematic verification against code reality
- [ ] No systematic development without evidence-based foundation

---

## 🎯 CONFIDENCE SCORING & SUCCESS METRICS

### **Forensic Investigation Effectiveness**
- **Code Evidence Quality**: ___/10 (completeness of implementation analysis)
- **Contamination Detection Accuracy**: ___/10 (precision in identifying doc/code gaps)
- **Ground Truth Reconstruction**: ___/10 (accuracy of actual system understanding)
- **Decontamination Plan Viability**: ___/10 (implementability of cleanup strategy)
- **Overall Forensic Analysis Quality**: ___/10 (comprehensive investigation capability)

**Threshold for actionable reconstruction**: Overall score ≥ 8/10

### **Success Criteria Framework**

**For DEFAULT MODE (Full Analysis + Implementation):**
1. **Forensic analysis complete** with ground truth established and contamination patterns identified
2. **User review completed** with implementation decision made based on forensic findings
3. **IF Implementation approved**: Decontamination operational, anti-contamination processes active, development capability enabled
4. **IF Implementation declined**: Forensic analysis available for future implementation decision

**For IMPLEMENTATION MODE (--implement existing report):**
1. **Existing forensic report validated** and loaded as implementation foundation
2. **Contaminated documentation quarantined** without replacement bloat creation
3. **Actual code issues fixed** (interface methods, test failures corrected)
4. **Immediate development capability enabled** using functional codebase with minimal process overhead

---

---

## 🛑 COMMAND EXECUTION TERMINATION (DEFAULT MODE)

**🎯 FORENSIC ANALYSIS COMPLETE**: Code reality established, contamination patterns identified, systematic decontamination and prevention protocols ready for implementation.

**📋 ANALYSIS DELIVERABLES:**
- **Forensic Report**: `$ARGUMENTS/forensic_reconstruction_report.md`
- **Ground Truth Assessment**: Code vs. documentation reality established  
- **Contamination Patterns**: Specific issues identified with evidence
- **Implementation Plan**: Ready for execution pending user approval

**⏸️ EXECUTION PAUSED**: Command halts at this point in DEFAULT mode

**👤 USER ACTION REQUIRED:**
1. **Review** the forensic reconstruction report
2. **Decide** whether to proceed with implementation
3. **Execute** implementation using one of the approved methods:
   - `y` - Continue implementation in current session
   - `/codefarm-forensic-reconstruction --implement $ARGUMENTS/forensic_reconstruction_report.md` - New session implementation

**🚫 NO AUTO-EXECUTION**: Phase D implementation requires explicit user approval

---

## ✅ COMMAND COMPLETION CRITERIA

**FOR DEFAULT MODE (Analysis Only):**
- Ground truth established through systematic code analysis
- Forensic reconstruction report generated with evidence-based findings
- User review opportunity provided with clear implementation options
- **EXECUTION HALTS** - no implementation without explicit user approval

**FOR IMPLEMENTATION MODE (--implement flag):**
- Existing forensic findings implemented through systematic decontamination
- Contaminated documentation quarantined without replacement bloat
- Actual code interface issues fixed
- Immediate development capability enabled using functional codebase foundation