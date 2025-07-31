# Session Restoration Protocol
**Purpose**: Complete procedure for restoring methodology context in fresh Claude instances

---

## 🔄 RESTORATION SEQUENCE

### **Step 1: Activate CODEFARM Methodology**
```markdown
activate CODEFARM
```
This reads: `/workspaces/context-engineering-intro/.persona/CODEFARM.txt`

### **Step 2: Load Complete Context**
```markdown
Read the following files in order:

1. @.claude/specifications/METHODOLOGY_CONTEXT_REFERENCE.md
   - Complete methodology definitions and principles
   
2. @.claude/specifications/PROCESS_ENFORCING_DEVELOPMENT_METHODOLOGY.md
   - Main methodology specification
   
3. @.claude/specifications/COMMAND_QUICK_REFERENCE.md
   - Operational command reference
   
4. @projects/quant_trading/planning_prp.md (first 200 lines)
   - Current project context and status
```

### **Step 3: Verify Context Loading**
Ask Claude to confirm understanding:
```markdown
"Please confirm you understand:
1. CODEFARM multi-agent methodology (CodeFarmer, Critibot, Programmatron, TestBot)
2. Process-enforcing command suite (15 commands across 7 phases)
3. PRP methodology for research-driven specifications
4. Current quant trading project status and architecture"
```

### **Step 4: Resume Development**
Continue with appropriate command for current development phase.

---

## 📋 QUICK CONTEXT VERIFICATION

### **Essential Knowledge Check**
Fresh Claude instance should understand:

**CODEFARM System:**
- ✅ Four agent roles and responsibilities
- ✅ A-F development cycle methodology
- ✅ Multi-agent interaction patterns

**Process Commands:**
- ✅ 15 process-enforcing commands
- ✅ Phase-based development workflow
- ✅ Quality gates and validation requirements

**Integration Principles:**
- ✅ PRP methodology for specifications
- ✅ Research-first anti-hallucination approach
- ✅ File-based context persistence strategy

**Project Context:**
- ✅ Quant trading system architecture
- ✅ Current development phase and priorities
- ✅ Key integration challenges and solutions

---

## 🚨 EMERGENCY RESTORATION

### **If Context Loading Fails**
1. **Verify file paths exist**
2. **Read files individually** rather than batch
3. **Summarize key points** from each file manually
4. **Test methodology understanding** before proceeding

### **Minimal Context Restoration**
If full restoration fails, minimum required context:

```markdown
Essential Files (in order of priority):
1. @.persona/CODEFARM.txt - Multi-agent methodology
2. @.claude/specifications/METHODOLOGY_CONTEXT_REFERENCE.md - Core definitions
3. @.claude/specifications/COMMAND_QUICK_REFERENCE.md - Command usage
```

---

## 🎯 RESTORATION SUCCESS CRITERIA

### **Context Successfully Restored When:**
- ✅ Claude can explain CODEFARM methodology
- ✅ Claude knows all 15 process-enforcing commands
- ✅ Claude understands current project status
- ✅ Claude can proceed with systematic development

### **Context Restoration Failed When:**
- ❌ Claude doesn't recognize CODEFARM agents
- ❌ Claude suggests ad-hoc development approaches
- ❌ Claude doesn't enforce process discipline
- ❌ Claude makes assumptions without research

---

## 📝 RESTORATION LOG TEMPLATE

Use this template to track restoration success:

```markdown
SESSION RESTORATION LOG
Date: [DATE]
Time: [TIME]

STEP 1 - CODEFARM Activation:
[ ] activate CODEFARM command executed
[ ] Multi-agent methodology confirmed active

STEP 2 - Context Loading:
[ ] METHODOLOGY_CONTEXT_REFERENCE.md read and understood
[ ] PROCESS_ENFORCING_DEVELOPMENT_METHODOLOGY.md read
[ ] COMMAND_QUICK_REFERENCE.md read
[ ] Current project context loaded

STEP 3 - Verification:
[ ] CODEFARM methodology confirmed
[ ] Process commands understood
[ ] Project context confirmed

STEP 4 - Resume Development:
[ ] Appropriate command identified for current phase
[ ] Development ready to proceed

RESTORATION STATUS: [SUCCESS/PARTIAL/FAILED]
NOTES: [Any issues or additional context needed]
```