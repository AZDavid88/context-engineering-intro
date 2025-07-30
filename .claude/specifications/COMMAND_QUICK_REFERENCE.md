# Process-Enforcing Commands - Quick Reference Guide

## 🚀 ESSENTIAL WORKFLOW

### **Standard Development Sequence**
```markdown
1. /codefarm-research-foundation [technology]     # Research first
2. /codefarm-spec-feature [feature-description]   # Detailed specification
3. /codefarm-implement-spec @spec-file.md         # Implement following spec
4. /codefarm-validate-system [component]          # System validation
```

### **Crisis Management Sequence**
```markdown
1. /codefarm-halt-and-analyze [error/issue]       # STOP and analyze
2. /codefarm-impact-assessment [proposed-fix]     # Impact analysis
3. /codefarm-architectural-gate [fix]             # Architecture check
4. [Proceed with normal workflow if approved]
```

---

## 📋 COMPLETE COMMAND LIST

### **PHASE 1: DISCOVERY & FOUNDATION**
- `/codefarm-initiate-project [project-description]`
- `/codefarm-research-foundation [technology/domain]`

### **PHASE 2: ARCHITECTURE & PLANNING**
- `/codefarm-architect-system [requirements]`
- `/codefarm-validate-architecture [architecture-spec]`

### **PHASE 3: SPECIFICATION & DESIGN**
- `/codefarm-spec-feature [feature-description]`
- `/codefarm-review-spec [spec-file]`

### **PHASE 4: IMPLEMENTATION**
- `/codefarm-implement-spec [spec-file]`
- `/codefarm-validate-implementation [component]`

### **PHASE 5: INTEGRATION & TESTING**
- `/codefarm-integrate-component [component] [target-system]`
- `/codefarm-validate-system [system-area]`

### **PHASE 6: CRISIS/PROBLEM RESOLUTION**
- `/codefarm-halt-and-analyze [error/issue]`
- `/codefarm-impact-assessment [proposed-change]`
- `/codefarm-architectural-gate [feature/change]`

### **PHASE 7: MAINTENANCE & EVOLUTION**
- `/codefarm-sync-documentation [component/system]`
- `/codefarm-evolve-architecture [new-requirements]`

---

## 🎯 USAGE EXAMPLES FOR QUANT TRADING PROJECT

### **Adding New Trading Feature**
```markdown
/codefarm-research-foundation "Hyperliquid margin trading API"
/codefarm-spec-feature "margin trading with risk controls"
/codefarm-implement-spec @margin-trading-spec.md
/codefarm-validate-system "trading system with margin"
```

### **Fixing Integration Issues**
```markdown
/codefarm-halt-and-analyze "rate limiting breaking genetic engine"
/codefarm-impact-assessment "modify rate limiter configuration"
/codefarm-architectural-gate "rate limiter changes"
```

### **System Evolution**
```markdown
/codefarm-evolve-architecture "Ray cluster scaling requirements"
/codefarm-architect-system "distributed genetic algorithm processing"
```

---

## 🔄 CONTEXT RESTORATION AFTER /COMPACT

### **Session Recovery Command**
```markdown
activate CODEFARM
/codefarm-sync-documentation "current project state"
# Review @PROCESS_ENFORCING_DEVELOPMENT_METHODOLOGY.md
# Continue with appropriate phase command
```

### **Key Files for Context**
- `@.claude/specifications/PROCESS_ENFORCING_DEVELOPMENT_METHODOLOGY.md`
- `@planning_prp.md` (project context)
- `@research/` (research documentation)
- Current specification files in progress

---

## 🚫 ANTI-PATTERNS TO AVOID

### **Don't Skip Phases**
❌ Directly implementing without research/spec
✅ Follow Research → Spec → Implement sequence

### **Don't Improvise Fixes**
❌ Quick fixes during implementation
✅ Use crisis management sequence

### **Don't Ignore Documentation**
❌ Code without updating docs
✅ Each command updates relevant documentation

---

## 📊 COMMAND PRIORITY FOR IMPLEMENTATION

### **HIGH PRIORITY (Implement First)**
1. `/codefarm-research-foundation` - Anti-hallucination
2. `/codefarm-spec-feature` - Specification discipline  
3. `/codefarm-implement-spec` - Controlled implementation
4. `/codefarm-halt-and-analyze` - Crisis management

### **MEDIUM PRIORITY**
5. `/codefarm-validate-system` - System validation
6. `/codefarm-impact-assessment` - Change analysis
7. `/codefarm-architectural-gate` - Architecture compliance

### **LOWER PRIORITY**
8. Remaining commands as needed

---

## 🎯 SUCCESS INDICATORS

### **Process Working**
- No implementation without specifications
- Reduced quick fixes and improvisation
- Better system stability
- Consistent documentation

### **Process Needs Adjustment**
- Frequent bypassing of commands
- Commands feeling too rigid
- Quality not improving
- Development slowing down significantly