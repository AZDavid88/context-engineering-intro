# Systematic Development Workflow Guide
**Project**: Quant Trading Genetic Algorithm System
**Methodology**: 7-Phase Development Process
**Version**: 1.0

## 🎯 WORKFLOW OVERVIEW

### **7-Phase Development Integration**
This project now follows the systematic 7-phase development methodology:
```
IDEATION → DISCOVERY → PLANNING → RESEARCH → SPECIFICATION → BUILD → VALIDATE/TEST
```

### **Project-Specific Workflow**
Given the project's current production-ready status with methodology compliance needs:
```
CURRENT STATE → SYSTEMATIZATION → COMPLIANCE → ENHANCEMENT → VALIDATION
```

## 📋 SYSTEMATIC COMMAND USAGE

### **Available CODEFARM Commands**
With the project now systematized, these commands are available:

#### **Research & Foundation Commands**
```bash
/codefarm-research-foundation [technology/domain]
# Use for: Adding new technology research to /research/ directory
# Example: /codefarm-research-foundation "distributed-trading-systems"
```

#### **Architecture & Planning Commands**
```bash
/codefarm-architect-system [requirements]
# Use for: Major system architecture changes
# Example: /codefarm-architect-system "add-multi-exchange-support"

/codefarm-validate-architecture [architecture-spec]
# Use for: Validating architecture changes before implementation
```

#### **Specification & Design Commands**
```bash
/codefarm-spec-feature [feature-description]
# Use for: Creating detailed specifications for new features
# Example: /codefarm-spec-feature "real-time-risk-monitoring"

/codefarm-review-spec [spec-file]
# Use for: Reviewing specifications before implementation
```

#### **Implementation Commands**
```bash
/codefarm-implement-spec [spec-file]
# Use for: Implementing features following specifications
# Ensures no improvisation, strict spec adherence

/codefarm-validate-implementation [component]
# Use for: Validating implementations against specifications
```

#### **Integration & Testing Commands**
```bash
/codefarm-integrate-component [component] [target-system]
# Use for: Systematic integration of new components

/codefarm-validate-system [system-area]
# Use for: System-wide validation after changes
```

## 🔧 PROJECT-SPECIFIC WORKFLOW PATTERNS

### **For File Systematization (Current Priority)**
```bash
# 1. Plan the systematization
/codefarm-spec-feature "split-monitoring-module-methodology-compliance"

# 2. Review the specification
/codefarm-review-spec "specs/monitoring_split_specification.md"

# 3. Implement the split safely
/codefarm-implement-spec "specs/monitoring_split_specification.md"

# 4. Validate functionality preservation
/codefarm-validate-implementation "monitoring-system"

# 5. Integrate with existing system
/codefarm-integrate-component "monitoring-modules" "trading-system"

# 6. System-wide validation
/codefarm-validate-system "complete-trading-system"
```

### **For New Feature Development**
```bash
# 1. Research new technology if needed
/codefarm-research-foundation "new-exchange-integration"

# 2. Create feature specification
/codefarm-spec-feature "add-binance-exchange-support"

# 3. Validate architecture impact
/codefarm-validate-architecture "specs/multi_exchange_architecture.md"

# 4. Implement following specification
/codefarm-implement-spec "specs/binance_integration_spec.md"

# 5. Complete systematic integration
/codefarm-integrate-component "binance-client" "trading-system"
```

### **For Crisis Management**
```bash
# 1. HALT and analyze any production issues
/codefarm-halt-and-analyze "production-error-description"

# 2. Assess system-wide impact before fixes
/codefarm-impact-assessment "proposed-fix-description"

# 3. Validate against architecture before proceeding
/codefarm-architectural-gate "emergency-fix-requirements"
```

## 📊 SYSTEMATIC QUALITY GATES

### **File-Level Quality Gates**
- **Line Limit**: All files must be ≤ 500 lines (currently 5 violations need fixing)
- **Import Organization**: Systematic import patterns and dependency management
- **Documentation**: All functions must have docstrings following Google style
- **Testing**: All new code must have comprehensive unit tests

### **System-Level Quality Gates**
- **Architecture Compliance**: All changes must align with documented architecture
- **Performance**: No performance degradation without explicit justification
- **Integration**: All external integrations must maintain compatibility
- **Security**: All changes must pass security review for trading system safety

## 🎯 DEVELOPMENT PRIORITIES

### **Immediate Priorities (Week 1)**
1. **File Systematization**: Complete splitting of 5 methodology-violating files
2. **Testing Validation**: Ensure 100% test suite compatibility after splits
3. **Import Optimization**: Clean up and optimize all import dependencies
4. **Documentation Updates**: Update all documentation for new structure

### **Short-term Development (Month 1)**
1. **Monitoring Integration**: Deploy production monitoring stack (Prometheus/Grafana)
2. **Performance Optimization**: Address any performance issues from systematization
3. **Security Hardening**: Complete security review and enhancements
4. **Distribution Testing**: Validate Ray cluster deployment with Anyscale

### **Medium-term Enhancement (Month 2-3)**
1. **Multi-Exchange Support**: Add additional exchange integrations
2. **Advanced Risk Management**: Enhanced risk controls and monitoring
3. **Strategy Expansion**: Additional genetic algorithm trading strategies
4. **Operational Excellence**: Complete monitoring and alerting deployment

## 📋 WORKFLOW CHECKLIST

### **Before Starting Any Work**
- [ ] Check current phase status in `/phases/current/`
- [ ] Review relevant specifications in `/specs/`
- [ ] Ensure all dependencies researched in `/research/`
- [ ] Run baseline tests to establish current system state

### **During Development**
- [ ] Follow systematic command workflow (spec → implement → validate)
- [ ] Maintain all files under 500-line methodology limit
- [ ] Update tests for all changes with comprehensive coverage
- [ ] Document all design decisions and rationale

### **After Completing Work**
- [ ] Run complete test suite with 100% pass rate
- [ ] Update relevant documentation and specifications
- [ ] Validate system-wide integration and performance
- [ ] Move completed work to `/phases/completed/`

## 🚀 SUCCESS METRICS

### **Methodology Compliance**
- **File Size Compliance**: 0 files exceeding 500-line limit
- **Documentation Quality**: All components have complete specifications
- **Test Coverage**: Maintain >90% test coverage across all components
- **Architecture Adherence**: All changes follow documented architecture patterns

### **Development Efficiency**
- **Systematic Process**: All development follows 7-phase methodology
- **Quality Gates**: All changes pass systematic validation before integration
- **Knowledge Preservation**: All decisions documented with rationale
- **Team Integration**: Clear workflow for multiple developers

---

**Status**: Systematic workflow established and ready for immediate use
**Next Steps**: Begin file systematization using systematic command workflow
**Documentation**: Complete methodology integration achieved with project-specific guidance