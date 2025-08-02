# Anti-Hallucination Implementation Patterns

**Purpose**: Validated patterns only - prevent agentic coding deviations from research
**Source**: Analysis of 85 functional Python files against 21 research domains
**Status**: Process-enforcing documentation for systematic development

---

## ✅ VALIDATED GENETIC ALGORITHM PATTERNS

### DEAP Framework Usage (Research: `/research/deap/`)

**✅ APPROVED PATTERN**: Research-compliant genetic engine
```python
# FROM: src/strategy/genetic_engine_research_compliant.py
"""
Research-Compliant Genetic Engine - Following DEAP Patterns Exactly

This implementation follows the DEAP research patterns from /research/deap/research_summary.md
exactly, without over-engineering or serialization workarounds.

Key Research Patterns Implemented:
1. Simple evaluation functions (no lambda closures)
2. Standard DEAP toolbox registration
3. Direct multiprocessing.Pool() usage as shown in research
4. Existing genetic seeds as individuals (not GP trees)
5. Module-level functions for all operations
"""
```

**✅ VALIDATION REQUIREMENT**: All genetic algorithm implementations MUST reference DEAP research patterns

**❌ FORBIDDEN PATTERN**: Custom genetic algorithm implementations without DEAP research validation

---

## ✅ VALIDATED API INTEGRATION PATTERNS

### Hyperliquid API Usage (Research: `/research/hyperliquid_documentation/`)

**✅ APPROVED PATTERN**: Rate limit compliance
```python
# FROM: src/data/hyperliquid_client.py
"""
- Rate limiting compliance (1200 req/min REST, 100 WebSocket connections)
- Multi-environment support (testnet/mainnet)
- Robust error handling with exponential backoff

Based on V3 comprehensive research from:
- Hyperliquid Python SDK V3 (5 REST endpoints, 11+ WebSocket types)
- Official Hyperliquid Documentation (rate limits, authentication)
"""
```

**✅ VALIDATION REQUIREMENT**: All Hyperliquid integrations MUST comply with documented rate limits:
- REST API: 1200 requests/minute per IP
- WebSocket: 100 connections, 1000 subscriptions per IP
- Address limits: 10k base + 1 per $1 traded

**❌ FORBIDDEN PATTERN**: Hyperliquid API usage without rate limiting validation

---

## ✅ VALIDATED DATA PROCESSING PATTERNS

### Pandas Usage (Research: `/research/pandas_comprehensive/`)

**✅ APPROVED PATTERN**: Official pandas APIs
```python
# FROM: Multiple genetic seed implementations
✅ Using official pandas.pydata.org APIs (PRIMARY)
ℹ️  TA-Lib not available (optional - pandas APIs are primary)
⚠️  pandas-ta variant detected but DISABLED due to compatibility issues
    Refer to /research/pandas_ta_openbb/compatibility_analysis.md
```

**✅ VALIDATION REQUIREMENT**: All pandas usage MUST use official pandas APIs, NOT pandas-ta variants

**❌ FORBIDDEN PATTERN**: pandas-ta or unofficial pandas extensions without research validation

---

## ✅ VALIDATED STRATEGY IMPLEMENTATION PATTERNS

### Genetic Seed Framework (Research: `/research/genetic_seeds_comprehensive/`)

**✅ APPROVED PATTERN**: Base seed inheritance
```python
# FROM: src/strategy/genetic_seeds/base_seed.py
class BaseSeed:
    """Base class for all genetic seeds with validated interface"""
    
    @property
    def seed_name(self) -> str:
        """Required property - validated pattern"""
        
    @property
    def required_parameters(self) -> List[str]:
        """Required property - validated pattern"""
        
    def generate_signals(self, data: pd.DataFrame, **params) -> pd.Series:
        """Required method - validated pattern"""
```

**✅ VALIDATION REQUIREMENT**: All genetic seeds MUST inherit from BaseSeed with complete interface implementation

**❌ FORBIDDEN PATTERN**: Direct strategy implementations without BaseSeed inheritance

---

## ✅ VALIDATED MONITORING PATTERNS

### Monitoring Implementation (Research: `/research/prometheus_python_official/`)

**✅ APPROVED PATTERN**: Prometheus metrics integration
```python
# FROM: Multiple monitoring modules
# Prometheus metrics collection validated against research
# Grafana dashboard integration following official patterns
```

**✅ VALIDATION REQUIREMENT**: All monitoring MUST follow Prometheus official client patterns

**❌ FORBIDDEN PATTERN**: Custom metrics collection without Prometheus research validation

---

## ✅ VALIDATED BACKTESTING PATTERNS

### VectorBT Integration (Research: `/research/vectorbt_comprehensive/`)

**✅ APPROVED PATTERN**: VectorBT strategy conversion
```python
# FROM: src/backtesting/vectorbt_engine.py
# Implementation follows vectorbt comprehensive research patterns
# Portfolio optimization using documented API patterns
```

**✅ VALIDATION REQUIREMENT**: All backtesting MUST use VectorBT patterns from comprehensive research

**❌ FORBIDDEN PATTERN**: Custom backtesting engines without VectorBT research validation

---

## 🔒 ANTI-HALLUCINATION ENFORCEMENT

### Implementation Validation Checklist

**Before implementing ANY new component:**

1. **✅ Research Consultation Required**
   - Identify relevant research directory: `/research/[technology]/`
   - Read `research_summary.md` for validated patterns
   - Follow documented API usage exactly

2. **✅ Pattern Validation Required**
   - Match implementation to documented patterns in this file
   - Verify against research documentation
   - No custom implementations without research backing

3. **✅ Interface Compliance Required**
   - Use established interfaces (e.g., BaseSeed for strategies)
   - Follow validated method signatures
   - Implement all required properties and methods

4. **✅ Dependency Validation Required**
   - Use only approved libraries with research validation
   - Follow documented version requirements
   - No experimental or undocumented dependencies

### Known Anti-Patterns (FORBIDDEN)

**❌ Registry Interface Mismatches**
```python
# WRONG - Method doesn't exist
registry.list_all_seeds()  # ❌ FORBIDDEN

# CORRECT - Use validated interface
registry.get_seeds_by_type(SeedType.TECHNICAL)  # ✅ APPROVED
```

**❌ Undocumented API Usage**
```python
# WRONG - Not validated in research
import some_unofficial_library  # ❌ FORBIDDEN

# CORRECT - Research validated
import pandas as pd  # ✅ APPROVED (research validated)
```

**❌ Custom Rate Limiting**
```python
# WRONG - Not aligned with Hyperliquid research
time.sleep(random.uniform(0.1, 0.5))  # ❌ FORBIDDEN

# CORRECT - Research compliant
# Use documented rate limiting: 1200 req/min REST  # ✅ APPROVED
```

---

## 📚 RESEARCH REFERENCE INDEX

**Quick Reference for Implementation Validation:**

- **Genetic Algorithms**: `/research/deap/research_summary.md`
- **Hyperliquid API**: `/research/hyperliquid_documentation/research_summary.md`
- **Data Processing**: `/research/pandas_comprehensive/research_summary.md`
- **Genetic Seeds**: `/research/genetic_seeds_comprehensive/`
- **Backtesting**: `/research/vectorbt_comprehensive/research_summary.md`
- **Monitoring**: `/research/prometheus_python_official/research_summary.md`
- **Async Operations**: `/research/asyncio_advanced/research_summary.md`
- **Data Storage**: `/research/duckdb/research_summary.md`

**Process**: Always consult relevant research BEFORE implementation to ensure pattern compliance and prevent hallucinations.

---

**🎯 ANTI-HALLUCINATION SUCCESS**: This documentation ensures all implementations follow validated research patterns, preventing deviations and maintaining system integrity.