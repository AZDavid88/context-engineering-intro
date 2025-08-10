# File Classification Summary
**Generated:** 2025-08-10T02:24:00Z  
**Updated:** 2025-08-10T16:00:00Z (Post-Cleanup)  
**Audit ID:** audit-20250810-022400

## Overview Statistics (Post-Cleanup)
- **Total Files Analyzed:** 389 (down from 411)
- **Python Files:** 167 (active source files)
- **Configuration Files:** 7  
- **Documentation Files:** 241 (cleaned redundant archives)
- **Script Files:** 168 (active scripts only)
- **Executable Files:** 0 (no explicit executable bits set)
- **Bytecode Files Removed:** 147 (.pyc files)
- **Cache Directories Removed:** 29 (__pycache__ directories)

## File Type Distribution

### Scripts & Code (164 files)
- **Python Scripts:** 163 files (`.py`)
  - All use `#!/usr/bin/env python3` shebang (54 confirmed)
  - Located primarily in `src/`, `tests/`, `scripts/` directories
  - Includes genetic algorithms, trading engines, data processing

- **Shell Scripts:** 1 file (`.sh`)
  - `docker/genetic-pool/entrypoint.sh` with `#!/bin/bash`

### Configuration Files (7 files)
- **Docker Compose:** `docker-compose.yml`
- **Python Project:** `pyproject.toml` 
- **Requirements:** `requirements.txt`
- **Pytest Config:** `pytest.ini`
- **Environment:** `.env`
- **Git Ignore:** `.gitignore`
- **Decision Rules:** `config/decision_rules.json`

### Documentation (233 files)
- **Markdown:** Primary format (`.md`)
- **Planning Documents:** `docs/planning/`, `phases/`
- **Reports:** `docs/reports/`, validation reports
- **Architecture:** `docs/infrastructure/`, system specs
- **Research:** `research/` directory with extensive documentation

## Probable Executables by Context

### Primary Entry Points
- **Trading System:** `src/execution/trading_system_manager.py`
- **Genetic Engine:** `src/strategy/genetic_engine.py`
- **Data Collection:** `src/data/hyperliquid_client.py`
- **Market Pipeline:** `src/data/market_data_pipeline.py`

### Docker Services
- **Container Entry:** `docker/genetic-pool/entrypoint.sh`
- **Health Check:** `docker/genetic-pool/health_check.py`

### Test Executables
- **Integration Tests:** `scripts/integration/`
- **Validation Scripts:** `scripts/validation/`
- **Debug Scripts:** `scripts/debug/`

### Poetry Console Scripts (from pyproject.toml)
- `quant-trader` → `src.main:main`
- `data-ingest` → `src.data.hyperliquid_client:main`
- `strategy-evolve` → `src.strategy.evolution_engine:main`
- `backtest-runner` → `src.backtesting.vectorbt_engine:main`
- `cli-dashboard` → `src.monitoring.cli_dashboard:main`

## Directory Structure Analysis

### Source Code Hierarchy
```
src/ (163 Python files)
├── analysis/ - Market regime and correlation detection
├── backtesting/ - VectorBT engine integration
├── config/ - Settings and rate limiting
├── data/ - Market data collection and storage
├── discovery/ - Asset universe filtering
├── execution/ - Trading system management
├── monitoring/ - System health monitoring
├── signals/ - Trading signal generation
├── strategy/ - Genetic algorithm engines
├── utils/ - Utility functions
└── validation/ - System validation pipeline
```

### Test Infrastructure (Post-Cleanup)
```
tests/ (34 test files - research archive removed)
├── comprehensive/ - Full system validation
├── integration/ - Cross-module testing (active tests)
├── unit/ - Individual component tests
├── system/ - System-level testing
├── utils/ - Test utilities and fixtures
└── config/ - Test configuration files
Note: Removed tests/research_archive/ (6 outdated files)
Note: Removed tests/archive/outdated/ (1 old file)
```

### Scripts & Tools
```
scripts/ (15+ utility scripts)
├── debug/ - Debugging tools
├── evolution/ - Genetic algorithm scripts
├── integration/ - Integration test runners
├── utilities/ - Cleanup and migration tools
└── validation/ - System validation scripts
```

## Risk Assessment - Script vs Documentation Split
- **Scripts (164 files):** Potential execution risk - require sandbox
- **Documentation (233 files):** Low risk - informational only
- **Configs (7 files):** Medium risk - may contain sensitive parameters
- **Unknown executables:** All Python files are potentially executable via interpreter

## Next Phase Preparation
- **163 Python files** require entry point analysis
- **1 shell script** needs command extraction
- **5 Poetry console scripts** defined in pyproject.toml
- **Docker services** need container execution analysis
- **Test runners** may contain additional execution paths