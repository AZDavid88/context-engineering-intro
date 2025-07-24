# Asset-Agnostic Quant Trading Organism - Planning PRP

## Project Overview

**Project Name**: The Quant Organism Genesis Protocol

**Vision**: Build a production-grade, self-regulating algorithmic trading system that discovers and exploits market inefficiencies through genetic evolution, without reliance on LLMs.

**Problem Statement**: Create a robust, antifragile trading system capable of:
- Automatically generating and testing trading strategies through genetic algorithms
- Evolving strategies using Darwinian selection principles
- Trading across crypto assets on Hyperliquid with minimal human intervention
- Self-improving through systematic analysis of strategy persmance

**Target Platform**: Hyperliquid (crypto perpetuals)
**Initial Capital**: $10,000
**Trading Style**: Day trading (swing, scalping, short-term) - NOT HFT
**Performance Requirement**: Sharpe Ratio > 2

## Core Features

### Feature 1: Genetic Strategy Evolution Engine
**User Story**: "As a quant trader, I want the system to automatically generate, test, and evolve trading strategies using genetic algorithms, so that I can discover profitable patterns without manual hypothesis creation."

**Technical Approach**:
- Genetic programming with Abstract Syntax Trees (AST)
- Strategy genes encode: technical indicators, entry/exit conditions, position sizing, risk parameters
- Crossover and mutation operators for strategy evolution
- Pure algorithmic approach (no LLM involvement)

### Feature 2: Automated Backtesting & Validation Pipeline
**User Story**: "As a risk-conscious trader, I want every strategy to pass through a multi-stage validation gauntlet before risking real capital."

**Validation Stages**:
1. In-sample optimization (60% of historical data)
2. Out-of-sample testing (20% of historical data)
3. Walk-forward analysis (20% of historical data)
4. Paper trading minimum 1 week
5. Only strategies with Sharpe > 2 across ALL stages proceed to live trading

### Feature 3: Real-time Execution & Risk Management System
**User Story**: "As a systematic trader, I want the system to execute approved strategies automatically on Hyperliquid while enforcing strict risk limits."

**Key Components**:
- WebSocket connection for real-time data
- Robust order management with retry logic
- Position tracking and PnL monitoring
- Circuit breakers (max drawdown, daily loss limits)
- Automatic strategy deactivation on underperformance

## Additional v1 Features

### Performance Analytics (CLI-based)
- Rich/Textual terminal dashboard
- Real-time metrics display
- Historical performance analysis
- No GUI required

### Strategy Lifecycle Management
- Birth → Testing → Production → Decay → Death cycle
- Automated retirement of underperforming strategies
- Post-mortem analysis for future strategy improvement
- Death reports feed back into genetic fitness function

### Portfolio Allocation Engine
- Multi-factor scoring (Sharpe, consistency, drawdown, age)
- Kelly Criterion with safety factor (max 25% per strategy)
- Hard limit: No strategy gets >40% of capital
- Dynamic rebalancing based on performance

### Market Regime Detection
- Statistical analysis of market conditions
- Strategy adaptation based on detected regimes
- Input to hypothesis generation process

## Technology Stack

### Core Language & Framework
- **Python 3.11+** with asyncio for concurrent operations
- **FastAPI** for internal service APIs
- **Pydantic** for data validation

### Data Layer
- **DuckDB** for analytical queries and backtesting (start here)
- **TimescaleDB** for time-series metrics (scale later)
- **Upstash** for state management and queuing (Redis alternative)
- **Parquet files** for historical data storage

### Trading & Evolution
- **Hyperliquid Python SDK** for exchange integration
- **Vectorbt** for backtesting engine
- **DEAP** for genetic programming
- **multiprocessing** for parallel evaluation (start here)
- **Ray** for distributed computing (scale to cloud VM later)
- **pandas-ta** for technical indicators

### Infrastructure
- **venv** for Python environments (Phase 1-2)
- **Docker** for containerization (Phase 3-4)
- **Supervisor** for process management
- **NordVPN CLI** for VPN automation (required for Hyperliquid)

### Monitoring & Logging
- **Rich/Textual** for CLI dashboard
- **Loguru** for structured logging
- **CSV/Parquet** for metrics storage (start simple)
- **Sentry** for error tracking (when scaled)

## Architecture Design

### VPN Zone Separation
```
┌─────────────────────────────────────────────────────────┐
│                   Non-VPN Zone (90%)                     │
│  - Strategy Evolution    - Backtesting Engine           │
│  - Performance Analytics - Data Storage                 │
│  - Market Regime Analysis- Strategy Lifecycle Mgmt      │
└─────────────────────────┬───────────────────────────────┘
                          │ Message Queue (Upstash)
┌─────────────────────────┴───────────────────────────────┐
│                    VPN Zone (10%)                        │
│  - WebSocket Feed  - Order Execution  - Position Monitor│
└─────────────────────────────────────────────────────────┘
```

**Benefits**:
- Cost optimization (VPN only for execution)
- Fault isolation
- Independent scaling
- Better security

### Key Architecture Decisions

1. **State Management**: Traditional database (PostgreSQL/SQLite) - simple tables for strategies, trades, performance
2. **Strategy Identity**: `{algorithm_type}_{generation}_{hash(genes)[:8]}` format
3. **Correlation Prevention**: 70% correlation threshold between active strategies
4. **Data Source**: Hyperliquid S3 historical data (L2 book snapshots, asset contexts)
5. **Paper Trading**: Hyperliquid testnet with realistic VPN latency

## Development Phases

### Phase 1: Foundation (Weeks 1-2)
- Hyperliquid WebSocket connection & data ingestion
- Basic strategy representation (AST-based)
- Simple backtesting engine
- CLI monitoring framework

### Phase 2: Intelligence Layer (Weeks 3-4)
- Genetic algorithm for strategy generation
- Multi-stage validation pipeline
- Strategy lifecycle tracking
- Performance database

### Phase 3: Execution Engine (Weeks 5-6)
- Order management system
- Risk management (position limits, stop losses)
- Paper trading mode
- Real-time P&L tracking

### Phase 4: Evolution & Optimization (Weeks 7-8)
- Portfolio allocation engine
- Market regime detection
- Strategy retirement & post-mortem analysis
- Self-improvement feedback loops

## Research Targets

### Priority 1 (Must Research First)
1. **Hyperliquid Documentation**: https://hyperliquid.gitbook.io/hyperliquid-docs
   - WebSocket API, REST API, Rate limits, Error handling
   - Market data structure, Order types

2. **Hyperliquid Python SDK**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk ✅ **COMPLETED**
   - Official Python integration examples
   - **Research Status**: Complete via V3 Comprehensive method
   - **API Documentation**: Fully discovered and documented
   - **Implementation Patterns**: Production-ready code templates available

3. **DEAP Documentation**: https://deap.readthedocs.io/
   - Genetic programming sections
   - Custom operators for strategy evolution
   - Distributed evaluation patterns

4. **Vectorbt Documentation**: https://vectorbt.dev/
   - Custom indicators, Portfolio simulation
   - Performance metrics calculation

5. **Supervisor Documentation**: http://supervisord.org/
   - Process management, Auto-restart configuration

### Priority 2 (Research During Implementation)
1. **TimescaleDB**: Time-series best practices | https://docs.tigerdata.com/
2. **Ray Core**: https://docs.ray.io/ (when scaling beyond multiprocessing)
3. **pandas-ta**: Technical indicator library | https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas
4. **Upstash**: Redis-compatible serverless database | https://upstash.com/docs/introduction

### Priority 3 (Research When Needed)
1. **Docker**: Deployment phase documentation | https://docs.docker.com/
2. **Advanced allocation algorithms**: PyPortfolioOpt | https://pyportfolioopt.readthedocs.io/en/latest/

## Constraints & Considerations

1. **VPN Requirement**: All Hyperliquid connections require VPN (NordVPN)
2. **No LLM Dependency**: Pure algorithmic/statistical approaches only
3. **Capital Constraints**: Start with $10k, strategies must achieve Sharpe > 2
4. **Latency Requirements**: Day trading only, no HFT (cloud VM acceptable)
5. **Development Philosophy**: Build incrementally, extend rather than rebuild

## Success Metrics

1. **Strategy Quality**: Consistent Sharpe > 2 across all validation stages
2. **System Reliability**: <0.1% downtime, automatic recovery from failures
3. **Evolution Effectiveness**: Measurable improvement in strategy performance over generations
4. **Risk Management**: Maximum drawdown <10%, no single strategy >40% allocation

## Next Steps

1. Review this planning document
2. Execute research phase using `/execute-research' (execute-research-playwright OR execute-research-brightdata) command
3. Begin Phase 1 implementation following the architectural blueprint
4. Set up development environment with specified tools
5. Configure VPN and test Hyperliquid connectivity
