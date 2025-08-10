# Execution Graph - Mermaid Format

```mermaid
graph TD
    %% Entry Points
    CS1[quant-trader] --> M1[src.main:main]
    CS2[data-ingest] --> M2[src.data.hyperliquid_client:main]
    CS3[strategy-evolve] --> M3[src.strategy.evolution_engine:main]
    CS4[backtest-runner] --> M4[src.backtesting.vectorbt_engine:main]
    CS5[cli-dashboard] --> M5[src.monitoring.cli_dashboard:main]
    
    %% Docker Services
    DS1[ray-head] --> CMD1[head mode]
    DS2[ray-worker-1] --> CMD2[worker mode]
    DS3[genetic-pool] --> CMD3[distributed mode]
    DS4[dev-tools] --> CMD4[shell mode]
    
    %% Shell Entrypoint
    SH1[entrypoint.sh] --> CMD1
    SH1 --> CMD2
    SH1 --> CMD3
    SH1 --> CMD4
    
    %% Core Python Modules
    M2 --> EXT1[Hyperliquid API]
    M1 --> CORE1[trading_system_manager.py]
    M3 --> CORE2[genetic_engine.py]
    M4 --> CORE3[vectorbt_engine.py]
    M5 --> CORE4[monitoring_core.py]
    
    %% Ray Remote Functions
    CORE1 --> RAY1[@ray.remote EvolutionWorker]
    CORE2 --> RAY2[@ray.remote GeneticEvolutionWorker]
    
    %% External Resources
    CORE1 --> EXT2[PostgreSQL]
    CORE1 --> EXT3[Redis Cache]
    CORE2 --> EXT4[Market Data APIs]
    CORE4 --> EXT5[Prometheus]
    CORE4 --> EXT6[Grafana]
    
    %% Test Suites
    TEST1[Unit Tests] -.-> CORE1
    TEST2[Integration Tests] -.-> CORE2
    TEST3[System Tests] -.-> CORE3
    TEST4[Validation Scripts] -.-> CORE4
    
    %% Styling
    classDef entryPoint fill:#e3f2fd,stroke:#2196f3
    classDef pythonModule fill:#e8f5e9,stroke:#4caf50
    classDef rayRemote fill:#fff8e1,stroke:#ffc107
    classDef external fill:#ffebee,stroke:#f44336
    classDef test fill:#fce4ec,stroke:#e91e63
    
    class CS1,CS2,CS3,CS4,CS5,DS1,DS2,DS3,DS4,SH1 entryPoint
    class M1,M2,M3,M4,M5,CORE1,CORE2,CORE3,CORE4 pythonModule
    class RAY1,RAY2 rayRemote
    class EXT1,EXT2,EXT3,EXT4,EXT5,EXT6 external
    class TEST1,TEST2,TEST3,TEST4 test
```

## Graph Statistics
- **Total Nodes**: 125 (90 orphaned, indicating high fragmentation)
- **Total Edges**: 19 (low connectivity)
- **Entry Points**: 14 (5 console scripts + 9 docker services)
- **Connected Components**: 106 (high fragmentation)

## Critical Findings
1. **High Fragmentation**: 90 orphaned nodes suggest many unused or disconnected components
2. **Low Connectivity**: Only 19 edges for 125 nodes indicates poor integration
3. **Multiple Entry Points**: 14 different ways to start the system
4. **Ray Distribution**: Distributed computing via Ray but limited remote functions identified