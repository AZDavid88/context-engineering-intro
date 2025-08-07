# Neon Architecture - Compute-Storage Separation for Hybrid Systems

**Source**: https://neon.com/docs/introduction/architecture-overview
**Extraction Date**: 2025-08-06
**Project Context**: Phase 4 - Hybrid Storage Architecture Design

## Core Architecture Principles

### Compute-Storage Separation
- **Neon Compute**: Runs PostgreSQL with full compatibility
- **Neon Storage**: Multi-tenant key-value store for PostgreSQL pages
- **Control Plane**: Orchestrates cloud resources across compute and storage

### Three-Layer Storage Architecture

#### 1. Safekeepers (Durability Layer)
- **Function**: Ultra-reliable write buffer for recent updates
- **Protocol**: Paxos consensus for reliability across availability zones
- **Data Flow**: PostgreSQL streams WAL → Safekeepers → Durable storage
- **Purpose**: Hold latest data until processed and uploaded to cloud storage

#### 2. Pageservers (Performance Layer)
- **Function**: Serve read requests with custom storage format
- **Processing**: Convert WAL stream into accessible page versions
- **Caching**: Function as read cache for cloud storage
- **Upload**: Process and upload data to cloud object storage

#### 3. Cloud Object Storage (Long-term Storage)
- **Provider**: Amazon S3 (99.999999999% durability)
- **Format**: Both raw WAL and materialized form
- **Access**: On-demand data download for Pageservers

## Durability Architecture for Trading Systems

### Multi-Layer Redundancy
```
Trading Data → PostgreSQL WAL → Safekeepers (Paxos Cluster)
                    ↓
Multiple Availability Zones → Cloud Object Store (S3)
                    ↓
Pageserver Caches → Fast Read Access for GA Evolution
```

### Critical Benefits for Quantitative Trading:
1. **Ultra-High Durability**: Multiple copies across AZs and cloud storage
2. **Fast Recovery**: Point-in-time restore from any WAL position
3. **Horizontal Scaling**: Pageservers can scale read performance
4. **Cost Efficiency**: Hot data in Pageservers, cold data in object storage

## Archive Storage for Hybrid Systems

### Branch Archiving Mechanism
- **Process**: Data evicted from Pageserver (not moved)
- **Storage**: Remains in cost-efficient object storage
- **Access**: Available on-demand when needed
- **Benefit**: Frees up performant Pageserver storage

### Hybrid Storage Implications
```
Active Trading Data (Hot):
├── Local DuckDB Cache → Pageserver Performance
├── Recent WAL → Safekeepers Reliability  
└── Live Queries → Direct Pageserver Access

Historical Data (Warm/Cold):
├── Archived Branches → Object Storage
├── On-Demand Loading → When needed for backtesting
└── Cost Optimization → S3 pricing vs Pageserver storage
```

## Integration Architecture for Phase 4

### Connection Pattern for Ray Workers
```python
# Multiple Ray Workers → Neon Connection Pool
AsyncConnectionPool:
├── Min Connections: 5 per worker
├── Max Connections: 20 per worker
├── Failover: Local DuckDB cache
└── Load Balancing: Across Pageservers
```

### Data Flow Architecture
```
GA Evolution Process:
├── Recent OHLCV → Pageserver (fast access)
├── Historical OHLCV → Object Storage (on-demand)
├── Evolution State → PostgreSQL tables (ACID)
└── Backup Strategy → WAL streaming + S3 storage
```

## Performance Characteristics

### Read Performance
- **Hot Data**: Pageserver cache (microsecond latency)
- **Warm Data**: Pageserver with object storage fetch
- **Cold Data**: Direct object storage access

### Write Performance
- **WAL Streaming**: Immediate Safekeeper acknowledgment
- **Durability**: Paxos consensus across AZs
- **Processing**: Asynchronous Pageserver materialization

## High Availability Features

### Automatic Failover
- **Compute**: Automatic restart with preserved state
- **Storage**: Multiple Pageservers and Safekeepers
- **Recovery**: WAL replay from last checkpoint

### Multi-AZ Deployment
- **Safekeepers**: Distributed across availability zones
- **Pageservers**: Can be replicated for read scaling
- **Object Storage**: S3 cross-region replication available

## Cost Optimization Strategy

### Tiered Storage Approach
1. **Active Data**: Keep in Pageservers for performance
2. **Archive Data**: Move to object storage for cost efficiency
3. **Hybrid Access**: On-demand loading when needed

### For Quantitative Trading:
- **Recent Market Data**: Pageserver storage for GA execution
- **Historical Backtesting**: Object storage with caching
- **Evolution State**: Always in Pageservers for consistency

## Next Research Areas

Based on this architecture understanding:
1. **Autoscaling Architecture**: How compute scales with workload
2. **Connection Patterns**: AsyncPG integration with connection pooling
3. **Security**: SSL/TLS and authentication in distributed architecture
4. **Monitoring**: Observability across compute-storage layers
5. **Performance Tuning**: Optimization for time-series workloads