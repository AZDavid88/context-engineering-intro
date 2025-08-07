# AWS EFS Performance Specifications - Phase 4 Optimization Guide

**Source**: https://docs.aws.amazon.com/efs/latest/ug/performance.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - EFS Performance Optimization for Ray Worker Workloads

## Performance Dimensions Overview

Amazon EFS performance is measured across three critical dimensions for Phase 4 quantitative trading workloads:
- **Latency**: Response time for file operations
- **IOPS**: Input/Output Operations Per Second
- **Throughput**: Data transfer capacity (GiBps/MiBps)

## Performance Configuration Impact

### File System Type Selection

**Regional File Systems (Recommended for Phase 4):**
- Multi-AZ data redundancy
- Higher performance limits
- Better resilience for distributed Ray workers

**One Zone File Systems (Cost-Optimized Alternative):**
- Single AZ storage
- Lower latency (1.6ms write vs 2.7ms)
- Lower performance limits but adequate for smaller workloads

## Performance Modes

### General Purpose Mode (Recommended)
- **Latency**: As low as 250 microseconds (reads), 2.7ms (writes)
- **Use Case**: Latency-sensitive trading applications  
- **Ray Worker Benefit**: Optimal for real-time genetic algorithm execution
- **Default**: All new file systems, One Zone file systems always use this mode

### Max I/O Mode (Legacy - Not Recommended)
- **Latency**: Higher per-operation latencies than General Purpose
- **Use Case**: Highly parallelized workloads that tolerate higher latency
- **Phase 4 Recommendation**: **Avoid** - Use General Purpose for better performance

## Throughput Modes for Phase 4 Workloads

### Elastic Throughput (Recommended)
**Best for**: Spiky or unpredictable Ray worker access patterns

**Performance Characteristics:**
- **Regional + Elastic**: 20-60 GiBps read, 1-5 GiBps write
- **Read IOPS**: 900,000-2,500,000 (scalable based on access patterns)
- **Write IOPS**: 500,000
- **Per-Client**: Up to 1,500 MiBps with EFS client v2.0+

**Phase 4 Benefits:**
- Automatic scaling for genetic algorithm burst requirements
- No need to predict Ray worker throughput patterns
- Pay only for actual usage
- No burst credit management required

### Provisioned Throughput
**Best for**: Predictable Ray worker performance requirements

**Performance Characteristics:**
- **Regional + Provisioned**: 3-10 GiBps read, 1-3.33 GiBps write  
- **Read IOPS**: 55,000
- **Write IOPS**: 25,000
- **Per-Client**: 500 MiBps

**When to Use:**
- Average-to-peak throughput ratio ≥ 5%
- Known, consistent Ray worker access patterns
- Predictable genetic algorithm workload sizes

### Bursting Throughput
**Best for**: Throughput scaling with storage size

**Performance Characteristics:**
- **Baseline Rate**: 50 KiBps per GiB of storage
- **Burst Capacity**: Up to 100 MiBps per TiB (minimum 100 MiBps)
- **Burst Credits**: Accrue during low activity, consumed during high activity

**Burst Credit Economics:**
```
100 GiB file system:
- Baseline: 5 MiBps continuous
- 24-hour inactivity = 432,000 MiB credits
- Burst: 100 MiBps for 72 minutes

1 TiB file system:  
- Baseline: 50 MiBps continuous
- Burst: 100 MiBps for 12 hours daily
```

## Performance Specifications Table

| Configuration | Read Latency | Write Latency | Read IOPS | Write IOPS | Read Throughput | Write Throughput |
|---------------|--------------|---------------|-----------|------------|-----------------|------------------|
| **Regional + Elastic** | 250µs | 2.7ms | 900K-2.5M | 500K | 20-60 GiBps | 1-5 GiBps |
| **Regional + Provisioned** | 250µs | 2.7ms | 55K | 25K | 3-10 GiBps | 1-3.33 GiBps |
| **Regional + Bursting** | 250µs | 2.7ms | 35K | 7K | 3-5 GiBps | 1-3 GiBps |
| **One Zone + Any Mode** | 250µs | 1.6ms | 35K | 7K | 3 GiBps | 1 GiBps |

## Storage Classes Performance Impact

### EFS Standard (Recommended for Trading Data)
- **Technology**: Solid State Drive (SSD) storage
- **Read Latency**: As low as 250 microseconds  
- **Write Latency**: As low as 2.7 milliseconds
- **Use Case**: Frequently accessed trading data, genetic algorithm inputs

### EFS Infrequent Access (IA) & Archive
- **Latency**: Tens of milliseconds (first-byte)
- **Use Case**: Historical data archives, infrequently accessed backtesting data
- **Cost Optimization**: Automatic lifecycle policies available

## Phase 4 Performance Optimization Strategy

### Recommended Configuration for Ray Workers

**File System Setup:**
```bash
# Optimal configuration for Phase 4 quantitative trading
aws efs create-file-system \
    --creation-token "phase4-ray-workers-$(date +%s)" \
    --file-system-type REGIONAL \
    --performance-mode GENERAL_PURPOSE \
    --throughput-mode ELASTIC \
    --encrypted true \
    --tags Key=Project,Value=QuantTradingPhase4
```

**Mount Optimization:**
```bash
# High-performance mount options for Ray workers
sudo mount -t efs \
    -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600 \
    fs-xxxxxxxxx.efs.us-east-1.amazonaws.com:/ /shared_data
```

### Performance Monitoring

**Critical CloudWatch Metrics:**
- **PercentIOLimit**: Ensure Ray workers stay within IOPS limits
- **MeteredIOBytes**: Monitor actual throughput usage
- **ThroughputUtilization**: Track percentage of throughput capacity used

**Monitoring Setup:**
```bash
# Set up CloudWatch alarms for performance monitoring
aws cloudwatch put-metric-alarm \
    --alarm-name "EFS-High-IOLimit" \
    --alarm-description "EFS approaching IO limit" \
    --metric-name PercentIOLimit \
    --namespace AWS/EFS \
    --statistic Maximum \
    --period 300 \
    --threshold 80.0 \
    --comparison-operator GreaterThanThreshold
```

## Ray Worker Access Pattern Optimization

### Concurrent Access Patterns
```python
# Optimized Ray worker data access for EFS
@ray.remote
class OptimizedGeneticWorker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.shared_data_path = "/shared_data"
        
    def load_market_data(self, symbols: List[str]):
        # Parallel data loading optimized for EFS throughput
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._load_symbol_data, symbol) 
                for symbol in symbols
            ]
            return [f.result() for f in futures]
    
    def _load_symbol_data(self, symbol: str):
        # Optimized read pattern for EFS performance
        file_path = f"{self.shared_data_path}/market_data/{symbol}.parquet"
        # Use buffered I/O for better EFS performance
        return pd.read_parquet(file_path, use_threads=True)
```

### Write Pattern Optimization
```python
# Efficient write patterns for genetic algorithm results
async def store_generation_results(generation_data: Dict, generation_id: int):
    # Batch writes for better EFS throughput utilization
    write_tasks = []
    for worker_id, results in generation_data.items():
        file_path = f"/shared_data/results/gen_{generation_id}/worker_{worker_id}.json"
        write_tasks.append(write_results_async(file_path, results))
    
    # Parallel writes to maximize EFS IOPS utilization
    await asyncio.gather(*write_tasks)
```

## Performance Troubleshooting

### Common Performance Issues

**Issue 1: High Latency**
- **Cause**: Incorrect performance mode (Max I/O instead of General Purpose)
- **Solution**: Use General Purpose mode for all Phase 4 workloads

**Issue 2: IOPS Limiting**
- **Symptoms**: PercentIOLimit CloudWatch metric > 80%
- **Solution**: Optimize file access patterns, consider read-heavy workloads

**Issue 3: Throughput Bottlenecks**
- **Symptoms**: ThroughputUtilization consistently high
- **Solution**: Switch from Bursting to Elastic throughput mode

### Performance Optimization Checklist

- ✅ **Regional File System**: Multi-AZ redundancy and higher performance
- ✅ **General Purpose Mode**: Lowest latency for trading applications
- ✅ **Elastic Throughput**: Automatic scaling for Ray worker bursts
- ✅ **EFS Client v2.0+**: Up to 1,500 MiBps per-client throughput
- ✅ **Optimized Mount Options**: rsize/wsize=1048576 for maximum throughput
- ✅ **CloudWatch Monitoring**: Track IOPS and throughput utilization
- ✅ **Parallel Access**: Multiple threads/processes for better IOPS utilization
- ✅ **Read-Heavy Optimization**: Leverage 3:1 read IOPS advantage

## Phase 4 Performance Expectations

**Small Ray Cluster (5-10 workers):**
- **Configuration**: Regional + Elastic + General Purpose
- **Expected Throughput**: 2-5 GiBps aggregate read, 1-2 GiBps write
- **Latency**: Sub-millisecond reads, ~3ms writes
- **IOPS**: 50K-100K mixed operations

**Medium Ray Cluster (10-50 workers):**
- **Configuration**: Regional + Elastic + General Purpose  
- **Expected Throughput**: 10-20 GiBps aggregate read, 2-4 GiBps write
- **Scaling**: Automatic with Elastic throughput
- **Monitoring**: Critical for cost and performance optimization

**Large Ray Cluster (50+ workers):**
- **Considerations**: May require custom performance tuning
- **Alternative**: Consider Neon database integration for better scalability
- **Monitoring**: Essential for identifying bottlenecks

## Cost-Performance Trade-offs

**Elastic vs Provisioned:**
- **Elastic**: Better for unpredictable genetic algorithm workloads
- **Provisioned**: More cost-effective for consistent, high-throughput usage

**Regional vs One Zone:**
- **Regional**: Better performance and resilience, higher cost
- **One Zone**: Lower cost, adequate for development/testing

**Storage Classes:**
- **Standard**: High performance, higher cost
- **IA/Archive**: Lower performance, significant cost savings for historical data

**Next Steps**: Proceed to [Security Considerations](./05_security_considerations.md) for compliance requirements, then [Resource Management](./06_creating_managing_resources.md) for operational procedures.