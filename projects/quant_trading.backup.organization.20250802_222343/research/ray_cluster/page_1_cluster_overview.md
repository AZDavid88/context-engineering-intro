# Ray Clusters Overview - Core Documentation

## Source URL
https://docs.ray.io/en/latest/cluster/getting-started.html

## Key Concepts for Distributed Genetic Algorithm Implementation

### Ray Cluster Architecture
- **Ray Cluster**: Set of worker nodes connected to a common Ray head node
- **Seamless Scaling**: Works from laptop to large cluster deployment
- **Autoscaling**: Clusters can scale up/down based on resource demands
- **Fixed or Dynamic**: Support for both fixed-size and autoscaling clusters

### Critical Implementation Points for Genetic Trading

#### 1. **Distributed Computing Model**
- **Ray Head Node**: Central coordinator for distributed tasks
- **Worker Nodes**: Execute distributed genetic evaluation tasks
- **Resource Management**: Automatic allocation based on workload demands

#### 2. **Deployment Options for Retail Trading**
Ray provides native cluster deployment on:
- **AWS and GCP**: Primary cloud deployment options
- **Kubernetes**: Via officially supported KubeRay project  
- **Anyscale**: Fully managed Ray platform (optimal for production)
- **Manual Deployment**: Advanced on-premises deployment

#### 3. **Cost-Efficient Architecture for Retail Trading**
**Recommendation**: Start with AWS/GCP for Phase 5 implementation
- **Autoscaling**: Only pay for resources during genetic evolution cycles
- **Resource Efficiency**: Workers spin up only during parameter evaluation
- **Cost Control**: Avoid idle cluster costs between trading sessions

### Ray Remote Functions Pattern (Critical for Genetic Evolution)

```python
# CORRECT Pattern for Distributed Genetic Evaluation
@ray.remote
def evaluate_strategy_parameters(seed_type: str, parameters: dict, market_data: dict) -> dict:
    """
    Stateless evaluation function for Ray workers.
    
    Args:
        seed_type: Strategy type (e.g., 'ema_crossover', 'rsi_mean_reversion')
        parameters: Genetic parameters to evaluate
        market_data: Historical market data for backtesting
        
    Returns:
        dict: Performance metrics (sharpe_ratio, total_return, max_drawdown)
    """
    # Import strategy locally (worker isolation)
    from strategy.genetic_seeds.seed_registry import create_seed_instance
    
    # Create strategy instance with parameters
    strategy = create_seed_instance(seed_type, parameters)
    
    # Run backtest evaluation
    results = strategy.backtest(market_data)
    
    return {
        'fitness': results.sharpe_ratio,
        'total_return': results.total_return,
        'max_drawdown': results.max_drawdown,
        'parameters': parameters
    }

# Usage in genetic evolution
def distributed_genetic_evaluation(population: List[dict]) -> List[dict]:
    """Evaluate entire population using Ray cluster."""
    
    # Submit all evaluations to Ray cluster
    futures = []
    for individual in population:
        future = evaluate_strategy_parameters.remote(
            individual['seed_type'],
            individual['parameters'], 
            market_data
        )
        futures.append(future)
    
    # Collect results from all workers
    results = ray.get(futures)
    return results
```

### Architecture Boundaries (Critical Understanding)

#### Local Components (Cannot Distribute):
- **Seed Registry**: Strategy discovery system
- **TradingSystemManager**: Live trading coordination
- **Connection Optimizer**: Real-time market connections
- **Order Management**: Live execution system

#### Distributable Components:
- **Strategy Parameter Evaluation**: Pure backtesting calculations
- **Genetic Algorithm Operations**: Population evolution, mutation, crossover
- **Performance Metrics Calculation**: Risk/return analysis
- **Market Data Processing**: Historical data analysis

### Resource Profile for Retail Trading
**Optimal Configuration**:
- **2-4GB RAM per worker**: Sufficient for strategy backtesting
- **1-2 CPU cores per worker**: Parallel genetic evaluation
- **10-50 workers**: Scale based on population size (100-1000 strategies)

**Cost Economics**:
- **Ray Cluster**: $7-20 per evolution cycle (2 hours)
- **Break-even Capital**: ~$647 (vs $30,417 for AWS Batch)
- **Autoscaling**: Workers only active during genetic evolution

### Platform Support
- **Multi-node Ray clusters**: Linux only (production requirement)
- **Development Testing**: Windows/OSX supported with `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1`

### Integration with Existing System

#### Phase 5A Implementation Strategy:
1. **Local Genetic Pool**: Initial implementation using existing BaseSeed architecture
2. **Ray Integration**: Add distributed evaluation capability 
3. **State Management**: Keep global state local, distribute pure functions
4. **Cloud Deployment**: Scale to Ray cluster for production evolution cycles

#### Critical Design Patterns:
- **Stateless Functions**: All Ray remote functions must be pure
- **Data Passing**: Market data shared via Ray object store
- **Result Aggregation**: Collect fitness scores for genetic selection
- **Fault Tolerance**: Handle worker failures gracefully

## Next Steps for Implementation

### Phase 5A: Local + Ray Ready
1. Implement `genetic_strategy_pool.py` with local-first approach
2. Add Ray remote function stubs for future distribution
3. Test 100-strategy evolution locally

### Phase 5B: Ray Cluster Integration  
1. Deploy Ray cluster with autoscaling
2. Migrate evaluation functions to Ray workers
3. Test distributed 1000-strategy evolution
4. Integrate with existing TradingSystemManager

This documentation provides the foundation for implementing distributed genetic algorithm evolution while maintaining the existing retail trading system architecture.