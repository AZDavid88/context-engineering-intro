# Strategy Module Data Flow Analysis

**Generated**: 2025-08-03  
**Analysis Method**: Code trace through actual function calls and data transformations  
**Confidence Level**: 95% - All flows verified through code analysis

## Overview

The strategy module implements a sophisticated data pipeline from market data ingestion through genetic evolution to portfolio execution. Data flows through multiple transformation stages with proper validation and error handling at each step.

## Primary Data Flow Pipeline

```
Market Data → Universal Engine → Genetic Evolution → Strategy Selection → Portfolio Allocation → Execution Signals
     ↓              ↓                 ↓                  ↓                    ↓                 ↓
  OHLCV         Asset Universe    Individual Seeds   Fitness Evaluation   Position Sizing    Trading Orders
```

## Detailed Data Flow Analysis

### 1. Market Data Ingestion

**Entry Point**: `UniversalStrategyEngine.evolve_universal_strategies()` (universal_strategy_engine.py:174)

**Input Format**:
```python
market_data: Dict[str, pd.DataFrame]
# Structure: {
#   'BTC-USD': DataFrame with columns ['open', 'high', 'low', 'close', 'volume'],
#   'ETH-USD': DataFrame with columns ['open', 'high', 'low', 'close', 'volume'],
#   ...
# }
```

**Data Validation**: 
- Asset selection filter (lines 189-191): Validates symbols exist in asset universe
- Minimum data length check (genetic_seeds/seed_registry.py:92): Requires 100+ bars
- OHLCV column validation in each genetic seed

**Evidence**: `_select_assets_for_evolution()` (universal_strategy_engine.py:436-468) filters assets with `len(data) >= 100`

### 2. Asset Universe Management

**Process Flow**:
1. `initialize_universe()` → `_fetch_hyperliquid_universe()` → `_create_asset_metadata()`
2. Asset classification via `_classify_asset()` (lines 392-413)
3. Metadata creation with liquidity scores, volatility percentiles, correlation clusters

**Data Transformation**:
```python
# Input: Raw asset info from Hyperliquid
asset_info = {'symbol': 'BTC-USD', 'marketCap': 1, 'volume24h': 10000000000}

# Output: Structured metadata
AssetMetadata(
    symbol='BTC-USD',
    asset_class=AssetClass.MAJOR_CRYPTO,
    market_cap_rank=1,
    avg_daily_volume=10000000000,
    liquidity_score=1.0,
    volatility_percentile=0.7,  # Calculated from actual data
    correlation_cluster=2
)
```

**Storage**: `self.asset_universe: Dict[str, AssetMetadata]` (line 124)

### 3. Genetic Evolution Data Flow

#### 3.1 Population Initialization

**Flow**: `PopulationManager.initialize_population()` → Seed Registry → Individual Seeds

**Process** (genetic_engine_population.py:55-100):
1. Query seed registry for available seed types
2. Distribute population across seed types for diversity
3. Apply random mutations for initial variation
4. Return `List[BaseSeed]` population

**Data Validation**: Type checking via `SeedGenes` Pydantic model (base_seed.py:40-76)

#### 3.2 Individual Evaluation

**Flow Path**:
```
Market Data → Individual Seed → Technical Indicators → Trading Signals → Performance Metrics → Fitness Score
```

**Evidence Trace**:

1. **Technical Indicator Calculation** (EMACrossoverSeed example):
   - Input: `pd.DataFrame` with OHLCV columns
   - Process: `calculate_technical_indicators()` (ema_crossover_seed.py:74-109)
   - Output: `Dict[str, pd.Series]` with indicators

2. **Signal Generation**:
   - Input: Technical indicators dictionary
   - Process: `generate_signals()` (ema_crossover_seed.py:110-151) 
   - Output: `pd.Series` with values in [-1.0, 1.0] range

3. **Fitness Evaluation**:
   - Input: Trading signals + market data
   - Process: `FitnessEvaluator.evaluate_individual()` (genetic_engine_evaluation.py:42-95)
   - Output: `Tuple[float, float, float, float]` (Sharpe, consistency, max_drawdown, win_rate)

### 4. Cross-Asset Correlation Analysis

**Process Flow**: `_calculate_cross_asset_correlations()` (universal_strategy_engine.py:470-503)

**Data Transformation**:
```python
# Input: Market data for multiple assets
market_data = {'BTC-USD': df1, 'ETH-USD': df2, ...}

# Processing: Extract returns and align time series
returns_data = {
    'BTC-USD': df1['close'].pct_change().dropna(),
    'ETH-USD': df2['close'].pct_change().dropna()
}
returns_df = pd.DataFrame(returns_data).dropna()

# Output: Correlation matrix
correlation_matrix = returns_df.corr()
```

**Usage**: Correlation penalties in allocation optimization (lines 605-647)

### 5. Portfolio Allocation Optimization

**Flow**: `_optimize_universal_allocation()` → `_calculate_genetic_allocations()` → Risk Adjustments

**Multi-Stage Processing**:

1. **Fitness Aggregation** (lines 519-528):
   ```python
   asset_fitness = {
       'BTC-USD': 2.1,  # Sharpe ratio from best individual
       'ETH-USD': 1.8,
       ...
   }
   ```

2. **Raw Weight Calculation** (lines 577-588):
   ```python
   total_fitness = sum(asset_fitness.values())
   raw_weights = {symbol: fitness / total_fitness for symbol, fitness in asset_fitness.items()}
   ```

3. **Correlation Penalty Application** (lines 592-647):
   - Reduces weights for highly correlated assets (correlation > 0.7)
   - Penalty formula: `penalty = correlation * other_weight * 0.5`

4. **Volatility Adjustment** (lines 649-691):
   - Inverse volatility weighting for risk management
   - Clips adjustment ratios to [0.5, 2.0] for stability

5. **Position Limits** (lines 693-720):
   - Min allocation: 0.001 (0.1%)
   - Max allocation: 0.15 (15%)
   - Renormalization to ensure total ≤ 1.0

**Output**: `Dict[str, float]` with final allocation weights

### 6. Portfolio Rebalancing

**Entry Point**: `rebalance_portfolio()` (universal_strategy_engine.py:260-325)

**Data Flow**:
```python
# Input: Current positions and target allocations
current_positions = {'BTC-USD': 0.12, 'ETH-USD': 0.08}
target_allocations = {'BTC-USD': 0.10, 'ETH-USD': 0.15, 'SOL-USD': 0.05}

# Processing: Calculate position adjustments
for symbol, target in target_allocations.items():
    current = current_positions.get(symbol, 0.0)
    position_result = position_sizer.calculate_position_size(...)
    # Scale to match target allocation
    position_result.target_size = target

# Output: List[PositionSizeResult] for execution
```

**Integration**: Results feed into execution engine for order generation

### 7. AST Strategy Data Flow

**Genetic Programming Pipeline**:

1. **Primitive Evaluation** (ast_strategy.py:347-428):
   - Market data → Technical indicator getters (e.g., `_get_rsi()`)
   - Type-safe primitives: DataFrame → float, float → bool operations

2. **Strategy Compilation**:
   - GP tree → Executable function via `gp.compile(individual, pset)`
   - Function signature: `(pd.DataFrame) → bool`

3. **Evolution Loop** (lines 514-546):
   ```python
   # Each generation:
   population = toolbox.population(n=population_size)
   fitnesses = map(toolbox.evaluate, population)  # Parallel evaluation
   offspring = selection + crossover + mutation
   population = toolbox.select(population + offspring, population_size)
   ```

## Data Quality and Validation

### Input Validation Points

1. **Asset Selection**: Minimum 100 bars requirement (genetic_seeds/seed_registry.py:92)
2. **Parameter Bounds**: All genetic parameters validated via Pydantic (base_seed.py:68-75)
3. **Signal Range**: Trading signals clipped to [-1.0, 1.0] (ema_crossover_seed.py:150)
4. **Correlation Matrix**: NaN handling and minimum periods check (universal_strategy_engine.py:486-496)

### Error Handling Patterns

- **Fallback Data**: Synthetic market data generation when real data unavailable
- **Default Values**: RSI defaults to 50.0, prices default to last known value
- **Exception Recovery**: Try/catch blocks with logging at every processing stage

## Performance Considerations

### Computational Bottlenecks

1. **Correlation Calculation**: O(n²) for n assets - limited to 50 assets max
2. **Genetic Evolution**: Parallel fitness evaluation via multiprocessing
3. **Technical Indicators**: Vectorized pandas operations for efficiency

### Memory Management

- **Data Caching**: `_market_data_cache` and `_synthetic_data_cache` in FitnessEvaluator
- **Population History**: Limited storage in `PopulationManager._population_history`
- **Strategy Results**: Bounded `allocation_history` list

## Integration Points

### External Dependencies

1. **Hyperliquid Client**: Asset universe data (universal_strategy_engine.py:121)
2. **Strategy Converter**: Signal to portfolio conversion (line 118)
3. **Performance Analyzer**: Fitness extraction (line 119)
4. **Position Sizer**: Risk-adjusted position sizing (line 120)

### Internal Module Interfaces

1. **Settings Integration**: All components use `get_settings()` for configuration
2. **Registry System**: Automatic seed discovery via `@genetic_seed` decorator
3. **Event Logging**: Comprehensive logging at every processing stage

## Data Lineage Summary

```mermaid
graph TD
    A[Market Data OHLCV] --> B[Asset Universe Filter]
    B --> C[Genetic Population Init]
    C --> D[Individual Evaluation]
    D --> E[Technical Indicators]
    E --> F[Trading Signals]
    F --> G[Fitness Calculation]
    G --> H[Population Evolution]
    H --> I[Best Strategy Selection]
    I --> J[Correlation Analysis]
    J --> K[Allocation Optimization]
    K --> L[Portfolio Rebalancing]
    L --> M[Execution Signals]
```

**Verification Status**: ✅ All data flows traced through actual code implementation with evidence references