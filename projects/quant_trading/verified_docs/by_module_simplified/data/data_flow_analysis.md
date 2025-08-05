# Data Module Data Flow Analysis
**Auto-generated from code verification on 2025-01-12**

## Module Overview
**Target Module**: `/workspaces/context-engineering-intro/projects/quant_trading/src/data`
**Purpose**: Real-time market data acquisition, processing, storage, and historical analysis
**Architecture**: Producer-consumer pipeline with dual storage (DuckDB + Parquet)

---

## Primary Data Flow Patterns

### 1. Real-Time Market Data Pipeline

```
External Data Sources → Clients → Pipeline → Aggregation → Storage
```

#### Flow Details:
**Input Sources:**
- Hyperliquid WebSocket API (real-time trades, L2 book, all mids)
- Alternative.me Fear & Greed Index API (market sentiment)
- AWS S3 Historical Archive (L2 snapshots, trade data)

**Processing Stages:**
1. **Data Ingestion** (`HyperliquidClient`, `FearGreedClient`)
   - WebSocket connection management with auto-reconnection
   - REST API calls with rate limiting (1200 req/min)
   - JSON message parsing with orjson for performance

2. **Real-Time Aggregation** (`MarketDataPipeline`)
   - AsyncIO producer-consumer queues (10,000+ msg/sec capacity)
   - Tick-to-OHLCV aggregation with time-based bucketing
   - Circuit breaker patterns for error handling

3. **Data Storage** (`DataStorage`)
   - Dual storage: DuckDB (fast queries) + Parquet (compression)
   - Thread-safe connection management
   - Zero-copy DataFrame integration with PyArrow

**Output Destinations:**
- DuckDB database for real-time queries
- Partitioned Parquet files for long-term storage
- Callback systems for real-time strategy feeds

---

### 2. Historical Data Collection Pipeline

```
S3 Archive → LZ4 Decompression → L2 Reconstruction → OHLCV Candles
```

#### Flow Details:
**S3 Historical Loader** (`S3HistoricalDataLoader`):
- Bucket structure: `hyperliquid-archive/market_data/[date]/[hour]/l2Book/[coin].lz4`
- Cost-optimized requester-pays access with local caching
- LZ4 decompression for compressed L2 snapshots
- OHLCV reconstruction from L2 book mid-prices

**Data Transformation:**
- L2 snapshots → Mid-price calculation → Time-bucketed OHLCV
- Volume reconstruction requires separate trade data
- Timestamp alignment across multiple timeframes

---

### 3. Dynamic Asset Data Collection

```
Asset Discovery → Tradeable Filtering → Multi-Timeframe Collection → Validation
```

#### Flow Details:
**Integration Pipeline** (`DynamicAssetDataCollector`):
- Enhanced asset filter integration for discovery
- API-only strategy (no S3 dependencies for production)
- Multi-timeframe collection: 5000 bars per timeframe
  - 1h: ~208 days of data
  - 15m: ~52 days of data

**Batch Processing:**
- Concurrent collection with rate limiting
- Memory-efficient batch processing (10 assets at a time)
- Data quality scoring and validation

---

## Data Transformation Points

### 1. WebSocket Message → TickData
**Location**: `MarketDataPipeline._parse_trade_message`
**Input**: Raw JSON WebSocket message
**Output**: Validated `TickData` object
**Transformations**:
- Timestamp conversion (milliseconds to datetime UTC)
- Price/volume type conversion to float
- Side normalization ('B'/'A' to 'buy'/'sell')

### 2. TickData → OHLCVBar
**Location**: `MarketDataAggregator.process_tick`
**Input**: Individual tick with price, volume, timestamp
**Output**: Completed OHLCV bar when time interval concludes
**Transformations**:
- Time-based bucketing with precise timestamp truncation
- Running OHLCV calculations (open, high, low, close)
- Volume-weighted average price (VWAP) computation
- Trade count accumulation

### 3. OHLCVBar → Database Storage
**Location**: `DataStorage.store_ohlcv_bars`
**Input**: List of OHLCV bar objects
**Output**: Dual storage in DuckDB + Parquet
**Transformations**:
- DataFrame conversion with proper schema
- DuckDB INSERT OR REPLACE for upserts
- Parquet partitioning by symbol and date
- Compression with Snappy codec

### 4. Raw API Response → FearGreedData
**Location**: `FearGreedClient.get_current_index`
**Input**: JSON API response from Alternative.me
**Output**: Validated `FearGreedData` with regime classification
**Transformations**:
- Pydantic validation with automatic field computation
- Market regime classification (extreme_fear to extreme_greed)
- Trading signal derivation (contrarian approach)
- Genetic algorithm pressure parameter calculation

---

## Data Storage Architecture

### DuckDB Analytical Database
**Tables**:
- `ohlcv_bars`: Real-time OHLCV data with time-series indexes
- `tick_data`: High-frequency tick data with date partitioning
- `strategy_performance`: Trading strategy execution tracking

**Optimizations**:
- Indexes on (symbol, timestamp DESC) for time-series queries
- Window functions for technical indicator calculations
- Larger-than-memory processing with automatic spilling

### Parquet Data Lake
**Structure**:
```
parquet_root/
├── ohlcv/
│   ├── symbol=BTC/
│   │   ├── date=2025-01-12/
│   │   │   └── data_1736708123.parquet
│   │   └── date=2025-01-13/
│   └── symbol=ETH/
└── tick_data/
```

**Features**:
- 5-10x compression with Snappy/ZSTD codecs
- Schema evolution support
- Zero-copy integration with DuckDB

---

## Error Handling and Data Quality

### Circuit Breaker Patterns
**MarketDataPipeline**:
- Maximum queue size enforcement (10,000 messages)
- Error rate monitoring (5% threshold triggers circuit breaker)
- Exponential backoff for WebSocket reconnection

### Data Validation
**Input Validation**:
- JSON schema validation for API responses
- Pydantic models for type safety and automatic conversion
- Range validation for prices and volumes

**Output Validation**:
- OHLCV consistency checks (high >= low, etc.)
- Timestamp alignment validation across timeframes
- Data completeness scoring for genetic evolution readiness

### Rate Limiting Integration
**API Compliance**:
- Hyperliquid: 1200 requests/minute for REST, 100 WebSocket connections
- Alternative.me: No documented limits, conservative 1 req/sec
- AWS S3: Requester-pays with cost optimization

---

## Integration Points

### External System Dependencies
1. **Hyperliquid Exchange API**
   - REST: Market data, account information, asset contexts
   - WebSocket: Real-time trades, L2 book updates, all mids

2. **Alternative.me Fear & Greed API**
   - Current index with market regime classification
   - Historical data for trend analysis

3. **AWS S3 Archive**
   - Historical L2 book snapshots (LZ4 compressed)
   - Trade execution data for volume validation

### Internal System Integration
1. **Configuration System** (`src.config.settings`)
   - Environment-aware configuration (testnet/mainnet)
   - Rate limiting parameters and database paths

2. **Discovery System** (`src.discovery.enhanced_asset_filter`)
   - Asset universe filtering for tradeable assets
   - Rate limiter sharing for efficient API usage

3. **Genetic Evolution System** (future integration)
   - Multi-timeframe data provision
   - Environmental pressure parameters from market sentiment

---

## Performance Characteristics

### Throughput Metrics
- **WebSocket Processing**: 10,000+ messages/second capacity
- **Database Writes**: Batch processing with thread pool execution
- **Memory Usage**: Zero-copy DataFrame operations where possible
- **Compression**: 5-10x reduction with Parquet storage

### Latency Optimization
- **Real-Time Path**: WebSocket → Queue → Aggregation → Storage (~1-10ms)
- **Historical Path**: S3 → Cache → Decompression → Processing (~100-500ms)
- **Query Path**: DuckDB indexes enable sub-millisecond OHLCV retrieval

### Scalability Considerations
- **Horizontal**: Multiple pipeline instances with shared storage
- **Vertical**: Thread-safe connection pooling and concurrent processing
- **Storage**: Automatic partitioning and compression for growth

---

**🎯 Data Flow Analysis Complete**: Comprehensive mapping of real-time and historical data flows with detailed transformation points, storage architecture, and integration patterns across the quantitative trading data infrastructure.