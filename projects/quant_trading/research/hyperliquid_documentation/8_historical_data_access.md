# Hyperliquid Historical Data Access

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data  
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## Historical Data Overview

Historical data is available through AWS S3 buckets with requester-pays model. The data supports backtesting and strategy development for the genetic algorithm trading system.

## Data Sources and Access

### AWS S3 Bucket Structure
**Main Bucket**: `hyperliquid-archive`
**Cost**: Requester pays for transfer costs
**Update Frequency**: Approximately once per month

### Required Tools
- **AWS CLI**: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- **LZ4 Compression**: https://github.com/lz4/lz4 (or install from package manager)

## Asset Data (Market Data)

### L2 Book Snapshots  
**Path Format**: `s3://hyperliquid-archive/market_data/[date]/[hour]/[datatype]/[coin].lz4`

**Example Access**:
```bash
# Download L2 book data for SOL on Sept 16, 2023 at 9 AM
aws s3 cp s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4 /tmp/SOL.lz4 --request-payer requester

# Decompress and view
unlz4 --rm /tmp/SOL.lz4
head /tmp/SOL
```

**Data Contains**:
- Order book snapshots at regular intervals
- Bid/ask levels with price, size, and order count
- Timestamp information for each snapshot

### Asset Contexts
**Path Format**: `s3://hyperliquid-archive/asset_ctxs/[date].csv.lz4`

**Data Contains**:
- Asset metadata and market context
- Funding rates for perpetuals
- Open interest data
- Oracle prices and mark prices

## Trade Data

### Modern Trade Data (Recommended)
**Path**: `s3://hl-mainnet-node-data/node_fills`
**Format**: Matches API format exactly
**Source**: Streamed via `--write-fills` from non-validating node

**Data Contains**:
- Individual trade executions
- Fill prices and sizes  
- Trade timestamps and IDs
- Buyer/seller information
- Fee information

### Legacy Trade Data
**Path**: `s3://hl-mainnet-node-data/node_trades`
**Format**: Different format from API (legacy)
**Note**: Older data in different format, use `node_fills` for consistency

## Historical Node Data

### Explorer Blocks
**Path**: `s3://hl-mainnet-node-data/explorer_blocks`
**Data Contains**:
- Complete block information
- Transaction details within blocks
- Block timestamps and hashes

### L1 Transactions  
**Path**: `s3://hl-mainnet-node-data/replica_cmds`
**Data Contains**:
- Raw L1 transaction data
- Command execution details
- State change information

## Data Limitations and Alternatives

### Not Available via S3
- **Candle Data**: Use API to record candle data yourself
- **Spot Asset Data**: Limited spot historical data
- **Real-time Feeds**: Use WebSocket API for live data collection

### API-Based Historical Data
For data not in S3, use API endpoints:
- **Candle Snapshot**: Get OHLCV data via API
- **User Fills**: Historical trade data for accounts
- **Asset Contexts**: Real-time market context

## Implementation for Genetic Algorithm System

### 1. Backtesting Data Pipeline
```python
class HistoricalDataLoader:
    def __init__(self, aws_profile=None):
        self.s3_client = boto3.client('s3', 
                                     aws_profile=aws_profile)
        self.bucket = 'hyperliquid-archive'
        
    def load_l2_book_data(self, date, hour, coin):
        """Load L2 book snapshots for backtesting"""
        key = f"market_data/{date}/{hour}/l2Book/{coin}.lz4"
        
        # Download and decompress
        local_path = f"/tmp/{coin}_{date}_{hour}.lz4"
        self.s3_client.download_file(
            self.bucket, key, local_path,
            ExtraArgs={'RequestPayer': 'requester'}
        )
        
        # Decompress LZ4
        with lz4.frame.LZ4FrameFile(local_path, 'r') as f:
            data = f.read()
            
        return self.parse_l2_data(data)
        
    def load_trade_data(self, date_range):
        """Load trade execution data"""
        bucket = 'hl-mainnet-node-data'
        prefix = 'node_fills'
        
        # Get files for date range
        files = self.list_s3_files(bucket, prefix, date_range)
        
        trades = []
        for file in files:
            trade_data = self.download_and_parse(bucket, file)
            trades.extend(trade_data)
            
        return trades
```

### 2. Strategy Validation Pipeline
```python
class BacktestEngine:
    def __init__(self, historical_data_loader):
        self.data_loader = historical_data_loader
        
    def validate_strategy(self, strategy, start_date, end_date):
        """Multi-stage validation using historical data"""
        
        # Stage 1: In-sample optimization (60% of data)
        in_sample_data = self.load_data_range(
            start_date, 
            start_date + (end_date - start_date) * 0.6
        )
        optimized_params = strategy.optimize(in_sample_data)
        
        # Stage 2: Out-of-sample testing (20% of data)
        oos_start = start_date + (end_date - start_date) * 0.6
        oos_end = start_date + (end_date - start_date) * 0.8
        oos_data = self.load_data_range(oos_start, oos_end)
        oos_performance = strategy.test(oos_data, optimized_params)
        
        # Stage 3: Walk-forward analysis (20% of data)
        wf_start = start_date + (end_date - start_date) * 0.8
        wf_data = self.load_data_range(wf_start, end_date)
        wf_performance = strategy.walk_forward_test(wf_data, optimized_params)
        
        return {
            'in_sample': strategy.performance,
            'out_of_sample': oos_performance,
            'walk_forward': wf_performance
        }
```

### 3. Data Quality Assessment
```python
class DataQualityChecker:
    def assess_data_completeness(self, date_range, assets):
        """Check data availability for backtesting period"""
        missing_data = []
        
        for date in date_range:
            for hour in range(24):
                for asset in assets:
                    key = f"market_data/{date}/{hour}/l2Book/{asset}.lz4"
                    if not self.s3_key_exists(key):
                        missing_data.append((date, hour, asset))
                        
        return missing_data
        
    def validate_data_integrity(self, data):
        """Validate downloaded data for consistency"""
        checks = {
            'timestamps_monotonic': self.check_timestamps(data),
            'price_sanity': self.check_price_ranges(data),
            'volume_consistency': self.check_volumes(data)
        }
        
        return all(checks.values()), checks
```

### 4. Cost Optimization
```python
class CostOptimizer:
    def estimate_download_costs(self, data_requirements):
        """Estimate AWS data transfer costs"""
        # Typical file sizes and transfer costs
        avg_l2_file_size_mb = 50
        avg_trade_file_size_mb = 100
        aws_transfer_cost_per_gb = 0.09  # USD
        
        total_size_gb = (
            data_requirements['l2_book_files'] * avg_l2_file_size_mb +
            data_requirements['trade_files'] * avg_trade_file_size_mb
        ) / 1024
        
        estimated_cost = total_size_gb * aws_transfer_cost_per_gb
        return estimated_cost
        
    def optimize_download_strategy(self, backtesting_period):
        """Determine most cost-effective data download approach"""
        # Compare costs: bulk download vs API historical queries
        bulk_cost = self.estimate_download_costs(backtesting_period)
        api_cost = self.estimate_api_costs(backtesting_period)
        
        return 'bulk' if bulk_cost < api_cost else 'api'
```

## Strategic Considerations

### 1. Data Coverage Assessment
Before implementing strategies:
- **Check Data Availability**: Verify historical data exists for backtesting period
- **Asset Coverage**: Ensure all target assets have sufficient historical data
- **Time Resolution**: Confirm data granularity meets strategy requirements

### 2. Cost Management
- **Selective Downloads**: Only download data needed for specific strategies
- **Local Caching**: Store frequently accessed data locally to avoid re-downloads
- **Bulk Processing**: Download larger date ranges to minimize per-request costs

### 3. Integration with Live Trading
- **Data Consistency**: Ensure historical data format matches live API data
- **Strategy Validation**: Use historical data to validate strategies before live deployment
- **Performance Benchmarking**: Compare live performance against historical backtests

This historical data access framework provides the foundation for implementing the comprehensive backtesting and validation pipeline described in the planning PRP, enabling rigorous strategy testing before live deployment.