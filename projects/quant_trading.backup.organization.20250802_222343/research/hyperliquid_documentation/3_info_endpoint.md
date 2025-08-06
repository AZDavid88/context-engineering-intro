# Hyperliquid Info Endpoint Documentation

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## Info Endpoint Overview

The info endpoint is used to fetch information about the exchange and specific users. Different request bodies result in different corresponding response body schemas.

**Endpoint**: `POST https://api.hyperliquid.xyz/info`

## Critical Features for Quant Trading

### Market Data Retrieval

#### 1. Retrieve Mids for All Coins
```json
{
  "type": "allMids",
  "dex": "" // Optional: perp dex name, defaults to first perp dex
}
```
**Response**: Object with coin symbols as keys and mid prices as values
**Use Case**: Essential for real-time price tracking across all markets

#### 2. L2 Book Snapshot
```json
{
  "type": "l2Book",
  "coin": "BTC",
  "nSigFigs": 5, // Optional: aggregate to significant figures
  "mantissa": 1  // Optional: when nSigFigs is 5
}
```
**Response**: Bid/ask levels with price, size, and order count
**Use Case**: Order book analysis for strategy execution

#### 3. Candle Snapshot
```json
{
  "type": "candleSnapshot",
  "req": {
    "coin": "BTC",
    "interval": "15m",
    "startTime": 1681923600000,
    "endTime": 1681924499999
  }
}
```
**Supported Intervals**: "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"
**Use Case**: Historical price data for backtesting and technical analysis

### Portfolio & Position Management

#### 4. User's Open Orders
```json
{
  "type": "openOrders",
  "user": "0x...", // 42-character hex address
  "dex": ""        // Optional: perp dex name
}
```
**Response**: Array of open orders with details
**Use Case**: Strategy lifecycle management and position tracking

#### 5. User's Fills
```json
{
  "type": "userFills",
  "user": "0x...",
  "aggregateByTime": false // Optional: combine partial fills
}
```
**Response**: Up to 2000 most recent fills
**Use Case**: Trade execution analysis and performance tracking

#### 6. User's Fills by Time Range
```json
{
  "type": "userFillsByTime",
  "user": "0x...",
  "startTime": 1681222254710, // milliseconds, inclusive
  "endTime": 1681222354710,   // milliseconds, inclusive
  "aggregateByTime": false    // Optional
}
```
**Response**: Up to 2000 fills in time range (max 10,000 most recent available)
**Use Case**: Historical performance analysis for strategy evaluation

### Risk Management & Monitoring

#### 7. Query User Rate Limits
```json
{
  "type": "userRateLimit",
  "user": "0x..."
}
```
**Response**: Current rate limit usage and capacity
**Use Case**: Ensure trading system stays within API limits

#### 8. Query Order Status
```json
{
  "type": "orderStatus",
  "user": "0x...",
  "oid": 12345 // order ID or client order ID
}
```
**Response**: Detailed order status and history
**Use Case**: Order execution monitoring and error handling

## Implementation Considerations

### Pagination
- Responses with time ranges return max 500 elements
- Use last returned timestamp as next `startTime` for pagination

### Asset Identification
- **Perpetuals**: Use coin name from `meta` response (e.g., "BTC")
- **Spot**: Use format like "PURR/USDC" or "@{index}" for other tokens

### User Address Requirements
- Must use actual address of master/sub-account, not agent wallet address
- Agent wallet addresses lead to empty results

### Rate Limiting
- Info requests have specific weights (2-60 depending on request type)
- `allMids`, `l2Book`, `clearinghouseState` etc. have weight 2
- Most user-specific requests have weight 20

## Critical for Genetic Algorithm Implementation

These endpoints provide the essential data feeds for:
1. **Real-time Price Discovery** - `allMids` for current market prices
2. **Strategy Execution Monitoring** - `openOrders` and `userFills` for trade tracking
3. **Historical Analysis** - `candleSnapshot` and `userFillsByTime` for backtesting
4. **Risk Management** - `userRateLimit` and `orderStatus` for system health monitoring

The info endpoint serves as the data foundation for implementing the multi-stage validation pipeline and strategy performance evaluation described in the planning PRP.