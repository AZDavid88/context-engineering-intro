# Hyperliquid WebSocket API Documentation

## Source
- **URLs**: 
  - https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket
  - https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## WebSocket Overview

WebSocket endpoints provide real-time data streaming and alternative request sending for optimal latency trading applications.

### Connection URLs
- **Mainnet**: `wss://api.hyperliquid.xyz/ws`
- **Testnet**: `wss://api.hyperliquid-testnet.xyz/ws`

### Connection Example
```bash
$ wscat -c wss://api.hyperliquid.xyz/ws
Connected (press CTRL+C to quit)
> { "method": "subscribe", "subscription": { "type": "trades", "coin": "SOL" } }
< {"channel":"subscriptionResponse","data":{"method":"subscribe","subscription":{"type":"trades","coin":"SOL"}}}
```

## Critical Subscriptions for Quant Trading

### 1. All Mid Prices
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "allMids",
    "dex": ""  // Optional: perp dex name
  }
}
```
**Data Format**: `{"mids": {"BTC": "29792.0", "ETH": "1891.4", ...}}`
**Use Case**: Real-time price feeds for all markets simultaneously

### 2. L2 Order Book
```json
{
  "method": "subscribe", 
  "subscription": {
    "type": "l2Book",
    "coin": "BTC",
    "nSigFigs": 5,  // Optional: aggregation level
    "mantissa": 1   // Optional: when nSigFigs is 5
  }
}
```
**Data Format**:
```typescript
interface WsBook {
  coin: string;
  levels: [Array<WsLevel>, Array<WsLevel>]; // [bids, asks]
  time: number;
}
interface WsLevel {
  px: string;  // price
  sz: string;  // size  
  n: number;   // number of orders
}
```
**Use Case**: Real-time order book depth for strategy execution decisions

### 3. Trade Feed
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "trades", 
    "coin": "BTC"
  }
}
```
**Data Format**:
```typescript
interface WsTrade {
  coin: string;
  side: string;    // "A" (ask/sell) or "B" (bid/buy)
  px: string;      // price
  sz: string;      // size
  hash: string;    // transaction hash
  time: number;    // timestamp
  tid: number;     // trade ID (50-bit hash)
  users: [string, string]; // [buyer, seller]
}
```
**Use Case**: Real-time trade execution data for market analysis

### 4. Candle Data
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "candle",
    "coin": "BTC", 
    "interval": "1m"  // "1m"|"3m"|"5m"|"15m"|"30m"|"1h"|"2h"|"4h"|"8h"|"12h"|"1d"|"3d"|"1w"|"1M"
  }
}
```
**Data Format**:
```typescript
interface Candle {
  t: number;  // open time millis
  T: number;  // close time millis  
  s: string;  // coin symbol
  i: string;  // interval
  o: number;  // open price
  c: number;  // close price
  h: number;  // high price
  l: number;  // low price
  v: number;  // volume (base unit)
  n: number;  // number of trades
}
```
**Use Case**: Real-time OHLCV data for technical indicators

## User-Specific Subscriptions

### 5. User Order Updates
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "orderUpdates",
    "user": "0x..."
  }
}
```
**Data Format**:
```typescript
interface WsOrder {
  order: WsBasicOrder;
  status: string;        // "open"|"filled"|"canceled"|"triggered"|etc.
  statusTimestamp: number;
}
```
**Use Case**: Real-time order status monitoring for strategy execution

### 6. User Fill Events
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "userFills",
    "user": "0x...",
    "aggregateByTime": false  // Optional: combine partial fills
  }
}
```
**Data Format**:
```typescript
interface WsUserFills {
  isSnapshot?: boolean;  // First message is snapshot
  user: string;
  fills: Array<WsFill>;
}

interface WsFill {
  coin: string;
  px: string;           // fill price
  sz: string;           // fill size
  side: string;         // "A"|"B"
  time: number;
  startPosition: string;  // position before fill
  dir: string;          // "Open Long"|"Close Short"|etc.
  closedPnl: string;    // realized PnL
  hash: string;         // transaction hash
  oid: number;          // order ID
  crossed: boolean;     // was taker order
  fee: string;          // fee paid (negative = rebate)
  tid: number;          // trade ID
  feeToken: string;     // fee currency
  builderFee?: string;  // builder fee if applicable
}
```
**Use Case**: Real-time trade execution tracking and P&L monitoring

### 7. User Events (Comprehensive)
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "userEvents", 
    "user": "0x..."
  }
}
```
**Data Format**: Union of fills, funding, liquidations, and cancellations
**Use Case**: Complete user activity monitoring

## Best-Bid-Offer (BBO) Feed
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "bbo",
    "coin": "BTC"
  }
}
```
**Data Format**:
```typescript
interface WsBbo {
  coin: string;
  time: number;
  bbo: [WsLevel | null, WsLevel | null]; // [best_bid, best_ask]
}
```
**Use Case**: Ultra-low latency top-of-book updates for high-frequency strategies

## WebSocket Rate Limits

### Connection Limits (Per IP)
- **Max Connections**: 100 WebSocket connections
- **Max Subscriptions**: 1000 total subscriptions
- **Max Users**: 10 unique users across user-specific subscriptions
- **Message Rate**: 2000 messages/minute across all connections
- **Inflight Posts**: 100 simultaneous post messages

## Implementation Guidelines for Genetic Algorithm System

### 1. Essential Feeds for Strategy Execution
```python
# Critical subscriptions for automated trading
essential_feeds = [
    {"type": "allMids"},                    # Price discovery
    {"type": "userFills", "user": address}, # Trade execution monitoring  
    {"type": "orderUpdates", "user": address}, # Order status tracking
    {"type": "userEvents", "user": address}     # Complete activity feed
]
```

### 2. Market Data for Strategy Development
```python
# Market analysis feeds
market_feeds = [
    {"type": "l2Book", "coin": "BTC"},      # Order book depth
    {"type": "trades", "coin": "BTC"},      # Trade flow analysis
    {"type": "candle", "coin": "BTC", "interval": "1m"}, # Technical analysis
    {"type": "bbo", "coin": "BTC"}          # High-frequency price updates
]
```

### 3. Connection Management
- **Reconnection Logic**: Implement automatic reconnection with exponential backoff
- **Subscription Recovery**: Re-subscribe to all feeds after reconnection
- **Heartbeat Monitoring**: Track connection health and latency
- **Error Handling**: Process WebSocket errors and connection drops gracefully

### 4. Data Processing Pipeline
- **Snapshot Handling**: First message has `isSnapshot: true`, can be ignored if already processed
- **Message Ordering**: Process messages in sequence to maintain data consistency  
- **Buffer Management**: Handle burst message rates during high volatility
- **Latency Optimization**: Process critical feeds (BBO, fills) with minimal latency

## Critical for Real-Time Strategy Execution

The WebSocket API provides the real-time data foundation for:
1. **Strategy Decision Making** - Live price feeds and market depth
2. **Execution Monitoring** - Real-time order and fill tracking
3. **Risk Management** - Immediate notification of position changes
4. **Performance Analysis** - Live P&L and trade execution metrics

This real-time infrastructure is essential for implementing the genetic algorithm trading system's execution engine and performance monitoring described in the planning PRP.