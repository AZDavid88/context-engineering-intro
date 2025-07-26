# Hyperliquid Exchange Endpoint Documentation

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## Exchange Endpoint Overview

The exchange endpoint is used to interact with and trade on the Hyperliquid chain. See the Python SDK for code to generate signatures for these requests.

**Endpoint**: `POST https://api.hyperliquid.xyz/exchange`

## Core Trading Operations

### 1. Place an Order
```json
{
  "action": {
    "type": "order",
    "orders": [{
      "a": 0,           // asset (index in universe)
      "b": true,        // isBuy
      "p": "29792.0",   // price
      "s": "0.01",      // size
      "r": false,       // reduceOnly
      "t": {            // type
        "limit": {
          "tif": "Gtc"  // "Alo"|"Ioc"|"Gtc"
        }
      },
      "c": "0x1234..."  // cloid (client order id, optional)
    }],
    "grouping": "na",   // "na"|"normalTpsl"|"positionTpsl"
    "builder": {        // Optional builder fee
      "b": "0x...",     // builder address
      "f": 10           // fee in tenths of basis point
    }
  },
  "nonce": 1681247412573,     // current timestamp recommended
  "signature": {...},         // signature object
  "vaultAddress": "0x...",    // Optional: for vault/subaccount trading
  "expiresAfter": 1681247512573 // Optional: expiration timestamp
}
```

**Order Types**:
- **Limit**: `{"limit": {"tif": "Gtc|Alo|Ioc"}}`
- **Trigger**: `{"trigger": {"isMarket": true, "triggerPx": "30000", "tpsl": "tp|sl"}}`

**Time-in-Force Options**:
- **ALO** (Add Liquidity Only): Post-only, canceled if would immediately match
- **IOC** (Immediate or Cancel): Unfilled portion canceled
- **GTC** (Good Till Canceled): Normal resting behavior

### 2. Cancel Orders
```json
{
  "action": {
    "type": "cancel",
    "cancels": [{
      "a": 0,      // asset
      "o": 91490942 // order ID (oid)
    }]
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

### 3. Cancel by Client Order ID
```json
{
  "action": {
    "type": "cancelByCloid",
    "cancels": [{
      "asset": 0,
      "cloid": "0x1234567890abcdef..."
    }]
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

### 4. Modify an Order
```json
{
  "action": {
    "type": "modify",
    "oid": 91490942,  // order ID or client order ID
    "order": {
      "a": 0,
      "b": true,
      "p": "29800.0",
      "s": "0.02",
      "r": false,
      "t": {"limit": {"tif": "Gtc"}},
      "c": "0x..."
    }
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

## Advanced Trading Features

### 5. TWAP Orders
```json
{
  "action": {
    "type": "twapOrder",
    "twap": {
      "a": 0,        // asset
      "b": true,     // isBuy
      "s": "10.0",   // size
      "r": false,    // reduceOnly
      "m": 30,       // minutes
      "t": true      // randomize timing
    }
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

**TWAP Details**:
- Divides large orders into 30-second intervals
- Maximum 3% slippage per suborder
- Attempts to meet execution target based on elapsed time
- Catches up with larger suborders if behind (max 3x normal size)

### 6. Schedule Cancel (Dead Man's Switch)
```json
{
  "action": {
    "type": "scheduleCancel",
    "time": 1681247512573  // Optional: timestamp (omit to remove)
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

**Features**:
- Automatically cancels all open orders at specified time
- Must be at least 5 seconds in future
- Max 10 triggers per day, resets at 00:00 UTC
- Critical for risk management

## Position & Risk Management

### 7. Update Leverage
```json
{
  "action": {
    "type": "updateLeverage",
    "asset": 0,
    "isCross": true,    // cross or isolated leverage
    "leverage": 10      // new leverage value
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

### 8. Update Isolated Margin
```json
{
  "action": {
    "type": "updateIsolatedMargin",
    "asset": 0,
    "isBuy": true,
    "ntli": 1000000     // amount in 6 decimals (1 USD = 1000000)
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

## Asset & Transfer Operations

### 9. Internal USDC Transfer
```json
{
  "action": {
    "type": "usdSend",
    "hyperliquidChain": "Mainnet",
    "signatureChainId": "0xa4b1",
    "destination": "0x...",
    "amount": "100.0",
    "time": 1681247412573
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

### 10. Spot to Perp Transfer
```json
{
  "action": {
    "type": "usdClassTransfer",
    "hyperliquidChain": "Mainnet",
    "signatureChainId": "0xa4b1",
    "amount": "100.0",
    "toPerp": true,     // true for spot->perp, false for perp->spot
    "nonce": 1681247412573
  },
  "nonce": 1681247412573,
  "signature": {...}
}
```

## Implementation Guidelines for Genetic Algorithm System

### Asset Identification
- **Perpetuals**: Use index from `universe` field in `meta` response
- **Spot**: Use `10000 + index` where index is from `spotMeta.universe`

### Error Handling
**Success Response**:
```json
{
  "status": "ok",
  "response": {
    "type": "order",
    "data": {
      "statuses": [{"resting": {"oid": 77738308}}]
    }
  }
}
```

**Error Response**:
```json
{
  "status": "ok",
  "response": {
    "type": "order", 
    "data": {
      "statuses": [{"error": "Order must have minimum value of $10."}]
    }
  }
}
```

### Rate Limiting
- Exchange actions have weight `1 + floor(batch_length / 40)`
- Address-based limits: 1 request per 1 USDC traded cumulatively
- Initial buffer of 10,000 requests per address
- When rate limited: 1 request every 10 seconds allowed

### Critical Features for Automated Trading
1. **Batch Operations** - Submit multiple orders/cancels in single request
2. **Client Order IDs** - Track orders with custom identifiers  
3. **Reduce-Only Orders** - Position management without flipping
4. **TWAP Execution** - Large order execution with minimal market impact
5. **Dead Man's Switch** - Automatic risk management if system fails

This endpoint provides the execution engine for the genetic algorithm trading system, enabling automated order placement, modification, and risk management as strategies evolve and execute.