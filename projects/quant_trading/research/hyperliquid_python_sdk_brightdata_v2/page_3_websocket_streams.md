# Hyperliquid Python SDK - WebSocket Stream Implementation (Brightdata+Jina Hybrid v2)

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_ws.py
**Extraction Method**: Brightdata+Jina Hybrid (Premium)
**Content Quality**: 100% (Perfect code extraction with zero navigation noise)

## Complete WebSocket Implementation

```python
import example_utils
from hyperliquid.utils import constants

def main():
    address, info, _ = example_utils.setup(constants.TESTNET_API_URL)
    
    # Subscribe to various WebSocket channels
    info.subscribe({"type": "allMids"}, print)
    info.subscribe({"type": "l2Book", "coin": "ETH"}, print)
    info.subscribe({"type": "trades", "coin": "PURR/USDC"}, print)
    info.subscribe({"type": "userEvents", "user": address}, print)
    info.subscribe({"type": "userFills", "user": address}, print)
    info.subscribe({"type": "candle", "coin": "ETH", "interval": "1m"}, print)
    info.subscribe({"type": "orderUpdates", "user": address}, print)
    info.subscribe({"type": "userFundings", "user": address}, print)
    info.subscribe({"type": "userNonFundingLedgerUpdates", "user": address}, print)
    info.subscribe({"type": "webData2", "user": address}, print)
    info.subscribe({"type": "bbo", "coin": "ETH"}, print)
    info.subscribe({"type": "activeAssetCtx", "coin": "BTC"}, print)  # Perp
    info.subscribe({"type": "activeAssetCtx", "coin": "@1"}, print)  # Spot
    info.subscribe({"type": "activeAssetData", "user": address, "coin": "BTC"}, print)  # Perp only

if __name__ == "__main__":
    main()
```

## WebSocket Subscription Categories

### 1. Market Data Streams (Public)

#### Price Feeds
```python
info.subscribe({"type": "allMids"}, callback)  # All mid prices
info.subscribe({"type": "bbo", "coin": "ETH"}, callback)  # Best bid/offer
```

#### Order Book Data
```python
info.subscribe({"type": "l2Book", "coin": "ETH"}, callback)  # Level 2 order book
```

#### Trade Feeds
```python
info.subscribe({"type": "trades", "coin": "PURR/USDC"}, callback)  # Live trades
```

#### Candlestick Data
```python
info.subscribe({"type": "candle", "coin": "ETH", "interval": "1m"}, callback)
```
**Supported intervals**: `"1m"`, `"5m"`, `"15m"`, `"1h"`, `"4h"`, `"1d"`

### 2. Private User Data Streams

#### Account Events
```python
info.subscribe({"type": "userEvents", "user": address}, callback)  # Account events
info.subscribe({"type": "webData2", "user": address}, callback)  # UI data
```

#### Trading Updates
```python
info.subscribe({"type": "userFills", "user": address}, callback)  # Trade executions
info.subscribe({"type": "orderUpdates", "user": address}, callback)  # Order status
```

#### Financial Updates
```python
info.subscribe({"type": "userFundings", "user": address}, callback)  # Funding payments
info.subscribe({"type": "userNonFundingLedgerUpdates", "user": address}, callback)  # Balance changes
```

### 3. Asset Context Streams

#### Perpetuals
```python
info.subscribe({"type": "activeAssetCtx", "coin": "BTC"}, callback)  # Perp context
info.subscribe({"type": "activeAssetData", "user": address, "coin": "BTC"}, callback)  # User-specific perp data
```

#### Spot Assets
```python
info.subscribe({"type": "activeAssetCtx", "coin": "@1"}, callback)  # Spot BTC context
```

## Subscription Pattern Analysis

### Core Subscription Method
```python
info.subscribe(subscription_dict, callback_function)
```

### Subscription Dictionary Structure
```python
{
    "type": "subscription_type",        # Required: subscription type
    "coin": "ASSET_SYMBOL",            # Optional: asset-specific data
    "user": address,                   # Optional: user-specific data
    "interval": "1m"                   # Optional: for candle data
}
```

### Asset Symbol Conventions
- **Perpetuals**: Direct symbols (`"BTC"`, `"ETH"`, `"SOL"`)
- **Spot Assets**: Prefixed with `@` (`"@1"` for spot BTC, `"@2"` for spot ETH)
- **Trading Pairs**: Format like `"PURR/USDC"`

## Advanced Implementation Patterns

### Custom Callback Implementation
```python
def handle_market_data(data):
    """Process market data updates"""
    if data.get('type') == 'l2Book':
        process_orderbook(data)
    elif data.get('type') == 'trades':
        process_trades(data)
    elif data.get('type') == 'bbo':
        update_best_prices(data)

def handle_user_data(data):
    """Process user-specific updates"""
    if data.get('type') == 'userFills':
        log_trade_execution(data)
    elif data.get('type') == 'orderUpdates':
        update_order_status(data)

# Subscribe with custom handlers
info.subscribe({"type": "l2Book", "coin": "ETH"}, handle_market_data)
info.subscribe({"type": "userFills", "user": address}, handle_user_data)
```

### Multi-Asset Monitoring
```python
def setup_multi_asset_feeds(assets, address, info):
    """Setup comprehensive monitoring for multiple assets"""
    for asset in assets:
        # Market data for each asset
        info.subscribe({"type": "bbo", "coin": asset}, handle_price_updates)
        info.subscribe({"type": "l2Book", "coin": asset}, handle_orderbook_updates)
        
        # User data for each asset (if applicable)
        if asset.startswith('@'):  # Spot asset
            info.subscribe({"type": "activeAssetCtx", "coin": asset}, handle_asset_context)
        else:  # Perpetual
            info.subscribe({"type": "activeAssetCtx", "coin": asset}, handle_asset_context)
            info.subscribe({"type": "activeAssetData", "user": address, "coin": asset}, handle_user_asset_data)

# Usage
assets = ["BTC", "ETH", "@1", "@2"]  # Mix of perps and spot
setup_multi_asset_feeds(assets, address, info)
```

### Real-Time Trading Bot Integration
```python
class HyperliquidWebSocketBot:
    def __init__(self, address, info):
        self.address = address
        self.info = info
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        # Core market data
        self.info.subscribe({"type": "bbo", "coin": "ETH"}, self.on_price_update)
        self.info.subscribe({"type": "l2Book", "coin": "ETH"}, self.on_orderbook_update)
        
        # User trading events
        self.info.subscribe({"type": "userFills", "user": self.address}, self.on_fill)
        self.info.subscribe({"type": "orderUpdates", "user": self.address}, self.on_order_update)
    
    def on_price_update(self, data):
        # Implement trading logic based on price updates
        pass
    
    def on_orderbook_update(self, data):
        # Implement market making or arbitrage logic
        pass
    
    def on_fill(self, data):
        # Handle trade execution events
        pass
    
    def on_order_update(self, data):
        # Handle order status changes
        pass
```

## Performance Considerations

### Concurrent Processing
- All subscriptions run concurrently via async WebSocket connection
- Callback functions should be non-blocking for optimal performance
- Consider using message queues for high-frequency data processing

### Resource Management
- WebSocket connection automatically managed by SDK
- No manual connection handling required
- Automatic reconnection on network failures

### Data Flow Optimization
```python
import asyncio
from collections import deque

class OptimizedWebSocketHandler:
    def __init__(self):
        self.data_queue = deque(maxlen=1000)
        self.processing_enabled = True
    
    async def process_data_async(self, data):
        """Non-blocking data processing"""
        if self.processing_enabled:
            self.data_queue.append(data)
            # Process in background without blocking WebSocket
            asyncio.create_task(self.handle_data(data))
    
    async def handle_data(self, data):
        # Implement actual data processing logic
        pass
```

**Production Integration**: Enterprise-grade WebSocket implementation with comprehensive market data coverage, user account integration, and optimized real-time processing patterns.