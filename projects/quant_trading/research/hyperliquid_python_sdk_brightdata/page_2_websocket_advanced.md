# Hyperliquid Python SDK - Advanced WebSocket Implementation (Brightdata Method)

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_ws.py
**Extraction Method**: Brightdata + Jina  
**Content Quality**: Excellent (Clean code extraction with implementation details)

## Complete WebSocket Implementation

```python
import example_utils
from hyperliquid.utils import constants

def main():
    address, info, _ = example_utils.setup(constants.TESTNET_API_URL)
    
    # Market data subscriptions
    info.subscribe({"type": "allMids"}, print)
    info.subscribe({"type": "l2Book", "coin": "ETH"}, print)
    info.subscribe({"type": "trades", "coin": "PURR/USDC"}, print)
    info.subscribe({"type": "bbo", "coin": "ETH"}, print)
    info.subscribe({"type": "candle", "coin": "ETH", "interval": "1m"}, print)
    
    # User-specific subscriptions
    info.subscribe({"type": "userEvents", "user": address}, print)
    info.subscribe({"type": "userFills", "user": address}, print)
    info.subscribe({"type": "orderUpdates", "user": address}, print)
    info.subscribe({"type": "userFundings", "user": address}, print)
    info.subscribe({"type": "userNonFundingLedgerUpdates", "user": address}, print)
    info.subscribe({"type": "webData2", "user": address}, print)
    
    # Asset context subscriptions
    info.subscribe({"type": "activeAssetCtx", "coin": "BTC"}, print)  # Perpetuals
    info.subscribe({"type": "activeAssetCtx", "coin": "@1"}, print)   # Spot assets
    info.subscribe({"type": "activeAssetData", "user": address, "coin": "BTC"}, print)  # Perp only

if __name__ == "__main__":
    main()
```

## Subscription Categories & Use Cases

### 1. Market Data Streams (Public)
- **`allMids`**: Real-time mid prices for all trading pairs
- **`l2Book`**: Level 2 order book depth for specific assets
- **`trades`**: Live trade feed with price/volume data
- **`bbo`**: Best bid/offer updates for rapid price monitoring
- **`candle`**: OHLCV candlestick data with configurable intervals

### 2. Private User Data Streams
- **`userEvents`**: Account-level events and notifications
- **`userFills`**: Trade execution confirmations and fill data
- **`orderUpdates`**: Real-time order status changes
- **`userFundings`**: Funding payment tracking for perpetuals
- **`userNonFundingLedgerUpdates`**: Balance changes excluding funding
- **`webData2`**: Consolidated user interface data

### 3. Asset Context Streams
- **`activeAssetCtx`**: Asset metadata and trading context
- **`activeAssetData`**: User-specific asset data (perpetuals only)

## Implementation Patterns

### Subscription Format
```python
subscription = {
    "type": "subscription_name",
    "coin": "ASSET_SYMBOL",      # For asset-specific data
    "user": address,             # For user-specific data  
    "interval": "1m"             # For candle data
}
info.subscribe(subscription, callback_function)
```

### Asset Symbol Conventions
- **Perpetuals**: Standard symbols (`"BTC"`, `"ETH"`, `"SOL"`)
- **Spot Assets**: Prefixed with `@` (`"@1"` for spot BTC, `"@2"` for spot ETH)
- **Trading Pairs**: Format like `"PURR/USDC"`

### Callback Implementation
```python
def handle_market_data(data):
    # Process real-time market updates
    if data.get('type') == 'l2Book':
        process_orderbook(data)
    elif data.get('type') == 'trades':
        process_trades(data)

info.subscribe({"type": "l2Book", "coin": "ETH"}, handle_market_data)
```

## Advanced Features

### Snapshot Behavior
- Some subscriptions provide initial snapshots upon connection
- Others only send updates when events occur
- Critical for state management in trading algorithms

### Performance Considerations
- All subscriptions run concurrently via async WebSocket connection
- Callback functions should be non-blocking for optimal performance
- Consider using message queues for high-frequency data processing

## Integration with Trading Systems

Perfect for implementing:
- **Real-time portfolio monitoring** via `userEvents` and `userFills`
- **Market data analysis** via `l2Book` and `trades` streams
- **Order management systems** via `orderUpdates` subscriptions
- **Risk management** via funding and balance update streams

**Production Readiness**: Enterprise-grade WebSocket implementation with comprehensive market data coverage and user account integration.