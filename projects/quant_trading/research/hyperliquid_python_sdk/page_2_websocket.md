# Hyperliquid Python SDK - WebSocket Subscriptions

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_ws.py
**Extraction Method**: Playwright + Jina
**Content Quality**: High (Complete WebSocket subscription examples)

## Core WebSocket Implementation

```python
import example_utils
from hyperliquid.utils import constants

def main():
    address, info, _ = example_utils.setup(constants.TESTNET_API_URL)
    
    # Subscribe to different types of real-time data
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
    info.subscribe({"type": "activeAssetCtx", "coin": "@1"}, print)   # Spot
    info.subscribe({"type": "activeAssetData", "user": address, "coin": "BTC"}, print)  # Perp only

if __name__ == "__main__":
    main()
```

## Available WebSocket Subscription Types

1. **Market Data**:
   - `allMids`: All mid prices
   - `l2Book`: Level 2 order book for specific coin
   - `trades`: Trade feed for specific coin
   - `bbo`: Best bid/offer for specific coin
   - `candle`: Candlestick data with interval

2. **User-Specific Data**:
   - `userEvents`: User account events
   - `userFills`: User trade fills
   - `orderUpdates`: Order status updates
   - `userFundings`: Funding payments
   - `userNonFundingLedgerUpdates`: Non-funding balance changes
   - `webData2`: General user web data

3. **Asset Context**:
   - `activeAssetCtx`: Asset context for perpetuals or spot
   - `activeAssetData`: Active asset data (perpetuals only)

## Implementation-Ready Patterns

- WebSocket setup: `info.subscribe(subscription_dict, callback_function)`
- Subscription format: `{"type": "subscription_type", "coin": "SYMBOL", "user": address}`
- Spot assets use "@" prefix (e.g., "@1" for spot BTC)
- Perpetual assets use regular symbols (e.g., "BTC", "ETH")
- All subscriptions use callback pattern for real-time data handling
- Some subscriptions require user address for private data
- Candle intervals supported: "1m", "5m", "15m", "1h", "4h", "1d"