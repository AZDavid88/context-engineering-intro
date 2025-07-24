# Hyperliquid Python SDK - Basic Order Example

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_order.py
**Extraction Method**: Playwright + Jina
**Content Quality**: High (Contains complete code example)

## Core Implementation Code

```python
import json
import example_utils
from hyperliquid.utils import constants

def main():
    address, info, exchange = example_utils.setup(base_url=constants.TESTNET_API_URL, skip_ws=True)
    
    # Get the user state and print out position information
    user_state = info.user_state(address)
    positions = []
    for position in user_state["assetPositions"]:
        positions.append(position["position"])
    
    if len(positions) > 0:
        print("positions:")
        for position in positions:
            print(json.dumps(position, indent=2))
    else:
        print("no open positions")
    
    # Place an order that should rest by setting the price very low
    order_result = exchange.order("ETH", True, 0.2, 1100, {"limit": {"tif": "Gtc"}})
    print(order_result)
    
    # Query the order status by oid
    if order_result["status"] == "ok":
        status = order_result["response"]["data"]["statuses"][0]
        if "resting" in status:
            order_status = info.query_order_by_oid(address, status["resting"]["oid"])
            print("Order status by oid:", order_status)
    
    # Cancel the order
    if order_result["status"] == "ok":
        status = order_result["response"]["data"]["statuses"][0]
        if "resting" in status:
            cancel_result = exchange.cancel("ETH", status["resting"]["oid"])
            print(cancel_result)

if __name__ == "__main__":
    main()
```

## Key API Patterns Discovered

1. **Setup Pattern**: `example_utils.setup(base_url=constants.TESTNET_API_URL, skip_ws=True)`
2. **User State Query**: `info.user_state(address)` returns position information
3. **Order Placement**: `exchange.order(symbol, is_buy, size, price, order_type)`
4. **Order Status Query**: `info.query_order_by_oid(address, oid)`
5. **Order Cancellation**: `exchange.cancel(symbol, oid)`

## Implementation-Ready Insights

- Uses testnet by default (`constants.TESTNET_API_URL`)
- Order structure: symbol, direction (bool), size, price, order type dict
- Order types include `{"limit": {"tif": "Gtc"}}` (Good Till Cancel)
- Response includes status and oid for tracking
- Position data accessible through `user_state["assetPositions"]`