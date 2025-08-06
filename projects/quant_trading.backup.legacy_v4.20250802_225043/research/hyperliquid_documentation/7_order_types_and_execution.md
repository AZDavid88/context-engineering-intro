# Hyperliquid Order Types and Execution

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/order-types
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## Order Types Available

### 1. Market Orders
- **Description**: Executes immediately at current market price
- **Use Case**: Immediate execution when speed is priority over price
- **Risk**: Subject to slippage in volatile markets

### 2. Limit Orders  
- **Description**: Executes at selected limit price or better
- **Use Case**: Price control and passive liquidity provision
- **Behavior**: Rests on order book until filled or canceled

### 3. Stop Market Orders
- **Description**: Market order activated when price reaches stop price
- **Use Case**: Stop losses and breakout strategies
- **Trigger**: Activated on adverse price movement

### 4. Stop Limit Orders
- **Description**: Limit order activated when price reaches stop price  
- **Use Case**: More precise stop loss execution with price control
- **Risk**: May not execute if market gaps past limit price

### 5. Scale Orders
- **Description**: Multiple limit orders across a price range
- **Use Case**: Dollar cost averaging and gradual position building
- **Strategy**: Systematic entry/exit across price levels

### 6. TWAP Orders (Time-Weighted Average Price)
- **Description**: Large order divided into smaller suborders executed over time
- **Execution**: 30-second intervals with 3% max slippage per suborder
- **Use Case**: Large position execution with minimal market impact

## TWAP Order Details (Critical for Large Strategies)

### Execution Mechanics
- **Interval**: Suborders sent every 30 seconds
- **Target**: Elapsed time / total time × total size
- **Slippage Control**: Maximum 3% slippage per suborder
- **Catch-up Logic**: Behind-schedule suborders can be up to 3× normal size
- **Completion**: May not fully complete if too many suborders fail to fill

### Size Constraints
```python
# TWAP suborder size calculation
normal_suborder_size = total_size / number_of_intervals
max_catchup_size = normal_suborder_size * 3

# If falling behind target:
if current_executed < target_executed:
    next_suborder_size = min(
        target_executed - current_executed,
        max_catchup_size
    )
```

### Market Conditions Impact
- **Wide Spreads**: May cause suborders to not fill completely
- **Low Liquidity**: Increases probability of partial fills
- **Network Upgrades**: TWAP suborders don't fill during post-only periods

## Order Options and Time-in-Force

### 1. Reduce Only
- **Purpose**: Only reduces existing position, never increases
- **Use Case**: Position management and risk reduction
- **Constraint**: Rejected if would increase position size

### 2. Good Till Cancel (GTC)
- **Behavior**: Order rests on book until filled or manually canceled  
- **Default**: Standard behavior for limit orders
- **Use Case**: Passive strategies and patient execution

### 3. Add Liquidity Only (ALO) / Post Only
- **Behavior**: Added to order book, never executes immediately
- **Rejection**: Canceled if would match immediately  
- **Use Case**: Market making and liquidity provision strategies
- **Fee Benefit**: Typically receives maker rebates

### 4. Immediate or Cancel (IOC)
- **Behavior**: Executes immediately available quantity, cancels remainder
- **No Resting**: Never remains on order book
- **Use Case**: Testing liquidity without committing to full size

### 5. Take Profit (TP) Orders
- **Trigger**: Activates when TP price is reached
- **Type**: Automatically converts to market order
- **Use Case**: Profit-taking automation
- **Configuration**: Can set specific position percentage

### 6. Stop Loss (SL) Orders  
- **Trigger**: Activates when SL price is reached
- **Type**: Automatically converts to market order
- **Use Case**: Loss limitation and risk management
- **Configuration**: Can set specific position percentage

## Implementation Guidelines for Genetic Algorithm System

### 1. Order Type Selection Strategy
```python
class OrderTypeSelector:
    def select_order_type(self, strategy_params, market_conditions):
        if strategy_params['urgency'] == 'high':
            return 'market'
        elif strategy_params['size'] > market_conditions['avg_volume'] * 0.1:
            return 'twap'  # Large size relative to volume
        elif strategy_params['patience'] == 'high':
            return 'limit_alo'  # Seek maker rebates
        else:
            return 'limit_gtc'  # Standard limit order
```

### 2. TWAP Size Threshold
```python
class TWAPDecisionEngine:
    def should_use_twap(self, order_size, market_data):
        avg_daily_volume = market_data['24h_volume'] 
        order_ratio = order_size / avg_daily_volume
        
        # Use TWAP for orders > 1% of daily volume
        return order_ratio > 0.01
        
    def calculate_twap_duration(self, order_size, urgency):
        base_duration = 30  # minutes
        size_multiplier = min(order_size / 10000, 3)  # Scale with size
        urgency_divisor = urgency  # 1=urgent, 2=normal, 3=patient
        
        return int(base_duration * size_multiplier / urgency_divisor)
```

### 3. Stop Loss Integration
```python
class RiskManagement:
    def create_position_with_stops(self, entry_order, risk_params):
        orders = [entry_order]
        
        if risk_params['stop_loss_pct']:
            sl_price = self.calculate_stop_price(
                entry_order['price'], 
                entry_order['side'],
                risk_params['stop_loss_pct']
            )
            
            sl_order = {
                'type': 'trigger',
                'trigger_px': sl_price,
                'tpsl': 'sl',
                'reduce_only': True,
                'size': entry_order['size']
            }
            orders.append(sl_order)
            
        return orders
```

### 4. Market Making with ALO Orders
```python
class MarketMaker:
    def create_spread_orders(self, mid_price, spread_bps, size):
        bid_price = mid_price * (1 - spread_bps / 10000)
        ask_price = mid_price * (1 + spread_bps / 10000)
        
        orders = [
            {
                'type': 'limit',
                'tif': 'Alo',  # Add Liquidity Only
                'side': 'buy',
                'price': bid_price,
                'size': size
            },
            {
                'type': 'limit', 
                'tif': 'Alo',
                'side': 'sell',
                'price': ask_price,
                'size': size
            }
        ]
        
        return orders
```

## Strategic Considerations for Genetic Evolution

### 1. Order Type as Genetic Parameter
```python
# Order type could be evolved as part of strategy genes
order_type_genes = {
    'market_threshold': 0.001,    # Use market if spread < 0.1%
    'twap_size_ratio': 0.01,      # Use TWAP if size > 1% daily volume  
    'alo_probability': 0.7,       # 70% chance to use ALO for liquidity
    'stop_loss_distance': 0.02    # 2% stop loss distance
}
```

### 2. Execution Cost Optimization
- **Maker vs Taker**: ALO orders earn rebates, market orders pay fees
- **Slippage Control**: TWAP orders limit slippage for large sizes
- **Timing Optimization**: IOC orders test liquidity without commitment

### 3. Risk Management Integration
- **Automatic Stops**: TP/SL orders provide automated risk management
- **Position Sizing**: Reduce-only constraint prevents over-leveraging
- **Execution Monitoring**: Track fill rates and execution quality

This order type framework provides the execution toolkit for implementing sophisticated genetic algorithm strategies with proper risk management and execution cost optimization on Hyperliquid.