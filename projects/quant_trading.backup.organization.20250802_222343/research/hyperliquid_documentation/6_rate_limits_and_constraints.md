# Hyperliquid Rate Limits and API Constraints

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## IP-Based Rate Limits

### REST API Limits (Per IP Address)
- **Aggregated Weight Limit**: 1200 per minute
- **Exchange API Weight**: `1 + floor(batch_length / 40)`
  - Unbatched actions: weight 1
  - Batch of 79 orders: weight 2
  - Batch length = array length in action (e.g., number of orders)

### Info API Weights
- **Weight 2**: `l2Book`, `allMids`, `clearinghouseState`, `orderStatus`, `spotClearinghouseState`, `exchangeStatus`
- **Weight 20**: Most other documented info requests
- **Weight 60**: `userRole`
- **Weight 40**: All explorer API requests

### WebSocket Limits (Per IP Address)
- **Max Connections**: 100 WebSocket connections
- **Max Subscriptions**: 1000 WebSocket subscriptions  
- **Max Unique Users**: 10 users across user-specific subscriptions
- **Message Rate**: 2000 messages/minute across all connections
- **Inflight Posts**: 100 simultaneous post messages

## Address-Based Limits (Per User)

### Trading Request Limits
**Core Formula**: 1 request per 1 USDC traded cumulatively since address inception
- **Initial Buffer**: 10,000 requests per new address
- **When Rate Limited**: 1 request every 10 seconds allowed
- **Sub-accounts**: Treated as separate users with independent limits

### Cancel Operation Limits
**Enhanced Limit**: `min(limit + 100000, limit * 2)`
- **Purpose**: Ensures ability to cancel orders even when rate limited
- **Example**: If normal limit is 50,000, cancel limit becomes 100,000

### Order Quantity Limits
**Default Limit**: 1000 open orders + 1 additional per 5M USDC volume
- **Maximum**: 5000 total open orders per user
- **Restriction**: Orders 1001+ rejected if reduce-only or trigger orders

## Batched Request Handling

### Rate Limit Application
- **IP Limits**: 1 batched request (regardless of size) = 1 request for IP limits
- **Address Limits**: 1 batched request with n orders = n requests for address limits

### Batch Weight Calculation
```python
# Exchange API batch weight
def calculate_weight(batch_length):
    return 1 + (batch_length // 40)

# Examples:
# 1 order = weight 1
# 40 orders = weight 2  
# 79 orders = weight 2
# 80 orders = weight 3
```

## Implementation Guidelines for Genetic Algorithm System

### 1. Rate Limit Management Strategy
```python
class RateLimitManager:
    def __init__(self):
        self.ip_weight_remaining = 1200  # Per minute
        self.address_requests_remaining = 10000  # Initial buffer
        self.last_reset_time = time.time()
        
    def can_make_request(self, weight=1, is_address_based=True):
        # Check IP limits
        if self.ip_weight_remaining < weight:
            return False
            
        # Check address limits for trading actions
        if is_address_based and self.address_requests_remaining < 1:
            return False
            
        return True
        
    def consume_request(self, weight=1, is_address_based=True):
        self.ip_weight_remaining -= weight
        if is_address_based:
            self.address_requests_remaining -= 1
```

### 2. Optimal Batching Strategy
```python
def optimize_batch_size(orders):
    """
    Optimize batch size for weight efficiency
    Weight increases every 40 orders, so batch in multiples of 40
    """
    max_batch_size = 40  # Optimal for weight efficiency
    batches = []
    
    for i in range(0, len(orders), max_batch_size):
        batch = orders[i:i + max_batch_size]
        batches.append(batch)
        
    return batches
```

### 3. WebSocket Connection Management
```python
class WebSocketManager:
    def __init__(self):
        self.max_connections = 100
        self.max_subscriptions = 1000
        self.max_users = 10
        self.active_connections = 0
        self.active_subscriptions = 0
        self.tracked_users = set()
        
    def can_create_connection(self):
        return self.active_connections < self.max_connections
        
    def can_add_subscription(self, user=None):
        if self.active_subscriptions >= self.max_subscriptions:
            return False
            
        if user and user not in self.tracked_users:
            if len(self.tracked_users) >= self.max_users:
                return False
                
        return True
```

### 4. Trading Volume Tracking
```python
class VolumeTracker:
    def __init__(self):
        self.cumulative_volume = 0  # USDC traded since inception
        self.base_limit = 10000     # Initial request buffer
        
    def calculate_request_limit(self):
        """Calculate current request limit based on trading volume"""
        volume_bonus = int(self.cumulative_volume)  # 1 request per 1 USDC
        return self.base_limit + volume_bonus
        
    def calculate_cancel_limit(self, normal_limit):
        """Enhanced limit for cancel operations"""
        return min(normal_limit + 100000, normal_limit * 2)
```

## Error Handling and Recovery

### Rate Limit Exceeded Responses
- **IP Rate Limit**: HTTP 429 with retry-after header
- **Address Rate Limit**: 10-second cooldown period
- **WebSocket Limits**: Connection rejection or forced disconnection

### Recovery Strategies
1. **Exponential Backoff**: Implement progressive delays after rate limit hits
2. **Request Queuing**: Queue requests during rate limit periods
3. **Graceful Degradation**: Reduce strategy execution frequency when limited
4. **Alternative Paths**: Use WebSocket for data, REST for essential trading only

## Monitoring and Alerting

### Key Metrics to Track
```python
monitoring_metrics = {
    "ip_weight_usage": "current_weight / 1200",
    "address_limit_usage": "requests_used / available_limit", 
    "websocket_connections": "active_connections / 100",
    "websocket_subscriptions": "active_subs / 1000",
    "rate_limit_hits": "count of 429 responses",
    "avg_request_latency": "milliseconds",
    "cumulative_trading_volume": "USDC total"
}
```

### Alert Thresholds
- **IP Weight**: Alert at 80% usage (960/1200)
- **Address Limit**: Alert at 90% usage
- **WebSocket**: Alert at 90% of connection/subscription limits
- **Rate Limit Hits**: Alert on any 429 responses

## Strategic Implications for Genetic Algorithm System

### 1. Strategy Execution Frequency
- **High-Volume Strategies**: Earn more request capacity through trading volume
- **Low-Volume Strategies**: May be constrained by base 10k request limit
- **Strategy Selection**: Factor rate limits into genetic fitness evaluation

### 2. Multi-Account Architecture
- **Sub-Accounts**: Each gets independent rate limits
- **Risk Distribution**: Spread strategies across multiple addresses
- **Capacity Scaling**: More accounts = more request capacity

### 3. Execution Optimization
- **Batch Operations**: Group orders/cancels for weight efficiency
- **WebSocket Priority**: Use WebSocket for high-frequency data access
- **Critical Path**: Prioritize essential trading operations over monitoring

This rate limiting framework is crucial for designing the genetic algorithm trading system's execution engine to operate within Hyperliquid's infrastructure constraints while maximizing strategy execution capability.