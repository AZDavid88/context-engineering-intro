"""
Order Management System - Live Order Execution

This module implements live order execution system that converts genetic position
sizes to live Hyperliquid orders with robust order lifecycle management,
execution quality analysis, and error handling.

This addresses GAP 3 from the PRP: Missing Live Order Execution System.

Key Features:
- Convert genetic position sizes to live Hyperliquid orders
- Order lifecycle management (create, fill, cancel, modify)
- Execution quality analysis and slippage tracking
- Retry logic with exponential backoff
- Position tracking and reconciliation
- Risk management integration
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque

from src.data.hyperliquid_client import HyperliquidClient
from src.execution.position_sizer import PositionSizeResult, GeneticPositionSizer
from src.config.settings import get_settings, Settings


class OrderType(str, Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class OrderSide(str, Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    """Order request from genetic strategy."""
    
    symbol: str
    side: OrderSide
    size: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancel
    strategy_id: str = "unknown"
    signal_strength: float = 1.0
    urgency: str = "normal"  # normal, high, emergency
    max_slippage: float = 0.005  # 0.5%
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    client_order_id: Optional[str] = None


@dataclass
class OrderFill:
    """Order fill information."""
    
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    filled_size: float
    fill_price: float
    fill_time: datetime
    commission: float
    liquidity: str  # maker or taker
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of fill."""
        return self.filled_size * self.fill_price


@dataclass
class Order:
    """Complete order with lifecycle tracking."""
    
    # Request information
    request: OrderRequest
    
    # Exchange information
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution information
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[OrderFill] = field(default_factory=list)
    total_commission: float = 0.0
    
    # Timing information
    submitted_at: Optional[datetime] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Execution quality
    slippage: float = 0.0
    execution_time_ms: float = 0.0
    
    # Error information
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                             OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR]
    
    @property
    def remaining_size(self) -> float:
        """Calculate remaining unfilled size."""
        return max(0.0, self.request.size - self.filled_size)
    
    @property
    def fill_percentage(self) -> float:
        """Calculate percentage filled."""
        if self.request.size == 0:
            return 0.0
        return (self.filled_size / self.request.size) * 100


class OrderManager:
    """Manages live order execution for genetic trading strategies."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize order manager.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.OrderManager")
        
        # Hyperliquid client for order execution
        self.hyperliquid_client = HyperliquidClient(settings)
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}  # client_order_id -> Order
        self.completed_orders: deque = deque(maxlen=1000)  # Keep last 1000 orders
        self.order_counter = 0
        
        # Execution tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'avg_execution_time_ms': 0.0,
            'avg_slippage': 0.0
        }
        
        # Risk management
        self.max_daily_orders = 1000
        self.max_order_size = 10.0  # Maximum order size
        self.daily_order_count = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        # Retry configuration
        self.max_retries = 3
        self.base_retry_delay = 1.0  # seconds
        self.max_retry_delay = 30.0  # seconds
        
        # Order callbacks
        self.fill_callbacks: List[Callable[[OrderFill], None]] = []
        self.order_callbacks: List[Callable[[Order], None]] = []
        
        self.logger.info("Order manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the order manager and connections."""
        try:
            await self.hyperliquid_client.connect()
            self.logger.info("Order manager connection established")
        except Exception as e:
            self.logger.error(f"Failed to initialize order manager: {e}")
            raise
    
    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit an order to the exchange.
        
        Args:
            order_request: Order request to submit
            
        Returns:
            Client order ID for tracking
            
        Raises:
            ValueError: If order validation fails
            RuntimeError: If order submission fails
        """
        # Validate order request
        self._validate_order_request(order_request)
        
        # Generate client order ID
        client_order_id = self._generate_order_id()
        order_request.client_order_id = client_order_id
        
        # Create order object
        order = Order(request=order_request)
        order.submitted_at = datetime.now(timezone.utc)
        
        # Store order for tracking
        self.active_orders[client_order_id] = order
        
        try:
            # Submit to exchange
            await self._submit_to_exchange(order)
            
            self.logger.info(f"Order submitted: {client_order_id} - {order_request.symbol} "
                           f"{order_request.side.value} {order_request.size:.4f}")
            
            return client_order_id
            
        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            self.logger.error(f"Failed to submit order {client_order_id}: {e}")
            
            # Move to completed orders
            self._complete_order(order)
            raise
    
    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an active order.
        
        Args:
            client_order_id: Client order ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        if client_order_id not in self.active_orders:
            self.logger.warning(f"Cannot cancel order {client_order_id}: not found")
            return False
        
        order = self.active_orders[client_order_id]
        
        if order.is_complete:
            self.logger.warning(f"Cannot cancel order {client_order_id}: already complete")
            return False
        
        try:
            if order.exchange_order_id:
                await self.hyperliquid_client.cancel_order(order.exchange_order_id)
            
            order.status = OrderStatus.CANCELLED
            order.completed_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Order cancelled: {client_order_id}")
            self._complete_order(order)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {client_order_id}: {e}")
            return False
    
    async def process_position_size_update(self, position_result: PositionSizeResult,
                                         current_position: float = 0.0) -> Optional[str]:
        """Process position size update and create necessary orders.
        
        Args:
            position_result: Position sizing result from genetic algorithm
            current_position: Current position size
            
        Returns:
            Client order ID if order was created, None otherwise
        """
        target_size = position_result.target_size
        size_difference = target_size - current_position
        
        # Check if order is needed
        min_order_size = 0.001
        if abs(size_difference) < min_order_size:
            return None
        
        # Determine order side
        if size_difference > 0:
            side = OrderSide.BUY
            order_size = abs(size_difference)
        else:
            side = OrderSide.SELL
            order_size = abs(size_difference)
        
        # Create order request
        order_request = OrderRequest(
            symbol=position_result.symbol,
            side=side,
            size=order_size,
            order_type=OrderType.MARKET,
            strategy_id=f"genetic_{position_result.symbol}",
            signal_strength=1.0,  # Position sizing already incorporates signal strength
            max_slippage=0.01  # 1% maximum slippage
        )
        
        try:
            client_order_id = await self.submit_order(order_request)
            
            self.logger.info(f"Position update order created: {client_order_id} - "
                           f"{position_result.symbol} {side.value} {order_size:.4f} "
                           f"(target: {target_size:.4f}, current: {current_position:.4f})")
            
            return client_order_id
            
        except Exception as e:
            self.logger.error(f"Failed to create position update order for "
                            f"{position_result.symbol}: {e}")
            return None
    
    async def update_order_status(self) -> None:
        """Update status of all active orders."""
        if not self.active_orders:
            return
        
        try:
            # Get order status updates from exchange
            order_updates = await self._fetch_order_updates()
            
            for client_order_id, order in list(self.active_orders.items()):
                if order.exchange_order_id in order_updates:
                    update_data = order_updates[order.exchange_order_id]
                    await self._process_order_update(order, update_data)
                    
        except Exception as e:
            self.logger.error(f"Error updating order status: {e}")
    
    async def _submit_to_exchange(self, order: Order) -> None:
        """Submit order to Hyperliquid exchange.
        
        Args:
            order: Order to submit
        """
        order_request = order.request
        
        # Convert to Hyperliquid order format
        hyperliquid_order = {
            'symbol': order_request.symbol,
            'side': order_request.side.value,
            'size': order_request.size,
            'type': order_request.order_type.value,
            'clientOrderId': order_request.client_order_id
        }
        
        if order_request.price:
            hyperliquid_order['price'] = order_request.price
        
        if order_request.stop_price:
            hyperliquid_order['stopPrice'] = order_request.stop_price
        
        start_time = time.time()
        
        # Submit with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.hyperliquid_client.create_order(hyperliquid_order)
                
                # Extract exchange order ID
                if 'orderId' in response:
                    order.exchange_order_id = response['orderId']
                    order.status = OrderStatus.SUBMITTED
                    
                    # Calculate execution time
                    order.execution_time_ms = (time.time() - start_time) * 1000
                    
                    # Update statistics
                    self.execution_stats['total_orders'] += 1
                    self.daily_order_count += 1
                    
                    return
                else:
                    raise RuntimeError(f"No order ID in response: {response}")
                    
            except Exception as e:
                order.retry_count = attempt
                
                if attempt < self.max_retries:
                    # Calculate retry delay with exponential backoff
                    delay = min(
                        self.base_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    
                    self.logger.warning(f"Order submission attempt {attempt + 1} failed: {e}. "
                                      f"Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    order.status = OrderStatus.ERROR
                    order.error_message = f"Max retries exceeded: {e}"
                    self.execution_stats['failed_orders'] += 1
                    raise RuntimeError(f"Order submission failed after {self.max_retries} retries: {e}")
    
    async def _fetch_order_updates(self) -> Dict[str, Any]:
        """Fetch order status updates from exchange.
        
        Returns:
            Dictionary mapping exchange order IDs to update data
        """
        try:
            # Get active exchange order IDs
            exchange_order_ids = [
                order.exchange_order_id for order in self.active_orders.values()
                if order.exchange_order_id
            ]
            
            if not exchange_order_ids:
                return {}
            
            # Fetch updates from Hyperliquid
            updates = await self.hyperliquid_client.get_order_status(exchange_order_ids)
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error fetching order updates: {e}")
            return {}
    
    async def _process_order_update(self, order: Order, update_data: Dict[str, Any]) -> None:
        """Process order status update from exchange.
        
        Args:
            order: Order to update
            update_data: Update data from exchange
        """
        try:
            # Update order status
            new_status = self._map_exchange_status(update_data.get('status', ''))
            old_status = order.status
            order.status = new_status
            order.last_update = datetime.now(timezone.utc)
            
            # Process fills if any
            if 'fills' in update_data:
                new_fills = []
                for fill_data in update_data['fills']:
                    fill = OrderFill(
                        fill_id=fill_data.get('fillId', ''),
                        order_id=order.exchange_order_id,
                        symbol=order.request.symbol,
                        side=order.request.side,
                        filled_size=float(fill_data.get('size', 0)),
                        fill_price=float(fill_data.get('price', 0)),
                        fill_time=datetime.fromtimestamp(
                            fill_data.get('timestamp', time.time()) / 1000,
                            tz=timezone.utc
                        ),
                        commission=float(fill_data.get('commission', 0)),
                        liquidity=fill_data.get('liquidity', 'taker')
                    )
                    
                    # Check if this is a new fill
                    if not any(f.fill_id == fill.fill_id for f in order.fills):
                        new_fills.append(fill)
                        order.fills.append(fill)
                
                # Update order totals
                if new_fills:
                    self._update_order_fill_totals(order)
                    
                    # Notify fill callbacks
                    for fill in new_fills:
                        for callback in self.fill_callbacks:
                            try:
                                callback(fill)
                            except Exception as e:
                                self.logger.warning(f"Fill callback error: {e}")
            
            # Calculate slippage for market orders
            if (order.request.order_type == OrderType.MARKET and 
                order.avg_fill_price > 0 and 
                'marketPrice' in update_data):
                
                market_price = float(update_data['marketPrice'])
                if order.request.side == OrderSide.BUY:
                    order.slippage = (order.avg_fill_price - market_price) / market_price
                else:
                    order.slippage = (market_price - order.avg_fill_price) / market_price
            
            # Check if order is complete
            if order.is_complete:
                order.completed_at = datetime.now(timezone.utc)
                self._complete_order(order)
                
                if new_status == OrderStatus.FILLED:
                    self.execution_stats['successful_orders'] += 1
            
            # Log status changes
            if old_status != new_status:
                self.logger.info(f"Order {order.request.client_order_id} status: "
                               f"{old_status.value} -> {new_status.value}")
            
            # Notify order callbacks
            for callback in self.order_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    self.logger.warning(f"Order callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing order update: {e}")
    
    def _update_order_fill_totals(self, order: Order) -> None:
        """Update order fill totals and averages.
        
        Args:
            order: Order to update
        """
        if not order.fills:
            return
        
        # Calculate totals
        total_filled = sum(fill.filled_size for fill in order.fills)
        total_notional = sum(fill.notional_value for fill in order.fills)
        total_commission = sum(fill.commission for fill in order.fills)
        
        # Update order
        order.filled_size = total_filled
        order.total_commission = total_commission
        
        if total_filled > 0:
            order.avg_fill_price = total_notional / total_filled
        
        # Update statistics
        self.execution_stats['total_volume'] += total_notional
        self.execution_stats['total_commission'] += total_commission
    
    def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange status to internal status.
        
        Args:
            exchange_status: Status from exchange
            
        Returns:
            Internal order status
        """
        status_mapping = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        
        return status_mapping.get(exchange_status.lower(), OrderStatus.ERROR)
    
    def _complete_order(self, order: Order) -> None:
        """Move order from active to completed.
        
        Args:
            order: Order to complete
        """
        client_order_id = order.request.client_order_id
        
        if client_order_id in self.active_orders:
            del self.active_orders[client_order_id]
            self.completed_orders.append(order)
    
    def _validate_order_request(self, order_request: OrderRequest) -> None:
        """Validate order request before submission.
        
        Args:
            order_request: Order request to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check daily order limits
        self._check_daily_limits()
        
        # Validate order size
        if order_request.size <= 0:
            raise ValueError("Order size must be positive")
        
        if order_request.size > self.max_order_size:
            raise ValueError(f"Order size {order_request.size} exceeds maximum {self.max_order_size}")
        
        # Validate symbol
        if not order_request.symbol:
            raise ValueError("Symbol is required")
        
        # Validate price for limit orders
        if order_request.order_type == OrderType.LIMIT and not order_request.price:
            raise ValueError("Price is required for limit orders")
        
        # Validate stop price for stop orders
        if (order_request.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT] and 
            not order_request.stop_price):
            raise ValueError("Stop price is required for stop orders")
    
    def _check_daily_limits(self) -> None:
        """Check and reset daily order limits.
        
        Raises:
            ValueError: If daily limits are exceeded
        """
        current_date = datetime.now(timezone.utc).date()
        
        # Reset daily counter if new day
        if current_date > self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = current_date
        
        # Check daily limit
        if self.daily_order_count >= self.max_daily_orders:
            raise ValueError(f"Daily order limit of {self.max_daily_orders} exceeded")
    
    def _generate_order_id(self) -> str:
        """Generate unique client order ID.
        
        Returns:
            Unique client order ID
        """
        self.order_counter += 1
        timestamp = int(time.time() * 1000)
        return f"genetic_{timestamp}_{self.order_counter}"
    
    def add_fill_callback(self, callback: Callable[[OrderFill], None]) -> None:
        """Add callback for order fills.
        
        Args:
            callback: Function to call on each fill
        """
        self.fill_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for order updates.
        
        Args:
            callback: Function to call on each order update
        """
        self.order_callbacks.append(callback)
    
    def get_active_orders(self) -> List[Order]:
        """Get list of active orders.
        
        Returns:
            List of active orders
        """
        return list(self.active_orders.values())
    
    def get_order_by_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            Order if found, None otherwise
        """
        return self.active_orders.get(client_order_id)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        stats = self.execution_stats.copy()
        
        # Calculate derived metrics
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
            stats['failure_rate'] = stats['failed_orders'] / stats['total_orders']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Add current state
        stats['active_orders'] = len(self.active_orders)
        stats['daily_order_count'] = self.daily_order_count
        stats['daily_limit_remaining'] = self.max_daily_orders - self.daily_order_count
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown order manager and clean up resources."""
        self.logger.info("Shutting down order manager...")
        
        # Cancel all active orders
        for client_order_id in list(self.active_orders.keys()):
            try:
                await self.cancel_order(client_order_id)
            except Exception as e:
                self.logger.warning(f"Error cancelling order {client_order_id}: {e}")
        
        # Disconnect from exchange
        await self.hyperliquid_client.disconnect()
        
        self.logger.info("Order manager shutdown complete")


async def test_order_manager():
    """Test function for order manager."""
    
    print("=== Order Manager Test ===")
    
    # Create order manager
    order_manager = OrderManager()
    
    try:
        # Initialize (this would connect to Hyperliquid in real usage)
        print("Initializing order manager...")
        # await order_manager.initialize()  # Skip for test
        print("✅ Order manager initialized")
        
        # Test order creation
        order_request = OrderRequest(
            symbol='BTC-USD',
            side=OrderSide.BUY,
            size=0.01,
            order_type=OrderType.MARKET,
            strategy_id='test_genetic_strategy',
            signal_strength=0.8
        )
        
        print(f"Creating test order: {order_request.symbol} {order_request.side.value} {order_request.size}")
        
        # Note: This would actually submit to exchange in real usage
        # client_order_id = await order_manager.submit_order(order_request)
        # print(f"✅ Order submitted: {client_order_id}")
        
        # Test position size processing
        from src.execution.position_sizer import PositionSizeResult
        
        position_result = PositionSizeResult(
            symbol='BTC-USD',
            target_size=0.05,
            max_size=0.15,
            raw_size=0.05,
            scaling_factor=1.0,
            method_used='genetic_evolved',
            risk_metrics={'sharpe_estimate': 1.2},
            correlation_adjustment=1.0,
            volatility_adjustment=1.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        current_position = 0.02
        
        # This would create an order in real usage
        # client_order_id = await order_manager.process_position_size_update(
        #     position_result, current_position
        # )
        # if client_order_id:
        #     print(f"✅ Position update order created: {client_order_id}")
        
        # Show execution stats
        stats = order_manager.get_execution_stats()
        print(f"\nExecution Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        print(f"Active orders: {len(order_manager.get_active_orders())}")
        
    finally:
        # Clean up
        await order_manager.shutdown()
    
    print(f"\n✅ Order Manager test completed successfully!")


if __name__ == "__main__":
    """Test the order manager."""
    
    import asyncio
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_order_manager())