"""
Hyperliquid Exchange Client - WebSocket & REST API Integration

This module provides comprehensive integration with Hyperliquid exchange, including:
- WebSocket real-time data feeds with 11+ subscription types
- REST API for market data queries and account information
- Rate limiting compliance (1200 req/min REST, 100 WebSocket connections)
- Multi-environment support (testnet/mainnet)
- Robust error handling with exponential backoff

Based on V3 comprehensive research from:
- Hyperliquid Python SDK V3 (5 REST endpoints, 11+ WebSocket types)
- Official Hyperliquid Documentation (rate limits, authentication)
- Production-ready error handling patterns
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI
from pydantic import BaseModel, Field, validator

# Import our configuration system - THIS IS THE KEY CONNECTION
from src.config.settings import get_settings, Settings


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class SubscriptionType(str, Enum):
    """Hyperliquid WebSocket subscription types - from V3 research."""
    
    # Market Data Subscriptions
    ALL_MIDS = "allMids"              # Current mid prices for all assets
    L2_BOOK = "l2Book"                # Level 2 order book (top 10 bids/asks)
    TRADES = "trades"                 # Recent trade history
    BBO = "bbo"                       # Best bid/offer updates
    CANDLE = "candle"                 # OHLCV candlestick data
    
    # User Data Subscriptions (require authentication)
    USER_EVENTS = "userEvents"        # Account events and notifications
    USER_FILLS = "userFills"          # Trade execution notifications
    ORDER_UPDATES = "orderUpdates"    # Order status changes
    USER_FUNDINGS = "userFundings"    # Funding payments received
    
    # Asset Context Subscriptions
    ACTIVE_ASSET_CTX = "activeAssetCtx"      # Perpetual asset contexts
    ACTIVE_ASSET_DATA = "activeAssetData"    # Spot asset contexts


class MarketDataMessage(BaseModel):
    """Validated market data message from WebSocket."""
    
    channel_type: str = Field(..., description="Subscription channel type")
    data: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RateLimiter:
    """Rate limiter for API requests based on Hyperliquid constraints."""
    
    def __init__(self, max_requests_per_second: int = 100):
        """Initialize rate limiter.
        
        Args:
            max_requests_per_second: Maximum requests per second (from settings)
        """
        self.max_requests = max_requests_per_second
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            
            # Remove requests older than 1 second
            self.requests = [req_time for req_time in self.requests if now - req_time < 1.0]
            
            # Check if we can make a request
            if len(self.requests) >= self.max_requests:
                wait_time = 1.0 - (now - self.requests[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self.requests.append(now)


class HyperliquidRESTClient:
    """REST API client for Hyperliquid exchange."""
    
    def __init__(self, settings: Settings):
        """Initialize REST client.
        
        Args:
            settings: Configuration settings from settings.py
        """
        self.settings = settings
        self.base_url = settings.hyperliquid_api_url  # Auto-selects testnet/mainnet
        self.api_key = settings.hyperliquid.api_key
        self.timeout = aiohttp.ClientTimeout(total=settings.hyperliquid.connection_timeout)
        self.rate_limiter = RateLimiter(settings.hyperliquid.max_requests_per_second)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Logger setup
        self.logger = logging.getLogger(f"{__name__}.REST")
        self.logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self.session is None:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "QuantTradingOrganism/1.0"
            }
            
            # Add API key if available
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
            
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )
            
            self.logger.info(f"REST client connected to {self.base_url}")
    
    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("REST client disconnected")
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request with rate limiting.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            
        Returns:
            Response data
            
        Raises:
            aiohttp.ClientError: On HTTP errors
            ValueError: On invalid response format
        """
        await self.rate_limiter.acquire()
        
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                
                try:
                    data = await response.json()
                    self.logger.debug(f"Request to {endpoint} successful")
                    return data
                except json.JSONDecodeError as e:
                    error_text = await response.text()
                    self.logger.error(f"Invalid JSON response from {endpoint}: {error_text}")
                    raise ValueError(f"Invalid JSON response: {error_text}") from e
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error for {endpoint}: {e}")
            raise
    
    async def get_user_state(self, user_address: Optional[str] = None) -> Dict[str, Any]:
        """Get user account state including positions and margin.
        
        Args:
            user_address: User wallet address (optional for API key auth)
            
        Returns:
            User state data including positions and margin summary
        """
        payload = {"type": "clearinghouseState"}
        if user_address:
            payload["user"] = user_address
        
        return await self._make_request("/info", payload)
    
    async def get_l2_book(self, symbol: str) -> Dict[str, Any]:
        """Get Level 2 order book for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            
        Returns:
            Order book with bids and asks
        """
        payload = {
            "type": "l2Book",
            "coin": symbol
        }
        return await self._make_request("/info", payload)
    
    async def get_all_mids(self) -> Dict[str, Any]:
        """Get current mid prices for all available assets.
        
        Returns:
            Dictionary of symbol -> mid price
        """
        payload = {"type": "allMids"}
        return await self._make_request("/info", payload)
    
    async def get_candles(self, symbol: str, interval: str = "1h", 
                         start_time: Optional[int] = None,
                         end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get candlestick data for a symbol.
        
        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of OHLCV candles
        """
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval
            }
        }
        
        if start_time:
            payload["req"]["startTime"] = start_time
        if end_time:
            payload["req"]["endTime"] = end_time
        
        return await self._make_request("/info", payload)
    
    async def get_asset_contexts(self) -> List[Dict[str, Any]]:
        """Get asset contexts with trading constraints and specifications.
        
        Returns:
            List of asset contexts with leverage limits, size decimals, etc.
        """
        payload = {"type": "meta"}
        response = await self._make_request("/info", payload)
        # Extract universe array from meta response
        return response.get("universe", [])


class HyperliquidWebSocketClient:
    """WebSocket client for Hyperliquid real-time data feeds."""
    
    def __init__(self, settings: Settings):
        """Initialize WebSocket client.
        
        Args:
            settings: Configuration settings from settings.py
        """
        self.settings = settings
        self.websocket_url = settings.hyperliquid_websocket_url  # Auto-selects testnet/mainnet
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_state = ConnectionState.DISCONNECTED
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[SubscriptionType, Callable] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = settings.hyperliquid.reconnect_attempts
        self.reconnect_delay = settings.hyperliquid.reconnect_delay
        
        # Task management
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Logger setup
        self.logger = logging.getLogger(f"{__name__}.WebSocket")
        self.logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    async def connect(self) -> bool:
        """Connect to Hyperliquid WebSocket.
        
        Returns:
            True if connection successful
        """
        if self.connection_state == ConnectionState.CONNECTED:
            return True
        
        self.connection_state = ConnectionState.CONNECTING
        self.logger.info(f"Connecting to WebSocket: {self.websocket_url}")
        
        try:
            # VPN check - critical for Hyperliquid
            if self.settings.hyperliquid.require_vpn:
                self.logger.info("VPN required for Hyperliquid - ensure VPN is connected")
            
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connection_state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            
            # Start background tasks
            self._listen_task = asyncio.create_task(self._listen_messages())
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            
            self.logger.info("WebSocket connected successfully")
            return True
            
        except Exception as e:
            self.connection_state = ConnectionState.FAILED
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._shutdown = True
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Cancel background tasks
        if self._listen_task:
            self._listen_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.logger.info("WebSocket disconnected")
    
    async def subscribe(self, subscription_type: SubscriptionType, 
                       symbol: Optional[str] = None,
                       handler: Optional[Callable] = None) -> bool:
        """Subscribe to a data feed.
        
        Args:
            subscription_type: Type of subscription
            symbol: Symbol for symbol-specific subscriptions
            handler: Optional message handler function
            
        Returns:
            True if subscription successful
        """
        if self.connection_state != ConnectionState.CONNECTED:
            if not await self.connect():
                return False
        
        # Build subscription message
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": subscription_type.value
            }
        }
        
        # Add symbol if required
        if symbol and subscription_type in [SubscriptionType.L2_BOOK, 
                                          SubscriptionType.TRADES, 
                                          SubscriptionType.CANDLE]:
            subscription["subscription"]["coin"] = symbol
        
        try:
            await self.websocket.send(json.dumps(subscription))
            
            # Store subscription and handler
            sub_key = f"{subscription_type.value}_{symbol or 'all'}"
            self.subscriptions[sub_key] = subscription
            if handler:
                self.message_handlers[subscription_type] = handler
            
            self.logger.info(f"Subscribed to {subscription_type.value}" + 
                           (f" for {symbol}" if symbol else ""))
            return True
            
        except Exception as e:
            self.logger.error(f"Subscription failed: {e}")
            return False
    
    async def unsubscribe(self, subscription_type: SubscriptionType, 
                         symbol: Optional[str] = None) -> bool:
        """Unsubscribe from a data feed.
        
        Args:
            subscription_type: Type of subscription
            symbol: Symbol for symbol-specific subscriptions
            
        Returns:
            True if unsubscription successful
        """
        sub_key = f"{subscription_type.value}_{symbol or 'all'}"
        
        if sub_key not in self.subscriptions:
            self.logger.warning(f"No active subscription for {sub_key}")
            return False
        
        unsubscribe_msg = {
            "method": "unsubscribe",
            "subscription": self.subscriptions[sub_key]["subscription"]
        }
        
        try:
            await self.websocket.send(json.dumps(unsubscribe_msg))
            del self.subscriptions[sub_key]
            
            self.logger.info(f"Unsubscribed from {subscription_type.value}" + 
                           (f" for {symbol}" if symbol else ""))
            return True
            
        except Exception as e:
            self.logger.error(f"Unsubscription failed: {e}")
            return False
    
    async def _listen_messages(self) -> None:
        """Listen for incoming WebSocket messages."""
        while not self._shutdown and self.websocket:
            try:
                message = await self.websocket.recv()
                await self._handle_message(message)
                
            except ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                if not self._shutdown:
                    await self._reconnect()
                break
                
            except Exception as e:
                self.logger.error(f"Error receiving message: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, raw_message: str) -> None:
        """Handle incoming WebSocket message.
        
        Args:
            raw_message: Raw JSON message string
        """
        try:
            message_data = json.loads(raw_message)
            
            # Create validated message object
            if "channel" in message_data and "data" in message_data:
                message = MarketDataMessage(
                    channel_type=message_data["channel"],
                    data=message_data["data"]
                )
                
                # Route to appropriate handler
                channel_type = message_data["channel"]
                for sub_type, handler in self.message_handlers.items():
                    if sub_type.value in channel_type:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                        break
                else:
                    # Default logging if no specific handler
                    self.logger.debug(f"Received {channel_type} message: {len(raw_message)} bytes")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def _heartbeat(self) -> None:
        """Send periodic heartbeat to maintain connection."""
        while not self._shutdown and self.websocket:
            try:
                await asyncio.sleep(30)  # 30-second heartbeat
                if self.websocket and self.connection_state == ConnectionState.CONNECTED:
                    pong = await self.websocket.ping()
                    await asyncio.wait_for(pong, timeout=10)
                    
            except asyncio.TimeoutError:
                self.logger.warning("Heartbeat timeout - connection may be stale")
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                break
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._shutdown or self.reconnect_attempts >= self.max_reconnect_attempts:
            return
        
        self.connection_state = ConnectionState.RECONNECTING
        self.reconnect_attempts += 1
        
        # Exponential backoff with jitter
        delay = min(self.reconnect_delay * (2 ** self.reconnect_attempts), 60)
        jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
        await asyncio.sleep(delay + jitter)
        
        self.logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        if await self.connect():
            # Resubscribe to all previous subscriptions
            for sub_key, subscription in self.subscriptions.copy().items():
                try:
                    await self.websocket.send(json.dumps(subscription))
                except Exception as e:
                    self.logger.error(f"Failed to resubscribe to {sub_key}: {e}")


class HyperliquidClient:
    """Unified Hyperliquid client combining REST and WebSocket functionality."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize unified Hyperliquid client.
        
        Args:
            settings: Configuration settings (uses global settings if None)
        """
        self.settings = settings or get_settings()  # KEY CONNECTION TO SETTINGS.PY
        self.rest_client = HyperliquidRESTClient(self.settings)
        self.websocket_client = HyperliquidWebSocketClient(self.settings)
        
        # Logger setup
        self.logger = logging.getLogger(f"{__name__}.Client")
        self.logger.setLevel(logging.DEBUG if self.settings.debug else logging.INFO)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """Connect both REST and WebSocket clients.
        
        Returns:
            True if both connections successful
        """
        self.logger.info("Connecting to Hyperliquid exchange...")
        
        # Connect REST client
        await self.rest_client.connect()
        
        # Connect WebSocket client
        ws_connected = await self.websocket_client.connect()
        
        if ws_connected:
            self.logger.info("Hyperliquid client fully connected")
            return True
        else:
            self.logger.error("Failed to establish WebSocket connection")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect both clients."""
        self.logger.info("Disconnecting from Hyperliquid exchange...")
        await self.websocket_client.disconnect()
        await self.rest_client.disconnect()
        self.logger.info("Hyperliquid client disconnected")
    
    # REST API Methods (delegate to REST client)
    async def get_user_state(self, user_address: Optional[str] = None) -> Dict[str, Any]:
        """Get user account state."""
        return await self.rest_client.get_user_state(user_address)
    
    async def get_l2_book(self, symbol: str) -> Dict[str, Any]:
        """Get Level 2 order book."""
        return await self.rest_client.get_l2_book(symbol)
    
    async def get_all_mids(self) -> Dict[str, Any]:
        """Get all mid prices."""
        return await self.rest_client.get_all_mids()
    
    async def get_candles(self, symbol: str, interval: str = "1h", 
                         start_time: Optional[int] = None,
                         end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get candlestick data."""
        return await self.rest_client.get_candles(symbol, interval, start_time, end_time)
    
    async def get_asset_contexts(self) -> List[Dict[str, Any]]:
        """Get asset contexts."""
        return await self.rest_client.get_asset_contexts()
    
    # WebSocket Methods (delegate to WebSocket client)
    async def subscribe_all_mids(self, handler: Optional[Callable] = None) -> bool:
        """Subscribe to all mid prices."""
        return await self.websocket_client.subscribe(SubscriptionType.ALL_MIDS, handler=handler)
    
    async def subscribe_l2_book(self, symbol: str, handler: Optional[Callable] = None) -> bool:
        """Subscribe to Level 2 order book for a symbol."""
        return await self.websocket_client.subscribe(SubscriptionType.L2_BOOK, symbol, handler)
    
    async def subscribe_trades(self, symbol: str, handler: Optional[Callable] = None) -> bool:
        """Subscribe to trade feed for a symbol."""
        return await self.websocket_client.subscribe(SubscriptionType.TRADES, symbol, handler)


async def test_hyperliquid_client():
    """Test function to validate client functionality."""
    
    # Use our settings system - DEMONSTRATES THE CONNECTION
    settings = get_settings()
    
    print("=== Hyperliquid Client Test ===")
    print(f"Environment: {settings.environment}")
    print(f"API URL: {settings.hyperliquid_api_url}")
    print(f"WebSocket URL: {settings.hyperliquid_websocket_url}")
    print(f"API Key configured: {'Yes' if settings.hyperliquid.api_key else 'No'}")
    print(f"Rate limit: {settings.hyperliquid.max_requests_per_second} req/sec")
    
    async with HyperliquidClient(settings) as client:
        try:
            # Test REST API
            print("\n=== Testing REST API ===")
            all_mids = await client.get_all_mids()
            print(f"Retrieved {len(all_mids)} mid prices")
            
            asset_contexts = await client.get_asset_contexts()
            print(f"Retrieved {len(asset_contexts)} asset contexts")
            
            # Test WebSocket
            print("\n=== Testing WebSocket ===")
            
            def handle_all_mids(message: MarketDataMessage):
                print(f"Received all mids update: {len(message.data)} assets")
            
            await client.subscribe_all_mids(handler=handle_all_mids)
            
            # Listen for 10 seconds
            print("Listening to real-time data for 10 seconds...")
            await asyncio.sleep(10)
            
            print("\n✅ Hyperliquid client test completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    """Test the Hyperliquid client."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_hyperliquid_client())