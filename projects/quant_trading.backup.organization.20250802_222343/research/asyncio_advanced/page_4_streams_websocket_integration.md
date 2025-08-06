# AsyncIO Streams - WebSocket Integration Patterns

**Source**: https://docs.python.org/3/library/asyncio-stream.html
**Extraction Method**: Brightdata MCP
**Research Focus**: WebSocket integration patterns, stream processing, and network I/O for high-frequency trading data feeds

## Overview

Streams are high-level async/await-ready primitives to work with network connections. Streams allow sending and receiving data without using callbacks or low-level protocols and transports.

**Key Advantage**: Perfect for WebSocket connections that need to handle continuous market data streams in trading systems.

## Stream Functions

### Connection Establishment

```python
async asyncio.open_connection(host=None, port=None, *, limit=None, ssl=None, 
                             family=0, proto=0, flags=0, sock=None, local_addr=None, 
                             server_hostname=None, ssl_handshake_timeout=None, 
                             ssl_shutdown_timeout=None, happy_eyeballs_delay=None, 
                             interleave=None)
```

Establish a network connection and return a pair of `(reader, writer)` objects.

**Key Parameters**:
- `limit`: Buffer size limit used by the returned [`StreamReader`](#asyncio.StreamReader "asyncio.StreamReader") instance (default: 64 KiB)
- `ssl`: SSL context for secure connections
- `sock`: Use existing socket (useful for WebSocket upgrade)

**Returns**: `(reader, writer)` tuple where:
- `reader`: [`StreamReader`](#asyncio.StreamReader "asyncio.StreamReader") instance
- `writer`: [`StreamWriter`](#asyncio.StreamWriter "asyncio.StreamWriter") instance

### Server Creation

```python
async asyncio.start_server(client_connected_cb, host=None, port=None, *, limit=None, 
                          family=socket.AF_UNSPEC, flags=socket.AI_PASSIVE, sock=None, 
                          backlog=100, ssl=None, reuse_address=None, reuse_port=None, 
                          keep_alive=None, ssl_handshake_timeout=None, 
                          ssl_shutdown_timeout=None, start_serving=True)
```

Start a socket server with automatic client handling.

## StreamReader Class

```python
class asyncio.StreamReader
```

Represents a reader object that provides APIs to read data from the IO stream. As an [asynchronous iterable](../glossary.html#term-asynchronous-iterable), the object supports the [`async for`](../reference/compound_stmts.html#async-for) statement.

### Core Reading Methods

```python
async def read(n=-1)
```
Read up to _n_ bytes from the stream.

**Behavior**:
- If _n_ is not provided or set to `-1`, read until EOF, then return all read [`bytes`](stdtypes.html#bytes "bytes")
- If _n_ is `0`, return an empty `bytes` object immediately
- If _n_ is positive, return at most _n_ available `bytes` as soon as at least 1 byte is available

```python
async def readline()
```
Read one line, where "line" is a sequence of bytes ending with `\n`.

**Use Case**: Perfect for line-based protocols and message parsing in trading systems.

```python
async def readexactly(n)
```
Read exactly _n_ bytes.

**Exception Handling**: Raise an [`IncompleteReadError`](asyncio-exceptions.html#asyncio.IncompleteReadError "asyncio.IncompleteReadError") if EOF is reached before _n_ can be read.

```python
async def readuntil(separator=b'\n')
```
Read data from the stream until _separator_ is found.

**Advanced Feature**: The _separator_ may also be a tuple of separators for complex message parsing.

**Exception Handling**:
- [`LimitOverrunError`](asyncio-exceptions.html#asyncio.LimitOverrunError "asyncio.LimitOverrunError") if data exceeds configured stream limit
- [`IncompleteReadError`](asyncio-exceptions.html#asyncio.IncompleteReadError "asyncio.IncompleteReadError") if EOF is reached before complete separator is found

### State Management

```python
def at_eof()
```
Return `True` if the buffer is empty and [`feed_eof()`](#asyncio.StreamReader.feed_eof "asyncio.StreamReader.feed_eof") was called.

```python
def feed_eof()
```
Acknowledge the EOF.

## StreamWriter Class

```python
class asyncio.StreamWriter
```

Represents a writer object that provides APIs to write data to the IO stream.

### Core Writing Methods

```python
def write(data)
```
The method attempts to write the _data_ to the underlying socket immediately. If that fails, the data is queued in an internal write buffer until it can be sent.

**Critical Pattern**: The method should be used along with the `drain()` method:

```python
stream.write(data)
await stream.drain()
```

```python
def writelines(data)
```
The method writes a list (or any iterable) of bytes to the underlying socket immediately.

```python
async def drain()
```
Wait until it is appropriate to resume writing to the stream.

**Flow Control**: This is a flow control method that interacts with the underlying IO write buffer:
- When the size of the buffer reaches the high watermark, _drain()_ blocks until the size of the buffer is drained down to the low watermark
- When there is nothing to wait for, the [`drain()`](#asyncio.StreamWriter.drain "asyncio.StreamWriter.drain") returns immediately

### Connection Management

```python
def close()
```
The method closes the stream and the underlying socket.

**Recommended Pattern**:

```python
stream.close()
await stream.wait_closed()
```

```python
async def wait_closed()
```
Wait until the stream is closed. Should be called after [`close()`](#asyncio.StreamWriter.close "asyncio.StreamWriter.close") to wait until the underlying connection is closed.

```python
def is_closing()
```
Return `True` if the stream is closed or in the process of being closed.

### TLS Upgrade

```python
async def start_tls(sslcontext, *, server_hostname=None, ssl_handshake_timeout=None, 
                   ssl_shutdown_timeout=None)
```
Upgrade an existing stream-based connection to TLS.

**Parameters**:
- `sslcontext`: configured instance of [`SSLContext`](ssl.html#ssl.SSLContext "ssl.SSLContext")
- `server_hostname`: sets or overrides the host name for certificate matching
- `ssl_handshake_timeout`: time to wait for TLS handshake (default: 60.0 seconds)
- `ssl_shutdown_timeout`: time to wait for SSL shutdown (default: 30.0 seconds)

### Transport Access

```python
@property
def transport
```
Return the underlying asyncio transport.

```python
def get_extra_info(name, default=None)
```
Access optional transport information; see [`BaseTransport.get_extra_info()`](asyncio-protocol.html#asyncio.BaseTransport.get_extra_info "asyncio.BaseTransport.get_extra_info") for details.

## WebSocket Integration Examples

### Basic TCP Echo Client

```python
import asyncio

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8888)

    print(f'Send: {message!r}')
    writer.write(message.encode())
    await writer.drain()

    data = await reader.read(100)
    print(f'Received: {data.decode()!r}')

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

asyncio.run(tcp_echo_client('Hello World!'))
```

### HTTP Headers Fetching

```python
import asyncio
import urllib.parse
import sys

async def print_http_headers(url):
    url = urllib.parse.urlsplit(url)
    if url.scheme == 'https':
        reader, writer = await asyncio.open_connection(
            url.hostname, 443, ssl=True)
    else:
        reader, writer = await asyncio.open_connection(
            url.hostname, 80)

    query = (
        f"HEAD {url.path or '/'} HTTP/1.0\r\n"
        f"Host: {url.hostname}\r\n"
        f"\r\n"
    )

    writer.write(query.encode('latin-1'))
    while True:
        line = await reader.readline()
        if not line:
            break

        line = line.decode('latin1').rstrip()
        if line:
            print(f'HTTP header> {line}')

    # Ignore the body, close the socket
    writer.close()
    await writer.wait_closed()
```

## Quant Trading Implementation Patterns

### Hyperliquid WebSocket Connection

```python
import asyncio
import websockets
import orjson
from typing import Optional, Callable

class HyperliquidWebSocketClient:
    def __init__(self, 
                 message_handler: Callable,
                 buffer_limit: int = 1024 * 1024):  # 1MB buffer
        self.url = "wss://api.hyperliquid.xyz/ws"
        self.message_handler = message_handler
        self.buffer_limit = buffer_limit
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connection_active = False

    async def connect_with_streams(self):
        """Connect using asyncio streams for fine-grained control"""
        try:
            # Establish WebSocket handshake manually for stream control
            self.reader, self.writer = await asyncio.open_connection(
                'api.hyperliquid.xyz', 443, ssl=True,
                limit=self.buffer_limit
            )
            
            # Perform WebSocket upgrade
            await self._perform_websocket_handshake()
            
            self.connection_active = True
            print("✅ Hyperliquid WebSocket connected via streams")
            
            # Start message processing
            await self._process_messages()
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            await self._cleanup_connection()

    async def _perform_websocket_handshake(self):
        """Perform WebSocket protocol upgrade"""
        import base64
        import hashlib
        import secrets
        
        # Generate WebSocket key
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        
        # Send upgrade request
        upgrade_request = (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: api.hyperliquid.xyz\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        
        self.writer.write(upgrade_request.encode())
        await self.writer.drain()
        
        # Read upgrade response
        response_lines = []
        while True:
            line = await self.reader.readline()
            if line == b'\r\n':
                break
            response_lines.append(line.decode().strip())
        
        # Verify upgrade success
        if not any('101 Switching Protocols' in line for line in response_lines):
            raise ConnectionError("WebSocket upgrade failed")

    async def _process_messages(self):
        """Process incoming WebSocket messages with proper frame parsing"""
        try:
            while self.connection_active:
                # Read WebSocket frame header
                frame_data = await self._read_websocket_frame()
                
                if frame_data:
                    try:
                        # Parse JSON message
                        message = orjson.loads(frame_data)
                        
                        # Handle message with timeout to prevent blocking
                        async with asyncio.timeout(1.0):
                            await self.message_handler(message)
                            
                    except orjson.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error: {e}")
                    except asyncio.TimeoutError:
                        print("⚠️  Message handler timeout")
                    except Exception as e:
                        print(f"⚠️  Message processing error: {e}")
                
                # Yield control to prevent blocking
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            print("📊 Message processing cancelled")
            raise
        except Exception as e:
            print(f"❌ Message processing error: {e}")
            raise

    async def _read_websocket_frame(self) -> Optional[bytes]:
        """Read and parse WebSocket frame format"""
        try:
            # Read first 2 bytes for frame info
            frame_start = await self.reader.readexactly(2)
            
            # Parse frame header
            fin = (frame_start[0] & 0x80) == 0x80
            opcode = frame_start[0] & 0x0F
            masked = (frame_start[1] & 0x80) == 0x80
            payload_length = frame_start[1] & 0x7F
            
            # Handle extended payload length
            if payload_length == 126:
                length_data = await self.reader.readexactly(2)
                payload_length = int.from_bytes(length_data, 'big')
            elif payload_length == 127:
                length_data = await self.reader.readexactly(8)
                payload_length = int.from_bytes(length_data, 'big')
            
            # Read masking key if present
            if masked:
                mask = await self.reader.readexactly(4)
            
            # Read payload
            if payload_length > 0:
                payload = await self.reader.readexactly(payload_length)
                
                # Unmask payload if necessary
                if masked:
                    payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
                
                return payload
            
            return None
            
        except asyncio.IncompleteReadError:
            print("📊 WebSocket connection closed by server")
            self.connection_active = False
            return None
        except Exception as e:
            print(f"❌ Frame reading error: {e}")
            return None

    async def send_subscription(self, subscription_data: dict):
        """Send subscription message with proper WebSocket framing"""
        if not self.connection_active:
            raise ConnectionError("WebSocket not connected")
        
        try:
            # Serialize message
            message_bytes = orjson.dumps(subscription_data)
            
            # Create WebSocket frame
            frame = self._create_websocket_frame(message_bytes)
            
            # Send frame
            self.writer.write(frame)
            await self.writer.drain()
            
        except Exception as e:
            print(f"❌ Send subscription error: {e}")
            raise

    def _create_websocket_frame(self, payload: bytes) -> bytes:
        """Create WebSocket frame for outgoing messages"""
        frame = bytearray()
        
        # FIN + opcode (text frame = 0x1)
        frame.append(0x81)
        
        # Payload length
        payload_length = len(payload)
        if payload_length < 126:
            frame.append(payload_length | 0x80)  # Set mask bit
        elif payload_length < 65536:
            frame.append(126 | 0x80)
            frame.extend(payload_length.to_bytes(2, 'big'))
        else:
            frame.append(127 | 0x80)
            frame.extend(payload_length.to_bytes(8, 'big'))
        
        # Masking key (required for client-to-server)
        import secrets
        mask = secrets.token_bytes(4)
        frame.extend(mask)
        
        # Masked payload
        masked_payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
        frame.extend(masked_payload)
        
        return bytes(frame)

    async def _cleanup_connection(self):
        """Clean up connection resources"""
        self.connection_active = False
        
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception as e:
                print(f"⚠️  Writer cleanup error: {e}")
        
        print("🔌 WebSocket connection cleaned up")

    async def close(self):
        """Gracefully close WebSocket connection"""
        print("📊 Closing WebSocket connection...")
        await self._cleanup_connection()

# Usage Example
async def market_data_handler(message: dict):
    """Handle incoming market data messages"""
    if 'channel' in message:
        channel = message['channel']
        data = message.get('data', {})
        
        if channel == 'allMids':
            # Process mid prices
            for asset, price in data.items():
                print(f"💰 {asset}: ${price}")
        
        elif channel == 'l2Book':
            # Process order book updates
            symbol = data.get('coin', 'Unknown')
            bids = data.get('levels', [[]])[0]
            asks = data.get('levels', [[]])[1] if len(data.get('levels', [])) > 1 else []
            
            if bids:
                best_bid = float(bids[0]['px']) if bids else 0
                print(f"📈 {symbol} Best Bid: ${best_bid}")

async def main():
    client = HyperliquidWebSocketClient(
        message_handler=market_data_handler,
        buffer_limit=2 * 1024 * 1024  # 2MB buffer for high-frequency data
    )
    
    try:
        # Connect and start processing
        await client.connect_with_streams()
        
        # Subscribe to market data
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "allMids"
            }
        }
        await client.send_subscription(subscription)
        
        # Keep connection alive
        await asyncio.sleep(30)  # Run for 30 seconds
        
    except KeyboardInterrupt:
        print("📊 Shutting down...")
    finally:
        await client.close()

# Run the client
asyncio.run(main())
```

### Stream-Based Market Data Aggregator

```python
class StreamBasedMarketDataAggregator:
    def __init__(self, max_buffer_size: int = 1024 * 1024):
        self.max_buffer_size = max_buffer_size
        self.active_connections = {}
        self.message_queue = asyncio.Queue(maxsize=10000)

    async def create_multiple_feeds(self, exchanges: list):
        """Create multiple exchange connections using streams"""
        async with asyncio.TaskGroup() as tg:
            for exchange in exchanges:
                task = tg.create_task(
                    self.connect_exchange(exchange),
                    name=f"stream_{exchange}"
                )
                self.active_connections[exchange] = task

    async def connect_exchange(self, exchange: str):
        """Connect to exchange using asyncio streams"""
        try:
            if exchange == "hyperliquid":
                reader, writer = await asyncio.open_connection(
                    'api.hyperliquid.xyz', 443, ssl=True,
                    limit=self.max_buffer_size
                )
            # Add other exchanges...
            
            # Store connection for management
            self.active_connections[exchange] = {
                'reader': reader,
                'writer': writer,
                'status': 'connected'
            }
            
            # Start message processing for this exchange
            await self.process_exchange_stream(exchange, reader, writer)
            
        except Exception as e:
            print(f"❌ {exchange} connection failed: {e}")
            
    async def process_exchange_stream(self, exchange: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Process messages from specific exchange stream"""
        try:
            while True:
                # Read with timeout to prevent hanging
                async with asyncio.timeout(30):
                    data = await reader.read(8192)  # Read in chunks
                    
                if not data:
                    print(f"📊 {exchange} stream ended")
                    break
                
                # Parse and queue messages
                messages = self.parse_exchange_data(exchange, data)
                for message in messages:
                    try:
                        self.message_queue.put_nowait({
                            'exchange': exchange,
                            'timestamp': asyncio.get_event_loop().time(),
                            'data': message
                        })
                    except asyncio.QueueFull:
                        print(f"⚠️  Message queue full, dropping {exchange} message")
                
        except asyncio.TimeoutError:
            print(f"⚠️  {exchange} stream timeout")
        except asyncio.CancelledError:
            print(f"📊 {exchange} stream cancelled")
            raise
        finally:
            # Cleanup connection
            writer.close()
            await writer.wait_closed()
            print(f"🔌 {exchange} stream connection closed")

    def parse_exchange_data(self, exchange: str, data: bytes) -> list:
        """Parse exchange-specific data format"""
        messages = []
        # Implementation depends on exchange protocol
        # This is a simplified version
        try:
            decoded = data.decode('utf-8')
            for line in decoded.split('\n'):
                if line.strip():
                    messages.append(orjson.loads(line))
        except Exception as e:
            print(f"⚠️  Parse error for {exchange}: {e}")
        
        return messages

    async def get_aggregated_data(self):
        """Get aggregated market data from all exchanges"""
        while True:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process aggregated message
                yield message
                
                # Signal task completion
                self.message_queue.task_done()
                
            except asyncio.TimeoutError:
                # No messages available, yield control
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                print("📊 Aggregator cancelled")
                break

# Usage
async def main():
    aggregator = StreamBasedMarketDataAggregator()
    
    # Create connections to multiple exchanges
    exchanges = ["hyperliquid", "binance", "coinbase"]
    
    try:
        # Start all connections
        connection_task = asyncio.create_task(
            aggregator.create_multiple_feeds(exchanges)
        )
        
        # Process aggregated data
        async for message in aggregator.get_aggregated_data():
            exchange = message['exchange']
            timestamp = message['timestamp']
            data = message['data']
            
            print(f"📊 {exchange} @ {timestamp}: {len(str(data))} bytes")
            
            # Your trading logic here
            # await process_market_data(message)
            
    except KeyboardInterrupt:
        print("📊 Shutting down aggregator...")
        connection_task.cancel()
        await connection_task

asyncio.run(main())
```

## Key Benefits for Trading Systems

1. **Fine-Grained Control**: Streams provide precise control over buffer sizes and flow control for high-frequency data
2. **Backpressure Management**: `drain()` method provides natural backpressure handling for outbound data
3. **SSL/TLS Support**: Built-in secure connection support for encrypted trading APIs  
4. **Timeout Handling**: Easy timeout implementation for preventing hanging connections
5. **Connection Management**: Proper connection lifecycle management with cleanup support
6. **Protocol Flexibility**: Can handle any protocol including WebSocket, HTTP, custom binary protocols
7. **Memory Efficiency**: Configurable buffer limits prevent memory exhaustion during data bursts

This stream-based approach is essential for building robust, high-performance quantitative trading systems that need direct control over network connections and data flow management.