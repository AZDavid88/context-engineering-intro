# Hyperliquid API Documentation

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP

## API Overview

Documentation for the Hyperliquid public API

### Available SDKs
- **Python SDK**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk (Primary/Official)
- **Rust SDK**: https://github.com/hyperliquid-dex/hyperliquid-rust-sdk (Less maintained)
- **Typescript SDKs** (Community):
  - https://github.com/nktkas/hyperliquid
  - https://github.com/nomeida/hyperliquid
- **CCXT Integration**: https://docs.ccxt.com/#/exchanges/hyperliquid (Multiple languages)

### Base URLs
- **Mainnet**: https://api.hyperliquid.xyz
- **Testnet**: https://api.hyperliquid-testnet.xyz

All example API calls use the Mainnet url, but you can make the same requests against Testnet using the corresponding url.

## Key Implementation Notes for Quant Trading

### API Architecture
The API is split into two main endpoints:
1. **Info Endpoint** - For retrieving market data, user data, and exchange information
2. **Exchange Endpoint** - For placing orders, cancels, and other trading actions

### Rate Limiting Considerations
- REST requests have aggregated weight limits per IP
- WebSocket connections have separate limits for optimal real-time data access
- Address-based limits apply per user for trading actions

### Authentication & Security
- API requires proper signature generation for trading actions
- Nonces and API wallets system for secure access
- Multiple signature schemes depending on action type

### Real-time Data Access
- WebSocket API available for low-latency market data
- Multiple subscription types for different data feeds
- Optimized for high-frequency data consumption

## Recommended Implementation Approach

For the quant trading system:
1. **Use Official Python SDK** - Most maintained and feature-complete
2. **WebSocket for Market Data** - Essential for real-time price feeds
3. **REST API for Trading** - Order placement, cancellation, portfolio management
4. **Testnet First** - Develop and test strategies on testnet before mainnet deployment

This API structure provides the foundation for implementing the genetic algorithm trading system with proper real-time data feeds and execution capabilities.