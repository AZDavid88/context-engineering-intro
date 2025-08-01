# Hyperliquid Documentation - Main Overview

## Source
- **URL**: https://hyperliquid.gitbook.io/hyperliquid-docs
- **Extracted**: 2025-01-25
- **Method**: Brightdata MCP + Enhancement

## What is Hyperliquid?

Hyperliquid is a performant blockchain built with the vision of a fully onchain open financial system. Liquidity, user applications, and trading activity synergize on a unified platform that will ultimately house all of finance.

## Technical Overview

Hyperliquid is a layer one blockchain (L1) written and optimized from first principles.

Hyperliquid uses a custom consensus algorithm called HyperBFT inspired by Hotstuff and its successors. Both the algorithm and networking stack are optimized from the ground up to support the unique demands of the L1.

Hyperliquid state execution is split into two broad components: HyperCore and the HyperEVM. HyperCore includes fully onchain perpetual futures and spot order books. Every order, cancel, trade, and liquidation happens transparently with one-block finality inherited from HyperBFT. HyperCore currently supports 200k orders / second, with throughput constantly improving as the node software is further optimized.

The HyperEVM brings the familiar general-purpose smart contract platform pioneered by Ethereum to the Hyperliquid blockchain. With the HyperEVM, the performant liquidity and financial primitives of HyperCore are available as permissionless building blocks for all users and builders.

## Key Information for Quant Trading

### API Access
- **Official Python SDK**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
- **Community Typescript SDKs** available
- **CCXT integration** for multiple languages

### API Endpoints
- **Mainnet URL**: https://api.hyperliquid.xyz
- **Testnet URL**: https://api.hyperliquid-testnet.xyz

### Critical API Components
1. **Info Endpoint** - Supports "Perpetuals" and "Spot" data retrieval
2. **Exchange Endpoint** - Allows trading interactions  
3. **WebSocket Features** - Supports subscriptions, post requests, timeouts and heartbeats

### Additional Considerations
- Requires "nonces and API wallets" for authentication
- Has defined "rate limits and user limits"
- Provides error response handling
- Developers can choose between direct API access or using available SDKs

## Architecture Highlights for Trading Systems

### Performance Characteristics
- **Order Processing**: 200k orders/second capacity
- **Finality**: One-block finality with HyperBFT consensus
- **Real-time Data**: WebSocket feeds for real-time market data
- **Latency**: Optimized networking stack for trading applications

### Trading Features
- **Perpetual Futures**: Fully onchain with transparent order books
- **Spot Trading**: Native spot markets integrated with perpetuals
- **Order Types**: Comprehensive order type support including market, limit, stop orders
- **Risk Management**: Built-in liquidation and margin systems

This documentation serves as the foundation for implementing the quant trading system described in the planning PRP, providing direct access to Hyperliquid's high-performance trading infrastructure.