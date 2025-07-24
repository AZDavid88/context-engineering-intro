# Hyperliquid Python SDK - Main Documentation (Brightdata Method)

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
**Extraction Method**: Brightdata + Jina
**Content Quality**: Very High (Premium content extraction with minimal navigation noise)

## Installation

```bash
pip install hyperliquid-python-sdk
```

## Configuration Steps

1. Set public key as `account_address` in examples/config.json
2. Set private key as `secret_key` in examples/config.json  
3. Reference examples/example_utils.py for config loading patterns

### API Wallet Setup (Optional)

- Generate API key at https://app.hyperliquid.xyz/API
- Use API wallet's private key as `secret_key`
- **Important**: Still use main wallet's public key as `account_address` (NOT API wallet address)

## Basic Usage Pattern

```python
from hyperliquid.info import Info
from hyperliquid.utils import constants

info = Info(constants.TESTNET_API_URL, skip_ws=True)
user_state = info.user_state("0xcd5051944f780a621ee62e39e493c489668acf4d")
print(user_state)
```

## Development Stack Requirements

- **Python 3.10** (exactly - issues with 3.11+, typing issues with older versions)
- **Poetry 1.x** (not v2.x supported)
- Pre-commit hooks for code quality

## Repository Structure Analysis

- **examples/**: Rich collection of implementation examples
- **hyperliquid/**: Core SDK modules
- **tests/**: Comprehensive test suite
- **api/**: API specification files

## Quality Indicators

- **859 stars**, **294 forks** (strong community adoption)
- **36+ contributors** (active development)
- **184 dependent projects** (production usage validation)
- **36 releases** with semantic versioning
- **98.6% Python code** (focused codebase)

## Available Examples Overview

The SDK includes 38+ examples covering:
- Basic order operations
- WebSocket subscriptions
- Multi-signature transactions
- Asset transfers and conversions
- Vault operations
- EVM interactions

## Development Commands

```bash
# Setup
make install

# Quality checks
make pre-commit

# Testing
make test

# View all commands
make help
```

## License & Citation

MIT Licensed with provided BibTeX citation format for academic use.

**Implementation Assessment**: Ready for immediate production integration with comprehensive examples and active maintenance.