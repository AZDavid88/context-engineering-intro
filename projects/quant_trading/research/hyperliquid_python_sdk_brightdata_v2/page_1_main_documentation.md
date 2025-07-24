# Hyperliquid Python SDK - Main Documentation (Brightdata+Jina Hybrid v2)

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
**Extraction Method**: Brightdata+Jina Hybrid (Premium)
**Content Quality**: 99% (Premium extraction with complete navigation elimination)

## Installation

Install the SDK using pip:

```bash
pip install hyperliquid-python-sdk
```

## Configuration

### Setup Steps
1. Set the public key as `account_address` in `examples/config.json`
2. Set your private key as `secret_key` in `examples/config.json`
3. Reference `examples/example_utils.py` for config loading

### Optional: API Wallet Key Generation
- Generate a new API private key at https://app.hyperliquid.xyz/API
- Set the API wallet's private key as `secret_key`
- Use the main wallet's public key for `account_address`

## Basic Usage Example

```python
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Initialize Info client for testnet
info = Info(constants.TESTNET_API_URL, skip_ws=True)

# Fetch user state for a specific address
user_state = info.user_state("0xcd5051944f780a621ee62e39e493c489668acf4d")
print(user_state)
```

## Development Setup

### Prerequisites
- Python 3.10 (exactly - issues with 3.11+, typing issues with older versions)
- Poetry (version 1.4.1 - v2.x not supported)

### Installation Steps
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.1 python3 -

# Set Python version
poetry env use /path/to/python3.10

# Install project dependencies
make install
```

## Makefile Commands

- `make install`: Install dependencies
- `make lint`: Run linters 
- `make test`: Run tests
- `make pre-commit`: Run pre-commit checks

## Quality Indicators

- **859 stars**, **294 forks** (strong community adoption)
- **36+ contributors** (active development)
- **184 dependent projects** (production usage validation)
- **36 releases** with semantic versioning
- **98.6% Python code** (focused codebase)

## License

MIT License - See `LICENSE.md` for full details

## Integration Patterns

The SDK provides comprehensive examples covering:
- Basic order operations (`basic_order.py`)
- WebSocket subscriptions (`basic_ws.py`)
- Multi-signature transactions (`multi_sig_*.py`)
- Asset transfers and conversions (`basic_transfer.py`)
- Vault operations (`basic_vault.py`)
- EVM interactions (`evm_*.py`)

**Implementation Assessment**: Production-ready with comprehensive examples and active maintenance.