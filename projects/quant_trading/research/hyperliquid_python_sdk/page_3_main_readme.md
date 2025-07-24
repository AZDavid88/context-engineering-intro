# Hyperliquid Python SDK - Main Documentation

**Source URL**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
**Extraction Method**: Playwright + Jina
**Content Quality**: High (Complete SDK overview and setup instructions)

## Installation

```bash
pip install hyperliquid-python-sdk
```

## Configuration

1. Set the public key as the `account_address` in examples/config.json
2. Set your private key as the `secret_key` in examples/config.json
3. See example_utils.py for config loading patterns

### Optional: Generate API Wallet

- Generate and authorize a new API private key on https://app.hyperliquid.xyz/API
- Set the API wallet's private key as `secret_key` in config.json
- Still use main wallet's public key as `account_address` (NOT the API wallet address)

## Basic Usage Example

```python
from hyperliquid.info import Info
from hyperliquid.utils import constants

info = Info(constants.TESTNET_API_URL, skip_ws=True)
user_state = info.user_state("0xcd5051944f780a621ee62e39e493c489668acf4d")
print(user_state)
```

## Project Structure

- **examples/**: 38+ complete usage examples
- **hyperliquid/**: Core SDK implementation
- **api/**: API specifications
- **tests/**: Test suite

## Development Requirements

- **Python 3.10 exactly** (dependency issues on 3.11+, typing issues on older versions)
- **Poetry** for dependency management (v1.x, not v2.x)

## Setup for Contributing

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.1 python3 -

# Set correct Python version
brew install python@3.10 && poetry env use /opt/homebrew/Cellar/python@3.10/3.10.16/bin/python3.10

# Install dependencies
make install
```

## Available Make Commands

- `make pre-commit`: Run linters and formatters
- `make test`: Run tests with pytest
- `make install`: Install dependencies
- `make check-safety`: Run safety checks
- `make help`: Show all available commands

## License

MIT License - See LICENSE.md for details

## Repository Stats

- **859 stars**, **294 forks**
- **36 contributors**
- **36 releases** (latest: v0.16.0)
- **Languages**: Python 98.6%, Makefile 1.4%
- **Used by 184 projects**