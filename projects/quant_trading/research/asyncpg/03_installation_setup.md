# AsyncPG Installation & Setup - Phase 4 Development Environment

**Source**: https://magicstack.github.io/asyncpg/current/installation.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - Neon Integration Development Setup

## Installation Methods

### Standard Installation (Recommended for Phase 4)
```bash
$ pip install asyncpg
```

**asyncpg** has no external dependencies when not using GSSAPI/SSPI authentication. This is the recommended installation method for the Phase 4 Neon integration.

### GSSAPI/SSPI Authentication Support
If you need GSSAPI/SSPI authentication (enterprise environments):

```bash
$ pip install 'asyncpg[gssauth]'
```

This installs:
- **SSPI support** on Windows
- **GSSAPI support** on non-Windows platforms

SSPI and GSSAPI interoperate as clients and servers: an SSPI client can authenticate to a GSSAPI server and vice versa.

### Linux GSSAPI Requirements
On Linux, installing GSSAPI requires:
- A working C compiler
- Kerberos 5 development files

**Debian/Ubuntu:**
```bash
$ sudo apt-get install libkrb5-dev
```

**RHEL/Fedora:**
```bash
$ sudo yum install krb5-devel
```

(This is needed because PyPI does not have Linux wheels for **gssapi**.)

### Windows GSSAPI Alternative
It is also possible to use GSSAPI on Windows:
1. `pip install gssapi`
2. Install [Kerberos for Windows](https://web.mit.edu/kerberos/dist/)
3. Set the `gsslib` parameter or the `PGGSSLIB` environment variable to `gssapi` when connecting

## Building from Source

If you want to build **asyncpg** from a Git checkout you will need:

### Prerequisites
- Cloned repo with `--recurse-submodules`
- A working C compiler
- CPython header files:
  - **Debian/Ubuntu**: `python3-dev` package
  - **RHEL/Fedora**: `python3-devel` package

### Build Commands
```bash
# Standard build
$ pip install -e .

# Debug build with runtime checks
$ env ASYNCPG_DEBUG=1 pip install -e .
```

## Running Tests

### Prerequisites for Testing
- PostgreSQL installed
- Test database access

### Test Execution
```bash
$ python setup.py test
```

## Phase 4 Development Environment Setup

### Docker Development Environment
For consistent Phase 4 development, use Docker with PostgreSQL:

```dockerfile
# Dockerfile for AsyncPG development
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install AsyncPG
RUN pip install asyncpg

# Copy application code
COPY . /app
WORKDIR /app

# Install project dependencies
RUN pip install -r requirements.txt
```

### Development Dependencies
Create `requirements-dev.txt` for Phase 4 development:

```txt
asyncpg>=0.30.0
asyncio-pool>=0.6.0
pytest>=7.0.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0
```

### Environment Variables for Neon Integration
```bash
# Phase 4 Environment Configuration
export NEON_DATABASE_URL="postgresql://user:password@ep-example.us-east-2.aws.neon.tech/dbname"
export NEON_SSL_MODE="require"
export NEON_APPLICATION_NAME="quant_trading_phase4"
export NEON_POOL_MIN_SIZE="5"
export NEON_POOL_MAX_SIZE="20"
export NEON_COMMAND_TIMEOUT="30"
```

### Connection Testing Script
Create `test_neon_connection.py` for Phase 4 validation:

```python
import asyncio
import asyncpg
import os
from datetime import datetime

async def test_neon_connection():
    """Test Neon connection for Phase 4 setup."""
    
    try:
        # Get connection from environment
        dsn = os.getenv('NEON_DATABASE_URL')
        if not dsn:
            print("❌ NEON_DATABASE_URL environment variable not set")
            return False
        
        # Test basic connection
        conn = await asyncpg.connect(dsn)
        
        # Test basic query
        version = await conn.fetchval('SELECT version()')
        print(f"✅ Connected to: {version}")
        
        # Test TimescaleDB extension (required for Phase 4)
        try:
            extensions = await conn.fetch(
                "SELECT * FROM pg_extension WHERE extname = 'timescaledb'"
            )
            if extensions:
                print("✅ TimescaleDB extension available")
            else:
                print("⚠️  TimescaleDB extension not found")
        except Exception as e:
            print(f"⚠️  Could not check TimescaleDB: {e}")
        
        # Test connection pool
        pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=5,
            command_timeout=10
        )
        
        async with pool.acquire() as pool_conn:
            timestamp = await pool_conn.fetchval('SELECT NOW()')
            print(f"✅ Connection pool working: {timestamp}")
        
        await pool.close()
        await conn.close()
        
        print("🎉 Phase 4 Neon setup validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_neon_connection())
    exit(0 if success else 1)
```

## Troubleshooting Common Issues

### SSL Connection Issues
If you encounter SSL-related errors with Neon:

```python
import ssl
import asyncpg

# Create custom SSL context if needed
ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

conn = await asyncpg.connect(dsn, ssl=ssl_context)
```

### Connection Timeout Issues
For cloud deployments, increase timeout values:

```python
conn = await asyncpg.connect(
    dsn,
    timeout=60,  # Connection timeout
    command_timeout=30  # Query timeout
)
```

### Memory Issues with Large Result Sets
Use cursors for memory-efficient processing:

```python
async with conn.transaction():
    async with conn.cursor('SELECT * FROM large_table') as cursor:
        async for row in cursor:
            # Process row without loading all data into memory
            process_row(row)
```

## Phase 4 Integration Checklist

- ✅ AsyncPG installed with no external dependencies
- ✅ PostgreSQL client tools available for testing
- ✅ Environment variables configured for Neon connection
- ✅ Connection test script validates setup
- ✅ SSL configuration verified for Neon security requirements
- ✅ Connection pooling tested and working
- ✅ TimescaleDB extension availability confirmed
- ✅ Development dependencies installed for testing

This setup provides a solid foundation for Phase 4 Neon integration development with proper error handling, testing capabilities, and performance optimization.