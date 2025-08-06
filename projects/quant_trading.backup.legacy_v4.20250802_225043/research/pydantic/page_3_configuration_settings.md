# Pydantic Configuration and Settings Management

**Source URLs**: 
- https://docs.pydantic.dev/latest/concepts/config/
- https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- https://docs.pydantic.dev/latest/api/fields/

**Extraction Method**: Brightdata MCP + WebFetch Enhancement
**Content Quality**: High - Complete configuration and settings documentation

## Configuration Overview

Pydantic behavior is controlled via `ConfigDict` class configuration values. Configuration can be applied to models, dataclasses, TypeAdapter, and other supported types.

## Model Configuration Methods

### Method 1: model_config Class Attribute
```python
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(
        str_max_length=5,
        validate_assignment=True,
        frozen=True
    )
    
    v: str
```

### Method 2: Class Arguments
```python
from pydantic import BaseModel

class Model(BaseModel, frozen=True, str_max_length=5):
    v: str
```

**Advantage**: Static type checkers recognize class arguments and flag mutation errors.

## Key Configuration Options

### Validation Controls
- **`strict`**: Use strict mode (no type coercion)
- **`validate_assignment`**: Validate on attribute assignment
- **`validate_default`**: Validate default values
- **`revalidate_instances`**: Control instance revalidation
- **`use_enum_values`**: Use enum values in validation

### String Controls
- **`str_max_length`**: Maximum string length
- **`str_min_length`**: Minimum string length
- **`str_strip_whitespace`**: Strip leading/trailing whitespace
- **`str_to_lower`**: Convert strings to lowercase
- **`str_to_upper`**: Convert strings to uppercase

### Numeric Controls
- **`coerce_numbers_to_str`**: Convert numbers to strings
- **`use_decimal_context`**: Use decimal context for validation

### Extra Data Handling
- **`extra`**: Control extra field behavior ('ignore', 'allow', 'forbid')
- **`frozen`**: Make model immutable
- **`populate_by_name`**: Allow population by field name and alias

### Error and Warning Controls
- **`arbitrary_types_allowed`**: Allow arbitrary types
- **`from_attributes`**: Enable ORM mode (create from class attributes)

## Global Configuration Inheritance

```python
from pydantic import BaseModel, ConfigDict

class GlobalParent(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        str_to_lower=False,
        validate_assignment=True
    )

class TradingModel(GlobalParent):
    model_config = ConfigDict(str_to_lower=True)  # Merged with parent config
    
    symbol: str
```

**Result**: `TradingModel` inherits `extra='allow'` and `validate_assignment=True` from parent, with `str_to_lower=True` override.

## Pydantic Settings Management

### Installation
```bash
pip install pydantic-settings
```

### Basic Settings Class
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class TradingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='TRADING_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    # API Configuration
    hyperliquid_api_key: str = Field(..., alias='HYPERLIQUID_KEY')
    hyperliquid_secret: str = Field(..., alias='HYPERLIQUID_SECRET')
    
    # Trading Parameters
    max_position_size: float = Field(10000.0, gt=0)
    risk_threshold: float = Field(0.02, ge=0.0, le=1.0)
    base_currency: str = Field('USDC', min_length=3, max_length=5)
    
    # System Configuration
    log_level: str = Field('INFO', regex=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    database_url: str = Field('sqlite:///trading.db')
    redis_url: str = Field('redis://localhost:6379/0')
```

### Configuration Sources Priority (Descending)
1. **CLI arguments** (if enabled)
2. **Initialization arguments**
3. **Environment variables**
4. **Dotenv files** (.env)
5. **Secrets directory**
6. **Default field values**

### Environment Variable Examples
```bash
# .env file
TRADING_HYPERLIQUID_API_KEY=your_api_key_here
TRADING_MAX_POSITION_SIZE=50000.0
TRADING_RISK_THRESHOLD=0.015
TRADING_LOG_LEVEL=DEBUG
```

### Advanced Settings Configuration
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict

class AdvancedTradingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='QUANT_',
        env_file=['.env', '.env.local'],  # Multiple env files
        env_nested_delimiter='__',  # For nested config: QUANT_DATABASE__HOST
        secrets_dir='/run/secrets',  # Docker secrets
        case_sensitive=True
    )
    
    # Nested Configuration
    database: Dict[str, str] = {
        'host': 'localhost',
        'port': '5432',
        'name': 'trading_db'
    }
    
    # List Configuration
    enabled_strategies: List[str] = ['momentum', 'mean_reversion']
    trading_pairs: List[str] = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # Complex Validation
    kelly_fraction: float = Field(
        0.25,
        ge=0.01,
        le=0.5,
        description="Kelly Criterion fraction for position sizing"
    )
```

## Field Definitions and Constraints

### Basic Field Configuration
```python
from pydantic import BaseModel, Field
from typing import Annotated

class TradingStrategy(BaseModel):
    # String validation
    name: str = Field(
        ...,  # Required field
        min_length=3,
        max_length=50,
        description="Strategy name",
        examples=["momentum_1", "mean_reversion_2"]
    )
    
    # Numeric validation
    sharpe_ratio: float = Field(
        ...,
        gt=0.5,  # Greater than 0.5
        le=10.0,  # Less than or equal to 10.0
        description="Required Sharpe ratio threshold"
    )
    
    # Pattern validation
    symbol: str = Field(
        ...,
        pattern=r'^[A-Z]{3,10}$',  # Regex pattern
        description="Trading symbol (uppercase letters only)"
    )
    
    # Alias configuration
    position_size: float = Field(
        alias='positionSize',  # JSON field name
        validation_alias='pos_size',  # Alternative validation name
        serialization_alias='position',  # Output field name
        gt=0
    )
```

### Advanced Field Constraints
```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List

class Order(BaseModel):
    # Multiple constraints
    quantity: float = Field(
        ...,
        gt=0,
        multiple_of=0.01,  # Must be multiple of 0.01
        description="Order quantity"
    )
    
    # List validation
    tags: List[str] = Field(
        default=[],
        max_items=10,
        unique_items=True,
        description="Order tags"
    )
    
    # Datetime validation
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Order creation timestamp"
    )
    
    # Custom validation
    @validator('quantity')
    def validate_quantity(cls, v):
        if v > 1000000:  # 1M limit
            raise ValueError('Quantity exceeds maximum limit')
        return v
```

## Trading System Configuration Examples

### 1. Environment-Based Configuration
```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional

class TradingEnvironmentConfig(BaseSettings):
    """Environment-based trading configuration"""
    
    class Config:
        env_prefix = 'QUANT_'
        case_sensitive = False
        env_file = '.env'
    
    # Environment: development, staging, production
    environment: str = Field('development', regex=r'^(development|staging|production)$')
    
    # VPN Configuration (required for Hyperliquid)
    vpn_enabled: bool = Field(True)
    vpn_server: Optional[str] = Field(None)
    
    # API Configuration
    hyperliquid_testnet: bool = Field(True)  # Use testnet by default
    api_rate_limit: int = Field(200000, ge=1000)  # Requests per second
    
    @validator('vpn_server')
    def vpn_server_required_for_production(cls, v, values):
        if values.get('environment') == 'production' and not v:
            raise ValueError('VPN server required for production')
        return v
```

### 2. Strategy-Specific Configuration
```python
class StrategyConfig(BaseSettings):
    """Individual strategy configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix='STRATEGY_',
        frozen=True,  # Immutable configuration
        validate_assignment=True
    )
    
    # Strategy Identity
    strategy_id: str = Field(..., min_length=1, max_length=32)
    strategy_type: str = Field(..., regex=r'^(genetic|momentum|mean_reversion|arbitrage)$')
    
    # Genetic Algorithm Parameters
    population_size: int = Field(100, ge=10, le=1000)
    mutation_rate: float = Field(0.1, ge=0.001, le=0.5)
    crossover_rate: float = Field(0.8, ge=0.1, le=1.0)
    
    # Risk Management
    max_drawdown: float = Field(0.1, ge=0.01, le=0.5)
    position_size_limit: float = Field(0.25, ge=0.01, le=0.4)  # Max 40% allocation
    
    # Performance Requirements
    min_sharpe_ratio: float = Field(2.0, ge=0.5)
    min_backtest_days: int = Field(30, ge=7, le=365)

# Usage
strategy_config = StrategyConfig(
    strategy_id="genetic_momentum_v1",
    strategy_type="genetic"
)
```

### 3. System-Wide Configuration
```python
from pydantic import BaseModel, Field
from typing import Dict, List

class SystemConfig(BaseSettings):
    """System-wide configuration management"""
    
    model_config = SettingsConfigDict(
        env_prefix='SYSTEM_',
        env_file=['.env', '.env.production'],
        secrets_dir='/app/secrets'
    )
    
    # Database Configuration
    database_url: str = Field(..., min_length=10)
    database_pool_size: int = Field(10, ge=1, le=100)
    
    # Message Queue
    redis_url: str = Field('redis://localhost:6379/0')
    
    # Logging
    log_level: str = Field('INFO', regex=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    log_format: str = Field('json')  # 'json' or 'text'
    
    # Process Management
    supervisor_config: Dict[str, any] = Field(default={
        'unix_http_server': {'file': '/tmp/supervisor.sock'},
        'supervisord': {'logfile': '/tmp/supervisord.log'},
        'rpcinterface': {'supervisor.rpcinterface_factory': 'supervisor.rpcinterface:make_main_rpcinterface'}
    })
    
    # Trading Pairs
    active_trading_pairs: List[str] = Field([
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD'
    ])
```

## Configuration Loading Patterns

### 1. Hierarchical Configuration
```python
# Load configuration with inheritance
base_config = SystemConfig()
strategy_config = StrategyConfig(_env_prefix='MOMENTUM_')
environment_config = TradingEnvironmentConfig()

# Merge configurations
merged_settings = {**base_config.model_dump(), **strategy_config.model_dump()}
```

### 2. Dynamic Reloading
```python
from pydantic_settings import BaseSettings

class ReloadableConfig(BaseSettings):
    def reload(self):
        """Reload configuration from sources"""
        new_instance = self.__class__()
        for field_name, field_value in new_instance.model_dump().items():
            setattr(self, field_name, field_value)

# Usage in production
config = ReloadableConfig()
# Later... reload config without restart
config.reload()
```

## Best Practices for Trading System

### 1. Immutable Configuration
```python
class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,  # Prevent accidental modification
        validate_assignment=True  # Validate on any assignment attempt
    )
```

### 2. Environment-Specific Validation
```python
@validator('hyperliquid_api_key')
def validate_api_key_format(cls, v):
    if not v.startswith('hl_'):
        raise ValueError('Invalid Hyperliquid API key format')
    return v
```

### 3. Fail-Fast Configuration
```python
class Config(BaseSettings):
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        return None if v == '' else v
```

### 4. Type-Safe Environment Integration
```python
# Ensure environment variables are properly typed
DATABASE_URL: str = Field(..., env='DATABASE_URL')
REDIS_URL: str = Field(..., env='REDIS_URL')
LOG_LEVEL: str = Field('INFO', env='LOG_LEVEL')
```

## Next Implementation Steps

1. **Create BaseSettings classes** for different configuration layers
2. **Set up environment files** (.env, .env.production, .env.development)
3. **Implement configuration validation** with custom validators
4. **Enable Docker secrets integration** for production deployment
5. **Create configuration hot-reloading** for dynamic updates
6. **Implement hierarchical config** with inheritance patterns