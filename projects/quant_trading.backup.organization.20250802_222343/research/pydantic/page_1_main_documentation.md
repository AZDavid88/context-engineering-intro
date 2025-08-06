# Pydantic Main Documentation

**Source URL**: https://docs.pydantic.dev/latest/
**Extraction Method**: Brightdata MCP
**Content Quality**: High - Complete technical documentation with examples

## Overview

Pydantic is the most widely used data validation library for Python. Fast and extensible, Pydantic plays nicely with your linters/IDE/brain. Define how data should be in pure, canonical Python 3.9+; validate it with Pydantic.

**Current Version**: v2.11.7

## Key Features

### 1. Powered by Type Hints
- Schema validation and serialization controlled by type annotations
- Less to learn, less code to write
- Seamless integration with IDE and static analysis tools

### 2. Speed
- Core validation logic written in Rust
- Among the fastest data validation libraries for Python

### 3. JSON Schema
- Pydantic models can emit JSON Schema
- Easy integration with other tools

### 4. Strict and Lax Mode
- **Strict mode**: Data is not converted
- **Lax mode**: Pydantic tries to coerce data to correct type where appropriate

### 5. Multiple Data Types Support
- Dataclasses
- TypedDicts
- Standard library types

### 6. Customisation
- Custom validators and serializers
- Alter how data is processed in powerful ways

### 7. Battle Tested
- Downloaded over 360M times/month
- Used by all FAANG companies and 20 of the 25 largest companies on NASDAQ
- Around 8,000 packages on PyPI use Pydantic

## Installation

```bash
pip install pydantic
```

## Core Usage Pattern

### Basic Model Definition

```python
from datetime import datetime
from pydantic import BaseModel, PositiveInt

class User(BaseModel):
    id: int  # Required field with type coercion
    name: str = 'John Doe'  # Optional field with default
    signup_ts: datetime | None  # Optional datetime field
    tastes: dict[str, PositiveInt]  # Dict with positive integer values
```

### Model Instantiation and Validation

```python
external_data = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',  # ISO 8601 format - will be converted
    'tastes': {
        'wine': 9,
        b'cheese': 7,  # bytes key - will be coerced to string
        'cabbage': '1',  # string value - will be coerced to int
    },
}

user = User(**external_data)
print(user.id)  # 123
print(user.model_dump())  # Dictionary representation
```

### Error Handling

```python
from pydantic import ValidationError

try:
    invalid_data = {'id': 'not an int', 'tastes': {}}
    User(**invalid_data)
except ValidationError as e:
    print(e.errors())
    # Returns detailed error information with location and type
```

## Key Documentation Sections

### Core Concepts
1. **Models** - BaseModel fundamentals
2. **Fields** - Field definitions and constraints
3. **JSON Schema** - Schema generation and validation
4. **JSON** - JSON serialization/deserialization
5. **Types** - Built-in and custom types
6. **Unions** - Union type handling
7. **Alias** - Field aliasing and mapping
8. **Configuration** - Model configuration options
9. **Serialization** - Data output formatting
10. **Validators** - Custom validation logic
11. **Dataclasses** - Pydantic dataclass integration
12. **Settings Management** - Configuration management patterns

### API Documentation Structure
- **BaseModel** - Core model class
- **RootModel** - Root-level validation
- **Pydantic Dataclasses** - Enhanced dataclass support
- **TypeAdapter** - Type validation without models
- **Validate Call** - Function validation decorator
- **Fields** - Field definition utilities
- **Configuration** - Model configuration
- **Errors** - Error handling and types
- **Functional Validators** - Validation functions
- **Standard Library Types** - Built-in type support
- **Pydantic Types** - Custom Pydantic types

## Integration Patterns for Quant Trading System

### Configuration Management
```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class TradingConfig(BaseSettings):
    api_key: str
    max_position_size: float = 10000.0
    risk_threshold: float = 0.02
    
    class Config:
        env_file = ".env"
```

### Data Validation for Trading Signals
```python
from pydantic import BaseModel, Field, validator
from typing import Literal
from datetime import datetime

class TradingSignal(BaseModel):
    symbol: str = Field(..., regex=r'^[A-Z]{3,10}$')
    action: Literal['buy', 'sell', 'hold']
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    price: float = Field(..., gt=0)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Custom validation for trading symbols
        return v.upper()
```

### Strategy Parameters Validation
```python
class StrategyParams(BaseModel):
    lookback_period: int = Field(..., ge=1, le=1000)
    sharpe_threshold: float = Field(..., ge=0.5)
    max_drawdown: float = Field(..., ge=0.01, le=0.5)
    position_size: float = Field(..., gt=0)
    
    class Config:
        # Validate assignment for real-time parameter updates
        validate_assignment = True
```

## Key Advantages for This Project

1. **Type Safety**: Ensures data integrity across the entire trading system
2. **Configuration Management**: pydantic-settings for robust config handling
3. **API Integration**: Perfect for validating external API responses (Hyperliquid)
4. **Performance**: Rust-based validation suitable for real-time trading
5. **Error Handling**: Detailed error messages for debugging trading logic
6. **Serialization**: JSON Schema generation for API documentation
7. **IDE Support**: Excellent autocomplete and type checking

## Companies Using Pydantic

Major organizations using Pydantic include:
- Adobe, Amazon/AWS, Anthropic, Apple
- Google, Microsoft, Netflix, OpenAI
- Facebook, GitHub, IBM, Intel
- NASA, NVIDIA, Oracle, Salesforce

**Next Steps for Implementation**:
1. Use BaseModel for all trading data structures
2. Implement pydantic-settings for configuration management
3. Create custom validators for trading-specific logic
4. Use TypeAdapter for Hyperliquid API response validation
5. Leverage JSON Schema for API documentation