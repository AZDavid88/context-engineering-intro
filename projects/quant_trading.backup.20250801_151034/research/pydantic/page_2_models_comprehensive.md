# Pydantic Models - Comprehensive Guide

**Source URL**: https://docs.pydantic.dev/latest/concepts/models/
**Extraction Method**: Brightdata MCP
**Content Quality**: High - Complete technical documentation with detailed examples

## Overview

Models are the primary way of defining schema in Pydantic. Models are classes that inherit from `BaseModel` and define fields as annotated attributes. Models are similar to structs in languages like C, or as requirements of a single endpoint in an API.

**Key Difference from Dataclasses**: Models have been designed with subtle-yet-important differences that streamline workflows related to validation, serialization, and JSON schema generation.

## Core Model Concepts

### Basic Model Usage

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    id: int
    name: str = 'Jane Doe'
    
    model_config = ConfigDict(str_max_length=10)
```

**Field Types**:
- `id`: integer, required field
- `name`: string with default value, not required

### Model Instantiation and Validation

```python
user = User(id='123')  # String '123' coerced to integer 123
assert user.name == 'Jane Doe'  # Default value used
assert user.id == 123  # Type coerced
assert isinstance(user.id, int)  # Confirms coercion
```

**Key Validation Principle**: Pydantic guarantees that fields of the resultant model instance will conform to the field types defined on the model.

## Essential Model Methods and Properties

### Core Methods
- **`model_validate()`**: Validates object against Pydantic model
- **`model_validate_json()`**: Validates JSON data against model
- **`model_construct()`**: Creates models without validation (performance)
- **`model_dump()`**: Returns dictionary of fields and values
- **`model_dump_json()`**: Returns JSON string representation
- **`model_copy()`**: Returns copy (shallow by default) of model
- **`model_json_schema()`**: Returns JSON Schema dictionary
- **`model_rebuild()`**: Rebuilds model schema for recursive/generic models

### Key Properties
- **`model_fields`**: Mapping between field names and FieldInfo instances
- **`model_computed_fields`**: Mapping of computed field definitions
- **`model_extra`**: Extra fields set during validation
- **`model_fields_set`**: Fields explicitly provided during initialization

## Data Conversion and Type Coercion

Pydantic performs intelligent type conversion:

```python
from pydantic import BaseModel

class Model(BaseModel):
    a: int
    b: float
    c: str

print(Model(a=3.000, b='2.72', c=b'binary data').model_dump())
#> {'a': 3, 'b': 2.72, 'c': 'binary data'}
```

**Important**: This may result in information loss but is deliberate and usually beneficial.

**Strict Mode Alternative**: Use `strict_mode` for no data conversion - values must match declared field types exactly.

## Extra Data Handling

### Default Behavior (Ignore)
```python
from pydantic import BaseModel

class Model(BaseModel):
    x: int

m = Model(x=1, y='a')  # 'y' is ignored
assert m.model_dump() == {'x': 1}
```

### Configuration Options
```python
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    x: int
    model_config = ConfigDict(extra='allow')  # or 'forbid'

m = Model(x=1, y='a')
assert m.model_dump() == {'x': 1, 'y': 'a'}
assert m.__pydantic_extra__ == {'y': 'a'}
```

**Extra Configuration Values**:
- `'ignore'`: Extra data ignored (default)
- `'forbid'`: Extra data not permitted
- `'allow'`: Extra data allowed and stored in `__pydantic_extra__`

## Nested Models

```python
from pydantic import BaseModel

class Foo(BaseModel):
    count: int
    size: float | None = None

class Bar(BaseModel):
    apple: str = 'x'
    banana: str = 'y'

class Spam(BaseModel):
    foo: Foo
    bars: list[Bar]

m = Spam(foo={'count': 4}, bars=[{'apple': 'x1'}, {'apple': 'x2'}])
```

**Self-referencing models** are supported using forward annotations.

## Validation Methods

### Three Primary Validation Methods

1. **`model_validate()`**: Dictionary or object input
```python
User.model_validate({'id': 123, 'name': 'James'})
```

2. **`model_validate_json()`**: JSON string/bytes input
```python
User.model_validate_json('{"id": 123, "name": "James"}')
```

3. **`model_validate_strings()`**: String dictionary with type coercion
```python
User.model_validate_strings({'id': '123', 'name': 'James'})
```

### Error Handling
```python
from pydantic import ValidationError

try:
    User.model_validate(['not', 'a', 'dict'])
except ValidationError as e:
    print(e.errors())  # Detailed error information
```

## Advanced Features

### Generic Models

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

DataT = TypeVar('DataT')

class Response(BaseModel, Generic[DataT]):
    data: DataT

# Usage
print(Response[int](data=1))  # Parametrized with int
print(Response[str](data='value'))  # Parametrized with str
```

### Dynamic Model Creation

```python
from pydantic import create_model

DynamicModel = create_model(
    'DynamicModel',
    foo=str,
    bar=(int, 123)  # (type, default_value)
)
```

### RootModel for Custom Root Types

```python
from pydantic import RootModel

Pets = RootModel[list[str]]
PetsByName = RootModel[dict[str, str]]

print(Pets(['dog', 'cat']))
#> root=['dog', 'cat']
```

### Immutable Models

```python
from pydantic import BaseModel, ConfigDict

class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    a: str
    b: dict

# Attempting to modify will raise ValidationError
```

### Private Attributes

```python
from pydantic import BaseModel, PrivateAttr
from datetime import datetime

class TimeAwareModel(BaseModel):
    _processed_at: datetime = PrivateAttr(default_factory=datetime.now)
    _secret_value: str
    
    def model_post_init(self, context):
        self._secret_value = "secret"
```

## Integration Patterns for Quant Trading

### Trading Configuration Model
```python
from pydantic import BaseModel, Field
from typing import Literal

class TradingConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,  # Immutable configuration
        validate_assignment=True  # Validate on assignment
    )
    
    symbol: str = Field(..., regex=r'^[A-Z]{3,10}$')
    max_position_size: float = Field(..., gt=0)
    risk_threshold: float = Field(..., ge=0.0, le=1.0)
    strategy_type: Literal['momentum', 'mean_reversion', 'arbitrage']
```

### Market Data Model
```python
from pydantic import BaseModel, validator
from datetime import datetime

class MarketData(BaseModel):
    symbol: str
    price: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    timestamp: datetime
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper()
    
    class Config:
        # Allow ORM integration (SQLAlchemy)
        from_attributes = True
```

### Strategy Parameters with Validation
```python
class StrategyParams(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,  # Real-time validation
        extra='forbid'  # Strict parameter control
    )
    
    lookback_period: int = Field(..., ge=1, le=1000)
    sharpe_threshold: float = Field(..., ge=0.5)
    max_drawdown: float = Field(..., ge=0.01, le=0.5)
    kelly_fraction: float = Field(0.25, ge=0.01, le=0.5)
    
    @validator('kelly_fraction')
    def validate_kelly(cls, v, values):
        # Custom validation logic for Kelly criterion
        return min(v, 0.25)  # Cap at 25% for safety
```

### Nested Order Management
```python
class OrderRequest(BaseModel):
    symbol: str
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    order_type: Literal['market', 'limit', 'stop']

class Portfolio(BaseModel):
    cash_balance: float = Field(..., ge=0)
    positions: dict[str, float] = {}
    pending_orders: list[OrderRequest] = []
    
    @validator('positions')
    def validate_positions(cls, v):
        # Ensure no negative positions in this strategy
        return {k: max(0, val) for k, val in v.items()}
```

## Performance Considerations

### Model Construction vs Validation
- **`model_construct()`**: Fastest, no validation (use with trusted data)
- **`model_validate()`**: Full validation with type conversion
- **Direct instantiation**: `Model(**data)` - standard approach

### Revalidation Control
```python
class Model(BaseModel):
    model_config = ConfigDict(
        revalidate_instances='always'  # Force revalidation of model instances
    )
```

## Key Advantages for Trading System

1. **Type Safety**: Guarantees data integrity across trading pipeline
2. **Performance**: Rust-based validation suitable for real-time trading
3. **Configuration Management**: Immutable, validated trading parameters
4. **Error Handling**: Detailed validation errors for debugging
5. **ORM Integration**: `from_attributes=True` for database models
6. **JSON Schema**: Automatic API documentation generation
7. **Custom Validation**: Domain-specific trading logic validation
8. **Nested Models**: Complex portfolio and strategy hierarchies

## Next Implementation Steps

1. **Core Models**: Create BaseModel classes for all trading entities
2. **Validation Logic**: Implement custom validators for trading rules
3. **Configuration**: Use frozen models for immutable trading config
4. **Error Handling**: Leverage ValidationError for robust error reporting
5. **Performance**: Use model_construct() for high-frequency operations
6. **Integration**: Enable from_attributes for ORM/database integration