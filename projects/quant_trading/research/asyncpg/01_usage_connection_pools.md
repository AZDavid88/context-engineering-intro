# AsyncPG Usage & Connection Pools - Essential for Phase 4 Neon Integration

**Source**: https://magicstack.github.io/asyncpg/current/usage.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - Neon Cloud Database Integration with Ray Workers

## Core AsyncPG Usage Patterns

### Database Connection
The interaction with the database normally starts with a call to `connect()`, which establishes a new database session and returns a new `Connection` instance, which provides methods to run queries and manage transactions.

```python
import asyncio
import asyncpg
import datetime

async def main():
    # Establish a connection to an existing database named "test"
    # as a "postgres" user.
    conn = await asyncpg.connect('postgresql://postgres@localhost/test')
    # Execute a statement to create a new table.
    await conn.execute('''
 CREATE TABLE users(
 id serial PRIMARY KEY,
 name text,
 dob date
 )
 ''')

    # Insert a record into the created table.
    await conn.execute('''
 INSERT INTO users(name, dob) VALUES($1, $2)
 ''', 'Bob', datetime.date(1984, 3, 1))

    # Select a row from the table.
    row = await conn.fetchrow(
        'SELECT * FROM users WHERE name = $1', 'Bob')
    # *row* now contains
    # asyncpg.Record(id=1, name='Bob', dob=datetime.date(1984, 3, 1))

    # Close the connection.
    await conn.close()

asyncio.run(main())
```

**Note**: asyncpg uses the native PostgreSQL syntax for query arguments: `$n`.

## Connection Pools - CRITICAL for Phase 4 Ray Workers

For server-type type applications, that handle frequent requests and need the database connection for a short period time while handling a request, the use of a connection pool is recommended. asyncpg provides an advanced pool implementation, which eliminates the need to use an external connection pooler such as PgBouncer.

To create a connection pool, use the `asyncpg.create_pool()` function. The resulting `Pool` object can then be used to borrow connections from the pool.

### Connection Pool Example for Web Services

Below is an example of how **asyncpg** can be used to implement a simple Web service that computes the requested power of two.

```python
import asyncio
import asyncpg
from aiohttp import web

async def handle(request):
 """Handle incoming requests."""
    pool = request.app['pool']
    power = int(request.match_info.get('power', 10))

    # Take a connection from the pool.
    async with pool.acquire() as connection:
        # Open a transaction.
        async with connection.transaction():
            # Run the query passing the request argument.
            result = await connection.fetchval('select 2 ^ $1', power)
            return web.Response(
                text="2 ^ {} is {}".format(power, result))

async def init_db(app):
 """Initialize a connection pool."""
     app['pool'] = await asyncpg.create_pool(database='postgres',
                                             user='postgres')
     yield
     await app['pool'].close()

def init_app():
 """Initialize the application server."""
    app = web.Application()
    # Create a database context
    app.cleanup_ctx.append(init_db)
    # Configure service routes
    app.router.add_route('GET', '/{power:\\d+}', handle)
    app.router.add_route('GET', '/', handle)
    return app

app = init_app()
web.run_app(app)
```

### Phase 4 Implementation Patterns

**Critical Connection Pool Pattern for Ray Workers:**

```python
import asyncpg
import asyncio

# Pattern for Phase 4 Ray Worker Connection Management
class NeonConnectionManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize_pool(self):
        """Initialize AsyncPG connection pool with Neon optimizations."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,      # Minimum connections per Ray worker
            max_size=20,     # Maximum connections per Ray worker
            command_timeout=30,
            server_settings={
                'application_name': 'quant_trading_ray_worker'
            }
        )
    
    async def acquire_connection(self):
        """Acquire connection for Ray worker operations."""
        if not self.pool:
            await self.initialize_pool()
        
        return self.pool.acquire()
```

## Transactions

To create transactions, the `Connection.transaction()` method should be used.

The most common way to use transactions is through an `async with` statement:

```python
async with connection.transaction():
    await connection.execute("INSERT INTO mytable VALUES(1, 2, 3)")
```

**Note**: When not in an explicit transaction block, any changes to the database will be applied immediately. This is also known as _auto-commit_.

## Type Conversion

asyncpg automatically converts PostgreSQL types to the corresponding Python types and vice versa. All standard data types are supported out of the box, including arrays, composite types, range types, enumerations and any combination of them.

### Key Type Mappings for Trading Data:
| PostgreSQL Type | Python Type |
|---|---|
| `timestamp with time zone` | offset-aware `datetime.datetime` |
| `numeric` | `Decimal` |
| `float`, `double precision` | `float` |
| `smallint`, `integer`, `bigint` | `int` |
| `json`, `jsonb` | `str` |
| `uuid` | `uuid.UUID` |
| `bool` | `bool` |

## Custom Type Conversions

asyncpg allows defining custom type conversion functions both for standard and user-defined types using the `Connection.set_type_codec()` and `Connection.set_builtin_type_codec()` methods.

### Example: Decoding numeric columns as floats
```python
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect()
    try:
        await conn.set_type_codec(
            'numeric', encoder=str, decoder=float,
            schema='pg_catalog', format='text'
        )
        res = await conn.fetchval("SELECT $1::numeric", 11.123)
        print(res, type(res))
    finally:
        await conn.close()

asyncio.run(main())
```

## Phase 4 Integration Notes

1. **Connection Pooling**: Essential for Ray workers to efficiently manage Neon connections
2. **Transaction Management**: Use async context managers for reliable transaction handling
3. **Type Handling**: Automatic PostgreSQL type conversion works well with trading data types
4. **Query Parameterization**: Always use `$n` parameterization for security and performance
5. **Connection Lifecycle**: Proper initialization and cleanup of connection pools in distributed environments

See [Connection Pools API documentation](https://magicstack.github.io/asyncpg/current/api/index.html#asyncpg-api-pool) for more information.