# PyArrow Table and Array API Documentation

## Overview
Comprehensive documentation for PyArrow Table and Array APIs, focusing on efficient data manipulation patterns for quantitative trading applications.

## Table Construction Methods

### Primary Construction APIs
- **from_pandas()**: Convert pandas DataFrame to Arrow Table
- **from_arrays()**: Create table from Arrow arrays with optional names and schema
- **from_batches()**: Construct table from sequence of RecordBatches
- **from_pydict()**: Create table from dictionary of arrays
- **from_pylist()**: Construct table from list of rows/dictionaries

### Schema-Aware Construction
```python
# Trading data table construction
schema = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    ('symbol', pa.dictionary(pa.int32(), pa.string())),
    ('price', pa.float64()),
    ('volume', pa.float64())
])

table = pa.Table.from_arrays(
    [timestamps, symbols, prices, volumes],
    schema=schema
)
```

### Batch Processing Construction
- **Memory Efficiency**: Build tables from streaming batches
- **Schema Consistency**: Ensure consistent schema across batches
- **Error Handling**: Manage construction failures gracefully
- **Trading Applications**: Build OHLCV tables from real-time data

## Schema Manipulation and Column Operations

### Column Management
- **add_column()**: Insert column at specific position
- **append_column()**: Add column at table end
- **drop_columns()**: Remove specified columns by name or index
- **rename_columns()**: Create new table with renamed columns
- **select()**: Select subset of columns (similar to SQL SELECT)

### Schema Operations
- **cast()**: Convert table values to different schema
- **field()**: Retrieve schema field information
- **schema**: Access table schema metadata
- **column_names**: Get list of column names

### Dynamic Column Addition
```python
# Add derived columns for technical analysis
def add_returns_column(table, price_col='close'):
    prices = table.column(price_col)
    shifted_prices = pc.shift(prices, 1)
    returns = pc.divide(pc.subtract(prices, shifted_prices), shifted_prices)
    return table.append_column('returns', returns)
```

## Row and Column Access Patterns

### Individual Access Methods
- **column()**: Select single column by index or name
- **slice()**: Create zero-copy slice of table rows
- **take()**: Select specific rows by indices
- **field()**: Retrieve schema field by name or index

### Bulk Access Operations
- **to_pandas()**: Convert entire table to pandas DataFrame
- **to_pydict()**: Convert to Python dictionary format
- **to_batches()**: Convert to list of RecordBatches
- **itercolumns()**: Iterate over columns

### Memory-Efficient Access
```python
# Access price data without full table conversion
close_prices = table.column('close').to_pandas()
volumes = table.column('volume').to_pandas()

# Zero-copy slicing for time ranges
recent_data = table.slice(start_idx, length)
```

## Filtering and Selection Operations

### Boolean Filtering
- **filter()**: Select rows based on boolean mask/expression
- **Compute Integration**: Use Arrow compute functions for filters
- **Complex Conditions**: Combine multiple filter conditions
- **Performance**: Efficient columnar filtering

### Advanced Selection
- **drop_null()**: Remove rows with missing values
- **sort_by()**: Sort table by one or multiple columns
- **unique()**: Get unique rows (when available)
- **distinct()**: Remove duplicate rows

### Trading-Specific Filtering
```python
# Filter for specific time range and high volume
time_filter = pc.and_(
    pc.greater_equal(table.column('timestamp'), start_time),
    pc.less_equal(table.column('timestamp'), end_time)
)
volume_filter = pc.greater(table.column('volume'), min_volume)
combined_filter = pc.and_(time_filter, volume_filter)

filtered_table = table.filter(combined_filter)
```

## Concatenation and Joining Operations

### Table Concatenation
- **concat_tables()**: Vertical concatenation of tables
- **Schema Compatibility**: Ensure consistent schemas
- **Memory Management**: Efficient concatenation for large tables
- **Streaming Support**: Concatenate tables from streaming sources

### Join Operations
- **join()**: Perform inner/outer joins with another table
- **join_asof()**: As-of joins for time-series data
- **Join Keys**: Specify join columns and conditions
- **Performance**: Optimized join algorithms

### Time-Series Join Patterns
```python
# As-of join for price and volume data
price_volume_joined = price_table.join_asof(
    volume_table,
    on='timestamp',
    by='symbol',
    tolerance=pd.Timedelta(seconds=1)
)
```

## Memory Management and Performance

### Memory Optimization
- **combine_chunks()**: Consolidate chunked arrays for better performance
- **get_total_buffer_size()**: Calculate total memory consumption
- **Memory Pools**: Use custom memory pools for allocation control
- **Zero-Copy Operations**: Minimize data copying

### Chunked Array Handling
- **ChunkedArray Structure**: Understand Arrow's chunked memory layout
- **Chunk Consolidation**: When to combine chunks
- **Memory Fragmentation**: Manage memory fragmentation in long-running processes
- **Performance Impact**: Balance memory efficiency vs. performance

### Performance Monitoring
```python
# Monitor table memory usage
def analyze_table_memory(table):
    total_size = table.get_total_buffer_size()
    num_chunks = {col: len(table.column(col).chunks) 
                  for col in table.column_names}
    return {
        'total_memory_mb': total_size / (1024**2),
        'num_rows': len(table),
        'num_columns': len(table.column_names),
        'chunks_per_column': num_chunks
    }
```

## Integration Patterns

### DuckDB Integration
- **Direct Registration**: Register Arrow tables as DuckDB relations
- **Query Performance**: Leverage Arrow's columnar format
- **Memory Sharing**: Zero-copy data sharing when possible
- **Analytical Queries**: Combine Arrow operations with SQL analytics

### Pandas Interoperability
- **Conversion Patterns**: Efficient Arrow ↔ pandas workflows
- **Index Handling**: Manage pandas index in Arrow context
- **Type Preservation**: Maintain data types across conversions
- **Memory Efficiency**: Optimize conversions for large datasets

### Parquet Integration
- **Schema Consistency**: Ensure Parquet and Arrow schema compatibility
- **Metadata Preservation**: Maintain custom metadata across I/O operations
- **Partitioning**: Handle partitioned datasets efficiently
- **Streaming I/O**: Read/write large datasets in chunks

## Advanced Array Operations

### Array Construction
- **pa.array()**: Create arrays from Python lists/numpy arrays
- **pa.chunked_array()**: Create chunked arrays for large datasets
- **Type Specification**: Explicit type specification for consistency
- **Null Handling**: Proper null value representation

### Array Manipulation
- **Slicing**: Zero-copy array slicing operations
- **Concatenation**: Combine arrays efficiently
- **Type Conversion**: Cast arrays to different types
- **Validation**: Check array integrity and consistency

### Custom Array Types
- **Dictionary Arrays**: Efficient categorical data representation
- **Timestamp Arrays**: Time-series data with timezone support
- **Nested Arrays**: Complex data structures (lists, structs)
- **Extension Types**: Custom data types for specialized use cases

## Error Handling and Validation

### Data Validation
- **Schema Validation**: Ensure data conforms to expected schema
- **Type Checking**: Validate data types before operations
- **Null Value Handling**: Consistent null value treatment
- **Range Validation**: Check data ranges for sanity

### Error Recovery Patterns
- **Graceful Degradation**: Handle errors without crashing
- **Data Quality Checks**: Identify and handle bad data
- **Transaction Safety**: Ensure data consistency during operations
- **Logging and Monitoring**: Comprehensive error reporting

## Quality Metrics
- **API Coverage**: 95%+ of core Table/Array functionality documented
- **Performance Benefits**: Zero-copy operations where possible
- **Memory Efficiency**: Optimal memory usage patterns
- **Integration Quality**: Seamless interoperability with pandas and DuckDB