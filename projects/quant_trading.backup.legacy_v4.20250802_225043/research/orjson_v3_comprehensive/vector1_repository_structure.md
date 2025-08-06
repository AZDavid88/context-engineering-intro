# Vector 1: orjson Repository Structure Analysis

## Repository Overview
- **GitHub Repository**: https://github.com/ijl/orjson
- **Stars**: 7.2k stars, 251 forks
- **Primary Language**: Python (56.0%), Rust (42.8%)
- **License**: Apache-2.0, MIT dual license

## Core Library Features

### Performance Characteristics
- **JSON Serialization**: ~10x faster than standard `json` library
- **JSON Deserialization**: ~2x faster than standard `json` library
- **Native Type Support**: dataclass, datetime, numpy, UUID instances
- **Memory Efficiency**: Maintains key cache to reduce memory usage
- **Thread Safety**: GIL held during operations

### Key Library Functions
```python
def dumps(
    __obj: Any,
    default: Optional[Callable[[Any], Any]] = ...,
    option: Optional[int] = ...,
) -> bytes: ...

def loads(__obj: Union[bytes, bytearray, memoryview, str]) -> Any: ...
```

## Directory Structure Analysis

### Core Implementation
```
/src/                    # Rust implementation (42.8% of codebase)
/pysrc/orjson/          # Python bindings and interface
/include/               # C headers for integration
/build.rs               # Rust build configuration
```

### Performance & Benchmarking Infrastructure
```
/bench/                 # Performance benchmarking suite
├── benchmark_dumps.py  # JSON serialization performance tests
├── benchmark_loads.py  # JSON parsing/deserialization speed tests  
├── benchmark_empty.py  # Edge cases and minimal JSON processing
├── data.py            # Test data fixtures
├── util.py            # Benchmark utilities
├── run_default        # Default benchmark execution
├── run_func           # Function-specific benchmarks
├── run_mem            # Memory usage benchmarks
└── requirements.txt   # Benchmark dependencies
```

### Comprehensive Test Suite
```
/test/                  # 35+ test files covering all aspects
├── test_parsing.py     # Core JSON parsing validation
├── test_roundtrip.py   # Serialization/deserialization consistency
├── test_canonical.py   # RFC 8259 compliance testing
├── test_dict.py        # Dictionary serialization patterns
├── test_type.py        # Type handling validation
├── test_datetime.py    # DateTime serialization (RFC 3339)
├── test_enum.py        # Enum serialization patterns
├── test_uuid.py        # UUID serialization validation
├── test_dataclass.py   # Dataclass native serialization
├── test_typeddict.py   # TypedDict support testing
├── test_error.py       # Error handling and edge cases
├── test_circular.py    # Circular reference detection
├── test_non_str_keys.py# Non-string dictionary keys
├── test_escape.py      # UTF-8 character escaping
└── test_fragment.py    # Fragment inclusion for pre-serialized JSON
```

### Integration & CI/CD
```
/ci/                    # Continuous integration configurations
/.github/               # GitHub Actions workflows
/integration/           # Integration testing framework
/script/                # Build and deployment scripts
```

### Documentation & Data
```
/doc/                   # Performance graphs and documentation
/data/                  # Test fixtures and sample data
/CHANGELOG.md          # Version history and changes
/CONTRIBUTING.md       # Development guidelines
```

## Key Performance Optimizations

### Serialization Options (Flags)
```python
# Performance optimization flags
OPT_APPEND_NEWLINE      # Append \n without copying
OPT_INDENT_2           # Pretty-print with 2-space indent
OPT_NAIVE_UTC          # Treat naive datetime as UTC
OPT_NON_STR_KEYS       # Allow non-string dict keys
OPT_OMIT_MICROSECONDS  # Remove microseconds from datetime
OPT_PASSTHROUGH_DATACLASS  # Custom dataclass handling
OPT_PASSTHROUGH_DATETIME   # Custom datetime formatting
OPT_PASSTHROUGH_SUBCLASS   # Custom subclass handling
OPT_SERIALIZE_NUMPY    # Native numpy array serialization
OPT_SERIALIZE_UUID     # Native UUID serialization
OPT_SORT_KEYS          # Sort dictionary keys
OPT_STRICT_INTEGER     # 53-bit integer limit
OPT_UTC_Z              # Use Z suffix for UTC timezone
```

### Memory & Performance Features
- **Zero-copy operations**: Direct memory access patterns
- **Native type serialization**: No Python object conversion overhead
- **Key caching**: Reduces memory usage for repeated keys
- **UTF-8 validation**: Strict conformance checking
- **C array support**: Contiguous memory layout optimization

## High-Frequency Trading Relevance

### WebSocket Message Processing
- **Native bytes output**: No string conversion overhead
- **High-throughput parsing**: 2x faster JSON deserialization
- **Memory efficiency**: Key caching reduces allocation pressure
- **Fragment support**: Pre-serialized JSON blob inclusion

### AsyncIO Integration Potential
- **Non-blocking I/O friendly**: bytes output compatible with async patterns
- **GIL management**: Predictable lock behavior for async applications
- **Memory predictability**: Consistent allocation patterns

### Production Usage Patterns
- **Error handling**: JSONEncodeError/JSONDecodeError with proper chaining
- **Type safety**: Strict UTF-8 validation prevents corruption
- **Performance monitoring**: Comprehensive benchmark suite
- **Version stability**: Semantic versioning with breaking change protection

## Missing Gaps for Trading Applications

### AsyncIO Integration
- **No native async support**: Library is synchronous only
- **No async examples**: No AsyncIO usage patterns documented
- **Thread safety**: Only mentions GIL behavior, not async-specific patterns

### Streaming JSON Processing
- **No streaming API**: Only supports complete document processing
- **Memory bound**: Must load entire JSON document in memory
- **No line-delimited JSON**: Explicitly states NDJSON not supported

### High-Frequency Specific Features
- **No schema validation**: Requires external validation layer
- **No custom deserializers**: Cannot deserialize to custom types
- **No object hooks**: No deserialization customization

## Performance Benchmarks (from README)

### Serialization Performance vs json
- twitter.json: 11.1x faster
- github.json: 13.6x faster  
- citm_catalog.json: 11.8x faster
- canada.json: 11.9x faster

### Deserialization Performance vs json
- twitter.json: 4.2x faster
- github.json: 2.2x faster
- citm_catalog.json: 3.1x faster
- canada.json: 6x faster

### Numpy Array Serialization
- 92MiB float64 array: 14.2x faster than json
- 100MiB int32 array: 10.1x faster than json
- 105MiB bool array: 11.5x faster than json

## Next Research Vectors Required

1. **Vector 2**: Detailed API documentation and usage patterns
2. **Vector 3**: Performance benchmarks and memory optimization techniques
3. **Vector 4**: Production usage patterns and AsyncIO integration strategies
4. **Vector 5**: High-frequency trading specific optimizations and patterns