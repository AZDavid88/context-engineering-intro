# aiofiles API Specifications (Vector 2)

## Core API Overview

aiofiles provides a comprehensive async file I/O API that mirrors Python's standard file operations while ensuring non-blocking behavior through thread pool delegation.

## Primary API Functions

### 1. `aiofiles.open()` - Core File Opening Function

```python
async def open(
    file: Union[str, bytes, int, os.PathLike],
    mode: str = 'r',
    buffering: int = -1,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    closefd: bool = True,
    opener: Optional[Callable] = None,
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    executor: Optional[Executor] = None
) -> AsyncFile
```

**Parameters:**
- `file`: File path, file descriptor, or path-like object
- `mode`: File mode ('r', 'w', 'a', 'x', 'b', 't', '+')
- `buffering`: Buffer size (-1 for default, 0 for unbuffered, >0 for specific size)
- `encoding`: Text encoding (None for binary mode)
- `errors`: Error handling strategy
- `newline`: Newline handling for text mode
- `closefd`: Whether to close file descriptor on file close
- `opener`: Custom file opener callable
- `loop`: Specific asyncio event loop (defaults to current running loop)
- `executor`: Custom thread pool executor (defaults to loop's default executor)

**Returns:** Async file object with standard file interface

**Usage Examples:**
```python
# Basic text file reading
async with aiofiles.open('data.txt', mode='r') as f:
    content = await f.read()

# Binary file with custom executor
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
async with aiofiles.open('data.bin', mode='rb', executor=executor) as f:
    data = await f.read(1024)

# Custom buffering for performance
async with aiofiles.open('large_file.dat', mode='rb', buffering=65536) as f:
    chunk = await f.read(8192)
```

## Async File Object Methods

### Reading Operations
```python
async def read(size: int = -1) -> Union[str, bytes]
async def read1(size: int = -1) -> bytes  # Binary mode only
async def readall() -> bytes              # Binary mode only
async def readinto(b: memoryview) -> int  # Binary mode only
async def readline(size: int = -1) -> Union[str, bytes]
async def readlines(hint: int = -1) -> List[Union[str, bytes]]
```

### Writing Operations
```python
async def write(data: Union[str, bytes]) -> int
async def writelines(lines: Iterable[Union[str, bytes]]) -> None
async def flush() -> None
```

### File Positioning
```python
async def seek(offset: int, whence: int = 0) -> int
async def tell() -> int
async def truncate(size: Optional[int] = None) -> int
```

### File State Operations
```python
async def close() -> None
async def readable() -> bool
async def writable() -> bool
async def seekable() -> bool
async def isatty() -> bool
```

### Async Context Manager Support
```python
async def __aenter__(self) -> 'AsyncFile'
async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
```

### Async Iteration Support
```python
def __aiter__(self) -> 'AsyncFile'
async def __anext__(self) -> Union[str, bytes]
```

## Standard Streams API

### Text Streams
```python
aiofiles.stdin: AsyncTextIOWrapper     # Async sys.stdin
aiofiles.stdout: AsyncTextIOWrapper    # Async sys.stdout
aiofiles.stderr: AsyncTextIOWrapper    # Async sys.stderr
```

### Binary Streams
```python
aiofiles.stdin_bytes: AsyncBufferedReader   # Async sys.stdin.buffer
aiofiles.stdout_bytes: AsyncBufferedWriter  # Async sys.stdout.buffer
aiofiles.stderr_bytes: AsyncBufferedWriter  # Async sys.stderr.buffer
```

**Usage Examples:**
```python
# Async stdin reading
async for line in aiofiles.stdin:
    await process_line(line)

# Async stdout writing
await aiofiles.stdout.write("Processing complete\n")
await aiofiles.stdout.flush()

# Binary stream operations
data = await aiofiles.stdin_bytes.read(1024)
await aiofiles.stdout_bytes.write(processed_data)
```

## OS Operations API (`aiofiles.os`)

### File Management
```python
async def remove(path: PathLike) -> None
async def unlink(path: PathLike) -> None  # Alias for remove
async def rename(src: PathLike, dst: PathLike) -> None
async def renames(old: PathLike, new: PathLike) -> None
async def replace(src: PathLike, dst: PathLike) -> None
```

### Directory Operations
```python
async def mkdir(path: PathLike, mode: int = 0o777, *, dir_fd: Optional[int] = None) -> None
async def makedirs(name: PathLike, mode: int = 0o777, exist_ok: bool = False) -> None
async def rmdir(path: PathLike, *, dir_fd: Optional[int] = None) -> None
async def removedirs(name: PathLike) -> None
async def listdir(path: PathLike = '.') -> List[str]
async def scandir(path: PathLike = '.') -> AsyncIterator[os.DirEntry]
```

### File Metadata and Status
```python
async def stat(path: PathLike, *, dir_fd: Optional[int] = None, follow_symlinks: bool = True) -> os.stat_result
async def access(path: PathLike, mode: int, *, dir_fd: Optional[int] = None, effective_ids: bool = False, follow_symlinks: bool = True) -> bool
async def getcwd() -> str
```

### Link Operations (Platform Dependent)
```python
async def link(src: PathLike, dst: PathLike, *, src_dir_fd: Optional[int] = None, dst_dir_fd: Optional[int] = None, follow_symlinks: bool = True) -> None
async def symlink(src: PathLike, dst: PathLike, target_is_directory: bool = False, *, dir_fd: Optional[int] = None) -> None
async def readlink(path: PathLike, *, dir_fd: Optional[int] = None) -> str
```

### Path Operations (`aiofiles.os.path`)
```python
async def exists(path: PathLike) -> bool
async def isfile(path: PathLike) -> bool
async def isdir(path: PathLike) -> bool
async def islink(path: PathLike) -> bool
async def ismount(path: PathLike) -> bool
async def getsize(path: PathLike) -> int
async def getatime(path: PathLike) -> float
async def getmtime(path: PathLike) -> float
async def getctime(path: PathLike) -> float
async def samefile(path1: PathLike, path2: PathLike) -> bool
async def sameopenfile(fp1: int, fp2: int) -> bool
async def abspath(path: PathLike) -> str
```

## Temporary File API (`aiofiles.tempfile`)

### Temporary File Classes
```python
class TemporaryFile:
    """Async version of tempfile.TemporaryFile"""
    
class NamedTemporaryFile:
    """Async version of tempfile.NamedTemporaryFile"""
    
class SpooledTemporaryFile:
    """Async version of tempfile.SpooledTemporaryFile"""
    
class TemporaryDirectory:
    """Async version of tempfile.TemporaryDirectory"""
```

**Usage Examples:**
```python
# Temporary file operations
async with aiofiles.tempfile.TemporaryFile('wb+') as f:
    await f.write(b'temporary data')
    await f.seek(0)
    data = await f.read()

# Named temporary file
async with aiofiles.tempfile.NamedTemporaryFile('w+', delete=False) as f:
    await f.write('temp content')
    temp_name = f.name

# Temporary directory
async with aiofiles.tempfile.TemporaryDirectory() as temp_dir:
    file_path = os.path.join(temp_dir, 'temp_file.txt')
    async with aiofiles.open(file_path, 'w') as f:
        await f.write('temporary file content')
```

## Performance Optimization Parameters

### Thread Pool Configuration
```python
# Custom executor for high-performance scenarios
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix='aiofiles'
)

async with aiofiles.open('data.bin', mode='rb', executor=executor) as f:
    data = await f.read()
```

### Buffering Strategies
```python
# High-performance binary reading with large buffer
async with aiofiles.open('large_file.dat', mode='rb', buffering=1048576) as f:  # 1MB buffer
    while True:
        chunk = await f.read(65536)  # 64KB chunks
        if not chunk:
            break
        await process_chunk(chunk)

# Unbuffered writes for immediate disk sync
async with aiofiles.open('log.txt', mode='w', buffering=0) as f:
    await f.write('critical log entry\n')
```

## Error Handling

### Standard Exception Propagation
```python
try:
    async with aiofiles.open('nonexistent.txt') as f:
        content = await f.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
except OSError as e:
    print(f"OS error: {e}")
```

### Async Context Manager Error Handling
```python
async with aiofiles.open('file.txt', 'w') as f:
    try:
        await f.write('data')
    except Exception as e:
        # File will be properly closed even if exception occurs
        print(f"Write failed: {e}")
        raise
```

## Integration Patterns for High-Performance Applications

### AsyncIO Data Pipeline Integration
```python
import asyncio
import aiofiles

async def process_file_pipeline(file_path: str) -> None:
    """High-performance async file processing pipeline"""
    async with aiofiles.open(file_path, 'rb', buffering=1048576) as f:
        while True:
            chunk = await f.read(65536)
            if not chunk:
                break
            
            # Process chunk asynchronously
            await asyncio.create_task(process_data_chunk(chunk))

async def concurrent_file_processing(file_paths: List[str]) -> None:
    """Process multiple files concurrently"""
    tasks = [process_file_pipeline(path) for path in file_paths]
    await asyncio.gather(*tasks)
```

### Producer-Consumer Pattern with aiofiles
```python
import asyncio
import aiofiles

async def file_producer(queue: asyncio.Queue, file_path: str) -> None:
    """Produce data from file to queue"""
    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(8192)
            if not chunk:
                break
            await queue.put(chunk)
    await queue.put(None)  # Sentinel

async def data_consumer(queue: asyncio.Queue) -> None:
    """Consume and process data from queue"""
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        await process_chunk(chunk)
        queue.task_done()
```

This API specification provides comprehensive coverage of aiofiles functionality with performance optimization patterns suitable for high-performance trading system data pipelines.