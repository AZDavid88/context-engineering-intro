"""
Pytest Configuration - Optimized Test Infrastructure

Implements shared fixtures and optimizations for fast test execution
following the Phase 4 remediation plan specifications.

Key Features:
- Shared resource pool for all tests (eliminates session creation overhead)
- Connection pooling for tests with automatic cleanup
- Test isolation and resource management  
- Parallel test execution configuration
- Performance monitoring and regression detection

Based on Phase 4 plan specifications:
- pytest fixtures for shared resources
- Connection pooling for tests
- Test isolation and cleanup
- Parallel test execution configuration
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Test infrastructure imports
from src.config.settings import get_settings, Settings
from src.infrastructure.shared_resource_pool import SharedResourcePool
from src.data.fear_greed_client import FearGreedClient
from tests.utils.market_data_fixtures import create_test_market_data
from src.strategy.genetic_seeds.base_seed import SeedGenes, SeedType

# Configure test logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise during tests
    format='%(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging for specific modules during tests
logging.getLogger('src.data.fear_greed_client').setLevel(logging.ERROR)
logging.getLogger('aiohttp').setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def shared_resource_pool():
    """
    Shared resource pool for all tests.
    
    Eliminates the overhead of creating new connection pools and
    FearGreedClient instances for each test.
    """
    settings = get_test_settings()
    pool = SharedResourcePool(settings)
    
    start_time = time.perf_counter()
    await pool.initialize()
    init_time = time.perf_counter() - start_time
    
    print(f"\\n🚀 Test resource pool initialized in {init_time:.2f}s")
    
    yield pool
    
    # Cleanup
    await pool.cleanup()
    print("\\n🧹 Test resource pool cleaned up")


@pytest.fixture(scope="session")
async def shared_fear_greed_client(shared_resource_pool):
    """Shared FearGreedClient for tests."""
    return await shared_resource_pool.get_fear_greed_client()


@pytest.fixture(scope="function")
async def test_market_data():
    """Pre-generated market data for tests."""
    return create_test_market_data(days=30, seed=42)


@pytest.fixture(scope="function")
def test_seed_genes():
    """Standard test genetic parameters for seed testing."""
    return SeedGenes(
        seed_id="test_seed_001",
        seed_type=SeedType.MOMENTUM,
        parameters={
            'fast_ema_period': 10.0,
            'slow_ema_period': 30.0,
            'momentum_threshold': 0.005,
            'signal_strength': 1.0,
            'trend_filter': 0.5
        },
        fitness_score=0.75,
        generation=1
    )


@pytest.fixture(scope="session")
def test_settings():
    """Optimized test settings."""
    return get_test_settings()


def get_test_settings() -> Settings:
    """Get optimized settings for test environment."""
    settings = get_settings()
    
    # Override with test-optimized values
    settings.fear_greed_cache_ttl = 300  # 5 minutes cache for tests
    settings.http_timeout = 10.0  # Shorter timeouts for tests
    settings.max_retries = 2  # Fewer retries for faster test execution
    settings.retry_delay = 0.5  # Faster retry delays
    
    return settings


# Performance monitoring fixture
@pytest.fixture(scope="function", autouse=True)
def test_performance_monitor(request):
    """Monitor test execution performance and detect regressions."""
    start_time = time.perf_counter()
    
    yield
    
    execution_time = time.perf_counter() - start_time
    
    # Log slow tests (> 5 seconds)
    if execution_time > 5.0:
        print(f"\\n⚠️ Slow test detected: {request.node.name} took {execution_time:.2f}s")
    
    # Store performance data for regression analysis
    if hasattr(request.config, 'test_performance_data'):
        request.config.test_performance_data[request.node.name] = execution_time


def pytest_configure(config):
    """Pytest configuration hook."""
    # Initialize performance tracking
    config.test_performance_data = {}
    
    # Configure test markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for optimization."""
    # Auto-mark slow tests based on naming patterns
    for item in items:
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
            
        if "comprehensive" in item.name or "system" in item.name:
            item.add_marker(pytest.mark.slow)


def pytest_unconfigure(config):
    """Cleanup hook - run performance analysis."""
    if hasattr(config, 'test_performance_data'):
        performance_data = config.test_performance_data
        
        if performance_data:
            total_time = sum(performance_data.values())
            slowest_test = max(performance_data.items(), key=lambda x: x[1])
            avg_time = total_time / len(performance_data)
            
            print(f"\\n📊 Test Performance Summary:")
            print(f"   Total execution time: {total_time:.2f}s")
            print(f"   Average test time: {avg_time:.2f}s")
            print(f"   Slowest test: {slowest_test[0]} ({slowest_test[1]:.2f}s)")
            print(f"   Tests executed: {len(performance_data)}")


# Pytest command line options
def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--performance-baseline",
        action="store",
        default=None,
        help="Set performance baseline file for regression testing"
    )


# Skip slow tests by default in CI environments
def pytest_runtest_setup(item):
    """Skip slow tests if running with --fast flag."""
    if item.config.getoption("--collect-only"):
        return
        
    # Skip integration tests if requested
    if item.get_closest_marker("slow") and item.config.getoption("-m") == "not slow":
        pytest.skip("Slow test skipped")