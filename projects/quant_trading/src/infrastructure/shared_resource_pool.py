"""
Shared Resource Pool - Centralized Resource Management

Implements centralized resource management for the quantitative trading system
to eliminate session management conflicts and resource leaks across all components.

Key Features:
- Single shared connection pool across all HTTP clients
- Centralized FearGreedClient management with dependency injection
- Health monitoring and automatic recovery
- Resource lifecycle tracking and cleanup
- Production-ready error handling and monitoring integration

Based on Phase 4 remediation plan specifications:
- SharedResourcePool class for system-wide resource management
- Connection pool sharing across all HTTP clients
- Health monitoring and automatic recovery
- Resource lifecycle tracking and cleanup

Research Foundation:
- /research/asyncio_advanced/page_4_streams_websocket_integration.md
- /research/aiofiles_v3/vector4_asyncio_integration.md
"""

import asyncio
import logging
import time
import aiohttp
from typing import Dict, Optional, Any, List, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from src.config.settings import get_settings, Settings
from src.data.fear_greed_client import FearGreedClient


class ResourceHealth(str, Enum):
    """Resource health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


@dataclass
class HealthCheckResult:
    """Result of a resource health check."""
    healthy: bool
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    overall_health: bool
    individual_checks: Dict[str, HealthCheckResult]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SharedResourcePool:
    """
    Centralized resource pool for system-wide resource management.
    
    Manages shared HTTP connections, FearGreedClient instances, and provides
    health monitoring with automatic recovery capabilities.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize shared resource pool."""
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Core shared resources
        self.connection_pool: Optional[aiohttp.ClientSession] = None
        self.fear_greed_client: Optional[FearGreedClient] = None
        
        # Resource health tracking
        self.resource_health: Dict[str, ResourceHealth] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_check_interval = timedelta(minutes=5)
        
        # Resource lifecycle tracking
        self.resource_creation_time: Dict[str, datetime] = {}
        self.resource_usage_count: Dict[str, int] = {}
        
        # Health check functions
        self.health_checks: Dict[str, Callable] = {
            'connection_pool': self._check_connection_pool_health,
            'fear_greed_client': self._check_fear_greed_client_health,
        }
        
        # Initialization state
        self.is_initialized = False
        self.initialization_time: Optional[datetime] = None
        
    async def initialize(self) -> None:
        """Initialize all shared resources."""
        if self.is_initialized:
            self.logger.warning("SharedResourcePool already initialized")
            return
            
        try:
            self.logger.info("🚀 Initializing SharedResourcePool...")
            start_time = time.perf_counter()
            
            # Initialize connection pool first
            await self._initialize_connection_pool()
            
            # Initialize FearGreedClient with shared connection pool
            await self._initialize_fear_greed_client()
            
            # Mark as initialized
            self.is_initialized = True
            self.initialization_time = datetime.now(timezone.utc)
            
            # Run initial health check
            health_report = await self.validate_system_health()
            
            elapsed_time = time.perf_counter() - start_time
            self.logger.info(
                f"✅ SharedResourcePool initialized in {elapsed_time:.2f}s - "
                f"Overall health: {health_report.overall_health}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SharedResourcePool: {e}")
            await self.cleanup()
            raise
    
    async def get_connection_pool(self) -> aiohttp.ClientSession:
        """Get shared HTTP connection pool."""
        if not self.is_initialized:
            raise RuntimeError("SharedResourcePool not initialized")
            
        if not self.connection_pool or self.connection_pool.closed:
            await self._initialize_connection_pool()
            
        self.resource_usage_count['connection_pool'] = (
            self.resource_usage_count.get('connection_pool', 0) + 1
        )
        
        return self.connection_pool
    
    async def get_fear_greed_client(self) -> FearGreedClient:
        """Get shared FearGreedClient with connection pooling."""
        if not self.is_initialized:
            raise RuntimeError("SharedResourcePool not initialized")
            
        if not self.fear_greed_client:
            await self._initialize_fear_greed_client()
            
        self.resource_usage_count['fear_greed_client'] = (
            self.resource_usage_count.get('fear_greed_client', 0) + 1
        )
        
        return self.fear_greed_client
    
    async def validate_system_health(self) -> SystemHealthReport:
        """Comprehensive system health validation."""
        self.logger.debug("🔍 Running system health validation...")
        
        health_results = {}
        overall_health = True
        
        for resource_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                health_results[resource_name] = result
                
                # Update resource health tracking
                self.resource_health[resource_name] = (
                    ResourceHealth.HEALTHY if result.healthy else ResourceHealth.UNHEALTHY
                )
                self.last_health_check[resource_name] = datetime.now(timezone.utc)
                
                if not result.healthy:
                    overall_health = False
                    self.logger.warning(f"⚠️ Resource {resource_name} unhealthy: {result.error}")
                    
            except Exception as e:
                health_results[resource_name] = HealthCheckResult(
                    healthy=False,
                    error=str(e),
                    timestamp=datetime.now(timezone.utc)
                )
                self.resource_health[resource_name] = ResourceHealth.UNHEALTHY
                overall_health = False
                self.logger.error(f"❌ Health check failed for {resource_name}: {e}")
        
        return SystemHealthReport(
            overall_health=overall_health,
            individual_checks=health_results,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def cleanup(self) -> None:
        """Clean up all shared resources."""
        self.logger.info("🧹 Cleaning up SharedResourcePool...")
        
        # Cleanup FearGreedClient
        if self.fear_greed_client:
            try:
                await self.fear_greed_client.disconnect()
                self.logger.info("✅ FearGreedClient disconnected")
            except Exception as e:
                self.logger.warning(f"⚠️ Error disconnecting FearGreedClient: {e}")
            finally:
                self.fear_greed_client = None
        
        # Cleanup connection pool
        if self.connection_pool and not self.connection_pool.closed:
            try:
                await self.connection_pool.close()
                self.logger.info("✅ Connection pool closed")
            except Exception as e:
                self.logger.warning(f"⚠️ Error closing connection pool: {e}")
            finally:
                self.connection_pool = None
        
        # Reset state
        self.is_initialized = False
        self.resource_health.clear()
        self.last_health_check.clear()
        self.resource_usage_count.clear()
        
        self.logger.info("✅ SharedResourcePool cleanup complete")
    
    async def _initialize_connection_pool(self) -> None:
        """Initialize shared HTTP connection pool."""
        self.logger.info("🌐 Initializing shared connection pool...")
        
        # Close existing pool if it exists
        if self.connection_pool and not self.connection_pool.closed:
            await self.connection_pool.close()
        
        # Create new connection pool with optimized settings
        timeout = aiohttp.ClientTimeout(
            total=30.0,
            connect=10.0,
            sock_connect=10.0,
            sock_read=20.0
        )
        
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        self.connection_pool = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'QuantTradingSystem/1.0'
            }
        )
        
        self.resource_creation_time['connection_pool'] = datetime.now(timezone.utc)
        self.resource_health['connection_pool'] = ResourceHealth.HEALTHY
        
        self.logger.info("✅ Shared connection pool initialized")
    
    async def _initialize_fear_greed_client(self) -> None:
        """Initialize FearGreedClient with shared connection pool."""
        self.logger.info("📊 Initializing shared FearGreedClient...")
        
        # Disconnect existing client if it exists
        if self.fear_greed_client:
            try:
                await self.fear_greed_client.disconnect()
            except Exception as e:
                self.logger.warning(f"⚠️ Error disconnecting existing FearGreedClient: {e}")
        
        # Ensure connection pool is available
        if not self.connection_pool or self.connection_pool.closed:
            await self._initialize_connection_pool()
        
        # Create FearGreedClient with shared session
        self.fear_greed_client = FearGreedClient(self.settings)
        
        # Inject shared session
        self.fear_greed_client.session = self.connection_pool
        
        self.resource_creation_time['fear_greed_client'] = datetime.now(timezone.utc)
        self.resource_health['fear_greed_client'] = ResourceHealth.HEALTHY
        
        self.logger.info("✅ Shared FearGreedClient initialized")
    
    async def _check_connection_pool_health(self) -> HealthCheckResult:
        """Check connection pool health."""
        try:
            if not self.connection_pool or self.connection_pool.closed:
                return HealthCheckResult(
                    healthy=False,
                    error="Connection pool is closed or None"
                )
            
            # Try a simple HTTP request to test connectivity
            start_time = time.perf_counter()
            try:
                async with self.connection_pool.get(
                    'https://httpbin.org/status/200',
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    response_time = time.perf_counter() - start_time
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            healthy=True,
                            metrics={
                                'response_time_seconds': response_time,
                                'status_code': response.status
                            }
                        )
                    else:
                        return HealthCheckResult(
                            healthy=False,
                            error=f"HTTP status {response.status}"
                        )
                        
            except asyncio.TimeoutError:
                return HealthCheckResult(
                    healthy=False,
                    error="Connection pool health check timeout"
                )
                
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                error=f"Connection pool health check error: {str(e)}"
            )
    
    async def _check_fear_greed_client_health(self) -> HealthCheckResult:
        """Check FearGreedClient health."""
        try:
            if not self.fear_greed_client:
                return HealthCheckResult(
                    healthy=False,
                    error="FearGreedClient is None"
                )
            
            # Test API call with timeout
            start_time = time.perf_counter()
            try:
                current_data = await self.fear_greed_client.get_current_index()
                response_time = time.perf_counter() - start_time
                
                if current_data and hasattr(current_data, 'value'):
                    return HealthCheckResult(
                        healthy=response_time < 10.0,  # 10 second threshold
                        metrics={
                            'response_time_seconds': response_time,
                            'fear_greed_value': current_data.value
                        }
                    )
                else:
                    return HealthCheckResult(
                        healthy=False,
                        error="FearGreedClient returned invalid data"
                    )
                    
            except asyncio.TimeoutError:
                return HealthCheckResult(
                    healthy=False,
                    error="FearGreedClient API timeout"
                )
                
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                error=f"FearGreedClient health check error: {str(e)}"
            )
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return {
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'is_initialized': self.is_initialized,
            'resource_health': {k: v.value for k, v in self.resource_health.items()},
            'resource_usage_count': self.resource_usage_count.copy(),
            'resource_creation_time': {
                k: v.isoformat() for k, v in self.resource_creation_time.items()
            },
            'last_health_check': {
                k: v.isoformat() for k, v in self.last_health_check.items()
            }
        }


# Global shared resource pool instance
_shared_resource_pool: Optional[SharedResourcePool] = None


async def get_shared_resource_pool() -> SharedResourcePool:
    """Get or create the global shared resource pool."""
    global _shared_resource_pool
    
    if _shared_resource_pool is None:
        _shared_resource_pool = SharedResourcePool()
        await _shared_resource_pool.initialize()
    
    return _shared_resource_pool


async def cleanup_shared_resource_pool() -> None:
    """Cleanup the global shared resource pool."""
    global _shared_resource_pool
    
    if _shared_resource_pool is not None:
        await _shared_resource_pool.cleanup()
        _shared_resource_pool = None