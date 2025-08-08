"""
Neon Connection Pool - Production-Ready AsyncPG Integration for Ray Workers

This module implements production-grade connection pooling for Neon PostgreSQL
database with TimescaleDB extension, optimized for distributed Ray worker
coordination and genetic algorithm evolution workloads.

Research-Based Implementation:
- /research/neon/04_neon_production_deployment.md - Production configuration
- /research/asyncpg/01_usage_connection_pools.md - AsyncPG patterns
- /research/neon/03_neon_connection_patterns.md - Connection optimization

Key Features:
- Production connection pooling (10-50 connections for Ray workers)
- TimescaleDB extension validation
- Automatic retry logic with exponential backoff
- SSL/TLS security with pooled connections (-pooler.neon.tech)
- Comprehensive health monitoring and metrics
- Ray worker coordination optimized settings
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass

# AsyncPG imports (conditional for graceful degradation)
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    # Create mock asyncpg for type hints when not available
    class MockAsyncPG:
        class Connection:
            pass
        class Pool:
            pass
    asyncpg = MockAsyncPG()

from src.config.settings import get_settings


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolStats:
    """Connection pool statistics for monitoring."""
    pool_size: int
    max_size: int
    min_size: int
    idle_count: int
    active_count: int
    total_connections_created: int
    health_status: str
    last_health_check: datetime


class NeonConnectionPool:
    """
    Production-ready AsyncPG connection pool for Neon PostgreSQL with TimescaleDB.
    
    Optimized for Ray worker coordination with distributed genetic algorithm
    evolution workloads requiring reliable database access and state management.
    """
    
    def __init__(self, 
                 connection_string: str,
                 min_connections: int = 10,
                 max_connections: int = 50,
                 production_mode: bool = True,
                 application_name: str = "quant_trading_ray_worker"):
        """
        Initialize Neon connection pool with production configuration.
        
        Args:
            connection_string: Base Neon connection string
            min_connections: Minimum pool size (default: 10 for Ray workers)
            max_connections: Maximum pool size (default: 50 for Ray workers)
            production_mode: Enable production optimizations
            application_name: Application identifier for monitoring
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required for Neon integration. Install with: pip install asyncpg")
        
        self.connection_string = self._enhance_connection_string(
            connection_string, application_name
        )
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.production_mode = production_mode
        self.application_name = application_name
        
        # Connection pool
        self.pool: Optional[asyncpg.Pool] = None
        self.pool_stats = ConnectionPoolStats(
            pool_size=0, max_size=max_connections, min_size=min_connections,
            idle_count=0, active_count=0, total_connections_created=0,
            health_status="uninitialized", last_health_check=datetime.now(timezone.utc)
        )
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.NeonPool")
        
        # Production settings based on research
        if production_mode:
            self.command_timeout = 120  # 2 minutes for complex GA operations
            self.query_timeout = 300    # 5 minutes for backtesting queries
            self.connection_retry_attempts = 5
            self.health_check_interval = 30  # seconds
        else:
            self.command_timeout = 30
            self.query_timeout = 60
            self.connection_retry_attempts = 3
            self.health_check_interval = 60
        
        # Metrics tracking
        self.connection_attempts = 0
        self.successful_connections = 0
        self.failed_connections = 0
        self.total_queries_executed = 0
        self.last_error: Optional[str] = None
        
    def _enhance_connection_string(self, base_string: str, app_name: str) -> str:
        """
        Enhance connection string with production parameters.
        
        Based on research from /research/neon/04_neon_production_deployment.md
        """
        # Convert to pooled connection for production (Neon research pattern)
        if "-pooler." not in base_string:
            base_string = base_string.replace(".neon.tech", "-pooler.neon.tech")
            self.logger.info("Converted to pooled Neon connection for production")
        
        # Production security and performance parameters
        params = [
            "sslmode=require",  # Production security requirement
            f"application_name={app_name}",
            "statement_timeout=300000",  # 5 minute statement timeout
            "idle_in_transaction_session_timeout=600000",  # 10 minute idle timeout
            "tcp_keepalives_idle=600",  # TCP keepalive for long connections
            "tcp_keepalives_interval=30",
            "tcp_keepalives_count=3"
        ]
        
        # Add parameters to connection string
        separator = "&" if "?" in base_string else "?"
        enhanced_string = base_string + separator + "&".join(params)
        
        self.logger.info("Enhanced connection string with production parameters")
        return enhanced_string
    
    async def initialize(self) -> None:
        """
        Initialize connection pool with TimescaleDB validation.
        
        Raises:
            RuntimeError: If TimescaleDB extension is not available
            ConnectionError: If pool initialization fails
        """
        try:
            self.logger.info(f"Initializing Neon connection pool: {self.min_connections}-{self.max_connections} connections")
            
            # Create AsyncPG pool with production settings
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=self.command_timeout,
                server_settings={
                    'application_name': self.application_name,
                    'timezone': 'UTC',
                    'statement_timeout': f'{self.query_timeout}s',
                    'lock_timeout': '30s'  # Prevent long locks
                },
                setup=self._setup_connection  # Per-connection setup
            )
            
            # Validate TimescaleDB extension (critical for Phase 4)
            await self._validate_timescaledb_extension()
            
            # Update pool stats
            self._update_pool_stats()
            self.pool_stats.health_status = "healthy"
            self.successful_connections += 1
            
            self.logger.info("Neon connection pool initialized successfully with TimescaleDB validation")
            
        except Exception as e:
            self.failed_connections += 1
            self.last_error = str(e)
            self.pool_stats.health_status = "failed"
            self.logger.error(f"Failed to initialize Neon connection pool: {e}")
            raise ConnectionError(f"Neon pool initialization failed: {e}") from e
    
    async def _setup_connection(self, connection: asyncpg.Connection) -> None:
        """
        Per-connection setup for optimal performance.
        
        Args:
            connection: AsyncPG connection to configure
        """
        try:
            # Set connection-specific parameters for trading workloads
            await connection.execute("SET statement_timeout = '300s'")
            await connection.execute("SET lock_timeout = '30s'")
            await connection.execute("SET idle_in_transaction_session_timeout = '600s'")
            
            # Optimize for time-series queries (TimescaleDB specific)
            await connection.execute("SET timescaledb.enable_optimizations = on")
            await connection.execute("SET timescaledb.max_background_workers = 4")
            
        except Exception as e:
            self.logger.warning(f"Connection setup warning (non-critical): {e}")
    
    async def _validate_timescaledb_extension(self) -> None:
        """
        Validate TimescaleDB extension is available and functional.
        
        Raises:
            RuntimeError: If TimescaleDB extension is not available
        """
        try:
            async with self.pool.acquire() as conn:
                # Check TimescaleDB extension
                result = await conn.fetch(
                    "SELECT * FROM pg_extension WHERE extname = 'timescaledb'"
                )
                
                if not result:
                    raise RuntimeError(
                        "TimescaleDB extension not available - required for Phase 4 implementation"
                    )
                
                # Validate TimescaleDB version
                version_result = await conn.fetchval(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                )
                
                self.logger.info(f"TimescaleDB extension validated: version {version_result}")
                
                # Test basic TimescaleDB functionality
                await conn.fetchval("SELECT timescaledb_version()")
                
        except Exception as e:
            self.logger.error(f"TimescaleDB validation failed: {e}")
            raise RuntimeError(f"TimescaleDB validation error: {e}") from e
    
    @asynccontextmanager
    async def acquire_connection(self):
        """
        Acquire connection with production retry logic and error handling.
        
        Yields:
            asyncpg.Connection: Database connection ready for use
            
        Raises:
            ConnectionError: If connection acquisition fails after retries
        """
        if not self.pool:
            await self.initialize()
        
        self.connection_attempts += 1
        last_exception = None
        
        for attempt in range(self.connection_retry_attempts):
            try:
                async with self.pool.acquire() as connection:
                    # Update metrics
                    self.successful_connections += 1
                    self._update_pool_stats()
                    
                    yield connection
                    break
                    
            except Exception as e:
                last_exception = e
                self.failed_connections += 1
                self.last_error = str(e)
                
                if attempt == self.connection_retry_attempts - 1:
                    self.logger.error(
                        f"Connection acquisition failed after {self.connection_retry_attempts} attempts: {e}"
                    )
                    self.pool_stats.health_status = "connection_failed"
                    raise ConnectionError(f"Neon database unavailable: {e}") from e
                
                # Exponential backoff
                wait_time = 2 ** attempt
                self.logger.warning(
                    f"Connection attempt {attempt + 1}/{self.connection_retry_attempts} failed, "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute query with automatic retry and metrics tracking.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            List of query results as dictionaries
        """
        start_time = time.time()
        
        try:
            async with self.acquire_connection() as conn:
                result = await conn.fetch(query, *args)
                
                # Convert to list of dicts
                result_dicts = [dict(row) for row in result]
                
                # Update metrics
                self.total_queries_executed += 1
                execution_time = time.time() - start_time
                
                self.logger.debug(
                    f"Query executed successfully in {execution_time:.3f}s: "
                    f"{len(result_dicts)} rows returned"
                )
                
                return result_dicts
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Query execution failed after {execution_time:.3f}s: {e}")
            self.last_error = str(e)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for monitoring and alerting.
        
        Returns:
            Dict containing detailed health status and metrics
        """
        health_start = time.time()
        
        try:
            if not self.pool:
                return {
                    "status": "disconnected",
                    "error": "Pool not initialized",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Test database responsiveness
            async with self.acquire_connection() as conn:
                # Basic connectivity test
                response_start = time.time()
                result = await conn.fetchval("SELECT 1")
                response_time_ms = (time.time() - response_start) * 1000
                
                # TimescaleDB health check
                timescale_version = await conn.fetchval("SELECT timescaledb_version()")
                
                # Database size and statistics
                db_stats = await conn.fetchrow("""
                    SELECT 
                        pg_size_pretty(pg_database_size(current_database())) as size_pretty,
                        pg_database_size(current_database()) / (1024*1024*1024.0) as size_gb
                """)
                
                # Active connections
                active_connections = await conn.fetchval("""
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%'
                """)
            
            # Update pool statistics
            self._update_pool_stats()
            self.pool_stats.health_status = "healthy"
            self.pool_stats.last_health_check = datetime.now(timezone.utc)
            
            total_health_time = time.time() - health_start
            
            return {
                "status": "healthy",
                "database_responsive": result == 1,
                "response_time_ms": response_time_ms,
                "health_check_time_ms": total_health_time * 1000,
                "timescaledb_version": timescale_version,
                "database_size_gb": float(db_stats["size_gb"]),
                "database_size_pretty": db_stats["size_pretty"],
                "active_database_connections": active_connections,
                "pool_stats": {
                    "size": self.pool_stats.pool_size,
                    "max_size": self.pool_stats.max_size,
                    "min_size": self.pool_stats.min_size,
                    "idle_count": self.pool_stats.idle_count,
                    "active_count": self.pool_stats.active_count
                },
                "connection_metrics": {
                    "total_attempts": self.connection_attempts,
                    "successful": self.successful_connections,
                    "failed": self.failed_connections,
                    "success_rate": (self.successful_connections / max(1, self.connection_attempts)) * 100,
                    "total_queries": self.total_queries_executed
                },
                "last_error": self.last_error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.pool_stats.health_status = "unhealthy"
            self.last_error = str(e)
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_available": self.pool is not None,
                "connection_metrics": {
                    "total_attempts": self.connection_attempts,
                    "successful": self.successful_connections,
                    "failed": self.failed_connections,
                    "total_queries": self.total_queries_executed
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _update_pool_stats(self) -> None:
        """Update internal pool statistics."""
        if self.pool:
            self.pool_stats.pool_size = self.pool.get_size()
            self.pool_stats.idle_count = self.pool.get_idle_size()
            self.pool_stats.active_count = self.pool_stats.pool_size - self.pool_stats.idle_count
    
    async def close(self) -> None:
        """Close connection pool gracefully."""
        if self.pool:
            self.logger.info("Closing Neon connection pool...")
            await self.pool.close()
            self.pool = None
            self.pool_stats.health_status = "closed"
            self.logger.info("Neon connection pool closed successfully")
    
    def get_connection_string_info(self) -> Dict[str, Any]:
        """
        Get sanitized connection string information for debugging.
        
        Returns:
            Dict with connection info (passwords masked)
        """
        # Mask sensitive information
        masked_string = self.connection_string
        if "@" in masked_string:
            parts = masked_string.split("@")
            if ":" in parts[0]:
                user_pass = parts[0].split(":")
                masked_string = f"{user_pass[0]}:***@{parts[1]}"
        
        return {
            "masked_connection_string": masked_string,
            "uses_pooled_connection": "-pooler." in self.connection_string,
            "ssl_enabled": "sslmode=require" in self.connection_string,
            "application_name": self.application_name,
            "production_mode": self.production_mode,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections
        }


def create_neon_pool_from_settings() -> NeonConnectionPool:
    """
    Create Neon connection pool from application settings.
    
    Returns:
        Configured NeonConnectionPool instance
        
    Raises:
        ValueError: If Neon configuration is missing
    """
    try:
        settings = get_settings()
        
        # Get Neon configuration from settings
        if hasattr(settings, 'neon') and hasattr(settings.neon, 'connection_string'):
            connection_string = settings.neon.connection_string
            min_connections = getattr(settings.neon, 'min_connections', 10)
            max_connections = getattr(settings.neon, 'max_connections', 50)
            production_mode = getattr(settings.neon, 'production_mode', True)
        else:
            # Fallback to environment variable
            import os
            connection_string = os.getenv('NEON_CONNECTION_STRING')
            if not connection_string:
                raise ValueError(
                    "Neon connection string not configured. Set NEON_CONNECTION_STRING environment variable "
                    "or configure settings.neon.connection_string"
                )
            
            min_connections = int(os.getenv('NEON_MIN_CONNECTIONS', '10'))
            max_connections = int(os.getenv('NEON_MAX_CONNECTIONS', '50'))
            production_mode = os.getenv('NEON_PRODUCTION_MODE', 'true').lower() == 'true'
        
        return NeonConnectionPool(
            connection_string=connection_string,
            min_connections=min_connections,
            max_connections=max_connections,
            production_mode=production_mode
        )
        
    except Exception as e:
        logger.error(f"Failed to create Neon pool from settings: {e}")
        raise ValueError(f"Neon pool configuration error: {e}") from e