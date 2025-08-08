"""
Neon Hybrid Storage - Phase 4 DataStorageInterface Implementation

This module implements the NeonHybridStorage class that provides seamless integration
between Neon PostgreSQL + TimescaleDB (cloud) and DuckDB (local cache) for optimal
performance in distributed genetic algorithm evolution workloads.

Research-Based Implementation:
- Implements existing DataStorageInterface for zero-code-change upgrade
- /research/neon/02_neon_architecture.md - Neon storage patterns
- /verified_docs/by_module_simplified/data/ - Existing storage patterns
- Production-ready with comprehensive error handling and monitoring

Key Features:
- Drop-in replacement for LocalDataStorage following DataStorageInterface
- Intelligent data placement using HybridCacheStrategy
- Automatic failover to local cache when Neon unavailable
- TimescaleDB optimization for time-series genetic algorithm queries
- Comprehensive health monitoring and performance metrics
- Production-ready error handling and retry logic
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.data.storage_interfaces import DataStorageInterface
from src.data.data_storage import DataStorage
from src.data.market_data_pipeline import OHLCVBar
from .neon_connection_pool import NeonConnectionPool, create_neon_pool_from_settings
from .neon_schema_manager import NeonSchemaManager, MigrationStats
from .hybrid_cache_strategy import HybridCacheStrategy, DataSource
from src.config.settings import get_settings, Settings


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class HybridStorageMetrics:
    """Performance metrics for hybrid storage operations."""
    total_queries: int = 0
    local_cache_hits: int = 0
    neon_direct_hits: int = 0
    hybrid_queries: int = 0
    average_response_time_ms: float = 0.0
    failover_events: int = 0
    sync_operations: int = 0
    last_metrics_reset: datetime = None
    
    def __post_init__(self):
        if self.last_metrics_reset is None:
            self.last_metrics_reset = datetime.now(timezone.utc)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return (self.local_cache_hits / self.total_queries) * 100
    
    @property
    def failover_rate(self) -> float:
        """Calculate failover rate."""
        if self.total_queries == 0:
            return 0.0
        return (self.failover_events / self.total_queries) * 100


class NeonHybridStorage(DataStorageInterface):
    """
    Production-ready hybrid storage implementing DataStorageInterface.
    
    Provides seamless integration between Neon cloud database and local DuckDB cache
    for optimal performance in distributed genetic algorithm workloads. Acts as a 
    drop-in replacement for LocalDataStorage with zero code changes required.
    """
    
    def __init__(self, settings: Optional[Settings] = None, auto_initialize: bool = True):
        """
        Initialize hybrid storage with Neon + DuckDB integration.
        
        Args:
            settings: Configuration settings (optional)
            auto_initialize: Whether to automatically initialize connections
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.NeonHybridStorage")
        
        # Initialize local DuckDB cache (existing implementation)
        self.local_cache = DataStorage(self.settings)
        self.logger.info("Initialized local DuckDB cache")
        
        # Initialize Neon components
        self.neon_pool: Optional[NeonConnectionPool] = None
        self.schema_manager: Optional[NeonSchemaManager] = None
        self.cache_strategy = HybridCacheStrategy()
        
        # Performance metrics and monitoring
        self.metrics = HybridStorageMetrics()
        self.initialization_complete = False
        self.neon_available = False
        
        # Sync management
        self.sync_queue: List[OHLCVBar] = []
        self.sync_lock = asyncio.Lock()
        self.background_sync_task: Optional[asyncio.Task] = None
        
        # Auto-initialize if requested
        if auto_initialize:
            asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self) -> None:
        """Initialize Neon connection asynchronously."""
        try:
            await self.initialize_neon_connection()
        except Exception as e:
            self.logger.warning(f"Neon initialization failed, continuing with local cache only: {e}")
    
    async def initialize_neon_connection(self) -> bool:
        """
        Initialize Neon connection pool and schema.
        
        Returns:
            True if initialization successful, False if fallback to local only
        """
        try:
            self.logger.info("Initializing Neon connection pool...")
            
            # Create connection pool from settings
            self.neon_pool = create_neon_pool_from_settings()
            await self.neon_pool.initialize()
            
            # Initialize schema manager
            self.schema_manager = NeonSchemaManager(self.neon_pool)
            
            # Validate or create schema
            schema_health = await self.schema_manager.validate_schema_health()
            if not schema_health.get("schema_valid", False):
                self.logger.info("Creating TimescaleDB schema...")
                await self.schema_manager.create_complete_schema()
            
            self.neon_available = True
            self.initialization_complete = True
            
            # Start background sync task
            self.background_sync_task = asyncio.create_task(self._background_sync_loop())
            
            self.logger.info("Neon hybrid storage initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Neon initialization failed: {e}")
            self.neon_available = False
            self.initialization_complete = True
            return False
    
    async def store_ohlcv_bars(self, bars: List[OHLCVBar]) -> None:
        """
        Store OHLCV bars using hybrid storage strategy.
        
        Stores data in local cache first for immediate access, then synchronizes
        to Neon asynchronously for centralized access across Ray workers.
        
        Args:
            bars: List of OHLCV bars to store
        """
        if not bars:
            return
        
        start_time = time.time()
        
        try:
            # Always store in local cache first (immediate access)
            await self.local_cache.store_ohlcv_bars(bars)
            self.logger.debug(f"Stored {len(bars)} bars in local cache")
            
            # Queue for Neon synchronization if available
            if self.neon_available:
                async with self.sync_lock:
                    self.sync_queue.extend(bars)
                    
                # Trigger immediate sync for small batches
                if len(self.sync_queue) < 100:
                    await self._sync_to_neon()
            else:
                self.logger.debug("Neon unavailable - stored locally only")
            
            # Update metrics
            response_time_ms = (time.time() - start_time) * 1000
            await self._update_metrics("store_operation", response_time_ms, len(bars))
            
        except Exception as e:
            self.logger.error(f"Hybrid storage store_ohlcv_bars failed: {e}")
            raise RuntimeError(f"Failed to store OHLCV bars: {e}") from e
    
    async def get_ohlcv_bars(self, 
                           symbol: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV bars using intelligent hybrid strategy.
        
        Uses HybridCacheStrategy to determine optimal data source based on
        data age, query characteristics, and performance patterns.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for data retrieval
            end_time: End time for data retrieval  
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        query_start_time = time.time()
        
        try:
            # Determine optimal data source using strategy
            optimal_source = await self.cache_strategy.determine_optimal_source(
                symbol, start_time, end_time, limit, query_context="genetic_algorithm"
            )
            
            # Execute query based on strategy decision
            if optimal_source == DataSource.LOCAL_CACHE:
                result = await self._get_ohlcv_from_local(symbol, start_time, end_time, limit)
                self.metrics.local_cache_hits += 1
                
            elif optimal_source == DataSource.NEON_DIRECT:
                result = await self._get_ohlcv_from_neon(symbol, start_time, end_time, limit)
                self.metrics.neon_direct_hits += 1
                
            else:  # DataSource.HYBRID
                result = await self._get_ohlcv_hybrid(symbol, start_time, end_time, limit)
                self.metrics.hybrid_queries += 1
            
            # Record performance metrics
            query_time_ms = (time.time() - query_start_time) * 1000
            await self.cache_strategy.record_query_performance(
                symbol, optimal_source, query_time_ms, len(result), success=True
            )
            
            await self._update_metrics("get_operation", query_time_ms, len(result))
            
            self.logger.debug(
                f"Retrieved {len(result)} bars for {symbol} via {optimal_source} "
                f"in {query_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            query_time_ms = (time.time() - query_start_time) * 1000
            await self.cache_strategy.record_query_performance(
                symbol, "unknown", query_time_ms, 0, success=False
            )
            
            self.logger.error(f"Hybrid get_ohlcv_bars failed for {symbol}: {e}")
            
            # Attempt fallback to local cache
            try:
                result = await self._get_ohlcv_from_local(symbol, start_time, end_time, limit)
                self.metrics.failover_events += 1
                self.logger.info(f"Fallback successful for {symbol}: {len(result)} bars")
                return result
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                raise RuntimeError(f"All data sources failed for {symbol}: {e}") from e
    
    async def _get_ohlcv_from_local(self, symbol: str, start_time: Optional[datetime],
                                   end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Get OHLCV data from local DuckDB cache."""
        return await self.local_cache.get_ohlcv_bars(symbol, start_time, end_time, limit)
    
    async def _get_ohlcv_from_neon(self, symbol: str, start_time: Optional[datetime],
                                  end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Get OHLCV data directly from Neon TimescaleDB."""
        if not self.neon_available or not self.neon_pool:
            raise RuntimeError("Neon database not available")
        
        # Build optimized TimescaleDB query
        query = """
        SELECT symbol, timestamp, open, high, low, close, volume, vwap, trade_count
        FROM ohlcv_bars 
        WHERE symbol = $1
        """
        params = [symbol]
        
        # Add time filtering
        if start_time:
            query += " AND timestamp >= $2"
            params.append(start_time)
            
        if end_time:
            param_num = len(params) + 1
            query += f" AND timestamp <= ${param_num}"
            params.append(end_time)
        
        # Order and limit
        query += " ORDER BY timestamp DESC"
        
        if limit:
            param_num = len(params) + 1
            query += f" LIMIT ${param_num}"
            params.append(limit)
        
        # Execute query
        async with self.neon_pool.acquire_connection() as conn:
            rows = await conn.fetch(query, *params)
        
        # Convert to DataFrame
        if rows:
            df = pd.DataFrame([dict(row) for row in rows])
            # Ensure proper data types
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['trade_count'] = pd.to_numeric(df['trade_count'], errors='coerce').astype('Int64')
            return df.sort_values('timestamp')
        else:
            return pd.DataFrame(columns=['symbol', 'timestamp', 'open', 'high', 'low', 
                                       'close', 'volume', 'vwap', 'trade_count'])
    
    async def _get_ohlcv_hybrid(self, symbol: str, start_time: Optional[datetime],
                               end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """
        Get OHLCV data using hybrid approach (combine local and Neon).
        
        Optimizes by getting recent data from cache and historical from Neon.
        """
        now = datetime.now(timezone.utc)
        
        # Define split point (7 days ago for hot data)
        hot_data_cutoff = now - timedelta(days=7)
        
        # Determine query split strategy
        if start_time and start_time >= hot_data_cutoff:
            # Recent data - use local cache
            return await self._get_ohlcv_from_local(symbol, start_time, end_time, limit)
            
        elif end_time and end_time < hot_data_cutoff:
            # Historical data - use Neon
            return await self._get_ohlcv_from_neon(symbol, start_time, end_time, limit)
            
        else:
            # Mixed data - combine both sources
            try:
                # Get recent data from local cache
                recent_data = await self._get_ohlcv_from_local(
                    symbol, hot_data_cutoff, end_time, None
                )
                
                # Get historical data from Neon
                historical_data = await self._get_ohlcv_from_neon(
                    symbol, start_time, hot_data_cutoff, None
                )
                
                # Combine and sort
                if not recent_data.empty and not historical_data.empty:
                    combined_data = pd.concat([historical_data, recent_data], ignore_index=True)
                elif not recent_data.empty:
                    combined_data = recent_data
                elif not historical_data.empty:
                    combined_data = historical_data
                else:
                    combined_data = pd.DataFrame(columns=['symbol', 'timestamp', 'open', 'high', 
                                                        'low', 'close', 'volume', 'vwap', 'trade_count'])
                
                # Sort by timestamp and apply limit
                if not combined_data.empty:
                    combined_data = combined_data.sort_values('timestamp')
                    if limit:
                        combined_data = combined_data.tail(limit)
                
                return combined_data
                
            except Exception as e:
                self.logger.warning(f"Hybrid query failed, falling back to local: {e}")
                return await self._get_ohlcv_from_local(symbol, start_time, end_time, limit)
    
    async def calculate_technical_indicators(self, 
                                           symbol: str,
                                           lookback_periods: int = 200) -> pd.DataFrame:
        """
        Calculate technical indicators using local cache for performance.
        
        Technical indicator calculations are optimized for local DuckDB performance
        as they require intensive computations better suited for local processing.
        
        Args:
            symbol: Trading symbol
            lookback_periods: Number of periods for calculations
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            # Use local cache for technical indicators (performance optimization)
            result = await self.local_cache.calculate_technical_indicators(
                symbol, lookback_periods
            )
            
            self.logger.debug(f"Calculated technical indicators for {symbol}: {len(result)} records")
            return result
            
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed for {symbol}: {e}")
            raise RuntimeError(f"Failed to calculate technical indicators for {symbol}: {e}") from e
    
    async def get_market_summary(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get market summary using local cache for performance.
        
        Market summary is optimized for local cache as it's frequently accessed
        and benefits from the performance characteristics of DuckDB.
        
        Args:
            symbols: List of symbols (optional)
            
        Returns:
            DataFrame with market summary
        """
        try:
            # Use local cache for market summary (performance optimization)
            result = await self.local_cache.get_market_summary(symbols)
            
            self.logger.debug(
                f"Retrieved market summary: {len(result)} symbols"
                f"{f' (filtered: {symbols})' if symbols else ''}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Market summary retrieval failed: {e}")
            raise RuntimeError(f"Failed to get market summary: {e}") from e
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for hybrid storage system.
        
        Returns:
            Detailed health status including both local cache and Neon components
        """
        health_start_time = time.time()
        
        try:
            # Check local cache health
            local_health = await self.local_cache.health_check()
            
            # Check Neon health if available
            neon_health = {"status": "unavailable", "message": "Not initialized"}
            if self.neon_available and self.neon_pool:
                try:
                    neon_health = await self.neon_pool.health_check()
                except Exception as e:
                    neon_health = {"status": "error", "error": str(e)}
            
            # Check schema health if available
            schema_health = {"status": "unavailable"}
            if self.schema_manager:
                try:
                    schema_health = await self.schema_manager.validate_schema_health()
                except Exception as e:
                    schema_health = {"status": "error", "error": str(e)}
            
            # Determine overall health
            local_healthy = local_health.get("status") == "healthy"
            neon_healthy = neon_health.get("status") == "healthy"
            
            if local_healthy and (neon_healthy or not self.neon_available):
                overall_status = "healthy"
            elif local_healthy:
                overall_status = "degraded"  # Local works, Neon issues
            else:
                overall_status = "unhealthy"  # Local issues
            
            health_check_time = (time.time() - health_start_time) * 1000
            
            return {
                "status": overall_status,
                "backend": "neon_hybrid",
                "initialization_complete": self.initialization_complete,
                "neon_available": self.neon_available,
                "health_check_time_ms": health_check_time,
                "components": {
                    "local_cache": local_health,
                    "neon_database": neon_health,
                    "schema_manager": schema_health
                },
                "performance_metrics": {
                    "total_queries": self.metrics.total_queries,
                    "cache_hit_rate_percent": self.metrics.cache_hit_rate,
                    "failover_rate_percent": self.metrics.failover_rate,
                    "average_response_time_ms": self.metrics.average_response_time_ms,
                    "sync_queue_size": len(self.sync_queue)
                },
                "strategy_metrics": await self.cache_strategy.get_strategy_metrics(),
                "sync_status": {
                    "background_sync_running": self.background_sync_task is not None and not self.background_sync_task.done(),
                    "queued_bars_count": len(self.sync_queue),
                    "sync_operations_completed": self.metrics.sync_operations
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "backend": "neon_hybrid", 
                "error": str(e),
                "initialization_complete": self.initialization_complete,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _sync_to_neon(self) -> None:
        """Synchronize queued data to Neon database."""
        if not self.neon_available or not self.sync_queue:
            return
        
        async with self.sync_lock:
            bars_to_sync = self.sync_queue.copy()
            self.sync_queue.clear()
        
        if not bars_to_sync:
            return
        
        try:
            # Group bars by symbol for efficient insertion
            bars_by_symbol = {}
            for bar in bars_to_sync:
                symbol = bar.symbol
                if symbol not in bars_by_symbol:
                    bars_by_symbol[symbol] = []
                bars_by_symbol[symbol].append(bar)
            
            # Sync each symbol's data
            for symbol, symbol_bars in bars_by_symbol.items():
                # Convert to DataFrame format for schema manager
                df_data = []
                for bar in symbol_bars:
                    df_data.append({
                        'symbol': bar.symbol,
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': bar.vwap,
                        'trade_count': bar.trade_count
                    })
                
                df = pd.DataFrame(df_data)
                await self.schema_manager._batch_insert_ohlcv(symbol, df)
            
            self.metrics.sync_operations += 1
            self.logger.debug(f"Synchronized {len(bars_to_sync)} bars to Neon")
            
        except Exception as e:
            # Re-queue failed bars for retry
            async with self.sync_lock:
                self.sync_queue.extend(bars_to_sync)
            
            self.logger.error(f"Neon synchronization failed: {e}")
            # Consider marking Neon as unavailable on repeated failures
    
    async def _background_sync_loop(self) -> None:
        """Background task for periodic Neon synchronization."""
        while self.neon_available:
            try:
                await asyncio.sleep(30)  # Sync every 30 seconds
                if self.sync_queue:
                    await self._sync_to_neon()
            except asyncio.CancelledError:
                self.logger.info("Background sync loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Background sync error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_metrics(self, operation_type: str, response_time_ms: float, record_count: int) -> None:
        """Update performance metrics."""
        self.metrics.total_queries += 1
        
        # Update average response time (exponential moving average)
        if self.metrics.average_response_time_ms == 0:
            self.metrics.average_response_time_ms = response_time_ms
        else:
            alpha = 0.1  # Smoothing factor
            self.metrics.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.metrics.average_response_time_ms
            )
    
    async def migrate_from_local_storage(self, local_storage: DataStorage) -> MigrationStats:
        """
        Migrate data from existing local storage to Neon hybrid system.
        
        Args:
            local_storage: Existing DataStorage instance to migrate from
            
        Returns:
            Migration statistics and results
        """
        if not self.schema_manager:
            raise RuntimeError("Schema manager not available - Neon not initialized")
        
        self.logger.info("Starting migration from local storage to Neon hybrid system")
        
        try:
            # Perform migration using schema manager
            migration_stats = await self.schema_manager.migrate_existing_data(local_storage)
            
            self.logger.info(
                f"Migration completed successfully: {migration_stats.symbols_migrated} symbols, "
                f"{migration_stats.total_bars_migrated:,} bars"
            )
            
            return migration_stats
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise RuntimeError(f"Migration from local storage failed: {e}") from e
    
    async def close(self) -> None:
        """Close hybrid storage connections gracefully."""
        self.logger.info("Closing Neon hybrid storage...")
        
        # Cancel background sync task
        if self.background_sync_task and not self.background_sync_task.done():
            self.background_sync_task.cancel()
            try:
                await self.background_sync_task
            except asyncio.CancelledError:
                pass
        
        # Final sync of remaining data
        if self.sync_queue and self.neon_available:
            try:
                await self._sync_to_neon()
            except Exception as e:
                self.logger.warning(f"Final sync failed: {e}")
        
        # Close Neon connection pool
        if self.neon_pool:
            await self.neon_pool.close()
        
        # Close local cache (if it has a close method)
        if hasattr(self.local_cache, 'close'):
            await self.local_cache.close()
        
        self.logger.info("Neon hybrid storage closed successfully")


async def create_neon_hybrid_storage(settings: Optional[Settings] = None, 
                                    auto_migrate: bool = False) -> NeonHybridStorage:
    """
    Create and initialize NeonHybridStorage with optional data migration.
    
    Args:
        settings: Configuration settings
        auto_migrate: Whether to automatically migrate existing local data
        
    Returns:
        Initialized NeonHybridStorage instance
    """
    # Create hybrid storage instance
    hybrid_storage = NeonHybridStorage(settings, auto_initialize=False)
    
    # Initialize Neon connection
    neon_initialized = await hybrid_storage.initialize_neon_connection()
    
    if neon_initialized and auto_migrate:
        try:
            # Create temporary local storage for migration
            local_storage = DataStorage(settings)
            migration_stats = await hybrid_storage.migrate_from_local_storage(local_storage)
            
            logger.info(
                f"Auto-migration completed: {migration_stats.symbols_migrated} symbols, "
                f"{migration_stats.total_bars_migrated:,} bars"
            )
            
        except Exception as e:
            logger.warning(f"Auto-migration failed (continuing anyway): {e}")
    
    return hybrid_storage