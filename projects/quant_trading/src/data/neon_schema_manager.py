"""
Neon Schema Manager - TimescaleDB Schema Setup and Data Migration

This module manages TimescaleDB schema creation, optimization, and data migration
from existing DuckDB storage to Neon PostgreSQL + TimescaleDB for distributed
genetic algorithm evolution and cloud-ready data access.

Research-Based Implementation:
- /research/timescaledb/1_timescaledb_hypertables_documentation.md - Hypertable patterns
- /research/neon/02_neon_architecture.md - Neon storage architecture
- /research/asyncpg/01_usage_connection_pools.md - Batch operations

Key Features:
- Optimized TimescaleDB hypertables for OHLCV time-series data
- Evolution state tracking for distributed GA coordination
- Efficient batch data migration from existing DuckDB storage
- Production-ready indexing strategy for genetic algorithm queries
- Comprehensive schema validation and health monitoring
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import pandas as pd

from .neon_connection_pool import NeonConnectionPool
from src.data.data_storage import DataStorage
from src.data.market_data_pipeline import OHLCVBar


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Data migration statistics and metrics."""
    symbols_migrated: int = 0
    total_bars_migrated: int = 0
    migration_start_time: Optional[datetime] = None
    migration_end_time: Optional[datetime] = None
    migration_duration_seconds: float = 0.0
    average_bars_per_symbol: float = 0.0
    migration_rate_bars_per_second: float = 0.0
    errors_encountered: List[str] = None
    
    def __post_init__(self):
        if self.errors_encountered is None:
            self.errors_encountered = []
    
    @property
    def success_rate(self) -> float:
        """Calculate migration success rate."""
        if self.symbols_migrated == 0:
            return 0.0
        return (self.symbols_migrated / max(1, self.symbols_migrated + len(self.errors_encountered))) * 100


class NeonSchemaManager:
    """
    TimescaleDB schema setup and data migration manager for Phase 4.
    
    Handles creation of optimized hypertables for OHLCV time-series data and
    evolution state tracking, with efficient migration from existing DuckDB storage.
    """
    
    def __init__(self, connection_pool: NeonConnectionPool):
        """
        Initialize schema manager with Neon connection pool.
        
        Args:
            connection_pool: Configured NeonConnectionPool instance
        """
        self.pool = connection_pool
        self.logger = logging.getLogger(f"{__name__}.SchemaManager")
        
        # Schema configuration based on TimescaleDB research
        self.ohlcv_chunk_interval = "INTERVAL '1 day'"  # Optimal for trading data
        self.evolution_chunk_interval = "INTERVAL '1 hour'"  # Fine-grained for GA state
        
        # Migration settings
        self.batch_size = 1000  # Records per batch for efficient insertion
        self.migration_timeout = 300  # 5 minutes per symbol
        
    async def create_complete_schema(self) -> Dict[str, Any]:
        """
        Create complete TimescaleDB schema for Phase 4 implementation.
        
        Returns:
            Dict containing schema creation results and statistics
        """
        self.logger.info("Starting complete TimescaleDB schema creation")
        schema_start_time = time.time()
        
        results = {
            "ohlcv_schema": False,
            "evolution_schema": False, 
            "indexes_created": False,
            "extensions_validated": False,
            "schema_creation_time": 0.0,
            "errors": []
        }
        
        try:
            # Step 1: Validate extensions
            await self._validate_required_extensions()
            results["extensions_validated"] = True
            
            # Step 2: Create OHLCV hypertable
            await self._create_ohlcv_hypertable()
            results["ohlcv_schema"] = True
            
            # Step 3: Create evolution state hypertable
            await self._create_evolution_state_hypertable()
            results["evolution_schema"] = True
            
            # Step 4: Create optimized indexes
            await self._create_performance_indexes()
            results["indexes_created"] = True
            
            results["schema_creation_time"] = time.time() - schema_start_time
            
            self.logger.info(f"Complete schema creation successful in {results['schema_creation_time']:.2f}s")
            return results
            
        except Exception as e:
            error_msg = f"Schema creation failed: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            results["schema_creation_time"] = time.time() - schema_start_time
            raise RuntimeError(error_msg) from e
    
    async def _validate_required_extensions(self) -> None:
        """Validate required PostgreSQL extensions are available."""
        required_extensions = ["timescaledb", "uuid-ossp"]
        
        async with self.pool.acquire_connection() as conn:
            for extension in required_extensions:
                # Check if extension exists
                result = await conn.fetch(
                    "SELECT * FROM pg_extension WHERE extname = $1", extension
                )
                
                if not result:
                    # Try to create extension
                    try:
                        await conn.execute(f"CREATE EXTENSION IF NOT EXISTS \"{extension}\"")
                        self.logger.info(f"Created extension: {extension}")
                    except Exception as e:
                        raise RuntimeError(f"Required extension '{extension}' not available: {e}")
                else:
                    self.logger.info(f"Extension '{extension}' already available")
    
    async def _create_ohlcv_hypertable(self) -> None:
        """
        Create optimized OHLCV hypertable for time-series data.
        
        Based on TimescaleDB research patterns for optimal performance.
        """
        create_ohlcv_table_sql = f"""
        -- Create OHLCV table optimized for time-series queries
        CREATE TABLE IF NOT EXISTS ohlcv_bars (
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL,
            vwap DOUBLE PRECISION NOT NULL DEFAULT 0,
            trade_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            
            -- Additional metadata for genetic algorithm optimization
            data_quality_score REAL DEFAULT 1.0,
            source_exchange TEXT DEFAULT 'hyperliquid'
        );
        
        -- Convert to hypertable with optimal chunk size for trading data
        SELECT create_hypertable('ohlcv_bars', 'timestamp', 
                                chunk_time_interval => {self.ohlcv_chunk_interval},
                                if_not_exists => TRUE);
        
        -- Add table comment for documentation
        COMMENT ON TABLE ohlcv_bars IS 'Time-series OHLCV data optimized for genetic algorithm backtesting and analysis';
        """
        
        async with self.pool.acquire_connection() as conn:
            await conn.execute(create_ohlcv_table_sql)
            
        self.logger.info("OHLCV hypertable created successfully")
    
    async def _create_evolution_state_hypertable(self) -> None:
        """
        Create evolution state hypertable for distributed GA coordination.
        
        Stores genetic algorithm population state and results for Ray worker coordination.
        """
        create_evolution_table_sql = f"""
        -- Create evolution state table for distributed genetic algorithm coordination
        CREATE TABLE IF NOT EXISTS evolution_state (
            evolution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            phase INTEGER NOT NULL,
            generation INTEGER NOT NULL,
            worker_id TEXT,
            
            -- Population and fitness data (JSONB for flexibility)
            population_data JSONB NOT NULL,
            fitness_metrics JSONB,
            best_individual JSONB,
            
            -- Evolution metadata
            status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'paused')),
            evolution_parameters JSONB,
            performance_metrics JSONB,
            
            -- Timestamps for tracking
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Convert to hypertable with fine-grained chunks for GA coordination
        SELECT create_hypertable('evolution_state', 'timestamp',
                                chunk_time_interval => {self.evolution_chunk_interval}, 
                                if_not_exists => TRUE);
        
        -- Create trigger to update 'updated_at' timestamp
        CREATE OR REPLACE FUNCTION update_evolution_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS evolution_state_update_timestamp ON evolution_state;
        CREATE TRIGGER evolution_state_update_timestamp
            BEFORE UPDATE ON evolution_state
            FOR EACH ROW EXECUTE FUNCTION update_evolution_timestamp();
        
        -- Add table comment
        COMMENT ON TABLE evolution_state IS 'Distributed genetic algorithm evolution state and coordination data';
        """
        
        async with self.pool.acquire_connection() as conn:
            await conn.execute(create_evolution_table_sql)
            
        self.logger.info("Evolution state hypertable created successfully")
    
    async def _create_performance_indexes(self) -> None:
        """
        Create optimized indexes for genetic algorithm query patterns.
        
        Based on expected query patterns from genetic algorithm backtesting.
        """
        index_creation_sql = """
        -- OHLCV table indexes for optimal query performance
        
        -- Primary compound index for symbol + timestamp queries (most common)
        CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_bars_symbol_time_idx
            ON ohlcv_bars (symbol, timestamp DESC);
        
        -- Symbol index with timestamp included for covering queries
        CREATE INDEX IF NOT EXISTS ohlcv_bars_symbol_covering_idx 
            ON ohlcv_bars (symbol) INCLUDE (timestamp, close, volume);
        
        -- Timestamp index for time-range queries across all symbols
        CREATE INDEX IF NOT EXISTS ohlcv_bars_timestamp_idx 
            ON ohlcv_bars (timestamp DESC) WHERE data_quality_score >= 0.8;
        
        -- Composite index for data quality filtering
        CREATE INDEX IF NOT EXISTS ohlcv_bars_quality_idx
            ON ohlcv_bars (symbol, data_quality_score, timestamp DESC)
            WHERE data_quality_score >= 0.8;
        
        -- Evolution state indexes for distributed GA coordination
        
        -- Active evolution queries (most frequent pattern)
        CREATE INDEX IF NOT EXISTS evolution_state_active_idx
            ON evolution_state (status, phase, generation DESC) 
            WHERE status = 'active';
        
        -- Worker coordination index
        CREATE INDEX IF NOT EXISTS evolution_state_worker_idx
            ON evolution_state (worker_id, timestamp DESC)
            WHERE worker_id IS NOT NULL;
        
        -- Evolution ID index for state retrieval
        CREATE INDEX IF NOT EXISTS evolution_state_evolution_id_idx
            ON evolution_state (evolution_id, generation DESC);
        
        -- Timestamp index for cleanup and archival
        CREATE INDEX IF NOT EXISTS evolution_state_timestamp_idx
            ON evolution_state (timestamp DESC);
        
        -- JSONB indexes for efficient population data queries
        CREATE INDEX IF NOT EXISTS evolution_state_fitness_idx
            ON evolution_state USING GIN (fitness_metrics)
            WHERE fitness_metrics IS NOT NULL;
        
        CREATE INDEX IF NOT EXISTS evolution_state_params_idx
            ON evolution_state USING GIN (evolution_parameters)
            WHERE evolution_parameters IS NOT NULL;
        """
        
        async with self.pool.acquire_connection() as conn:
            await conn.execute(index_creation_sql)
            
        self.logger.info("Performance indexes created successfully")
    
    async def migrate_existing_data(self, local_storage: DataStorage) -> MigrationStats:
        """
        Migrate existing OHLCV data from DuckDB to Neon TimescaleDB.
        
        Args:
            local_storage: Existing DataStorage instance to migrate from
            
        Returns:
            MigrationStats with detailed migration results
        """
        self.logger.info("Starting data migration from DuckDB to Neon TimescaleDB")
        
        stats = MigrationStats()
        stats.migration_start_time = datetime.now(timezone.utc)
        
        try:
            # Get list of symbols from local storage
            symbols_df = await local_storage.get_market_summary()
            
            if symbols_df.empty:
                self.logger.warning("No symbols found in local storage - migration skipped")
                stats.migration_end_time = datetime.now(timezone.utc)
                stats.migration_duration_seconds = 0.0
                return stats
            
            total_symbols = len(symbols_df)
            self.logger.info(f"Found {total_symbols} symbols to migrate")
            
            # Migrate each symbol's data
            for index, symbol_row in symbols_df.iterrows():
                symbol = symbol_row['symbol']
                
                try:
                    # Get OHLCV data for symbol
                    symbol_start_time = time.time()
                    ohlcv_data = await local_storage.get_ohlcv_bars(symbol)
                    
                    if not ohlcv_data.empty:
                        # Batch insert to TimescaleDB
                        bars_migrated = await self._batch_insert_ohlcv(symbol, ohlcv_data)
                        
                        stats.symbols_migrated += 1
                        stats.total_bars_migrated += bars_migrated
                        
                        symbol_duration = time.time() - symbol_start_time
                        self.logger.info(
                            f"Migrated {bars_migrated} bars for {symbol} in {symbol_duration:.2f}s "
                            f"({index + 1}/{total_symbols})"
                        )
                    else:
                        self.logger.info(f"No data found for {symbol} - skipping")
                        
                except Exception as e:
                    error_msg = f"Failed to migrate {symbol}: {e}"
                    self.logger.error(error_msg)
                    stats.errors_encountered.append(error_msg)
            
            # Calculate final statistics
            stats.migration_end_time = datetime.now(timezone.utc)
            stats.migration_duration_seconds = (stats.migration_end_time - stats.migration_start_time).total_seconds()
            
            if stats.symbols_migrated > 0:
                stats.average_bars_per_symbol = stats.total_bars_migrated / stats.symbols_migrated
                stats.migration_rate_bars_per_second = stats.total_bars_migrated / max(1, stats.migration_duration_seconds)
            
            self.logger.info(
                f"Migration completed: {stats.symbols_migrated}/{total_symbols} symbols, "
                f"{stats.total_bars_migrated:,} bars in {stats.migration_duration_seconds:.1f}s "
                f"({stats.migration_rate_bars_per_second:.1f} bars/sec)"
            )
            
            return stats
            
        except Exception as e:
            stats.migration_end_time = datetime.now(timezone.utc)
            if stats.migration_start_time:
                stats.migration_duration_seconds = (stats.migration_end_time - stats.migration_start_time).total_seconds()
            
            error_msg = f"Migration failed: {e}"
            self.logger.error(error_msg)
            stats.errors_encountered.append(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _batch_insert_ohlcv(self, symbol: str, ohlcv_df: pd.DataFrame) -> int:
        """
        Efficient batch insert of OHLCV data using asyncpg copy.
        
        Args:
            symbol: Trading symbol
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            Number of bars inserted
        """
        if ohlcv_df.empty:
            return 0
        
        # Prepare data tuples for efficient insertion
        data_tuples = []
        for _, row in ohlcv_df.iterrows():
            tuple_data = (
                symbol,
                row.get('timestamp'),
                float(row.get('open', 0)),
                float(row.get('high', 0)),
                float(row.get('low', 0)),
                float(row.get('close', 0)),
                float(row.get('volume', 0)),
                float(row.get('vwap', 0)),
                int(row.get('trade_count', 0)),
                1.0,  # data_quality_score
                'hyperliquid'  # source_exchange
            )
            data_tuples.append(tuple_data)
        
        # Batch insert using asyncpg copy (most efficient method)
        async with self.pool.acquire_connection() as conn:
            await conn.copy_records_to_table(
                'ohlcv_bars',
                records=data_tuples,
                columns=[
                    'symbol', 'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'vwap', 'trade_count', 'data_quality_score', 'source_exchange'
                ]
            )
        
        return len(data_tuples)
    
    async def create_evolution_state_entry(self, 
                                         evolution_id: str,
                                         phase: int,
                                         generation: int,
                                         population_data: Dict[str, Any],
                                         worker_id: Optional[str] = None,
                                         evolution_parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create new evolution state entry for distributed GA coordination.
        
        Args:
            evolution_id: Unique evolution identifier
            phase: Evolution phase number
            generation: Generation number
            population_data: Serializable population data
            worker_id: Ray worker identifier
            evolution_parameters: Evolution configuration parameters
            
        Returns:
            Created record UUID
        """
        insert_sql = """
        INSERT INTO evolution_state 
        (evolution_id, phase, generation, population_data, worker_id, evolution_parameters)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING evolution_id
        """
        
        async with self.pool.acquire_connection() as conn:
            result = await conn.fetchval(
                insert_sql,
                evolution_id,
                phase,
                generation,
                population_data,  # JSONB automatically serializes dict
                worker_id,
                evolution_parameters
            )
        
        self.logger.debug(f"Created evolution state entry: {evolution_id}, gen {generation}")
        return result
    
    async def update_evolution_state(self,
                                   evolution_id: str,
                                   generation: int,
                                   status: str,
                                   fitness_metrics: Optional[Dict[str, Any]] = None,
                                   best_individual: Optional[Dict[str, Any]] = None,
                                   performance_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update evolution state with results and metrics.
        
        Args:
            evolution_id: Evolution identifier
            generation: Generation number to update
            status: New status ('active', 'completed', 'failed', 'paused')
            fitness_metrics: Fitness evaluation results
            best_individual: Best individual from generation
            performance_metrics: Performance measurements
            
        Returns:
            True if update was successful
        """
        update_sql = """
        UPDATE evolution_state 
        SET status = $3,
            fitness_metrics = $4,
            best_individual = $5,
            performance_metrics = $6,
            updated_at = NOW()
        WHERE evolution_id = $1 AND generation = $2
        """
        
        async with self.pool.acquire_connection() as conn:
            result = await conn.execute(
                update_sql,
                evolution_id,
                generation,
                status,
                fitness_metrics,
                best_individual,
                performance_metrics
            )
        
        # Check if update was successful
        rows_affected = int(result.split()[-1])
        success = rows_affected > 0
        
        if success:
            self.logger.debug(f"Updated evolution state: {evolution_id}, gen {generation}, status {status}")
        else:
            self.logger.warning(f"No evolution state found to update: {evolution_id}, gen {generation}")
        
        return success
    
    async def get_evolution_state(self, evolution_id: str, generation: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve evolution state data for coordination.
        
        Args:
            evolution_id: Evolution identifier
            generation: Specific generation (None for all generations)
            
        Returns:
            List of evolution state records
        """
        if generation is not None:
            query_sql = """
            SELECT * FROM evolution_state 
            WHERE evolution_id = $1 AND generation = $2
            ORDER BY timestamp DESC
            """
            params = [evolution_id, generation]
        else:
            query_sql = """
            SELECT * FROM evolution_state 
            WHERE evolution_id = $1
            ORDER BY generation DESC, timestamp DESC
            """
            params = [evolution_id]
        
        async with self.pool.acquire_connection() as conn:
            result = await conn.fetch(query_sql, *params)
        
        # Convert to list of dictionaries
        return [dict(row) for row in result]
    
    async def cleanup_old_evolution_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old evolution state data to manage storage.
        
        Args:
            days_to_keep: Number of days of evolution data to retain
            
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        cleanup_sql = """
        DELETE FROM evolution_state 
        WHERE timestamp < $1 AND status IN ('completed', 'failed')
        """
        
        async with self.pool.acquire_connection() as conn:
            result = await conn.execute(cleanup_sql, cutoff_date)
        
        # Extract number of deleted rows
        rows_deleted = int(result.split()[-1])
        
        self.logger.info(f"Cleaned up {rows_deleted} old evolution state records (older than {days_to_keep} days)")
        return rows_deleted
    
    async def validate_schema_health(self) -> Dict[str, Any]:
        """
        Comprehensive schema health validation.
        
        Returns:
            Dict with detailed schema health status
        """
        health_results = {
            "schema_valid": False,
            "hypertables_healthy": False,
            "indexes_present": False,
            "data_integrity": False,
            "performance_metrics": {},
            "errors": []
        }
        
        try:
            async with self.pool.acquire_connection() as conn:
                # Check hypertables exist and are healthy
                hypertables_query = """
                SELECT schemaname, tablename, owner 
                FROM timescaledb_information.hypertables 
                WHERE tablename IN ('ohlcv_bars', 'evolution_state')
                """
                hypertables = await conn.fetch(hypertables_query)
                health_results["hypertables_healthy"] = len(hypertables) == 2
                
                # Check indexes exist
                indexes_query = """
                SELECT indexname FROM pg_indexes 
                WHERE tablename IN ('ohlcv_bars', 'evolution_state')
                AND indexname LIKE '%_idx'
                """
                indexes = await conn.fetch(indexes_query)
                health_results["indexes_present"] = len(indexes) >= 6  # Expected number of indexes
                
                # Check data integrity
                ohlcv_count = await conn.fetchval("SELECT count(*) FROM ohlcv_bars")
                evolution_count = await conn.fetchval("SELECT count(*) FROM evolution_state")
                
                health_results["data_integrity"] = True  # Basic existence check
                health_results["performance_metrics"] = {
                    "ohlcv_records": ohlcv_count,
                    "evolution_records": evolution_count,
                    "hypertables_count": len(hypertables),
                    "indexes_count": len(indexes)
                }
                
                # Overall health status
                health_results["schema_valid"] = (
                    health_results["hypertables_healthy"] and
                    health_results["indexes_present"] and
                    health_results["data_integrity"]
                )
                
        except Exception as e:
            error_msg = f"Schema health validation failed: {e}"
            health_results["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        return health_results