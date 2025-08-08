"""
Strategic Storage Interface - Phase 1 Foundation for Clean Progression

This module implements the minimal DataStorageInterface abstraction that enables
clean progression through Phases 1-4 without over-engineering EFS complexity.

Strategic Design Principles:
- Minimal abstraction for immediate Ray cluster functionality
- Clean upgrade path for Phase 4 Neon integration
- Forest-level architecture over trees-level complexity
- Business value focus over infrastructure complexity

Research-Based Implementation:
- /research/duckdb/ - Local storage optimization patterns
- Current data_storage.py patterns for backward compatibility
- Phase progression requirements analysis
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import os

# Import existing data structures
from src.data.market_data_pipeline import OHLCVBar
from src.data.data_storage import DataStorage
from src.config.settings import get_settings

# Set up logging
logger = logging.getLogger(__name__)


class DataStorageInterface(ABC):
    """
    Strategic storage interface for clean phase progression.
    
    This interface enables seamless backend switching from local storage
    (Phase 1-3) to cloud database (Phase 4) without code changes.
    """
    
    @abstractmethod
    async def store_ohlcv_bars(self, bars: List[OHLCVBar]) -> None:
        """Store OHLCV bars in the storage backend."""
        pass
    
    @abstractmethod
    async def get_ohlcv_bars(self, 
                           symbol: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve OHLCV bars from storage backend."""
        pass
    
    @abstractmethod  
    async def calculate_technical_indicators(self, 
                                           symbol: str,
                                           lookback_periods: int = 200) -> pd.DataFrame:
        """Calculate technical indicators using backend-specific optimizations."""
        pass
    
    @abstractmethod
    async def get_market_summary(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get market summary statistics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform storage backend health check."""
        pass


class LocalDataStorage(DataStorageInterface):
    """
    Local DuckDB storage implementation - Phase 1-3 foundation.
    
    This provides immediate Ray cluster functionality while preparing
    for seamless Phase 4 Neon upgrade via interface abstraction.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize local storage with thread-safe DuckDB backend."""
        self.db_path = db_path or "data/trading.duckdb"
        self.storage = DataStorage(db_path=self.db_path)
        logger.info(f"LocalDataStorage initialized with path: {self.db_path}")
    
    async def store_ohlcv_bars(self, bars: List[OHLCVBar]) -> None:
        """Store OHLCV bars using existing DuckDB storage."""
        try:
            # Use existing storage implementation
            await self.storage.store_ohlcv_bars(bars)
            logger.debug(f"Stored {len(bars)} OHLCV bars via LocalDataStorage")
        except Exception as e:
            logger.error(f"LocalDataStorage store_ohlcv_bars failed: {e}")
            raise
    
    async def get_ohlcv_bars(self, 
                           symbol: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Get OHLCV bars using existing DuckDB storage."""
        try:
            result = await self.storage.get_ohlcv_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            logger.debug(f"Retrieved {len(result)} OHLCV bars for {symbol}")
            return result
        except Exception as e:
            logger.error(f"LocalDataStorage get_ohlcv_bars failed: {e}")
            raise
    
    async def calculate_technical_indicators(self, 
                                           symbol: str,
                                           lookback_periods: int = 200) -> pd.DataFrame:
        """Calculate technical indicators using existing DuckDB storage."""
        try:
            result = await self.storage.calculate_technical_indicators(
                symbol=symbol, 
                lookback_periods=lookback_periods
            )
            logger.debug(f"Calculated technical indicators for {symbol}")
            return result
        except Exception as e:
            logger.error(f"LocalDataStorage calculate_technical_indicators failed: {e}")
            raise
    
    async def get_market_summary(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get market summary using existing DuckDB storage."""
        try:
            result = await self.storage.get_market_summary(symbols=symbols)
            logger.debug(f"Retrieved market summary for {len(symbols) if symbols else 'all'} symbols")
            return result
        except Exception as e:
            logger.error(f"LocalDataStorage get_market_summary failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive local storage health check with robust validation."""
        try:
            test_query_start = datetime.now()
            db_path = Path(self.db_path)
            
            # Test 1: Database connectivity and basic schema validation
            basic_connectivity = True
            schema_validation = True
            query_performance = 0.0
            
            try:
                # Test basic database connection and schema existence
                # This tests core functionality without requiring specific data
                test_result = await self.storage.get_market_summary(symbols=None)  # Get all available symbols
                query_performance = (datetime.now() - test_query_start).total_seconds()
                
                # If we have any data, great! If not, that's also valid for a fresh system
                data_available = test_result is not None and len(test_result) > 0
                
            except Exception as connectivity_error:
                logger.warning(f"Storage connectivity test encountered issue: {connectivity_error}")
                basic_connectivity = False
                schema_validation = False
                
            # Test 2: File system health
            db_accessible = db_path.exists() and os.access(db_path.parent, os.W_OK)
            
            # Determine overall health
            is_healthy = basic_connectivity and schema_validation and db_accessible
            
            return {
                "status": "healthy" if is_healthy else "degraded" if basic_connectivity else "unhealthy",
                "backend": "local_duckdb",
                "db_path": str(db_path.absolute()),
                "db_exists": db_path.exists(),
                "db_accessible": db_accessible,
                "db_size_mb": db_path.stat().st_size / (1024**2) if db_path.exists() else 0,
                "basic_connectivity": basic_connectivity,
                "schema_validation": schema_validation,
                "query_latency_ms": query_performance * 1000,
                "data_available": locals().get('data_available', False),
                "functional_validation": "robust_pipeline_validation",
                "validation_type": "fresh_system_compatible",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"LocalDataStorage health check failed: {e}")
            return {
                "status": "unhealthy",
                "backend": "local_duckdb",
                "error": str(e),
                "error_type": "health_check_exception",
                "timestamp": datetime.now().isoformat()
            }


class SharedDataStorage(DataStorageInterface):
    """
    Shared storage implementation for distributed Ray workers.
    
    This enables multiple Ray workers to access the same data storage,
    supporting immediate cloud deployment while preparing for Phase 4 Neon.
    
    Implementation Options:
    - Option A: Shared file system (NFS/EFS) - for cloud deployment
    - Option B: Shared local directory - for development testing
    """
    
    def __init__(self, shared_path: str = "/shared/data"):
        """Initialize shared storage with configurable path."""
        self.shared_path = Path(shared_path)
        self.db_path = self.shared_path / "shared_trading.duckdb"
        
        # Ensure shared directory exists
        self.shared_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize underlying storage
        self.storage = DataStorage(db_path=str(self.db_path))
        logger.info(f"SharedDataStorage initialized with path: {self.shared_path}")
    
    async def store_ohlcv_bars(self, bars: List[OHLCVBar]) -> None:
        """Store OHLCV bars on shared storage."""
        try:
            await self.storage.store_ohlcv_bars(bars)
            logger.debug(f"Stored {len(bars)} OHLCV bars via SharedDataStorage")
        except Exception as e:
            logger.error(f"SharedDataStorage store_ohlcv_bars failed: {e}")
            raise
    
    async def get_ohlcv_bars(self, 
                           symbol: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Get OHLCV bars from shared storage."""
        try:
            result = await self.storage.get_ohlcv_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            logger.debug(f"Retrieved {len(result)} OHLCV bars for {symbol} from shared storage")
            return result
        except Exception as e:
            logger.error(f"SharedDataStorage get_ohlcv_bars failed: {e}")
            raise
    
    async def calculate_technical_indicators(self, 
                                           symbol: str,
                                           lookback_periods: int = 200) -> pd.DataFrame:
        """Calculate technical indicators using shared storage."""
        try:
            result = await self.storage.calculate_technical_indicators(
                symbol=symbol, 
                lookback_periods=lookback_periods
            )
            logger.debug(f"Calculated technical indicators for {symbol} via shared storage")
            return result
        except Exception as e:
            logger.error(f"SharedDataStorage calculate_technical_indicators failed: {e}")
            raise
    
    async def get_market_summary(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get market summary from shared storage."""
        try:
            result = await self.storage.get_market_summary(symbols=symbols)
            logger.debug(f"Retrieved market summary from shared storage")
            return result
        except Exception as e:
            logger.error(f"SharedDataStorage get_market_summary failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform robust shared storage health check."""
        try:
            test_query_start = datetime.now()
            
            # Test 1: Shared directory accessibility
            shared_accessible = self.shared_path.exists() and os.access(self.shared_path, os.W_OK)
            
            # Test 2: Database connectivity (robust - doesn't require specific data)
            basic_connectivity = True
            query_performance = 0.0
            
            try:
                test_result = await self.storage.get_market_summary(symbols=None)
                query_performance = (datetime.now() - test_query_start).total_seconds()
                data_available = test_result is not None and len(test_result) > 0
            except Exception as connectivity_error:
                logger.warning(f"Shared storage connectivity test: {connectivity_error}")
                basic_connectivity = False
                data_available = False
            
            # Test 3: File system health
            db_accessible = self.db_path.exists() and os.access(self.db_path.parent, os.W_OK)
            
            # Overall health assessment
            is_healthy = shared_accessible and basic_connectivity and db_accessible
            
            return {
                "status": "healthy" if is_healthy else "degraded" if basic_connectivity else "unhealthy",
                "backend": "shared_duckdb", 
                "shared_path": str(self.shared_path.absolute()),
                "shared_directory_accessible": shared_accessible,
                "db_path": str(self.db_path.absolute()),
                "db_exists": self.db_path.exists(),
                "db_accessible": db_accessible,
                "db_size_mb": self.db_path.stat().st_size / (1024**2) if self.db_path.exists() else 0,
                "basic_connectivity": basic_connectivity,
                "query_latency_ms": query_performance * 1000,
                "data_available": locals().get('data_available', False),
                "shared_directory_writable": os.access(self.shared_path, os.W_OK) if self.shared_path.exists() else False,
                "functional_validation": "robust_shared_validation",
                "validation_type": "distributed_compatible",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"SharedDataStorage health check failed: {e}")
            return {
                "status": "unhealthy",
                "backend": "shared_duckdb",
                "error": str(e),
                "error_type": "health_check_exception",
                "shared_path": str(self.shared_path),
                "timestamp": datetime.now().isoformat()
            }


def get_storage_implementation() -> DataStorageInterface:
    """
    Get configured storage implementation based on environment settings.
    
    Strategic implementation selection:
    - 'local': LocalDataStorage for single-machine development
    - 'shared': SharedDataStorage for distributed Ray workers
    - 'neon': NeonDataStorage (Phase 4 - future implementation)
    """
    try:
        settings = get_settings()
        # Check environment variable first, then settings attribute
        storage_backend = os.environ.get('STORAGE_BACKEND') or getattr(settings, 'storage_backend', 'local')
        
        if storage_backend == 'shared':
            shared_path = getattr(settings, 'shared_storage_path', '/shared/data')
            logger.info(f"Using SharedDataStorage backend with path: {shared_path}")
            return SharedDataStorage(shared_path=shared_path)
        
        elif storage_backend == 'neon':
            # Phase 4: NeonHybridStorage implementation
            try:
                from .neon_hybrid_storage import create_neon_hybrid_storage
                logger.info("Using NeonHybridStorage backend")
                # Create async task to initialize hybrid storage
                import asyncio
                hybrid_storage = asyncio.create_task(create_neon_hybrid_storage(settings))
                # For now, return the hybrid storage instance (initialization happens async)
                from .neon_hybrid_storage import NeonHybridStorage
                return NeonHybridStorage(settings, auto_initialize=True)
            except ImportError as e:
                logger.warning(f"Neon storage backend not available: {e}. Using local storage.")
                return LocalDataStorage()
            except Exception as e:
                logger.error(f"Failed to initialize Neon storage: {e}. Using local storage.")
                return LocalDataStorage()
        
        else:
            # Default: Local storage
            db_path = getattr(settings, 'duckdb_path', None)
            logger.info("Using LocalDataStorage backend")
            return LocalDataStorage(db_path=db_path)
            
    except Exception as e:
        logger.warning(f"Storage configuration failed, using default local storage: {e}")
        return LocalDataStorage()


# Backward compatibility ensured by import at top