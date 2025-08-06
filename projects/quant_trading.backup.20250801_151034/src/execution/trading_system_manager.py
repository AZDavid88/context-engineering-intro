"""
Trading System Manager - Centralized Async Session Coordination

This module implements a centralized async context manager that coordinates all 
async sessions and resources across the quantitative trading system to eliminate
session warnings and optimize resource management.

Based on research from:
- /research/asyncio_advanced/page_4_streams_websocket_integration.md
- /research/aiofiles_v3/vector4_asyncio_integration.md

Key Features:
- Centralized async session lifecycle management
- Dependency-aware initialization and cleanup order
- Connection pooling and resource optimization
- Graceful error recovery and timeout handling
- Production-ready monitoring integration
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import aiohttp
from collections import defaultdict

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import get_settings, Settings
from src.data.fear_greed_client import FearGreedClient
from src.execution.risk_management import GeneticRiskManager, GeneticRiskGenome
from src.execution.paper_trading import PaperTradingEngine, PaperTradingMode
from src.execution.position_sizer import GeneticPositionSizer
from src.execution.monitoring import RealTimeMonitoringSystem
from src.execution.retail_connection_optimizer import (
    RetailConnectionOptimizer, TradingSessionProfile, TradingTimeframe, 
    ConnectionUsagePattern, SCALPING_SESSION, INTRADAY_SESSION, SWING_SESSION
)

# Configure logging
logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session status states for tracking component health."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    ERROR = "error"


@dataclass
class SessionHealth:
    """Health metrics for async session management."""
    component_name: str
    status: SessionStatus
    connection_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    session_id: Optional[str] = None


class AsyncResourceManager:
    """Helper class for managing async resources with proper cleanup."""
    
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self.resources: List[Any] = []
        self.cleanup_callbacks: List[callable] = []
    
    def register_resource(self, resource: Any, cleanup_callback: Optional[callable] = None):
        """Register a resource for automatic cleanup."""
        self.resources.append(resource)
        if cleanup_callback:
            self.cleanup_callbacks.append(cleanup_callback)
    
    async def cleanup_all(self):
        """Clean up all registered resources."""
        errors = []
        
        # Execute cleanup callbacks first
        for callback in reversed(self.cleanup_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    # Handle lambda functions that return coroutines
                    result = callback()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                errors.append(f"Callback cleanup error: {e}")
                self.logger.warning(f"Resource cleanup callback failed: {e}")
        
        # Clean up resources
        for resource in reversed(self.resources):
            try:
                if hasattr(resource, 'close') and callable(resource.close):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                elif hasattr(resource, 'disconnect') and callable(resource.disconnect):
                    if asyncio.iscoroutinefunction(resource.disconnect):
                        await resource.disconnect()
                    else:
                        resource.disconnect()
            except Exception as e:
                errors.append(f"Resource cleanup error: {e}")
                self.logger.warning(f"Resource cleanup failed for {resource}: {e}")
        
        if errors:
            self.logger.warning(f"Resource manager '{self.name}' had {len(errors)} cleanup errors")
        else:
            self.logger.info(f"Resource manager '{self.name}' cleaned up {len(self.resources)} resources")
        
        self.resources.clear()
        self.cleanup_callbacks.clear()


class TradingSystemManager:
    """
    Centralized async session manager for the quantitative trading system.
    
    This class coordinates all async sessions and resources to ensure:
    - Proper initialization order based on dependencies
    - Clean resource management and session cleanup
    - Elimination of async session warnings
    - Optimal resource utilization and performance
    """
    
    def __init__(self, settings: Optional[Settings] = None, 
                 trading_session: Optional[TradingSessionProfile] = None):
        """Initialize trading system manager.
        
        Args:
            settings: Configuration settings (uses global settings if None)
            trading_session: Trading session profile for connection optimization
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.TradingSystemManager")
        
        # Initialize retail connection optimizer
        self.connection_optimizer = RetailConnectionOptimizer(self.settings)
        if trading_session:
            self.connection_optimizer.register_trading_session(trading_session)
            self.logger.info(f"Trading session registered: {trading_session.timeframe.value}")
        
        # Component health tracking
        self.component_health: Dict[str, SessionHealth] = {}
        self.initialization_order: List[str] = []
        self.startup_time: Optional[datetime] = None
        
        # Resource managers for different component types
        self.resource_managers: Dict[str, AsyncResourceManager] = {
            'data_clients': AsyncResourceManager('DataClients', self.logger),
            'trading_engines': AsyncResourceManager('TradingEngines', self.logger),
            'monitoring': AsyncResourceManager('Monitoring', self.logger)
        }
        
        # Core components (will be initialized in __aenter__)
        self.fear_greed_client: Optional[FearGreedClient] = None
        self.risk_manager: Optional[GeneticRiskManager] = None
        self.paper_trading: Optional[PaperTradingEngine] = None
        self.position_sizer: Optional[GeneticPositionSizer] = None
        self.monitoring: Optional[RealTimeMonitoringSystem] = None
        
        # Session management state
        self.active_sessions: Dict[str, Any] = {}
        self.connection_pool: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # Performance metrics
        self.operation_count = 0
        self.total_operation_time = 0.0
    
    async def __aenter__(self):
        """Async context manager entry - initialize all components."""
        self.logger.info("🚀 Starting trading system initialization...")
        self.startup_time = datetime.now(timezone.utc)
        
        try:
            # Step 1: Initialize shared connection pool
            await self._initialize_connection_pool()
            
            # Step 2: Initialize core data clients (foundation layer)
            await self._initialize_data_clients()
            
            # Step 3: Initialize trading engines (business logic layer)
            await self._initialize_trading_engines()
            
            # Step 4: Initialize monitoring system (observability layer)
            await self._initialize_monitoring()
            
            # Step 5: Validate all components are healthy
            await self._validate_system_health()
            
            self.is_initialized = True
            self.logger.info("✅ Trading system initialization complete")
            
            return self
            
        except Exception as e:
            self.logger.error(f"❌ Trading system initialization failed: {e}")
            # Cleanup on failure
            await self._emergency_cleanup()
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - clean up all components."""
        self.logger.info("🔄 Starting trading system shutdown...")
        
        try:
            # Shutdown in reverse order of initialization
            await self._shutdown_monitoring()
            await self._shutdown_trading_engines()
            await self._shutdown_data_clients()
            await self._shutdown_connection_pool()
            
            # Final cleanup of resource managers
            for manager in self.resource_managers.values():
                await manager.cleanup_all()
            
            self.is_initialized = False
            self.logger.info("✅ Trading system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Trading system shutdown error: {e}")
            # Still attempt emergency cleanup
            await self._emergency_cleanup()
            
        finally:
            # Reset state
            self.component_health.clear()
            self.active_sessions.clear()
            self.initialization_order.clear()
    
    async def _initialize_connection_pool(self):
        """Initialize shared HTTP connection pool with retail trading optimization."""
        self.logger.info("🔧 Initializing optimized connection pool...")
        
        # Get optimized settings from retail connection optimizer
        timeout = self.connection_optimizer.get_optimal_timeout_settings()
        connector_settings = self.connection_optimizer.get_optimal_connector_settings()
        
        # Create optimized connector for retail trading
        connector = aiohttp.TCPConnector(**connector_settings)
        
        # Create shared session with trading-optimized headers
        headers = {
            "User-Agent": "QuantTradingOrganism/4.1.0",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
        
        # Use json for serialization (orjson is optional optimization)
        try:
            import orjson
            json_serialize = lambda obj: orjson.dumps(obj).decode()
        except ImportError:
            import json
            json_serialize = json.dumps
        
        self.connection_pool = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
            json_serialize=json_serialize
        )
        
        # Register for cleanup
        self.resource_managers['data_clients'].register_resource(
            self.connection_pool,
            self.connection_pool.close
        )
        
        self._update_component_health("connection_pool", SessionStatus.CONNECTED)
        self.logger.info("✅ Connection pool initialized")
    
    async def _initialize_data_clients(self):
        """Initialize data client components."""
        self.logger.info("📊 Initializing data clients...")
        
        # Initialize Fear & Greed client with shared session
        self.fear_greed_client = FearGreedClient(self.settings)
        self.fear_greed_client.set_shared_session(self.connection_pool)  # Share connection pool safely
        
        # Connect to external API
        self._update_component_health("fear_greed_client", SessionStatus.CONNECTING)
        try:
            # Test connection to validate API availability
            await self.fear_greed_client.get_current_index(use_cache=False)
            self._update_component_health("fear_greed_client", SessionStatus.CONNECTED)
            self.logger.info("✅ Fear & Greed client connected")
        except Exception as e:
            self._update_component_health("fear_greed_client", SessionStatus.ERROR, str(e))
            self.logger.warning(f"⚠️  Fear & Greed client connection issue: {e}")
        
        # Register for cleanup (but don't close shared session)
        self.resource_managers['data_clients'].register_resource(
            self.fear_greed_client,
            lambda: self._safe_disconnect(self.fear_greed_client)
        )
        
        self.initialization_order.append("data_clients")
    
    async def _initialize_trading_engines(self):
        """Initialize trading engine components."""
        self.logger.info("⚡ Initializing trading engines...")
        
        # Initialize risk manager with genetic genome
        genetic_genome = GeneticRiskGenome(
            stop_loss_percentage=0.03,
            max_position_size=0.15,
            daily_drawdown_limit=0.02
        )
        
        self.risk_manager = GeneticRiskManager(self.settings, genetic_genome)
        
        # Replace the auto-created FearGreedClient with our managed one
        # First, cleanup any session that might have been created by the original client
        original_client = self.risk_manager.regime_detector.fear_greed_client
        if original_client and original_client.session:
            try:
                # Only disconnect if it's not already our shared session
                if original_client.session != self.connection_pool:
                    await original_client.disconnect()
                    self.logger.info("🧹 Cleaned up original FearGreedClient session")
                else:
                    self.logger.info("🔄 Original FearGreedClient already using shared session")
            except Exception as e:
                self.logger.warning(f"⚠️  Error cleaning up original FearGreedClient: {e}")
        
        # Now use our managed client with shared session
        self.risk_manager.regime_detector.fear_greed_client = self.fear_greed_client
        self.logger.info("🔗 Risk manager now using shared FearGreedClient")
        
        # Initialize position sizer
        self.position_sizer = GeneticPositionSizer(self.settings)
        
        # Initialize paper trading engine
        self.paper_trading = PaperTradingEngine(self.settings, PaperTradingMode.SIMULATION)
        
        # Register components for cleanup
        self.resource_managers['trading_engines'].register_resource(
            self.risk_manager,
            self.risk_manager.cleanup
        )
        
        self.resource_managers['trading_engines'].register_resource(
            self.position_sizer,
            lambda: self._safe_disconnect(self.position_sizer)
        )
        
        self.resource_managers['trading_engines'].register_resource(
            self.paper_trading,
            lambda: self._safe_disconnect(self.paper_trading)
        )
        
        # Update health status
        self._update_component_health("risk_manager", SessionStatus.CONNECTED)
        self._update_component_health("position_sizer", SessionStatus.CONNECTED)
        self._update_component_health("paper_trading", SessionStatus.CONNECTED)
        
        self.initialization_order.append("trading_engines")
        self.logger.info("✅ Trading engines initialized")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring system."""
        self.logger.info("📈 Initializing monitoring system...")
        
        # Initialize monitoring system
        self.monitoring = RealTimeMonitoringSystem(self.settings)
        
        # Inject all components for monitoring
        self.monitoring.inject_components(
            risk_manager=self.risk_manager,
            paper_trading=self.paper_trading,
            position_sizer=self.position_sizer
        )
        
        # Register for cleanup
        self.resource_managers['monitoring'].register_resource(
            self.monitoring,
            lambda: self._safe_disconnect(self.monitoring)
        )
        
        self._update_component_health("monitoring", SessionStatus.CONNECTED)
        self.initialization_order.append("monitoring")
        self.logger.info("✅ Monitoring system initialized")
    
    async def _validate_system_health(self):
        """Validate that all components are healthy."""
        self.logger.info("🔍 Validating system health...")
        
        healthy_components = 0
        total_components = len(self.component_health)
        
        for component_name, health in self.component_health.items():
            if health.status == SessionStatus.CONNECTED:
                healthy_components += 1
            elif health.status == SessionStatus.ERROR:
                self.logger.warning(f"⚠️  Component {component_name} has errors: {health.last_error}")
        
        health_ratio = healthy_components / total_components if total_components > 0 else 0
        
        if health_ratio >= 0.8:  # 80% components healthy
            self.logger.info(f"✅ System health validation passed: {healthy_components}/{total_components} components healthy")
        else:
            self.logger.warning(f"⚠️  System health degraded: {healthy_components}/{total_components} components healthy")
    
    async def _shutdown_monitoring(self):
        """Shutdown monitoring system."""
        if self.monitoring:
            self.logger.info("📈 Shutting down monitoring system...")
            self._update_component_health("monitoring", SessionStatus.DISCONNECTING)
            await self.resource_managers['monitoring'].cleanup_all()
            self._update_component_health("monitoring", SessionStatus.DISCONNECTED)
    
    async def _shutdown_trading_engines(self):
        """Shutdown trading engines."""
        self.logger.info("⚡ Shutting down trading engines...")
        
        for component in ["risk_manager", "position_sizer", "paper_trading"]:
            if component in self.component_health:
                self._update_component_health(component, SessionStatus.DISCONNECTING)
        
        await self.resource_managers['trading_engines'].cleanup_all()
        
        for component in ["risk_manager", "position_sizer", "paper_trading"]:
            if component in self.component_health:
                self._update_component_health(component, SessionStatus.DISCONNECTED)
    
    async def _shutdown_data_clients(self):
        """Shutdown data clients."""
        self.logger.info("📊 Shutting down data clients...")
        
        if "fear_greed_client" in self.component_health:
            self._update_component_health("fear_greed_client", SessionStatus.DISCONNECTING)
        
        # Don't close the shared session here - it will be closed with connection pool
        if self.fear_greed_client:
            self.fear_greed_client.session = None  # Remove reference
        
        self._update_component_health("fear_greed_client", SessionStatus.DISCONNECTED)
    
    async def _shutdown_connection_pool(self):
        """Shutdown shared connection pool."""
        if self.connection_pool:
            self.logger.info("🔧 Shutting down connection pool...")
            self._update_component_health("connection_pool", SessionStatus.DISCONNECTING)
            
            try:
                await self.connection_pool.close()
                # Wait for underlying connections to close
                await asyncio.sleep(0.1)
                self._update_component_health("connection_pool", SessionStatus.DISCONNECTED)
                self.logger.info("✅ Connection pool shut down cleanly")
            except Exception as e:
                self.logger.warning(f"⚠️  Connection pool shutdown warning: {e}")
    
    async def _emergency_cleanup(self):
        """Emergency cleanup in case of initialization failure."""
        self.logger.warning("🚨 Performing emergency cleanup...")
        
        try:
            # Attempt to clean up all resource managers
            for manager in self.resource_managers.values():
                try:
                    await manager.cleanup_all()
                except Exception as e:
                    self.logger.error(f"Emergency cleanup error in {manager.name}: {e}")
            
            # Force close connection pool if it exists
            if self.connection_pool:
                try:
                    await self.connection_pool.close()
                except Exception as e:
                    self.logger.error(f"Emergency connection pool cleanup error: {e}")
            
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")
    
    async def _safe_disconnect(self, component):
        """Safely disconnect a component, handling various interface types."""
        if component is None:
            return
        
        try:
            if hasattr(component, 'disconnect') and callable(component.disconnect):
                if asyncio.iscoroutinefunction(component.disconnect):
                    await component.disconnect()
                else:
                    component.disconnect()
            elif hasattr(component, 'cleanup') and callable(component.cleanup):
                if asyncio.iscoroutinefunction(component.cleanup):
                    await component.cleanup()
                else:
                    component.cleanup()
            elif hasattr(component, 'close') and callable(component.close):
                if asyncio.iscoroutinefunction(component.close):
                    await component.close()
                else:
                    component.close()
        except Exception as e:
            self.logger.warning(f"Safe disconnect warning for {component}: {e}")
    
    def _update_component_health(self, component_name: str, status: SessionStatus, error: Optional[str] = None):
        """Update component health status."""
        if component_name not in self.component_health:
            self.component_health[component_name] = SessionHealth(
                component_name=component_name,
                status=status
            )
        
        health = self.component_health[component_name]
        health.status = status
        health.last_activity = datetime.now(timezone.utc)
        
        if status == SessionStatus.CONNECTED and health.connection_time is None:
            health.connection_time = datetime.now(timezone.utc)
        
        if error:
            health.error_count += 1
            health.last_error = error
    
    # Public API methods for trading operations
    
    async def execute_trading_operation(self, operation_name: str, **kwargs) -> Any:
        """Execute a trading operation with performance monitoring and optimization."""
        if not self.is_initialized:
            raise RuntimeError("Trading system not initialized")
        
        start_time = time.perf_counter()
        self.operation_count += 1
        
        try:
            self.logger.debug(f"🔄 Executing trading operation: {operation_name}")
            
            # Route to appropriate component based on operation
            if operation_name == "get_fear_greed_index":
                result = await self.fear_greed_client.get_current_index()
                api_name = "fear_greed"
            elif operation_name == "evaluate_risk":
                result = await self.risk_manager.evaluate_trade_risk(**kwargs)
                api_name = "risk_evaluation"
            elif operation_name == "execute_paper_trade":
                result = await self.paper_trading.execute_paper_trade(**kwargs)
                api_name = "paper_trading"
            elif operation_name == "collect_monitoring_snapshot":
                result = self.monitoring.collect_monitoring_snapshot()
                api_name = "monitoring"
            else:
                raise ValueError(f"Unknown trading operation: {operation_name}")
            
            operation_time = time.perf_counter() - start_time
            operation_time_ms = operation_time * 1000  # Convert to milliseconds
            self.total_operation_time += operation_time
            
            # Record performance for optimization
            self.connection_optimizer.record_api_performance(
                api_name=api_name,
                response_time_ms=operation_time_ms,
                success=True,
                bytes_transferred=1024  # Estimated bytes
            )
            
            self.logger.debug(f"✅ Trading operation '{operation_name}' completed in {operation_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            operation_time = time.perf_counter() - start_time
            operation_time_ms = operation_time * 1000
            
            # Record failed operation
            if 'api_name' in locals():
                self.connection_optimizer.record_api_performance(
                    api_name=api_name,
                    response_time_ms=operation_time_ms,
                    success=False
                )
            
            self.logger.error(f"❌ Trading operation '{operation_name}' failed after {operation_time_ms:.1f}ms: {e}")
            raise
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        if not self.is_initialized:
            return {"status": "not_initialized", "components": {}}
        
        component_statuses = {}
        connected_count = 0
        error_count = 0
        
        for name, health in self.component_health.items():
            component_statuses[name] = {
                "status": health.status.value,
                "connection_time": health.connection_time.isoformat() if health.connection_time else None,
                "last_activity": health.last_activity.isoformat() if health.last_activity else None,
                "error_count": health.error_count,
                "last_error": health.last_error
            }
            
            if health.status == SessionStatus.CONNECTED:
                connected_count += 1
            elif health.status == SessionStatus.ERROR:
                error_count += 1
        
        total_components = len(self.component_health)
        health_score = (connected_count / total_components * 100) if total_components > 0 else 0
        
        uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds() if self.startup_time else 0
        avg_operation_time = (self.total_operation_time / self.operation_count) if self.operation_count > 0 else 0
        
        return {
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "critical",
            "health_score": health_score,
            "uptime_seconds": uptime,
            "components": component_statuses,
            "performance": {
                "total_operations": self.operation_count,
                "average_operation_time": avg_operation_time,
                "operations_per_second": self.operation_count / uptime if uptime > 0 else 0
            },
            "summary": {
                "total_components": total_components,
                "connected_components": connected_count,
                "error_components": error_count,
                "initialization_order": self.initialization_order
            },
            "connection_optimization": self.connection_optimizer.get_performance_summary()
        }


# Factory function for easy instantiation
def create_trading_system_manager(settings: Optional[Settings] = None, 
                                trading_session: Optional[TradingSessionProfile] = None) -> TradingSystemManager:
    """Factory function to create a properly configured trading system manager.
    
    Args:
        settings: Configuration settings (uses global settings if None)
        trading_session: Trading session profile for connection optimization
    """
    return TradingSystemManager(settings, trading_session)

# Convenience factory functions for common retail trading patterns
def create_scalping_trading_system(settings: Optional[Settings] = None) -> TradingSystemManager:
    """Create a trading system optimized for scalping (1-5 minute strategies)."""
    return create_trading_system_manager(settings, SCALPING_SESSION)

def create_intraday_trading_system(settings: Optional[Settings] = None) -> TradingSystemManager:
    """Create a trading system optimized for intraday trading (15min-4h strategies)."""
    return create_trading_system_manager(settings, INTRADAY_SESSION)

def create_swing_trading_system(settings: Optional[Settings] = None) -> TradingSystemManager:
    """Create a trading system optimized for swing trading (4h-1d strategies)."""
    return create_trading_system_manager(settings, SWING_SESSION)


if __name__ == "__main__":
    """Test trading system manager functionality."""
    
    async def test_trading_system_manager():
        """Test the trading system manager."""
        
        print("=== Trading System Manager Test ===")
        
        try:
            async with TradingSystemManager() as manager:
                print("✅ Trading system initialized successfully")
                
                # Test system health
                health = manager.get_system_health_summary()
                print(f"📊 System Health Score: {health['health_score']:.1f}%")
                print(f"📈 Connected Components: {health['summary']['connected_components']}/{health['summary']['total_components']}")
                
                # Test trading operations
                fear_greed_data = await manager.execute_trading_operation("get_fear_greed_index")
                print(f"📊 Fear & Greed Index: {fear_greed_data.value} ({fear_greed_data.regime.value})")
                
                # Test monitoring snapshot
                snapshot = await manager.execute_trading_operation("collect_monitoring_snapshot")
                print(f"📈 Monitoring Status: {snapshot.status}")
                
                print("✅ All trading operations completed successfully")
                
        except Exception as e:
            print(f"❌ Trading system manager test failed: {e}")
            raise
        
        print("✅ Trading system shutdown completed - checking for session warnings...")
        
        # Wait a moment to see if any async warnings appear
        await asyncio.sleep(1)
        
        print("✅ Trading System Manager test completed")
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_trading_system_manager())