"""
Real-time Monitoring Core Engine - Central monitoring system for genetic trading.
Core monitoring engine with system health tracking and genetic algorithm evolution monitoring.
"""

import asyncio
import logging
import time
import sys
import os
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import warnings

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import get_settings, Settings
from src.execution.risk_management import GeneticRiskManager, RiskLevel, RiskMetrics, MarketRegime
from src.execution.paper_trading import PaperTradingEngine, StrategyPerformance, TradeExecutionQuality
from src.execution.position_sizer import GeneticPositionSizer, PositionSizeResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@dataclass
class SystemHealthMetrics:
    """System health monitoring metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_threads: int
    open_file_handles: int
    network_connections: int
    gc_collections: int
    system_load_avg: float


@dataclass
class GeneticEvolutionMetrics:
    """Genetic algorithm evolution monitoring metrics."""
    timestamp: datetime
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity_score: float
    evaluation_time: float
    population_size: int
    memory_usage_gb: float
    convergence_rate: float


@dataclass
class TradingPerformanceMetrics:
    """Trading performance monitoring metrics."""
    timestamp: datetime
    total_trades: int
    successful_trades: int
    failed_trades: int
    execution_success_rate: float
    avg_execution_time_ms: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    position_count: int
    exposure_percentage: float


class MonitoringStatus(Enum):
    """Overall monitoring system status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Alert categories for classification."""
    SYSTEM_HEALTH = "system_health"
    TRADING_PERFORMANCE = "trading_performance"
    RISK_MANAGEMENT = "risk_management"
    GENETIC_EVOLUTION = "genetic_evolution"
    DATA_PIPELINE = "data_pipeline"


@dataclass
class MonitoringSnapshot:
    """Complete monitoring snapshot at a point in time."""
    timestamp: datetime
    status: MonitoringStatus
    system_metrics: SystemHealthMetrics
    genetic_metrics: GeneticEvolutionMetrics
    trading_metrics: TradingPerformanceMetrics
    risk_level: RiskLevel
    active_circuit_breakers: List[str]
    market_regime: MarketRegime


class MonitoringEngine:
    """Core monitoring engine for real-time system observation."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize monitoring engine."""
        self.settings = settings or get_settings()
        self.is_running = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # seconds
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=1000)
        self.genetic_metrics_history = deque(maxlen=1000)
        self.trading_metrics_history = deque(maxlen=1000)
        self.monitoring_snapshots = deque(maxlen=1000)
        
        # Component references
        self.risk_manager = None
        self.paper_trading_engine = None
        self.position_sizer = None
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        
        logger.info("MonitoringEngine initialized")
    
    def start_monitoring(self, risk_manager: GeneticRiskManager = None,
                        paper_trading_engine: PaperTradingEngine = None,
                        position_sizer: GeneticPositionSizer = None):
        """Start the monitoring system."""
        if self.is_running:
            logger.warning("Monitoring system is already running")
            return
        
        # Store component references
        self.risk_manager = risk_manager
        self.paper_trading_engine = paper_trading_engine
        self.position_sizer = position_sizer
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_running:
            logger.warning("Monitoring system is not running")
            return
        
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Monitoring loop started")
        
        while self.is_running:
            try:
                # Collect metrics
                system_metrics = self._collect_system_metrics()
                genetic_metrics = self._collect_genetic_metrics()
                trading_metrics = self._collect_trading_metrics()
                risk_level, circuit_breakers, market_regime = self._collect_risk_metrics()
                
                # Create monitoring snapshot
                snapshot = MonitoringSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    status=self._determine_monitoring_status(
                        system_metrics, genetic_metrics, trading_metrics, risk_level
                    ),
                    system_metrics=system_metrics,
                    genetic_metrics=genetic_metrics,
                    trading_metrics=trading_metrics,
                    risk_level=risk_level,
                    active_circuit_breakers=circuit_breakers,
                    market_regime=market_regime
                )
                
                # Store snapshot
                with self._metrics_lock:
                    self.monitoring_snapshots.append(snapshot)
                    self.system_metrics_history.append(system_metrics)
                    self.genetic_metrics_history.append(genetic_metrics)
                    self.trading_metrics_history.append(trading_metrics)
                
                # Log status periodically
                self._log_monitoring_status(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect system health metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            threads = process.num_threads()
            fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            connections = len(process.connections())
            
            # System load
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            
            # GC collections (approximate)
            import gc
            gc_collections = len(gc.get_objects())
            
            return SystemHealthMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_threads=threads,
                open_file_handles=fds,
                network_connections=connections,
                gc_collections=gc_collections,
                system_load_avg=load_avg
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemHealthMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                active_threads=0,
                open_file_handles=0,
                network_connections=0,
                gc_collections=0,
                system_load_avg=0.0
            )
    
    def _collect_genetic_metrics(self) -> GeneticEvolutionMetrics:
        """Collect genetic algorithm evolution metrics."""
        # Default metrics for when genetic engine is not available
        default_metrics = GeneticEvolutionMetrics(
            timestamp=datetime.now(timezone.utc),
            generation=0,
            best_fitness=0.0,
            avg_fitness=0.0,
            worst_fitness=0.0,
            diversity_score=0.0,
            evaluation_time=0.0,
            population_size=0,
            memory_usage_gb=0.0,
            convergence_rate=0.0
        )
        
        try:
            # TODO: Integrate with GeneticEngine when available
            # For now, return default metrics
            return default_metrics
        except Exception as e:
            logger.error(f"Error collecting genetic metrics: {e}")
            return default_metrics
    
    def _collect_trading_metrics(self) -> TradingPerformanceMetrics:
        """Collect trading performance metrics."""
        default_metrics = TradingPerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            total_trades=0,
            successful_trades=0,
            failed_trades=0,
            execution_success_rate=1.0,
            avg_execution_time_ms=0.0,
            total_pnl=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            position_count=0,
            exposure_percentage=0.0
        )
        
        try:
            if self.paper_trading_engine:
                # Get performance metrics from paper trading engine
                performance = self.paper_trading_engine.get_performance_summary()
                
                return TradingPerformanceMetrics(
                    timestamp=datetime.now(timezone.utc),
                    total_trades=performance.get('total_trades', 0),
                    successful_trades=performance.get('successful_trades', 0),
                    failed_trades=performance.get('failed_trades', 0),
                    execution_success_rate=performance.get('success_rate', 1.0),
                    avg_execution_time_ms=performance.get('avg_execution_time', 0.0),
                    total_pnl=performance.get('total_pnl', 0.0),
                    realized_pnl=performance.get('realized_pnl', 0.0),
                    unrealized_pnl=performance.get('unrealized_pnl', 0.0),
                    sharpe_ratio=performance.get('sharpe_ratio'),
                    max_drawdown=performance.get('max_drawdown', 0.0),
                    position_count=performance.get('position_count', 0),
                    exposure_percentage=performance.get('exposure_percentage', 0.0)
                )
            
            return default_metrics
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return default_metrics
    
    def _collect_risk_metrics(self) -> Tuple[RiskLevel, List[str], MarketRegime]:
        """Collect risk management metrics."""
        try:
            if self.risk_manager:
                risk_metrics = self.risk_manager.get_risk_metrics()
                return (
                    risk_metrics.risk_level,
                    [str(cb) for cb in risk_metrics.active_circuit_breakers],
                    risk_metrics.current_regime
                )
            return RiskLevel.LOW, [], MarketRegime.UNKNOWN
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
            return RiskLevel.CRITICAL, [], MarketRegime.UNKNOWN
    
    def _determine_monitoring_status(self, system_metrics: SystemHealthMetrics,
                                   genetic_metrics: GeneticEvolutionMetrics,
                                   trading_metrics: TradingPerformanceMetrics,
                                   risk_level: RiskLevel) -> MonitoringStatus:
        """Determine overall monitoring system status."""
        
        # Critical conditions
        if (risk_level == RiskLevel.EMERGENCY or
            system_metrics.cpu_usage_percent > 95 or
            system_metrics.memory_usage_percent > 95):
            return MonitoringStatus.CRITICAL
        
        # Degraded conditions
        if (risk_level == RiskLevel.CRITICAL or
            system_metrics.cpu_usage_percent > 80 or
            system_metrics.memory_usage_percent > 80 or
            trading_metrics.execution_success_rate < 0.9):
            return MonitoringStatus.DEGRADED
        
        # Warning conditions
        if (risk_level == RiskLevel.HIGH or
            system_metrics.cpu_usage_percent > 70 or
            system_metrics.memory_usage_percent > 70 or
            trading_metrics.execution_success_rate < 0.95):
            return MonitoringStatus.WARNING
        
        return MonitoringStatus.HEALTHY
    
    def _log_monitoring_status(self, snapshot: MonitoringSnapshot):
        """Log periodic monitoring status."""
        logger.info(f"Monitoring Status: {snapshot.status}")
        logger.debug(f"System: CPU {snapshot.system_metrics.cpu_usage_percent:.1f}% "
                    f"Memory {snapshot.system_metrics.memory_usage_percent:.1f}%")
        logger.debug(f"Trading: {snapshot.trading_metrics.total_trades} trades "
                    f"Success rate: {snapshot.trading_metrics.execution_success_rate:.2f}")
    
    def get_latest_snapshot(self) -> Optional[MonitoringSnapshot]:
        """Get the latest monitoring snapshot."""
        with self._metrics_lock:
            return self.monitoring_snapshots[-1] if self.monitoring_snapshots else None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        snapshot = self.get_latest_snapshot()
        if not snapshot:
            return {'status': 'unknown', 'message': 'No monitoring data available'}
        
        return {
            'status': snapshot.status.value,
            'timestamp': snapshot.timestamp.isoformat(),
            'system': {
                'cpu_percent': snapshot.system_metrics.cpu_usage_percent,
                'memory_percent': snapshot.system_metrics.memory_usage_percent,
                'disk_percent': snapshot.system_metrics.disk_usage_percent,
                'threads': snapshot.system_metrics.active_threads
            },
            'trading': {
                'total_trades': snapshot.trading_metrics.total_trades,
                'success_rate': snapshot.trading_metrics.execution_success_rate,
                'pnl': snapshot.trading_metrics.total_pnl
            },
            'risk_level': snapshot.risk_level.value,
            'active_breakers': snapshot.active_circuit_breakers
        }


class MetricCollector:
    """Collects and aggregates monitoring metrics."""
    
    def __init__(self):
        """Initialize metric collector."""
        self.metrics_cache = {}
        self._cache_lock = threading.Lock()
    
    def collect_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Collect a single metric."""
        timestamp = datetime.now(timezone.utc)
        tags = tags or {}
        
        with self._cache_lock:
            if metric_name not in self.metrics_cache:
                self.metrics_cache[metric_name] = deque(maxlen=1000)
            
            self.metrics_cache[metric_name].append({
                'timestamp': timestamp,
                'value': value,
                'tags': tags
            })
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get metric history for specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._cache_lock:
            if metric_name not in self.metrics_cache:
                return []
            
            return [
                metric for metric in self.metrics_cache[metric_name]
                if metric['timestamp'] > cutoff_time
            ]


class SystemHealthTracker:
    """Tracks system health metrics and trends."""
    
    def __init__(self):
        """Initialize system health tracker."""
        self.health_history = deque(maxlen=1000)
        self._health_lock = threading.Lock()
        logger.info("SystemHealthTracker initialized")
    
    def record_health_check(self, metrics: SystemHealthMetrics):
        """Record a system health check."""
        with self._health_lock:
            self.health_history.append(metrics)
    
    def get_health_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trend analysis."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._health_lock:
            recent_metrics = [
                m for m in self.health_history
                if m.timestamp > cutoff_time
            ]
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_percent for m in recent_metrics])
        avg_disk = np.mean([m.disk_usage_percent for m in recent_metrics])
        
        return {
            'avg_cpu': float(avg_cpu),
            'avg_memory': float(avg_memory),
            'avg_disk': float(avg_disk),
            'sample_count': len(recent_metrics),
            'time_period_hours': hours
        }