"""
Retail Trading Connection Optimizer - Phase 4B Implementation

This module implements connection pool optimization specifically designed for 
retail quantitative trading systems focusing on scalping, intraday, and swing trading.

Target Performance Improvements:
- API Response Time: 200ms → 150ms average  
- Resource Efficiency: 30% reduction in connection overhead
- Reliability: 99.9% uptime for trading sessions
- Scalability: Handle 10-50 concurrent strategy evaluations

Based on research from:
- /research/asyncio_advanced/page_2_queues_producer_consumer.md
- /research/aiofiles_v3/vector4_asyncio_integration.md
- Real-world retail trading patterns and requirements
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Tuple, Deque
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import aiohttp

# Add project root to Python path for imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our configuration system
from src.config.settings import get_settings, Settings


class TradingTimeframe(str, Enum):
    """Trading timeframes for retail optimization."""
    SCALPING = "scalping"        # 1-5 minute strategies
    INTRADAY = "intraday"        # 15min-4h strategies  
    SWING = "swing"              # 4h-1d strategies
    PORTFOLIO = "portfolio"      # Multi-day position management


class ConnectionUsagePattern(str, Enum):
    """Connection usage patterns for optimization."""
    BURST = "burst"              # High frequency during market open/close
    STEADY = "steady"            # Consistent usage throughout session
    INTERMITTENT = "intermittent"  # Periodic checks and updates
    IDLE = "idle"                # Minimal usage, maintain connectivity


@dataclass
class ConnectionMetrics:
    """Real-time connection performance metrics."""
    
    response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_transferred: int = 0
    last_activity: Optional[datetime] = None
    current_connections: int = 0
    peak_connections: int = 0
    
    def add_response_time(self, response_time_ms: float):
        """Add response time measurement."""
        self.response_times.append(response_time_ms)
        self.successful_requests += 1
        self.last_activity = datetime.now(timezone.utc)
    
    def get_average_response_time(self) -> float:
        """Get average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if len(self.response_times) < 5:
            return self.get_average_response_time()
        sorted_times = sorted(self.response_times)
        p95_index = int(len(sorted_times) * 0.95)
        return sorted_times[p95_index]


@dataclass
class TradingSessionProfile:
    """Trading session characteristics for connection optimization."""
    
    timeframe: TradingTimeframe
    expected_api_calls_per_minute: int
    max_concurrent_strategies: int
    usage_pattern: ConnectionUsagePattern
    session_duration_hours: float
    priority_apis: List[str] = field(default_factory=list)


class RetailConnectionOptimizer:
    """
    Connection pool optimizer designed for retail quantitative trading.
    
    Optimizes connection pools based on actual trading patterns rather than
    theoretical maximum throughput, focusing on:
    - Scalping: Quick API responses for 1-5min decisions  
    - Intraday: Reliable connections for position management
    - Swing: Efficient resource usage for longer timeframes
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize retail connection optimizer.
        
        Args:
            settings: Configuration settings (uses global settings if None)
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(f"{__name__}.RetailOptimizer")
        
        # Performance tracking
        self.metrics_by_api: Dict[str, ConnectionMetrics] = {}
        self.session_profiles: Dict[str, TradingSessionProfile] = {}
        self.optimization_history: Deque[Dict] = deque(maxlen=1000)
        
        # Connection pool configuration
        self.base_pool_size = 20  # Conservative baseline for retail
        self.max_pool_size = 100  # Upper limit to prevent resource exhaustion
        self.min_pool_size = 5    # Minimum to maintain connectivity
        
        # Trading session state
        self.current_session_profile: Optional[TradingSessionProfile] = None
        self.session_start_time: Optional[datetime] = None
        self.active_strategies = 0
        
        # Optimization parameters tuned for retail trading
        self.target_response_time_ms = 150  # Target: 200ms → 150ms
        self.acceptable_response_time_ms = 200  # Still acceptable threshold
        self.optimization_interval_seconds = 60  # Optimize every minute
        
    def register_trading_session(self, profile: TradingSessionProfile) -> None:
        """Register a trading session profile for optimization.
        
        Args:
            profile: Trading session characteristics
        """
        self.current_session_profile = profile
        self.session_start_time = datetime.now(timezone.utc)
        
        session_id = f"{profile.timeframe}_{profile.usage_pattern}_{int(time.time())}"
        self.session_profiles[session_id] = profile
        
        self.logger.info(f"Registered trading session: {profile.timeframe} "
                        f"({profile.expected_api_calls_per_minute} calls/min, "
                        f"{profile.max_concurrent_strategies} strategies)")
    
    def get_optimal_connector_settings(self) -> Dict:
        """Get optimized aiohttp connector settings for current session.
        
        Returns:
            Dictionary with optimal connector configuration
        """
        if not self.current_session_profile:
            return self._get_default_connector_settings()
        
        profile = self.current_session_profile
        
        # Calculate optimal pool size based on trading profile
        base_connections = max(profile.max_concurrent_strategies * 2, self.min_pool_size)
        
        # Adjust based on timeframe and usage pattern
        if profile.timeframe == TradingTimeframe.SCALPING:
            # Scalping needs more connections for quick decisions
            pool_size = min(base_connections * 2, self.max_pool_size)
            per_host_limit = min(pool_size // 2, 30)
            keepalive_timeout = 30  # Shorter for scalping
        elif profile.timeframe == TradingTimeframe.INTRADAY:
            # Balanced approach for intraday
            pool_size = min(base_connections * 1.5, self.max_pool_size)
            per_host_limit = min(pool_size // 3, 25)
            keepalive_timeout = 60  # Standard keepalive
        else:  # SWING or PORTFOLIO
            # Conservative for longer timeframes
            pool_size = min(base_connections, self.max_pool_size)
            per_host_limit = min(pool_size // 4, 20)
            keepalive_timeout = 120  # Longer keepalive for swing trading
        
        # Adjust for usage patterns
        if profile.usage_pattern == ConnectionUsagePattern.BURST:
            pool_size = min(pool_size * 1.3, self.max_pool_size)
        elif profile.usage_pattern == ConnectionUsagePattern.INTERMITTENT:
            pool_size = max(pool_size * 0.7, self.min_pool_size)
        
        return {
            'limit': int(pool_size),
            'limit_per_host': int(per_host_limit),
            'ttl_dns_cache': 600,  # 10 minutes DNS cache
            'use_dns_cache': True,
            'keepalive_timeout': keepalive_timeout,
            'enable_cleanup_closed': True
        }
    
    def get_optimal_timeout_settings(self) -> aiohttp.ClientTimeout:
        """Get optimized timeout settings for current trading session.
        
        Returns:
            Optimized ClientTimeout configuration
        """
        if not self.current_session_profile:
            return self._get_default_timeout_settings()
        
        profile = self.current_session_profile
        
        # Adjust timeouts based on trading timeframe
        if profile.timeframe == TradingTimeframe.SCALPING:
            # Aggressive timeouts for scalping
            return aiohttp.ClientTimeout(
                total=15.0,     # Quick total timeout
                connect=5.0,    # Fast connection establishment
                sock_read=10.0  # Quick data reading
            )
        elif profile.timeframe == TradingTimeframe.INTRADAY:
            # Balanced timeouts for intraday
            return aiohttp.ClientTimeout(
                total=25.0,
                connect=8.0,
                sock_read=15.0
            )
        else:  # SWING or PORTFOLIO
            # More generous timeouts for longer timeframes
            return aiohttp.ClientTimeout(
                total=30.0,
                connect=10.0,
                sock_read=20.0
            )
    
    def record_api_performance(self, api_name: str, response_time_ms: float, 
                             success: bool = True, bytes_transferred: int = 0) -> None:
        """Record API performance metrics for optimization.
        
        Args:
            api_name: Name of the API endpoint
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            bytes_transferred: Number of bytes transferred
        """
        if api_name not in self.metrics_by_api:
            self.metrics_by_api[api_name] = ConnectionMetrics()
        
        metrics = self.metrics_by_api[api_name]
        
        if success:
            metrics.add_response_time(response_time_ms)
            metrics.total_bytes_transferred += bytes_transferred
        else:
            metrics.failed_requests += 1
        
        # Check if optimization is needed
        avg_response_time = metrics.get_average_response_time()
        if avg_response_time > self.acceptable_response_time_ms:
            self.logger.warning(f"API {api_name} response time degraded: {avg_response_time:.1f}ms")
            # Trigger optimization if we have enough data
            if len(metrics.response_times) >= 10:
                asyncio.create_task(self._optimize_for_api(api_name))
    
    async def _optimize_for_api(self, api_name: str) -> None:
        """Optimize connection settings for specific API performance.
        
        Args:
            api_name: Name of the API to optimize for
        """
        metrics = self.metrics_by_api.get(api_name)
        if not metrics:
            return
        
        avg_response_time = metrics.get_average_response_time()
        p95_response_time = metrics.get_p95_response_time()
        
        optimization_suggestion = {
            'timestamp': datetime.now(timezone.utc),
            'api_name': api_name,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'total_requests': metrics.successful_requests + metrics.failed_requests,
            'success_rate': metrics.successful_requests / (metrics.successful_requests + metrics.failed_requests),
            'suggested_action': self._get_optimization_suggestion(avg_response_time, p95_response_time)
        }
        
        self.optimization_history.append(optimization_suggestion)
        
        self.logger.info(f"Optimization analysis for {api_name}: "
                        f"avg={avg_response_time:.1f}ms, p95={p95_response_time:.1f}ms, "
                        f"suggestion={optimization_suggestion['suggested_action']}")
    
    def _get_optimization_suggestion(self, avg_time: float, p95_time: float) -> str:
        """Get optimization suggestion based on performance metrics.
        
        Args:
            avg_time: Average response time
            p95_time: 95th percentile response time
            
        Returns:
            Optimization suggestion string
        """
        if avg_time < self.target_response_time_ms:
            return "performance_excellent"
        elif avg_time < self.acceptable_response_time_ms:
            return "performance_acceptable"
        elif p95_time > self.acceptable_response_time_ms * 2:
            return "increase_connection_pool"
        elif avg_time > self.acceptable_response_time_ms * 1.5:
            return "optimize_timeout_settings"
        else:
            return "monitor_closely"
    
    def _get_default_connector_settings(self) -> Dict:
        """Get default connector settings for unknown session profile."""
        return {
            'limit': self.base_pool_size,
            'limit_per_host': 15,
            'ttl_dns_cache': 300,
            'use_dns_cache': True,
            'keepalive_timeout': 60,
            'enable_cleanup_closed': True
        }
    
    def _get_default_timeout_settings(self) -> aiohttp.ClientTimeout:
        """Get default timeout settings."""
        return aiohttp.ClientTimeout(
            total=30.0,
            connect=10.0,
            sock_read=20.0
        )
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary for all APIs.
        
        Returns:
            Performance summary dictionary
        """
        summary = {
            'session_profile': self.current_session_profile.timeframe.value if self.current_session_profile else None,
            'session_duration_minutes': 0,
            'apis': {},
            'overall': {
                'total_requests': 0,
                'total_failed_requests': 0,
                'average_response_time': 0,
                'best_performing_api': None,
                'worst_performing_api': None
            }
        }
        
        if self.session_start_time:
            duration = datetime.now(timezone.utc) - self.session_start_time
            summary['session_duration_minutes'] = duration.total_seconds() / 60
        
        all_response_times = []
        total_requests = 0
        total_failed = 0
        
        api_performance = {}
        
        for api_name, metrics in self.metrics_by_api.items():
            api_summary = {
                'average_response_time': metrics.get_average_response_time(),
                'p95_response_time': metrics.get_p95_response_time(),
                'total_requests': metrics.successful_requests + metrics.failed_requests,
                'success_rate': metrics.successful_requests / (metrics.successful_requests + metrics.failed_requests) if (metrics.successful_requests + metrics.failed_requests) > 0 else 0,
                'total_bytes': metrics.total_bytes_transferred
            }
            
            summary['apis'][api_name] = api_summary
            api_performance[api_name] = api_summary['average_response_time']
            
            all_response_times.extend(metrics.response_times)
            total_requests += api_summary['total_requests']
            total_failed += metrics.failed_requests
        
        # Overall summary
        summary['overall']['total_requests'] = total_requests
        summary['overall']['total_failed_requests'] = total_failed
        summary['overall']['average_response_time'] = statistics.mean(all_response_times) if all_response_times else 0
        
        if api_performance:
            summary['overall']['best_performing_api'] = min(api_performance, key=api_performance.get)
            summary['overall']['worst_performing_api'] = max(api_performance, key=api_performance.get)
        
        return summary


# Predefined trading session profiles for common retail patterns
SCALPING_SESSION = TradingSessionProfile(
    timeframe=TradingTimeframe.SCALPING,
    expected_api_calls_per_minute=20,  # 20 calls/minute for scalping
    max_concurrent_strategies=3,       # 2-3 scalping strategies max
    usage_pattern=ConnectionUsagePattern.BURST,
    session_duration_hours=2.0,        # Short intense sessions
    priority_apis=["fear_greed", "hyperliquid_ticker"]
)

INTRADAY_SESSION = TradingSessionProfile(
    timeframe=TradingTimeframe.INTRADAY,
    expected_api_calls_per_minute=10,  # 10 calls/minute for intraday
    max_concurrent_strategies=5,       # 3-5 intraday strategies
    usage_pattern=ConnectionUsagePattern.STEADY,
    session_duration_hours=6.0,        # Full trading session
    priority_apis=["fear_greed", "hyperliquid_orderbook", "monitoring"]
)

SWING_SESSION = TradingSessionProfile(
    timeframe=TradingTimeframe.SWING,
    expected_api_calls_per_minute=3,   # 3 calls/minute for swing
    max_concurrent_strategies=8,       # 5-8 swing strategies
    usage_pattern=ConnectionUsagePattern.INTERMITTENT,
    session_duration_hours=12.0,       # Extended monitoring
    priority_apis=["fear_greed", "monitoring", "portfolio_analysis"]
)


if __name__ == "__main__":
    """Test retail connection optimizer functionality."""
    
    async def test_retail_optimizer():
        """Test the retail connection optimizer."""
        
        print("=== Retail Connection Optimizer Test ===")
        
        # Initialize optimizer
        optimizer = RetailConnectionOptimizer()
        print("✅ Optimizer initialized")
        
        # Register scalping session
        optimizer.register_trading_session(SCALPING_SESSION)
        print("✅ Scalping session registered")
        
        # Get optimized settings
        connector_settings = optimizer.get_optimal_connector_settings()
        timeout_settings = optimizer.get_optimal_timeout_settings()
        
        print(f"📊 Optimized connector settings: {connector_settings}")
        print(f"⏱️  Optimized timeout settings: total={timeout_settings.total}s")
        
        # Simulate API performance recording
        for i in range(10):
            response_time = 80 + i * 5  # Simulate increasing response times
            optimizer.record_api_performance("fear_greed", response_time, True, 1024)
            await asyncio.sleep(0.01)
        
        # Get performance summary
        summary = optimizer.get_performance_summary()
        print(f"📈 Performance summary: {summary['overall']['average_response_time']:.1f}ms avg")
        
        print("✅ Retail Connection Optimizer test completed")
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_retail_optimizer())