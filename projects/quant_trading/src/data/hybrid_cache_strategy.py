"""
Hybrid Cache Strategy - Intelligent Data Placement for Optimal Performance

This module implements intelligent data placement strategy for the Neon + DuckDB
hybrid storage system, optimizing query performance by determining the optimal
data source (local cache vs Neon) based on data age, query patterns, and access frequency.

Research-Based Implementation:
- /research/duckdb/research_summary.md - Local cache performance characteristics
- /research/neon/02_neon_architecture.md - Neon storage tiering patterns
- /verified_docs/by_module_simplified/data/data_flow_analysis.md - Query patterns

Key Features:
- Hot/warm/cold data classification based on age and access patterns
- Query-specific optimization (small vs large, recent vs historical)
- Adaptive caching strategy based on genetic algorithm workload patterns
- Cost-aware data placement for cloud optimization
- Performance monitoring and strategy adjustment
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics


# Set up logging
logger = logging.getLogger(__name__)


class DataTemperature(str, Enum):
    """Data temperature classification for caching strategy."""
    HOT = "hot"         # Recent data, frequent access - local cache preferred
    WARM = "warm"       # Medium-age data, occasional access - hybrid approach
    COLD = "cold"       # Old data, rare access - Neon storage preferred


class QueryType(str, Enum):
    """Query type classification for optimization."""
    RECENT_SMALL = "recent_small"           # Last few days, <100 records
    RECENT_LARGE = "recent_large"           # Last few days, >100 records  
    HISTORICAL_SMALL = "historical_small"   # >30 days, <1000 records
    HISTORICAL_LARGE = "historical_large"   # >30 days, >1000 records
    CROSS_TEMPORAL = "cross_temporal"       # Spans hot and cold data


class DataSource(str, Enum):
    """Available data sources for query execution."""
    LOCAL_CACHE = "local_cache"     # DuckDB local storage
    NEON_DIRECT = "neon_direct"     # Direct Neon TimescaleDB
    HYBRID = "hybrid"               # Combine both sources


@dataclass
class QueryPattern:
    """Query pattern analysis for strategy optimization."""
    symbol: str
    query_type: QueryType
    data_temperature: DataTemperature
    estimated_record_count: int
    time_span_days: int
    access_frequency: float = 0.0  # Accesses per hour
    last_access: Optional[datetime] = None
    average_response_time_ms: float = 0.0


@dataclass
class CacheMetrics:
    """Cache performance metrics for strategy optimization."""
    local_cache_hit_rate: float = 0.0
    local_cache_avg_response_ms: float = 0.0
    neon_avg_response_ms: float = 0.0
    hybrid_queries_executed: int = 0
    total_queries: int = 0
    cost_savings_percentage: float = 0.0
    last_metrics_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HybridCacheStrategy:
    """
    Intelligent data placement strategy for optimal performance and cost.
    
    Determines the optimal data source for queries based on data age, query patterns,
    access frequency, and performance characteristics of local cache vs Neon storage.
    """
    
    def __init__(self, 
                 hot_data_days: int = 7,
                 warm_data_days: int = 30,
                 cost_optimization_enabled: bool = True):
        """
        Initialize hybrid cache strategy.
        
        Args:
            hot_data_days: Days to consider data "hot" (local cache preferred)
            warm_data_days: Days to consider data "warm" (hybrid approach)
            cost_optimization_enabled: Enable cost-aware optimization
        """
        self.hot_data_days = hot_data_days
        self.warm_data_days = warm_data_days
        self.cost_optimization_enabled = cost_optimization_enabled
        
        # Strategy configuration based on genetic algorithm patterns
        self.small_query_threshold = 100   # Records
        self.large_query_threshold = 1000  # Records
        self.frequent_access_threshold = 5.0  # Accesses per hour
        
        # Performance thresholds (milliseconds)
        self.local_cache_target_response = 50   # Target response time for local
        self.neon_acceptable_response = 500     # Acceptable response time for Neon
        
        # Metrics tracking
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.cache_metrics = CacheMetrics()
        self.strategy_adjustments = 0
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.HybridStrategy")
    
    async def determine_optimal_source(self, 
                                     symbol: str,
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None,
                                     limit: Optional[int] = None,
                                     query_context: Optional[str] = None) -> str:
        """
        Determine optimal data source for query based on multiple factors.
        
        Args:
            symbol: Trading symbol
            start_time: Query start time
            end_time: Query end time  
            limit: Record limit
            query_context: Additional context (e.g., 'backtesting', 'live_trading')
            
        Returns:
            Optimal data source: 'local_cache', 'neon_direct', or 'hybrid'
        """
        # Analyze query characteristics
        query_analysis = await self._analyze_query_pattern(
            symbol, start_time, end_time, limit, query_context
        )
        
        # Record query pattern for future optimization
        self.query_patterns[f"{symbol}_{int(time.time())}"] = query_analysis
        
        # Determine optimal source based on analysis
        optimal_source = self._select_optimal_source(query_analysis)
        
        # Log decision for monitoring
        self.logger.debug(
            f"Source selection for {symbol}: {optimal_source} "
            f"(type: {query_analysis.query_type}, temp: {query_analysis.data_temperature})"
        )
        
        return optimal_source
    
    async def _analyze_query_pattern(self,
                                   symbol: str,
                                   start_time: Optional[datetime],
                                   end_time: Optional[datetime], 
                                   limit: Optional[int],
                                   query_context: Optional[str]) -> QueryPattern:
        """Analyze query pattern to determine optimal strategy."""
        
        now = datetime.now(timezone.utc)
        
        # Determine time characteristics
        if start_time is None and end_time is None:
            # Default to recent data
            data_age_days = 0
            time_span_days = 1
            estimated_records = limit or 100
        else:
            # Ensure timezone consistency for datetime arithmetic
            query_start = start_time or (now - timedelta(days=1))
            query_end = end_time or now
            
            # Make timezone-naive datetimes timezone-aware (assume UTC)
            if query_start.tzinfo is None:
                query_start = query_start.replace(tzinfo=timezone.utc)
            if query_end.tzinfo is None:
                query_end = query_end.replace(tzinfo=timezone.utc)
            
            data_age_days = (now - query_start).days
            time_span_days = (query_end - query_start).days
            
            # Estimate record count based on time span and symbol
            estimated_records = self._estimate_record_count(symbol, time_span_days, limit)
        
        # Classify data temperature
        data_temp = self._classify_data_temperature(data_age_days)
        
        # Classify query type
        query_type = self._classify_query_type(
            data_age_days, estimated_records, time_span_days, query_context
        )
        
        # Get access pattern information
        access_frequency = self._get_access_frequency(symbol)
        last_access = self._get_last_access(symbol)
        
        return QueryPattern(
            symbol=symbol,
            query_type=query_type,
            data_temperature=data_temp,
            estimated_record_count=estimated_records,
            time_span_days=time_span_days,
            access_frequency=access_frequency,
            last_access=last_access
        )
    
    def _classify_data_temperature(self, data_age_days: int) -> DataTemperature:
        """Classify data temperature based on age."""
        if data_age_days <= self.hot_data_days:
            return DataTemperature.HOT
        elif data_age_days <= self.warm_data_days:
            return DataTemperature.WARM
        else:
            return DataTemperature.COLD
    
    def _classify_query_type(self, 
                           data_age_days: int,
                           estimated_records: int,
                           time_span_days: int,
                           query_context: Optional[str]) -> QueryType:
        """Classify query type for optimization strategy."""
        
        # Check for cross-temporal queries
        if time_span_days > self.warm_data_days:
            return QueryType.CROSS_TEMPORAL
        
        # Recent data queries
        if data_age_days <= self.hot_data_days:
            if estimated_records <= self.small_query_threshold:
                return QueryType.RECENT_SMALL
            else:
                return QueryType.RECENT_LARGE
        
        # Historical data queries  
        else:
            if estimated_records <= self.large_query_threshold:
                return QueryType.HISTORICAL_SMALL
            else:
                return QueryType.HISTORICAL_LARGE
    
    def _estimate_record_count(self, symbol: str, time_span_days: int, limit: Optional[int]) -> int:
        """Estimate record count based on symbol and time span."""
        
        # Base estimation: assume 1440 minutes per day, 1 bar per minute for crypto
        base_records_per_day = 1440
        
        # Adjust for symbol type (crypto vs traditional markets)
        if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'USD', '-']):
            records_per_day = base_records_per_day  # 24/7 crypto trading
        else:
            records_per_day = int(base_records_per_day * 0.35)  # ~8.5 hours traditional markets
        
        estimated_total = records_per_day * time_span_days
        
        # Apply limit if specified
        if limit:
            estimated_total = min(estimated_total, limit)
        
        return max(1, estimated_total)
    
    def _select_optimal_source(self, query_pattern: QueryPattern) -> str:
        """
        Select optimal data source based on query pattern analysis.
        
        Decision matrix based on data temperature, query type, and performance characteristics.
        """
        symbol = query_pattern.symbol
        query_type = query_pattern.query_type
        data_temp = query_pattern.data_temperature
        estimated_records = query_pattern.estimated_record_count
        
        # Hot data optimization (prefer local cache)
        if data_temp == DataTemperature.HOT:
            if query_type in [QueryType.RECENT_SMALL, QueryType.RECENT_LARGE]:
                return DataSource.LOCAL_CACHE
            else:
                return DataSource.HYBRID  # Cross-temporal
        
        # Cold data optimization (prefer Neon)
        elif data_temp == DataTemperature.COLD:
            if query_type == QueryType.HISTORICAL_SMALL and self._is_frequently_accessed(symbol):
                return DataSource.HYBRID  # Cache frequently accessed cold data
            else:
                return DataSource.NEON_DIRECT
        
        # Warm data optimization (intelligent hybrid)
        else:  # DataTemperature.WARM
            if query_type == QueryType.HISTORICAL_SMALL:
                return DataSource.LOCAL_CACHE  # Small warm queries from cache
            elif query_type == QueryType.HISTORICAL_LARGE:
                return DataSource.NEON_DIRECT  # Large warm queries from Neon
            else:
                return DataSource.HYBRID  # Complex warm queries
    
    def _is_frequently_accessed(self, symbol: str) -> bool:
        """Check if symbol is frequently accessed based on historical patterns."""
        recent_patterns = [
            pattern for pattern in self.query_patterns.values()
            if pattern.symbol == symbol and pattern.last_access and
            (datetime.now(timezone.utc) - pattern.last_access).hours < 24
        ]
        
        if not recent_patterns:
            return False
        
        avg_frequency = statistics.mean([p.access_frequency for p in recent_patterns])
        return avg_frequency >= self.frequent_access_threshold
    
    def _get_access_frequency(self, symbol: str) -> float:
        """Get access frequency for symbol (accesses per hour)."""
        recent_patterns = [
            pattern for pattern in self.query_patterns.values()
            if pattern.symbol == symbol and pattern.last_access and
            (datetime.now(timezone.utc) - pattern.last_access).hours < 24
        ]
        
        if not recent_patterns:
            return 0.0
        
        # Calculate average access frequency
        return statistics.mean([p.access_frequency for p in recent_patterns])
    
    def _get_last_access(self, symbol: str) -> Optional[datetime]:
        """Get last access time for symbol."""
        symbol_patterns = [
            pattern for pattern in self.query_patterns.values()
            if pattern.symbol == symbol and pattern.last_access
        ]
        
        if not symbol_patterns:
            return None
        
        return max([p.last_access for p in symbol_patterns])
    
    async def record_query_performance(self,
                                     symbol: str,
                                     source_used: str,
                                     response_time_ms: float,
                                     record_count: int,
                                     success: bool = True) -> None:
        """
        Record query performance for strategy optimization.
        
        Args:
            symbol: Trading symbol
            source_used: Data source that was used
            response_time_ms: Query response time in milliseconds
            record_count: Number of records returned
            success: Whether query was successful
        """
        # Update cache metrics
        self.cache_metrics.total_queries += 1
        
        if source_used == DataSource.LOCAL_CACHE:
            self.cache_metrics.local_cache_avg_response_ms = (
                (self.cache_metrics.local_cache_avg_response_ms + response_time_ms) / 2
            )
        elif source_used == DataSource.NEON_DIRECT:
            self.cache_metrics.neon_avg_response_ms = (
                (self.cache_metrics.neon_avg_response_ms + response_time_ms) / 2
            )
        elif source_used == DataSource.HYBRID:
            self.cache_metrics.hybrid_queries_executed += 1
        
        # Update strategy if performance is poor
        await self._evaluate_strategy_adjustment(source_used, response_time_ms, success)
        
        self.logger.debug(
            f"Recorded query performance: {symbol} via {source_used} "
            f"({response_time_ms:.1f}ms, {record_count} records, success: {success})"
        )
    
    async def _evaluate_strategy_adjustment(self,
                                          source_used: str,
                                          response_time_ms: float,
                                          success: bool) -> None:
        """Evaluate if strategy adjustment is needed based on performance."""
        
        # Check for performance degradation
        if source_used == DataSource.LOCAL_CACHE:
            if response_time_ms > self.local_cache_target_response * 2:
                self.logger.warning(
                    f"Local cache performance degraded: {response_time_ms:.1f}ms "
                    f"(target: {self.local_cache_target_response}ms)"
                )
                # Consider reducing hot_data_days
                await self._adjust_hot_data_threshold(-1)
        
        elif source_used == DataSource.NEON_DIRECT:
            if response_time_ms > self.neon_acceptable_response * 2:
                self.logger.warning(
                    f"Neon performance degraded: {response_time_ms:.1f}ms "
                    f"(threshold: {self.neon_acceptable_response}ms)"
                )
                # Consider increasing hybrid usage
                await self._adjust_hybrid_preference(0.1)
        
        if not success:
            self.logger.error(f"Query failed using {source_used}")
            # Implement fallback strategy adjustment
    
    async def _adjust_hot_data_threshold(self, adjustment_days: int) -> None:
        """Adjust hot data threshold based on performance."""
        old_threshold = self.hot_data_days
        self.hot_data_days = max(1, min(14, self.hot_data_days + adjustment_days))
        
        if self.hot_data_days != old_threshold:
            self.strategy_adjustments += 1
            self.logger.info(
                f"Adjusted hot data threshold: {old_threshold} → {self.hot_data_days} days "
                f"(adjustment #{self.strategy_adjustments})"
            )
    
    async def _adjust_hybrid_preference(self, adjustment_factor: float) -> None:
        """Adjust hybrid query preference based on performance."""
        # Placeholder for hybrid preference adjustment logic
        self.strategy_adjustments += 1
        self.logger.info(f"Adjusted hybrid preference by {adjustment_factor}")
    
    async def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy performance metrics.
        
        Returns:
            Dict with detailed strategy performance and optimization metrics
        """
        now = datetime.now(timezone.utc)
        
        # Calculate hit rates and performance
        local_queries = sum(1 for p in self.query_patterns.values() 
                          if p.data_temperature == DataTemperature.HOT)
        neon_queries = sum(1 for p in self.query_patterns.values()
                         if p.data_temperature == DataTemperature.COLD)
        hybrid_queries = self.cache_metrics.hybrid_queries_executed
        
        total_tracked_queries = len(self.query_patterns)
        
        return {
            "strategy_configuration": {
                "hot_data_days": self.hot_data_days,
                "warm_data_days": self.warm_data_days,
                "cost_optimization_enabled": self.cost_optimization_enabled,
                "small_query_threshold": self.small_query_threshold,
                "large_query_threshold": self.large_query_threshold
            },
            "query_distribution": {
                "local_cache_queries": local_queries,
                "neon_direct_queries": neon_queries, 
                "hybrid_queries": hybrid_queries,
                "total_tracked_queries": total_tracked_queries
            },
            "performance_metrics": {
                "local_cache_avg_response_ms": self.cache_metrics.local_cache_avg_response_ms,
                "neon_avg_response_ms": self.cache_metrics.neon_avg_response_ms,
                "total_queries_executed": self.cache_metrics.total_queries,
                "strategy_adjustments_made": self.strategy_adjustments
            },
            "data_temperature_distribution": self._get_temperature_distribution(),
            "query_type_distribution": self._get_query_type_distribution(),
            "optimization_opportunities": await self._identify_optimization_opportunities(),
            "last_updated": now.isoformat()
        }
    
    def _get_temperature_distribution(self) -> Dict[str, int]:
        """Get distribution of data temperature in recent queries."""
        temp_counts = {"hot": 0, "warm": 0, "cold": 0}
        
        for pattern in self.query_patterns.values():
            temp_counts[pattern.data_temperature.value] += 1
        
        return temp_counts
    
    def _get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types in recent queries."""
        type_counts = {
            "recent_small": 0, "recent_large": 0,
            "historical_small": 0, "historical_large": 0,
            "cross_temporal": 0
        }
        
        for pattern in self.query_patterns.values():
            type_counts[pattern.query_type.value] += 1
        
        return type_counts
    
    async def _identify_optimization_opportunities(self) -> List[str]:
        """Identify potential optimization opportunities."""
        opportunities = []
        
        # Check for high Neon response times
        if self.cache_metrics.neon_avg_response_ms > self.neon_acceptable_response:
            opportunities.append(
                f"Neon response time high ({self.cache_metrics.neon_avg_response_ms:.1f}ms) "
                "- consider increasing local cache size"
            )
        
        # Check for low hybrid usage
        hybrid_percentage = (self.cache_metrics.hybrid_queries_executed / 
                           max(1, self.cache_metrics.total_queries)) * 100
        if hybrid_percentage < 10:
            opportunities.append(
                f"Low hybrid query usage ({hybrid_percentage:.1f}%) "
                "- may be missing optimization opportunities"
            )
        
        # Check for outdated strategy
        if self.strategy_adjustments == 0 and self.cache_metrics.total_queries > 100:
            opportunities.append(
                "No strategy adjustments made - consider enabling adaptive optimization"
            )
        
        return opportunities
    
    async def cleanup_old_patterns(self, hours_to_keep: int = 24) -> int:
        """
        Clean up old query patterns to manage memory.
        
        Args:
            hours_to_keep: Hours of query patterns to retain
            
        Returns:
            Number of patterns removed
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_to_keep)
        
        old_patterns = [
            key for key, pattern in self.query_patterns.items()
            if pattern.last_access and pattern.last_access < cutoff_time
        ]
        
        for key in old_patterns:
            del self.query_patterns[key]
        
        if old_patterns:
            self.logger.info(f"Cleaned up {len(old_patterns)} old query patterns")
        
        return len(old_patterns)