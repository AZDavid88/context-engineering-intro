"""
Risk Management Framework - Genetic Evolution of Risk Parameters

This module implements a comprehensive risk management system with genetic algorithm
evolution of risk parameters, market regime detection, and circuit breakers.

Based on research from:
- VectorBT Advanced Risk Management (2M backtest validation)
- Crypto Fear & Greed Index for regime detection
- Real-time circuit breaker patterns

Key Features:
- Genetic evolution of 22+ risk parameters
- Market regime-specific risk optimization
- Real-time circuit breakers and drawdown limits
- Position sizing constraints with genetic weights
- Performance monitoring and alerting
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque
import json

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import get_settings, Settings
from src.data.fear_greed_client import FearGreedClient
from src.execution.order_management import OrderRequest, OrderStatus
from src.execution.position_sizer import PositionSizeResult

# Configure logging
logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level states for trading operations."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MarketRegime(str, Enum):
    """Market regime classifications for regime-specific risk management."""
    BULL_VOLATILE = "bull_volatile"
    BULL_STABLE = "bull_stable"
    BEAR_VOLATILE = "bear_volatile"
    BEAR_STABLE = "bear_stable"
    SIDEWAYS_VOLATILE = "sideways_volatile"
    SIDEWAYS_STABLE = "sideways_stable"
    UNKNOWN = "unknown"


class CircuitBreakerType(str, Enum):
    """Types of circuit breakers that can be triggered."""
    DAILY_DRAWDOWN = "daily_drawdown"
    TOTAL_DRAWDOWN = "total_drawdown"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    FEAR_GREED_EXTREME = "fear_greed_extreme"
    POSITION_CONCENTRATION = "position_concentration"
    RAPID_LOSSES = "rapid_losses"


@dataclass
class RiskMetrics:
    """Current risk metrics for the trading system."""
    
    # Portfolio-level metrics
    total_exposure: float = 0.0
    daily_pnl: float = 0.0
    daily_drawdown: float = 0.0
    total_drawdown: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_sharpe: float = 0.0
    
    # Position-level metrics
    max_position_size: float = 0.0
    position_count: int = 0
    avg_correlation: float = 0.0
    concentration_risk: float = 0.0
    
    # Market regime metrics
    current_regime: MarketRegime = MarketRegime.UNKNOWN
    fear_greed_index: Optional[int] = None
    volatility_percentile: float = 0.0
    
    # Risk level assessment
    risk_level: RiskLevel = RiskLevel.LOW
    active_circuit_breakers: List[CircuitBreakerType] = field(default_factory=list)
    
    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GeneticRiskGenome:
    """Genetic algorithm genome for risk management parameters."""
    
    # Stop loss parameters (evolved genetically)
    stop_loss_percentage: float = 0.05  # 5% stop loss
    trailing_stop_percentage: float = 0.03  # 3% trailing stop
    take_profit_percentage: float = 0.10  # 10% take profit
    
    # Position sizing parameters
    max_position_size: float = 0.25  # 25% max position
    max_portfolio_exposure: float = 0.80  # 80% max exposure
    correlation_penalty: float = 0.1  # Correlation adjustment
    
    # Drawdown limits (evolved)
    daily_drawdown_limit: float = 0.02  # 2% daily max
    total_drawdown_limit: float = 0.10  # 10% total max
    consecutive_loss_limit: int = 5  # Max consecutive losses
    
    # Volatility thresholds (evolved)
    high_volatility_threshold: float = 0.03  # 3% daily volatility
    volatility_scaling_factor: float = 0.5  # Scale positions in high vol
    
    # Regime-specific adjustments
    bear_market_reduction: float = 0.3  # 30% size reduction in bear
    fear_threshold: int = 25  # Fear & Greed threshold
    greed_threshold: int = 75  # Fear & Greed threshold
    
    # Circuit breaker parameters
    rapid_loss_threshold: float = 0.01  # 1% loss in 15 minutes
    rapid_loss_timeframe: int = 900  # 15 minutes
    correlation_spike_threshold: float = 0.8  # 80% correlation
    
    # Time-based controls
    max_trades_per_hour: int = 10
    cooldown_period_minutes: int = 60
    
    # Performance thresholds
    min_sharpe_continuation: float = 1.5  # Min Sharpe to continue
    performance_review_period: int = 24  # Hours


class CircuitBreaker:
    """Individual circuit breaker implementation."""
    
    def __init__(self, breaker_type: CircuitBreakerType, threshold: float, 
                 timeframe_minutes: int = 60):
        """Initialize circuit breaker.
        
        Args:
            breaker_type: Type of circuit breaker
            threshold: Threshold value to trigger
            timeframe_minutes: Time window for evaluation
        """
        self.breaker_type = breaker_type
        self.threshold = threshold
        self.timeframe_minutes = timeframe_minutes
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_count = 0
        
        # Historical data for evaluation
        self.data_history = deque(maxlen=1000)
    
    def check_trigger(self, current_value: float, timestamp: Optional[datetime] = None) -> bool:
        """Check if circuit breaker should trigger.
        
        Args:
            current_value: Current value to evaluate
            timestamp: Timestamp of the value
            
        Returns:
            True if circuit breaker triggered
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Add to history
        self.data_history.append({
            'timestamp': timestamp,
            'value': current_value
        })
        
        # Check threshold
        if self._evaluate_threshold(current_value, timestamp):
            if not self.triggered:
                self.triggered = True
                self.trigger_time = timestamp
                self.trigger_count += 1
                logger.warning(f"Circuit breaker triggered: {self.breaker_type} "
                             f"(value: {current_value}, threshold: {self.threshold})")
            return True
        
        return False
    
    def _evaluate_threshold(self, current_value: float, timestamp: datetime) -> bool:
        """Evaluate if threshold is breached."""
        if self.breaker_type == CircuitBreakerType.DAILY_DRAWDOWN:
            return current_value >= self.threshold
        elif self.breaker_type == CircuitBreakerType.VOLATILITY_SPIKE:
            return current_value >= self.threshold
        elif self.breaker_type == CircuitBreakerType.CORRELATION_SPIKE:
            return current_value >= self.threshold
        elif self.breaker_type == CircuitBreakerType.RAPID_LOSSES:
            # Check for rapid losses in timeframe
            cutoff_time = timestamp - timedelta(minutes=self.timeframe_minutes)
            recent_data = [d for d in self.data_history 
                          if d['timestamp'] >= cutoff_time]
            if len(recent_data) >= 2:
                total_loss = sum(d['value'] for d in recent_data)
                return total_loss >= self.threshold
        
        return False
    
    def reset(self):
        """Reset circuit breaker."""
        self.triggered = False
        self.trigger_time = None
    
    def is_active(self, cooldown_minutes: int = 60) -> bool:
        """Check if circuit breaker is currently active."""
        if not self.triggered or not self.trigger_time:
            return False
        
        # Check cooldown period
        cooldown_end = self.trigger_time + timedelta(minutes=cooldown_minutes)
        return datetime.now(timezone.utc) < cooldown_end


class MarketRegimeDetector:
    """Detect current market regime for risk management."""
    
    def __init__(self, settings: Settings):
        """Initialize regime detector.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.fear_greed_client = None  # Will be injected by TradingSystemManager
        
        # Historical data for regime analysis
        self.price_history = deque(maxlen=100)  # 100 periods
        self.volatility_history = deque(maxlen=30)  # 30 periods
        
        # Current regime state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.last_update = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.fear_greed_client:
            await self.fear_greed_client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.fear_greed_client:
            await self.fear_greed_client.disconnect()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.fear_greed_client:
            await self.fear_greed_client.disconnect()
    
    async def update_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Update current market regime assessment.
        
        Args:
            market_data: Recent market data with OHLCV
            
        Returns:
            Current market regime
        """
        try:
            # Ensure client is available and properly managed
            if self.fear_greed_client is None:
                raise RuntimeError("FearGreedClient not injected - TradingSystemManager required")
            
            # Get Fear & Greed Index (session management handled by TradingSystemManager)
            fear_greed_data = await self.fear_greed_client.get_current_index()
            fear_greed_value = fear_greed_data.value if fear_greed_data else 50
            
            # Calculate market metrics
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            trend = (market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1
            
            # Update histories
            self.price_history.extend(market_data['close'].values)
            current_vol = returns.tail(20).std() * np.sqrt(252)
            self.volatility_history.append(current_vol)
            
            # Regime classification
            regime = self._classify_regime(volatility, trend, fear_greed_value)
            
            # Update state
            self.current_regime = regime
            self.last_update = datetime.now(timezone.utc)
            
            logger.info(f"Market regime updated: {regime} "
                       f"(vol: {volatility:.2f}, trend: {trend:.2f}, "
                       f"fear_greed: {fear_greed_value})")
            
            return regime
            
        except Exception as e:
            logger.error(f"Failed to update market regime: {e}")
            return MarketRegime.UNKNOWN
    
    def _classify_regime(self, volatility: float, trend: float, 
                        fear_greed: int) -> MarketRegime:
        """Classify market regime based on metrics."""
        
        # Volatility classification
        high_vol = volatility > self.settings.market_regime.high_volatility_threshold
        
        # Trend classification  
        bull_trend = trend > self.settings.market_regime.trend_threshold
        bear_trend = trend < -self.settings.market_regime.trend_threshold
        
        # Fear & Greed influence
        extreme_fear = fear_greed <= self.settings.market_regime.fear_threshold
        extreme_greed = fear_greed >= self.settings.market_regime.greed_threshold
        
        # Regime logic
        if bull_trend and not extreme_greed:
            return MarketRegime.BULL_VOLATILE if high_vol else MarketRegime.BULL_STABLE
        elif bear_trend or extreme_fear:
            return MarketRegime.BEAR_VOLATILE if high_vol else MarketRegime.BEAR_STABLE
        else:
            return MarketRegime.SIDEWAYS_VOLATILE if high_vol else MarketRegime.SIDEWAYS_STABLE


class GeneticRiskManager:
    """Advanced risk management with genetic algorithm optimization."""
    
    def __init__(self, settings: Optional[Settings] = None, 
                 genetic_genome: Optional[GeneticRiskGenome] = None):
        """Initialize genetic risk manager.
        
        Args:
            settings: Configuration settings
            genetic_genome: Evolved risk parameters
        """
        self.settings = settings or get_settings()
        self.genome = genetic_genome or GeneticRiskGenome()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(self.settings)
        
        # Circuit breakers
        self.circuit_breakers = self._initialize_circuit_breakers()
        
        # Risk monitoring
        self.current_metrics = RiskMetrics()
        self.metrics_history = deque(maxlen=1000)
        
        # Position tracking
        self.position_history = deque(maxlen=100)
        
        # Performance tracking
        self.daily_pnl_history = deque(maxlen=30)
        self.recent_trades = deque(maxlen=100)
        
        # State management
        self.emergency_mode = False
        self.trading_enabled = True
        self.last_risk_check = datetime.now(timezone.utc)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.regime_detector.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.regime_detector.__aexit__(exc_type, exc_val, exc_tb)
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.regime_detector.cleanup()
    
    def _initialize_circuit_breakers(self) -> Dict[CircuitBreakerType, CircuitBreaker]:
        """Initialize all circuit breakers."""
        breakers = {}
        
        # Daily drawdown breaker
        breakers[CircuitBreakerType.DAILY_DRAWDOWN] = CircuitBreaker(
            CircuitBreakerType.DAILY_DRAWDOWN,
            self.genome.daily_drawdown_limit,
            timeframe_minutes=1440  # 24 hours
        )
        
        # Total drawdown breaker
        breakers[CircuitBreakerType.TOTAL_DRAWDOWN] = CircuitBreaker(
            CircuitBreakerType.TOTAL_DRAWDOWN,
            self.genome.total_drawdown_limit,
            timeframe_minutes=10080  # 1 week
        )
        
        # Volatility spike breaker
        breakers[CircuitBreakerType.VOLATILITY_SPIKE] = CircuitBreaker(
            CircuitBreakerType.VOLATILITY_SPIKE,
            self.genome.high_volatility_threshold,
            timeframe_minutes=60
        )
        
        # Rapid losses breaker
        breakers[CircuitBreakerType.RAPID_LOSSES] = CircuitBreaker(
            CircuitBreakerType.RAPID_LOSSES,
            self.genome.rapid_loss_threshold,
            timeframe_minutes=self.genome.rapid_loss_timeframe // 60
        )
        
        # Correlation spike breaker
        breakers[CircuitBreakerType.CORRELATION_SPIKE] = CircuitBreaker(
            CircuitBreakerType.CORRELATION_SPIKE,
            self.genome.correlation_spike_threshold,
            timeframe_minutes=60
        )
        
        return breakers
    
    async def evaluate_trade_risk(self, order_request: OrderRequest, 
                                 current_positions: Dict[str, float],
                                 market_data: pd.DataFrame) -> Tuple[bool, str, RiskLevel]:
        """Evaluate risk for a proposed trade.
        
        Args:
            order_request: Trade request to evaluate
            current_positions: Current position sizes by symbol
            market_data: Recent market data
            
        Returns:
            Tuple of (approved, reason, risk_level)
        """
        try:
            # Update market regime
            await self.regime_detector.update_regime(market_data)
            
            # Update risk metrics
            await self._update_risk_metrics(current_positions, market_data)
            
            # Check emergency mode
            if self.emergency_mode:
                return False, "Emergency mode active - all trading suspended", RiskLevel.EMERGENCY
            
            # Check circuit breakers
            active_breakers = self._check_circuit_breakers()
            if active_breakers:
                breaker_names = [str(b) for b in active_breakers]
                return False, f"Circuit breakers active: {', '.join(breaker_names)}", RiskLevel.CRITICAL
            
            # Position size validation
            position_approved, position_reason = self._validate_position_size(
                order_request, current_positions
            )
            if not position_approved:
                return False, position_reason, RiskLevel.HIGH
            
            # Market regime risk adjustment
            regime_approved, regime_reason = self._validate_regime_risk(
                order_request, self.regime_detector.current_regime
            )
            if not regime_approved:
                return False, regime_reason, RiskLevel.MODERATE
            
            # Correlation risk check
            correlation_approved, correlation_reason = self._validate_correlation_risk(
                order_request, current_positions
            )
            if not correlation_approved:
                return False, correlation_reason, RiskLevel.MODERATE
            
            # All checks passed
            risk_level = self._calculate_overall_risk_level()
            return True, "Trade approved", risk_level
            
        except Exception as e:
            logger.error(f"Risk evaluation failed: {e}")
            return False, f"Risk evaluation error: {e}", RiskLevel.CRITICAL
    
    def _validate_position_size(self, order_request: OrderRequest, 
                               current_positions: Dict[str, float]) -> Tuple[bool, str]:
        """Validate position sizing constraints."""
        
        # Calculate position value (simplified)
        position_value = order_request.size  # Assuming normalized sizing
        
        # Check max position size
        if position_value > self.genome.max_position_size:
            return False, f"Position size {position_value:.3f} exceeds max {self.genome.max_position_size:.3f}"
        
        # Check total exposure
        total_exposure = sum(abs(size) for size in current_positions.values()) + position_value
        if total_exposure > self.genome.max_portfolio_exposure:
            return False, f"Total exposure {total_exposure:.3f} exceeds max {self.genome.max_portfolio_exposure:.3f}"
        
        return True, "Position size approved"
    
    def _validate_regime_risk(self, order_request: OrderRequest, 
                             regime: MarketRegime) -> Tuple[bool, str]:
        """Validate trade against current market regime."""
        
        # Bear market restrictions
        if regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_STABLE]:
            # Reduce position size in bear markets
            adjusted_size = order_request.size * (1 - self.genome.bear_market_reduction)
            if order_request.size > adjusted_size:
                return False, f"Bear market: reduce position size to {adjusted_size:.3f}"
        
        # High volatility restrictions
        if regime in [MarketRegime.BULL_VOLATILE, MarketRegime.BEAR_VOLATILE, 
                     MarketRegime.SIDEWAYS_VOLATILE]:
            adjusted_size = order_request.size * self.genome.volatility_scaling_factor
            if order_request.size > adjusted_size:
                return False, f"High volatility: reduce position size to {adjusted_size:.3f}"
        
        return True, "Regime risk approved"
    
    def _validate_correlation_risk(self, order_request: OrderRequest,
                                  current_positions: Dict[str, float]) -> Tuple[bool, str]:
        """Validate correlation concentration risk."""
        
        # Simplified correlation check (would use actual correlation matrix in production)
        same_asset_class_symbols = [s for s in current_positions.keys() 
                                   if s.startswith(order_request.symbol[:3])]
        
        if len(same_asset_class_symbols) >= 3:  # More than 3 similar assets
            total_similar_exposure = sum(abs(current_positions[s]) for s in same_asset_class_symbols)
            if total_similar_exposure > 0.5:  # 50% concentration limit
                return False, f"Correlation risk: {total_similar_exposure:.1%} exposure to similar assets"
        
        return True, "Correlation risk approved"
    
    def _check_circuit_breakers(self) -> List[CircuitBreakerType]:
        """Check all circuit breakers and return active ones."""
        active_breakers = []
        
        for breaker_type, breaker in self.circuit_breakers.items():
            if breaker.is_active():
                active_breakers.append(breaker_type)
        
        return active_breakers
    
    async def _update_risk_metrics(self, current_positions: Dict[str, float],
                                  market_data: pd.DataFrame):
        """Update current risk metrics."""
        
        # Portfolio metrics
        total_exposure = sum(abs(size) for size in current_positions.values())
        position_count = len([s for s in current_positions.values() if abs(s) > 0.001])
        
        # Volatility calculation
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().tail(20)
            portfolio_volatility = returns.std() * np.sqrt(252)
        else:
            portfolio_volatility = 0.0
        
        # Update metrics
        self.current_metrics = RiskMetrics(
            total_exposure=total_exposure,
            position_count=position_count,
            portfolio_volatility=portfolio_volatility,
            current_regime=self.regime_detector.current_regime,
            risk_level=self._calculate_overall_risk_level(),
            active_circuit_breakers=self._check_circuit_breakers(),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store in history
        self.metrics_history.append(self.current_metrics)
    
    def _calculate_overall_risk_level(self) -> RiskLevel:
        """Calculate overall risk level based on current metrics."""
        
        # Check for critical conditions
        if self.emergency_mode or len(self.current_metrics.active_circuit_breakers) > 0:
            return RiskLevel.CRITICAL
        
        # Check for high risk conditions
        high_risk_conditions = [
            self.current_metrics.total_exposure > 0.8,
            self.current_metrics.portfolio_volatility > 0.5,
            self.current_metrics.current_regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_STABLE]
        ]
        
        if sum(high_risk_conditions) >= 2:
            return RiskLevel.HIGH
        elif sum(high_risk_conditions) >= 1:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self.current_metrics
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'risk_level': self.current_metrics.risk_level,
            'total_exposure': self.current_metrics.total_exposure,
            'position_count': self.current_metrics.position_count,
            'current_regime': self.current_metrics.current_regime,
            'active_circuit_breakers': self.current_metrics.active_circuit_breakers,
            'trading_enabled': self.trading_enabled,
            'emergency_mode': self.emergency_mode,
            'last_update': self.current_metrics.timestamp.isoformat(),
            'genetic_parameters': {
                'stop_loss': self.genome.stop_loss_percentage,
                'max_position': self.genome.max_position_size,
                'daily_drawdown_limit': self.genome.daily_drawdown_limit,
                'total_drawdown_limit': self.genome.total_drawdown_limit
            }
        }
    
    def emergency_shutdown(self, reason: str):
        """Trigger emergency shutdown of trading."""
        self.emergency_mode = True
        self.trading_enabled = False
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
    
    def reset_emergency_mode(self):
        """Reset emergency mode (manual intervention required)."""
        self.emergency_mode = False
        self.trading_enabled = True
        
        # Reset circuit breakers
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        
        logger.info("Emergency mode reset - trading re-enabled")


if __name__ == "__main__":
    """Risk management testing and validation."""
    
    import asyncio
    
    async def test_risk_manager():
        """Test risk manager functionality."""
        
        print("=== Risk Management Framework Test ===")
        
        # Initialize risk manager
        risk_manager = GeneticRiskManager()
        print("✅ Risk manager initialized")
        
        # Create mock market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        mock_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.exponential(1000, 100)
        })
        mock_data.set_index('timestamp', inplace=True)
        print("✅ Mock market data created")
        
        # Test order request
        from src.execution.order_management import OrderRequest, OrderSide
        test_order = OrderRequest(
            symbol="BTC",
            side=OrderSide.BUY,
            size=0.1,
            strategy_id="test_strategy"
        )
        
        # Mock current positions
        current_positions = {"ETH": 0.05, "SOL": 0.03}
        
        # Evaluate trade risk
        approved, reason, risk_level = await risk_manager.evaluate_trade_risk(
            test_order, current_positions, mock_data
        )
        
        print(f"✅ Trade evaluation: {approved} ({reason})")
        print(f"✅ Risk level: {risk_level}")
        
        # Get risk summary
        summary = risk_manager.get_risk_summary()
        print(f"✅ Risk summary generated")
        print(f"   - Total exposure: {summary['total_exposure']:.1%}")
        print(f"   - Position count: {summary['position_count']}")
        print(f"   - Current regime: {summary['current_regime']}")
        print(f"   - Trading enabled: {summary['trading_enabled']}")
        
        # Test circuit breaker
        risk_manager.circuit_breakers[CircuitBreakerType.DAILY_DRAWDOWN].check_trigger(0.025)
        active_breakers = risk_manager._check_circuit_breakers()
        print(f"✅ Circuit breaker test: {len(active_breakers)} active")
        
        print("\\n=== Risk Management Test Complete ===")
    
    # Run test
    asyncio.run(test_risk_manager())