"""
Automated Decision Engine - Lone Trader Intelligence Framework

This module implements intelligent automated decision making for quantitative
trading operations, handling 95% of routine decisions automatically while
providing human-in-the-loop alerts for critical judgment calls.

Integration Points:
- ConfigStrategyLoader for strategy lifecycle management
- PerformanceAnalyzer for strategy performance metrics  
- GeneticRiskManager for portfolio risk assessment
- MonitoringEngine for alerting and notifications
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from pathlib import Path

from src.config.settings import get_settings, Settings
from src.execution.retail_connection_optimizer import TradingTimeframe, ConnectionUsagePattern
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.execution.risk_management import GeneticRiskManager, RiskLevel
from src.execution.monitoring_core import MonitoringEngine
from src.strategy.config_strategy_loader import ConfigStrategyLoader

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of automated decisions."""
    STRATEGY_POOL_SIZING = "strategy_pool_sizing"
    STRATEGY_RETIREMENT = "strategy_retirement" 
    NEW_STRATEGY_APPROVAL = "new_strategy_approval"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    RISK_ADJUSTMENT = "risk_adjustment"
    TRADING_SESSION_OPTIMIZATION = "trading_session_optimization"


class DecisionUrgency(str, Enum):
    """Urgency levels for human alerts."""
    LOW = "low"           # Informational, no action required
    MEDIUM = "medium"     # Review within 24 hours
    HIGH = "high"         # Review within 4 hours
    CRITICAL = "critical" # Immediate attention required


@dataclass  
class DecisionContext:
    """Context information for decision making."""
    
    # Market conditions
    market_volatility: float = 0.02
    average_volume_ratio: float = 1.0
    average_correlation: float = 0.5
    fear_greed_index: int = 50
    
    # Portfolio state
    total_capital: float = 10000.0
    active_strategies: int = 5
    daily_pnl_percentage: float = 0.0
    weekly_pnl_percentage: float = 0.0
    current_drawdown: float = 0.0
    
    # Strategy performance
    average_sharpe_ratio: float = 1.0
    best_strategy_sharpe: float = 1.5
    worst_strategy_sharpe: float = 0.5
    strategies_in_loss: int = 1
    
    # System health
    system_uptime_hours: float = 24.0
    api_error_rate: float = 0.01
    execution_latency_ms: float = 150.0


@dataclass
class DecisionResult:
    """Result of an automated decision."""
    
    decision_type: DecisionType
    decision: Any  # The actual decision made
    confidence: float = 1.0  # Confidence in decision (0.0-1.0)
    reasoning: str = ""  # Explanation of decision logic
    requires_human_review: bool = False
    urgency: DecisionUrgency = DecisionUrgency.LOW
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionRules:
    """Configurable decision rules loaded from JSON."""
    
    def __init__(self, rules_file: str = "config/decision_rules.json"):
        self.rules_file = Path(rules_file)
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load decision rules from JSON configuration."""
        
        if not self.rules_file.exists():
            # Create default rules file
            default_rules = {
                "strategy_pool_sizing": {
                    "base_strategies_per_1k_capital": 1,
                    "min_strategies": 3,
                    "max_strategies": 25,
                    "performance_adjustment": {
                        "high_performance_threshold": 2.0,  # Sharpe > 2.0
                        "high_performance_bonus": 5,
                        "low_performance_threshold": 0.5,   # Sharpe < 0.5
                        "low_performance_penalty": 3
                    }
                },
                
                "strategy_retirement": {
                    "negative_sharpe_days": 7,      # Negative Sharpe for 7 days
                    "max_drawdown_threshold": 0.15, # >15% drawdown
                    "max_drawdown_days": 3,         # For 3 days
                    "low_win_rate_threshold": 0.3,  # <30% win rate
                    "low_win_rate_days": 14         # For 14 days
                },
                
                "new_strategy_approval": {
                    "min_backtest_sharpe": 1.0,
                    "min_paper_trading_days": 3,
                    "min_paper_sharpe": 0.8,
                    "max_drawdown_threshold": 0.12
                },
                
                "emergency_shutdown": {
                    "daily_loss_threshold": 0.05,   # -5% daily loss
                    "weekly_loss_threshold": 0.15,  # -15% weekly loss
                    "max_drawdown_threshold": 0.20  # -20% portfolio drawdown
                },
                
                "risk_adjustment": {
                    "high_volatility_threshold": 0.05,  # >5% market volatility
                    "high_correlation_threshold": 0.8,  # >80% asset correlation
                    "low_volume_threshold": 0.7,        # <70% average volume
                    "position_size_adjustment": {
                        "high_risk_multiplier": 0.5,    # Reduce positions by 50%
                        "normal_risk_multiplier": 1.0,
                        "low_risk_multiplier": 1.2      # Increase positions by 20%
                    }
                }
            }
            
            self.rules_file.parent.mkdir(parents=True, exist_ok=True)
            self.rules_file.write_text(json.dumps(default_rules, indent=2))
            logger.info(f"Created default decision rules: {self.rules_file}")
            
            return default_rules
        
        try:
            return json.loads(self.rules_file.read_text())
        except Exception as e:
            logger.error(f"Failed to load decision rules: {e}")
            return {}
    
    def get_rule(self, decision_type: str, rule_name: str, default: Any = None) -> Any:
        """Get specific decision rule value."""
        return self.rules.get(decision_type, {}).get(rule_name, default)


class AutomatedDecisionEngine:
    """Intelligent decision automation for lone quantitative traders."""
    
    def __init__(self, 
                 config_loader: Optional[ConfigStrategyLoader] = None,
                 performance_analyzer: Optional[PerformanceAnalyzer] = None,
                 risk_manager: Optional[GeneticRiskManager] = None,
                 monitoring_engine: Optional[MonitoringEngine] = None,
                 settings: Optional[Settings] = None):
        
        self.settings = settings or get_settings()
        self.config_loader = config_loader or ConfigStrategyLoader()
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()
        self.risk_manager = risk_manager or GeneticRiskManager()
        self.monitoring_engine = monitoring_engine or MonitoringEngine()
        
        # Decision framework
        self.decision_rules = DecisionRules()
        self.decision_history: List[DecisionResult] = []
        
        # Alerting system integration
        from src.execution.alerting_system import AlertingSystem
        self.alerting_system = AlertingSystem()
        
        logger.info("AutomatedDecisionEngine initialized with verified architectural patterns")
    
    async def make_decision(self, 
                          decision_type: DecisionType,
                          context: DecisionContext) -> DecisionResult:
        """
        Make an automated decision based on context and rules.
        
        Args:
            decision_type: Type of decision to make
            context: Current market and portfolio context
            
        Returns:
            DecisionResult with decision and metadata
        """
        
        decision_method = {
            DecisionType.STRATEGY_POOL_SIZING: self._decide_strategy_pool_size,
            DecisionType.STRATEGY_RETIREMENT: self._decide_strategy_retirement,
            DecisionType.NEW_STRATEGY_APPROVAL: self._decide_strategy_approval,
            DecisionType.EMERGENCY_SHUTDOWN: self._decide_emergency_shutdown,
            DecisionType.RISK_ADJUSTMENT: self._decide_risk_adjustment,
            DecisionType.TRADING_SESSION_OPTIMIZATION: self._decide_trading_session
        }.get(decision_type)
        
        if not decision_method:
            return DecisionResult(
                decision_type=decision_type,
                decision=None,
                confidence=0.0,
                reasoning="Unknown decision type",
                requires_human_review=True,
                urgency=DecisionUrgency.MEDIUM
            )
        
        try:
            result = await decision_method(context)
            
            # Log decision
            self.decision_history.append(result)
            logger.info(f"Decision made: {decision_type.value} = {result.decision} ({result.confidence:.2f} confidence)")
            
            # Send alert if human review required
            if result.requires_human_review:
                await self.alerting_system.send_decision_alert(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Decision making failed for {decision_type.value}: {e}")
            
            return DecisionResult(
                decision_type=decision_type,
                decision=None,
                confidence=0.0,
                reasoning=f"Decision engine error: {e}",
                requires_human_review=True,
                urgency=DecisionUrgency.HIGH
            )
    
    async def _decide_strategy_pool_size(self, context: DecisionContext) -> DecisionResult:
        """Decide optimal strategy pool size based on capital and performance."""
        
        rules = self.decision_rules.rules.get("strategy_pool_sizing", {})
        
        # Base calculation: strategies per capital
        base_per_1k = rules.get("base_strategies_per_1k_capital", 1)
        base_size = max(int(context.total_capital / 1000 * base_per_1k), 
                       rules.get("min_strategies", 3))
        
        # Performance adjustments
        perf_rules = rules.get("performance_adjustment", {})
        if context.average_sharpe_ratio > perf_rules.get("high_performance_threshold", 2.0):
            bonus = perf_rules.get("high_performance_bonus", 5)
            adjusted_size = min(base_size + bonus, rules.get("max_strategies", 25))
            reasoning = f"High performance (Sharpe {context.average_sharpe_ratio:.2f}) - increasing pool size"
            
        elif context.average_sharpe_ratio < perf_rules.get("low_performance_threshold", 0.5):
            penalty = perf_rules.get("low_performance_penalty", 3)
            adjusted_size = max(base_size - penalty, rules.get("min_strategies", 3))
            reasoning = f"Low performance (Sharpe {context.average_sharpe_ratio:.2f}) - reducing pool size"
            
        else:
            adjusted_size = base_size
            reasoning = f"Normal performance - maintaining base pool size"
        
        # Check if decision requires human review
        requires_review = (
            adjusted_size > rules.get("max_strategies", 25) * 0.8 or  # Near maximum
            adjusted_size < rules.get("min_strategies", 3) * 1.5      # Near minimum
        )
        
        return DecisionResult(
            decision_type=DecisionType.STRATEGY_POOL_SIZING,
            decision=adjusted_size,
            confidence=0.9 if not requires_review else 0.7,
            reasoning=reasoning,
            requires_human_review=requires_review,
            urgency=DecisionUrgency.LOW if not requires_review else DecisionUrgency.MEDIUM,
            metadata={
                "base_size": base_size,
                "capital": context.total_capital,
                "current_sharpe": context.average_sharpe_ratio
            }
        )
    
    async def _decide_strategy_retirement(self, context: DecisionContext) -> DecisionResult:
        """Decide whether to retire underperforming strategies."""
        
        # Get strategy performance data from config loader
        strategy_summary = self.config_loader.get_strategy_summary()
        
        # Load retirement rules
        rules = self.decision_rules.rules.get("strategy_retirement", {})
        
        retirement_decisions = []
        
        # Check each active strategy (simplified logic for now)
        # In production, this would iterate through actual strategy performance data
        if context.worst_strategy_sharpe < 0.0:
            days_negative = 7  # This would come from performance tracking
            if days_negative >= rules.get("negative_sharpe_days", 7):
                retirement_decisions.append({
                    "strategy": "worst_performer",
                    "reason": f"Negative Sharpe ({context.worst_strategy_sharpe:.2f}) for {days_negative} days",
                    "confidence": 0.95
                })
        
        # High drawdown check
        if context.current_drawdown > rules.get("max_drawdown_threshold", 0.15):
            drawdown_days = 3  # This would come from performance tracking
            if drawdown_days >= rules.get("max_drawdown_days", 3):
                retirement_decisions.append({
                    "strategy": "high_drawdown",
                    "reason": f"Drawdown {context.current_drawdown:.1%} for {drawdown_days} days",
                    "confidence": 0.9
                })
        
        # Determine if human review is needed
        requires_review = len(retirement_decisions) > 2  # Multiple retirements
        
        return DecisionResult(
            decision_type=DecisionType.STRATEGY_RETIREMENT,
            decision=retirement_decisions,
            confidence=0.8 if retirement_decisions else 1.0,
            reasoning=f"Analyzed {strategy_summary.get('active_strategies', 0)} strategies, found {len(retirement_decisions)} candidates for retirement",
            requires_human_review=requires_review,
            urgency=DecisionUrgency.MEDIUM if retirement_decisions else DecisionUrgency.LOW,
            metadata={
                "total_active": strategy_summary.get('active_strategies', 0),
                "retirement_candidates": len(retirement_decisions)
            }
        )
    
    async def _decide_strategy_approval(self, context: DecisionContext) -> DecisionResult:
        """Decide whether to approve new strategies for deployment."""
        
        rules = self.decision_rules.rules.get("new_strategy_approval", {})
        
        # This would normally receive actual strategy performance data
        # For now, simulate with context values
        approval_criteria = {
            "backtest_sharpe": context.best_strategy_sharpe,
            "paper_trading_days": 5,  # Would come from actual data
            "paper_sharpe": context.average_sharpe_ratio * 0.9,  # Conservative estimate
            "max_drawdown": context.current_drawdown
        }
        
        approval_checks = []
        
        # Check each approval criterion
        if approval_criteria["backtest_sharpe"] >= rules.get("min_backtest_sharpe", 1.0):
            approval_checks.append("backtest_sharpe_passed")
        
        if approval_criteria["paper_trading_days"] >= rules.get("min_paper_trading_days", 3):
            approval_checks.append("paper_trading_duration_passed")
        
        if approval_criteria["paper_sharpe"] >= rules.get("min_paper_sharpe", 0.8):
            approval_checks.append("paper_performance_passed")
        
        if approval_criteria["max_drawdown"] <= rules.get("max_drawdown_threshold", 0.12):
            approval_checks.append("drawdown_check_passed")
        
        # Approval decision
        approval_rate = len(approval_checks) / 4
        should_approve = approval_rate >= 0.75  # Need 3/4 criteria
        
        requires_review = approval_rate < 0.9  # Review if not clearly passing
        
        return DecisionResult(
            decision_type=DecisionType.NEW_STRATEGY_APPROVAL,
            decision=should_approve,
            confidence=approval_rate,
            reasoning=f"Approval criteria met: {len(approval_checks)}/4 checks passed",
            requires_human_review=requires_review,
            urgency=DecisionUrgency.MEDIUM if requires_review else DecisionUrgency.LOW,
            metadata={
                "approval_checks": approval_checks,
                "approval_rate": approval_rate,
                "criteria": approval_criteria
            }
        )
    
    async def _decide_emergency_shutdown(self, context: DecisionContext) -> DecisionResult:
        """Check for emergency shutdown conditions."""
        
        rules = self.decision_rules.rules.get("emergency_shutdown", {})
        
        shutdown_reasons = []
        
        # Daily loss check
        if context.daily_pnl_percentage <= -rules.get("daily_loss_threshold", 0.05):
            shutdown_reasons.append(f"Daily loss {context.daily_pnl_percentage:.1%} exceeds threshold")
        
        # Weekly loss check  
        if context.weekly_pnl_percentage <= -rules.get("weekly_loss_threshold", 0.15):
            shutdown_reasons.append(f"Weekly loss {context.weekly_pnl_percentage:.1%} exceeds threshold")
        
        # Portfolio drawdown check
        if context.current_drawdown >= rules.get("max_drawdown_threshold", 0.20):
            shutdown_reasons.append(f"Portfolio drawdown {context.current_drawdown:.1%} exceeds threshold")
        
        should_shutdown = len(shutdown_reasons) > 0
        
        return DecisionResult(
            decision_type=DecisionType.EMERGENCY_SHUTDOWN,
            decision=should_shutdown,
            confidence=1.0,
            reasoning="; ".join(shutdown_reasons) if shutdown_reasons else "No emergency conditions detected",
            requires_human_review=should_shutdown,  # Always alert human for shutdown
            urgency=DecisionUrgency.CRITICAL if should_shutdown else DecisionUrgency.LOW,
            metadata={
                "daily_pnl": context.daily_pnl_percentage,
                "weekly_pnl": context.weekly_pnl_percentage,
                "current_drawdown": context.current_drawdown,
                "shutdown_reasons": shutdown_reasons
            }
        )
    
    async def _decide_risk_adjustment(self, context: DecisionContext) -> DecisionResult:
        """Decide risk adjustment based on market conditions."""
        
        rules = self.decision_rules.rules.get("risk_adjustment", {})
        multipliers = rules.get("position_size_adjustment", {})
        
        # Analyze risk factors
        risk_factors = []
        
        if context.market_volatility > rules.get("high_volatility_threshold", 0.05):
            risk_factors.append("high_volatility")
        
        if context.average_correlation > rules.get("high_correlation_threshold", 0.8):
            risk_factors.append("high_correlation")
        
        if context.average_volume_ratio < rules.get("low_volume_threshold", 0.7):
            risk_factors.append("low_volume")
        
        # Determine adjustment
        if len(risk_factors) >= 2:
            adjustment = multipliers.get("high_risk_multiplier", 0.5)
            risk_level = "HIGH"
        elif len(risk_factors) == 1:
            adjustment = multipliers.get("normal_risk_multiplier", 1.0)
            risk_level = "MEDIUM"
        else:
            adjustment = multipliers.get("low_risk_multiplier", 1.2)
            risk_level = "LOW"
        
        return DecisionResult(
            decision_type=DecisionType.RISK_ADJUSTMENT,
            decision={
                "position_size_multiplier": adjustment,
                "risk_level": risk_level,
                "risk_factors": risk_factors
            },
            confidence=0.85,
            reasoning=f"Risk level: {risk_level} based on {len(risk_factors)} risk factors",
            requires_human_review=len(risk_factors) >= 2,  # Review high risk situations
            urgency=DecisionUrgency.MEDIUM if len(risk_factors) >= 2 else DecisionUrgency.LOW,
            metadata={
                "volatility": context.market_volatility,
                "correlation": context.average_correlation,
                "volume_ratio": context.average_volume_ratio,
                "risk_factors": risk_factors
            }
        )
    
    async def _decide_trading_session(self, context: DecisionContext) -> DecisionResult:
        """Decide trading session optimization (asset-agnostic approach)."""
        
        # For GA-evolved strategies monitoring 200+ assets, traditional session profiles are less relevant
        # Focus on risk and position sizing adjustments based on market conditions
        
        if context.market_volatility > 0.05:
            session_adjustment = {
                "position_size_multiplier": 0.8,  # Reduce position sizes in high volatility
                "max_concurrent_positions": 15,   # Limit concurrent positions
                "risk_adjustment_factor": 0.7     # Conservative risk
            }
            reasoning = "High volatility detected - reducing position sizes and risk"
            
        elif context.average_correlation > 0.8:
            session_adjustment = {
                "position_size_multiplier": 0.9,
                "max_concurrent_positions": 10,   # Fewer positions when assets are highly correlated
                "risk_adjustment_factor": 0.8
            }
            reasoning = "High asset correlation - reducing position diversity"
            
        else:
            session_adjustment = {
                "position_size_multiplier": 1.0,  # Normal position sizing
                "max_concurrent_positions": 20,   # Standard concurrent positions
                "risk_adjustment_factor": 1.0     # Normal risk
            }
            reasoning = "Normal market conditions - standard position sizing"
        
        return DecisionResult(
            decision_type=DecisionType.TRADING_SESSION_OPTIMIZATION,
            decision=session_adjustment,
            confidence=0.9,
            reasoning=reasoning,
            requires_human_review=False,  # This is routine adjustment
            urgency=DecisionUrgency.LOW,
            metadata={
                "market_volatility": context.market_volatility,
                "asset_correlation": context.average_correlation
            }
        )
    
    async def send_system_alert(self, alert_type: str, message: str, priority: str = "warning") -> bool:
        """Send system-level alert through integrated alerting system."""
        
        try:
            from src.execution.alerting_system import AlertPriority
            
            priority_map = {
                "informational": AlertPriority.INFORMATIONAL,
                "warning": AlertPriority.WARNING,
                "urgent": AlertPriority.URGENT,
                "critical": AlertPriority.CRITICAL
            }
            
            alert_priority = priority_map.get(priority.lower(), AlertPriority.WARNING)
            return await self.alerting_system.send_system_alert(alert_type, message, alert_priority)
            
        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
            return False
    
    def get_decision_history(self, 
                           decision_type: Optional[DecisionType] = None,
                           hours_back: int = 24) -> List[DecisionResult]:
        """Get recent decision history."""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        filtered_decisions = [
            decision for decision in self.decision_history
            if decision.timestamp >= cutoff_time
        ]
        
        if decision_type:
            filtered_decisions = [
                decision for decision in filtered_decisions
                if decision.decision_type == decision_type
            ]
        
        return filtered_decisions
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision engine statistics."""
        
        if not self.decision_history:
            return {"total_decisions": 0}
        
        # Calculate statistics
        total_decisions = len(self.decision_history)
        automated_decisions = len([d for d in self.decision_history if not d.requires_human_review])
        average_confidence = statistics.mean([d.confidence for d in self.decision_history])
        
        decision_types = {}
        for decision in self.decision_history:
            decision_type = decision.decision_type.value
            decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "automated_decisions": automated_decisions,
            "automation_rate": automated_decisions / total_decisions if total_decisions > 0 else 0,
            "average_confidence": average_confidence,
            "decision_type_breakdown": decision_types,
            "recent_24h": len(self.get_decision_history(hours_back=24))
        }


# Factory function for easy integration
async def get_decision_engine() -> AutomatedDecisionEngine:
    """Factory function to get AutomatedDecisionEngine instance."""
    return AutomatedDecisionEngine()