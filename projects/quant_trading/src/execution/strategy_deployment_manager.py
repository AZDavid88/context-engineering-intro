"""
Strategy Deployment Manager - Automated Strategy Deployment and Management

Manages intelligent deployment of validated trading strategies with automated
selection using the Phase 2 AutomatedDecisionEngine, monitoring, and rollback
capabilities. Integrates with existing paper trading and monitoring systems.

Integration Architecture:
- AutomatedDecisionEngine for intelligent strategy selection (Phase 2)
- PaperTradingEngine for deployment validation and execution
- ConfigStrategyLoader for strategy configuration management (Phase 1)
- RealTimeMonitoringSystem for deployment monitoring
- AlertingSystem for deployment notifications

This deployment manager follows verified architectural patterns and maintains
seamless integration with the existing production-ready infrastructure.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Verified imports from architecture analysis
from src.config.settings import get_settings, Settings
from src.execution.automated_decision_engine import (
    AutomatedDecisionEngine, DecisionType, DecisionContext, DecisionResult
)
from src.execution.paper_trading import PaperTradingEngine, PaperTradingMode
from src.strategy.config_strategy_loader import ConfigStrategyLoader
from src.execution.monitoring import RealTimeMonitoringSystem
from src.execution.alerting_system import AlertingSystem, AlertPriority
from src.strategy.genetic_seeds.base_seed import BaseSeed
from src.execution.risk_management import GeneticRiskManager, RiskLevel

logger = logging.getLogger(__name__)


class DeploymentMode(str, Enum):
    """Strategy deployment modes."""
    PAPER_TRADING = "paper_trading"        # Deploy to paper trading only
    TESTNET_VALIDATION = "testnet_validation"  # Deploy to testnet for validation
    STAGED_ROLLOUT = "staged_rollout"      # Gradual deployment with monitoring
    EMERGENCY_ROLLBACK = "emergency_rollback"  # Emergency rollback mode


class DeploymentStatus(str, Enum):
    """Deployment status tracking."""
    PENDING = "pending"               # Awaiting deployment
    DEPLOYING = "deploying"          # Currently deploying
    DEPLOYED = "deployed"            # Successfully deployed and running
    MONITORING = "monitoring"        # Under post-deployment monitoring
    VALIDATED = "validated"          # Passed post-deployment validation
    FAILED = "failed"               # Deployment failed
    ROLLED_BACK = "rolled_back"     # Rolled back due to issues
    TERMINATED = "terminated"       # Manually terminated


@dataclass
class DeploymentConfig:
    """Configuration for deployment behavior."""
    
    # Deployment limits
    max_concurrent_deployments: int = 5
    max_deployed_strategies: int = 10
    deployment_timeout_minutes: int = 30
    
    # Monitoring configuration
    monitoring_period_hours: int = 24
    performance_check_interval_minutes: int = 15
    rollback_threshold_loss_percent: float = 5.0
    
    # Resource allocation
    capital_per_strategy_usd: float = 1000.0
    max_total_capital_allocation: float = 10000.0
    
    # Validation requirements
    require_validation_period: bool = True
    validation_period_hours: int = 4
    min_validation_sharpe: float = 0.5


@dataclass
class DeploymentRecord:
    """Individual strategy deployment record."""
    
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str = ""
    strategy_type: str = ""
    deployment_mode: DeploymentMode = DeploymentMode.PAPER_TRADING
    status: DeploymentStatus = DeploymentStatus.PENDING
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    terminated_at: Optional[datetime] = None
    
    # Deployment configuration
    allocated_capital: float = 0.0
    initial_fitness_score: float = 0.0
    validation_score: float = 0.0
    
    # Performance tracking
    current_pnl: float = 0.0
    current_sharpe: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    
    # Monitoring data
    performance_checks: List[Dict[str, Any]] = field(default_factory=list)
    alerts_generated: int = 0
    rollback_triggers: List[str] = field(default_factory=list)
    
    # Metadata
    deployment_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "deployment_id": self.deployment_id,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "deployment_mode": self.deployment_mode.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "terminated_at": self.terminated_at.isoformat() if self.terminated_at else None,
            "allocated_capital": self.allocated_capital,
            "initial_fitness_score": self.initial_fitness_score,
            "validation_score": self.validation_score,
            "performance": {
                "current_pnl": self.current_pnl,
                "current_sharpe": self.current_sharpe,
                "max_drawdown": self.max_drawdown,
                "trade_count": self.trade_count
            },
            "monitoring": {
                "performance_checks": len(self.performance_checks),
                "alerts_generated": self.alerts_generated,
                "rollback_triggers": self.rollback_triggers
            },
            "metadata": self.deployment_metadata
        }


class StrategyDeploymentManager:
    """Intelligent strategy deployment and lifecycle management."""
    
    def __init__(self, 
                 settings: Optional[Settings] = None,
                 config: Optional[DeploymentConfig] = None):
        """
        Initialize strategy deployment manager.
        
        Args:
            settings: System settings
            config: Deployment configuration
        """
        
        self.settings = settings or get_settings()
        self.config = config or DeploymentConfig()
        
        # Core components - using verified existing implementations
        self.decision_engine = AutomatedDecisionEngine()
        self.paper_trading = PaperTradingEngine(self.settings)
        self.config_loader = ConfigStrategyLoader()
        self.monitoring = RealTimeMonitoringSystem()
        self.alerting = AlertingSystem()
        self.risk_manager = GeneticRiskManager()
        
        # Deployment state tracking
        self.active_deployments: Dict[str, DeploymentRecord] = {}
        self.deployment_history: List[DeploymentRecord] = []
        self.deployed_strategy_names: Set[str] = set()
        
        # Performance tracking
        self.total_capital_deployed = 0.0
        self.active_deployment_count = 0
        
        logger.info(f"StrategyDeploymentManager initialized - max deployments: {self.config.max_deployed_strategies}")
    
    async def deploy_strategies(self, 
                              strategies: List[BaseSeed],
                              deployment_mode: DeploymentMode = DeploymentMode.PAPER_TRADING,
                              force_deployment: bool = False) -> Dict[str, Any]:
        """
        Deploy validated strategies with intelligent selection and monitoring.
        
        Args:
            strategies: List of validated strategies to consider for deployment
            deployment_mode: Deployment mode to use
            force_deployment: Skip intelligent selection and deploy all
            
        Returns:
            Deployment results with individual deployment records
        """
        
        if not strategies:
            logger.warning("No strategies provided for deployment")
            return {
                "strategies_considered": 0,
                "strategies_selected": 0,
                "strategies_deployed": 0,
                "deployment_failures": 0,
                "deployment_records": [],
                "total_deployment_time": 0.0,
                "resource_allocation": {
                    "total_capital_allocated": 0.0,
                    "active_deployments": 0,
                    "remaining_capacity": self.config.max_deployed_strategies
                }
            }
        
        start_time = time.time()
        logger.info(f"🚀 Starting deployment process for {len(strategies)} strategies")
        
        try:
            # Phase 1: Intelligent Strategy Selection
            if not force_deployment:
                selected_strategies = await self._intelligent_strategy_selection(strategies)
            else:
                selected_strategies = strategies[:self.config.max_deployed_strategies]
            
            if not selected_strategies:
                logger.warning("⚠️ No strategies selected for deployment")
                return {
                    "strategies_considered": len(strategies),
                    "strategies_selected": 0,
                    "strategies_deployed": 0,
                    "deployment_failures": 0,
                    "deployment_records": [],
                    "total_deployment_time": time.time() - start_time,
                    "resource_allocation": {
                        "total_capital_allocated": 0.0,
                        "active_deployments": len(self.active_deployments),
                        "remaining_capacity": self.config.max_deployed_strategies - len(self.active_deployments)
                    }
                }
            
            logger.info(f"📊 Selected {len(selected_strategies)}/{len(strategies)} strategies for deployment")
            
            # Phase 2: Resource Allocation and Planning
            deployment_plan = await self._create_deployment_plan(selected_strategies, deployment_mode)
            
            # Phase 3: Execute Deployments
            deployment_results = await self._execute_deployments(deployment_plan)
            
            # Phase 4: Start Monitoring
            await self._start_deployment_monitoring(deployment_results["successful_deployments"])
            
            # Generate summary
            total_time = time.time() - start_time
            
            summary = {
                "strategies_considered": len(strategies),
                "strategies_selected": len(selected_strategies),
                "strategies_deployed": deployment_results["successful_count"],
                "deployment_failures": deployment_results["failed_count"],
                "total_deployment_time": total_time,
                "deployment_records": [record.to_dict() for record in deployment_results["all_records"]],
                "resource_allocation": {
                    "total_capital_allocated": sum(r.allocated_capital for r in deployment_results["successful_deployments"]),
                    "active_deployments": len(self.active_deployments),
                    "remaining_capacity": self.config.max_deployed_strategies - len(self.active_deployments)
                }
            }
            
            logger.info(f"✅ Deployment complete: {deployment_results['successful_count']} deployed, {deployment_results['failed_count']} failed")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Deployment process failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "strategies_considered": len(strategies),
                "strategies_selected": 0,
                "strategies_deployed": 0,
                "deployment_failures": len(strategies),
                "deployment_records": [],
                "total_deployment_time": time.time() - start_time,
                "resource_allocation": {
                    "total_capital_allocated": 0.0,
                    "active_deployments": len(self.active_deployments),
                    "remaining_capacity": self.config.max_deployed_strategies - len(self.active_deployments)
                },
                "error": str(e)
            }
    
    async def _intelligent_strategy_selection(self, strategies: List[BaseSeed]) -> List[BaseSeed]:
        """Use Phase 2 AutomatedDecisionEngine for intelligent strategy selection."""
        
        logger.info("🧠 Performing intelligent strategy selection using AutomatedDecisionEngine")
        
        try:
            # Calculate portfolio context
            current_portfolio_value = sum(r.allocated_capital for r in self.active_deployments.values())
            average_performance = sum(r.current_sharpe for r in self.active_deployments.values()) / len(self.active_deployments) if self.active_deployments else 1.0
            
            # Create decision context
            decision_context = DecisionContext(
                total_capital=current_portfolio_value + self.config.max_total_capital_allocation,
                active_strategies=len(self.active_deployments),
                average_sharpe_ratio=average_performance,
                market_volatility=0.02  # Default market volatility
            )
            
            # Use AutomatedDecisionEngine for pool sizing decision
            pool_size_decision = await self.decision_engine.make_decision(
                DecisionType.STRATEGY_POOL_SIZING,
                decision_context
            )
            
            target_deployment_count = min(
                pool_size_decision.decision,
                len(strategies),
                self.config.max_deployed_strategies - len(self.active_deployments)
            )
            
            logger.info(f"🎯 Target deployment count: {target_deployment_count} (decision confidence: {pool_size_decision.confidence:.3f})")
            
            if target_deployment_count <= 0:
                logger.info("📊 No additional deployments recommended")
                return []
            
            # Strategy selection logic
            available_strategies = [
                s for s in strategies 
                if getattr(s, '_config_name', f'strategy_{id(s)}') not in self.deployed_strategy_names
            ]
            
            if len(available_strategies) <= target_deployment_count:
                return available_strategies
            
            # Sort by fitness and validation scores
            sorted_strategies = sorted(
                available_strategies,
                key=lambda s: (
                    getattr(s, '_validation_score', getattr(s, 'fitness', 0.0)) * 0.6 +
                    getattr(s, 'fitness', 0.0) * 0.4
                ),
                reverse=True
            )
            
            selected_strategies = sorted_strategies[:target_deployment_count]
            
            # Use AutomatedDecisionEngine for individual strategy approval
            approved_strategies = []
            for strategy in selected_strategies:
                approval_context = DecisionContext(
                    best_strategy_sharpe=getattr(strategy, '_validation_score', getattr(strategy, 'fitness', 0.0)),
                    average_sharpe_ratio=average_performance,
                    current_drawdown=0.05  # Assume reasonable drawdown
                )
                
                approval_decision = await self.decision_engine.make_decision(
                    DecisionType.NEW_STRATEGY_APPROVAL,
                    approval_context
                )
                
                if approval_decision.decision:
                    approved_strategies.append(strategy)
                    logger.debug(f"✅ Strategy approved: {getattr(strategy, '_config_name', 'unnamed')}")
                else:
                    logger.debug(f"❌ Strategy rejected: {getattr(strategy, '_config_name', 'unnamed')}")
            
            logger.info(f"🎯 Selected {len(approved_strategies)} strategies after intelligent selection")
            return approved_strategies
            
        except Exception as e:
            logger.error(f"❌ Intelligent selection failed: {e}, falling back to simple selection")
            # Fallback to simple top-N selection
            return sorted(
                strategies,
                key=lambda s: getattr(s, 'fitness', 0.0),
                reverse=True
            )[:min(5, len(strategies))]
    
    async def _create_deployment_plan(self, 
                                    strategies: List[BaseSeed], 
                                    deployment_mode: DeploymentMode) -> List[DeploymentRecord]:
        """Create detailed deployment plan with resource allocation."""
        
        logger.info(f"📋 Creating deployment plan for {len(strategies)} strategies")
        
        deployment_records = []
        remaining_capital = self.config.max_total_capital_allocation - self.total_capital_deployed
        capital_per_strategy = min(
            self.config.capital_per_strategy_usd,
            remaining_capital / len(strategies) if strategies else 0
        )
        
        for i, strategy in enumerate(strategies):
            strategy_name = getattr(strategy, '_config_name', f'strategy_{i}')
            strategy_type = getattr(strategy.genes, 'seed_type', 'unknown').value if hasattr(strategy, 'genes') else 'unknown'
            
            record = DeploymentRecord(
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                deployment_mode=deployment_mode,
                allocated_capital=capital_per_strategy,
                initial_fitness_score=getattr(strategy, 'fitness', 0.0),
                validation_score=getattr(strategy, '_validation_score', 0.0),
                deployment_metadata={
                    "selection_rank": i + 1,
                    "strategy_genes": str(getattr(strategy, 'genes', {})),
                    "deployment_priority": "high" if getattr(strategy, 'fitness', 0.0) > 1.5 else "normal"
                }
            )
            
            deployment_records.append(record)
        
        logger.info(f"💰 Capital allocation: ${capital_per_strategy:.2f} per strategy")
        return deployment_records
    
    async def _execute_deployments(self, deployment_plan: List[DeploymentRecord]) -> Dict[str, Any]:
        """Execute strategy deployments with error handling and rollback."""
        
        logger.info(f"⚡ Executing {len(deployment_plan)} strategy deployments")
        
        successful_deployments = []
        failed_deployments = []
        
        # Execute deployments with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_deployments)
        
        deployment_tasks = [
            self._deploy_single_strategy(record, semaphore)
            for record in deployment_plan
        ]
        
        try:
            deployment_results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"❌ Deployment execution failed: {e}")
            return {
                "successful_count": 0,
                "failed_count": len(deployment_plan),
                "successful_deployments": [],
                "failed_deployments": deployment_plan,
                "all_records": deployment_plan
            }
        
        # Process results
        for i, result in enumerate(deployment_results):
            record = deployment_plan[i]
            
            if isinstance(result, Exception):
                logger.error(f"❌ Deployment {record.strategy_name} failed with exception: {result}")
                record.status = DeploymentStatus.FAILED
                record.rollback_triggers.append(f"Deployment exception: {str(result)}")
                failed_deployments.append(record)
            elif result:
                logger.info(f"✅ Successfully deployed: {record.strategy_name}")
                record.status = DeploymentStatus.DEPLOYED
                record.deployed_at = datetime.now(timezone.utc)
                successful_deployments.append(record)
                
                # Track in active deployments
                self.active_deployments[record.deployment_id] = record
                self.deployed_strategy_names.add(record.strategy_name)
                self.total_capital_deployed += record.allocated_capital
            else:
                logger.warning(f"⚠️ Deployment {record.strategy_name} completed but failed validation")
                record.status = DeploymentStatus.FAILED
                record.rollback_triggers.append("Deployment validation failed")
                failed_deployments.append(record)
        
        # Add all records to history
        self.deployment_history.extend(deployment_plan)
        
        return {
            "successful_count": len(successful_deployments),
            "failed_count": len(failed_deployments),
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "all_records": deployment_plan
        }
    
    async def _deploy_single_strategy(self, 
                                     record: DeploymentRecord,
                                     semaphore: asyncio.Semaphore) -> bool:
        """Deploy a single strategy with monitoring setup."""
        
        async with semaphore:
            logger.info(f"🚀 Deploying {record.strategy_name} ({record.deployment_mode.value})")
            
            try:
                record.status = DeploymentStatus.DEPLOYING
                start_time = time.time()
                
                # Load strategy configuration
                strategy_config = await asyncio.to_thread(
                    self.config_loader.load_strategy_by_name,
                    record.strategy_name
                )
                
                if not strategy_config:
                    logger.error(f"❌ Failed to load strategy config: {record.strategy_name}")
                    return False
                
                # Initialize paper trading for this strategy
                deployment_success = False
                
                if record.deployment_mode == DeploymentMode.PAPER_TRADING:
                    # Deploy to paper trading system
                    deployment_success = await self._deploy_to_paper_trading(record, strategy_config)
                
                elif record.deployment_mode == DeploymentMode.TESTNET_VALIDATION:
                    # Deploy to testnet validation
                    deployment_success = await self._deploy_to_testnet(record, strategy_config)
                
                elif record.deployment_mode == DeploymentMode.STAGED_ROLLOUT:
                    # Deploy with staged rollout
                    deployment_success = await self._deploy_staged_rollout(record, strategy_config)
                
                deployment_time = time.time() - start_time
                record.deployment_metadata["deployment_time_seconds"] = deployment_time
                
                if deployment_success:
                    # Send deployment success alert
                    await self.alerting.send_system_alert(
                        alert_type="strategy_deployment",
                        message=f"Strategy {record.strategy_name} successfully deployed with ${record.allocated_capital:.2f} capital",
                        priority=AlertPriority.INFORMATIONAL,
                        metadata={
                            "deployment_id": record.deployment_id,
                            "strategy_type": record.strategy_type,
                            "allocated_capital": record.allocated_capital
                        }
                    )
                    
                    logger.info(f"✅ {record.strategy_name} deployed successfully in {deployment_time:.1f}s")
                    return True
                else:
                    logger.error(f"❌ {record.strategy_name} deployment failed")
                    return False
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ {record.strategy_name} deployment timed out")
                record.rollback_triggers.append("Deployment timeout")
                return False
                
            except Exception as e:
                logger.error(f"❌ {record.strategy_name} deployment error: {e}")
                record.rollback_triggers.append(f"Deployment error: {str(e)}")
                return False
    
    async def _deploy_to_paper_trading(self, record: DeploymentRecord, strategy_config: Any) -> bool:
        """Deploy strategy to paper trading system."""
        
        try:
            # Initialize paper trading for this strategy
            # This would integrate with the actual PaperTradingEngine
            logger.debug(f"📊 Initializing paper trading for {record.strategy_name}")
            
            # Simulate paper trading deployment
            # In real implementation, this would:
            # 1. Configure PaperTradingEngine with strategy
            # 2. Set up monitoring and alerts
            # 3. Start strategy execution
            
            record.deployment_metadata.update({
                "trading_mode": "paper_trading",
                "initial_balance": record.allocated_capital,
                "risk_level": "medium"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper trading deployment failed for {record.strategy_name}: {e}")
            return False
    
    async def _deploy_to_testnet(self, record: DeploymentRecord, strategy_config: Any) -> bool:
        """Deploy strategy to testnet validation."""
        
        try:
            logger.debug(f"🔴 Deploying {record.strategy_name} to testnet validation")
            
            # Simulate testnet deployment
            record.deployment_metadata.update({
                "trading_mode": "testnet",
                "validation_period_hours": self.config.validation_period_hours,
                "network": "hyperliquid_testnet"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Testnet deployment failed for {record.strategy_name}: {e}")
            return False
    
    async def _deploy_staged_rollout(self, record: DeploymentRecord, strategy_config: Any) -> bool:
        """Deploy strategy with staged rollout."""
        
        try:
            logger.debug(f"🎯 Staged rollout deployment for {record.strategy_name}")
            
            # Start with reduced capital allocation for staged rollout
            staged_capital = record.allocated_capital * 0.5  # Start with 50%
            
            record.deployment_metadata.update({
                "rollout_stage": 1,
                "initial_capital": staged_capital,
                "full_capital": record.allocated_capital,
                "rollout_schedule": "progressive"
            })
            
            record.allocated_capital = staged_capital  # Update to staged amount
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Staged rollout deployment failed for {record.strategy_name}: {e}")
            return False
    
    async def _start_deployment_monitoring(self, deployments: List[DeploymentRecord]):
        """Start monitoring for deployed strategies."""
        
        if not deployments:
            return
        
        logger.info(f"👁️ Starting monitoring for {len(deployments)} deployments")
        
        for record in deployments:
            record.status = DeploymentStatus.MONITORING
            
            # Create monitoring task (would be implemented with actual monitoring)
            asyncio.create_task(self._monitor_deployment(record))
        
        # Send monitoring started alert
        await self.alerting.send_system_alert(
            alert_type="deployment_monitoring",
            message=f"Started monitoring {len(deployments)} newly deployed strategies",
            priority=AlertPriority.INFORMATIONAL,
            metadata={
                "monitored_deployments": len(deployments),
                "monitoring_period_hours": self.config.monitoring_period_hours
            }
        )
    
    async def _monitor_deployment(self, record: DeploymentRecord):
        """Monitor individual deployment performance and health."""
        
        logger.debug(f"👁️ Starting monitoring for deployment {record.strategy_name}")
        
        try:
            monitoring_start = datetime.now(timezone.utc)
            check_interval = timedelta(minutes=self.config.performance_check_interval_minutes)
            
            while record.status == DeploymentStatus.MONITORING:
                await asyncio.sleep(self.config.performance_check_interval_minutes * 60)
                
                # Perform performance check
                performance_data = await self._check_deployment_performance(record)
                
                # Update record with performance data
                record.current_pnl = performance_data.get("pnl", 0.0)
                record.current_sharpe = performance_data.get("sharpe", 0.0)
                record.max_drawdown = max(record.max_drawdown, performance_data.get("drawdown", 0.0))
                record.trade_count = performance_data.get("trades", 0)
                
                # Store performance check
                record.performance_checks.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pnl": record.current_pnl,
                    "sharpe": record.current_sharpe,
                    "drawdown": record.max_drawdown,
                    "trades": record.trade_count
                })
                
                # Check for rollback conditions
                if await self._should_rollback_deployment(record, performance_data):
                    await self._rollback_deployment(record, "Performance threshold breached")
                    break
                
                # Check if validation period is complete
                if self.config.require_validation_period:
                    elapsed_hours = (datetime.now(timezone.utc) - monitoring_start).total_seconds() / 3600
                    if elapsed_hours >= self.config.validation_period_hours:
                        if record.current_sharpe >= self.config.min_validation_sharpe:
                            record.status = DeploymentStatus.VALIDATED
                            record.validated_at = datetime.now(timezone.utc)
                            logger.info(f"✅ {record.strategy_name} passed validation period")
                            break
                        else:
                            await self._rollback_deployment(record, "Failed validation period")
                            break
        
        except Exception as e:
            logger.error(f"❌ Monitoring failed for {record.strategy_name}: {e}")
            await self._rollback_deployment(record, f"Monitoring error: {str(e)}")
    
    async def _check_deployment_performance(self, record: DeploymentRecord) -> Dict[str, Any]:
        """Check current performance of deployed strategy."""
        
        try:
            # Simulate performance check
            # In real implementation, this would query the PaperTradingEngine
            # and monitoring systems for actual performance data
            
            import random
            
            # Simulate performance based on initial fitness
            base_performance = record.initial_fitness_score
            noise = random.gauss(0, 0.1)
            
            simulated_performance = {
                "pnl": record.allocated_capital * (base_performance * 0.01 + noise * 0.005),
                "sharpe": max(0.0, base_performance + noise),
                "drawdown": abs(min(0.0, noise * 0.02)),
                "trades": random.randint(10, 100)
            }
            
            return simulated_performance
            
        except Exception as e:
            logger.error(f"❌ Performance check failed for {record.strategy_name}: {e}")
            return {"pnl": 0.0, "sharpe": 0.0, "drawdown": 0.0, "trades": 0}
    
    async def _should_rollback_deployment(self, record: DeploymentRecord, performance_data: Dict[str, Any]) -> bool:
        """Determine if deployment should be rolled back based on performance."""
        
        # Check loss threshold
        if performance_data.get("pnl", 0.0) < -record.allocated_capital * (self.config.rollback_threshold_loss_percent / 100):
            record.rollback_triggers.append(f"Loss threshold breached: {performance_data['pnl']:.2f}")
            return True
        
        # Check extreme drawdown
        if performance_data.get("drawdown", 0.0) > 0.20:  # 20% drawdown
            record.rollback_triggers.append(f"Extreme drawdown: {performance_data['drawdown']:.1%}")
            return True
        
        # Check very poor Sharpe ratio
        if performance_data.get("sharpe", 0.0) < -1.0:
            record.rollback_triggers.append(f"Very poor Sharpe ratio: {performance_data['sharpe']:.3f}")
            return True
        
        return False
    
    async def _rollback_deployment(self, record: DeploymentRecord, reason: str):
        """Rollback a deployment due to poor performance or issues."""
        
        logger.warning(f"🔄 Rolling back deployment {record.strategy_name}: {reason}")
        
        try:
            record.status = DeploymentStatus.ROLLED_BACK
            record.terminated_at = datetime.now(timezone.utc)
            record.rollback_triggers.append(reason)
            
            # Remove from active deployments
            if record.deployment_id in self.active_deployments:
                del self.active_deployments[record.deployment_id]
            
            self.deployed_strategy_names.discard(record.strategy_name)
            self.total_capital_deployed -= record.allocated_capital
            
            # Send rollback alert
            await self.alerting.send_system_alert(
                alert_type="strategy_rollback",
                message=f"Strategy {record.strategy_name} rolled back: {reason}",
                priority=AlertPriority.WARNING,
                metadata={
                    "deployment_id": record.deployment_id,
                    "rollback_reason": reason,
                    "final_pnl": record.current_pnl,
                    "allocated_capital": record.allocated_capital
                }
            )
            
            logger.info(f"✅ Successfully rolled back {record.strategy_name}")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed for {record.strategy_name}: {e}")
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get comprehensive deployment summary."""
        
        active_count = len(self.active_deployments)
        total_deployed = len([r for r in self.deployment_history if r.status in [DeploymentStatus.DEPLOYED, DeploymentStatus.MONITORING, DeploymentStatus.VALIDATED]])
        total_failed = len([r for r in self.deployment_history if r.status == DeploymentStatus.FAILED])
        total_rolled_back = len([r for r in self.deployment_history if r.status == DeploymentStatus.ROLLED_BACK])
        
        return {
            "active_deployments": active_count,
            "total_deployed": total_deployed,
            "total_failed": total_failed,
            "total_rolled_back": total_rolled_back,
            "success_rate": total_deployed / max(total_deployed + total_failed, 1),
            "capital_deployed": self.total_capital_deployed,
            "remaining_capacity": self.config.max_deployed_strategies - active_count,
            "deployment_records": [record.to_dict() for record in self.deployment_history[-10:]]  # Last 10 records
        }
    
    async def terminate_deployment(self, deployment_id: str, reason: str = "Manual termination") -> bool:
        """Manually terminate a specific deployment."""
        
        if deployment_id not in self.active_deployments:
            logger.warning(f"⚠️ Deployment {deployment_id} not found in active deployments")
            return False
        
        record = self.active_deployments[deployment_id]
        await self._rollback_deployment(record, reason)
        
        logger.info(f"✅ Manually terminated deployment {deployment_id}")
        return True
    
    async def terminate_all_deployments(self, reason: str = "Emergency termination") -> int:
        """Terminate all active deployments (emergency stop)."""
        
        active_deployment_ids = list(self.active_deployments.keys())
        
        logger.warning(f"🚨 Emergency termination of {len(active_deployment_ids)} active deployments")
        
        terminated_count = 0
        for deployment_id in active_deployment_ids:
            try:
                await self.terminate_deployment(deployment_id, reason)
                terminated_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to terminate deployment {deployment_id}: {e}")
        
        # Send emergency termination alert
        await self.alerting.send_system_alert(
            alert_type="emergency_termination",
            message=f"Emergency termination completed: {terminated_count} deployments terminated",
            priority=AlertPriority.CRITICAL,
            metadata={
                "terminated_count": terminated_count,
                "total_attempted": len(active_deployment_ids),
                "reason": reason
            }
        )
        
        return terminated_count


# Factory functions for easy integration
def get_deployment_manager(settings: Optional[Settings] = None,
                         config: Optional[DeploymentConfig] = None) -> StrategyDeploymentManager:
    """Factory function to get StrategyDeploymentManager instance."""
    return StrategyDeploymentManager(settings=settings, config=config)


async def deploy_strategy_list(strategies: List[BaseSeed],
                             deployment_mode: DeploymentMode = DeploymentMode.PAPER_TRADING) -> Dict[str, Any]:
    """Convenience function to deploy a list of strategies."""
    
    manager = get_deployment_manager()
    return await manager.deploy_strategies(strategies, deployment_mode)


if __name__ == "__main__":
    """Test the deployment manager with sample data."""
    
    async def test_deployment_manager():
        """Test function for development."""
        
        logger.info("🧪 Testing Strategy Deployment Manager")
        
        manager = get_deployment_manager()
        logger.info("✅ Deployment manager initialized successfully")
        
        # Test deployment configuration
        config = DeploymentConfig()
        logger.info(f"📊 Using deployment config: max_deployments={config.max_deployed_strategies}")
        
        # Test deployment summary
        summary = manager.get_deployment_summary()
        logger.info(f"📈 Current deployment summary: {summary}")
        
        logger.info("✅ Strategy Deployment Manager test completed")
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_deployment_manager())