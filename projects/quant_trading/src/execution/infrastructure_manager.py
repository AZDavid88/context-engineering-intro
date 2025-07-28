"""
Infrastructure Manager - Integration Bridge for Trading System

This module provides the integration bridge between the new infrastructure
abstractions and the existing TradingSystemManager and GeneticStrategyPool,
enabling seamless deployment and management of genetic algorithm workloads.

Integration Pattern:
- Integrates with existing TradingSystemManager async coordination
- Extends GeneticStrategyPool with infrastructure-aware deployment
- Maintains backward compatibility with existing system architecture
- Provides infrastructure observability to existing monitoring systems

Key Features:
- Seamless integration with existing async session management
- Dynamic infrastructure scaling based on genetic algorithm workloads
- Cost-aware deployment with budget management
- Health monitoring integration with existing RealTimeMonitoringSystem
"""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Infrastructure abstractions
from infrastructure.core.deployment_interface import (
    DeploymentManager, GeneticPoolConfig, DeploymentResult, PlatformType, CostLimits
)
from infrastructure.core.cluster_manager import (
    ClusterManager, ClusterConfig, ClusterInfo, WorkloadType
)
from infrastructure.core.monitoring_interface import (
    MonitoringManager, MetricType, Alert, AlertSeverity
)
from infrastructure.core.config_manager import (
    ConfigurationManager, FileBasedConfigurationManager, Environment
)

# Existing system integration
from src.config.settings import get_settings
from src.execution.monitoring import RealTimeMonitoringSystem
from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionMode

# Set up logging
logger = logging.getLogger(__name__)


class InfrastructureState(str, Enum):
    """Infrastructure deployment states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEPLOYING = "deploying"
    SCALING = "scaling"
    MONITORING = "monitoring"
    TERMINATING = "terminating"
    ERROR = "error"


@dataclass
class InfrastructureStatus:
    """Current infrastructure status"""
    state: InfrastructureState
    platform: Optional[PlatformType] = None
    environment: Optional[Environment] = None
    deployment_id: Optional[str] = None
    cluster_id: Optional[str] = None
    
    # Resource information
    active_nodes: int = 0
    total_cost_per_hour: float = 0.0
    
    # Integration status
    genetic_pool_connected: bool = False
    monitoring_active: bool = False
    
    # Health information
    last_health_check: Optional[datetime] = None
    health_score: float = 0.0
    
    # Metadata
    initialized_at: Optional[datetime] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)


class InfrastructureManager:
    """
    Infrastructure management integration for existing trading system.
    
    This class provides the bridge between infrastructure abstractions and
    the existing TradingSystemManager, enabling seamless infrastructure
    deployment and management for genetic algorithm workloads.
    """
    
    def __init__(self, 
                 platform: PlatformType = PlatformType.ANYSCALE,
                 environment: Environment = Environment.DEVELOPMENT,
                 config_path: Optional[str] = None):
        """
        Initialize infrastructure manager.
        
        Args:
            platform: Target deployment platform
            environment: Deployment environment
            config_path: Path to configuration files
        """
        self.platform = platform
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers (will be set by platform-specific implementations)
        self.deployment_manager: Optional[DeploymentManager] = None
        self.cluster_manager: Optional[ClusterManager] = None
        self.monitoring_manager: Optional[MonitoringManager] = None
        self.config_manager = FileBasedConfigurationManager(config_path)
        
        # Infrastructure state
        self.status = InfrastructureStatus(state=InfrastructureState.UNINITIALIZED)
        
        # Integration with existing system
        self.trading_monitor: Optional[RealTimeMonitoringSystem] = None
        self.genetic_pool: Optional[GeneticStrategyPool] = None
        
        # Active deployments tracking
        self._active_deployments: Dict[str, DeploymentResult] = {}
        self._active_clusters: Dict[str, ClusterInfo] = {}
        
        self.logger.info(f"Infrastructure manager initialized for {platform}/{environment}")
    
    async def initialize(self, trading_monitor: Optional[RealTimeMonitoringSystem] = None):
        """
        Initialize infrastructure manager with existing system integration.
        
        Args:
            trading_monitor: Existing monitoring system for integration
        """
        try:
            self.status.state = InfrastructureState.INITIALIZING
            self.status.initialized_at = datetime.now(timezone.utc)
            
            # Store reference to existing monitoring system
            self.trading_monitor = trading_monitor
            
            # Load configuration
            infrastructure_config = await self.config_manager.load_configuration(
                self.environment, self.platform
            )
            
            # Initialize platform-specific managers (placeholder - will be implemented by concrete platforms)
            if self.platform == PlatformType.ANYSCALE:
                from infrastructure.platforms.anyscale.anyscale_deployer import AnyscaleDeploymentManager
                from infrastructure.platforms.anyscale.anyscale_cluster_manager import AnyscaleClusterManager
                from infrastructure.platforms.anyscale.anyscale_monitoring import AnyscaleMonitoringManager
                
                # These would be implemented in the next phase
                # self.deployment_manager = AnyscaleDeploymentManager(...)
                # self.cluster_manager = AnyscaleClusterManager(...)
                # self.monitoring_manager = AnyscaleMonitoringManager(...)
            
            # For now, use mock implementations for testing
            self.logger.warning("Using mock infrastructure managers - implement platform-specific versions")
            
            # Integrate with existing monitoring system
            if self.trading_monitor:
                await self._integrate_monitoring()
            
            self.status.state = InfrastructureState.READY
            self.logger.info("Infrastructure manager initialized successfully")
            
        except Exception as e:
            self.status.state = InfrastructureState.ERROR
            self.logger.error(f"Failed to initialize infrastructure manager: {e}")
            raise
    
    async def deploy_genetic_infrastructure(self, 
                                          genetic_pool: GeneticStrategyPool,
                                          population_size: int,
                                          max_generations: int) -> DeploymentResult:
        """
        Deploy infrastructure optimized for genetic algorithm workload.
        
        This method integrates with the existing GeneticStrategyPool to deploy
        appropriate infrastructure based on the genetic algorithm parameters.
        
        Args:
            genetic_pool: Existing genetic strategy pool instance
            population_size: Size of genetic population
            max_generations: Maximum number of generations
            
        Returns:
            DeploymentResult with deployment information
        """
        try:
            self.status.state = InfrastructureState.DEPLOYING
            self.genetic_pool = genetic_pool
            
            # Calculate infrastructure requirements based on genetic parameters
            infrastructure_requirements = self._calculate_infrastructure_requirements(
                population_size, max_generations
            )
            
            # Create deployment configuration
            config = GeneticPoolConfig(
                population_size=population_size,
                max_generations=max_generations,
                evaluation_timeout=300,  # 5 minutes
                platform=self.platform,
                platform_config=infrastructure_requirements,
                cost_limits=CostLimits(
                    max_hourly_cost=50.0,
                    max_total_cost=500.0
                ),
                use_spot_instances=True,
                health_check_interval=30,
                metrics_collection=True
            )
            
            # Validate configuration
            config.validate()
            
            # Deploy infrastructure (mock implementation for now)
            if self.deployment_manager:
                deployment_result = await self.deployment_manager.deploy_genetic_pool(config)
            else:
                # Mock deployment result for testing
                deployment_result = DeploymentResult(
                    deployment_id=f"mock_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    cluster_id=f"mock_cluster_{population_size}",
                    platform=self.platform,
                    status="running",
                    endpoint_url="ray://mock-cluster:10001",
                    estimated_cost_per_hour=25.0,
                    total_nodes=min(population_size // 20, 10),
                    optimization_applied=True,
                    genetic_pool_endpoint="ray://mock-cluster:10001"
                )
                
                self.logger.warning("Using mock deployment - implement actual deployment manager")
            
            # Store deployment information
            self._active_deployments[deployment_result.deployment_id] = deployment_result
            self.status.deployment_id = deployment_result.deployment_id
            self.status.cluster_id = deployment_result.cluster_id
            self.status.active_nodes = deployment_result.total_nodes
            self.status.total_cost_per_hour = deployment_result.estimated_cost_per_hour
            
            # Connect genetic pool to infrastructure
            await self._connect_genetic_pool(genetic_pool, deployment_result)
            
            # Start monitoring
            await self._start_infrastructure_monitoring(deployment_result)
            
            self.status.state = InfrastructureState.READY
            self.status.genetic_pool_connected = True
            self.status.monitoring_active = True
            
            self.logger.info(
                f"Deployed genetic infrastructure: {deployment_result.total_nodes} nodes, "
                f"${deployment_result.estimated_cost_per_hour:.2f}/hour"
            )
            
            return deployment_result
            
        except Exception as e:
            self.status.state = InfrastructureState.ERROR
            self.logger.error(f"Failed to deploy genetic infrastructure: {e}")
            raise
    
    async def scale_for_workload(self, 
                               population_size: int,
                               complexity: str = "medium") -> bool:
        """
        Scale infrastructure based on genetic algorithm workload.
        
        Args:
            population_size: New population size
            complexity: Workload complexity ("simple", "medium", "complex")
            
        Returns:
            True if scaling successful
        """
        try:
            if not self.status.cluster_id or not self.cluster_manager:
                self.logger.warning("No active cluster to scale")
                return False
            
            self.status.state = InfrastructureState.SCALING
            
            # Optimize cluster for new workload
            success = await self.cluster_manager.optimize_for_genetic_workload(
                self.status.cluster_id,
                population_size,
                complexity
            )
            
            if success:
                # Update status
                cluster_info = await self.cluster_manager.get_cluster_info(self.status.cluster_id)
                self.status.active_nodes = cluster_info.total_nodes
                self.status.total_cost_per_hour = cluster_info.estimated_cost_per_hour
                
                self.logger.info(f"Scaled infrastructure for population {population_size}")
            
            self.status.state = InfrastructureState.READY
            return success
            
        except Exception as e:
            self.status.state = InfrastructureState.ERROR
            self.logger.error(f"Failed to scale infrastructure: {e}")
            return False
    
    async def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """
        Get current infrastructure metrics for integration with existing monitoring.
        
        Returns:
            Dictionary with infrastructure metrics
        """
        try:
            if not self.monitoring_manager or not self.status.cluster_id:
                return {"status": "no_active_infrastructure"}
            
            # Collect genetic algorithm specific metrics
            genetic_metrics = [
                MetricType.GENETIC_EVALUATIONS_PER_SECOND,
                MetricType.COST_PER_HOUR,
                MetricType.CLUSTER_HEALTH_SCORE,
                MetricType.CPU_UTILIZATION,
                MetricType.MEMORY_UTILIZATION
            ]
            
            metrics = await self.monitoring_manager.collect_metrics(
                self.status.cluster_id, genetic_metrics
            )
            
            # Convert to dictionary format for existing monitoring system
            metrics_dict = {
                "infrastructure_status": self.status.state,
                "platform": self.platform,
                "active_nodes": self.status.active_nodes,
                "cost_per_hour": self.status.total_cost_per_hour,
                "genetic_pool_connected": self.status.genetic_pool_connected,
                "last_updated": self.status.last_updated.isoformat()
            }
            
            # Add collected metrics
            for metric in metrics:
                metrics_dict[metric.metric_type] = metric.value
            
            return metrics_dict
            
        except Exception as e:
            self.logger.error(f"Failed to get infrastructure metrics: {e}")
            return {"error": str(e)}
    
    async def terminate_infrastructure(self, preserve_results: bool = True) -> bool:
        """
        Terminate infrastructure deployment with optional result preservation.
        
        Args:
            preserve_results: Whether to preserve genetic algorithm results
            
        Returns:
            True if termination successful
        """
        try:
            self.status.state = InfrastructureState.TERMINATING
            
            # Disconnect genetic pool
            if self.genetic_pool:
                # Reset genetic pool to local mode
                self.genetic_pool.evolution_mode = EvolutionMode.LOCAL
                self.status.genetic_pool_connected = False
            
            # Terminate deployment
            if self.deployment_manager and self.status.deployment_id:
                success = await self.deployment_manager.terminate_deployment(
                    self.status.deployment_id, preserve_results
                )
                
                if success:
                    # Clean up tracking
                    if self.status.deployment_id in self._active_deployments:
                        del self._active_deployments[self.status.deployment_id]
                    
                    # Reset status
                    self.status = InfrastructureStatus(state=InfrastructureState.UNINITIALIZED)
                    
                    self.logger.info("Infrastructure terminated successfully")
                    return True
            
            return False
            
        except Exception as e:
            self.status.state = InfrastructureState.ERROR
            self.logger.error(f"Failed to terminate infrastructure: {e}")
            return False
    
    def get_status(self) -> InfrastructureStatus:
        """Get current infrastructure status"""
        self.status.last_updated = datetime.now(timezone.utc)
        return self.status
    
    # Private helper methods
    
    def _calculate_infrastructure_requirements(self, 
                                             population_size: int,
                                             max_generations: int) -> Dict[str, Any]:
        """Calculate infrastructure requirements based on genetic parameters"""
        # Simple heuristic for infrastructure sizing
        if population_size <= 50:
            node_count = 2
            instance_type = "m5.large"
        elif population_size <= 200:
            node_count = min(population_size // 25, 8)
            instance_type = "c5.xlarge"
        else:
            node_count = min(population_size // 20, 15)
            instance_type = "c5.2xlarge"
        
        return {
            "min_nodes": max(2, node_count // 2),
            "max_nodes": node_count,
            "instance_type": instance_type,
            "auto_scaling": True,
            "spot_instances": True
        }
    
    async def _connect_genetic_pool(self, 
                                  genetic_pool: GeneticStrategyPool,
                                  deployment_result: DeploymentResult):
        """Connect genetic pool to deployed infrastructure"""
        try:
            # Get Ray client configuration
            if self.cluster_manager:
                ray_config = self.cluster_manager.get_ray_client_config(deployment_result.cluster_id)
                
                # Update genetic pool with Ray cluster information
                genetic_pool.evolution_mode = EvolutionMode.DISTRIBUTED
                # Set Ray address (this would be done through genetic pool's Ray integration)
                
                self.logger.info(f"Connected genetic pool to Ray cluster: {ray_config['ray_address']}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect genetic pool to infrastructure: {e}")
            raise
    
    async def _start_infrastructure_monitoring(self, deployment_result: DeploymentResult):
        """Start monitoring for deployed infrastructure"""
        try:
            if not self.monitoring_manager:
                return
            
            # Create infrastructure-specific dashboard
            # This would integrate with existing monitoring system
            
            # Start metric collection
            asyncio.create_task(self._monitor_infrastructure_health())
            
            self.logger.info("Started infrastructure monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to start infrastructure monitoring: {e}")
    
    async def _integrate_monitoring(self):
        """Integrate with existing RealTimeMonitoringSystem"""
        try:
            if not self.trading_monitor or not self.monitoring_manager:
                return
            
            # Configure integration
            integration_config = {
                "alert_channels": ["infrastructure_alerts"],
                "max_hourly_cost": 50.0,
                "cost_alert_channels": ["cost_alerts"]
            }
            
            # Set up monitoring integration
            await self.monitoring_manager.integrate_with_trading_monitor(integration_config)
            
            self.logger.info("Integrated with existing trading system monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate monitoring: {e}")
    
    async def _monitor_infrastructure_health(self):
        """Background task for monitoring infrastructure health"""
        try:
            while self.status.state in [InfrastructureState.READY, InfrastructureState.MONITORING]:
                if self.cluster_manager and self.status.cluster_id:
                    # Get cluster health
                    cluster_info = await self.cluster_manager.get_cluster_info(self.status.cluster_id)
                    
                    # Update status
                    self.status.health_score = cluster_info.cluster_health.overall_status
                    self.status.last_health_check = datetime.now(timezone.utc)
                    self.status.active_nodes = cluster_info.total_nodes
                    self.status.total_cost_per_hour = cluster_info.estimated_cost_per_hour
                    
                    # Check for auto-scaling opportunities
                    if cluster_info.state == "running":
                        await self.cluster_manager.auto_scale_cluster(self.status.cluster_id)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
        except asyncio.CancelledError:
            self.logger.info("Infrastructure health monitoring cancelled")
        except Exception as e:
            self.logger.error(f"Infrastructure health monitoring error: {e}")


class InfrastructureError(Exception):
    """Base exception for infrastructure operations"""
    pass


class InfrastructureNotReadyError(InfrastructureError):
    """Raised when infrastructure is not ready for operation"""
    pass


class InfrastructureCostExceededError(InfrastructureError):
    """Raised when infrastructure costs exceed limits"""
    pass