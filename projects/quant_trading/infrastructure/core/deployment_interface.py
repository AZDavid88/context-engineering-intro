"""
Production Infrastructure - Core Deployment Interface

This module defines universal deployment contracts for the genetic algorithm
infrastructure, enabling platform-agnostic deployment with platform-specific
optimizations.

Research-Based Implementation:
- /research/anyscale/research_summary.md - Platform optimization patterns
- PHASE_5B5_INFRASTRUCTURE_ARCHITECTURE.md - Architecture design
- Existing genetic_strategy_pool.py integration patterns

Key Features:
- Platform-agnostic deployment interface
- Cost-aware resource management
- Health monitoring and failure recovery
- Integration with existing genetic algorithm core
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json

# Set up logging
logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Supported deployment platforms"""
    ANYSCALE = "anyscale"
    AWS_ECS = "aws_ecs"
    AWS_EKS = "aws_eks"
    DIGITALOCEAN = "digitalocean"
    LOCAL = "local"


class DeploymentStatus(str, Enum):
    """Deployment lifecycle status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    UNHEALTHY = "unhealthy"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"


class NodeType(str, Enum):
    """Ray cluster node types for genetic algorithm workloads"""
    HEAD_NODE = "head_node"
    CPU_WORKER = "cpu_worker"
    GPU_WORKER = "gpu_worker"
    MEMORY_WORKER = "memory_worker"


@dataclass
class CostLimits:
    """Cost management configuration for genetic algorithm deployments"""
    max_hourly_cost: float
    max_total_cost: Optional[float] = None
    cost_alert_threshold: float = 0.8  # Alert at 80% of limit
    auto_scale_down_threshold: float = 0.9  # Scale down at 90% of limit
    hard_stop_threshold: float = 0.95  # Hard stop at 95% of limit


@dataclass
class GeneticPoolConfig:
    """Configuration for genetic algorithm pool deployment"""
    # Genetic Algorithm Configuration
    population_size: int
    max_generations: int
    evaluation_timeout: int
    
    # Platform Configuration
    platform: PlatformType
    platform_config: Dict[str, Any]
    
    # Resource Management
    min_nodes: int = 2
    max_nodes: int = 10
    node_type: NodeType = NodeType.CPU_WORKER
    
    # Cost Management
    cost_limits: Optional[CostLimits] = None
    use_spot_instances: bool = True
    spot_fallback_on_demand: bool = True
    
    # Monitoring Configuration
    health_check_interval: int = 30  # seconds
    metrics_collection: bool = True
    custom_monitoring: Dict[str, Any] = field(default_factory=dict)
    
    # Integration Configuration
    genetic_algorithm_image: str = "genetic-pool:latest"
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate configuration for deployment"""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.max_generations <= 0:
            raise ValueError("Max generations must be positive")
        if self.min_nodes > self.max_nodes:
            raise ValueError("Min nodes cannot exceed max nodes")
        if self.cost_limits and self.cost_limits.max_hourly_cost <= 0:
            raise ValueError("Max hourly cost must be positive")
        return True


@dataclass
class NodeInfo:
    """Information about individual cluster nodes"""
    node_id: str
    node_type: NodeType
    instance_type: str
    status: str
    cpu_cores: int
    memory_gb: int
    cost_per_hour: float
    is_spot_instance: bool
    availability_zone: str
    created_at: datetime


@dataclass
class ClusterHealth:
    """Comprehensive cluster health information"""
    overall_status: str
    healthy_nodes: int
    total_nodes: int
    cpu_utilization: float
    memory_utilization: float
    active_genetic_evaluations: int
    failed_evaluations_last_hour: int
    cost_per_hour: float
    estimated_completion_time: Optional[int] = None  # seconds
    last_health_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DeploymentResult:
    """Result of genetic pool deployment"""
    deployment_id: str
    cluster_id: str
    platform: PlatformType
    status: DeploymentStatus
    endpoint_url: str
    
    # Cost Information
    estimated_cost_per_hour: float
    total_cost_estimate: Optional[float] = None
    optimization_applied: bool = False
    
    # Cluster Information
    total_nodes: int = 0
    node_types: List[NodeType] = field(default_factory=list)
    
    # Integration Information
    genetic_pool_endpoint: Optional[str] = None
    monitoring_dashboard_url: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_ready_time: Optional[datetime] = None


@dataclass
class ScalingResult:
    """Result of cluster scaling operation"""
    success: bool
    previous_node_count: int
    new_node_count: int
    scaling_duration: float  # seconds
    cost_impact: float  # change in cost per hour
    message: str


class DeploymentManager(ABC):
    """
    Universal deployment interface for genetic algorithm infrastructure.
    
    This abstract base class defines the contract for deploying, managing,
    and monitoring genetic algorithm workloads across different platforms.
    Platform-specific implementations provide optimized deployment strategies
    while maintaining consistent interfaces.
    """
    
    def __init__(self, platform: PlatformType, config: Dict[str, Any]):
        """
        Initialize deployment manager for specific platform.
        
        Args:
            platform: Target deployment platform
            config: Platform-specific configuration
        """
        self.platform = platform
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{platform}")
    
    @abstractmethod  
    async def deploy_genetic_pool(self, config: GeneticPoolConfig) -> DeploymentResult:
        """
        Deploy genetic algorithm pool with platform-specific optimizations.
        
        This method creates a distributed Ray cluster optimized for genetic
        algorithm workloads, integrating with the existing genetic_strategy_pool.py
        implementation.
        
        Args:
            config: Genetic pool deployment configuration
            
        Returns:
            DeploymentResult with cluster information and cost estimates
            
        Raises:
            DeploymentError: If deployment fails
            CostLimitExceededError: If deployment would exceed cost limits
        """
        pass
    
    @abstractmethod
    async def scale_cluster(self, cluster_id: str, target_nodes: int) -> ScalingResult:
        """
        Scale cluster up or down based on genetic workload requirements.
        
        Args:
            cluster_id: Unique cluster identifier
            target_nodes: Desired number of worker nodes
            
        Returns:
            ScalingResult with scaling operation details
        """
        pass
    
    @abstractmethod
    async def get_cluster_health(self, cluster_id: str) -> ClusterHealth:
        """
        Get comprehensive cluster health and genetic algorithm status.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            ClusterHealth with detailed health metrics
        """
        pass
    
    @abstractmethod
    async def get_deployment_status(self, deployment_id: str) -> DeploymentResult:
        """
        Get current deployment status and information.
        
        Args:
            deployment_id: Unique deployment identifier
            
        Returns:
            DeploymentResult with current status
        """
        pass
    
    @abstractmethod
    async def terminate_deployment(self, deployment_id: str, 
                                 preserve_data: bool = True) -> bool:
        """
        Terminate genetic algorithm deployment and cleanup resources.
        
        Args:
            deployment_id: Unique deployment identifier
            preserve_data: Whether to preserve genetic algorithm results
            
        Returns:
            True if termination successful
        """
        pass
    
    @abstractmethod
    async def get_cost_metrics(self, deployment_id: str) -> Dict[str, float]:
        """
        Get real-time cost metrics for genetic algorithm deployment.
        
        Args:
            deployment_id: Unique deployment identifier
            
        Returns:
            Dictionary with cost metrics (hourly, total, projected)
        """
        pass
    
    # Common utility methods for all platforms
    
    async def validate_genetic_integration(self, deployment_result: DeploymentResult) -> bool:
        """
        Validate integration with existing genetic_strategy_pool.py.
        
        This method ensures the deployed infrastructure can properly execute
        the existing genetic algorithm implementation.
        """
        try:
            # Test Ray cluster connectivity
            endpoint = deployment_result.genetic_pool_endpoint
            if not endpoint:
                self.logger.error("No genetic pool endpoint available")
                return False
            
            # Test genetic algorithm execution (placeholder for actual integration test)
            self.logger.info(f"Validating genetic integration for deployment {deployment_result.deployment_id}")
            
            # TODO: Implement actual genetic algorithm test execution
            # This would test a small genetic population to ensure everything works
            
            return True
            
        except Exception as e:
            self.logger.error(f"Genetic integration validation failed: {e}")
            return False
    
    def calculate_cost_efficiency(self, config: GeneticPoolConfig, 
                                result: DeploymentResult) -> Dict[str, float]:
        """
        Calculate cost efficiency metrics for genetic algorithm deployment.
        
        Args:
            config: Original deployment configuration
            result: Deployment result with cost information
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Baseline cost calculation (traditional cloud deployment)
        baseline_cost_per_hour = config.population_size * 0.1  # Rough estimate
        
        # Actual deployment cost
        actual_cost_per_hour = result.estimated_cost_per_hour
        
        # Calculate efficiency metrics
        cost_savings = max(0, baseline_cost_per_hour - actual_cost_per_hour)
        efficiency_percentage = (cost_savings / baseline_cost_per_hour) * 100 if baseline_cost_per_hour > 0 else 0
        
        return {
            "baseline_cost_per_hour": baseline_cost_per_hour,
            "actual_cost_per_hour": actual_cost_per_hour,
            "cost_savings_per_hour": cost_savings,
            "efficiency_percentage": efficiency_percentage,
            "optimization_applied": result.optimization_applied
        }


class DeploymentError(Exception):
    """Base exception for deployment operations"""
    pass


class CostLimitExceededError(DeploymentError):
    """Raised when deployment would exceed cost limits"""
    def __init__(self, estimated_cost: float, limit: float):
        self.estimated_cost = estimated_cost
        self.limit = limit
        super().__init__(f"Estimated cost ${estimated_cost:.2f}/hour exceeds limit ${limit:.2f}/hour")


class ClusterUnhealthyError(DeploymentError):
    """Raised when cluster is in unhealthy state"""
    pass


class PlatformNotSupportedError(DeploymentError):
    """Raised when requested platform is not supported"""
    def __init__(self, platform: str):
        self.platform = platform
        super().__init__(f"Platform '{platform}' is not supported")