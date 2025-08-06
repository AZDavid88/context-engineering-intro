"""
Production Infrastructure - Core Cluster Manager Interface

This module defines universal cluster management contracts for Ray-based
genetic algorithm infrastructure, enabling platform-agnostic cluster
operations with platform-specific optimizations.

Research-Based Implementation:
- /research/ray_cluster/research_summary.md - Ray cluster patterns
- /research/anyscale/research_summary.md - Anyscale optimization patterns
- Existing genetic_strategy_pool.py Ray integration patterns

Key Features:
- Platform-agnostic Ray cluster management
- Heterogeneous node group support (CPU/GPU workers)
- Auto-scaling based on genetic algorithm workload
- Integration with existing GeneticStrategyPool architecture
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json

# Integration with existing system
from .deployment_interface import (
    NodeType, NodeInfo, ClusterHealth, PlatformType, 
    DeploymentError, ClusterUnhealthyError
)

# Set up logging
logger = logging.getLogger(__name__)


class ClusterState(str, Enum):
    """Ray cluster lifecycle states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    UPDATING = "updating"
    UNHEALTHY = "unhealthy"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"


class WorkloadType(str, Enum):
    """Genetic algorithm workload types for cluster optimization"""
    GENETIC_EVOLUTION = "genetic_evolution"      # CPU-intensive genetic operations
    STRATEGY_EVALUATION = "strategy_evaluation"  # Mixed CPU/memory workload
    ML_INFERENCE = "ml_inference"                # GPU-intensive ML operations
    DATA_PROCESSING = "data_processing"          # Memory-intensive operations


@dataclass
class ClusterConfig:
    """Configuration for Ray cluster deployment"""
    # Basic cluster configuration
    cluster_name: str
    region: str
    platform: PlatformType
    
    # Node group configurations
    head_node_config: Dict[str, Any]
    worker_groups: List[Dict[str, Any]]
    
    # Ray-specific configuration
    ray_version: str = "2.8.0"
    python_version: str = "3.10"
    
    # Genetic algorithm optimizations
    genetic_workload_config: Dict[WorkloadType, Dict[str, Any]] = field(default_factory=dict)
    
    # Auto-scaling configuration
    enable_autoscaling: bool = True
    max_worker_nodes: int = 50
    min_worker_nodes: int = 2
    target_utilization: float = 0.8
    
    # Cost optimization
    use_spot_instances: bool = True
    spot_fallback_on_demand: bool = True
    max_spot_interruption_rate: float = 0.05  # 5% acceptable interruption rate
    
    # Integration configuration
    custom_setup_commands: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate cluster configuration"""
        if not self.cluster_name:
            raise ValueError("Cluster name is required")
        if not self.head_node_config:
            raise ValueError("Head node configuration is required")
        if not self.worker_groups:
            raise ValueError("At least one worker group is required")
        if self.min_worker_nodes > self.max_worker_nodes:
            raise ValueError("Min worker nodes cannot exceed max worker nodes")
        if not 0.0 < self.target_utilization <= 1.0:
            raise ValueError("Target utilization must be between 0 and 1")
        return True


@dataclass
class WorkerGroupInfo:
    """Information about a worker group in the cluster"""
    group_name: str
    node_type: NodeType
    instance_type: str
    min_nodes: int
    max_nodes: int
    current_nodes: int
    target_nodes: int
    cpu_cores_per_node: int
    memory_gb_per_node: int
    cost_per_hour_per_node: float
    is_spot_group: bool
    availability_zones: List[str]
    workload_types: List[WorkloadType]


@dataclass
class ClusterInfo:
    """Comprehensive cluster information"""
    cluster_id: str
    cluster_name: str
    platform: PlatformType
    state: ClusterState
    
    # Node information
    head_node: NodeInfo
    worker_groups: List[WorkerGroupInfo]
    total_nodes: int
    healthy_nodes: int
    
    # Resource information
    total_cpu_cores: int
    total_memory_gb: int
    total_gpu_count: int
    
    # Cost information
    estimated_cost_per_hour: float
    actual_cost_current_hour: float
    
    # Ray cluster information
    ray_version: str
    ray_dashboard_url: Optional[str] = None
    ray_client_url: Optional[str] = None
    
    # Health and performance
    cluster_health: ClusterHealth
    active_tasks: int = 0
    queued_tasks: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScalingDecision:
    """Auto-scaling decision information"""
    should_scale: bool
    target_nodes: int
    current_nodes: int
    scaling_reason: str
    workload_metrics: Dict[str, float]
    cost_impact: float
    confidence_score: float  # 0.0 to 1.0


class ClusterManager(ABC):
    """
    Universal cluster management interface for genetic algorithm infrastructure.
    
    This abstract base class defines the contract for managing Ray clusters
    across different platforms, with optimizations for genetic algorithm
    workloads and integration with the existing GeneticStrategyPool.
    """
    
    def __init__(self, platform: PlatformType, config: Dict[str, Any]):
        """
        Initialize cluster manager for specific platform.
        
        Args:
            platform: Target deployment platform
            config: Platform-specific configuration
        """
        self.platform = platform
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{platform}")
        self._active_clusters: Dict[str, ClusterInfo] = {}
    
    @abstractmethod
    async def initialize_ray_cluster(self, config: ClusterConfig) -> ClusterInfo:
        """
        Initialize Ray cluster optimized for genetic algorithm workloads.
        
        This method creates a new Ray cluster with heterogeneous node groups
        optimized for different phases of genetic algorithm execution:
        - CPU workers for genetic evolution operations
        - GPU workers for ML inference (if needed)
        - Memory workers for large population handling
        
        Args:
            config: Cluster configuration with genetic algorithm optimizations
            
        Returns:
            ClusterInfo with complete cluster details
            
        Raises:
            DeploymentError: If cluster initialization fails
            CostLimitExceededError: If cluster would exceed cost limits
        """
        pass
    
    @abstractmethod
    async def add_worker_nodes(self, cluster_id: str, 
                             node_type: NodeType, 
                             count: int,
                             workload_type: WorkloadType) -> List[NodeInfo]:
        """
        Add worker nodes optimized for specific genetic algorithm workloads.
        
        Args:
            cluster_id: Unique cluster identifier
            node_type: Type of nodes to add (CPU/GPU/Memory workers)
            count: Number of nodes to add
            workload_type: Genetic algorithm workload type for optimization
            
        Returns:
            List of NodeInfo for newly added nodes
        """
        pass
    
    @abstractmethod
    async def remove_worker_nodes(self, cluster_id: str, 
                                node_ids: List[str],
                                drain_tasks: bool = True) -> bool:
        """
        Remove worker nodes with graceful task draining.
        
        Args:
            cluster_id: Unique cluster identifier
            node_ids: List of node IDs to remove
            drain_tasks: Whether to wait for tasks to complete before removal
            
        Returns:
            True if removal successful
        """
        pass
    
    @abstractmethod
    async def get_cluster_info(self, cluster_id: str) -> ClusterInfo:
        """
        Get comprehensive cluster information and status.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            ClusterInfo with detailed cluster state
        """
        pass
    
    @abstractmethod
    async def terminate_cluster(self, cluster_id: str, 
                              preserve_results: bool = True) -> bool:
        """
        Terminate Ray cluster with optional result preservation.
        
        Args:
            cluster_id: Unique cluster identifier
            preserve_results: Whether to preserve genetic algorithm results
            
        Returns:
            True if termination successful
        """
        pass
    
    @abstractmethod
    async def get_cluster_metrics(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get real-time cluster performance metrics.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    # Auto-scaling methods
    
    async def evaluate_scaling_decision(self, cluster_id: str) -> ScalingDecision:
        """
        Evaluate whether cluster should be scaled based on workload metrics.
        
        This method analyzes genetic algorithm workload patterns and makes
        intelligent scaling decisions based on:
        - Current CPU/memory utilization
        - Queue depth for genetic evaluations
        - Cost optimization targets
        - Spot instance interruption rates
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            ScalingDecision with recommended scaling action
        """
        try:
            cluster_info = await self.get_cluster_info(cluster_id)
            metrics = await self.get_cluster_metrics(cluster_id)
            
            # Analyze workload patterns
            cpu_utilization = metrics.get("cpu_utilization", 0.0)
            memory_utilization = metrics.get("memory_utilization", 0.0)
            queue_depth = metrics.get("queued_tasks", 0)
            active_evaluations = metrics.get("active_genetic_evaluations", 0)
            
            # Calculate scaling need
            current_nodes = cluster_info.total_nodes - 1  # Exclude head node
            target_utilization = 0.8  # Target 80% utilization
            
            # Simple scaling logic (platform implementations can override)
            should_scale_up = (
                cpu_utilization > target_utilization or 
                memory_utilization > target_utilization or
                queue_depth > current_nodes * 2
            )
            
            should_scale_down = (
                cpu_utilization < target_utilization * 0.5 and
                memory_utilization < target_utilization * 0.5 and
                queue_depth == 0 and
                active_evaluations == 0
            )
            
            if should_scale_up:
                target_nodes = min(current_nodes + 2, cluster_info.worker_groups[0].max_nodes)
                reason = "High utilization or queue depth"
                confidence = 0.8
            elif should_scale_down:
                target_nodes = max(current_nodes - 1, cluster_info.worker_groups[0].min_nodes)
                reason = "Low utilization and no active work"
                confidence = 0.7
            else:
                target_nodes = current_nodes
                reason = "Cluster optimally sized"
                confidence = 0.9
            
            # Estimate cost impact
            cost_per_node = cluster_info.estimated_cost_per_hour / current_nodes if current_nodes > 0 else 0
            cost_impact = (target_nodes - current_nodes) * cost_per_node
            
            return ScalingDecision(
                should_scale=(target_nodes != current_nodes),
                target_nodes=target_nodes,
                current_nodes=current_nodes,
                scaling_reason=reason,
                workload_metrics={
                    "cpu_utilization": cpu_utilization,
                    "memory_utilization": memory_utilization,
                    "queue_depth": queue_depth,
                    "active_evaluations": active_evaluations
                },
                cost_impact=cost_impact,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate scaling decision: {e}")
            return ScalingDecision(
                should_scale=False,
                target_nodes=0,
                current_nodes=0,
                scaling_reason=f"Error: {e}",
                workload_metrics={},
                cost_impact=0.0,
                confidence_score=0.0
            )
    
    async def auto_scale_cluster(self, cluster_id: str) -> bool:
        """
        Automatically scale cluster based on workload analysis.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            True if scaling was performed
        """
        try:
            decision = await self.evaluate_scaling_decision(cluster_id)
            
            if not decision.should_scale:
                self.logger.debug(f"No scaling needed for cluster {cluster_id}: {decision.scaling_reason}")
                return False
            
            if decision.confidence_score < 0.6:
                self.logger.warning(f"Low confidence scaling decision for cluster {cluster_id}, skipping")
                return False
            
            cluster_info = await self.get_cluster_info(cluster_id)
            current_workers = cluster_info.total_nodes - 1  # Exclude head node
            
            if decision.target_nodes > current_workers:
                # Scale up
                nodes_to_add = decision.target_nodes - current_workers
                worker_group = cluster_info.worker_groups[0]  # Use first worker group
                
                added_nodes = await self.add_worker_nodes(
                    cluster_id=cluster_id,
                    node_type=NodeType.CPU_WORKER,
                    count=nodes_to_add,
                    workload_type=WorkloadType.GENETIC_EVOLUTION
                )
                
                self.logger.info(f"Scaled up cluster {cluster_id}: added {len(added_nodes)} nodes")
                
            elif decision.target_nodes < current_workers:
                # Scale down
                nodes_to_remove = current_workers - decision.target_nodes
                
                # Select nodes to remove (prefer spot instances, newer nodes)
                all_worker_nodes = []
                for group in cluster_info.worker_groups:
                    # This would be implemented by platform-specific managers
                    pass
                
                # For now, just log the intent (platform implementations will handle actual removal)
                self.logger.info(f"Would scale down cluster {cluster_id}: remove {nodes_to_remove} nodes")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Auto-scaling failed for cluster {cluster_id}: {e}")
            return False
    
    # Integration methods for existing GeneticStrategyPool
    
    async def optimize_for_genetic_workload(self, cluster_id: str, 
                                          population_size: int,
                                          evaluation_complexity: str = "medium") -> bool:
        """
        Optimize cluster configuration for specific genetic algorithm workload.
        
        This method adjusts cluster settings based on genetic algorithm
        characteristics to maximize performance and minimize cost.
        
        Args:
            cluster_id: Unique cluster identifier
            population_size: Size of genetic population
            evaluation_complexity: "simple", "medium", "complex"
            
        Returns:
            True if optimization applied successfully
        """
        try:
            cluster_info = await self.get_cluster_info(cluster_id)
            
            # Calculate optimal node count based on population size
            if evaluation_complexity == "simple":
                evals_per_node = 50
            elif evaluation_complexity == "medium":
                evals_per_node = 20
            else:  # complex
                evals_per_node = 10
            
            optimal_nodes = max(2, min(population_size // evals_per_node, 20))
            current_workers = cluster_info.total_nodes - 1
            
            if optimal_nodes != current_workers:
                self.logger.info(
                    f"Optimizing cluster {cluster_id} for population {population_size}: "
                    f"{current_workers} -> {optimal_nodes} nodes"
                )
                
                if optimal_nodes > current_workers:
                    await self.add_worker_nodes(
                        cluster_id=cluster_id,
                        node_type=NodeType.CPU_WORKER,
                        count=optimal_nodes - current_workers,
                        workload_type=WorkloadType.GENETIC_EVOLUTION
                    )
                # Note: Scale down would be handled by auto-scaling
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to optimize cluster for genetic workload: {e}")
            return False
    
    def get_ray_client_config(self, cluster_id: str) -> Dict[str, str]:
        """
        Get Ray client configuration for connecting to cluster.
        
        This method returns the configuration needed for the existing
        GeneticStrategyPool to connect to the Ray cluster.
        
        Args:
            cluster_id: Unique cluster identifier
            
        Returns:
            Dictionary with Ray client configuration
        """
        if cluster_id not in self._active_clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        cluster_info = self._active_clusters[cluster_id]
        
        return {
            "ray_address": cluster_info.ray_client_url or "auto",
            "ray_dashboard_url": cluster_info.ray_dashboard_url or "",
            "cluster_id": cluster_id,
            "platform": cluster_info.platform,
        }


class ClusterStateError(DeploymentError):
    """Raised when cluster is in invalid state for operation"""
    def __init__(self, cluster_id: str, current_state: ClusterState, required_state: ClusterState):
        self.cluster_id = cluster_id
        self.current_state = current_state
        self.required_state = required_state
        super().__init__(
            f"Cluster {cluster_id} is in state {current_state}, "
            f"but operation requires state {required_state}"
        )


class InsufficientResourcesError(DeploymentError):
    """Raised when cluster lacks resources for requested operation"""
    def __init__(self, requested: Dict[str, int], available: Dict[str, int]):
        self.requested = requested
        self.available = available
        super().__init__(f"Insufficient resources: requested {requested}, available {available}")


class WorkloadOptimizationError(DeploymentError):
    """Raised when workload optimization fails"""
    pass