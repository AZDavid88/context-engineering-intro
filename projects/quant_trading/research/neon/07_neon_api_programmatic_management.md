# Neon API Programmatic Management - Complete Reference for Ray Workers

**Source**: https://api-docs.neon.tech/reference/getting-started-with-neon-api  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - Automated Database Management for Distributed Trading System

## Executive Summary

Complete Neon API reference for programmatic management of databases, compute endpoints, and branches in distributed Ray worker environments. Essential for Phase 4 automated genetic algorithm coordination and cloud database orchestration.

## API Foundation

### Base Configuration
```python
# File: src/data/neon_api_client.py

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class NeonAPIConfig:
    """Neon API configuration for Phase 4."""
    base_url: str = "https://console.neon.tech/api/v2"
    api_key: str = None
    timeout: int = 30
    max_retries: int = 3

class NeonAPIClient:
    """Production-ready Neon API client for Ray worker coordination."""
    
    def __init__(self, config: NeonAPIConfig):
        self.config = config
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }
        
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=self.headers,
            timeout=config.timeout
        )
```

## Critical API Endpoints for Phase 4

### 1. Project Management

#### Create Project for Trading System
```python
async def create_trading_project(self, region: str = "us-east-1") -> Dict[str, Any]:
    """Create new Neon project optimized for trading workloads."""
    
    payload = {
        "project": {
            "name": f"quant-trading-{region}",
            "region_id": region,
            "settings": {
                "quota": {
                    "active_time_seconds": -1,  # No time limits
                    "compute_time_seconds": -1,
                    "written_data_bytes": -1,
                    "data_transfer_bytes": -1
                }
            }
        }
    }
    
    response = await self.client.post("/projects", json=payload)
    return response.json()

async def list_projects(self) -> Dict[str, Any]:
    """List all projects for management."""
    response = await self.client.get("/projects")
    return response.json()

async def get_project_details(self, project_id: str) -> Dict[str, Any]:
    """Get detailed project information."""
    response = await self.client.get(f"/projects/{project_id}")
    return response.json()
```

### 2. Branch Management (Critical for GA Coordination)

#### Dynamic Branch Creation for Strategy Testing
```python
async def create_strategy_branch(self, project_id: str, strategy_id: str, 
                               parent_branch: str = "main") -> Dict[str, Any]:
    """Create branch for isolated strategy development/testing."""
    
    payload = {
        "branch": {
            "name": f"strategy-{strategy_id}",
            "parent_id": parent_branch,
            "parent_lsn": None,  # Use latest data
            "parent_timestamp": None
        }
    }
    
    response = await self.client.post(
        f"/projects/{project_id}/branches", 
        json=payload
    )
    return response.json()

async def list_branches(self, project_id: str) -> Dict[str, Any]:
    """List all branches in project."""
    response = await self.client.get(f"/projects/{project_id}/branches")
    return response.json()

async def delete_strategy_branch(self, project_id: str, branch_id: str) -> Dict[str, Any]:
    """Clean up completed strategy branches."""
    response = await self.client.delete(f"/projects/{project_id}/branches/{branch_id}")
    return response.json()

async def restore_branch_to_timestamp(self, project_id: str, branch_id: str, 
                                    timestamp: str) -> Dict[str, Any]:
    """Restore branch to specific point in time for analysis."""
    
    payload = {
        "timestamp": timestamp,
        "preserve_under_name": f"backup-{int(time.time())}"
    }
    
    response = await self.client.post(
        f"/projects/{project_id}/branches/{branch_id}/restore",
        json=payload
    )
    return response.json()
```

### 3. Compute Endpoint Management

#### Ray Worker Compute Orchestration
```python
async def create_ray_worker_endpoint(self, project_id: str, branch_id: str, 
                                   worker_id: str) -> Dict[str, Any]:
    """Create dedicated compute endpoint for Ray worker."""
    
    payload = {
        "endpoint": {
            "branch_id": branch_id,
            "type": "read_write",
            "compute_settings": {
                "min": 2.0,     # 8GB RAM minimum
                "max": 16.0,    # 64GB RAM maximum for intensive backtests
                "autoscaling": {
                    "scale_down_delay_seconds": 300,  # 5 min before scale down
                    "scale_up_delay_seconds": 60      # 1 min to scale up
                }
            },
            "settings": {
                "pg_settings": {
                    "shared_preload_libraries": "timescaledb,pg_stat_statements",
                    "max_connections": "1000",
                    "work_mem": "256MB",
                    "effective_cache_size": "12GB"
                }
            }
        }
    }
    
    response = await self.client.post(
        f"/projects/{project_id}/endpoints",
        json=payload
    )
    return response.json()

async def scale_compute_for_workload(self, project_id: str, endpoint_id: str,
                                   target_cu: float) -> Dict[str, Any]:
    """Dynamically scale compute based on workload."""
    
    payload = {
        "endpoint": {
            "compute_settings": {
                "min": max(target_cu * 0.5, 1.0),  # Minimum 50% of target
                "max": target_cu * 2.0              # Maximum 200% of target
            }
        }
    }
    
    response = await self.client.patch(
        f"/projects/{project_id}/endpoints/{endpoint_id}",
        json=payload
    )
    return response.json()

async def restart_compute_endpoint(self, project_id: str, endpoint_id: str) -> Dict[str, Any]:
    """Restart compute endpoint for maintenance or recovery."""
    response = await self.client.post(
        f"/projects/{project_id}/endpoints/{endpoint_id}/restart"
    )
    return response.json()
```

### 4. Database Management

#### Trading Database Orchestration
```python
async def create_trading_database(self, project_id: str, branch_id: str,
                                database_name: str) -> Dict[str, Any]:
    """Create database optimized for trading data."""
    
    payload = {
        "database": {
            "name": database_name,
            "owner_name": "trading_admin"
        }
    }
    
    response = await self.client.post(
        f"/projects/{project_id}/branches/{branch_id}/databases",
        json=payload
    )
    return response.json()

async def setup_trading_schema(self, project_id: str, branch_id: str,
                             database_name: str) -> Dict[str, Any]:
    """Get database schema for validation."""
    response = await self.client.get(
        f"/projects/{project_id}/branches/{branch_id}/schema?database={database_name}"
    )
    return response.json()
```

### 5. Role Management for Security

#### Dynamic Role Creation for Ray Workers
```python
async def create_ray_worker_role(self, project_id: str, branch_id: str,
                               worker_id: str) -> Dict[str, Any]:
    """Create secure role for Ray worker with limited permissions."""
    
    payload = {
        "role": {
            "name": f"ray_worker_{worker_id}",
            "protected": False  # Allow password resets
        }
    }
    
    response = await self.client.post(
        f"/projects/{project_id}/branches/{branch_id}/roles",
        json=payload
    )
    return response.json()

async def rotate_worker_password(self, project_id: str, branch_id: str,
                               role_name: str) -> Dict[str, Any]:
    """Rotate Ray worker password for security."""
    response = await self.client.post(
        f"/projects/{project_id}/branches/{branch_id}/roles/{role_name}/reset_password"
    )
    return response.json()

async def get_role_password(self, project_id: str, branch_id: str,
                          role_name: str) -> Dict[str, Any]:
    """Retrieve role password for connection string generation."""
    response = await self.client.get(
        f"/projects/{project_id}/branches/{branch_id}/roles/{role_name}/password"
    )
    return response.json()
```

### 6. Operations Monitoring

#### Track Long-Running Operations
```python
async def monitor_operation(self, project_id: str, operation_id: str) -> Dict[str, Any]:
    """Monitor long-running operations like branch creation."""
    response = await self.client.get(
        f"/projects/{project_id}/operations/{operation_id}"
    )
    return response.json()

async def wait_for_operation_completion(self, project_id: str, operation_id: str,
                                      timeout: int = 300) -> Dict[str, Any]:
    """Wait for operation completion with polling."""
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        operation = await self.monitor_operation(project_id, operation_id)
        
        if operation["operation"]["status"] == "finished":
            return operation
        elif operation["operation"]["status"] == "error":
            raise RuntimeError(f"Operation failed: {operation['operation']['error']}")
            
        await asyncio.sleep(5)  # Poll every 5 seconds
    
    raise TimeoutError(f"Operation {operation_id} did not complete within {timeout}s")
```

## Advanced Ray Worker Coordination

### Automated Environment Management
```python
class NeonRayCoordinator:
    """Coordinate Neon resources for distributed Ray workers."""
    
    def __init__(self, api_client: NeonAPIClient):
        self.api = api_client
        self.active_workers = {}
        self.resource_pool = {}
        
    async def provision_worker_environment(self, worker_id: str, 
                                         workload_type: str) -> Dict[str, Any]:
        """Provision complete Neon environment for Ray worker."""
        
        # Determine resource requirements
        compute_requirements = self._calculate_compute_needs(workload_type)
        
        # Create dedicated branch for worker
        branch = await self.api.create_strategy_branch(
            self.project_id, 
            f"worker-{worker_id}"
        )
        
        # Create compute endpoint
        endpoint = await self.api.create_ray_worker_endpoint(
            self.project_id,
            branch["branch"]["id"],
            worker_id
        )
        
        # Create worker role
        role = await self.api.create_ray_worker_role(
            self.project_id,
            branch["branch"]["id"], 
            worker_id
        )
        
        # Generate connection string
        connection_uri = await self._build_connection_string(
            endpoint, role, worker_id
        )
        
        # Track resources
        self.active_workers[worker_id] = {
            "branch_id": branch["branch"]["id"],
            "endpoint_id": endpoint["endpoint"]["id"], 
            "role_name": role["role"]["name"],
            "connection_uri": connection_uri,
            "workload_type": workload_type
        }
        
        return self.active_workers[worker_id]
    
    async def cleanup_worker_environment(self, worker_id: str) -> None:
        """Clean up Neon resources when worker completes."""
        
        if worker_id not in self.active_workers:
            return
            
        worker_resources = self.active_workers[worker_id]
        
        # Delete compute endpoint
        await self.api.client.delete(
            f"/projects/{self.project_id}/endpoints/{worker_resources['endpoint_id']}"
        )
        
        # Delete branch (this also removes roles and databases)
        await self.api.delete_strategy_branch(
            self.project_id,
            worker_resources["branch_id"]
        )
        
        # Remove from tracking
        del self.active_workers[worker_id]
        
    async def scale_worker_resources(self, worker_id: str, 
                                   new_workload_type: str) -> Dict[str, Any]:
        """Dynamically scale worker resources based on workload changes."""
        
        if worker_id not in self.active_workers:
            raise ValueError(f"Worker {worker_id} not found")
            
        worker_resources = self.active_workers[worker_id]
        compute_requirements = self._calculate_compute_needs(new_workload_type)
        
        # Scale compute endpoint
        result = await self.api.scale_compute_for_workload(
            self.project_id,
            worker_resources["endpoint_id"],
            compute_requirements["target_cu"]
        )
        
        # Update tracking
        self.active_workers[worker_id]["workload_type"] = new_workload_type
        
        return result
    
    def _calculate_compute_needs(self, workload_type: str) -> Dict[str, float]:
        """Calculate compute requirements based on workload type."""
        
        requirements = {
            "data_ingestion": {"target_cu": 2.0},
            "backtesting_light": {"target_cu": 4.0},
            "backtesting_intensive": {"target_cu": 8.0},
            "genetic_evolution": {"target_cu": 16.0},
            "model_training": {"target_cu": 32.0}
        }
        
        return requirements.get(workload_type, {"target_cu": 4.0})
    
    async def _build_connection_string(self, endpoint: Dict, role: Dict,
                                     worker_id: str) -> str:
        """Build connection string for Ray worker."""
        
        # Get role password
        password_info = await self.api.get_role_password(
            self.project_id,
            endpoint["endpoint"]["branch_id"],
            role["role"]["name"]
        )
        
        # Build pooled connection string
        host = endpoint["endpoint"]["host"]
        database = "trading"  # Default database name
        user = role["role"]["name"]  
        password = password_info["password"]
        
        connection_string = (
            f"postgresql://{user}:{password}@{host}/{database}"
            f"?sslmode=require&application_name=ray_worker_{worker_id}"
        )
        
        return connection_string
```

## Consumption & Cost Monitoring

### Usage Tracking
```python
async def get_project_consumption(self, project_id: str, 
                                from_time: str, to_time: str) -> Dict[str, Any]:
    """Track project consumption for cost optimization."""
    
    params = {
        "from": from_time,
        "to": to_time,
        "granularity": "hourly"
    }
    
    response = await self.client.get(
        f"/consumption_history/projects/{project_id}",
        params=params
    )
    return response.json()

async def analyze_worker_costs(self, time_period: str = "7d") -> Dict[str, Any]:
    """Analyze costs per Ray worker for optimization."""
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    consumption = await self.get_project_consumption(
        self.project_id,
        start_time.isoformat(),
        end_time.isoformat()
    )
    
    # Analyze consumption patterns
    analysis = {
        "total_compute_hours": sum(c["compute_time"] for c in consumption["periods"]),
        "total_storage_gb": max(c["data_storage_bytes_hour"] for c in consumption["periods"]) / (1024**3),
        "total_data_transfer_gb": sum(c["data_transfer_bytes"] for c in consumption["periods"]) / (1024**3),
        "estimated_cost": self._calculate_estimated_cost(consumption)
    }
    
    return analysis
```

## Error Handling & Resilience

### Robust API Client
```python
class ResilientNeonAPI:
    """Production-ready Neon API client with comprehensive error handling."""
    
    async def execute_with_retry(self, method: str, url: str, 
                               **kwargs) -> Dict[str, Any]:
        """Execute API call with automatic retry and error handling."""
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = int(e.response.headers.get("Retry-After", 60))
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code >= 500:  # Server error
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                raise
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
```

## Integration with Phase 4 Components

### Genetic Algorithm Coordination
```python
class GeneticAlgorithmNeonOrchestrator:
    """Orchestrate Neon resources for genetic algorithm evolution."""
    
    async def provision_evolution_environment(self, evolution_id: str,
                                            population_size: int) -> Dict[str, Any]:
        """Provision Neon environment for genetic algorithm evolution."""
        
        # Calculate required workers based on population size
        required_workers = min(population_size // 10, 20)  # Max 20 workers
        
        # Create main evolution branch
        evolution_branch = await self.api.create_strategy_branch(
            self.project_id,
            f"evolution-{evolution_id}"
        )
        
        # Provision worker environments
        worker_environments = []
        for worker_idx in range(required_workers):
            worker_id = f"evo-{evolution_id}-w{worker_idx}"
            
            worker_env = await self.coordinator.provision_worker_environment(
                worker_id,
                "genetic_evolution"
            )
            worker_environments.append(worker_env)
        
        return {
            "evolution_id": evolution_id,
            "main_branch_id": evolution_branch["branch"]["id"],
            "worker_environments": worker_environments,
            "total_workers": required_workers
        }
    
    async def coordinate_evolution_step(self, evolution_id: str,
                                      generation: int) -> Dict[str, Any]:
        """Coordinate database operations for evolution step."""
        
        # Store evolution state
        await self._store_evolution_state(evolution_id, generation)
        
        # Coordinate worker synchronization
        await self._sync_worker_populations(evolution_id)
        
        # Monitor progress
        progress = await self._monitor_evolution_progress(evolution_id, generation)
        
        return progress
```

This comprehensive API integration enables full programmatic control of Neon resources for Phase 4 distributed genetic algorithm trading systems, ensuring automated provisioning, scaling, and coordination of database resources across Ray worker clusters.