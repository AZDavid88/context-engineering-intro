# Phase 5B: Production Deployment Infrastructure

**Generated**: 2025-08-10  
**Author**: Daedalus Watt - Performance Optimization Architect  
**Priority**: P0 - CRITICAL  
**Timeline**: 5 Days  
**Status**: PENDING

## Executive Summary

The system lacks concrete production deployment implementation despite having abstract interfaces. This phase implements AWS ECS deployment, Kubernetes manifests, CI/CD pipeline, and production monitoring to enable actual production trading operations.

## Problem Analysis

### Current State
- **Abstract DeploymentInterface** without concrete implementations
- **Docker Compose** for development only
- **No CI/CD pipeline** for automated deployment
- **Missing production monitoring** setup
- **No deployment scripts** or automation

### Root Causes
1. Focus on development without production planning
2. Abstract interfaces created without implementation
3. No DevOps integration strategy
4. Missing production requirements gathering

### Business Impact
- **Cannot deploy to production** despite "production-ready" code
- **No automated testing** in deployment pipeline
- **No monitoring** for production issues
- **Manual deployment** prone to errors
- **No rollback strategy** for failed deployments

## Implementation Architecture

### Day 1-2: Anyscale Deployment Implementation (Preferred for Potato Laptop)

#### 1.1 Concrete Anyscale Deployment Manager
```python
# File: infrastructure/anyscale/anyscale_deployment_manager.py
import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import anyscale
from anyscale.sdk import AnyscaleSDK
import ray

from infrastructure.core.deployment_interface import (
    DeploymentManager, DeploymentResult, GeneticPoolConfig,
    ClusterHealth, ScalingResult, PlatformType
)

class AnyscaleDeploymentManager(DeploymentManager):
    """
    Concrete Anyscale implementation for cloud deployment.
    Perfect for potato laptops - runs compute in the cloud!
    
    This addresses the missing concrete implementation:
    - Actually deploys to Anyscale cloud
    - Manages Ray clusters remotely
    - No local compute required
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anyscale deployment manager.
        
        Args:
            api_key: Anyscale API key (or set ANYSCALE_API_KEY env var)
        """
        super().__init__(PlatformType.ANYSCALE, {})
        
        # Initialize Anyscale client
        self.api_key = api_key or os.getenv('ANYSCALE_API_KEY')
        if not self.api_key:
            raise ValueError("Anyscale API key required")
        
        self.client = AnyscaleSDK(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        
    async def deploy_genetic_pool(self, config: GeneticPoolConfig) -> DeploymentResult:
        """
        Deploy genetic algorithm pool to Anyscale cloud.
        
        This is the ACTUAL IMPLEMENTATION that was missing.
        Your laptop just orchestrates - all compute happens in cloud.
        """
        self.logger.info("Starting Anyscale deployment...")
        
        try:
            # Create compute configuration
            compute_config = self._create_compute_config(config)
            
            # Create cluster configuration
            cluster_config = {
                "name": f"genetic-pool-{config.population_size}",
                "compute_config": compute_config,
                "ray_version": "2.9.0",
                "idle_timeout_minutes": 15,
                "cloud_id": "aws",  # or "gcp", "azure"
                "region": "us-west-2"
            }
            
            # Create Anyscale cluster
            self.logger.info("Creating Anyscale cluster...")
            cluster_response = self.client.create_cluster(cluster_config)
            cluster_id = cluster_response['id']
            
            # Wait for cluster to be ready
            await self._wait_for_cluster_ready(cluster_id)
            
            # Deploy genetic algorithm application
            app_config = self._create_app_config(config)
            app_response = self.client.deploy_app(
                cluster_id=cluster_id,
                config=app_config
            )
            
            # Get cluster endpoint
            cluster_info = self.client.get_cluster(cluster_id)
            ray_address = cluster_info['ray_address']
            
            # Connect to Ray cluster
            ray.init(address=ray_address)
            
            # Calculate costs
            estimated_cost = self._calculate_estimated_cost(compute_config)
            
            return DeploymentResult(
                deployment_id=app_response['deployment_id'],
                cluster_id=cluster_id,
                platform=PlatformType.ANYSCALE,
                status=DeploymentStatus.DEPLOYED,
                endpoint_url=ray_address,
                estimated_cost_per_hour=estimated_cost,
                total_nodes=config.min_nodes,
                genetic_pool_endpoint=f"ray://{ray_address}",
                monitoring_dashboard_url=f"https://console.anyscale.com/clusters/{cluster_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Anyscale deployment failed: {e}")
            raise DeploymentError(f"Failed to deploy to Anyscale: {e}")
    
    def _create_compute_config(self, config: GeneticPoolConfig) -> Dict:
        """Create Anyscale compute configuration."""
        # Choose instance types based on workload
        if config.node_type == NodeType.CPU_WORKER:
            instance_type = "m5.2xlarge"  # 8 vCPU, 32 GB RAM
        elif config.node_type == NodeType.GPU_WORKER:
            instance_type = "g4dn.xlarge"  # GPU instance
        else:
            instance_type = "m5.xlarge"  # 4 vCPU, 16 GB RAM
        
        return {
            "cloud_id": "aws",
            "region": "us-west-2",
            "head_node_type": {
                "instance_type": "m5.large",
                "min_workers": 0,
                "max_workers": 0
            },
            "worker_node_types": [
                {
                    "name": "genetic-worker",
                    "instance_type": instance_type,
                    "min_workers": config.min_nodes,
                    "max_workers": config.max_nodes,
                    "use_spot": config.use_spot_instances
                }
            ]
        }
    
    def _create_app_config(self, config: GeneticPoolConfig) -> Dict:
        """Create application deployment configuration."""
        return {
            "name": "genetic-trading-system",
            "working_dir": ".",
            "pip": [
                "pandas", "numpy", "deap", "vectorbt",
                "asyncio", "aiohttp", "pydantic"
            ],
            "runtime_env": {
                "env_vars": config.environment_variables,
                "working_dir": "."
            },
            "entrypoint": "python -m src.execution.genetic_strategy_pool",
            "num_replicas": 1,
            "ray_actor_options": {
                "num_cpus": 2,
                "memory": 4000 * 1024 * 1024  # 4GB
            }
        }
    
    async def _wait_for_cluster_ready(self, cluster_id: str, timeout: int = 600):
        """Wait for cluster to be ready."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            cluster = self.client.get_cluster(cluster_id)
            
            if cluster['state'] == 'RUNNING':
                self.logger.info(f"Cluster {cluster_id} is ready")
                return
            elif cluster['state'] in ['TERMINATED', 'FAILED']:
                raise DeploymentError(f"Cluster failed to start: {cluster['state']}")
            
            await asyncio.sleep(10)
        
        raise DeploymentError("Cluster startup timeout")
    
    def _calculate_estimated_cost(self, compute_config: Dict) -> float:
        """Calculate estimated hourly cost."""
        # Rough cost estimates (update with actual Anyscale pricing)
        instance_costs = {
            "m5.large": 0.096,
            "m5.xlarge": 0.192,
            "m5.2xlarge": 0.384,
            "g4dn.xlarge": 0.526
        }
        
        total_cost = 0.0
        
        # Head node cost
        head_type = compute_config['head_node_type']['instance_type']
        total_cost += instance_costs.get(head_type, 0.1)
        
        # Worker nodes cost
        for worker_type in compute_config['worker_node_types']:
            instance_type = worker_type['instance_type']
            min_workers = worker_type['min_workers']
            cost_per_instance = instance_costs.get(instance_type, 0.2)
            
            if worker_type.get('use_spot'):
                cost_per_instance *= 0.3  # Spot discount
            
            total_cost += cost_per_instance * min_workers
        
        # Add Anyscale platform fee (estimated 20%)
        total_cost *= 1.2
        
        return total_cost
    
    async def scale_cluster(self, cluster_id: str, target_nodes: int) -> ScalingResult:
        """Scale Anyscale cluster up or down."""
        try:
            current_cluster = self.client.get_cluster(cluster_id)
            current_nodes = current_cluster['num_workers']
            
            # Update cluster configuration
            self.client.update_cluster(
                cluster_id=cluster_id,
                min_workers=target_nodes,
                max_workers=target_nodes * 2  # Allow autoscaling headroom
            )
            
            # Wait for scaling to complete
            await self._wait_for_scaling(cluster_id, target_nodes)
            
            # Calculate cost impact
            new_cluster = self.client.get_cluster(cluster_id)
            new_cost = self._calculate_cluster_cost(new_cluster)
            old_cost = self._calculate_cluster_cost(current_cluster)
            
            return ScalingResult(
                success=True,
                previous_node_count=current_nodes,
                new_node_count=target_nodes,
                scaling_duration=30.0,  # Estimated
                cost_impact=new_cost - old_cost,
                message=f"Scaled from {current_nodes} to {target_nodes} nodes"
            )
            
        except Exception as e:
            return ScalingResult(
                success=False,
                previous_node_count=0,
                new_node_count=0,
                scaling_duration=0,
                cost_impact=0,
                message=f"Scaling failed: {e}"
            )
    
    async def get_cluster_health(self, cluster_id: str) -> ClusterHealth:
        """Get Anyscale cluster health metrics."""
        cluster = self.client.get_cluster(cluster_id)
        metrics = self.client.get_cluster_metrics(cluster_id)
        
        return ClusterHealth(
            overall_status=cluster['state'],
            healthy_nodes=metrics.get('healthy_nodes', 0),
            total_nodes=metrics.get('total_nodes', 0),
            cpu_utilization=metrics.get('cpu_utilization', 0.0),
            memory_utilization=metrics.get('memory_utilization', 0.0),
            active_genetic_evaluations=metrics.get('active_tasks', 0),
            failed_evaluations_last_hour=metrics.get('failed_tasks', 0),
            cost_per_hour=self._calculate_cluster_cost(cluster)
        )
    
    async def terminate_deployment(self, deployment_id: str, preserve_data: bool = True) -> bool:
        """Terminate Anyscale deployment."""
        try:
            # Get deployment info
            deployment = self.client.get_deployment(deployment_id)
            cluster_id = deployment['cluster_id']
            
            if preserve_data:
                # Save results to cloud storage
                await self._save_deployment_data(deployment_id)
            
            # Terminate application
            self.client.terminate_app(deployment_id)
            
            # Terminate cluster
            self.client.terminate_cluster(cluster_id)
            
            self.logger.info(f"Terminated deployment {deployment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to terminate deployment: {e}")
            return False
```

#### 1.2 Anyscale Configuration and Setup
```python
# File: infrastructure/anyscale/anyscale_setup.py
import os
import yaml
from pathlib import Path
from typing import Dict, Optional

class AnyscaleSetup:
    """
    Setup and configuration for Anyscale deployment.
    Handles API keys, project setup, and configuration.
    """
    
    @staticmethod
    def setup_anyscale_project() -> Dict:
        """
        Set up Anyscale project for genetic trading system.
        Run this once to configure your Anyscale account.
        """
        config = {
            "project_name": "genetic-trading-system",
            "cloud_provider": "aws",  # or "gcp", "azure"
            "region": "us-west-2",
            "cluster_compute": {
                "head_node_type": "m5.large",
                "worker_node_type": "m5.2xlarge",
                "min_workers": 2,
                "max_workers": 10,
                "use_spot_instances": True,
                "spot_max_price": 0.15  # Max price for spot instances
            },
            "ray_config": {
                "ray_version": "2.9.0",
                "dashboard": True,
                "metrics_export_port": 8080
            },
            "autoscaling": {
                "enabled": True,
                "target_cpu_utilization": 70,
                "scale_up_rate": 1.5,
                "scale_down_rate": 0.5
            },
            "cost_controls": {
                "max_hourly_cost": 10.0,
                "alert_threshold": 8.0,
                "auto_shutdown_hours": 4
            }
        }
        
        # Save configuration
        config_path = Path("infrastructure/anyscale/config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print("Anyscale configuration saved to infrastructure/anyscale/config.yaml")
        print("\nNext steps:")
        print("1. Set your Anyscale API key: export ANYSCALE_API_KEY='your-key'")
        print("2. Install Anyscale CLI: pip install anyscale")
        print("3. Login: anyscale login")
        print("4. Create project: anyscale project create genetic-trading-system")
        
        return config
    
    @staticmethod
    def create_ray_runtime_env() -> Dict:
        """Create Ray runtime environment for Anyscale."""
        return {
            "working_dir": ".",
            "pip": [
                "pandas==2.0.3",
                "numpy==1.24.3",
                "deap==1.4.1",
                "vectorbt==0.26.0",
                "aiohttp==3.9.0",
                "pydantic==2.5.0",
                "asyncpg==0.29.0",
                "orjson==3.9.10"
            ],
            "env_vars": {
                "PYTHONPATH": ".",
                "RAY_DEDUP_LOGS": "0",
                "GENETIC_MODE": "distributed"
            },
            "excludes": [
                "*.pyc",
                "__pycache__",
                ".git",
                "tests/",
                "docs/"
            ]
        }
```

### Alternative Day 1-2: AWS ECS Deployment (If Anyscale Not Available)

#### 1.1 ECS Task Definitions
```python
# File: infrastructure/aws/ecs_deployment.py
import boto3
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class ECSConfig:
    """ECS deployment configuration."""
    cluster_name: str = "quant-trading-cluster"
    service_name: str = "genetic-trading-service"
    task_family: str = "genetic-trading-task"
    cpu: str = "4096"  # 4 vCPU
    memory: str = "8192"  # 8 GB
    desired_count: int = 2
    region: str = "us-east-1"

class ECSDeploymentManager:
    """Concrete implementation for AWS ECS deployment."""
    
    def __init__(self, config: ECSConfig):
        self.config = config
        self.ecs_client = boto3.client('ecs', region_name=config.region)
        self.ec2_client = boto3.client('ec2', region_name=config.region)
        self.logger = logging.getLogger(__name__)
    
    def create_task_definition(self) -> str:
        """Create ECS task definition for trading system."""
        task_def = {
            'family': self.config.task_family,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': self.config.cpu,
            'memory': self.config.memory,
            'containerDefinitions': [
                {
                    'name': 'trading-system',
                    'image': 'YOUR_ECR_REPO/genetic-trading:latest',
                    'essential': True,
                    'portMappings': [
                        {'containerPort': 8000, 'protocol': 'tcp'},
                        {'containerPort': 8265, 'protocol': 'tcp'}  # Ray dashboard
                    ],
                    'environment': [
                        {'name': 'ENV', 'value': 'production'},
                        {'name': 'RAY_HEAD_NODE_HOST', 'value': '0.0.0.0'},
                        {'name': 'GENETIC_ALGORITHM_MODE', 'value': 'distributed'}
                    ],
                    'secrets': [
                        {'name': 'HYPERLIQUID_API_KEY', 'valueFrom': 'arn:aws:secretsmanager:region:account:secret:hyperliquid-api'},
                        {'name': 'DATABASE_URL', 'valueFrom': 'arn:aws:secretsmanager:region:account:secret:neon-db-url'}
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': '/ecs/genetic-trading',
                            'awslogs-region': self.config.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    },
                    'healthCheck': {
                        'command': ['CMD-SHELL', 'python /app/health_check.py'],
                        'interval': 30,
                        'timeout': 10,
                        'retries': 3,
                        'startPeriod': 60
                    }
                },
                {
                    'name': 'ray-worker',
                    'image': 'YOUR_ECR_REPO/genetic-trading:latest',
                    'essential': False,
                    'command': ['worker'],
                    'environment': [
                        {'name': 'RAY_HEAD_ADDRESS', 'value': 'localhost:10001'}
                    ]
                }
            ]
        }
        
        response = self.ecs_client.register_task_definition(**task_def)
        return response['taskDefinition']['taskDefinitionArn']
    
    def create_service(self, task_definition_arn: str) -> str:
        """Create ECS service for continuous operation."""
        service_config = {
            'cluster': self.config.cluster_name,
            'serviceName': self.config.service_name,
            'taskDefinition': task_definition_arn,
            'desiredCount': self.config.desired_count,
            'launchType': 'FARGATE',
            'networkConfiguration': {
                'awsvpcConfiguration': {
                    'subnets': self._get_subnet_ids(),
                    'securityGroups': self._get_security_groups(),
                    'assignPublicIp': 'ENABLED'
                }
            },
            'healthCheckGracePeriodSeconds': 60,
            'deploymentConfiguration': {
                'maximumPercent': 200,
                'minimumHealthyPercent': 100,
                'deploymentCircuitBreaker': {
                    'enable': True,
                    'rollback': True
                }
            },
            'enableECSManagedTags': True,
            'propagateTags': 'SERVICE'
        }
        
        response = self.ecs_client.create_service(**service_config)
        return response['service']['serviceArn']
    
    def deploy(self) -> Dict[str, str]:
        """Full deployment pipeline."""
        self.logger.info("Starting ECS deployment...")
        
        # Create cluster if not exists
        self._ensure_cluster_exists()
        
        # Register task definition
        task_arn = self.create_task_definition()
        self.logger.info(f"Task definition created: {task_arn}")
        
        # Create or update service
        service_arn = self.create_service(task_arn)
        self.logger.info(f"Service created: {service_arn}")
        
        return {
            'cluster': self.config.cluster_name,
            'service': service_arn,
            'task_definition': task_arn,
            'status': 'deployed'
        }
```

#### 1.2 Auto-scaling Configuration
```yaml
# File: infrastructure/aws/auto_scaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genetic-trading-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genetic-trading
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Day 3: Kubernetes Deployment Manifests

#### 3.1 Kubernetes Deployment
```yaml
# File: infrastructure/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genetic-trading
  namespace: production
  labels:
    app: genetic-trading
    component: trading-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genetic-trading
  template:
    metadata:
      labels:
        app: genetic-trading
    spec:
      containers:
      - name: trading-system
        image: YOUR_REGISTRY/genetic-trading:latest
        ports:
        - containerPort: 8000
          name: api
        - containerPort: 8265
          name: ray-dashboard
        env:
        - name: ENV
          value: "production"
        - name: RAY_HEAD_NODE_HOST
          value: "0.0.0.0"
        envFrom:
        - secretRef:
            name: trading-secrets
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: trading-config
---
apiVersion: v1
kind: Service
metadata:
  name: genetic-trading-service
  namespace: production
spec:
  selector:
    app: genetic-trading
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: ray
    port: 8265
    targetPort: 8265
  type: LoadBalancer
```

#### 3.2 ConfigMap and Secrets
```yaml
# File: infrastructure/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-config
  namespace: production
data:
  settings.yaml: |
    trading:
      mode: production
      max_positions: 10
      risk_limit: 0.02
    genetic_algorithm:
      population_size: 200
      generations: 50
      mutation_rate: 0.1
    monitoring:
      enabled: true
      interval: 30
---
apiVersion: v1
kind: Secret
metadata:
  name: trading-secrets
  namespace: production
type: Opaque
stringData:
  HYPERLIQUID_API_KEY: "your-encrypted-key"
  DATABASE_URL: "postgresql://user:pass@host/db"
  NEON_API_KEY: "your-neon-key"
```

### Day 4: GitHub Actions CI/CD Pipeline

#### 4.1 CI/CD Workflow
```yaml
# File: .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: genetic-trading
  ECS_CLUSTER: quant-trading-cluster
  ECS_SERVICE: genetic-trading-service

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service \
          --cluster ${{ env.ECS_CLUSTER }} \
          --service ${{ env.ECS_SERVICE }} \
          --force-new-deployment \
          --region ${{ env.AWS_REGION }}
    
    - name: Wait for deployment
      run: |
        aws ecs wait services-stable \
          --cluster ${{ env.ECS_CLUSTER }} \
          --services ${{ env.ECS_SERVICE }} \
          --region ${{ env.AWS_REGION }}
    
    - name: Notify deployment status
      if: always()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Deployment to production ${{ job.status }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

#### 4.2 Rollback Workflow
```yaml
# File: .github/workflows/rollback.yml
name: Rollback Production

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to rollback to'
        required: true

jobs:
  rollback:
    runs-on: ubuntu-latest
    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Rollback ECS service
      run: |
        # Update task definition to previous version
        aws ecs update-service \
          --cluster quant-trading-cluster \
          --service genetic-trading-service \
          --task-definition genetic-trading-task:${{ github.event.inputs.version }} \
          --force-new-deployment
```

### Day 5: Production Monitoring Setup

#### 5.1 Prometheus Configuration
```yaml
# File: infrastructure/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'genetic-trading'
    static_configs:
    - targets: ['localhost:8000']
    metrics_path: '/metrics'
    
  - job_name: 'ray-cluster'
    static_configs:
    - targets: ['localhost:8265']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
    - targets: ['localhost:9100']

rule_files:
  - '/etc/prometheus/alerts.yml'

alerting:
  alertmanagers:
  - static_configs:
    - targets: ['localhost:9093']
```

#### 5.2 Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Genetic Trading System",
    "panels": [
      {
        "title": "Strategy Performance",
        "targets": [
          {
            "expr": "rate(strategy_evaluations_total[5m])",
            "legendFormat": "Evaluations/sec"
          },
          {
            "expr": "strategy_fitness_score",
            "legendFormat": "Fitness Score"
          }
        ]
      },
      {
        "title": "System Health",
        "targets": [
          {
            "expr": "up{job='genetic-trading'}",
            "legendFormat": "Service Status"
          },
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Trading Metrics",
        "targets": [
          {
            "expr": "trading_positions_open",
            "legendFormat": "Open Positions"
          },
          {
            "expr": "trading_pnl_total",
            "legendFormat": "Total P&L"
          }
        ]
      }
    ]
  }
}
```

#### 5.3 Application Metrics Integration
```python
# File: src/monitoring/metrics_exporter.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

class MetricsExporter:
    """Export metrics for Prometheus monitoring."""
    
    def __init__(self):
        # Strategy metrics
        self.strategy_evaluations = Counter(
            'strategy_evaluations_total',
            'Total strategy evaluations'
        )
        self.fitness_score = Gauge(
            'strategy_fitness_score',
            'Current best fitness score'
        )
        
        # Trading metrics
        self.positions_open = Gauge(
            'trading_positions_open',
            'Number of open positions'
        )
        self.pnl_total = Gauge(
            'trading_pnl_total',
            'Total P&L'
        )
        
        # Performance metrics
        self.evaluation_duration = Histogram(
            'strategy_evaluation_duration_seconds',
            'Time spent evaluating strategies'
        )
        
    def start_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        start_http_server(port)
    
    def record_evaluation(self, fitness: float, duration: float):
        """Record strategy evaluation metrics."""
        self.strategy_evaluations.inc()
        self.fitness_score.set(fitness)
        self.evaluation_duration.observe(duration)
    
    def update_trading_metrics(self, positions: int, pnl: float):
        """Update trading metrics."""
        self.positions_open.set(positions)
        self.pnl_total.set(pnl)
```

## Success Metrics

### Deployment Capabilities
- ✅ **One-command deployment** to AWS ECS
- ✅ **Kubernetes manifests** for cloud-agnostic deployment
- ✅ **Automated CI/CD** with GitHub Actions
- ✅ **Rollback capability** for failed deployments

### Monitoring & Observability
- ✅ **Real-time metrics** in Grafana
- ✅ **Alert rules** for critical issues
- ✅ **Application metrics** exported
- ✅ **Log aggregation** configured

### Reliability
- ✅ **Health checks** configured
- ✅ **Auto-scaling** based on load
- ✅ **Circuit breakers** for deployments
- ✅ **Blue-green deployment** support

## Risk Mitigation

### Potential Risks
1. **AWS Costs**: Uncontrolled scaling could increase costs
   - Mitigation: Cost alerts, scaling limits
   
2. **Deployment Failures**: Bad deployments affecting production
   - Mitigation: Automated rollback, canary deployments
   
3. **Security**: Exposed secrets or vulnerabilities
   - Mitigation: Secrets management, security scanning

## Validation Steps

1. **ECS Deployment Test**:
   - Deploy to staging environment
   - Verify all containers healthy
   - Test auto-scaling behavior

2. **Kubernetes Test**:
   - Deploy to local k8s cluster
   - Verify manifests work correctly
   - Test rolling updates

3. **CI/CD Pipeline Test**:
   - Push to feature branch
   - Verify tests run
   - Check deployment to staging

4. **Monitoring Test**:
   - Generate load on system
   - Verify metrics appear in Grafana
   - Test alert triggers

## Dependencies

- AWS Account with ECS configured
- Docker images built and pushed to registry
- GitHub repository with Actions enabled
- Prometheus and Grafana instances

## Next Phase

After deployment infrastructure is complete, proceed to Phase 5C (Performance Optimization) to optimize the deployed system.