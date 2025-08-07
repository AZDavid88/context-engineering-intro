# AWS EFS Docker Compose Integration - Phase 4 Ray Workers

**Source**: https://docs.aws.amazon.com/efs/latest/ug/mounting-fs-mount-cmd-general.html  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - EFS Integration for Ray Workers with Docker Compose

## EFS Mount Configuration

### Docker Compose NFS Volume Configuration

AWS EFS can be mounted as NFS volumes in Docker containers, providing shared storage across Ray workers:

```yaml
# File: docker-compose.cloud.yml - Phase 4 EFS Integration

version: '3.8'

services:
  ray-head-cloud:
    image: rayproject/ray:latest
    container_name: ray-head-cloud
    command: ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--dashboard-port=8265"]
    ports:
      - "8265:8265"  # Ray Dashboard
      - "10001:10001"  # Ray Client
    volumes:
      - type: volume
        source: efs-shared-data
        target: /data
        volume:
          nocopy: true
      - type: volume
        source: efs-model-cache
        target: /models
        volume:
          nocopy: true
    environment:
      - RAY_DISABLE_IMPORT_WARNING=1
      - RAY_DEDUP_LOGS=0
    networks:
      - ray-network
    healthcheck:
      test: ["CMD-SHELL", "ray status || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  ray-worker-cloud:
    image: rayproject/ray:latest
    command: ["ray", "start", "--address=ray-head-cloud:10001"]
    deploy:
      replicas: 4  # Scale Ray workers as needed
    volumes:
      - type: volume
        source: efs-shared-data
        target: /data
        volume:
          nocopy: true
      - type: volume
        source: efs-model-cache
        target: /models
        volume:
          nocopy: true
    environment:
      - RAY_DISABLE_IMPORT_WARNING=1
    networks:
      - ray-network
    depends_on:
      ray-head-cloud:
        condition: service_healthy

  neon-connector:
    build:
      context: .
      dockerfile: docker/Dockerfile.neon
    container_name: neon-connector
    volumes:
      - type: volume
        source: efs-shared-data
        target: /data
        volume:
          nocopy: true
    environment:
      - NEON_CONNECTION_STRING=${NEON_CONNECTION_STRING}
      - EFS_MOUNT_PATH=/data
    networks:
      - ray-network
    depends_on:
      - ray-head-cloud

volumes:
  efs-shared-data:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=${EFS_DNS_NAME},nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600"
      device: ":/"
  
  efs-model-cache:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=${EFS_DNS_NAME},nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600"
      device: ":/models"

networks:
  ray-network:
    driver: bridge
```

### Environment Configuration

```bash
# File: .env.cloud - Phase 4 Cloud Configuration

# EFS Configuration
EFS_DNS_NAME=fs-0123456789abcdef0.efs.us-east-1.amazonaws.com
EFS_MOUNT_TARGET_IP=10.0.1.100
EFS_REGION=us-east-1

# Ray Configuration
RAY_HEAD_HOST=ray-head-cloud
RAY_HEAD_PORT=10001

# Neon Configuration
NEON_CONNECTION_STRING=postgresql://user:pass@ep-xyz.us-east-1.aws.neon.tech/neondb
NEON_POOL_MIN_SIZE=10
NEON_POOL_MAX_SIZE=50

# AWS Configuration
AWS_REGION=us-east-1
AWS_AVAILABILITY_ZONE=us-east-1a
```

## EFS Performance Optimization

### Mount Options for Trading Workloads

```python
# File: src/infrastructure/efs_mount_manager.py

import subprocess
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class EFSMountConfig:
    """EFS mount configuration for Ray workers."""
    
    efs_dns_name: str
    mount_path: str
    nfs_version: str = "4.1"
    performance_mode: str = "generalPurpose"  # or "maxIO"
    throughput_mode: str = "provisioned"      # or "bursting"
    
    def get_mount_options(self) -> List[str]:
        """Get optimized mount options for trading workloads."""
        base_options = [
            f"nfsvers={self.nfs_version}",
            "rsize=1048576",    # 1MB read size for large files
            "wsize=1048576",    # 1MB write size for large files
            "hard",             # Hard mount for reliability
            "intr",             # Interruptible for better control
            "timeo=600",        # 60 second timeout
            "retrans=2",        # 2 retransmissions before error
        ]
        
        # Performance optimizations based on mode
        if self.performance_mode == "maxIO":
            base_options.extend([
                "rsize=65536",   # Smaller read size for maxIO
                "wsize=65536",   # Smaller write size for maxIO
            ])
        
        return base_options

class EFSMountManager:
    """Manage EFS mounts for Ray worker containers."""
    
    def __init__(self, mount_config: EFSMountConfig):
        self.config = mount_config
        self.logger = logging.getLogger(f"{__name__}.EFSMountManager")
        
    def validate_efs_availability(self) -> bool:
        """Validate EFS file system is available."""
        try:
            # Test DNS resolution
            result = subprocess.run(
                ["nslookup", self.config.efs_dns_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.logger.error(f"DNS lookup timeout for {self.config.efs_dns_name}")
            return False
        except Exception as e:
            self.logger.error(f"EFS availability check failed: {e}")
            return False
    
    def create_mount_point(self) -> bool:
        """Create mount point directory if it doesn't exist."""
        try:
            os.makedirs(self.config.mount_path, mode=0o755, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create mount point {self.config.mount_path}: {e}")
            return False
    
    def mount_efs(self) -> bool:
        """Mount EFS file system with optimal settings."""
        if not self.validate_efs_availability():
            return False
            
        if not self.create_mount_point():
            return False
        
        mount_options = self.config.get_mount_options()
        mount_command = [
            "mount",
            "-t", "nfs4",
            "-o", ",".join(mount_options),
            f"{self.config.efs_dns_name}:/",
            self.config.mount_path
        ]
        
        try:
            result = subprocess.run(
                mount_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully mounted EFS at {self.config.mount_path}")
                return True
            else:
                self.logger.error(f"Mount failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("EFS mount timeout")
            return False
        except Exception as e:
            self.logger.error(f"EFS mount error: {e}")
            return False
    
    def check_mount_health(self) -> Dict[str, any]:
        """Check EFS mount health and performance."""
        try:
            # Check if mount point is accessible
            test_file = os.path.join(self.config.mount_path, ".health_check")
            
            # Write test
            write_start = time.time()
            with open(test_file, "w") as f:
                f.write("health_check")
            write_time = time.time() - write_start
            
            # Read test  
            read_start = time.time()
            with open(test_file, "r") as f:
                content = f.read()
            read_time = time.time() - read_start
            
            # Cleanup
            os.remove(test_file)
            
            return {
                "status": "healthy",
                "write_time_ms": round(write_time * 1000, 2),
                "read_time_ms": round(read_time * 1000, 2),
                "mount_path": self.config.mount_path,
                "accessible": content == "health_check"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "mount_path": self.config.mount_path,
                "accessible": False
            }
```

## Ray Worker EFS Integration

### Shared Data Management

```python
# File: src/data/efs_data_manager.py

import os
import json
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

@dataclass
class EFSDataPath:
    """EFS data path configuration."""
    base_path: str
    market_data: str
    models: str
    results: str
    logs: str
    
    @classmethod
    def from_mount_path(cls, mount_path: str) -> 'EFSDataPath':
        """Create EFS data paths from mount point."""
        return cls(
            base_path=mount_path,
            market_data=os.path.join(mount_path, "market_data"),
            models=os.path.join(mount_path, "models"), 
            results=os.path.join(mount_path, "results"),
            logs=os.path.join(mount_path, "logs")
        )

class EFSDataManager:
    """Manage shared data storage on EFS for Ray workers."""
    
    def __init__(self, efs_mount_path: str):
        self.mount_path = efs_mount_path
        self.paths = EFSDataPath.from_mount_path(efs_mount_path)
        self.logger = logging.getLogger(f"{__name__}.EFSDataManager")
        
    async def initialize_efs_structure(self):
        """Initialize EFS directory structure for trading system."""
        directories = [
            self.paths.market_data,
            self.paths.models,
            self.paths.results,
            self.paths.logs,
            os.path.join(self.paths.market_data, "ohlcv"),
            os.path.join(self.paths.market_data, "indicators"),
            os.path.join(self.paths.models, "trained"),
            os.path.join(self.paths.models, "checkpoints"),
            os.path.join(self.paths.results, "backtests"),
            os.path.join(self.paths.results, "performance"),
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, mode=0o755, exist_ok=True)
                self.logger.info(f"Created EFS directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    async def save_market_data(self, symbol: str, timeframe: str, data: Dict[str, Any]):
        """Save market data to EFS shared storage."""
        file_path = os.path.join(
            self.paths.market_data, 
            "ohlcv", 
            f"{symbol}_{timeframe}_{int(time.time())}.json"
        )
        
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Saved market data to EFS: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save market data to EFS: {e}")
            raise
    
    async def load_market_data(self, symbol: str, timeframe: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load market data from EFS shared storage."""
        data_dir = os.path.join(self.paths.market_data, "ohlcv")
        pattern = f"{symbol}_{timeframe}_"
        
        try:
            matching_files = [
                f for f in os.listdir(data_dir) 
                if f.startswith(pattern) and f.endswith(".json")
            ]
            
            # Sort by timestamp (newest first)
            matching_files.sort(reverse=True)
            
            market_data = []
            files_loaded = 0
            
            for file_name in matching_files:
                if files_loaded >= limit:
                    break
                    
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    market_data.append(data)
                    files_loaded += 1
            
            self.logger.info(f"Loaded {len(market_data)} market data files from EFS")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to load market data from EFS: {e}")
            return []
    
    async def save_model_checkpoint(self, model_id: str, checkpoint_data: bytes, metadata: Dict[str, Any]):
        """Save model checkpoint to EFS."""
        checkpoint_dir = os.path.join(self.paths.models, "checkpoints", model_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save binary checkpoint data
        checkpoint_path = os.path.join(checkpoint_dir, "model.pkl")
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        
        try:
            # Save model data
            with open(checkpoint_path, "wb") as f:
                f.write(checkpoint_data)
            
            # Save metadata
            metadata["saved_at"] = time.time()
            metadata["checkpoint_path"] = checkpoint_path
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Saved model checkpoint to EFS: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model checkpoint: {e}")
            raise
    
    async def load_model_checkpoint(self, model_id: str) -> Optional[tuple]:
        """Load model checkpoint from EFS."""
        checkpoint_dir = os.path.join(self.paths.models, "checkpoints", model_id)
        checkpoint_path = os.path.join(checkpoint_dir, "model.pkl")
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        
        if not os.path.exists(checkpoint_path) or not os.path.exists(metadata_path):
            self.logger.warning(f"Model checkpoint not found: {model_id}")
            return None
        
        try:
            # Load checkpoint data
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = f.read()
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            self.logger.info(f"Loaded model checkpoint from EFS: {model_id}")
            return (checkpoint_data, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to load model checkpoint: {e}")
            return None
    
    async def save_backtest_results(self, strategy_id: str, results: Dict[str, Any]):
        """Save backtest results to EFS."""
        results_file = os.path.join(
            self.paths.results,
            "backtests", 
            f"{strategy_id}_{int(time.time())}.json"
        )
        
        try:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved backtest results to EFS: {results_file}")
            return results_file
            
        except Exception as e:
            self.logger.error(f"Failed to save backtest results: {e}")
            raise
    
    async def cleanup_old_data(self, days_to_keep: int = 7):
        """Cleanup old data from EFS to manage storage costs."""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        cleanup_dirs = [
            os.path.join(self.paths.market_data, "ohlcv"),
            os.path.join(self.paths.results, "backtests"),
            self.paths.logs
        ]
        
        files_removed = 0
        bytes_freed = 0
        
        for cleanup_dir in cleanup_dirs:
            if not os.path.exists(cleanup_dir):
                continue
                
            try:
                for file_name in os.listdir(cleanup_dir):
                    file_path = os.path.join(cleanup_dir, file_name)
                    
                    if os.path.isfile(file_path):
                        file_stat = os.stat(file_path)
                        
                        # Remove files older than cutoff
                        if file_stat.st_mtime < cutoff_time:
                            file_size = file_stat.st_size
                            os.remove(file_path)
                            
                            files_removed += 1
                            bytes_freed += file_size
                            
            except Exception as e:
                self.logger.error(f"Cleanup error in {cleanup_dir}: {e}")
        
        self.logger.info(f"EFS cleanup complete: {files_removed} files removed, {bytes_freed / 1024 / 1024:.2f} MB freed")
```

## Docker Compose Production Configuration

### Multi-Environment Setup

```yaml
# File: docker-compose.production.yml - Full Production EFS Setup

version: '3.8'

services:
  # Ray Head with EFS Storage
  ray-head:
    image: rayproject/ray:2.9.0
    container_name: ray-head-production
    restart: unless-stopped
    command: >
      bash -c "
      ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=0 &&
      python -c 'import time; time.sleep(99999)'
      "
    ports:
      - "8265:8265"
      - "10001:10001"
    volumes:
      - efs-shared-data:/data:rw
      - efs-model-cache:/models:rw
      - efs-logs:/logs:rw
      - ./src:/app/src:ro
    environment:
      - RAY_DISABLE_IMPORT_WARNING=1
      - RAY_ADDRESS=0.0.0.0:10001
      - PYTHONPATH=/app/src
    networks:
      - trading-network
    healthcheck:
      test: ["CMD-SHELL", "python -c 'import ray; ray.init(\"ray://localhost:10001\"); print(ray.cluster_resources())'"]
      interval: 30s
      timeout: 15s
      retries: 5
      start_period: 60s

  # Ray Workers with EFS Storage
  ray-worker:
    image: rayproject/ray:2.9.0
    restart: unless-stopped
    command: >
      bash -c "
      ray start --address=ray-head:10001 &&
      python -c 'import time; time.sleep(99999)'
      "
    deploy:
      replicas: 8  # Scale based on workload
    volumes:
      - efs-shared-data:/data:rw
      - efs-model-cache:/models:rw
      - efs-logs:/logs:rw
      - ./src:/app/src:ro
    environment:
      - RAY_DISABLE_IMPORT_WARNING=1
      - PYTHONPATH=/app/src
    networks:
      - trading-network
    depends_on:
      ray-head:
        condition: service_healthy

  # Neon Database Connector
  neon-connector:
    build:
      context: .
      dockerfile: docker/Dockerfile.neon
    container_name: neon-connector
    restart: unless-stopped
    volumes:
      - efs-shared-data:/data:rw
      - efs-logs:/logs:rw
    environment:
      - NEON_CONNECTION_STRING=${NEON_CONNECTION_STRING}
      - NEON_POOL_MIN_SIZE=${NEON_POOL_MIN_SIZE:-10}
      - NEON_POOL_MAX_SIZE=${NEON_POOL_MAX_SIZE:-50}
      - EFS_MOUNT_PATH=/data
    networks:
      - trading-network
    depends_on:
      - ray-head

  # EFS Health Monitor
  efs-monitor:
    image: alpine:latest
    container_name: efs-health-monitor
    restart: unless-stopped
    command: >
      sh -c "
      while true; do
        echo 'EFS Health Check:' && 
        df -h /data /models /logs &&
        ls -la /data /models /logs &&
        sleep 60
      done
      "
    volumes:
      - efs-shared-data:/data:ro
      - efs-model-cache:/models:ro
      - efs-logs:/logs:ro
    networks:
      - trading-network

volumes:
  efs-shared-data:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=${EFS_DNS_NAME},nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600,retrans=2"
      device: ":/data"
  
  efs-model-cache:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=${EFS_DNS_NAME},nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600,retrans=2"
      device: ":/models"
  
  efs-logs:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=${EFS_DNS_NAME},nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600,retrans=2"
      device: ":/logs"

networks:
  trading-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Performance Monitoring

### EFS Performance Metrics

```python
# File: src/monitoring/efs_performance_monitor.py

import os
import time
import psutil
import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EFSPerformanceMetrics:
    """EFS performance metrics."""
    mount_path: str
    read_latency_ms: float
    write_latency_ms: float
    disk_usage_gb: float
    disk_free_gb: float
    read_throughput_mbps: float
    write_throughput_mbps: float
    timestamp: datetime

class EFSPerformanceMonitor:
    """Monitor EFS performance for Ray workers."""
    
    def __init__(self, mount_paths: List[str]):
        self.mount_paths = mount_paths
        self.logger = logging.getLogger(f"{__name__}.EFSPerformanceMonitor")
        
    async def collect_efs_metrics(self) -> List[EFSPerformanceMetrics]:
        """Collect performance metrics for all EFS mount points."""
        metrics = []
        
        for mount_path in self.mount_paths:
            try:
                metric = await self._collect_mount_metrics(mount_path)
                metrics.append(metric)
            except Exception as e:
                self.logger.error(f"Failed to collect metrics for {mount_path}: {e}")
        
        return metrics
    
    async def _collect_mount_metrics(self, mount_path: str) -> EFSPerformanceMetrics:
        """Collect metrics for a specific mount point."""
        # Disk usage
        disk_usage = psutil.disk_usage(mount_path)
        disk_usage_gb = disk_usage.used / (1024**3)
        disk_free_gb = disk_usage.free / (1024**3)
        
        # Performance tests
        read_latency = await self._measure_read_latency(mount_path)
        write_latency = await self._measure_write_latency(mount_path)
        read_throughput = await self._measure_read_throughput(mount_path)
        write_throughput = await self._measure_write_throughput(mount_path)
        
        return EFSPerformanceMetrics(
            mount_path=mount_path,
            read_latency_ms=read_latency,
            write_latency_ms=write_latency,
            disk_usage_gb=disk_usage_gb,
            disk_free_gb=disk_free_gb,
            read_throughput_mbps=read_throughput,
            write_throughput_mbps=write_throughput,
            timestamp=datetime.utcnow()
        )
    
    async def _measure_read_latency(self, mount_path: str) -> float:
        """Measure read latency in milliseconds."""
        test_file = os.path.join(mount_path, ".read_latency_test")
        
        # Create test file if it doesn't exist
        if not os.path.exists(test_file):
            with open(test_file, "w") as f:
                f.write("latency test data")
        
        # Measure read latency
        start_time = time.perf_counter()
        with open(test_file, "r") as f:
            _ = f.read()
        end_time = time.perf_counter()
        
        return (end_time - start_time) * 1000
    
    async def _measure_write_latency(self, mount_path: str) -> float:
        """Measure write latency in milliseconds."""
        test_file = os.path.join(mount_path, ".write_latency_test")
        
        start_time = time.perf_counter()
        with open(test_file, "w") as f:
            f.write(f"write test {time.time()}")
        end_time = time.perf_counter()
        
        # Cleanup
        try:
            os.remove(test_file)
        except:
            pass
        
        return (end_time - start_time) * 1000
    
    # Additional throughput measurement methods...
```

## Implementation Priority

1. **✅ CRITICAL**: Docker Compose EFS volume configuration
2. **✅ HIGH**: EFS mount management and health checks
3. **✅ HIGH**: Shared data management for Ray workers
4. **✅ MEDIUM**: Performance monitoring and optimization
5. **✅ LOW**: Cost optimization and cleanup procedures

## Next Steps

1. Configure EFS file system in AWS with proper security groups
2. Implement Docker Compose configuration with NFS volumes
3. Set up shared data management for Ray worker coordination
4. Deploy performance monitoring for EFS mount points
5. Test failover and recovery scenarios

**EFS Integration Success Criteria**: Reliable shared storage across Ray workers, optimal performance for trading workloads, proper monitoring and alerting, seamless integration with Neon database transition.