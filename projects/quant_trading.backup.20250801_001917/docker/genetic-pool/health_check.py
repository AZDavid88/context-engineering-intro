#!/usr/bin/env python3
"""
Health Check Script for Genetic Algorithm Pool Container

This script performs comprehensive health checks for the genetic algorithm
container, validating Ray cluster connectivity, genetic pool functionality,
and system resources.

Health Check Categories:
- Ray cluster connectivity and status
- Genetic algorithm pool initialization
- System resource availability
- Integration component health
"""

import asyncio
import sys
import os
import logging
import time
import psutil
from typing import Dict, Any, List
from datetime import datetime, timezone

# Add project root to Python path
sys.path.insert(0, '/app')

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Configure logging for health checks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - HEALTH_CHECK - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthCheckResult:
    """Health check result container"""
    
    def __init__(self, component: str, status: str, message: str, details: Dict[str, Any] = None):
        self.component = component
        self.status = status  # "healthy", "warning", "critical"
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def is_healthy(self) -> bool:
        return self.status == "healthy"
    
    def __str__(self) -> str:
        return f"{self.component}: {self.status.upper()} - {self.message}"


class GeneticPoolHealthChecker:
    """Comprehensive health checker for genetic algorithm pool"""
    
    def __init__(self):
        self.logger = logger
        self.results: List[HealthCheckResult] = []
    
    async def run_all_checks(self) -> bool:
        """
        Run all health checks and return overall health status.
        
        Returns:
            True if all critical checks pass
        """
        try:
            # Core system checks
            await self.check_system_resources()
            await self.check_python_environment()
            
            # Ray cluster checks
            if RAY_AVAILABLE:
                await self.check_ray_cluster()
            else:
                self.results.append(HealthCheckResult(
                    "ray_availability", "critical", "Ray not available in container"
                ))
            
            # Genetic algorithm specific checks
            await self.check_genetic_pool_imports()
            await self.check_data_directories()
            await self.check_configuration()
            
            # Integration checks
            await self.check_infrastructure_integration()
            
            return self._evaluate_overall_health()
            
        except Exception as e:
            self.logger.error(f"Health check failed with exception: {e}")
            self.results.append(HealthCheckResult(
                "health_check_system", "critical", f"Health check system failure: {e}"
            ))
            return False
    
    async def check_system_resources(self):
        """Check system resource availability"""
        try:
            # Check CPU availability
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                status = "warning"
                message = f"High CPU usage: {cpu_percent}%"
            else:
                status = "healthy"
                message = f"CPU usage normal: {cpu_percent}%"
            
            self.results.append(HealthCheckResult(
                "cpu_resources", status, message, {"cpu_percent": cpu_percent}
            ))
            
            # Check memory availability
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = "critical"
                message = f"Critical memory usage: {memory_percent}%"
            elif memory_percent > 80:
                status = "warning"
                message = f"High memory usage: {memory_percent}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory_percent}%"
            
            self.results.append(HealthCheckResult(
                "memory_resources", status, message, {
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3)
                }
            ))
            
            # Check disk space
            disk = psutil.disk_usage('/app')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 90:
                status = "warning"
                message = f"Low disk space: {disk_percent:.1f}% used"
            else:
                status = "healthy"
                message = f"Disk space adequate: {disk_percent:.1f}% used"
            
            self.results.append(HealthCheckResult(
                "disk_resources", status, message, {"disk_percent": disk_percent}
            ))
            
        except Exception as e:
            self.results.append(HealthCheckResult(
                "system_resources", "critical", f"System resource check failed: {e}"
            ))
    
    async def check_python_environment(self):
        """Check Python environment and critical imports"""
        try:
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if sys.version_info >= (3, 10):
                status = "healthy"
                message = f"Python version compatible: {python_version}"
            else:
                status = "critical"
                message = f"Python version incompatible: {python_version} (requires >= 3.10)"
            
            self.results.append(HealthCheckResult(
                "python_version", status, message, {"version": python_version}
            ))
            
            # Check critical imports
            critical_imports = [
                "numpy", "pandas", "asyncio", "logging", "yaml", "json"
            ]
            
            failed_imports = []
            for module_name in critical_imports:
                try:
                    __import__(module_name)
                except ImportError:
                    failed_imports.append(module_name)
            
            if failed_imports:
                self.results.append(HealthCheckResult(
                    "python_imports", "critical", 
                    f"Critical imports failed: {failed_imports}"
                ))
            else:
                self.results.append(HealthCheckResult(
                    "python_imports", "healthy", "All critical imports available"
                ))
                
        except Exception as e:
            self.results.append(HealthCheckResult(
                "python_environment", "critical", f"Python environment check failed: {e}"
            ))
    
    async def check_ray_cluster(self):
        """Check Ray cluster connectivity and status"""
        try:
            if not RAY_AVAILABLE:
                self.results.append(HealthCheckResult(
                    "ray_cluster", "critical", "Ray not available"
                ))
                return
            
            # Check if Ray is already initialized
            if ray.is_initialized():
                # Get Ray cluster info
                try:
                    cluster_resources = ray.cluster_resources()
                    node_info = ray.nodes()
                    
                    total_cpus = cluster_resources.get("CPU", 0)
                    total_memory = cluster_resources.get("memory", 0) / (1024**3)  # Convert to GB
                    num_nodes = len([n for n in node_info if n.get("Alive", False)])
                    
                    if num_nodes > 0 and total_cpus > 0:
                        status = "healthy"
                        message = f"Ray cluster active: {num_nodes} nodes, {total_cpus} CPUs, {total_memory:.1f}GB memory"
                    else:
                        status = "warning"
                        message = "Ray cluster has limited resources"
                    
                    self.results.append(HealthCheckResult(
                        "ray_cluster", status, message, {
                            "num_nodes": num_nodes,
                            "total_cpus": total_cpus,
                            "total_memory_gb": total_memory
                        }
                    ))
                    
                except Exception as e:
                    self.results.append(HealthCheckResult(
                        "ray_cluster", "warning", f"Ray cluster info unavailable: {e}"
                    ))
                    
            else:
                # Try to connect to Ray cluster
                try:
                    ray_address = os.environ.get("RAY_ADDRESS", "auto")
                    ray.init(address=ray_address, ignore_reinit_error=True)
                    
                    if ray.is_initialized():
                        self.results.append(HealthCheckResult(
                            "ray_cluster", "healthy", f"Successfully connected to Ray cluster at {ray_address}"
                        ))
                    else:
                        self.results.append(HealthCheckResult(
                            "ray_cluster", "critical", "Failed to initialize Ray cluster"
                        ))
                        
                except Exception as e:
                    self.results.append(HealthCheckResult(
                        "ray_cluster", "critical", f"Ray cluster connection failed: {e}"
                    ))
            
        except Exception as e:
            self.results.append(HealthCheckResult(
                "ray_cluster", "critical", f"Ray cluster check failed: {e}"
            ))
    
    async def check_genetic_pool_imports(self):
        """Check genetic algorithm pool imports and basic functionality"""
        try:
            # Test critical genetic algorithm imports
            from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionMode
            from src.strategy.genetic_seeds.seed_registry import get_registry
            from src.strategy.genetic_seeds.base_seed import BaseSeed
            
            # Test seed registry functionality
            registry = get_registry()
            available_seeds = registry.get_available_seeds()
            
            if len(available_seeds) > 0:
                status = "healthy"
                message = f"Genetic pool imports successful, {len(available_seeds)} seeds available"
                details = {"available_seeds": list(available_seeds.keys())}
            else:
                status = "warning"
                message = "Genetic pool imports successful but no seeds available"
                details = {"available_seeds": []}
            
            self.results.append(HealthCheckResult(
                "genetic_pool_imports", status, message, details
            ))
            
        except ImportError as e:
            self.results.append(HealthCheckResult(
                "genetic_pool_imports", "critical", f"Genetic pool import failed: {e}"
            ))
        except Exception as e:
            self.results.append(HealthCheckResult(
                "genetic_pool_imports", "critical", f"Genetic pool check failed: {e}"
            ))
    
    async def check_data_directories(self):
        """Check required data directories and permissions"""
        required_dirs = [
            "/app/data",
            "/app/logs", 
            "/app/results",
            "/tmp/ray"
        ]
        
        missing_dirs = []
        permission_issues = []
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
            elif not os.access(dir_path, os.W_OK):
                permission_issues.append(dir_path)
        
        if missing_dirs or permission_issues:
            status = "warning"
            message = f"Directory issues - Missing: {missing_dirs}, No write access: {permission_issues}"
        else:
            status = "healthy"
            message = "All required directories available and writable"
        
        self.results.append(HealthCheckResult(
            "data_directories", status, message, {
                "missing": missing_dirs,
                "permission_issues": permission_issues
            }
        ))
    
    async def check_configuration(self):
        """Check configuration availability and validity"""
        try:
            # Check for configuration files
            config_paths = [
                "/app/config",
                "/app/infrastructure/config"
            ]
            
            config_available = any(os.path.exists(path) for path in config_paths)
            
            if config_available:
                status = "healthy"
                message = "Configuration directories available"
            else:
                status = "warning"
                message = "Configuration directories not found, using defaults"
            
            self.results.append(HealthCheckResult(
                "configuration", status, message
            ))
            
        except Exception as e:
            self.results.append(HealthCheckResult(
                "configuration", "warning", f"Configuration check failed: {e}"
            ))
    
    async def check_infrastructure_integration(self):
        """Check infrastructure integration components"""
        try:
            # Test infrastructure imports
            from infrastructure.core.deployment_interface import DeploymentManager
            from infrastructure.core.cluster_manager import ClusterManager
            from src.execution.infrastructure_manager import InfrastructureManager
            
            status = "healthy"
            message = "Infrastructure integration components available"
            
            self.results.append(HealthCheckResult(
                "infrastructure_integration", status, message
            ))
            
        except ImportError as e:
            self.results.append(HealthCheckResult(
                "infrastructure_integration", "warning", f"Infrastructure imports failed: {e}"
            ))
        except Exception as e:
            self.results.append(HealthCheckResult(
                "infrastructure_integration", "warning", f"Infrastructure check failed: {e}"
            ))
    
    def _evaluate_overall_health(self) -> bool:
        """Evaluate overall health based on individual check results"""
        critical_failures = [r for r in self.results if r.status == "critical"]
        warnings = [r for r in self.results if r.status == "warning"]
        healthy = [r for r in self.results if r.status == "healthy"]
        
        # Log summary
        self.logger.info(f"Health check summary: {len(healthy)} healthy, {len(warnings)} warnings, {len(critical_failures)} critical")
        
        # Log all results
        for result in self.results:
            if result.status == "critical":
                self.logger.error(str(result))
            elif result.status == "warning":
                self.logger.warning(str(result))
            else:
                self.logger.info(str(result))
        
        # Return False if any critical failures
        return len(critical_failures) == 0


async def main():
    """Main health check execution"""
    health_checker = GeneticPoolHealthChecker()
    
    try:
        is_healthy = await health_checker.run_all_checks()
        
        if is_healthy:
            logger.info("All health checks passed - container is healthy")
            sys.exit(0)
        else:
            logger.error("Health checks failed - container is unhealthy")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Health check execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())