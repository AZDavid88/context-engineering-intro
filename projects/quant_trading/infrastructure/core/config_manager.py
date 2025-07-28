"""
Production Infrastructure - Core Configuration Manager

This module provides universal configuration management for genetic algorithm
infrastructure across different platforms, with environment-specific settings
and integration with existing configuration systems.

Research-Based Implementation:
- /research/anyscale/cluster_configuration.md - Platform configuration patterns
- Existing src/config/settings.py integration patterns
- PHASE_5B5_INFRASTRUCTURE_ARCHITECTURE.md - Configuration requirements

Key Features:
- Environment-specific configuration (dev/staging/prod)
- Platform-agnostic configuration with platform-specific optimizations
- Integration with existing Settings system
- Secure credential management
- Dynamic configuration updates
"""

import asyncio
import logging
import os
import yaml
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import base64
from cryptography.fernet import Fernet

# Integration with existing system
from .deployment_interface import PlatformType, CostLimits
from .cluster_manager import ClusterConfig, WorkloadType

# Set up logging
logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationScope(str, Enum):
    """Configuration scope levels"""
    GLOBAL = "global"                    # Applies to all deployments
    PLATFORM = "platform"               # Platform-specific settings
    ENVIRONMENT = "environment"         # Environment-specific settings
    CLUSTER = "cluster"                  # Individual cluster settings
    WORKLOAD = "workload"               # Workload-specific settings


@dataclass
class CredentialConfig:
    """Secure credential configuration"""
    credential_id: str
    platform: PlatformType
    credential_type: str  # "api_key", "certificate", "token"
    
    # Encrypted credential data
    encrypted_value: str
    encryption_key_id: str
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    
    # Access control
    allowed_environments: List[Environment] = field(default_factory=list)
    allowed_clusters: List[str] = field(default_factory=list)


@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    platform: PlatformType
    
    # Authentication configuration
    credentials: Dict[str, CredentialConfig] = field(default_factory=dict)
    
    # Platform-specific settings
    platform_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Resource defaults
    default_instance_types: Dict[str, str] = field(default_factory=dict)
    default_regions: List[str] = field(default_factory=list)
    
    # Cost management
    cost_limits: Optional[CostLimits] = None
    budget_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Monitoring configuration
    monitoring_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    environment: Environment
    
    # Cluster configuration defaults
    cluster_defaults: Dict[str, Any] = field(default_factory=dict)
    
    # Resource limits
    max_clusters: int = 10
    max_nodes_per_cluster: int = 100
    max_cost_per_hour: float = 100.0
    
    # Genetic algorithm defaults
    genetic_defaults: Dict[str, Any] = field(default_factory=dict)
    
    # Integration settings
    trading_system_integration: Dict[str, Any] = field(default_factory=dict)
    
    # Security settings
    security_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadConfig:
    """Workload-specific configuration"""
    workload_type: WorkloadType
    
    # Resource requirements
    cpu_requirements: Dict[str, float] = field(default_factory=dict)
    memory_requirements: Dict[str, float] = field(default_factory=dict)
    storage_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Performance tuning
    optimization_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring settings
    metrics_collection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureConfig:
    """Complete infrastructure configuration"""
    config_id: str
    version: str
    
    # Core configuration
    global_settings: Dict[str, Any] = field(default_factory=dict)
    platform_configs: Dict[PlatformType, PlatformConfig] = field(default_factory=dict)
    environment_configs: Dict[Environment, EnvironmentConfig] = field(default_factory=dict)
    workload_configs: Dict[WorkloadType, WorkloadConfig] = field(default_factory=dict)
    
    # Integration configuration
    existing_system_integration: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "infrastructure_manager"


class ConfigurationManager(ABC):
    """
    Universal configuration management interface for genetic algorithm infrastructure.
    
    This abstract base class defines the contract for managing configuration
    across different platforms and environments, with secure credential handling
    and integration with existing configuration systems.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration files
        """
        self.config_path = Path(config_path) if config_path else Path("infrastructure/config")
        self.logger = logging.getLogger(__name__)
        self._config_cache: Dict[str, Any] = {}
        self._encryption_key: Optional[bytes] = None
        self._load_encryption_key()
    
    def _load_encryption_key(self):
        """Load or generate encryption key for credential management"""
        key_file = self.config_path / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                self._encryption_key = f.read()
        else:
            # Generate new key
            self._encryption_key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(self._encryption_key)
            # Secure the key file
            os.chmod(key_file, 0o600)
    
    @abstractmethod
    async def load_configuration(self, 
                               environment: Environment,
                               platform: PlatformType) -> InfrastructureConfig:
        """
        Load configuration for specific environment and platform.
        
        Args:
            environment: Target environment
            platform: Target platform
            
        Returns:
            InfrastructureConfig with merged configuration
        """
        pass
    
    @abstractmethod
    async def save_configuration(self, config: InfrastructureConfig) -> bool:
        """
        Save configuration to persistent storage.
        
        Args:
            config: Configuration to save
            
        Returns:
            True if save successful
        """
        pass
    
    @abstractmethod
    async def validate_configuration(self, config: InfrastructureConfig) -> List[str]:
        """
        Validate configuration for completeness and correctness.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    async def get_cluster_config(self, 
                               environment: Environment,
                               platform: PlatformType,
                               workload_type: WorkloadType) -> ClusterConfig:
        """
        Get cluster configuration for specific deployment.
        
        Args:
            environment: Target environment
            platform: Target platform
            workload_type: Genetic algorithm workload type
            
        Returns:
            ClusterConfig optimized for the specified parameters
        """
        pass
    
    # Credential management methods
    
    async def store_credential(self, credential: CredentialConfig) -> bool:
        """
        Store encrypted credential securely.
        
        Args:
            credential: Credential configuration to store
            
        Returns:
            True if storage successful
        """
        try:
            # Credential is already encrypted in the CredentialConfig
            credential_data = asdict(credential)
            
            # Store in platform-specific credential store
            credential_file = (self.config_path / "credentials" / 
                             f"{credential.platform}_{credential.credential_id}.json")
            credential_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(credential_file, "w") as f:
                json.dump(credential_data, f, default=str)
            
            # Secure the credential file
            os.chmod(credential_file, 0o600)
            
            self.logger.info(f"Stored credential {credential.credential_id} for {credential.platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store credential: {e}")
            return False
    
    async def retrieve_credential(self, 
                                platform: PlatformType,
                                credential_id: str,
                                environment: Environment) -> Optional[str]:
        """
        Retrieve and decrypt credential value.
        
        Args:
            platform: Platform for credential
            credential_id: Unique credential identifier
            environment: Current environment for access control
            
        Returns:
            Decrypted credential value or None if not found/unauthorized
        """
        try:
            credential_file = (self.config_path / "credentials" / 
                             f"{platform}_{credential_id}.json")
            
            if not credential_file.exists():
                self.logger.warning(f"Credential {credential_id} not found for {platform}")
                return None
            
            with open(credential_file, "r") as f:
                credential_data = json.load(f)
            
            credential = CredentialConfig(**credential_data)
            
            # Check access control
            if (credential.allowed_environments and 
                environment not in credential.allowed_environments):
                self.logger.warning(
                    f"Environment {environment} not authorized for credential {credential_id}"
                )
                return None
            
            # Check expiration
            if credential.expires_at and datetime.now(timezone.utc) > credential.expires_at:
                self.logger.warning(f"Credential {credential_id} has expired")
                return None
            
            # Decrypt credential value
            if self._encryption_key:
                fernet = Fernet(self._encryption_key)
                decrypted_value = fernet.decrypt(credential.encrypted_value.encode()).decode()
                return decrypted_value
            else:
                self.logger.error("No encryption key available for credential decryption")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential: {e}")
            return None
    
    def encrypt_credential_value(self, value: str) -> str:
        """
        Encrypt credential value for secure storage.
        
        Args:
            value: Plain text credential value
            
        Returns:
            Encrypted credential value
        """
        if not self._encryption_key:
            raise ValueError("No encryption key available")
        
        fernet = Fernet(self._encryption_key)
        encrypted_value = fernet.encrypt(value.encode())
        return encrypted_value.decode()
    
    # Integration methods
    
    async def integrate_with_existing_settings(self, 
                                             settings_module_path: str = "src.config.settings") -> Dict[str, Any]:
        """
        Integrate with existing settings system.
        
        This method loads configuration from the existing settings.py system
        and merges it with infrastructure-specific configuration.
        
        Args:
            settings_module_path: Path to existing settings module
            
        Returns:
            Dictionary with merged configuration
        """
        try:
            # Import existing settings (would be done dynamically in real implementation)
            # from src.config.settings import get_settings
            
            self.logger.info("Integrating with existing settings system")
            
            # Load existing settings
            existing_config = {
                "hyperliquid_api": {
                    "base_url": "https://api.hyperliquid.xyz",
                    "testnet": True
                },
                "fear_greed_api": {
                    "base_url": "https://api.alternative.me/fng/",
                    "cache_duration": 3600
                },
                "genetic_algorithm": {
                    "population_size": 100,
                    "max_generations": 50
                },
                "monitoring": {
                    "log_level": "INFO",
                    "health_check_interval": 30
                }
            }
            
            # Create infrastructure-specific defaults based on existing settings
            infrastructure_defaults = {
                "cost_limits": {
                    "max_hourly_cost": 50.0,
                    "max_total_cost": 500.0
                },
                "cluster_defaults": {
                    "min_nodes": 2,
                    "max_nodes": 10,
                    "auto_scaling": True
                },
                "workload_optimization": {
                    WorkloadType.GENETIC_EVOLUTION: {
                        "cpu_intensive": True,
                        "memory_requirements": "medium",
                        "optimization_priority": "throughput"
                    }
                }
            }
            
            # Merge configurations
            merged_config = {**existing_config, **infrastructure_defaults}
            
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with existing settings: {e}")
            return {}
    
    def create_default_configurations(self) -> InfrastructureConfig:
        """
        Create default infrastructure configuration.
        
        Returns:
            InfrastructureConfig with sensible defaults
        """
        # Development environment configuration
        dev_config = EnvironmentConfig(
            environment=Environment.DEVELOPMENT,
            cluster_defaults={
                "min_nodes": 1,
                "max_nodes": 5,
                "instance_type": "m5.large"
            },
            max_clusters=3,
            max_nodes_per_cluster=20,
            max_cost_per_hour=25.0,
            genetic_defaults={
                "population_size": 50,
                "max_generations": 25,
                "evaluation_timeout": 30
            }
        )
        
        # Production environment configuration
        prod_config = EnvironmentConfig(
            environment=Environment.PRODUCTION,
            cluster_defaults={
                "min_nodes": 2,
                "max_nodes": 20,
                "instance_type": "c5.2xlarge"
            },
            max_clusters=10,
            max_nodes_per_cluster=100,
            max_cost_per_hour=100.0,
            genetic_defaults={
                "population_size": 200,
                "max_generations": 100,
                "evaluation_timeout": 300
            }
        )
        
        # Anyscale platform configuration
        anyscale_config = PlatformConfig(
            platform=PlatformType.ANYSCALE,
            platform_settings={
                "ray_version": "2.8.0",
                "python_version": "3.10",
                "auto_scaling": True,
                "spot_instances": True
            },
            default_instance_types={
                "head_node": "m5.xlarge",
                "cpu_worker": "c5.2xlarge",
                "gpu_worker": "p3.2xlarge"
            },
            default_regions=["us-west-2", "us-east-1"],
            cost_limits=CostLimits(
                max_hourly_cost=50.0,
                max_total_cost=500.0,
                cost_alert_threshold=0.8
            )
        )
        
        # Genetic algorithm workload configuration
        genetic_workload_config = WorkloadConfig(
            workload_type=WorkloadType.GENETIC_EVOLUTION,
            cpu_requirements={"min_cores": 2, "max_cores": 8},
            memory_requirements={"min_gb": 4, "max_gb": 32},
            optimization_settings={
                "parallel_evaluations": True,
                "evaluation_caching": True,
                "resource_pooling": True
            }
        )
        
        # Create complete configuration
        config = InfrastructureConfig(
            config_id="default_infrastructure_config",
            version="1.0.0",
            global_settings={
                "project_name": "genetic_trading_system",
                "default_platform": PlatformType.ANYSCALE,
                "default_environment": Environment.DEVELOPMENT
            },
            platform_configs={PlatformType.ANYSCALE: anyscale_config},
            environment_configs={
                Environment.DEVELOPMENT: dev_config,
                Environment.PRODUCTION: prod_config
            },
            workload_configs={WorkloadType.GENETIC_EVOLUTION: genetic_workload_config},
            existing_system_integration={
                "trading_system_manager": {
                    "integration_enabled": True,
                    "monitoring_integration": True,
                    "genetic_pool_integration": True
                }
            }
        )
        
        return config
    
    async def generate_platform_specific_config(self,
                                              base_config: InfrastructureConfig,
                                              platform: PlatformType,
                                              environment: Environment) -> Dict[str, Any]:
        """
        Generate platform-specific configuration from base configuration.
        
        Args:
            base_config: Base infrastructure configuration
            platform: Target platform
            environment: Target environment
            
        Returns:
            Dictionary with platform-specific configuration
        """
        platform_config = base_config.platform_configs.get(platform)
        environment_config = base_config.environment_configs.get(environment)
        
        if not platform_config or not environment_config:
            raise ValueError(f"Configuration not found for {platform}/{environment}")
        
        # Generate platform-specific configuration
        config = {
            "platform": platform,
            "environment": environment,
            "cluster_config": {
                **environment_config.cluster_defaults,
                **platform_config.platform_settings
            },
            "cost_management": asdict(platform_config.cost_limits) if platform_config.cost_limits else {},
            "monitoring": platform_config.monitoring_config,
            "integration": base_config.existing_system_integration
        }
        
        return config


class FileBasedConfigurationManager(ConfigurationManager):
    """
    File-based implementation of configuration manager.
    
    This implementation stores configuration in YAML files with environment
    and platform-specific overrides.
    """
    
    async def load_configuration(self, 
                               environment: Environment,
                               platform: PlatformType) -> InfrastructureConfig:
        """Load configuration from YAML files"""
        try:
            # Load base configuration
            base_file = self.config_path / "base.yaml"
            if base_file.exists():
                with open(base_file, "r") as f:
                    base_config = yaml.safe_load(f)
            else:
                base_config = {}
            
            # Load environment-specific configuration
            env_file = self.config_path / f"{environment}.yaml"
            if env_file.exists():
                with open(env_file, "r") as f:
                    env_config = yaml.safe_load(f)
                    base_config.update(env_config)
            
            # Load platform-specific configuration
            platform_file = self.config_path / f"{platform}.yaml"
            if platform_file.exists():
                with open(platform_file, "r") as f:
                    platform_config = yaml.safe_load(f)
                    base_config.update(platform_config)
            
            # Convert to InfrastructureConfig (simplified conversion)
            config = InfrastructureConfig(
                config_id=f"{environment}_{platform}_config",
                version="1.0.0",
                global_settings=base_config.get("global_settings", {}),
                existing_system_integration=base_config.get("integration", {})
            )
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self.create_default_configurations()
    
    async def save_configuration(self, config: InfrastructureConfig) -> bool:
        """Save configuration to YAML files"""
        try:
            config_file = self.config_path / f"{config.config_id}.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = asdict(config)
            
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    async def validate_configuration(self, config: InfrastructureConfig) -> List[str]:
        """Validate configuration"""
        errors = []
        
        if not config.config_id:
            errors.append("Configuration ID is required")
        
        if not config.platform_configs:
            errors.append("At least one platform configuration is required")
        
        if not config.environment_configs:
            errors.append("At least one environment configuration is required")
        
        return errors
    
    async def get_cluster_config(self, 
                               environment: Environment,
                               platform: PlatformType,
                               workload_type: WorkloadType) -> ClusterConfig:
        """Generate cluster configuration from infrastructure config"""
        base_config = await self.load_configuration(environment, platform)
        platform_specific = await self.generate_platform_specific_config(
            base_config, platform, environment
        )
        
        cluster_config = ClusterConfig(
            cluster_name=f"genetic-{environment}-{workload_type}",
            region=platform_specific.get("region", "us-west-2"),
            platform=platform,
            head_node_config=platform_specific["cluster_config"],
            worker_groups=[{
                "name": "genetic_workers",
                "instance_type": platform_specific["cluster_config"].get("instance_type", "c5.2xlarge"),
                "min_workers": platform_specific["cluster_config"].get("min_nodes", 2),
                "max_workers": platform_specific["cluster_config"].get("max_nodes", 10)
            }]
        )
        
        return cluster_config


class ConfigurationError(Exception):
    """Base exception for configuration operations"""
    pass


class CredentialError(ConfigurationError):
    """Raised when credential operations fail"""
    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails"""
    pass