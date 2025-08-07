# Neon Security & SSL/TLS - Phase 4 Production Implementation

**Source**: https://neon.com/docs/security/security-overview  
**Extraction Date**: 2025-08-06  
**Project Context**: Phase 4 - Production Security for Neon Database Integration with Ray Workers

## Security Architecture Overview

### Multi-Layer Security Model

Neon implements comprehensive security across multiple layers:

**Infrastructure Security:**
- SOC 2 Type 2 compliance framework
- AWS infrastructure with VPC isolation
- Multi-tenant architecture with strict tenant isolation
- Regular security audits and penetration testing

**Network Security:**
- SSL/TLS encryption for all connections (mandatory)
- IP allowlisting for production environments
- Connection pooling with security parameters
- VPC integration for private connectivity

**Data Security:**
- Encryption at rest using AES-256
- Encryption in transit using TLS 1.2+
- Point-in-time recovery with encrypted backups
- Automatic security patches and updates

## SSL/TLS Configuration for Phase 4

### Production SSL Configuration

```python
# File: src/security/neon_ssl_config.py

import ssl
import asyncpg
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass

@dataclass
class NeonSSLConfig:
    """SSL configuration for Neon production deployment."""
    
    # SSL Mode Configuration
    SSL_MODE_REQUIRE = "require"      # Require SSL, but don't verify CA
    SSL_MODE_VERIFY_CA = "verify-ca"  # Require SSL and verify CA
    SSL_MODE_VERIFY_FULL = "verify-full"  # Require SSL, verify CA and hostname
    
    def __init__(self, ssl_mode: str = "require"):
        self.ssl_mode = ssl_mode
        self.ssl_context = self._create_ssl_context()
        self.connection_params = self._build_ssl_params()
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for Neon connections."""
        # Use TLS 1.2+ for production security
        context = ssl.create_default_context()
        
        # Production SSL settings
        context.check_hostname = (self.ssl_mode == "verify-full")
        context.verify_mode = ssl.CERT_REQUIRED if self.ssl_mode in ["verify-ca", "verify-full"] else ssl.CERT_NONE
        
        # Security hardening
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        return context
    
    def _build_ssl_params(self) -> Dict[str, Any]:
        """Build SSL parameters for asyncpg connection."""
        return {
            'ssl': self.ssl_context,
            'server_settings': {
                'application_name': 'quant_trading_secure',
                'ssl_min_protocol_version': 'TLSv1.2',
            }
        }

class NeonProductionSecurityManager:
    """Comprehensive security manager for Neon production deployment."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.ssl_config = NeonSSLConfig("require" if environment == "production" else "prefer")
        
    def build_secure_connection_string(
        self, 
        base_connection_string: str,
        additional_params: Optional[Dict[str, str]] = None
    ) -> str:
        """Build production-secure connection string."""
        
        # Ensure pooled connection for production
        if "-pooler." not in base_connection_string:
            base_connection_string = base_connection_string.replace(".neon.tech", "-pooler.neon.tech")
        
        # Production security parameters
        security_params = {
            "sslmode": self.ssl_config.ssl_mode,
            "application_name": "quant_trading_production",
            "connect_timeout": "30",
            "statement_timeout": "120000",  # 2 minutes
            "idle_in_transaction_session_timeout": "300000",  # 5 minutes
        }
        
        # Add custom parameters if provided
        if additional_params:
            security_params.update(additional_params)
        
        # Build query string
        separator = "&" if "?" in base_connection_string else "?"
        param_string = "&".join([f"{k}={v}" for k, v in security_params.items()])
        
        return f"{base_connection_string}{separator}{param_string}"
    
    async def create_secure_pool(
        self, 
        connection_string: str,
        pool_config: Optional[Dict[str, Any]] = None
    ) -> asyncpg.Pool:
        """Create production-secure connection pool."""
        
        # Default pool configuration for production
        default_pool_config = {
            "min_size": 10,
            "max_size": 50,
            "command_timeout": 30,
            "server_settings": {
                "application_name": "quant_trading_production",
                "timezone": "UTC",
                "statement_timeout": "120s",
                "lock_timeout": "30s",
                "idle_in_transaction_session_timeout": "300s",
            },
            **self.ssl_config.connection_params
        }
        
        # Override with custom config if provided
        if pool_config:
            default_pool_config.update(pool_config)
        
        # Create secure connection string
        secure_connection_string = self.build_secure_connection_string(connection_string)
        
        return await asyncpg.create_pool(secure_connection_string, **default_pool_config)
    
    def get_production_ip_allowlist(self) -> Dict[str, list]:
        """Get IP allowlist configuration for production deployment."""
        return {
            "ray_worker_cidrs": [
                "10.0.0.0/8",      # VPC CIDR range
                "172.16.0.0/12",   # Docker bridge networks  
                "192.168.0.0/16",  # Private networks
            ],
            "monitoring_cidrs": [
                "34.74.90.64/28",  # Google Cloud monitoring
                "35.235.240.0/20", # GCP health checks
            ],
            "admin_cidrs": [
                # Add specific admin IP ranges here
                # "203.0.113.0/24",  # Example admin network
            ]
        }
```

### SSL Certificate Management

```python
# File: src/security/neon_certificate_manager.py

import os
import logging
from pathlib import Path
from typing import Optional
import asyncio

class NeonCertificateManager:
    """Manage SSL certificates for Neon connections."""
    
    def __init__(self, cert_directory: str = "/etc/ssl/neon"):
        self.cert_directory = Path(cert_directory)
        self.logger = logging.getLogger(f"{__name__}.CertificateManager")
        
    def setup_certificate_directory(self):
        """Create certificate directory structure."""
        self.cert_directory.mkdir(parents=True, exist_ok=True)
        
        # Set appropriate permissions
        os.chmod(self.cert_directory, 0o700)
        
    def validate_certificates(self) -> Dict[str, bool]:
        """Validate SSL certificate configuration."""
        validation_results = {}
        
        cert_files = {
            "ca_cert": self.cert_directory / "ca-cert.pem",
            "client_cert": self.cert_directory / "client-cert.pem", 
            "client_key": self.cert_directory / "client-key.pem",
        }
        
        for cert_name, cert_path in cert_files.items():
            validation_results[cert_name] = cert_path.exists() and cert_path.is_file()
            
            if validation_results[cert_name]:
                # Check file permissions
                file_stat = cert_path.stat()
                if cert_name == "client_key" and (file_stat.st_mode & 0o077) != 0:
                    self.logger.warning(f"Client key {cert_path} has overly permissive permissions")
                    validation_results[cert_name] = False
        
        return validation_results
    
    def get_ssl_connection_params(self) -> Dict[str, str]:
        """Get SSL connection parameters with certificate paths."""
        cert_validation = self.validate_certificates()
        
        if not all(cert_validation.values()):
            self.logger.warning("Not all certificates are valid, using basic SSL mode")
            return {"sslmode": "require"}
        
        return {
            "sslmode": "verify-full",
            "sslcert": str(self.cert_directory / "client-cert.pem"),
            "sslkey": str(self.cert_directory / "client-key.pem"),
            "sslrootcert": str(self.cert_directory / "ca-cert.pem"),
        }
```

## Authentication & Authorization

### Environment-Based Credential Management

```python
# File: src/security/neon_auth_manager.py

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import base64
import json

@dataclass
class NeonCredentials:
    """Secure credential management for Neon connections."""
    host: str
    database: str
    username: str
    password: str
    port: int = 5432
    
    def to_connection_string(self) -> str:
        """Convert credentials to connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @classmethod
    def from_environment(cls) -> 'NeonCredentials':
        """Load credentials from environment variables."""
        required_vars = {
            'NEON_HOST': 'host',
            'NEON_DATABASE': 'database', 
            'NEON_USERNAME': 'username',
            'NEON_PASSWORD': 'password'
        }
        
        credentials = {}
        for env_var, field_name in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Required environment variable {env_var} not found")
            credentials[field_name] = value
        
        # Optional port override
        port = os.getenv('NEON_PORT', '5432')
        credentials['port'] = int(port)
        
        return cls(**credentials)
    
    @classmethod 
    def from_secret_manager(cls, secret_name: str) -> 'NeonCredentials':
        """Load credentials from AWS Secrets Manager or similar."""
        # Implementation would depend on your secret manager
        # This is a placeholder for the pattern
        secret_value = cls._get_secret(secret_name)
        credentials = json.loads(secret_value)
        return cls(**credentials)
    
    @staticmethod
    def _get_secret(secret_name: str) -> str:
        """Get secret from secret manager - implement based on your provider."""
        # AWS Secrets Manager example:
        # import boto3
        # client = boto3.client('secretsmanager')
        # response = client.get_secret_value(SecretId=secret_name)
        # return response['SecretString']
        
        # For now, fallback to environment
        return os.getenv(f"NEON_SECRET_{secret_name.upper()}", "{}")

class NeonAuthManager:
    """Authentication manager for Neon production deployment."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.logger = logging.getLogger(f"{__name__}.AuthManager")
        
    async def get_production_credentials(self) -> NeonCredentials:
        """Get production credentials with proper security."""
        
        if self.environment == "production":
            # In production, use secret manager or secure environment variables
            try:
                return NeonCredentials.from_environment()
            except ValueError as e:
                self.logger.error(f"Failed to load production credentials: {e}")
                raise
        else:
            # Development/testing credentials
            return self._get_development_credentials()
    
    def _get_development_credentials(self) -> NeonCredentials:
        """Get development credentials with defaults."""
        return NeonCredentials(
            host=os.getenv('NEON_DEV_HOST', 'localhost'),
            database=os.getenv('NEON_DEV_DATABASE', 'trading_dev'),
            username=os.getenv('NEON_DEV_USERNAME', 'postgres'),
            password=os.getenv('NEON_DEV_PASSWORD', 'development'),
            port=int(os.getenv('NEON_DEV_PORT', '5432'))
        )
    
    def validate_connection_security(self, connection_string: str) -> Dict[str, bool]:
        """Validate connection string security parameters."""
        validation_results = {
            "uses_ssl": "sslmode=" in connection_string,
            "uses_pooler": "-pooler." in connection_string,
            "has_timeout": "statement_timeout" in connection_string,
            "has_app_name": "application_name" in connection_string,
        }
        
        # Log security warnings
        for check, passed in validation_results.items():
            if not passed:
                self.logger.warning(f"Security check failed: {check}")
        
        return validation_results
```

## Network Security Configuration

### Ray Worker IP Allowlisting

```python
# File: src/security/neon_network_security.py

import ipaddress
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class NetworkSecurityPolicy:
    """Network security policy for Neon database access."""
    
    allowed_cidrs: List[str]
    environment: str
    description: str
    
    def validate_ip_access(self, client_ip: str) -> bool:
        """Validate if client IP is allowed access."""
        client_ip_obj = ipaddress.ip_address(client_ip)
        
        for cidr in self.allowed_cidrs:
            network = ipaddress.ip_network(cidr, strict=False)
            if client_ip_obj in network:
                return True
        
        return False

class NeonNetworkSecurityManager:
    """Manage network security for Neon database access."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NetworkSecurity")
        self.security_policies = self._define_security_policies()
    
    def _define_security_policies(self) -> Dict[str, NetworkSecurityPolicy]:
        """Define network security policies by environment."""
        return {
            "production": NetworkSecurityPolicy(
                allowed_cidrs=[
                    "10.0.0.0/8",      # VPC networks
                    "172.16.0.0/12",   # Docker bridge networks
                    "192.168.0.0/16",  # Private networks
                ],
                environment="production",
                description="Production Ray worker access"
            ),
            "staging": NetworkSecurityPolicy(
                allowed_cidrs=[
                    "10.0.0.0/8",      # VPC networks
                    "172.16.0.0/12",   # Docker networks
                    "203.0.113.0/24",  # Staging environment
                ],
                environment="staging", 
                description="Staging environment access"
            ),
            "development": NetworkSecurityPolicy(
                allowed_cidrs=[
                    "0.0.0.0/0",       # Open access for development
                ],
                environment="development",
                description="Development environment access"
            )
        }
    
    def get_neon_ip_allowlist(self, environment: str) -> List[str]:
        """Get IP allowlist for Neon database configuration."""
        if environment not in self.security_policies:
            raise ValueError(f"Unknown environment: {environment}")
        
        return self.security_policies[environment].allowed_cidrs
    
    async def validate_ray_worker_access(
        self, 
        worker_ip: str, 
        environment: str
    ) -> Dict[str, Any]:
        """Validate Ray worker IP access against security policy."""
        
        if environment not in self.security_policies:
            return {
                "allowed": False,
                "reason": f"Unknown environment: {environment}",
                "policy": None
            }
        
        policy = self.security_policies[environment]
        access_allowed = policy.validate_ip_access(worker_ip)
        
        result = {
            "allowed": access_allowed,
            "worker_ip": worker_ip,
            "environment": environment,
            "policy": policy.description,
            "matched_cidrs": []
        }
        
        # Find which CIDRs matched
        if access_allowed:
            client_ip_obj = ipaddress.ip_address(worker_ip)
            for cidr in policy.allowed_cidrs:
                network = ipaddress.ip_network(cidr, strict=False)
                if client_ip_obj in network:
                    result["matched_cidrs"].append(cidr)
        else:
            result["reason"] = f"IP {worker_ip} not in allowed CIDRs: {policy.allowed_cidrs}"
        
        # Log access attempt
        if access_allowed:
            self.logger.info(f"Allowed Ray worker access: {worker_ip} [{environment}]")
        else:
            self.logger.warning(f"Blocked Ray worker access: {worker_ip} [{environment}]")
        
        return result
```

## Security Monitoring & Alerting

```python
# File: src/security/neon_security_monitoring.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class SecurityEvent:
    """Security event for monitoring and alerting."""
    timestamp: datetime
    event_type: str
    severity: str  # 'info', 'warning', 'critical'
    source_ip: str
    details: Dict[str, Any]
    
@dataclass
class SecurityMetrics:
    """Security metrics for monitoring."""
    failed_connections: int = 0
    successful_connections: int = 0
    blocked_ips: List[str] = field(default_factory=list)
    ssl_errors: int = 0
    auth_failures: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

class NeonSecurityMonitor:
    """Monitor security events for Neon database connections."""
    
    def __init__(self, alert_threshold: int = 10):
        self.logger = logging.getLogger(f"{__name__}.SecurityMonitor")
        self.alert_threshold = alert_threshold
        self.security_events: List[SecurityEvent] = []
        self.metrics_history: List[SecurityMetrics] = []
        self.ip_failure_counts = defaultdict(int)
        
    def record_security_event(
        self, 
        event_type: str, 
        severity: str,
        source_ip: str, 
        details: Dict[str, Any]
    ):
        """Record a security event for monitoring."""
        
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            details=details
        )
        
        self.security_events.append(event)
        
        # Track failure counts per IP
        if event_type in ["connection_failed", "auth_failed"]:
            self.ip_failure_counts[source_ip] += 1
            
            # Check for potential security threats
            if self.ip_failure_counts[source_ip] >= self.alert_threshold:
                self._trigger_security_alert(source_ip, event)
        
        # Log security event
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(
            log_level, 
            f"Security event [{event_type}] from {source_ip}: {details}"
        )
    
    def _trigger_security_alert(self, source_ip: str, event: SecurityEvent):
        """Trigger security alert for suspicious activity."""
        alert_details = {
            "alert_type": "suspicious_activity",
            "source_ip": source_ip,
            "failure_count": self.ip_failure_counts[source_ip],
            "threshold": self.alert_threshold,
            "recent_events": [e.event_type for e in self.security_events[-10:] if e.source_ip == source_ip]
        }
        
        self.logger.critical(f"SECURITY ALERT: Suspicious activity from {source_ip}: {alert_details}")
        
        # In production, this would integrate with alerting system
        # e.g., send to Slack, PagerDuty, email, etc.
        
    async def collect_security_metrics(self) -> SecurityMetrics:
        """Collect current security metrics."""
        recent_events = [
            e for e in self.security_events 
            if e.timestamp >= datetime.utcnow() - timedelta(minutes=5)
        ]
        
        metrics = SecurityMetrics(
            failed_connections=len([e for e in recent_events if e.event_type == "connection_failed"]),
            successful_connections=len([e for e in recent_events if e.event_type == "connection_success"]),
            blocked_ips=list(set([e.source_ip for e in recent_events if e.event_type == "ip_blocked"])),
            ssl_errors=len([e for e in recent_events if e.event_type == "ssl_error"]),
            auth_failures=len([e for e in recent_events if e.event_type == "auth_failed"]),
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:  # Keep last 100 metrics
            self.metrics_history.pop(0)
            
        return metrics
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary for reporting."""
        if not self.metrics_history:
            return {"status": "no_data", "summary": "No security metrics available"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate failure rate
        total_connections = latest_metrics.failed_connections + latest_metrics.successful_connections
        failure_rate = (latest_metrics.failed_connections / total_connections * 100) if total_connections > 0 else 0
        
        return {
            "status": "healthy" if failure_rate < 5 else "warning" if failure_rate < 20 else "critical",
            "failure_rate_percent": round(failure_rate, 2),
            "metrics": {
                "failed_connections": latest_metrics.failed_connections,
                "successful_connections": latest_metrics.successful_connections,
                "blocked_ips_count": len(latest_metrics.blocked_ips),
                "ssl_errors": latest_metrics.ssl_errors,
                "auth_failures": latest_metrics.auth_failures,
            },
            "alerts": {
                "high_failure_ips": [ip for ip, count in self.ip_failure_counts.items() if count >= self.alert_threshold],
                "recent_blocked_ips": latest_metrics.blocked_ips,
            },
            "timestamp": latest_metrics.timestamp.isoformat()
        }
```

## Production Security Checklist

### Pre-Deployment Security Validation

```python
# File: src/security/neon_security_validation.py

class NeonSecurityValidator:
    """Validate security configuration before production deployment."""
    
    def __init__(self, connection_pool):
        self.pool = connection_pool
        self.validation_results = {}
        
    async def run_security_validation(self) -> Dict[str, bool]:
        """Run comprehensive security validation."""
        
        validations = [
            ("ssl_configuration", self._validate_ssl_configuration),
            ("authentication", self._validate_authentication),
            ("network_security", self._validate_network_security),
            ("connection_parameters", self._validate_connection_parameters),
            ("monitoring_setup", self._validate_monitoring_setup),
        ]
        
        for validation_name, validation_func in validations:
            try:
                result = await validation_func()
                self.validation_results[validation_name] = result
            except Exception as e:
                logging.error(f"Security validation {validation_name} failed: {e}")
                self.validation_results[validation_name] = False
        
        return self.validation_results
    
    async def _validate_ssl_configuration(self) -> bool:
        """Validate SSL/TLS configuration."""
        async with self.pool.acquire() as conn:
            # Check if SSL is enabled
            ssl_result = await conn.fetchval("SHOW ssl")
            if ssl_result != "on":
                return False
                
            # Check SSL version
            version_result = await conn.fetchval("SELECT version()")
            return "SSL" in version_result
    
    async def _validate_authentication(self) -> bool:
        """Validate authentication configuration."""
        # This would validate that proper authentication is configured
        # Implementation depends on your specific auth setup
        return True
    
    # Additional validation methods...
```

## Implementation Priority

1. **✅ CRITICAL**: SSL/TLS configuration with proper certificate management
2. **✅ CRITICAL**: Environment-based credential management
3. **✅ HIGH**: Network security with IP allowlisting
4. **✅ HIGH**: Security monitoring and alerting
5. **✅ MEDIUM**: Security validation and compliance checking

## Next Steps

1. Implement `NeonProductionSecurityManager` with SSL/TLS configuration
2. Set up environment-based credential management
3. Configure IP allowlisting for Ray worker networks
4. Deploy security monitoring and alerting system
5. Run security validation before production deployment

**Security Success Criteria**: Full SSL/TLS encryption, proper credential management, network access control, comprehensive monitoring, zero security vulnerabilities in production deployment.