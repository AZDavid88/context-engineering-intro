---
allowed-tools: Bash(docker:*), Bash(git:*), Bash(python:*), Write, Read, Grep, Glob
description: Production readiness checks and deployment preparation
argument-hint: [environment] | production deployment prep if no environment specified
---

# Production Deployment Preparation

**Context**: You are using the CodeFarm methodology for comprehensive production deployment preparation. This ensures system reliability, performance, and maintainability in production environments.

## Deployment Environment Setup

### 1. Environment Configuration
**Target Environment**: ${ARGUMENTS:-"production"}

Available environments:
- `staging` - Pre-production testing environment
- `production` - Live production environment
- `test` - Integration testing environment
- `development` - Development environment validation

### 2. Pre-Deployment Checklist
Initial deployment readiness assessment using Claude Code tools:

**Git status check:**
- Use Bash tool with command: `git status --porcelain | wc -l` to count uncommitted changes

**Current branch verification:**
- Use Bash tool with command: `git branch --show-current` to check current branch

**Recent commits review:**
- Use Bash tool with command: `git log --oneline -5` to see recent commits

**Docker availability check:**
- Use Bash tool with command: `docker --version 2>/dev/null || echo "Docker not available"` to verify Docker installation

**Python environment verification:**
- Use Bash tool with command: `python --version && echo "Virtual env: $VIRTUAL_ENV"` to check Python setup

### 3. Code Quality Gates
Ensure code meets production standards using Bash tool:

```bash
# Run comprehensive test suite
echo "=== Running Production Quality Gates ==="

# Unit tests with coverage
python -m pytest tests/unit/ -v --cov=src --cov-fail-under=80 || echo "❌ Unit tests failed"

# Integration tests
python -m pytest tests/integration/ -v || echo "❌ Integration tests failed"

# Code quality checks
python -m flake8 src/ --max-line-length=88 --ignore=E203,W503 || echo "❌ Code style issues"

# Type checking
python -m mypy src/ --ignore-missing-imports || echo "❌ Type checking failed"

# Security scan
python -m bandit -r src/ -ll || echo "❌ Security issues found"
```

## Production Configuration

### 4. Environment Variables Setup
Create production-ready environment configuration:

```python
# deployment_config_generator.py
import os
import secrets
import string
from typing import Dict, Any

class DeploymentConfigGenerator:
    """Generate production deployment configurations"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = {}
    
    def generate_secret_key(self, length: int = 32) -> str:
        """Generate cryptographically secure secret key"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def create_environment_config(self) -> Dict[str, Any]:
        """Create environment-specific configuration"""
        
        base_config = {
            # Application Settings
            "ENVIRONMENT": self.environment.upper(),
            "DEBUG": "False" if self.environment == "production" else "True",
            "SECRET_KEY": self.generate_secret_key(),
            
            # Database
            "DATABASE_URL": "${DATABASE_CONNECTION_STRING}",
            "DATABASE_POOL_SIZE": "10" if self.environment == "production" else "5",
            "DATABASE_MAX_OVERFLOW": "20" if self.environment == "production" else "10",
            
            # Redis/Cache
            "REDIS_URL": "${REDIS_CONNECTION_STRING}",
            "CACHE_TTL": "3600",
            
            # Security
            "ALLOWED_HOSTS": "yourdomain.com,api.yourdomain.com" if self.environment == "production" else "localhost,127.0.0.1",
            "CORS_ORIGINS": "https://yourdomain.com" if self.environment == "production" else "http://localhost:3000",
            "SESSION_COOKIE_SECURE": "True" if self.environment == "production" else "False",
            "CSRF_COOKIE_SECURE": "True" if self.environment == "production" else "False",
            
            # API Keys
            "API_KEY": "${YOUR_API_KEY}",
            "JWT_SECRET": self.generate_secret_key(),
            
            # Logging
            "LOG_LEVEL": "INFO" if self.environment == "production" else "DEBUG",
            "LOG_FORMAT": "json" if self.environment == "production" else "text",
            
            # Performance
            "WORKERS": "4" if self.environment == "production" else "1",
            "WORKER_TIMEOUT": "30",
            "MAX_REQUESTS": "1000",
            "MAX_REQUESTS_JITTER": "100",
            
            # Monitoring
            "SENTRY_DSN": "${SENTRY_DSN}" if self.environment == "production" else "",
            "PROMETHEUS_ENABLED": "True" if self.environment == "production" else "False",
            
            # Rate Limiting
            "RATE_LIMIT_ENABLED": "True",
            "RATE_LIMIT_PER_MINUTE": "60",
            "RATE_LIMIT_PER_HOUR": "1000",
        }
        
        return base_config
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration"""
        
        compose_config = f"""
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT={self.environment.upper()}
      - DATABASE_URL=${{DATABASE_URL}}
      - REDIS_URL=${{REDIS_URL}}
      - SECRET_KEY=${{SECRET_KEY}}
    volumes:
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=app_db
      - POSTGRES_USER=${{DB_USER}}
      - POSTGRES_PASSWORD=${{DB_PASSWORD}}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${DB_USER} -d app_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
"""
        
        return compose_config.strip()
    
    def generate_dockerfile(self) -> str:
        """Generate production Dockerfile"""
        
        dockerfile = """
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN adduser --disabled-password --gecos '' appuser

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        curl \\
        build-essential \\
        libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "main:app"]
"""
        
        return dockerfile.strip()
    
    def generate_gunicorn_config(self) -> str:
        """Generate Gunicorn configuration for production"""
        
        config = """
import os
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = int(os.getenv("WORKER_TIMEOUT", 30))
keepalive = 2

# Restart workers
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 100))
preload_app = True

# Logging
accesslog = "/app/logs/access.log"
errorlog = "/app/logs/error.log"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "app"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
worker_tmp_dir = "/dev/shm"
"""
        
        return config.strip()

if __name__ == "__main__":
    import sys
    
    environment = sys.argv[1] if len(sys.argv) > 1 else "production"
    generator = DeploymentConfigGenerator(environment)
    
    # Generate configurations
    env_config = generator.create_environment_config()
    
    # Create deployment directory
    os.makedirs("deployment", exist_ok=True)
    
    # Write environment file
    with open(f"deployment/.env.{environment}", "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\\n")
    
    # Write Docker configurations
    with open("deployment/docker-compose.yml", "w") as f:
        f.write(generator.generate_docker_compose())
    
    with open("deployment/Dockerfile", "w") as f:
        f.write(generator.generate_dockerfile())
    
    with open("deployment/gunicorn.conf.py", "w") as f:
        f.write(generator.generate_gunicorn_config())
    
    print(f"Deployment configuration generated for {environment} environment")
    print("Files created:")
    print(f"  deployment/.env.{environment}")
    print("  deployment/docker-compose.yml")
    print("  deployment/Dockerfile")
    print("  deployment/gunicorn.conf.py")
```

### 5. Database Migration Preparation
Ensure database is production-ready:

```python
# database_migration_prep.py
import os
from typing import List, Dict, Any

class DatabaseMigrationPrep:
    """Prepare database for production deployment"""
    
    def __init__(self):
        self.migration_checks = []
    
    def validate_migrations(self) -> Dict[str, Any]:
        """Validate database migrations are ready"""
        
        results = {
            "migrations_pending": [],
            "migration_conflicts": [],
            "backup_required": True,
            "rollback_plan": True
        }
        
        # Check for pending migrations
        try:
            # This would integrate with your migration system
            # Example for Alembic:
            # from alembic import command
            # from alembic.config import Config
            
            # config = Config("alembic.ini")
            # command.check(config)
            
            print("✓ Migration validation completed")
            
        except Exception as e:
            results["migrations_pending"].append(str(e))
        
        return results
    
    def generate_backup_script(self) -> str:
        """Generate database backup script"""
        
        backup_script = """#!/bin/bash
# Database backup script for deployment

# Configuration
DB_NAME="${DB_NAME:-app_db}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/pre_deploy_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
echo "Creating database backup..."
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "✓ Backup created: $BACKUP_FILE"
    
    # Compress backup
    gzip $BACKUP_FILE
    echo "✓ Backup compressed: ${BACKUP_FILE}.gz"
    
    # Verify backup
    if [ -f "${BACKUP_FILE}.gz" ]; then
        echo "✓ Backup verification successful"
        echo "Backup size: $(du -h ${BACKUP_FILE}.gz | cut -f1)"
    else
        echo "❌ Backup verification failed"
        exit 1
    fi
else
    echo "❌ Backup creation failed"
    exit 1
fi

# Cleanup old backups (keep last 10)
echo "Cleaning up old backups..."
cd $BACKUP_DIR
ls -t *.sql.gz 2>/dev/null | tail -n +11 | xargs -r rm
echo "✓ Backup cleanup completed"
"""
        
        return backup_script.strip()
    
    def generate_rollback_script(self) -> str:
        """Generate rollback script"""
        
        rollback_script = """#!/bin/bash
# Database rollback script

# Configuration
DB_NAME="${DB_NAME:-app_db}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
BACKUP_DIR="./backups"

# Find latest backup
LATEST_BACKUP=$(ls -t ${BACKUP_DIR}/*.sql.gz 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No backup found for rollback"
    exit 1
fi

echo "Rolling back to: $LATEST_BACKUP"

# Confirm rollback
read -p "Are you sure you want to rollback the database? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Rollback cancelled"
    exit 0
fi

# Restore backup
echo "Restoring database from backup..."
gunzip -c $LATEST_BACKUP | psql -h $DB_HOST -U $DB_USER -d $DB_NAME

if [ $? -eq 0 ]; then
    echo "✓ Database rollback completed successfully"
else
    echo "❌ Database rollback failed"
    exit 1
fi
"""
        
        return rollback_script.strip()

if __name__ == "__main__":
    db_prep = DatabaseMigrationPrep()
    
    # Validate migrations
    results = db_prep.validate_migrations()
    print("Database migration validation:")
    print(f"  Pending migrations: {len(results['migrations_pending'])}")
    print(f"  Migration conflicts: {len(results['migration_conflicts'])}")
    
    # Generate scripts
    os.makedirs("deployment/scripts", exist_ok=True)
    
    with open("deployment/scripts/backup.sh", "w") as f:
        f.write(db_prep.generate_backup_script())
    
    with open("deployment/scripts/rollback.sh", "w") as f:
        f.write(db_prep.generate_rollback_script())
    
    # Make scripts executable
    os.chmod("deployment/scripts/backup.sh", 0o755)
    os.chmod("deployment/scripts/rollback.sh", 0o755)
    
    print("Database scripts generated:")
    print("  deployment/scripts/backup.sh")
    print("  deployment/scripts/rollback.sh")
```

## Production Monitoring Setup

### 6. Health Check Implementation
Implement comprehensive health checks:

```python
# health_check_system.py
import time
import psutil
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

class HealthCheckSystem:
    """Comprehensive health check system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_checks = {}
    
    async def check_application_health(self) -> Dict[str, Any]:
        """Check application-level health"""
        
        return {
            "status": "healthy",
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.utcnow().isoformat(),
            "version": self._get_version(),
            "environment": self._get_environment()
        }
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        
        try:
            # This would integrate with your database
            # Example implementation:
            start_time = time.time()
            
            # Simulate database check
            await asyncio.sleep(0.01)  # Simulate DB query
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "connections": {
                    "active": 5,  # Would get from actual pool
                    "idle": 3,
                    "max": 10
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_external_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies"""
        
        dependencies = {
            "redis": await self._check_redis(),
            "external_api": await self._check_external_api(),
            "file_storage": await self._check_file_storage()
        }
        
        all_healthy = all(dep["status"] == "healthy" for dep in dependencies.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "dependencies": dependencies
        }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Define warning thresholds
        cpu_warning = cpu_percent > 80
        memory_warning = memory.percent > 80
        disk_warning = disk.percent > 80
        
        status = "healthy"
        if cpu_warning or memory_warning or disk_warning:
            status = "warning"
        if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
            status = "critical"
        
        return {
            "status": status,
            "cpu": {
                "percent": cpu_percent,
                "warning": cpu_warning
            },
            "memory": {
                "percent": memory.percent,
                "available_mb": memory.available / 1024 / 1024,
                "warning": memory_warning
            },
            "disk": {
                "percent": disk.percent,
                "free_gb": disk.free / 1024 / 1024 / 1024,
                "warning": disk_warning
            }
        }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        
        start_time = time.time()
        
        # Run all checks concurrently
        app_health, db_health, deps_health, system_health = await asyncio.gather(
            self.check_application_health(),
            self.check_database_health(),
            self.check_external_dependencies(),
            self.check_system_resources()
        )
        
        # Determine overall status
        statuses = [
            app_health["status"],
            db_health["status"],
            deps_health["status"],
            system_health["status"]
        ]
        
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses or "degraded" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        check_duration = time.time() - start_time
        
        return {
            "status": overall_status,
            "check_duration_ms": check_duration * 1000,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "application": app_health,
                "database": db_health,
                "dependencies": deps_health,
                "system": system_health
            }
        }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            # Simulate Redis check
            await asyncio.sleep(0.005)
            return {"status": "healthy", "response_time_ms": 5}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_external_api(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        try:
            # Simulate external API check
            await asyncio.sleep(0.01)
            return {"status": "healthy", "response_time_ms": 10}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_file_storage(self) -> Dict[str, Any]:
        """Check file storage accessibility"""
        try:
            # Simulate storage check
            return {"status": "healthy", "available_space_gb": 100}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _get_version(self) -> str:
        """Get application version"""
        try:
            with open("VERSION", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "unknown"
    
    def _get_environment(self) -> str:
        """Get current environment"""
        return os.getenv("ENVIRONMENT", "unknown")

# FastAPI health check endpoints
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

def setup_health_endpoints(app: FastAPI):
    """Set up health check endpoints"""
    
    health_system = HealthCheckSystem()
    
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        health = await health_system.check_application_health()
        return JSONResponse(content=health)
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Comprehensive health check endpoint"""
        health = await health_system.comprehensive_health_check()
        
        # Set appropriate HTTP status code
        status_code = 200
        if health["status"] in ["warning", "degraded"]:
            status_code = 503
        elif health["status"] in ["unhealthy", "critical"]:
            status_code = 503
        
        return JSONResponse(content=health, status_code=status_code)
    
    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe"""
        db_health = await health_system.check_database_health()
        deps_health = await health_system.check_external_dependencies()
        
        if db_health["status"] == "healthy" and deps_health["status"] in ["healthy", "degraded"]:
            return {"status": "ready"}
        else:
            return Response(status_code=503, content="Not ready")
    
    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe"""
        app_health = await health_system.check_application_health()
        system_health = await health_system.check_system_resources()
        
        if app_health["status"] == "healthy" and system_health["status"] != "critical":
            return {"status": "alive"}
        else:
            return Response(status_code=503, content="Not alive")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        health_system = HealthCheckSystem()
        
        print("Running comprehensive health check...")
        health_result = await health_system.comprehensive_health_check()
        
        print(f"Overall Status: {health_result['status']}")
        print(f"Check Duration: {health_result['check_duration_ms']:.2f}ms")
        
        for check_name, check_result in health_result['checks'].items():
            print(f"  {check_name}: {check_result['status']}")
    
    asyncio.run(main())
```

### 7. Logging and Monitoring Setup
Configure comprehensive logging:

```python
# logging_config.py
import os
import json
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any

class ProductionLoggingConfig:
    """Production logging configuration"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.log_format = os.getenv("LOG_FORMAT", "json" if environment == "production" else "text")
        self.log_level = os.getenv("LOG_LEVEL", "INFO" if environment == "production" else "DEBUG")
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get comprehensive logging configuration"""
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "logging.Formatter",
                    "format": json.dumps({
                        "timestamp": "%(asctime)s",
                        "level": "%(levelname)s",
                        "logger": "%(name)s",
                        "message": "%(message)s",
                        "module": "%(module)s",
                        "function": "%(funcName)s",
                        "line": "%(lineno)d"
                    })
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level,
                    "formatter": self.log_format,
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.log_level,
                    "formatter": self.log_format,
                    "filename": "/app/logs/app.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": self.log_format,
                    "filename": "/app/logs/error.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "security": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "WARNING",
                    "formatter": self.log_format,
                    "filename": "/app/logs/security.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10
                }
            },
            "loggers": {
                "": {  # Root logger
                    "level": self.log_level,
                    "handlers": ["console", "file", "error_file"]
                },
                "security": {
                    "level": "WARNING",
                    "handlers": ["security", "console"],
                    "propagate": False
                },
                "access": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                },
                "sqlalchemy.engine": {
                    "level": "WARNING" if self.environment == "production" else "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            }
        }
        
        return config
    
    def setup_logging(self):
        """Initialize logging configuration"""
        
        # Create logs directory
        os.makedirs("/app/logs", exist_ok=True)
        
        # Apply configuration
        logging.config.dictConfig(self.get_logging_config())
        
        # Log initialization
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized for {self.environment} environment")
        logger.info(f"Log level: {self.log_level}, Format: {self.log_format}")

# Monitoring integration
class ApplicationMetrics:
    """Application metrics for monitoring"""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_by_status": {},
            "response_time_sum": 0.0,
            "response_time_count": 0,
            "errors_total": 0,
            "active_connections": 0
        }
    
    def record_request(self, status_code: int, response_time: float):
        """Record request metrics"""
        
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_status"][status_code] = self.metrics["requests_by_status"].get(status_code, 0) + 1
        self.metrics["response_time_sum"] += response_time
        self.metrics["response_time_count"] += 1
        
        if status_code >= 400:
            self.metrics["errors_total"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        
        avg_response_time = 0
        if self.metrics["response_time_count"] > 0:
            avg_response_time = self.metrics["response_time_sum"] / self.metrics["response_time_count"]
        
        return {
            **self.metrics,
            "avg_response_time": avg_response_time,
            "error_rate": self.metrics["errors_total"] / max(1, self.metrics["requests_total"])
        }

if __name__ == "__main__":
    # Set up production logging
    logging_config = ProductionLoggingConfig("production")
    logging_config.setup_logging()
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Production logging configuration applied")
    
    # Test security logging
    security_logger = logging.getLogger("security")
    security_logger.warning("Security event test")
    
    print("Logging configuration saved to deployment/logging_config.py")
```

## Deployment Execution

### 8. Deployment Scripts
Create automated deployment scripts:

```bash
# deployment/deploy.sh
#!/bin/bash

# Production Deployment Script
set -e  # Exit on any error

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_NAME="your-app"
COMPOSE_FILE="docker-compose.yml"
BACKUP_REQUIRED=${BACKUP_REQUIRED:-true}

echo "=== Starting $ENVIRONMENT deployment ==="
echo "Project: $PROJECT_NAME"
echo "Timestamp: $(date)"

# Pre-deployment checks
echo "Running pre-deployment checks..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Check if environment file exists
if [ ! -f ".env.$ENVIRONMENT" ]; then
    echo "❌ Environment file .env.$ENVIRONMENT not found"
    exit 1
fi

# Check if Git is clean (production only)
if [ "$ENVIRONMENT" = "production" ]; then
    if [ ! -z "$(git status --porcelain)" ]; then
        echo "❌ Git working directory is not clean"
        echo "Commit or stash changes before deploying to production"
        exit 1
    fi
fi

# Database backup (if required)
if [ "$BACKUP_REQUIRED" = "true" ]; then
    echo "Creating database backup..."
    ./scripts/backup.sh || {
        echo "❌ Database backup failed"
        exit 1
    }
    echo "✓ Database backup completed"
fi

# Build and deploy
echo "Building application..."
docker-compose -f $COMPOSE_FILE build || {
    echo "❌ Build failed"
    exit 1
}

echo "Starting deployment..."
docker-compose -f $COMPOSE_FILE --env-file .env.$ENVIRONMENT up -d || {
    echo "❌ Deployment failed"
    
    # Rollback on failure
    echo "Rolling back deployment..."
    docker-compose -f $COMPOSE_FILE down
    
    if [ "$BACKUP_REQUIRED" = "true" ]; then
        ./scripts/rollback.sh
    fi
    
    exit 1
}

# Health check
echo "Performing health checks..."
sleep 30  # Wait for services to start

for i in {1..10}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Health check passed"
        break
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ Health check failed after 10 attempts"
        echo "Rolling back deployment..."
        docker-compose -f $COMPOSE_FILE down
        exit 1
    fi
    
    echo "Health check attempt $i failed, retrying in 10 seconds..."
    sleep 10
done

# Clean up old images
echo "Cleaning up old Docker images..."
docker image prune -f

echo "✓ Deployment completed successfully"
echo "Application is running at: http://localhost:8000"
echo "View logs with: docker-compose logs -f"
```

### 9. Monitoring and Alerting Setup
Configure monitoring systems:

```python
# monitoring_setup.py
import os
from typing import Dict, Any

def generate_prometheus_config() -> str:
    """Generate Prometheus configuration"""
    
    config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis_exporter:9121']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node_exporter:9100']
"""
    
    return config.strip()

def generate_alert_rules() -> str:
    """Generate Prometheus alert rules"""
    
    rules = """
groups:
  - name: application_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
      
      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is down"
      
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
      
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
      
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"}) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space"
          description: "Disk space is {{ $value | humanizePercentage }} full"
"""
    
    return rules.strip()

def generate_grafana_dashboard() -> Dict[str, Any]:
    """Generate Grafana dashboard configuration"""
    
    dashboard = {
        "dashboard": {
            "title": "Application Monitoring",
            "panels": [
                {
                    "title": "Request Rate",
                    "type": "graph",
                    "targets": [
                        {"expr": "rate(http_requests_total[5m])"}
                    ]
                },
                {
                    "title": "Response Time",
                    "type": "graph", 
                    "targets": [
                        {"expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"}
                    ]
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "targets": [
                        {"expr": "rate(http_requests_total{status=~\"5..\"}[5m])"}
                    ]
                },
                {
                    "title": "System Resources",
                    "type": "graph",
                    "targets": [
                        {"expr": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"},
                        {"expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100"}
                    ]
                }
            ]
        }
    }
    
    return dashboard

if __name__ == "__main__":
    import json
    
    # Create monitoring directory
    os.makedirs("deployment/monitoring", exist_ok=True)
    
    # Generate Prometheus configuration
    with open("deployment/monitoring/prometheus.yml", "w") as f:
        f.write(generate_prometheus_config())
    
    # Generate alert rules
    with open("deployment/monitoring/alert_rules.yml", "w") as f:
        f.write(generate_alert_rules())
    
    # Generate Grafana dashboard
    with open("deployment/monitoring/grafana_dashboard.json", "w") as f:
        json.dump(generate_grafana_dashboard(), f, indent=2)
    
    print("Monitoring configuration generated:")
    print("  deployment/monitoring/prometheus.yml")
    print("  deployment/monitoring/alert_rules.yml")
    print("  deployment/monitoring/grafana_dashboard.json")
```

### 10. Success Criteria
Deployment preparation complete when:

- [ ] All tests pass in production-like environment
- [ ] Database migrations validated and backed up
- [ ] Environment configurations secure and tested
- [ ] Docker containers build and run successfully
- [ ] Health checks implemented and functional
- [ ] Monitoring and alerting configured
- [ ] Rollback procedures tested
- [ ] Performance benchmarks met
- [ ] Security audit passed
- [ ] Documentation updated
- [ ] Team deployment runbook complete

---

This comprehensive deployment preparation ensures production readiness with proper monitoring, security, and reliability measures in place.