# Docker Python Containerization Guide - Official Documentation

## Overview
Official Docker guide for containerizing Python applications with best practices and production-ready patterns.

## Key Dockerfile Patterns

### Production-Ready Dockerfile Template
```dockerfile
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-privileged user - SECURITY BEST PRACTICE
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies with cache optimization
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to non-privileged user
USER appuser

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python3", "-m", "uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]
```

## Critical Security & Performance Features

### 1. Non-Root User Pattern
```dockerfile
# Create dedicated user instead of using root
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser
```

### 2. Cache Optimization
```dockerfile
# Use cache mounts for pip to speed builds
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
```

### 3. Python Environment Variables
```dockerfile
# Prevent .pyc files (reduces container size)
ENV PYTHONDONTWRITEBYTECODE=1

# Force stdout/stderr to be unbuffered (better logging)
ENV PYTHONUNBUFFERED=1
```

## Docker Compose Patterns

### Basic Service Configuration
```yaml
services:
  server:
    build:
      context: .
    ports:
      - 8000:8000
```

### Advanced Multi-Service Configuration
```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./app:/app
      
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
```

## .dockerignore Best Practices
```
**/.DS_Store
**/__pycache__
**/.venv
**/.env
**/.git
**/.gitignore
**/node_modules
**/secrets.dev.yaml
LICENSE
README.md
```

## Application for Our Genetic Algorithm Infrastructure

### Fixes for Our Dockerfile Issues
1. **Permission Problem**: Our Dockerfile needs `USER root` before package installation, then switch to non-root user
2. **Security Enhancement**: Use dedicated `rayuser` with proper UID
3. **Cache Optimization**: Use mount caches for pip installations
4. **Environment Variables**: Add Python optimization flags

### Integration with Infrastructure Manager
The Docker Python SDK patterns can be integrated into our `infrastructure_manager.py`:

```python
import docker

class DockerInfrastructureManager:
    def __init__(self):
        self.client = docker.from_env()
    
    async def deploy_genetic_containers(self, population_size: int):
        # Build genetic pool image
        image = self.client.images.build(
            path="docker/genetic-pool/",
            tag="genetic-pool:latest"
        )
        
        # Calculate required containers
        worker_count = min(population_size // 20, 10)
        
        # Deploy Ray head node
        head_container = self.client.containers.run(
            "genetic-pool:latest",
            command="head",
            ports={'8265': 8265, '10001': 10001},
            detach=True,
            name="genetic-ray-head"
        )
        
        # Deploy worker containers
        workers = []
        for i in range(worker_count):
            worker = self.client.containers.run(
                "genetic-pool:latest",
                command="worker",
                environment={"RAY_HEAD_ADDRESS": "genetic-ray-head:10001"},
                detach=True,
                name=f"genetic-ray-worker-{i}"
            )
            workers.append(worker)
        
        return head_container, workers
```

## Production Deployment Patterns
- Use multi-stage builds for smaller production images
- Implement health checks with specific endpoints
- Use secrets management for API keys and credentials
- Configure resource limits and logging
- Use Docker Compose profiles for different environments