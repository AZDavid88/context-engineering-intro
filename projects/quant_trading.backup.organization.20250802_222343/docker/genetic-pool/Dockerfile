# Production Docker Container for Genetic Algorithm Pool
# 
# This container packages the genetic algorithm pool with Ray cluster
# integration for distributed execution on Anyscale and other platforms.
#
# Research-Based Implementation (Official Docker Documentation):
# - /research/docker_python/research_summary.md - Official Docker Python patterns
# - /research/ray_cluster/research_summary.md - Ray cluster integration patterns  
# - /research/anyscale/research_summary.md - Anyscale optimization patterns

FROM rayproject/ray:2.8.0-py310

# Switch to root user for package installation (Official Docker Pattern)
USER root

# Set working directory
WORKDIR /app

# Python optimization environment variables (Official Docker Best Practice)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ray and genetic algorithm optimization
ENV RAY_HEAD_NODE_HOST=0.0.0.0
ENV RAY_HEAD_NODE_PORT=10001
ENV RAY_GENETIC_ALGORITHM_MODE=hybrid
ENV RAY_OBJECT_STORE_MEMORY=2000000000
ENV RAY_DISABLE_IMPORT_WARNING=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for genetic algorithm optimization
RUN pip install --no-cache-dir \
    deap==1.4.1 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.24.4 \
    orjson==3.9.10 \
    aiofiles==23.2.1 \
    cryptography==41.0.7 \
    pyyaml==6.0.1

# Copy source code
COPY src/ /app/src/
COPY infrastructure/ /app/infrastructure/

# Copy configuration files
COPY config/ /app/config/
RUN mkdir -p /app/infrastructure/config

# Copy genetic algorithm seeds
COPY src/strategy/genetic_seeds/ /app/src/strategy/genetic_seeds/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/results

# Set up Ray configuration for genetic algorithm workloads
RUN mkdir -p /tmp/ray

# Create health check script
COPY docker/genetic-pool/health_check.py /app/
RUN chmod +x /app/health_check.py

# Create entrypoint script for flexible startup
COPY docker/genetic-pool/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/health_check.py

# Expose Ray ports
EXPOSE 10001 8265 8000

# Set resource limits for genetic algorithm optimization
ENV RAY_memory_usage_threshold=0.95
ENV RAY_memory_monitor_refresh_ms=1000

# Create non-root user for security (Official Docker Pattern)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    rayuser && \
    chown -R rayuser:rayuser /app /tmp/ray
USER rayuser

# Default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command runs genetic strategy pool in distributed mode
CMD ["python", "/app/src/execution/genetic_strategy_pool.py", "--mode", "distributed"]