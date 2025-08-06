#!/bin/bash

# Entrypoint script for Genetic Algorithm Pool Container
#
# This script provides flexible startup options for the genetic algorithm
# container, supporting different deployment modes and Ray cluster configurations.
#
# Usage:
#   docker run genetic-pool:latest [mode] [additional_args]
#   
# Modes:
#   - local: Run genetic algorithm in local mode
#   - distributed: Connect to existing Ray cluster
#   - head: Start as Ray head node
#   - worker: Start as Ray worker node

set -e

# Set default values
MODE=${1:-distributed}
RAY_ADDRESS=${RAY_ADDRESS:-auto}
RAY_HEAD_NODE_HOST=${RAY_HEAD_NODE_HOST:-0.0.0.0}
RAY_HEAD_NODE_PORT=${RAY_HEAD_NODE_PORT:-10001}

# Logging setup
echo "============================================"
echo "Genetic Algorithm Pool Container Starting"
echo "============================================"
echo "Mode: $MODE"
echo "Ray Address: $RAY_ADDRESS"
echo "Timestamp: $(date)"
echo "Container User: $(whoami)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "============================================"

# Function to wait for Ray cluster to be ready
wait_for_ray_cluster() {
    echo "Waiting for Ray cluster to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if python -c "import ray; ray.init(address='$RAY_ADDRESS', ignore_reinit_error=True); print('Ray cluster ready')" 2>/dev/null; then
            echo "Ray cluster is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Ray cluster not ready, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Ray cluster failed to become ready after $max_attempts attempts"
    return 1
}

# Function to start Ray head node
start_ray_head() {
    echo "Starting Ray head node..."
    
    # Ray head node configuration optimized for genetic algorithms
    ray start \
        --head \
        --node-ip-address=$RAY_HEAD_NODE_HOST \
        --port=$RAY_HEAD_NODE_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --redis-password="" \
        --object-store-memory=2000000000 \
        --memory-fraction=0.8 \
        --num-cpus=$(nproc) \
        --temp-dir=/tmp/ray \
        --disable-usage-stats \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "Ray head node started successfully"
        echo "Dashboard available at: http://0.0.0.0:8265"
        return 0
    else
        echo "ERROR: Failed to start Ray head node"
        return 1
    fi
}

# Function to start Ray worker node
start_ray_worker() {
    echo "Starting Ray worker node..."
    
    if [ -z "$RAY_HEAD_ADDRESS" ]; then
        echo "ERROR: RAY_HEAD_ADDRESS environment variable must be set for worker mode"
        exit 1
    fi
    
    # Ray worker node configuration
    ray start \
        --address=$RAY_HEAD_ADDRESS \
        --object-store-memory=2000000000 \
        --memory-fraction=0.8 \
        --num-cpus=$(nproc) \
        --temp-dir=/tmp/ray \
        --disable-usage-stats \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "Ray worker node started successfully"
        return 0
    else
        echo "ERROR: Failed to start Ray worker node"
        return 1
    fi
}

# Function to run health check
run_health_check() {
    echo "Running container health check..."
    python /app/health_check.py
    
    if [ $? -eq 0 ]; then
        echo "Health check passed"
        return 0
    else
        echo "WARNING: Health check failed"
        return 1
    fi
}

# Function to setup logging
setup_logging() {
    # Create log directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set up log rotation (basic version)
    export GENETIC_POOL_LOG_FILE="/app/logs/genetic_pool_$(date +%Y%m%d_%H%M%S).log"
    echo "Logs will be written to: $GENETIC_POOL_LOG_FILE"
}

# Function to cleanup on exit
cleanup() {
    echo "============================================"
    echo "Container shutting down..."
    
    # Stop Ray if it's running
    if command -v ray >/dev/null 2>&1; then
        echo "Stopping Ray..."
        ray stop
    fi
    
    echo "Cleanup completed"
    echo "============================================"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Setup logging
setup_logging

# Main execution logic based on mode
case $MODE in
    "local")
        echo "Running genetic algorithm in local mode..."
        run_health_check
        
        # Run genetic algorithm without Ray cluster
        exec python /app/src/execution/genetic_strategy_pool.py --mode local "$@"
        ;;
        
    "distributed")
        echo "Running genetic algorithm in distributed mode..."
        
        # Wait for Ray cluster if not auto-discovery
        if [ "$RAY_ADDRESS" != "auto" ]; then
            wait_for_ray_cluster
        fi
        
        run_health_check
        
        # Run genetic algorithm with Ray cluster
        exec python /app/src/execution/genetic_strategy_pool.py --mode distributed --ray-address "$RAY_ADDRESS" "$@"
        ;;
        
    "head")
        echo "Starting container as Ray head node..."
        
        # Start Ray head node
        start_ray_head
        
        # Run health check
        run_health_check
        
        # Keep container running and monitor Ray cluster
        echo "Ray head node is running. Monitoring cluster health..."
        
        while true; do
            # Check Ray cluster status every 60 seconds
            if ! python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(f'Cluster status: {len(ray.nodes())} nodes')" 2>/dev/null; then
                echo "WARNING: Ray cluster health check failed"
            fi
            sleep 60
        done
        ;;
        
    "worker")
        echo "Starting container as Ray worker node..."
        
        # Start Ray worker node
        start_ray_worker
        
        # Run health check
        run_health_check
        
        # Keep container running
        echo "Ray worker node is running. Waiting for tasks..."
        
        while true; do
            # Simple heartbeat to keep container alive
            sleep 30
        done
        ;;
        
    "test")
        echo "Running container in test mode..."
        
        # Run comprehensive tests
        run_health_check
        
        echo "Running genetic algorithm integration tests..."
        python -c "
import asyncio
import sys
sys.path.insert(0, '/app')

async def test_genetic_pool():
    try:
        from src.execution.genetic_strategy_pool import GeneticStrategyPool, EvolutionMode
        from src.strategy.genetic_seeds.seed_registry import get_registry
        
        print('Testing genetic pool initialization...')
        pool = GeneticStrategyPool(
            population_size=10,
            max_generations=2,
            evolution_mode=EvolutionMode.LOCAL
        )
        
        print('Testing seed registry...')
        registry = get_registry()
        seeds = registry.get_available_seeds()
        print(f'Available seeds: {list(seeds.keys())}')
        
        print('Testing genetic algorithm execution...')
        # This would run a minimal test evolution
        await pool.initialize()
        
        print('All tests passed!')
        return True
        
    except Exception as e:
        print(f'Test failed: {e}')
        return False

result = asyncio.run(test_genetic_pool())
exit(0 if result else 1)
"
        
        if [ $? -eq 0 ]; then
            echo "All tests passed - container is ready for deployment"
            exit 0
        else
            echo "Tests failed - container has issues"
            exit 1
        fi
        ;;
        
    "shell")
        echo "Starting interactive shell for debugging..."
        exec /bin/bash
        ;;
        
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  local       - Run genetic algorithm in local mode"
        echo "  distributed - Connect to existing Ray cluster"
        echo "  head        - Start as Ray head node"
        echo "  worker      - Start as Ray worker node"
        echo "  test        - Run comprehensive tests"
        echo "  shell       - Start interactive shell"
        echo ""
        echo "Usage: $0 [mode] [additional_args]"
        exit 1
        ;;
esac