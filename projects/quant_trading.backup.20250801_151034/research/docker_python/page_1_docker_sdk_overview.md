# Docker SDK for Python - Official Documentation

## Overview
A Python library for the Docker Engine API that lets you do anything the `docker` command does, but from within Python apps – run containers, manage containers, manage Swarms, etc.

## Installation
```bash
pip install docker
```

## Basic Usage

### Client Connection
```python
import docker
client = docker.from_env()
```

### Container Operations
```python
# Run containers
>>> client.containers.run("ubuntu", "echo hello world")
'hello world\n'

# Run containers in background
>>> client.containers.run("bfirsh/reticulate-splines", detach=True)
<Container '45e6d2de7c54'>

# List containers
>>> client.containers.list()
[<Container '45e6d2de7c54'>, <Container 'db18e4f20eaa'>, ...]

# Get specific container
>>> container = client.containers.get('45e6d2de7c54')

# Container attributes
>>> container.attrs['Config']['Image']
"bfirsh/reticulate-splines"

# Container logs
>>> container.logs()
"Reticulating spline 1...\n"

# Stop container
>>> container.stop()
```

### Streaming Logs
```python
>>> for line in container.logs(stream=True):
...   print(line.strip())
Reticulating spline 2...
Reticulating spline 3...
...
```

### Image Management
```python
# Pull images
>>> client.images.pull('nginx')
<Image 'nginx'>

# List images
>>> client.images.list()
[<Image 'ubuntu'>, <Image 'nginx'>, ...]
```

## Key Components
- **Client**: Main interface to Docker daemon
- **Containers**: Container management operations
- **Images**: Image management operations
- **Networks**: Network management
- **Volumes**: Volume management
- **Services**: Docker Swarm services
- **Secrets**: Docker Swarm secrets
- **Configs**: Docker Swarm configs

## Integration Potential for Genetic Algorithm Infrastructure
This SDK can be used in our `infrastructure_manager.py` to:
- Programmatically build genetic algorithm containers
- Deploy Ray worker containers dynamically
- Monitor container health and resource usage
- Scale Ray clusters up/down based on genetic workload
- Manage container networking for distributed genetic operations