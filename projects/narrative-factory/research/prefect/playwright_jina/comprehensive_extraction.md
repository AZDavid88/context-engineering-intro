# Prefect v3 Comprehensive Documentation (Playwright + Jina Method)

**Extraction Method:** Playwright navigation + Jina WebFetch processing
**Date:** 2025-07-24
**Target:** Multi-agent narrative pipeline research

## Introduction: Prefect as Orchestration Engine

Prefect is an open-source orchestration engine that turns Python functions into production-grade data pipelines with minimal friction. Key benefits for multi-agent systems:

### Core Philosophy
- **Pythonic**: Write workflows in native Python—no DSLs, YAML, or special syntax
- **Dynamic Runtime**: Create tasks dynamically at runtime based on actual data or conditions
- **State & Recovery**: Robust state management with resume capabilities from failure points
- **Event-Driven**: Trigger flows on schedules, external events, or via API

---

## Quickstart Implementation

### Installation & Setup
```bash
# Install modern Python package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Login to Prefect Cloud for hosted database
uvx prefect-cloud login
```

### Multi-Agent Pipeline Example
```python
from prefect import flow, task
import random

@task
def get_customer_ids() -> list[str]:
    # Simulates Director agent gathering narrative contexts
    return [f"customer{n}" for n in random.choices(range(100), k=10)]

@task
def process_customer(customer_id: str) -> str:
    # Simulates individual persona processing
    return f"Processed {customer_id}"

@flow
def main() -> list[str]:
    # Orchestrates multi-agent collaboration
    customer_ids = get_customer_ids()
    
    # Map tasks across agents (parallel processing)
    results = process_customer.map(customer_ids)
    return results
```

### Deployment Workflow
```bash
# Clone and run locally
git clone https://github.com/PrefectHQ/quickstart && cd quickstart
uv run 01_getting_started.py

# Deploy to cloud infrastructure
uvx prefect-cloud deploy 01_getting_started.py:main \
--name my_first_deployment \
--from https://github.com/PrefectHQ/quickstart

# Execute remotely
uvx prefect-cloud run main/my_first_deployment

# Schedule automatic execution (daily at 8 AM)
uvx prefect-cloud schedule main/my_first_deployment "0 8 * * *"
```

---

## Advanced Workflow Patterns

### 1. Class-Based Agent Workflows
```python
from prefect import flow

class NarrativeAgent:
    @flow
    def instance_workflow(self):
        return "Instance-based agent workflow"

    @flow
    @classmethod  
    def class_workflow(cls):
        return "Class-level agent coordination"

    @flow
    @staticmethod
    def static_workflow():
        return "Stateless utility workflow"
```

### 2. Generator-Based Streaming
```python
@flow
def narrative_stream():
    for chapter in range(10):
        yield f"Chapter {chapter} content"

# Usage for streaming narrative generation
for content in narrative_stream():
    print(content)
```

### 3. Child Flows and Task Organization
```python
from prefect import flow, task
import random

@task
def generate_narrative_seed():
    return random.randint(0, 100)

@flow
def evaluate_narrative_quality(content: str) -> bool:
    # Child flow for quality assessment
    return len(content) > 10

@flow
def narrative_pipeline():
    # Main orchestration flow
    seed = generate_narrative_seed()
    
    if evaluate_narrative_quality(f"Generated content {seed}"):
        print("High quality narrative")
    else:
        print("Needs revision")
```

---

## Comprehensive Configuration

### Flow Configuration Options

```python
@flow(
    description="Multi-agent narrative generation pipeline",
    name="narrative_orchestrator", 
    retries=3,                     # Retry on failure
    retry_delay_seconds=10,        # Wait between retries
    timeout_seconds=300,           # 5-minute timeout
    validate_parameters=True,      # Pydantic validation
    task_runner=ThreadPoolTaskRunner()  # Parallel execution
)
def advanced_narrative_flow():
    # Sophisticated orchestration logic
    pass
```

**Flow Parameters:**
- `description`: Optional string description from docstring
- `name`: Custom flow name (inferred from function if not provided)
- `retries`: Number of retry attempts on failure
- `retry_delay_seconds`: Delay between retry attempts  
- `flow_run_name`: Dynamic run naming (template or function)
- `task_runner`: Task execution strategy (ThreadPool, Process, etc.)
- `timeout_seconds`: Maximum runtime before marking as failed
- `validate_parameters`: Boolean for Pydantic parameter validation
- `version`: Version string (hash of source file if not provided)

### Task Configuration Options

```python
@task(
    name="narrative_processor",
    description="Transform narrative content",  
    tags=["narrative", "processing"],
    timeout_seconds=60,
    retries=2,
    retry_delay_seconds=5,
    cache_policy=INPUTS + TASK_SOURCE,
    cache_expiration=timedelta(hours=1),
    log_prints=True
)
def process_narrative_content(content: str) -> str:
    """Process narrative content with full configuration."""
    return f"Processed: {content}"
```

**Task Parameters:**
- `name`: Custom task identifier  
- `description`: Task description from docstring
- `tags`: Set of tags combined with runtime context
- `timeout_seconds`: Maximum task runtime
- `cache_key_fn`: Callable for generating cache keys
- `cache_policy`: Caching strategy (INPUTS, TASK_SOURCE, RUN_ID, FLOW_PARAMETERS, NO_CACHE)
- `cache_expiration`: Duration for cache validity
- `retries`: Number of retry attempts
- `retry_delay_seconds`: Delay between task retries
- `log_prints`: Boolean for logging print statements

### Dynamic Task Naming

```python
import datetime
from prefect import flow, task

@task(
    name="Agent Task",
    description="Agent-specific processing",
    task_run_name="agent-{agent_name}-on-{date:%A}"
)
def agent_task(agent_name: str, date: datetime.datetime):
    pass

@flow
def multi_agent_flow():
    # Creates run name like "agent-director-on-Thursday"
    agent_task(
        agent_name="director", 
        date=datetime.datetime.now(datetime.timezone.utc)
    )
```

### Function-Based Dynamic Naming

```python
def generate_agent_name():
    date = datetime.datetime.now(datetime.timezone.utc)
    return f"{date:%A}-narrative-session"

@task(
    name="Narrative Agent",
    description="Dynamic narrative processing",
    task_run_name=generate_agent_name
)
def narrative_agent_task(prompt: str):
    pass
```

---

## Essential Patterns for Multi-Agent Systems

### 1. Timeout Management
```python
@flow(timeout_seconds=1, log_prints=True)
def timeout_example():
    print("I will execute")
    time.sleep(5)  # Will timeout after 1 second
    print("I will not execute")
```

### 2. State Tracking and Recovery
- Automatic state tracking for success, failure, retry states
- Resume interrupted runs from last successful point
- Cache expensive computations to avoid rework

### 3. Event-Driven Orchestration  
- Schedule-based triggers
- External event triggers
- API-triggered workflows
- Human intervention support (pause/resume)

### 4. Infrastructure Flexibility
- Local development execution
- Container deployment
- Kubernetes orchestration  
- Cloud service integration
- Infrastructure as code

---

## Multi-Agent Implementation Strategy

### For Your Narrative Pipeline:

1. **Director Agent Flow**: Main orchestration workflow
2. **Persona Task Mapping**: Parallel processing across Director, Tactician, Weaver, Canonist, Librarian
3. **State Management**: Shared context through Qdrant vector database
4. **Error Recovery**: Automatic retry with state preservation
5. **Event Triggers**: API-driven persona interactions
6. **Quality Gates**: Child flows for content validation

### Key Implementation Benefits:
- **90% Performance Improvement** in Prefect 3.0
- **Dynamic Task Creation** for adaptive agent behavior
- **Native Python Control Flow** (if/else, loops, etc.)
- **Built-in Observability** through modern UI
- **Production-Grade State Management**

This comprehensive foundation provides everything needed to orchestrate your five AI personas (Director, Tactician, Weaver, Canonist, Librarian) with robust state management, error recovery, and scalable execution patterns.