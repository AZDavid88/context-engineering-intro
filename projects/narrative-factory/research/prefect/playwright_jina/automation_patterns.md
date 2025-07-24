# Prefect v3 Automation Patterns (Playwright + Jina Method)

**Source:** https://docs.prefect.io/v3/how-to-guides/automations/creating-automations
**Extraction Method:** Playwright navigation + Jina WebFetch processing
**Date:** 2025-07-24
**Focus:** Event-driven automation for multi-agent narrative pipeline

---

## Automation Architecture

Automations enable **event-driven orchestration** perfect for multi-agent systems:

### Core Components:
1. **Triggers**: Define event conditions that initiate actions
2. **Actions**: Specify responses to trigger conditions  
3. **Details**: Metadata like name, description, and enablement status

### Creation Methods:
- **Web Interface**: Visual automation builder
- **CLI**: YAML/JSON file-based creation
- **Python SDK**: Programmatic automation management

---

## CLI Automation Creation

### Basic Commands
```bash
# Create from YAML file
prefect automation create --from-file automation.yaml
prefect automation create -f automation.yaml

# Create from JSON file  
prefect automation create --from-file automation.json

# Create from JSON string
prefect automation create --from-json '{"name": "my-automation", "trigger": {...}, "actions": [...]}'
prefect automation create -j '{"name": "my-automation", "trigger": {...}, "actions": [...]}'
```

### Single Automation Example (YAML)
```yaml
name: Cancel Long Running Flows
description: Cancels flows running longer than 5 minutes
enabled: true
trigger:
  type: event
  posture: Reactive
  match_state_name: RUNNING
  match_state_duration_s: 300  # 5 minutes
actions:
  - type: cancel-flow-run
```

### Multi-Automation Batch Creation
```yaml
automations:
  - name: Cancel Long Running Flows
    description: Cancels flows running longer than 30 minutes
    enabled: true
    trigger:
      type: event
      posture: Reactive
      expect:
        - prefect.flow-run.Running
      threshold: 1
      for_each:
        - prefect.resource.id
    actions:
      - type: cancel-flow-run

  - name: Notify on Flow Failure
    description: Send notification when flows fail
    enabled: true
    trigger:
      type: event
      posture: Reactive
      expect:
        - prefect.flow-run.Failed
      threshold: 1
    actions:
      - type: send-notification
        subject: "Flow Failed: {{ event.resource.name }}"
        body: "Flow run {{ event.resource.name }} has failed."
```

---

## Python SDK Automation Management

### Complete SDK Example
```python
from prefect.automations import Automation
from prefect.events.schemas.automations import EventTrigger
from prefect.events.actions import CancelFlowRun

# Create automation programmatically
automation = Automation(
    name="woodchonk",
    trigger=EventTrigger(
        expect={"animal.walked"},
        match={
            "genus": "Marmota",
            "species": "monax",
        },
        posture="Reactive",
        threshold=3,
    ),
    actions=[CancelFlowRun()],
).create()

# Read automation by ID
automation = Automation.read(id=automation.id)

# Read automation by name
automation = Automation.read(name="woodchonk")
```

---

## Trigger Configuration Options

### Event Trigger Parameters:
- **`type`**: Event trigger type (typically "event")
- **`posture`**: Response style
  - `"Reactive"`: Respond after events occur
  - `"Proactive"`: Anticipate and prevent events
- **`expect`**: List of expected event types (e.g., `["prefect.flow-run.Failed"]`)
- **`match`**: Event attribute matching criteria
- **`match_state_name`**: Specific workflow state to monitor (e.g., `"RUNNING"`)
- **`match_state_duration_s`**: Duration threshold in seconds
- **`threshold`**: Number of matching events required to trigger
- **`for_each`**: Scoping mechanism for resource-specific actions

### State-Based Triggers for Agent Coordination:
```yaml
# Monitor agent task duration
trigger:
  type: event
  posture: Reactive
  match_state_name: RUNNING
  match_state_duration_s: 600  # 10 minutes
  threshold: 1
```

---

## Action Types and Configuration

### Core Action Types:

#### 1. Cancel Flow Run
```yaml
actions:
  - type: cancel-flow-run
```

#### 2. Send Notification
```yaml
actions:
  - type: send-notification
    subject: "Agent Alert: {{ event.resource.name }}"
    body: "Agent {{ event.resource.name }} requires attention."
```

### Template Variables:
- **`{{ event.resource.name }}`**: Resource identifier
- **`{{ event.resource.id }}`**: Unique resource ID
- Dynamic event data accessible through template syntax

---

## Multi-Agent Pipeline Automation Patterns

### 1. Agent Timeout Management
```yaml
name: Agent Timeout Guardian
description: Cancel agents running longer than expected
trigger:
  type: event
  posture: Reactive
  expect:
    - prefect.flow-run.Running
    - prefect.task-run.Running
  match_state_duration_s: 1800  # 30 minutes
  for_each:
    - prefect.resource.id
actions:
  - type: cancel-flow-run
  - type: send-notification
    subject: "Agent Timeout: {{ event.resource.name }}"
    body: "Agent exceeded 30-minute runtime limit"
```

### 2. Failure Recovery Automation
```yaml
name: Agent Failure Recovery
description: Handle agent failures with notification and cleanup
trigger:
  type: event
  posture: Reactive
  expect:
    - prefect.flow-run.Failed
    - prefect.task-run.Failed
  threshold: 1
actions:
  - type: send-notification
    subject: "Agent Failed: {{ event.resource.name }}"
    body: "Agent {{ event.resource.name }} failed. Review required."
```

### 3. Success Chain Automation
```yaml
name: Agent Success Orchestration
description: Trigger next agent when previous completes
trigger:
  type: event
  posture: Reactive
  expect:
    - prefect.flow-run.Completed
  match:
    flow_name: "director_agent"
  threshold: 1
actions:
  - type: run-deployment
    deployment_name: "tactician_agent"
```

---

## JSON Configuration Format

### Single Automation (JSON)
```json
{
  "name": "Agent Coordination",
  "trigger": {
    "type": "event",
    "posture": "Reactive",
    "expect": ["prefect.flow-run.Completed"],
    "threshold": 1
  },
  "actions": [
    {
      "type": "send-notification",
      "subject": "Agent Ready: {{ event.resource.name }}",
      "body": "Agent completed successfully"
    }
  ]
}
```

### Multiple Automations (JSON Array)
```json
[
  {
    "name": "First Automation",
    "trigger": { ... },
    "actions": [ ... ]
  },
  {
    "name": "Second Automation", 
    "trigger": { ... },
    "actions": [ ... ]
  }
]
```

---

## Advanced Automation Management

### CLI Management Commands:
```bash
# List all automations
prefect automation ls

# Update automation
prefect automation update <automation-id>

# Delete automation
prefect automation delete <automation-id>

# Enable/disable automation
prefect automation enable <automation-id>
prefect automation disable <automation-id>
```

### Integration Options:
- **Terraform Provider**: Infrastructure as code automation management
- **Prefect API**: Direct REST API automation control
- **Web Interface**: Visual automation management and monitoring

---

## Multi-Agent Implementation Strategy

### For Your Narrative Pipeline:

1. **Director Agent Completion** → Trigger Tactician Agent
2. **Persona Task Failures** → Alert and recovery automation  
3. **Long-Running Agents** → Timeout and resource management
4. **Quality Gate Failures** → Notification and retry automation
5. **Context Synchronization** → Database update automation

### Event-Driven Benefits:
- **Reactive Orchestration**: Respond to actual agent states
- **Failure Resilience**: Automatic error handling and recovery
- **Resource Management**: Prevent runaway processes
- **Notification Systems**: Real-time status updates
- **Chain Orchestration**: Sequential agent coordination

This automation framework provides the event-driven foundation needed to orchestrate your five AI personas (Director, Tactician, Weaver, Canonist, Librarian) with intelligent failure handling, timeout management, and seamless agent-to-agent transitions.