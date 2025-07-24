# ControlFlow Documentation (Brightdata + Jina Hybrid Method)

**URL Source:** https://controlflow.ai/welcome
**Extraction Method:** Brightdata MCP + Jina WebFetch processing
**Date:** 2025-07-24

## Overview

ControlFlow is a Python framework for building agentic AI workflows. It enables developers to create sophisticated AI-powered applications with fine-grained control.

## Key Concepts

### Tasks
- Discrete, observable units of work for AI to solve
- Can return structured data using Pydantic models

### Agents
- Specialized AI entities assigned to tasks
- Can collaborate on complex workflows

### Flows
- Orchestrate complex behaviors by combining tasks
- Maintain shared context and message history

## Installation

```python
# Typical installation method (exact command not specified in document)
import controlflow as cf
```

## Basic Usage

### Simple Task Execution

```python
import controlflow as cf

# Basic task execution
result = cf.run("Write a short poem about artificial intelligence")
print(result)
```

### Structured Results

```python
from pydantic import BaseModel

class Poem(BaseModel):
    title: str
    content: str
    num_lines: int

result = cf.run(
    "Write a haiku about AI", 
    result_type=Poem
)
```

### Custom Tools

```python
import controlflow as cf
import random

def roll_dice(num_dice: int) -> list[int]:
    """Roll multiple dice and return the results."""
    return [random.randint(1, 6) for _ in range(num_dice)]

result = cf.run("Roll 3 dice and return the results", tools=[roll_dice])
```

### Multi-Agent Collaboration

```python
scientist = cf.Agent(name="Scientist", instructions="Explain scientific concepts.")
poet = cf.Agent(name="Poet", instructions="Write poetic content.")

result = cf.run(
    "Explain entropy briefly, then write a haiku about it",
    agents=[scientist, poet]
)
```

## Advanced Features

- Structured result types
- Custom tool integration
- Interactive tasks
- Multi-agent workflows
- Flexible task and flow configuration

## Why ControlFlow?

- Seamless AI integration
- Fine-grained control
- Scalable architecture
- Transparent AI decision-making
- Rapid prototyping