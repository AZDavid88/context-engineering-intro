# ControlFlow Documentation (Playwright + Jina Method)

**URL Source:** https://controlflow.ai/welcome
**Extraction Method:** Playwright navigation + content extraction
**Date:** 2025-07-24

## What is ControlFlow?

ControlFlow is a Python framework for building agentic AI workflows.

An agentic workflow is a process that delegates at least some of its work to an LLM agent. An agent is an autonomous entity that is invoked repeatedly to make decisions and perform complex tasks.

ControlFlow provides a structured, developer-focused framework for defining workflows and delegating work to LLMs, without sacrificing control or transparency:

- Create discrete, observable tasks for an AI to solve.
- Assign one or more specialized AI agents to each task.
- Combine tasks into a flow to orchestrate more complex behaviors.

This task-centric approach allows you to harness the power of AI for complex workflows while maintaining fine-grained control.

## Quickstart

Here's a simple but complete ControlFlow script that writes a poem:

```python
import controlflow as cf

result = cf.run("Write a short poem about artificial intelligence")

print(result)
```

The run() function is the main entry point for ControlFlow. This single line of code creates a task, assigns it to an agent, and immediately executes it, returning the result.

## Key Features

### Structured Results

ControlFlow tasks can return more than just text, including any structured data type supported by Pydantic:

```python
import controlflow as cf
from pydantic import BaseModel

class Poem(BaseModel):
    title: str
    content: str
    num_lines: int

result = cf.run("Write a haiku about AI", result_type=Poem)

print(f"Title: {result.title}")
print(f"Content:\n{result.content}")
print(f"Number of lines: {result.num_lines}")
```

You can also output a list of strings or choose from a list of predefined options:

```python
import controlflow as cf

text = "SpaceX successfully launched 60 Starlink satellites into orbit yesterday."

result = cf.run(
    "Tag the given text with the most appropriate category",
    context=dict(text=text),
    result_type=["Technology", "Science", "Politics", "Entertainment"]
)

print(f"Text: {text}")
print(f"Category: {result}")
```

### Custom Tools

Provide any Python function as a tool for agents to use:

```python
import controlflow as cf
import random

def roll_dice(num_dice: int) -> list[int]:
    """Roll multiple dice and return the results."""
    return [random.randint(1, 6) for _ in range(num_dice)]

result = cf.run("Roll 3 dice and return the results", tools=[roll_dice])

print(result)
```

### Multi-agent Collaboration

Assign multiple agents to a task to enable collaboration:

```python
import controlflow as cf

scientist = cf.Agent(name="Scientist", instructions="Explain scientific concepts.")
poet = cf.Agent(name="Poet", instructions="Write poetic content.")

result = cf.run(
    "Explain entropy briefly, then write a haiku about it",
    agents=[scientist, poet]
)

print(result)
```

### User Interaction

Quickly give agents the ability to chat with users:

```python
import controlflow as cf

name = cf.run("Get the user's name", interactive=True)
```

### Flows

Use flows to create complex workflows by running all tasks with a shared context and message history:

```python
import controlflow as cf

@cf.flow
def create_story():
    # get the topic from the user
    topic = cf.run(
        "Ask the user to provide a topic for a short story", interactive=True
    )

    # choose a genre
    genre_selector = cf.Agent(
        name="GenreSelector",
        instructions="You are an expert at selecting appropriate genres based on prompts.",
    )
    genre = genre_selector.run(
        "Select a genre for a short story",
        result_type=["Science Fiction", "Fantasy", "Mystery"],
        context=dict(topic=topic),
    )

    # choose a setting based on the genre
    if genre == "Science Fiction":
        setting = cf.run("Describe a distant planet in a binary star system")
    elif genre == "Fantasy":
        setting = cf.run("Create a magical floating city in the clouds")
    else:  # Mystery
        setting = cf.run("Design an isolated mansion with secret passages")

    # create a writer agent
    writer = cf.Agent(
        name="StoryWriter", instructions=f"You are an expert {genre} writer."
    )

    # create characters
    characters = writer.run(
        f"Create three unique characters suitable for a the provided genre, setting, and topic.",
        context=dict(genre=genre, setting=setting, topic=topic),
    )

    # write the story
    story = writer.run(
        f"Write a short story using the provided genre, setting, topic, and characters.",
        context=dict(genre=genre, setting=setting, topic=topic, characters=characters),
    )

    return dict(
        topic=topic,
        genre=genre,
        setting=setting,
        characters=characters,
        story=story,
    )

result = create_story()
print(result)
```

## Why ControlFlow?

- 🔗 **Seamless Integration**: Blend AI capabilities with your existing Python codebase effortlessly.
- 🎛️ **Fine-grained Control**: Balance automation with oversight, maintaining control over your AI workflows.
- 📈 **Scalability**: From simple scripts to complex applications, ControlFlow grows with your needs.
- 🔍 **Transparency**: Gain insights into your AI's decision-making process with built-in observability.
- 🚀 **Rapid Prototyping**: Quickly experiment with AI-powered features in your applications.
- 🤝 **Productivity**: Focus on your application logic while ControlFlow handles the intricacies of AI orchestration.

## Next Steps

- Install ControlFlow
- Explore the Core Concepts
- Browse Patterns for common use cases
- Check out the API Reference