# Pydantic AI Introduction

## Overview
Agent Framework / shim to use Pydantic with LLMs

Pydantic AI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI. Built by the Pydantic team (creators of Pydantic Validation used by OpenAI SDK, Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more).

## Key Features

### Why use Pydantic AI
- **Built by the Pydantic Team**: Production-ready validation patterns
- **Model-agnostic**: Supports OpenAI, Anthropic, Gemini, Deepseek, Ollama, Groq, Cohere, and Mistral
- **Pydantic Logfire Integration**: Real-time debugging, performance monitoring, and behavior tracking 
- **Type-safe**: Designed to make type checking as powerful and informative as possible
- **Python-centric Design**: Leverages Python's familiar control flow and agent composition
- **Structured Responses**: Harnesses Pydantic Validation to validate and structure model outputs
- **Dependency Injection System**: Optional system to provide data and services to system prompts, tools and output validators
- **Streamed Responses**: Stream LLM responses continuously with immediate validation
- **Graph Support**: Pydantic Graph provides powerful way to define graphs using typing hints

## Hello World Example

```python
from pydantic_ai import Agent

agent = Agent(
    'google-gla:gemini-1.5-flash',
    system_prompt='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
# "The first known use of "hello, world" was in a 1974 textbook about the C programming language."
```

## Tools & Dependency Injection Example

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn

class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)

support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)

@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"

@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )

async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)
    print(result.output)
    # support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
```

## Core Concepts

### Agent Components
- **System prompt(s)**: Instructions for the LLM written by the developer
- **Function tool(s)**: Functions that the LLM may call to get information while generating a response
- **Structured output type**: The structured datatype the LLM must return at the end of a run
- **Dependency type constraint**: System prompt functions, tools, and output validators may all use dependencies
- **LLM model**: Optional default LLM model associated with the agent
- **Model Settings**: Optional default model settings to help fine tune requests

### Agents are Generic
Agents are generic in their dependency and output types, e.g., `Agent[Foobar, list[str]]`

### Reusable Design
Agents are intended to be instantiated once (frequently as module globals) and reused throughout your application, similar to a small FastAPI app or an APIRouter.

## Implementation Patterns for Multi-Agent Narrative Factory

### Agent Creation Pattern
```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

class PersonaDependencies(BaseModel):
    memory_client: QdrantClient
    persona_name: str
    
class PersonaOutput(BaseModel):
    response: str
    memory_updates: list[str]

persona_agent = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=PersonaDependencies,
    output_type=PersonaOutput,
    system_prompt="You are a {persona_name} in a narrative factory..."
)
```

### Memory Integration Tool Pattern
```python
@persona_agent.tool
async def retrieve_context(
    ctx: RunContext[PersonaDependencies], 
    query: str
) -> str:
    """Retrieve relevant context from the persona's memory."""
    results = await ctx.deps.memory_client.query_points(
        collection_name=f"{ctx.deps.persona_name}_memory",
        query_vector=embed_query(query),
        limit=5
    )
    return format_context(results)

@persona_agent.tool
async def store_memory(
    ctx: RunContext[PersonaDependencies],
    content: str,
    memory_type: str
) -> str:
    """Store new information in the persona's memory."""
    vector = embed_content(content)
    await ctx.deps.memory_client.upsert(
        collection_name=f"{ctx.deps.persona_name}_memory",
        points=[{
            "vector": vector,
            "payload": {"content": content, "type": memory_type}
        }]
    )
    return "Memory stored successfully"
```

### Streaming Pattern for Chat Interfaces
```python
async def stream_persona_response(user_input: str, deps: PersonaDependencies):
    async with persona_agent.run_stream(user_input, deps=deps) as response:
        async for chunk in response.stream_text():
            yield chunk
```

## Key Integration Points for Narrative Factory

1. **Gemini Integration**: Use `'google-gla:gemini-1.5-flash'` model string
2. **Dependency Injection**: Pass Qdrant client, FastEmbed model, persona config through deps
3. **Structured Outputs**: Define Pydantic models for persona responses and memory operations
4. **Tools for Memory**: Create tools for querying and updating Qdrant vector database
5. **Async Support**: All operations support async which is essential for Reflex.dev integration
6. **Type Safety**: Generic typing ensures compile-time safety for dependency and output types