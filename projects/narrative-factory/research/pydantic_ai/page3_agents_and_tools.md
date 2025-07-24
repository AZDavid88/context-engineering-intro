# Pydantic AI - Agents and Tools for Multi-Persona System

## Agent Architecture Overview

### Agent Components
Each agent is a container for:
- **System prompt(s)**: Instructions for the LLM written by the developer
- **Function tool(s)**: Functions the LLM can call to get information or perform actions
- **Structured output type**: The structured datatype the LLM must return
- **Dependency type constraint**: System prompt functions, tools, and output validators use dependencies
- **LLM model**: Default LLM model associated with the agent
- **Model Settings**: Default model settings for fine-tuning requests

### Agent Generic Types
```python
# Agent[DependencyType, OutputType]
Agent[PersonaDependencies, PersonaOutput]
```

## Running Agents (4 Methods)

### 1. Synchronous Run
```python
result = agent.run_sync('What is the capital of Italy?')
print(result.output)  # Rome
```

### 2. Asynchronous Run (Recommended for Reflex)
```python
result = await agent.run('What is the capital of France?')
print(result.output)  # Paris
```

### 3. Streaming Run (For Chat Interfaces)
```python
async with agent.run_stream('What is the capital of the UK?') as response:
    print(await response.get_output())  # London
```

### 4. Graph Iteration (Advanced Control)
```python
async with agent.iter('User input') as agent_run:
    async for node in agent_run:
        # Process each step of the agent's execution
        if Agent.is_model_request_node(node):
            # Handle model requests
            pass
        elif Agent.is_call_tools_node(node):
            # Handle tool calls
            pass
        elif Agent.is_end_node(node):
            # Final result
            print(agent_run.result.output)
```

## System Prompts vs Instructions

### Use Instructions (Recommended)
```python
# Static instructions
agent = Agent(
    'google-gla:gemini-1.5-flash',
    instructions="You are a Director persona in a narrative factory."
)

# Dynamic instructions (re-evaluated each run)
@agent.instructions
def add_current_context(ctx: RunContext[PersonaDependencies]) -> str:
    return f"Current project: {ctx.deps.project_name}"
```

### Use System Prompts (When preserving history)
```python
# Static system prompt
agent = Agent(
    'google-gla:gemini-1.5-flash',
    system_prompt="You are a Director persona."
)

# Dynamic system prompt
@agent.system_prompt  
def add_persona_context(ctx: RunContext[PersonaDependencies]) -> str:
    return f"Your persona name is {ctx.deps.persona_name}"
```

## Function Tools

### Tool Registration Methods

#### 1. Via Decorator (with context)
```python
@agent.tool
async def retrieve_memory(
    ctx: RunContext[PersonaDependencies], 
    query: str
) -> str:
    """Retrieve relevant memories from Qdrant database."""
    results = await ctx.deps.qdrant_client.query_points(
        collection_name=f"{ctx.deps.persona_name}_memory",
        query_vector=await embed_query(query),
        limit=5
    )
    return format_memory_results(results)
```

#### 2. Via Plain Decorator (no context)
```python
@agent.tool_plain
def generate_random_number() -> int:
    """Generate a random number for story elements."""
    return random.randint(1, 100)
```

#### 3. Via Agent Constructor
```python
def memory_search(ctx: RunContext[PersonaDependencies], query: str) -> str:
    """Search persona memories."""
    # Implementation here
    pass

agent = Agent(
    'google-gla:gemini-1.5-flash',
    tools=[memory_search],
    deps_type=PersonaDependencies
)
```

### Tool Schema Generation
Pydantic AI automatically extracts function signatures and docstrings to create tool schemas:

```python
@agent.tool_plain
def character_interaction(
    character_a: str, 
    character_b: str, 
    interaction_type: str
) -> str:
    """Plan an interaction between two characters.
    
    Args:
        character_a: Name of the first character
        character_b: Name of the second character  
        interaction_type: Type of interaction (dialogue, conflict, collaboration)
    """
    return plan_interaction(character_a, character_b, interaction_type)
```

### Advanced Tool Returns
For rich multi-modal content:

```python
from pydantic_ai.messages import ToolReturn, BinaryContent

@agent.tool_plain
def generate_scene_visualization(scene_description: str) -> ToolReturn:
    """Generate visual representation of a scene."""
    # Generate image
    image_data = create_scene_image(scene_description)
    
    return ToolReturn(
        return_value=f"Scene visualization generated for: {scene_description}",
        content=[
            "Generated scene visualization:",
            BinaryContent(data=image_data, media_type="image/png"),
            "Please analyze this visual and suggest narrative improvements."
        ],
        metadata={
            "scene_type": "narrative_visualization",
            "timestamp": time.time()
        }
    )
```

## Dynamic Tools (Conditional Tool Availability)

### Per-Tool Prepare Method
```python
from pydantic_ai.tools import ToolDefinition

async def only_if_director(
    ctx: RunContext[PersonaDependencies], 
    tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Only include this tool if the persona is Director."""
    if ctx.deps.persona_type == "Director":
        return tool_def
    return None

@agent.tool(prepare=only_if_director)
async def set_story_direction(
    ctx: RunContext[PersonaDependencies],
    direction: str
) -> str:
    """Set the overall story direction (Director only)."""
    # Store direction in Qdrant
    await store_story_direction(ctx.deps.qdrant_client, direction)
    return f"Story direction set: {direction}"
```

### Agent-Wide Tool Preparation
```python
async def prepare_persona_tools(
    ctx: RunContext[PersonaDependencies], 
    tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Filter tools based on persona type and current state."""
    persona_type = ctx.deps.persona_type
    
    # Filter tools based on persona
    filtered_tools = []
    for tool_def in tool_defs:
        if tool_def.name.startswith(persona_type.lower()):
            filtered_tools.append(tool_def)
        elif tool_def.name in SHARED_TOOLS:
            filtered_tools.append(tool_def)
    
    return filtered_tools

agent = Agent(
    'google-gla:gemini-1.5-flash',
    prepare_tools=prepare_persona_tools,
    deps_type=PersonaDependencies
)
```

## Tool Execution and Retries

### Automatic Validation Retries
```python
@agent.tool
async def update_character_info(
    ctx: RunContext[PersonaDependencies],
    character_name: str,
    attribute: str,
    new_value: str
) -> str:
    """Update character information."""
    # Pydantic automatically validates parameters
    # If validation fails, LLM gets retry prompt
    return await update_character(character_name, attribute, new_value)
```

### Manual Retry Logic
```python
from pydantic_ai import ModelRetry

@agent.tool(retries=3)
async def query_external_api(
    ctx: RunContext[PersonaDependencies],
    query: str
) -> str:
    """Query external narrative API."""
    try:
        result = await external_api_call(query)
        if not result:
            raise ModelRetry("No results found, try refining your query")
        return result
    except APIError as e:
        raise ModelRetry(f"API error: {e}. Please try again with different parameters")
```

## Multi-Agent Integration Patterns

### Agent-to-Agent Communication
```python
class NarrativeFactorySystem:
    def __init__(self):
        self.director = create_director_agent()
        self.tactician = create_tactician_agent() 
        self.weaver = create_weaver_agent()
        self.canonist = create_canonist_agent()
        self.librarian = create_librarian_agent()
    
    async def collaborative_story_generation(self, initial_prompt: str):
        # Director sets direction
        direction = await self.director.run(
            f"Set story direction for: {initial_prompt}",
            deps=DirectorDependencies(...)
        )
        
        # Tactician creates plot points
        plot = await self.tactician.run(
            f"Create plot based on direction: {direction.output}",
            deps=TacticianDependencies(...)
        )
        
        # Continue with other personas...
        return collaborative_result
```

### Message History Continuation
```python
# First run with Director
director_result = await director_agent.run(
    "What should be the main theme of our fantasy story?",
    deps=director_deps
)

# Continue conversation with Tactician using history
tactician_result = await tactician_agent.run(
    "How can we implement this theme through character conflicts?",
    message_history=director_result.new_messages(),
    deps=tactician_deps
)
```

## Usage Limits and Error Handling

### Cost Control
```python
from pydantic_ai.usage import UsageLimits

result = await agent.run(
    user_input,
    deps=dependencies,
    usage_limits=UsageLimits(
        response_tokens_limit=2000,
        request_limit=10  # Prevent infinite tool calling
    )
)
```

### Error Capture
```python
from pydantic_ai import UnexpectedModelBehavior, capture_run_messages

with capture_run_messages() as messages:
    try:
        result = await agent.run(user_input, deps=dependencies)
    except UnexpectedModelBehavior as e:
        # Log the full conversation for debugging
        logger.error(f"Agent error: {e}")
        logger.error(f"Messages: {messages}")
        # Handle gracefully
```

## Performance Optimization

### Model Settings
```python
from pydantic_ai.settings import ModelSettings

# Optimize for consistent persona responses
persona_settings = ModelSettings(
    temperature=0.2,  # Lower for consistency
    max_tokens=1500,  # Reasonable limit
    timeout=30.0      # Prevent hanging
)

agent = Agent(
    'google-gla:gemini-1.5-flash',
    model_settings=persona_settings
)
```

### Streaming for Real-time Chat
```python
async def stream_persona_chat(agent, user_input, deps):
    """Stream persona responses for real-time chat interface."""
    async with agent.run_stream(user_input, deps=deps) as response:
        async for chunk in response.stream_text():
            yield chunk  # Send to Reflex UI immediately
```

## Implementation Checklist for Narrative Factory

1. ✅ **Agent Creation**: Each persona gets its own Agent instance
2. ✅ **Tool Registration**: Memory tools for Qdrant operations
3. ✅ **Dependency Injection**: Pass Qdrant client, persona config
4. ✅ **Structured Outputs**: Pydantic models for each persona's output
5. ✅ **Async Operations**: All agent runs use `await` for Reflex
6. ✅ **Error Handling**: Proper exception handling and retries
7. ✅ **Usage Limits**: Prevent runaway costs and infinite loops
8. ✅ **Streaming**: Real-time responses for chat interfaces