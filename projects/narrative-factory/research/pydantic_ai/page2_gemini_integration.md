# Pydantic AI - Gemini Integration

## Model Selection for Narrative Factory
Based on planning PRP, using **Google Gemini API** via Generative Language API

## Installation & Setup

### Installation
No extra dependencies required - just `pydantic-ai` or `pydantic-ai-slim`

### API Key Configuration
Get API key from [aistudio.google.com](https://aistudio.google.com/apikey)

**Environment variable:**
```bash
export GEMINI_API_KEY=your-api-key
```

## Model Usage Patterns

### Basic Agent Creation
```python
from pydantic_ai import Agent

# Method 1: By name (recommended for Narrative Factory)
agent = Agent('google-gla:gemini-1.5-flash')

# Method 2: Direct model initialization
from pydantic_ai.models.gemini import GeminiModel
model = GeminiModel('gemini-1.5-flash', provider='google-gla')
agent = Agent(model)
```

### Provider Prefix Convention
- `google-gla` = Google **G**enerative **L**anguage **A**PI (what we're using)
- `google-vertex` = Vertex AI (enterprise option)

### Custom Provider Configuration
```python
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

model = GeminiModel(
    'gemini-1.5-flash', 
    provider=GoogleGLAProvider(api_key='your-api-key')
)
agent = Agent(model)
```

### Custom HTTP Client
```python
from httpx import AsyncClient
from pydantic_ai.providers.google_gla import GoogleGLAProvider

custom_http_client = AsyncClient(timeout=30)
model = GeminiModel(
    'gemini-1.5-flash',
    provider=GoogleGLAProvider(
        api_key='your-api-key', 
        http_client=custom_http_client
    ),
)
```

## Available Models
- `gemini-1.5-flash` (recommended for PoC - fast and cheap)
- `gemini-2.0-flash` (newer version)
- Full list available in `GeminiModelName`

## Model Settings & Customization

### Basic Model Settings
```python
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings

model_settings = GeminiModelSettings(
    temperature=0.7,
    max_tokens=1000,
)

model = GeminiModel('gemini-1.5-flash')
agent = Agent(model, model_settings=model_settings)
```

### Disable Thinking (if needed)
```python
model_settings = GeminiModelSettings(
    gemini_thinking_config={'thinking_budget': 0}
)
```

### Safety Settings
```python
model_settings = GeminiModelSettings(
    gemini_safety_settings=[
        {
            'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
            'threshold': 'BLOCK_ONLY_HIGH',
        }
    ]
)
```

## Implementation for Narrative Factory Personas

### Director Persona Agent
```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

class DirectorDependencies(BaseModel):
    qdrant_client: QdrantClient
    project_context: dict

class DirectorOutput(BaseModel):
    story_direction: str
    key_themes: list[str]
    next_actions: list[str]

director_agent = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=DirectorDependencies,
    output_type=DirectorOutput,
    system_prompt="""You are the Director persona in a narrative factory.
    Your role is to provide high-level creative direction and maintain
    narrative coherence across the story."""
)

@director_agent.tool
async def analyze_story_structure(
    ctx: RunContext[DirectorDependencies],
    current_chapter: str
) -> str:
    """Analyze the current story structure and provide direction."""
    # Query Qdrant for story context
    results = await ctx.deps.qdrant_client.query_points(
        collection_name="story_structure",
        query_vector=embed_text(current_chapter),
        limit=10
    )
    return analyze_narrative_flow(results)
```

### Tactician Persona Agent
```python
class TacticianDependencies(BaseModel):
    qdrant_client: QdrantClient
    character_database: dict

class TacticianOutput(BaseModel):
    plot_points: list[str]
    character_actions: dict[str, str]
    conflicts: list[str]

tactician_agent = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=TacticianDependencies, 
    output_type=TacticianOutput,
    system_prompt="""You are the Tactician persona in a narrative factory.
    Your role is to design specific plot points, character interactions,
    and dramatic conflicts that serve the Director's vision."""
)

@tactician_agent.tool
async def plan_character_arc(
    ctx: RunContext[TacticianDependencies],
    character_name: str,
    story_phase: str
) -> str:
    """Plan character development for specific story phase."""
    character_history = await ctx.deps.qdrant_client.query_points(
        collection_name="character_arcs",
        query_filter={
            "must": [{"key": "character", "match": {"value": character_name}}]
        },
        limit=20
    )
    return develop_character_arc(character_history, story_phase)
```

### Async Integration Pattern (Critical for Reflex)
```python
async def run_persona_analysis(
    persona_agent: Agent,
    user_input: str,
    dependencies: Any
) -> str:
    """Run persona analysis with proper async handling for Reflex integration."""
    result = await persona_agent.run(user_input, deps=dependencies)
    return result.output

# For streaming responses in chat interfaces
async def stream_persona_response(
    persona_agent: Agent,
    user_input: str, 
    dependencies: Any
):
    """Stream persona response for real-time chat interface."""
    async with persona_agent.run_stream(user_input, deps=dependencies) as response:
        async for chunk in response.stream_text():
            yield chunk
```

## Error Handling for Production

### Model Behavior Errors
```python
from pydantic_ai import UnexpectedModelBehavior, capture_run_messages

with capture_run_messages() as messages:
    try:
        result = await persona_agent.run(user_input, deps=dependencies)
    except UnexpectedModelBehavior as e:
        print(f'Model error: {e}')
        print(f'Messages exchanged: {messages}')
        # Handle error appropriately
```

### Usage Limits (for cost control)
```python
from pydantic_ai.usage import UsageLimits

result = await persona_agent.run(
    user_input,
    deps=dependencies,
    usage_limits=UsageLimits(
        response_tokens_limit=2000,
        request_limit=5
    )
)
```

## Integration Checklist for Narrative Factory

1. ✅ **Model Selection**: Use `'google-gla:gemini-1.5-flash'` 
2. ✅ **API Key**: Set `GEMINI_API_KEY` environment variable
3. ✅ **Async Support**: All operations use `await` for Reflex integration
4. ✅ **Dependency Injection**: Pass Qdrant client through `deps_type`
5. ✅ **Structured Outputs**: Define Pydantic models for each persona output
6. ✅ **Tools Integration**: Create tools for Qdrant memory operations
7. ✅ **Error Handling**: Implement proper exception handling for production
8. ✅ **Streaming Support**: Use `run_stream()` for chat interfaces

## Performance Considerations

- **Temperature**: Lower (0.1-0.3) for consistent persona responses
- **Max Tokens**: Set reasonable limits (1000-2000) for each persona
- **Request Limits**: Prevent infinite loops in tool calling
- **Caching**: Consider implementing response caching for similar queries