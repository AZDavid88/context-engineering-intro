# Pydantic AI - OpenAI Model Integration

**URL**: https://ai.pydantic.dev/models/openai/  
**Purpose**: Complete guide for integrating OpenAI models with Pydantic AI, including configuration, custom clients, and OpenAI-compatible providers  
**Extraction Method**: Brightdata MCP scraping  
**Content Quality Score**: High (comprehensive API documentation with code examples)  

## Installation & Setup

### Basic Installation
```bash
# Install with OpenAI support
pip install "pydantic-ai-slim[openai]"
# OR
uv add "pydantic-ai-slim[openai]"
```

### Environment Configuration
```bash
export OPENAI_API_KEY='your-api-key'
```

## Basic Usage Patterns

### Simple Agent Creation
```python
from pydantic_ai import Agent

# Using string notation
agent = Agent('openai:gpt-4o')

# OR direct model initialization
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o')
agent = Agent(model)
```

### Programmatic Provider Configuration
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'gpt-4o', 
    provider=OpenAIProvider(api_key='your-api-key')
)
agent = Agent(model)
```

## Advanced Configuration

### Custom OpenAI Client
```python
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Custom client with specific settings
client = AsyncOpenAI(max_retries=3)
model = OpenAIModel(
    'gpt-4o', 
    provider=OpenAIProvider(openai_client=client)
)
agent = Agent(model)
```

### Azure OpenAI Integration
```python
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncAzureOpenAI(
    azure_endpoint='...',
    api_version='2024-07-01-preview',
    api_key='your-api-key',
)

model = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=client),
)
agent = Agent(model)
```

## OpenAI Responses API

### Basic Responses API Usage
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model)
```

### Built-in Tools Integration
```python
from openai.types.responses import WebSearchToolParam
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)
model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model=model, model_settings=model_settings)

# Agent can now use web search
result = agent.run_sync('What is the weather in Tokyo?')
```

**Available Built-in Tools:**
- **Web search**: Search the web for latest information
- **File search**: Search uploaded files for relevant information  
- **Computer use**: Perform tasks on computer behalf

## OpenAI-Compatible Providers

### Generic OpenAI-Compatible API
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com',
        api_key='your-api-key'
    ),
)
agent = Agent(model)
```

### DeepSeek Integration
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# Using shorthand
agent = Agent("deepseek:deepseek-chat")

# OR explicit configuration
model = OpenAIModel(
    'deepseek-chat',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
```

### Ollama Local/Remote Usage
```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

class CityLocation(BaseModel):
    city: str
    country: str

# Local Ollama server
ollama_model = OpenAIModel(
    model_name='llama3.2',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

# Remote Ollama server
ollama_remote = OpenAIModel(
    model_name='qwen2.5-coder:7b',
    provider=OpenAIProvider(base_url='http://192.168.1.74:11434/v1'),
)

agent = Agent(ollama_model, output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
```

### Additional Supported Providers

**Azure AI Foundry:**
```python
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIModel(
    'gpt-4o',
    provider=AzureProvider(
        azure_endpoint='your-azure-endpoint',
        api_version='your-api-version',
        api_key='your-api-key',
    ),
)
```

**OpenRouter:**
```python
from pydantic_ai.providers.openrouter import OpenRouterProvider

# Shorthand
agent = Agent("openrouter:google/gemini-2.5-pro-preview")

# OR explicit
model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
```

**Other Providers:**
- **Grok (xAI)**: `GrokProvider` with `grok-2-1212` model
- **GitHub Models**: `GitHubProvider` with prefixed model names like `xai/grok-3-mini`
- **Perplexity**: Generic `OpenAIProvider` with `https://api.perplexity.ai` base URL
- **Fireworks AI**: `FireworksProvider` with model library access
- **Together AI**: `TogetherProvider` with `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`
- **Heroku AI**: `HerokuProvider` with `claude-3-7-sonnet`

## Model Profiles & Customization

### Custom Model Profile for Non-Standard APIs
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com',
        api_key='your-api-key'
    ),
    profile=OpenAIModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        openai_supports_strict_tool_definition=False
    )
)
agent = Agent(model)
```

## Key Integration Points for Multi-Agent Pipeline

1. **Model Flexibility**: Easy switching between OpenAI, Gemini, and other providers
2. **Structured Output**: Perfect for agent-to-agent communication with Pydantic models
3. **Custom Client Support**: Allows for authentication, retries, and custom configurations
4. **Built-in Tool Support**: Web search, file search, and computer use capabilities
5. **Async Support**: Full async/await support for concurrent agent operations

## Environment Variables Summary
```bash
# Core OpenAI
export OPENAI_API_KEY='your-openai-key'

# Alternative providers
export DEEPSEEK_API_KEY='your-deepseek-key'
export OPENROUTER_API_KEY='your-openrouter-key'
export GITHUB_API_KEY='your-github-token'
export HEROKU_INFERENCE_KEY='your-heroku-key'
export HEROKU_INFERENCE_URL='https://us.inference.heroku.com'
```

This comprehensive OpenAI integration documentation provides all necessary patterns for implementing flexible model switching in the multi-agent narrative pipeline.