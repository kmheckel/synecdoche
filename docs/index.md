# synecdoche

**JIT AI code synthesis as a functional paradigm.**

Write a typed Python function signature. Decorate it. At first call an LLM
compiles a body, runs it in a sandbox, archives the artifact, and self-heals
from exceptions on the next run.

No `Agent` classes. No ambient state. No conversation history. Just types
and decorators.

```python
from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel
from synecdoche import Runtime

class Sentiment(BaseModel):
    label: str
    confidence: float

rt = Runtime(model=AnthropicModel("claude-sonnet-4-6"))

@rt.infer
def classify_sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral."""

print(classify_sentiment("I love this!"))
```

## Why

Most agent frameworks are object-oriented: you subclass `Agent`, register
`@tool`s, and thread state through `self`. The surface area grows with every
new capability.

synecdoche is the opposite bet: **one runtime, two decorators, types
everywhere, MCP for effects, sandbox for execution, exceptions as the repair
signal.**

- [**pydantic-ai**](https://ai.pydantic.dev/) — provider-agnostic models and
  structured output. Swap Anthropic for OpenAI, Gemini, Ollama, or anything
  else it supports without touching synecdoche.
- [**FastMCP**](https://gofastmcp.com/) — the tool surface. Mount MCP
  servers on the runtime and generated bodies call them uniformly.
- [**pydantic-monty**](https://pydantic.dev/articles/pydantic-monty) —
  a sandboxed Python interpreter that yields at external calls.

Head to [Getting started](getting-started/quickstart.md) or the
[design spec](design.md).
