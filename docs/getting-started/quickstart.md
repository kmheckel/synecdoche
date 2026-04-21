# Quickstart

Install:

```bash
uv add synecdoche[anthropic]
# or: uv add synecdoche[openai]
```

Set your API key (or use [Ollama](ollama.md) — no key required):

```bash
export ANTHROPIC_API_KEY=sk-...
```

## `@rt.infer` — terminal inference

A single structured-output call. No sandbox, no archive.

```python
from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel
from synecdoche import Runtime

class Language(BaseModel):
    code: str       # ISO 639-1
    name: str
    confidence: float

rt = Runtime(model=AnthropicModel("claude-haiku-4-5"))

@rt.infer
def detect_language(text: str) -> Language:
    """Identify the natural language of the given text."""

print(detect_language("Bonjour, le monde."))
# code='fr' name='French' confidence=0.99
```

## `@rt.recursion` — compiled body

The compiler synthesizes a Python body; it runs in a sandbox; the
resulting artifact is archived and reused on subsequent calls. If the
body raises, the repair loop regenerates it.

```python
from pathlib import Path
from fastmcp import Client
from pydantic_ai.models.anthropic import AnthropicModel
from synecdoche import Runtime

rt = Runtime(
    model_recursion=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),
    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.archive",
    heal=True,
    trace="tree",
)

@rt.recursion
def summarize_codebase(root: Path) -> Summary:
    """Summarize the architecture of a codebase at the given root."""
```

First call pays the compilation cost. Subsequent calls hit the archive.

## Next

- [Concepts / Runtime](../concepts/runtime.md)
- [Concepts / Repair loop](../concepts/repair.md)
- [API reference](../api/runtime.md)
