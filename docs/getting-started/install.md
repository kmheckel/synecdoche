# Install + providers

```bash
uv add synecdoche
```

For a specific provider:

```bash
uv add synecdoche[anthropic]   # Anthropic
uv add synecdoche[openai]      # OpenAI (also works for Ollama + OpenRouter)
uv add synecdoche[all]         # everything pydantic-ai supports
```

## Setting the model

Any `pydantic_ai.models.Model` works:

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

rt = Runtime(model=AnthropicModel("claude-sonnet-4-6"))
# or split by role:
rt = Runtime(
    model_recursion=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),
)
```

## Runtime options

See [`Runtime` API reference](../api/runtime.md) for the full list. Highlights:

- `archive="./.archive"` — path-backed SQLite; use `None` for ephemeral.
- `heal=True` + `max_repair_attempts=2` — the exception-driven repair loop.
- `shadow_validate=True` + `shadow_validate_samples=3` — verifies repaired
  bodies don't regress prior successes.
- `max_recursion_depth`, `max_tool_calls_per_frame`, `wall_time_seconds` —
  budgets.
- `trace="tree"` — indented stdout. Also: `"stdout"`, `"logfire"`, or any
  callable `(TraceEvent) -> None`.
