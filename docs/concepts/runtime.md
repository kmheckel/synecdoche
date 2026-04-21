# Runtime

The `Runtime` is the framework's only top-level object. It holds configuration
and hosts the two decorators. Swap runtimes for tests, multi-tenant, or
per-environment.

```python
rt = Runtime(
    # Either a single model…
    model=AnthropicModel("claude-sonnet-4-6"),
    # …or split by role:
    model_recursion=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),

    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.archive",

    heal=True,
    max_repair_attempts=2,
    shadow_validate=True,
    shadow_validate_samples=3,

    max_recursion_depth=6,
    max_tool_calls_per_frame=200,
    wall_time_seconds=30.0,

    trace="tree",
)
```

## Sync vs async

Decorated callables default to sync:

```python
@rt.infer
def classify(text: str) -> Sentiment: ...

classify("hi")                     # works from sync code
```

From async code they're transparently awaitable — the wrapper detects an
active loop and returns a coroutine:

```python
async def main():
    return await classify("hi")
```

## Internals

Under the hood, the Runtime uses `pydantic_ai.Agent` for structured output
— but that's an implementation detail, not user-facing surface.
