# synecdoche

**JIT AI code synthesis as a functional paradigm.**

Write a typed Python function signature. Decorate it. At first call, an LLM
compiles a body, runs it in a sandbox, archives the result, and self-heals
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
# label='positive' confidence=0.95
```

## The idea

Most agent frameworks are object-oriented at their core: you subclass an
`Agent`, register `@tool`s on it, and thread state through `self`. The
surface area of the framework grows with every new capability.

synecdoche is the opposite bet: **one runtime, two decorators, types
everywhere, MCP for effects, sandbox for execution, exceptions as the
repair signal.**

```
┌──────────────────┐
│  @rt.infer       │  terminal — one structured-output call, returns typed value
│  @rt.recursion   │  non-terminal — LLM emits a Python body, runs in a sandbox
└──────────────────┘
```

Behind the decorators:

- [**pydantic-ai**](https://ai.pydantic.dev/) — provider-agnostic model
  abstraction and structured output. Swap Anthropic for OpenAI, Gemini,
  Ollama, or anything else it supports without touching synecdoche.
- [**FastMCP**](https://gofastmcp.com/) — the tool surface. Mount MCP
  servers on the runtime and generated bodies call them uniformly. No
  `@tool` decorator in synecdoche itself — tool definitions live where
  they belong.
- [**pydantic-monty**](https://pydantic.dev/articles/pydantic-monty) —
  a sandboxed Python interpreter that yields at external calls so the
  host can dispatch them.

## How it works

When you call a `@rt.recursion` function for the first time:

1. The runtime computes a `(signature_hash, tool_surface_hash)` key.
2. Miss in archive → the compiler is invoked with the signature, return
   type schema, tool surface, and inputs. It emits a structured
   `GeneratedBody` containing the `solve` function source.
3. The sandbox assembles imports + type stubs + the body, type-checks it,
   and executes it. Every external call (MCP tool, inline helper) yields
   out of Monty and is dispatched by the host.
4. The return value is validated against the declared return type.
5. On success: archived, promoted, returned.
6. On failure: the exception (with its structured args) becomes the input
   to a *repair* compilation. The revised body is shadow-validated
   against the failing inputs before being promoted.

Subsequent calls: cache hit. The archived body runs directly.

## Installation

```bash
uv add synecdoche
# or
pip install synecdoche
```

For providers:

```bash
uv add "synecdoche[anthropic]"   # or [openai], [all]
```

## Usage

### `@rt.infer` — terminal inference

A single typed structured-output call. No sandbox, no archive.

```python
@rt.infer
def detect_language(text: str) -> Language:
    """Identify the natural language of the given text."""
```

### `@rt.recursion` — compiled body

The compiler synthesizes a body that may call MCP tools and inline
`@ai.infer` / `@ai.recursion` helpers. The body is archived and reused.

```python
@rt.recursion
def summarize_codebase(root: Path) -> Summary:
    """Summarize the architecture of a codebase at the given root."""
```

### Per-call overrides

```python
@rt.infer(model=AnthropicModel("claude-opus-4-7"))
def detect_subtle_bias(text: str) -> BiasReport: ...

@rt.recursion(
    model=AnthropicModel("claude-sonnet-4-6"),
    max_repair_attempts=4,
)
def list_recent_files(root: Path) -> list[Path]: ...
```

### Runtime configuration

```python
rt = Runtime(
    # Either a single model…
    model=AnthropicModel("claude-sonnet-4-6"),
    # …or split by role:
    # model_recursion=AnthropicModel("claude-opus-4-7"),
    # model_infer=AnthropicModel("claude-haiku-4-5"),

    mcp=[
        Client("stdio://mcp-server-filesystem"),
        Client("https://tools.example.com/mcp"),
    ],

    archive="./.archive",           # or path, or custom Archive impl
    heal=True,
    max_repair_attempts=2,
    shadow_validate=True,
    max_recursion_depth=6,
    trace="stdout",                 # or a callable, or a Tracer instance
)
```

## Rich exceptions are the API

The repair compiler is only as good as the information you give it. A raw
`ValueError("something went wrong")` teaches it nothing. A structured
exception carrying `{ "measured": 2_100_000, "limit": 200_000, "at_call":
"derive_summary" }` teaches it exactly what to fix.

Framework-defined exceptions the compiler learns to recognize:

- `ContextWindowExceeded(measured, limit, at_call)`
- `ToolSurfaceDrift(tool_name, expected_schema, actual_schema)`
- `BudgetExceeded(kind, limit, measured)`

Your own exceptions can carry anything — the more structure, the better
the repair.

## Status

This is a **proof of concept** exploring what a functional, type-first
alternative to OOP agent frameworks looks like. Expect the API to evolve.

Milestones implemented:

- [x] M1: Core compiler + sandbox loop.
- [x] M2: SQLite archive + caching + tool-surface invalidation.
- [x] M3: Exception-driven repair with shadow validation.
- [ ] M4: Full observability (Logfire) and archive CLI.

Deferred: soft-signal healing, offline optimization, semantic
cross-signature search, distributed archives, MCP-server export of
compiled functions.

## License

MIT — see [LICENSE](LICENSE).

The `synecdoche` name has been on PyPI since 2023 (originally a JAX/Haiku
hypernetwork experiment); this POC reuses it with the original author's
permission for a rewrite under the same MIT terms.
