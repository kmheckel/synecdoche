# Design Specification: A Recursive, Self-Healing LLM Function Framework

## 1. Overview

This framework lets users write Python functions whose bodies are compiled by an LLM at first call, cached as typed artifacts, executed in a sandboxed environment, and regenerated when they fail. The decorator is the interface; the runtime is the compiler, sandbox, and archive; exceptions are the self-healing signal.

The framework composes three pieces of ecosystem:

- **pydantic-ai** for typed model abstractions, provider-agnostic structured output, and the `CodeModeToolset` integration path.
- **FastMCP** for the external capability surface — all non-pure effects cross the boundary through mounted MCP servers.
- **pydantic/monty** for the sandboxed Python interpreter that executes compiled bodies, with snapshot/resume as the yield-at-syscall mechanism.

The contribution is the layer between them: a single decorator, a two-strategy compiler, a signature-keyed archive of generated bodies, and an exception-driven repair loop.

## 2. Design Principles

**One primitive, two strategies.** A user writes typed function signatures and declares whether each is `recursion` (non-terminal: emits a sandboxed Python body) or `infer` (terminal: resolves via a single typed inference). Nothing else varies at the decorator level.

**Types are the contract.** Every function has a pydantic-validated signature. Arguments in, typed return out. The return type schema goes into the structured-output schema for `infer`, and into the validation step for `recursion`. Exceptions raised by validation are first-class repair signals.

**Explicit state transfer.** Frames communicate by typed arguments and typed returns only. No ambient context, no shared memory, no conversation history. Large resources live behind MCP tools and are referenced by URI/ID arguments; the child dereferences at point of use.

**MCP is the syscall layer.** External capabilities — filesystems, email, databases, bespoke domain tools — are all mounted as MCP servers on the runtime. Generated bodies call them identically. No `@tool` decorator in the framework itself; tool definitions live where they belong, in FastMCP servers.

**Bodies are cached compilation artifacts.** The first call to an `@rt.recursion` function pays compilation cost; subsequent calls with a matching `(signature_hash, tool_surface_hash)` execute the archived body directly. The archive is searchable, versioned, and persists across runs.

**Exceptions drive repair.** When a body raises, the runtime captures the trace, packages a repair context, regenerates the failing frame with the exception information in the prompt, shadow-validates the new body, and promotes on success. The compiler's role is to be more defensive next time.

## 3. Public Interface

See the [README](../README.md) for the current, implemented surface. What follows is the
aspirational full spec.

### 3.1 Runtime Construction

```python
from fastmcp import Client
from pydantic_ai.models.anthropic import AnthropicModel
from synecdoche import Runtime

rt = Runtime(
    model_recursion=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),
    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.archive",
    heal=True,
    max_repair_attempts=2,
    shadow_validate=True,
    max_recursion_depth=6,
    max_tool_calls_per_frame=200,
    trace=None,
)
```

### 3.2 Decorators

```python
@rt.recursion
def summarize_codebase(root: Path) -> Summary:
    """Summarize the architecture of a codebase at the given root."""

@rt.infer
def classify_sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral."""
```

### 3.3 Inline Helper Declaration

Inside a `@rt.recursion` body the compiler can declare further helpers via the structured
output's `helpers` field. Each helper is compiled recursively (`kind="recursion"`) or
resolved via a single typed inference (`kind="infer"`).

## 4. Call Execution

### 4.1 `@rt.infer` — Terminal Inference

Build a prompt from the signature + inputs, invoke the configured `model_infer` with
`output_type=return_type`, validate the response, return it.

### 4.2 `@rt.recursion` — Compiled Body

1. Compute `cache_key = hash(signature_canonical) ⊕ hash(tool_surface)`.
2. Archive lookup. On hit, skip to step 6.
3. Build the `JITContext` and render the compilation prompt.
4. Invoke `model_recursion` with `output_type=GeneratedBody`.
5. Archive the resulting body keyed by `cache_key`.
6. Assemble body into a sandbox instance: imports + helper stubs + `solve`.
7. Execute via the sandbox's start/resume loop.
8. Validate the final return value.
9. Record trace + metrics.
10. Return the value.

Exceptions during 7–8 trigger the repair loop.

## 5. Compilation

### 5.1 JITContext IR

`CallSignature`, `ToolSurface`, `ToolSpec`, `RepairContext`, `JITContext` — see
`src/synecdoche/{signature,surface,jit,repair}.py`.

### 5.2 Structured Output

```python
class InlineHelper(BaseModel):
    kind: Literal["recursion", "infer"]
    name: str
    signature: str
    docstring: str

class GeneratedBody(BaseModel):
    reasoning: str
    helpers: list[InlineHelper] = []
    imports: list[str] = []
    body: str
```

For `@rt.infer` there's no body — structured output is the return type itself.

### 5.3 Prompt Structure

System prompt is constant per runtime (prefix caching). User prompt carries the
call-specific payload. Repair variant appends the `RepairContext`.

## 6. Archive

Keyed by `(signature_hash, tool_surface_hash, version)`. Current version = highest
promoted row. Versioning preserves rollback. SQLite default, pluggable via protocol.

## 7. Repair Loop

Triggered by: sandbox exception, return-type validation failure, or sandbox typecheck
failure at assembly time. Builds a `RepairContext` from the captured trace + exception,
re-invokes the compiler, shadow-validates the revised body, and promotes on success.

Scope is **frame-local** — only the raising frame's body is revised.

Framework-defined exceptions the compiler learns to recognize:

- `ContextWindowExceeded(measured, limit, at_call)`
- `ToolSurfaceDrift(tool_name, expected_schema, actual_schema)`
- `BudgetExceeded(kind, limit, measured)`

## 8. Sandbox Abstraction

Small protocol; default `MontySandbox` wraps `pydantic-monty`. Alternative sandboxes
(Pyodide, containerized Python) plug in behind the same protocol.

## 9. Integration with pydantic-ai

- `Model` abstraction for provider independence.
- Structured output (tool-emulation fallback for providers lacking native JSON Schema).
- `Agent` is used **internally only**; never exposed in our public API.

## 10. Observability

Pluggable `Tracer`: `NullTracer`, `StdoutTracer`, `CallableTracer`. Logfire integration
is opt-in.

## 11. Out of Scope for v1

Soft-signal healing, offline optimization, cross-signature callable archive primitives,
multi-runtime federation, native MCP-export of compiled functions, automatic
parallelization of nested calls.

## 12. Minimal Example

```python
from pathlib import Path
from pydantic import BaseModel
from fastmcp import Client
from pydantic_ai.models.anthropic import AnthropicModel
from synecdoche import Runtime

class Summary(BaseModel):
    top_level_purpose: str

rt = Runtime(
    model_recursion=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),
    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.archive",
    heal=True,
)

@rt.recursion
def summarize_codebase(root: Path) -> Summary:
    """Summarize the architecture of a codebase at the given root."""

if __name__ == "__main__":
    print(summarize_codebase(Path("./my-project")))
```

## 13. Development Roadmap

- **M1** — Core compiler + sandbox loop. ✅
- **M2** — Archive + caching + tool-surface invalidation. ✅
- **M3** — Exception-driven repair. ✅
- **M4** — Observability, archive CLI, Logfire tracer.

See open [issues](https://github.com/kmheckel/synecdoche/issues) for specific work items.
