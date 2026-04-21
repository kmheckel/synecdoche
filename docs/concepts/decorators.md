# Decorators

Two decorators, both methods on `Runtime`.

## `@rt.infer` — terminal inference

Resolves via a single typed structured-output call. The model emits the
return value directly; pydantic validates it.

```python
@rt.infer
def classify_sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral."""
```

Per-call override:

```python
@rt.infer(model=AnthropicModel("claude-opus-4-7"))
def detect_subtle_bias(text: str) -> BiasReport: ...
```

## `@rt.recursion` — compiled body

Non-terminal: an LLM compiles a Python body; it runs inside a sandbox; the
artifact is archived and reused on subsequent calls with matching signature
and tool surface.

```python
@rt.recursion
def summarize_codebase(root: Path) -> Summary:
    """Summarize the architecture of a codebase at the given root."""
```

Overrides:

```python
@rt.recursion(
    model=AnthropicModel("claude-sonnet-4-6"),
    max_repair_attempts=4,
    archive_key="custom_key",   # decouple archive identity from signature
)
def list_recent_files(root: Path) -> list[Path]: ...
```

## What's NOT a decorator

- No `@tool` — tools live in your FastMCP servers.
- No `@agent` / `@task` — no orchestration sugar.
- No `@cache` — caching falls out of the archive automatically.

## Inline helpers

Inside a generated `@rt.recursion` body, the compiler may declare its own
helpers — again with `kind="infer"` or `kind="recursion"`. These come
through the `GeneratedBody.helpers` structured field (not AST-walked).

> Inline helper dispatch is tracked in [issue #20](https://github.com/kmheckel/synecdoche/issues/20).
