# synecdoche

**Composable transformations of typed Python functions, with a neural
sequence model as the compiler.**

The shape is deliberately that of an array framework, transplanted to
general computing. JAX transforms numeric functions over arrays and
differentiates them with calculus. synecdoche transforms typed functions
over ordinary Python values and "differentiates" them with language:
critiques in, revised programs out.

```python
import synecdoche as syn
from pathlib import Path
from pydantic_ai.models.anthropic import AnthropicModel

syn.configure(model=AnthropicModel("claude-sonnet-4-6"))

@syn.jit
def summarize(root: Path) -> Summary:
    """Summarize the architecture of the codebase at root."""

@syn.oracle
def sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral."""

labels = syn.vmap(sentiment)(texts)              # concurrent batching

syn.feedback(summarize, 0.3, "missed the tests directory")
syn.descend(summarize)                           # critique -> revised program

report = syn.evolve(summarize, examples)         # the training loop
print(syn.solidify(summarize))                   # the artifact is just Python
```

## The correspondence

| array computing | synecdoche |
|---|---|
| `jax.jit` — trace to XLA, cache by shapes | `syn.jit` — compile intent (or source) to Python, cache by `(signature, tool surface)` |
| primitives run on a device | bodies run in a sandbox; effects cross only through MCP tools |
| `jax.vmap` — batch over the leading axis | `syn.vmap` — map over the leading argument, concurrently |
| backprop accumulates gradients | `syn.feedback` accumulates critiques (exceptions record themselves) |
| optimizer step updates parameters | `syn.descend` — recompile under the critiques; **the program is the parameter** |
| training loop | `syn.evolve` — breed variants against examples, promote the fittest |
| inspect the jaxpr / lowered HLO | `syn.lineage`, `syn.champion`, `syn.solidify` — except the artifact is ordinary Python you can commit |

There is no `grad`, on purpose: programs, not tensors, are the parameters,
so the derivative of a function with respect to a critique is another
function. `feedback` plays backprop; `descend` plays the optimizer step.

## `syn.jit` — compile whatever you give it

`jit` accepts the full range between "I wrote the code" and "I wrote the
intent":

```python
@syn.jit                      # handwritten body: runs natively, as written
def parse_semver(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)

@syn.jit                      # empty body: compiled from signature + docstring
def dedupe(records: list[Record]) -> list[Record]:
    """Merge records that refer to the same real-world entity."""
```

Both are the same object afterwards. A handwritten body is kept as
generation zero of a lineage; when it raises, the exception — with all its
structured attributes — becomes the compile context for a descendant that
fixes the defect, and the call succeeds instead of unwinding. A compiled
body starts life sandboxed and earns its keep the same way. The call site
cannot tell which is which, and `syn.solidify(f)` closes the loop: it
renders the current body back into source, renamed and provenance-stamped,
ready to commit — at which point it is deterministic code that can seed
the next lineage.

The cache key is `(signature, tool surface)`. Change the types, or mount
different MCP tools, and you get a fresh compilation — retracing, in
effect: the types are the shapes.

## `syn.oracle` — the model as a black-box function

The other end of the spectrum: no code at all. The neural sequence model
is invoked directly as an implementation of the signature, once per call,
its output validated against the declared return type. Use it where the
body would just be "judge this text" anyway.

```python
@syn.oracle
def detect_language(text: str) -> Language:
    """Identify the natural language of the given text."""
```

## Failure is compile context

Every compilation is a **variation**: `spawn` (no parents), `mutate` (one
parent plus the signals recorded against it), `cross` (two parents,
during evolution). "Repair" is not a subsystem — it is `mutate` under a
hard signal.

Signals are the selection pressure, and they come in two strengths:

- **Hard** — exceptions and validation failures, recorded automatically.
  Structured exceptions are the richest signal you can emit: a
  `BudgetExceeded(kind="tool_calls", limit=200, measured=214)` tells the
  compiler exactly what to fix. Raise rich exceptions; they are the API.
- **Soft** — `syn.feedback(f, score, note)`, for the failures that don't
  raise. The note is what the compiler reads; say what was wrong, not
  just how wrong it was.

Both flow through one `Signal` type into the same variation prompt.

Every variant lands in the **archive** — a cache (champion lookup skips
compilation), a fossil record (lineage is never rewritten), and a gene
pool (`evolve` draws parents from it). Inspect it anytime:

```python
syn.champion(f)      # the variant currently serving calls
syn.lineage(f)       # every variant ever compiled: operator, parents, metrics
syn.signals(f)       # everything recorded against the champion
syn.rollback(f)      # demote the champion to its parent
```

## `syn.evolve` — the training loop

When you have examples, selection becomes deliberate:

```python
report = syn.evolve(
    dedupe,
    examples=[{"records": batch1}, {"records": batch2}],
    generations=3,
    population=4,
    score=lambda inputs, out: judge(out),   # optional; default = validated success
)
print(f"v{report.champion.version} fitness={report.champion.fitness:.2f}")
```

Each generation breeds `population` offspring — mutations of the fittest,
crossovers of the top two, fresh spawns — scores every one against all
examples, and promotes the overall champion.

## The machinery

```
call ──> champion lookup ──> execute (native | sandbox) ──> validate ──> value
             │ miss                      │ raise
             ▼                           ▼
         vary(spawn)               signal recorded
                                   vary(mutate) ──> descendant promoted ──> retry
```

- [**pydantic-ai**](https://ai.pydantic.dev/) — provider-agnostic models
  and structured output. Swap Anthropic for OpenAI, Gemini, Ollama, or
  anything else it supports without touching synecdoche.
- [**FastMCP**](https://gofastmcp.com/) — the tool surface. Mount MCP
  servers on the backend; compiled bodies call them uniformly. The surface
  is hashed into the cache key, so a changed tool schema is a changed
  world.
- [**pydantic-monty**](https://pydantic.dev/articles/pydantic-monty) — a
  sandboxed Python interpreter that yields at external calls so the host
  dispatches them. Compiled code never touches your process, filesystem,
  or network directly.

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

## Configuration

`configure()` builds the default backend; every transform also accepts an
explicit `backend=` for isolation (tests, multiple archives):

```python
syn.configure(
    # Either a single model…
    model=AnthropicModel("claude-sonnet-4-6"),
    # …or split by role:
    # model_code=AnthropicModel("claude-opus-4-7"),     # compiles bodies
    # model_oracle=AnthropicModel("claude-haiku-4-5"),  # answers oracles

    mcp=[
        Client("stdio://mcp-server-filesystem"),
        Client("https://tools.example.com/mcp"),
    ],

    archive="./.archive",       # path -> SQLite; omit -> in-memory; or your own Archive
    heal=True,                  # recompile failing champions instead of raising
    max_repairs=2,              # recompile attempts per call
    shadow_validate=True,       # descend() validates before promoting
    max_recursion_depth=6,
    trace="stdout",             # or a callable, or a Tracer instance
)
```

Per-function overrides:

```python
@syn.jit(model=AnthropicModel("claude-opus-4-7"), max_repairs=4)
def gnarly(root: Path) -> Report:
    """..."""
```

## Status

A **proof of concept**. The bet: the array-framework recipe — pure typed
functions plus a small set of composable transformations, with an
optimizing compiler underneath — is the right shape for LLM-backed
computing too, and the "agent framework" (classes, tool registries,
conversation state) is the detour.

Implemented:

- [x] `jit` across handwritten and compiled bodies; seeds heal on
      exception; single-flight compilation under concurrency.
- [x] `oracle` and `vmap` (concurrent, bounded fan-out).
- [x] One variation path (`spawn` / `mutate` / `cross`) with signals as
      the only compile context; SQLite + in-memory archives with lineage,
      metrics, and per-variant signals.
- [x] `feedback` / `descend` with shadow validation; `evolve` with
      measured fitness and champion promotion; `solidify` back to source.

Deferred: inline-helper dispatch inside compiled bodies, semantic
neighbor search, full observability (Logfire), archive CLI, distributed
archives, MCP-server export of compiled functions.

## License

MIT — see [LICENSE](LICENSE).

The `synecdoche` name has been on PyPI since 2023 (originally a JAX/Haiku
hypernetwork experiment); this project reuses it with the original
author's permission for a rewrite under the same MIT terms. The name
still fits: the part — a signature — stands for the whole.
