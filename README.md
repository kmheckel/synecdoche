# synecdoche

**The part stands for the whole.** You write the part — a typed signature
and a sentence of intent. The runtime supplies the whole: a body, compiled
by a neural sequence model, sandboxed, archived, and evolved under
selection pressure from types, exceptions, and feedback.

```python
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel
from synecdoche import Runtime

rt = Runtime(model=AnthropicModel("claude-sonnet-4-6"))

@rt.fn                      # solid — your body, and it self-heals
def parse_semver(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)

@rt.fn                      # synth — the body is grown at first call
def summarize(root: Path) -> Summary:
    """Summarize the architecture of the codebase at root."""

@rt.fn(mode="oracle")       # oracle — the model *is* the body
def sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral."""
```

One decorator. Three phases of the same matter.

## Code is a phase of matter

Most frameworks draw a hard line between the code you write and the code a
model writes. synecdoche's bet is that the line is a phase boundary, not a
wall — and that a function should be able to cross it in both directions
without the call site changing:

| phase | what it is | how it runs |
|---|---|---|
| **solid** | a handwritten body | natively — it's your trusted code |
| **synth** | a generated body | sandboxed, archived, evolvable |
| **oracle** | no body at all | one typed inference per call |

The contract — `def name(params) -> Return:` plus a docstring — is the
fixed point. The implementation is fluid:

- A handwritten body is registered as the **seed** (generation zero) of a
  lineage. When it raises, the exception doesn't unwind to your caller —
  it becomes a signal, and the seed *melts*: the runtime mutates it into a
  sandboxed descendant that fixes the defect and serves the call.
- A synthesized champion can be **frozen** the other way:
  `fn.solidify("fn.py")` renders it back into committable source, with its
  provenance in the header. Decorate it again and it's the seed of the
  next lineage. Metamorphosis is the workflow, not a trick.

## Everything is variation

There is exactly one way code comes into being here — a **variation
operator** applied to parents under signals:

| operator | parents | produced by |
|---|---|---|
| `spawn` | 0 | first call of a synth fn; exploration during `evolve()` |
| `mutate` | 1 | healing (hard signal), `backward()` (soft signals), `evolve()` |
| `cross` | 2 | `evolve()` recombining the top two variants |

"Repair" is not a subsystem; it's `mutate` under an exception. Offline
optimization is not a subsystem; it's the same operators run in a loop
with measured fitness. The compiler has one entrypoint — `vary` — and the
prompt simply renders whichever parents and signals the variation carries.

Every variant lands in the **archive**: a cache (champion lookup skips
compilation), a fossil record (lineage is never rewritten), and a gene
pool (parents and cross-signature style neighbors are drawn from it).

## Exceptions and feedback are gradients

The landscape a body lives on is not differentiable, so synecdoche uses
the two gradient surrogates code actually has:

- **Hard gradients** — exceptions and validation failures, weight −1.
  Structured exceptions are the richest signal you can emit: a
  `BudgetExceeded(kind="tool_calls", limit=200, measured=214)` teaches the
  compiler exactly what to fix. Raise rich exceptions; they are the API.
- **Soft gradients** — graded feedback:

```python
result = summarize(Path("."))
summarize.feedback(0.3, "missed the tests directory entirely")
summarize.backward()        # textual gradient step: mutate under the
                            # accumulated signals, shadow-validate, promote
```

Both kinds flow through the same `Signal` type into the same `mutate`
operator. `backward()` is descent; the archive is the optimizer state.

## Evolution, when you want it deliberate

Online, selection is ambient — failures demote, descendants promote. When
you have examples, run it as an explicit evolutionary loop:

```python
report = summarize.evolve(
    [{"root": Path("./demo-repo")}, {"root": Path("./other-repo")}],
    generations=3,
    population=4,
    score=lambda inputs, out: judge(out),   # optional; default = validated success
)
print(f"champion v{report.champion.version} fitness={report.champion.fitness:.2f}")
```

Each generation breeds `population` offspring — mutations of the fittest,
crossovers of the top two, fresh spawns — scores every one against all
examples, and promotes the overall champion.

## Every function is reflective

`@rt.fn` returns an `Fn`: callable exactly like the function you wrote,
and also an object about itself.

```python
summarize.mode          # 'solid' | 'synth' | 'oracle'
summarize.champion      # the Variant currently serving calls
summarize.lineage()     # every variant ever bred, with operators and parents
summarize.signals()     # everything recorded against the champion
summarize.feedback(s, note)   # soft gradient
summarize.backward()    # gradient step
summarize.evolve(...)   # offline evolution
summarize.solidify()    # freeze the champion into source
summarize.rollback()    # demote the champion to its parent
```

## The machinery

```
contract ──> champion lookup ──> execute (native | sandbox) ──> validate ──> value
                 │ miss                     │ raise
                 ▼                          ▼
               vary(spawn)              signal recorded
                                        vary(mutate) ──> descendant promoted ──> retry
```

- [**pydantic-ai**](https://ai.pydantic.dev/) — provider-agnostic models
  and structured output. Swap Anthropic for OpenAI, Gemini, Ollama, or
  anything else it supports without touching synecdoche.
- [**FastMCP**](https://gofastmcp.com/) — the tool surface. Mount MCP
  servers on the runtime; generated bodies call them uniformly. The
  surface is hashed, so a changed tool schema is a changed world: variants
  bred against the old surface don't leak into the new one.
- [**pydantic-monty**](https://pydantic.dev/articles/pydantic-monty) — a
  sandboxed Python interpreter that yields at external calls so the host
  dispatches them. Generated code never touches your process, your
  filesystem, or the network directly.

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

## Runtime configuration

```python
rt = Runtime(
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
    heal=True,                  # melt failing champions into descendants
    max_repairs=2,              # mutation attempts per call
    shadow_validate=True,       # backward() validates before promoting
    max_recursion_depth=6,
    trace="stdout",             # or a callable, or a Tracer instance
)
```

Per-function overrides:

```python
@rt.fn(model=AnthropicModel("claude-opus-4-7"), max_repairs=4)
def gnarly(root: Path) -> Report:
    """..."""
```

## Status

A **proof of concept** exploring what a functional, type-first,
evolutionary alternative to OOP agent frameworks looks like. Expect the
API to keep evolving — it would be embarrassing if it didn't.

Implemented:

- [x] One decorator across solid / synth / oracle; seeds heal by melting.
- [x] Unified variation (`spawn` / `mutate` / `cross`) with signals as the
      only compile context; SQLite + in-memory population archives with
      lineage, metrics, and signals.
- [x] Soft gradients: `feedback()` / `backward()` with shadow validation.
- [x] Offline evolution: `evolve()` with fitness measurement and champion
      promotion; `solidify()` back to source.

Deferred: inline-helper dispatch inside generated bodies, semantic
neighbor search, full observability (Logfire), archive CLI, distributed
archives, MCP-server export of compiled functions.

## License

MIT — see [LICENSE](LICENSE).

The `synecdoche` name has been on PyPI since 2023 (originally a JAX/Haiku
hypernetwork experiment); this project reuses it with the original
author's permission for a rewrite under the same MIT terms.
