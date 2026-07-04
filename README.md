# synecdoche

**One decorator. You write the spec; the runtime learns the code.**

```python
import synecdoche as syn
from pydantic_ai.models.anthropic import AnthropicModel

syn.configure(model=AnthropicModel("claude-sonnet-4-6"), archive="./.synecdoche")

@syn
def dedupe(records: list[Record]) -> list[Record]:
    """Merge records that refer to the same real-world entity."""
```

First call: a body is generated to meet the spec, type-checked, run in a
sandbox, validated against the return type, and cached. Every call after:
the cached body runs. When it raises, the exception becomes the compile
context for a fixed replacement — failure is selection pressure, not an
error page. And the learned code is never hidden: the current champion for
every function is mirrored to `./.synecdoche/champions/` as an ordinary
`.py` file you can read, diff, and commit.

## Heuristic learning

The design target is what Jiayi Weng's
[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)
calls **heuristic learning**: the standard learning loop — state, action,
feedback, update — where the object being updated is *program structure,
not neural-network weights*. The model is the update channel; the program
is the parameter; history stays explicit, readable, and refactorable
instead of being compressed into a checkpoint.

synecdoche makes that loop a language primitive, per function:

| learning loop | synecdoche |
|---|---|
| parameters | the current body (the *champion*) for each `@syn` spec |
| forward pass | calling the function |
| loss signals | exceptions & validation failures (hard), `syn.feedback(f, score, note)` (soft) |
| update step | `syn.descend(f)` — regenerate the body under its accumulated signals |
| training loop | `syn.evolve(f, examples, score=...)` — generate, measure, select, repeat |
| checkpoint | a readable `.py` file with provenance in the header |

The same loop is what [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
and its successors used to discover new bin-packing heuristics and
mathematical constructions, and what LLM-driven
[heuristic discovery](https://arxiv.org/html/2501.18784v2) does for
planning: programs as hypotheses, an evaluator as fitness, selection over
readable code — evolutionary programming and symbolic regression with a
neural proposal distribution. `examples/heuristic_learning.py` (bin
packing) and `examples/symbolic_regression.py` are exactly these
experiments, each in under a hundred lines.

## The decorator

`@syn` accepts the full range between "I wrote the code" and "I wrote the
intent":

```python
@syn                          # spec only: the body is generated at first call
def choose_shard(key: str, load: dict[str, float]) -> str:
    """Pick the shard for this key, balancing load against locality."""

@syn                          # handwritten body: generation zero of a lineage
def pack(items: list[float], capacity: float) -> list[int]:
    """Assign every item to a bin; use as few bins as possible."""
    ...your naive first-fit, for evolution to beat...
```

Both are the same object afterwards. A handwritten body runs natively, as
written, with zero model involvement — until it raises, at which point the
exception (with all its structured attributes) drives the generation of a
sandboxed descendant that fixes the defect. The call site cannot tell the
difference, and `syn.solidify(f)` renders any champion back into source
you can paste under `@syn` to seed the next lineage.

The cache key is `(signature, tool surface)`: change the types or the
mounted tools and you get a fresh generation. Types are the contract —
every return value is validated against the annotation, whoever wrote the
body.

## Recursion is controlled by construction

Generated code cannot spawn more generated code. A body is flat: it may
call the MCP tools you mounted and one builtin —

```python
await infer(instruction="is this address residential? yes/no", data=text)
```

— a single typed judgment call, for the sub-problems that are perception
rather than algorithm. That's the entire nesting story. Composition of
specs happens in *your* Python, where the call graph is visible, bounded,
and reviewable:

```python
@syn
def extract(doc: str) -> Claims: ...

@syn
def verify(claims: Claims) -> Report: ...

def audit(doc: str) -> Report:      # plain Python — you own the structure
    return verify(extract(doc))
```

No hidden agent trees, no runaway recursion, no depth budget you have to
tune — the hierarchy is exactly what you wrote.

## The cache is code, not a blob

Every variant ever generated is archived in SQLite with its full lineage:
which parent it was varied from, by which operator (`spawn` / `mutate` /
`cross`), what signals were recorded against it, its live success metrics
and measured fitness. On top of that, every *promoted* champion is
mirrored to a source file:

```
.synecdoche/
├── synecdoche.sqlite          # lineage, metrics, signals — the fossil record
└── champions/
    └── pack_3f9a01bc.py       # the code that runs, readable and diffable
```

```python
# Champion for pack (signature 3f9a01bc...)
# v3 (mutate <- v2) — promoted 2026-07-04T18:22:11+00:00
# reasoning: Sort items descending and use best-fit; the v2 failure showed
#   first-fit fragmenting bins on the large-item-late instances.
# Regenerated by synecdoche on every promotion; do not edit in place.
# To take ownership, move the body into your source under @syn.
async def pack(items: list[float], capacity: float) -> list[int]:
    ...
```

Check the directory into git and every promotion shows up as a reviewable
diff. This is the maintainability contract: what the system learned is
always inspectable as ordinary source. (`syn.lineage(f)`, `syn.champion(f)`,
`syn.signals(f)`, and `syn.rollback(f)` give you the same view from code.)

## How is this different from just using a coding agent?

A coding agent is a *session*: you ask for a function, it writes one, the
session ends, and the feedback loop ends with it. synecdoche is the loop
made permanent and attached to the call site:

- **The spec is the durable artifact.** The implementation is disposable
  and regenerable — against today's model, today's tool schemas, today's
  types. A coding agent's output goes stale; a spec doesn't.
- **Feedback arrives where the code runs.** Production exceptions, with
  their structured payloads, land in the archive against the exact variant
  that raised — not in a bug report a human relays into a chat window.
- **Selection is continuous.** Metrics, signals, and fitness accumulate
  per variant; champions are demoted and replaced on evidence, per
  deployment context (the tool-surface hash), without anyone driving.
- **And it degrades gracefully into the coding-agent workflow**: when a
  champion is good, `solidify` it, commit it, own it. The decorator then
  costs nothing but the safety net.

The honest caveat: a coding agent sees your whole repo; a `@syn` body sees
one signature, its tools, and its lineage. This is a bet on narrow scopes
with tight feedback beating broad scopes with none — for leaf functions,
heuristics, and glue, not for architecture.

## The machinery

```
call ──> champion lookup ──> execute (native | sandbox) ──> validate ──> value
             │ miss                      │ raise
             ▼                           ▼
         generate (spawn)          signal recorded
                                   regenerate (mutate) ──> promote ──> retry
```

- [**pydantic-ai**](https://ai.pydantic.dev/) — provider-agnostic models
  and structured output; swap providers without touching synecdoche.
- [**FastMCP**](https://gofastmcp.com/) — the tool surface. Mount MCP
  servers; generated bodies call them uniformly; the surface hash is part
  of the cache key, so a changed tool schema is a changed world.
- [**pydantic-monty**](https://pydantic.dev/articles/pydantic-monty) — a
  sandboxed Python interpreter that yields at external calls so the host
  dispatches them. Generated code never touches your process, filesystem,
  or network directly.

## Install & configure

```bash
uv add synecdoche          # or: pip install synecdoche
uv add "synecdoche[anthropic]"   # provider extras: [openai], [all]
```

```python
syn.configure(
    model=AnthropicModel("claude-sonnet-4-6"),
    # or split by role:
    # model_code=AnthropicModel("claude-opus-4-7"),    # generates bodies
    # model_infer=AnthropicModel("claude-haiku-4-5"),  # answers the infer builtin

    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.synecdoche",    # dir -> SQLite + champions/; omit -> in-memory
    heal=True,                  # regenerate failing champions instead of raising
    max_repairs=2,
    shadow_validate=True,       # descend() validates before promoting
    trace="stdout",
)
```

Every function of the API also takes an explicit `backend=` (tests,
multiple archives): `@syn(backend=be)`, and per-function overrides:
`@syn(model=..., max_repairs=4)`.

## Status

A **proof of concept** of heuristic learning as a language primitive.
Expect the API to keep evolving — it would be embarrassing if it didn't.

Implemented: the `@syn` decorator across handwritten and generated bodies
with healing; flat-body sandbox execution with the `infer` builtin and MCP
tools; the variation core (`spawn`/`mutate`/`cross`) with signals as the
only compile context; SQLite + in-memory archives with lineage, metrics,
signals, and champion mirroring to source files; `feedback`/`descend`;
`evolve` with measured fitness; `vmap`; `solidify`.

Deferred: island-model populations and novelty pressure for `evolve`,
semantic neighbor search, editing mirrored champions back into the lineage,
full observability (Logfire), archive CLI, MCP-server export of learned
functions.

## License

MIT — see [LICENSE](LICENSE).

The `synecdoche` name has been on PyPI since 2023 (originally a JAX/Haiku
hypernetwork experiment); this project reuses it with the original
author's permission for a rewrite under the same MIT terms. The name
still fits: the part — a signature — stands for the whole.
