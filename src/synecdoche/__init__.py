"""synecdoche — composable transformations of typed Python functions,
with a neural sequence model as the compiler.

The shape is deliberately that of an array framework, transplanted to
general computing: where JAX transforms numeric functions over arrays and
differentiates them with calculus, synecdoche transforms typed functions
over ordinary Python values and "differentiates" them with language —
critiques in, revised programs out.

    import synecdoche as syn
    from pydantic_ai.models.anthropic import AnthropicModel

    syn.configure(model=AnthropicModel("claude-sonnet-4-6"))

    @syn.jit                      # compile at first call, cache by signature
    def summarize(root: Path) -> Summary:
        \"\"\"Summarize the architecture of the codebase at root.\"\"\"

    @syn.oracle                   # the model *is* the function
    def sentiment(text: str) -> Sentiment:
        \"\"\"Classify sentiment as positive, negative, or neutral.\"\"\"

    labels = syn.vmap(sentiment)(texts)          # concurrent batching

    syn.feedback(summarize, 0.3, "missed the tests directory")
    syn.descend(summarize)                       # critique -> revised program

    report = syn.evolve(summarize, examples)     # the training loop
    print(syn.solidify(summarize))               # the artifact is just Python
"""

from __future__ import annotations

from .archive import Archive, MemoryArchive, Metrics, SqliteArchive, Variant
from .backend import Backend
from .compiler import GeneratedBody, InlineHelper
from .config import configure, current_backend, set_default_backend
from .evolution import Budget, EvolutionReport, Signal, Variation, default_fitness
from .exceptions import (
    BudgetExceeded,
    CompilationError,
    ContextWindowExceeded,
    FrameworkError,
    RepairAttempt,
    SandboxError,
    ToolSurfaceDrift,
    ValidationError,
)
from .fn import Fn
from .sandbox import MontySandbox, Sandbox
from .signature import CallSignature, Param
from .surface import ToolSpec, ToolSurface
from .trace import CallableTracer, NullTracer, StdoutTracer, TraceEvent, Tracer
from .transforms import (
    champion,
    descend,
    evolve,
    feedback,
    jit,
    lineage,
    oracle,
    rollback,
    signals,
    solidify,
    vmap,
)

__version__ = "0.2.0"

__all__ = [
    "Archive",
    "Backend",
    "Budget",
    "BudgetExceeded",
    "CallSignature",
    "CallableTracer",
    "CompilationError",
    "ContextWindowExceeded",
    "EvolutionReport",
    "Fn",
    "FrameworkError",
    "GeneratedBody",
    "InlineHelper",
    "MemoryArchive",
    "Metrics",
    "MontySandbox",
    "NullTracer",
    "Param",
    "RepairAttempt",
    "Sandbox",
    "SandboxError",
    "Signal",
    "SqliteArchive",
    "StdoutTracer",
    "ToolSpec",
    "ToolSurface",
    "ToolSurfaceDrift",
    "TraceEvent",
    "Tracer",
    "ValidationError",
    "Variant",
    "Variation",
    "__version__",
    "champion",
    "configure",
    "current_backend",
    "default_fitness",
    "descend",
    "evolve",
    "feedback",
    "jit",
    "lineage",
    "oracle",
    "rollback",
    "set_default_backend",
    "signals",
    "solidify",
    "vmap",
]
