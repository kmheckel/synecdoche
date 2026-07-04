"""synecdoche — the part stands for the whole.

You write the part: a typed signature and a sentence of intent. The runtime
supplies the whole: a body, compiled by a neural sequence model, sandboxed,
archived, and evolved under selection pressure from types, exceptions, and
feedback.

One decorator covers the whole spectrum of code:

    from pydantic_ai.models.anthropic import AnthropicModel
    from synecdoche import Runtime

    rt = Runtime(model=AnthropicModel("claude-sonnet-4-6"))

    @rt.fn                      # solid — your body is the seed, and it self-heals
    def total(xs: list[float]) -> float:
        return sum(xs)

    @rt.fn                      # synth — the body is grown at first call
    def summarize(root: Path) -> Summary:
        \"\"\"Summarize the architecture of the codebase at root.\"\"\"

    @rt.fn(mode="oracle")       # oracle — the model *is* the body
    def sentiment(text: str) -> Sentiment:
        \"\"\"Classify sentiment as positive, negative, or neutral.\"\"\"

Every decorated function is a reflective handle: ``.champion``,
``.lineage()``, ``.feedback(score, note)``, ``.backward()``,
``.evolve(examples)``, ``.solidify()``.
"""

from __future__ import annotations

from .archive import Archive, MemoryArchive, Metrics, SqliteArchive, Variant
from .compiler import GeneratedBody, InlineHelper
from .evolution import Budget, Signal, Variation, default_fitness
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
from .fn import EvolutionReport, Fn
from .runtime import Runtime
from .sandbox import MontySandbox, Sandbox
from .signature import CallSignature, Param
from .surface import ToolSpec, ToolSurface
from .trace import CallableTracer, NullTracer, StdoutTracer, TraceEvent, Tracer

__version__ = "0.2.0"

__all__ = [
    "Archive",
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
    "Runtime",
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
    "default_fitness",
]
