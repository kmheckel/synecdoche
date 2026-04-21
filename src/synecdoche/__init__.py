"""synecdoche — JIT AI code synthesis as a functional paradigm.

Write a typed Python function signature. Decorate it. At first call, an LLM
compiles a body, which runs in a sandbox, is archived, and self-heals from
exceptions. No Agent classes, no ambient state, no conversation history —
just types and decorators.

Quickstart
----------

    from pydantic_ai.models.anthropic import AnthropicModel
    from synecdoche import Runtime

    rt = Runtime(model=AnthropicModel("claude-sonnet-4-6"))

    @rt.infer
    def classify_sentiment(text: str) -> Sentiment:
        \"\"\"Classify sentiment as positive, negative, or neutral.\"\"\"

    print(classify_sentiment("I love this!"))
"""

from __future__ import annotations

from .archive import Archive, ArchiveEntry, ArchiveMetrics, MemoryArchive, SqliteArchive
from .compiler import GeneratedBody, InlineHelper
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
from .runtime import Runtime
from .sandbox import MontySandbox, Sandbox
from .signature import CallSignature, Param
from .surface import ToolSpec, ToolSurface
from .trace import CallableTracer, NullTracer, StdoutTracer, TraceEvent, Tracer

__version__ = "0.1.0"

__all__ = [
    "Archive",
    "ArchiveEntry",
    "ArchiveMetrics",
    "BudgetExceeded",
    "CallSignature",
    "CallableTracer",
    "CompilationError",
    "ContextWindowExceeded",
    "FrameworkError",
    "GeneratedBody",
    "InlineHelper",
    "MemoryArchive",
    "MontySandbox",
    "NullTracer",
    "Param",
    "RepairAttempt",
    "Runtime",
    "Sandbox",
    "SandboxError",
    "SqliteArchive",
    "StdoutTracer",
    "ToolSpec",
    "ToolSurface",
    "ToolSurfaceDrift",
    "TraceEvent",
    "Tracer",
    "ValidationError",
    "__version__",
]
