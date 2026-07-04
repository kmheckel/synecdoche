"""synecdoche — one decorator that generates code to meet a function spec,
and a small set of functions that improve it under feedback.

The module itself is the decorator:

    import synecdoche as syn
    from pydantic_ai.models.anthropic import AnthropicModel

    syn.configure(model=AnthropicModel("claude-sonnet-4-6"))

    @syn                          # generate a body to meet the spec
    def dedupe(records: list[Record]) -> list[Record]:
        \"\"\"Merge records that refer to the same real-world entity.\"\"\"

    @syn                          # or start from your own code — it heals
    def parse_semver(version: str) -> tuple[int, int, int]:
        major, minor, patch = version.split(".")
        return int(major), int(minor), int(patch)

    syn.feedback(dedupe, 0.3, "merged two records that are clearly distinct")
    syn.descend(dedupe)           # critiques -> revised program

    report = syn.evolve(dedupe, examples)   # the training loop
    print(syn.solidify(dedupe))             # the artifact is just Python

This is heuristic learning (Weng, "Learning Beyond Gradients"): the loop
of state, action, feedback, update — where the thing being updated is
program structure, not weights. History stays explicit: every variant is
archived with its lineage, and champions are mirrored to readable ``.py``
files you can diff and commit.
"""

from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any as _Any

from .archive import Archive, MemoryArchive, Metrics, SqliteArchive, Variant, render_champion
from .backend import Backend
from .compiler import GeneratedBody
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
from .surface import INFER_SPEC, ToolSpec, ToolSurface
from .trace import CallableTracer, NullTracer, StdoutTracer, TraceEvent, Tracer
from .transforms import (
    champion,
    descend,
    evolve,
    feedback,
    fn,
    lineage,
    rollback,
    signals,
    solidify,
    vmap,
)

__version__ = "0.2.0"


class _SynModule(_ModuleType):
    """Make the module itself the decorator: ``@syn`` and ``@syn(...)``."""

    def __call__(self, f: _Any = None, **kwargs: _Any) -> _Any:
        return fn(f, **kwargs)


_sys.modules[__name__].__class__ = _SynModule

__all__ = [
    "INFER_SPEC",
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
    "fn",
    "lineage",
    "render_champion",
    "rollback",
    "set_default_backend",
    "signals",
    "solidify",
    "vmap",
]
