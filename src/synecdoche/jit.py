"""JIT compilation context: the IR handed to the compiler for each frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compiler import GeneratedBody
    from .signature import CallSignature
    from .surface import ToolSurface


@dataclass
class Budget:
    max_recursion_depth: int = 6
    max_tool_calls_per_frame: int = 200
    wall_time_seconds: float | None = None


@dataclass
class TraceFrame:
    kind: str  # "tool" | "helper"
    name: str
    args_summary: str
    result_summary: str


@dataclass
class RepairContext:
    prior_body: GeneratedBody
    exception_type: str
    exception_message: str
    exception_args: dict[str, Any]
    failing_line: int | None
    trace_excerpt: list[TraceFrame]
    prior_attempts: int


@dataclass
class JITContext:
    signature: CallSignature
    surface: ToolSurface
    inputs: dict[str, Any]
    archive_neighbors: tuple[GeneratedBody, ...] = field(default_factory=tuple)
    remaining_depth: int = 6
    budget: Budget = field(default_factory=Budget)
    repair: RepairContext | None = None


__all__ = ["Budget", "JITContext", "RepairContext", "TraceFrame"]
