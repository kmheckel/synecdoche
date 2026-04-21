"""Observability: a small pluggable tracer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from .archive import now_utc


@dataclass
class TraceEvent:
    kind: str  # "call_start" | "call_end" | "tool_call" | "helper_call" | "cache_hit" | "compile" | "repair"
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=now_utc)


class Tracer(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class NullTracer:
    def emit(self, event: TraceEvent) -> None:
        return


class CallableTracer:
    """Delegate every event to a callable — handy for tests and one-liners."""

    def __init__(self, fn: Callable[[TraceEvent], None]) -> None:
        self._fn = fn

    def emit(self, event: TraceEvent) -> None:
        self._fn(event)


class StdoutTracer:
    def emit(self, event: TraceEvent) -> None:
        data = " ".join(f"{k}={v!r}" for k, v in event.data.items())
        print(f"[synecdoche {event.at.isoformat()}] {event.kind} {event.name} {data}")


def resolve_tracer(spec: Any) -> Tracer:
    if spec is None:
        return NullTracer()
    if isinstance(spec, str) and spec == "stdout":
        return StdoutTracer()
    if callable(spec):
        return CallableTracer(spec)
    return spec  # assume already a Tracer


__all__ = [
    "CallableTracer",
    "NullTracer",
    "StdoutTracer",
    "TraceEvent",
    "Tracer",
    "resolve_tracer",
]
