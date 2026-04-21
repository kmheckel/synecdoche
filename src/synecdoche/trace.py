"""Observability: a small pluggable tracer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Protocol

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


class TreeTracer:
    """Indented tree formatter.

    Each `call_start` opens a frame (increasing indent); the matching
    `call_end` closes it. Tool calls are inline children. Repairs and cache
    events are one-liners at the current indent.
    """

    _ICONS: ClassVar[dict[str, str]] = {
        "call_start": "▶",
        "call_end": "◀",
        "cache_hit": "⚡",
        "compile": "✦",
        "repair": "🔁",
        "tool_call": "→",
        "helper_call": "↳",
    }

    def __init__(self, *, indent: str = "  ", stream: Any = None) -> None:
        self._indent = indent
        self._depth = 0
        self._stream = stream  # None → print()

    def _write(self, line: str) -> None:
        if self._stream is None:
            print(line)
        else:
            self._stream.write(line + "\n")

    def emit(self, event: TraceEvent) -> None:
        icon = self._ICONS.get(event.kind, "·")
        data = event.data
        if event.kind == "call_start":
            self._write(
                f"{self._indent * self._depth}{icon} {event.name}  [{data.get('mode', '')}]"
            )
            self._depth += 1
            return
        if event.kind == "call_end":
            self._depth = max(0, self._depth - 1)
            ok = data.get("ok")
            suffix = "" if ok is None else (" ✓" if ok else f" ✗ {data.get('error', '')}")
            self._write(f"{self._indent * self._depth}{icon} {event.name}{suffix}")
            return
        # Inline events within the current frame.
        args = data.get("args") or ""
        detail = " ".join(f"{k}={v!r}" for k, v in data.items() if k not in ("args",))
        line = f"{self._indent * self._depth}{icon} {event.name}"
        if args:
            line += f"({args})"
        if detail:
            line += f"  {detail}"
        self._write(line)


def resolve_tracer(spec: Any) -> Tracer:
    if spec is None:
        return NullTracer()
    if isinstance(spec, str):
        if spec == "stdout":
            return StdoutTracer()
        if spec == "tree":
            return TreeTracer()
        if spec == "logfire":
            try:
                from .integrations.logfire_tracer import (
                    LogfireTracer,  # type: ignore[import-not-found]
                )
            except ImportError as e:
                raise RuntimeError(
                    "trace='logfire' requires the 'logfire' extra: uv add 'synecdoche[logfire]'"
                ) from e
            return LogfireTracer()
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
