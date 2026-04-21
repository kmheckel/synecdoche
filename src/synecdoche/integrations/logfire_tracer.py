"""Logfire tracer — maps synecdoche events onto Pydantic Logfire spans.

This integration is lazy-imported from `trace.resolve_tracer` when a user
passes `trace="logfire"`. If `logfire` is not installed, `resolve_tracer`
raises with a helpful install hint before this module is loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..trace import TraceEvent


class LogfireTracer:
    """Tracer that emits Logfire spans / logs for each event.

    `call_start` opens a span; `call_end` closes it. Other events become
    `logfire.info` log records with structured attributes attached to the
    current active span. `logfire.configure()` must be called by the user
    before traces begin (we don't assume or override their config).
    """

    def __init__(self) -> None:
        import logfire  # noqa: F401  — import check

        self._span_stack: list[Any] = []

    def emit(self, event: TraceEvent) -> None:
        import logfire

        if event.kind == "call_start":
            span = logfire.span(
                "synecdoche {name}",
                name=event.name,
                kind=event.data.get("mode", ""),
                depth=event.data.get("depth", 0),
            )
            span.__enter__()
            self._span_stack.append(span)
            return
        if event.kind == "call_end":
            if self._span_stack:
                span = self._span_stack.pop()
                span.__exit__(None, None, None)
            return
        # Inline events attach to the current span.
        logfire.info(
            f"synecdoche {event.kind} {event.name}",
            _tags=["synecdoche"],
            **{k: v for k, v in event.data.items() if not callable(v)},
        )


__all__ = ["LogfireTracer"]
