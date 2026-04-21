from __future__ import annotations

import io

from synecdoche import CallableTracer, NullTracer, StdoutTracer, TraceEvent, TreeTracer
from synecdoche.trace import resolve_tracer


def test_null_tracer_does_nothing() -> None:
    NullTracer().emit(TraceEvent("anything", "x"))


def test_callable_tracer_forwards() -> None:
    events = []
    t = CallableTracer(events.append)
    t.emit(TraceEvent("call_start", "fn"))
    assert len(events) == 1
    assert events[0].kind == "call_start"


def test_tree_tracer_indents_frames() -> None:
    buf = io.StringIO()
    t = TreeTracer(stream=buf)
    t.emit(TraceEvent("call_start", "outer", {"mode": "recursion"}))
    t.emit(TraceEvent("tool_call", "fs_read", {"args": "path='x'"}))
    t.emit(TraceEvent("call_start", "inner", {"mode": "infer"}))
    t.emit(TraceEvent("call_end", "inner", {"mode": "infer", "ok": True}))
    t.emit(TraceEvent("call_end", "outer", {"mode": "recursion", "ok": True}))
    out = buf.getvalue()
    assert "outer" in out
    assert "  →" in out  # tool_call indented under outer
    assert "  ▶ inner" in out  # inner nested
    assert "✓" in out  # success marker


def test_resolver_recognizes_string_shortcuts() -> None:
    assert isinstance(resolve_tracer(None), NullTracer)
    assert isinstance(resolve_tracer("stdout"), StdoutTracer)
    assert isinstance(resolve_tracer("tree"), TreeTracer)

    calls = []
    t = resolve_tracer(calls.append)
    assert isinstance(t, CallableTracer)
    t.emit(TraceEvent("x", "y"))
    assert len(calls) == 1
