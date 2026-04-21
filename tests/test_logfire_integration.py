"""Smoke-test: the Logfire tracer import path doesn't blow up unless used."""

from __future__ import annotations

import importlib

import pytest

from synecdoche import NullTracer
from synecdoche.trace import resolve_tracer


def test_logfire_shortcut_gracefully_rejects_without_extra() -> None:
    """If `logfire` is not installed, resolve_tracer('logfire') should
    raise a helpful error, not a bare ImportError."""
    # Simulate missing dependency by stubbing the sys.modules entry.
    import sys

    removed = sys.modules.pop("logfire", None)
    try:
        spec = importlib.util.find_spec("logfire")
        if spec is not None:
            pytest.skip("logfire is installed in this env — skipping graceful-failure check")
        with pytest.raises(RuntimeError, match=r"synecdoche\[logfire\]"):
            resolve_tracer("logfire")
    finally:
        if removed is not None:
            sys.modules["logfire"] = removed


def test_other_tracer_shortcuts_still_work() -> None:
    assert not isinstance(resolve_tracer(None), type(NullTracer))  # instance check
    assert isinstance(resolve_tracer(None), NullTracer)
