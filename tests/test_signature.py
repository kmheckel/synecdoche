from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from synecdoche.signature import CallSignature


class Out(BaseModel):
    value: int


def test_signature_captures_params_and_return_type() -> None:
    def fn(x: int, y: str, path: Path) -> Out:
        """Some docstring."""
        ...

    sig = CallSignature.from_function(fn)
    assert sig.qualname.endswith("fn")
    assert sig.docstring == "Some docstring."
    assert [p.name for p in sig.params] == ["x", "y", "path"]
    assert sig.return_type is Out
    assert sig.return_schema  # non-empty JSON schema


def test_signature_hash_is_stable_across_calls() -> None:
    def fn(a: int, b: str = "x") -> int: ...

    sig1 = CallSignature.from_function(fn)
    sig2 = CallSignature.from_function(fn)
    assert sig1.signature_hash == sig2.signature_hash
    assert len(sig1.signature_hash) == 16


def test_signature_hash_differs_when_types_change() -> None:
    def fn_int(x: int) -> int: ...
    def fn_str(x: str) -> int: ...

    a = CallSignature.from_function(fn_int)
    b = CallSignature.from_function(fn_str)
    assert a.signature_hash != b.signature_hash


def test_signature_render_is_python_syntax() -> None:
    def fn(x: int, y: str = "hi") -> int: ...

    sig = CallSignature.from_function(fn)
    line = sig.render()
    assert line.startswith("def fn(")
    assert "x: int" in line
    assert "-> int" in line
