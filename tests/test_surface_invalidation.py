from __future__ import annotations

from pydantic import BaseModel

from synecdoche import Runtime
from synecdoche.surface import ToolSpec, ToolSurface


class Out(BaseModel):
    v: int


def _body(expr: str) -> dict:
    return {
        "reasoning": "r",
        "helpers": [],
        "imports": [],
        "body": f"async def solve(x: int) -> dict:\n    return {{'v': {expr}}}\n",
    }


def test_tool_surface_change_invalidates_cache(stub_model) -> None:
    """When the tool_surface_hash changes the archive lookup should miss and
    a fresh compile should fire."""
    stub_model.extend([_body("x * 2"), _body("x * 3")])
    rt = Runtime(model=stub_model.as_model())

    # First surface — empty.
    surface_a = ToolSurface(tools=(), tool_surface_hash="hash_a")
    rt._surface = surface_a

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    assert f(1).v == 2

    # Swap the surface — same empty tool list but a different hash.
    surface_b = ToolSurface(tools=(), tool_surface_hash="hash_b")
    rt._surface = surface_b

    assert f(1).v == 3

    # Both compile invocations consumed — archive has entries under two
    # different tool_surface_hash values.
    assert not stub_model._queue, "stub model should have served both bodies"


def test_same_surface_hits_cache(stub_model) -> None:
    """If the surface is identical across calls, no second compile should fire."""
    stub_model.push(_body("x + 1"))
    rt = Runtime(model=stub_model.as_model())
    rt._surface = ToolSurface(tools=(), tool_surface_hash="stable")

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    assert f(1).v == 2
    assert f(5).v == 6  # cache hit — no extra push needed
    assert not stub_model._queue


# Unused import kept intentionally for future tests with real tool specs.
_ = ToolSpec
