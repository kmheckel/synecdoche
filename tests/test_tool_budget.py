from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from synecdoche import BudgetExceeded, Runtime
from synecdoche.jit import Budget
from synecdoche.surface import ToolSpec, ToolSurface


@dataclass
class _FakeResult:
    structured_content: Any = None
    data: Any = None
    content: list = field(default_factory=list)


class Out(BaseModel):
    v: int


class _StubClient:
    """Minimal MCP client stub with async context manager + call_tool."""

    def __init__(self) -> None:
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name: str, kwargs: dict):
        self.calls += 1
        return _FakeResult(structured_content={"ok": True})


def _body_calling_tool_n_times(n: int) -> dict:
    body = (
        f"async def solve(x: int) -> dict:\n"
        f"    total = 0\n"
        f"    for _ in range({n}):\n"
        f"        r = await noop(x=x)\n"
        f"        total += 1\n"
        f"    return {{'v': total}}\n"
    )
    return {"reasoning": "r", "helpers": [], "imports": [], "body": body}


def test_tool_calls_per_frame_budget(stub_model) -> None:
    """A body that calls the tool 10 times should trip the budget when it's 3."""
    stub_model.push(_body_calling_tool_n_times(10))
    rt = Runtime(
        model=stub_model.as_model(),
        max_tool_calls_per_frame=3,
        heal=False,
    )
    rt._mcp_clients = [_StubClient()]
    rt._surface = ToolSurface(
        tools=(
            ToolSpec(
                name="noop",
                description="no-op",
                params_schema={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                return_schema={"type": "object"},
                source="mcp",
                mcp_client_id=0,
            ),
        ),
        tool_surface_hash="noop_hash",
    )

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    with pytest.raises(BudgetExceeded) as excinfo:
        f(1)

    assert excinfo.value.kind == "tool_calls"
    assert excinfo.value.limit == 3


def test_wall_time_budget_config_plumbed_through() -> None:
    """wall_time_seconds reaches the Budget and is readable on Runtime."""
    from pydantic_ai.models.test import TestModel

    rt = Runtime(model=TestModel(), wall_time_seconds=0.25)
    assert rt.budget.wall_time_seconds == 0.25
    assert isinstance(rt.budget, Budget)


def test_wall_time_budget_trips_on_slow_tool(stub_model) -> None:
    """A slow tool should trip the wall-time budget."""
    stub_model.push(_body_calling_tool_n_times(10))

    class _SlowClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def call_tool(self, name, kwargs):
            await asyncio.sleep(0.05)
            return _FakeResult(structured_content={"ok": True})

    rt = Runtime(
        model=stub_model.as_model(),
        wall_time_seconds=0.05,  # tight enough that the 2nd tool call trips
        max_tool_calls_per_frame=20,
        heal=False,
    )
    rt._mcp_clients = [_SlowClient()]
    rt._surface = ToolSurface(
        tools=(
            ToolSpec(
                name="noop",
                description="",
                params_schema={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                return_schema={"type": "object"},
                source="mcp",
                mcp_client_id=0,
            ),
        ),
        tool_surface_hash="slow_hash",
    )

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    with pytest.raises(BudgetExceeded) as excinfo:
        f(1)

    assert excinfo.value.kind == "wall_time"
