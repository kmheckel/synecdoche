from __future__ import annotations

from synecdoche.surface import ToolSpec


def test_tool_spec_renders_async_stub_with_types() -> None:
    spec = ToolSpec(
        name="fs_read",
        description="Read a file.",
        params_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "required": ["path"],
        },
        return_schema={"type": "string"},
        source="mcp",
    )
    stub = spec.render_stub()
    assert "async def fs_read(" in stub
    assert "path: str" in stub
    assert "encoding: str | None = None" in stub
    assert "-> str" in stub
    assert "Read a file." in stub


def test_surface_hash_is_stable() -> None:
    t1 = ToolSpec(
        name="a",
        description="",
        params_schema={"type": "object"},
        return_schema={"type": "object"},
        source="mcp",
    )
    t2 = ToolSpec(
        name="b",
        description="",
        params_schema={"type": "object"},
        return_schema={"type": "object"},
        source="mcp",
    )
    from synecdoche.surface import _hash_surface

    assert _hash_surface([t1, t2]) == _hash_surface([t2, t1])


def test_builtins_do_not_perturb_the_surface_hash() -> None:
    # The hash covers MCP tools only: builtins are constant across backends,
    # so their presence must not invalidate archived variants.
    from synecdoche.surface import INFER_SPEC, _hash_surface

    mcp_tool = ToolSpec(
        name="fs_read",
        description="",
        params_schema={"type": "object"},
        return_schema={"type": "object"},
        source="mcp",
    )
    assert _hash_surface([mcp_tool, INFER_SPEC]) == _hash_surface([mcp_tool])
