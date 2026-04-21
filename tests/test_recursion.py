from __future__ import annotations

from pydantic import BaseModel

from synecdoche import Runtime
from synecdoche.compiler import GeneratedBody
from synecdoche.signature import CallSignature
from synecdoche.surface import ToolSurface


class Summary(BaseModel):
    value: int
    note: str


def test_recursion_end_to_end_with_stub_model(stub_model) -> None:
    # Emit a GeneratedBody whose `body` returns a dict that validates as Summary.
    body_src = "async def solve(x: int) -> dict:\n    return {'value': x * 2, 'note': 'doubled'}\n"
    stub_model.push(
        {
            "reasoning": "Double x and tag it.",
            "helpers": [],
            "imports": [],
            "body": body_src,
        }
    )
    rt = Runtime(model=stub_model.as_model())

    @rt.recursion
    def double(x: int) -> Summary:
        """Double x and tag it."""

    result = double(21)
    assert isinstance(result, Summary)
    assert result.value == 42
    assert result.note == "doubled"


def test_recursion_cache_hit_skips_compile(stub_model) -> None:
    # Only push once — if the runtime calls the compiler twice, the stub raises.
    body_src = "async def solve(x: int) -> dict:\n    return {'value': x + 1, 'note': 'plus'}\n"
    stub_model.push(
        {
            "reasoning": "Add one.",
            "helpers": [],
            "imports": [],
            "body": body_src,
        }
    )
    rt = Runtime(model=stub_model.as_model())

    @rt.recursion
    def inc(x: int) -> Summary:
        """Add one."""

    a = inc(1)
    b = inc(2)
    assert a.value == 2
    assert b.value == 3


def test_assemble_script_appends_await_solve() -> None:
    from synecdoche.sandbox import assemble_script

    body = GeneratedBody(
        reasoning="r",
        helpers=[],
        imports=[],
        body="async def solve(x: int) -> int:\n    return x * 2\n",
    )

    def f(x: int) -> int: ...

    sig = CallSignature.from_function(f)
    surface = ToolSurface(tools=(), tool_surface_hash="empty")
    script = assemble_script(body=body, signature=sig, surface=surface)
    assert "async def solve" in script.source
    assert script.source.rstrip().endswith("await solve(x)")
    assert script.input_names == ["x"]
