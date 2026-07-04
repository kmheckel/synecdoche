from __future__ import annotations

from pydantic import BaseModel

import synecdoche as syn
from synecdoche.compiler import GeneratedBody
from synecdoche.signature import CallSignature
from synecdoche.surface import ToolSurface


class Summary(BaseModel):
    value: int
    note: str


def test_jit_compiles_at_first_call(stub_model) -> None:
    # Emit a GeneratedBody whose `body` returns a dict that validates as Summary.
    body_src = "async def solve(x: int) -> dict:\n    return {'value': x * 2, 'note': 'doubled'}\n"
    stub_model.push(
        {"reasoning": "Double x and tag it.", "helpers": [], "imports": [], "body": body_src}
    )
    be = syn.Backend(model=stub_model.as_model())

    @syn.jit(backend=be)
    def double(x: int) -> Summary:
        """Double x and tag it."""

    result = double(21)
    assert isinstance(result, Summary)
    assert result.value == 42
    assert result.note == "doubled"

    champ = syn.champion(double)
    assert champ is not None
    assert champ.operator == "spawn"
    assert champ.parents == ()


def test_jit_cache_hit_skips_compile(stub_model) -> None:
    # Only push once — if the backend calls the compiler twice, the stub raises.
    body_src = "async def solve(x: int) -> dict:\n    return {'value': x + 1, 'note': 'plus'}\n"
    stub_model.push({"reasoning": "Add one.", "helpers": [], "imports": [], "body": body_src})
    be = syn.Backend(model=stub_model.as_model())

    @syn.jit(backend=be)
    def inc(x: int) -> Summary:
        """Add one."""

    a = inc(1)
    b = inc(2)
    assert a.value == 2
    assert b.value == 3


def test_default_backend_via_configure(stub_model) -> None:
    body_src = "async def solve(x: int) -> dict:\n    return {'value': x, 'note': 'id'}\n"
    stub_model.push({"reasoning": "r", "helpers": [], "imports": [], "body": body_src})
    syn.configure(model=stub_model.as_model())
    try:

        @syn.jit
        def ident(x: int) -> Summary:
            """."""

        assert ident(9).value == 9
    finally:
        syn.set_default_backend(None)


def test_no_backend_is_a_clear_error() -> None:
    @syn.jit
    def orphan(x: int) -> int:
        """."""

    import pytest

    with pytest.raises(syn.FrameworkError, match="configure"):
        orphan(1)


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
    surface = ToolSurface(tools=(), surface_hash="empty")
    script = assemble_script(body=body, signature=sig, surface=surface)
    assert "async def solve" in script.source
    assert script.source.rstrip().endswith("await solve(x)")
    assert script.input_names == ["x"]
