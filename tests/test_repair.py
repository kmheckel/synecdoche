from __future__ import annotations

from pydantic import BaseModel

from synecdoche import Runtime


class Out(BaseModel):
    v: int


def test_repair_loop_promotes_revised_body(stub_model) -> None:
    # First compile: body that raises at runtime.
    bad_body = "async def solve(x: int) -> dict:\n    raise ValueError('boom')\n"
    # Second compile (repair): body that succeeds.
    good_body = "async def solve(x: int) -> dict:\n    return {'v': x}\n"
    stub_model.extend(
        [
            {"reasoning": "first attempt", "helpers": [], "imports": [], "body": bad_body},
            {"reasoning": "repaired", "helpers": [], "imports": [], "body": good_body},
            # Shadow validation re-runs the revised body — but it uses the
            # same archive entry without re-compiling, so no extra model call.
        ]
    )
    rt = Runtime(model=stub_model.as_model(), heal=True, max_repair_attempts=2)

    @rt.recursion
    def go(x: int) -> Out:
        """."""

    result = go(7)
    assert result.v == 7


def test_repair_disabled_raises_compilation_error(stub_model) -> None:
    bad_body = "async def solve(x: int) -> dict:\n    raise ValueError('boom')\n"
    stub_model.push({"reasoning": "first attempt", "helpers": [], "imports": [], "body": bad_body})
    rt = Runtime(model=stub_model.as_model(), heal=False)

    @rt.recursion
    def go(x: int) -> Out:
        """."""

    import pytest

    from synecdoche import CompilationError

    with pytest.raises(CompilationError):
        go(1)
