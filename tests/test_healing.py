from __future__ import annotations

import pytest
from pydantic import BaseModel

from synecdoche import CompilationError, Runtime


class Out(BaseModel):
    v: int


def test_healing_promotes_mutated_descendant(stub_model) -> None:
    # First compile: body that raises at runtime.
    bad_body = "async def solve(x: int) -> dict:\n    raise ValueError('boom')\n"
    # Second compile (mutation under the exception signal): body that succeeds.
    good_body = "async def solve(x: int) -> dict:\n    return {'v': x}\n"
    stub_model.extend(
        [
            {"reasoning": "first attempt", "helpers": [], "imports": [], "body": bad_body},
            {"reasoning": "healed", "helpers": [], "imports": [], "body": good_body},
        ]
    )
    rt = Runtime(model=stub_model.as_model(), heal=True, max_repairs=2)

    @rt.fn
    def go(x: int) -> Out:
        """."""

    result = go(7)
    assert result.v == 7

    # The lineage records the whole story: spawn -> mutate, with the
    # exception archived as a signal against the failed spawn.
    lineage = go.lineage()
    assert [v.operator for v in lineage] == ["spawn", "mutate"]
    assert lineage[1].parents == (lineage[0].version,)
    failed_signals = rt.archive.signals_for(
        lineage[0].signature_hash, lineage[0].surface_hash, lineage[0].version
    )
    assert any(s.kind == "exception" and "boom" in s.content for s in failed_signals)


def test_healing_disabled_raises_compilation_error(stub_model) -> None:
    bad_body = "async def solve(x: int) -> dict:\n    raise ValueError('boom')\n"
    stub_model.push({"reasoning": "first attempt", "helpers": [], "imports": [], "body": bad_body})
    rt = Runtime(model=stub_model.as_model(), heal=False)

    @rt.fn
    def go(x: int) -> Out:
        """."""

    with pytest.raises(CompilationError):
        go(1)
