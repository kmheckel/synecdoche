from __future__ import annotations

import pytest
from pydantic import BaseModel

from synecdoche import BudgetExceeded, Runtime
from synecdoche.exceptions import CompilationError


class Out(BaseModel):
    v: int


def test_max_recursion_depth_raises(stub_model) -> None:
    """If a recursion body internally triggered further recursion at depth > the
    limit, we should surface BudgetExceeded (wrapped in CompilationError).

    We simulate depth by invoking _call_recursion directly with a synthetic
    `depth` above the limit — the public decorator path always enters at 0,
    but this exercises the budget check itself.
    """
    rt = Runtime(model=stub_model.as_model(), max_recursion_depth=2)

    from synecdoche.runtime import _Overrides
    from synecdoche.signature import CallSignature

    def dummy(x: int) -> Out:
        """."""

    sig = CallSignature.from_function(dummy)

    import asyncio

    async def run() -> None:
        # Depth = limit + 1 → should raise BudgetExceeded(kind="depth").
        await rt._call_recursion(sig, {"x": 1}, overrides=_Overrides(), depth=3)

    with pytest.raises(BudgetExceeded) as excinfo:
        asyncio.run(run())
    assert excinfo.value.kind == "depth"
    assert excinfo.value.limit == 2


def test_depth_budget_message_includes_measured(stub_model) -> None:
    rt = Runtime(model=stub_model.as_model(), max_recursion_depth=0)
    from synecdoche.runtime import _Overrides
    from synecdoche.signature import CallSignature

    def dummy(x: int) -> Out:
        """."""

    sig = CallSignature.from_function(dummy)

    import asyncio

    with pytest.raises(BudgetExceeded) as excinfo:
        asyncio.run(rt._call_recursion(sig, {"x": 0}, overrides=_Overrides(), depth=1))
    assert "measured=1" in str(excinfo.value)
    # CompilationError is not raised here — BudgetExceeded is direct.
    assert not isinstance(excinfo.value, CompilationError)
