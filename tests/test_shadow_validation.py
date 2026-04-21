from __future__ import annotations

from pydantic import BaseModel

from synecdoche import CompilationError, Runtime


class Out(BaseModel):
    v: int


def _body(expr: str) -> dict:
    return {
        "reasoning": "r",
        "helpers": [],
        "imports": [],
        "body": f"async def solve(x: int) -> dict:\n    return {{'v': {expr}}}\n",
    }


def test_shadow_validation_catches_regression_on_prior_success(stub_model) -> None:
    """A repair that 'fixes' the failing input but regresses a prior success
    should be rejected by shadow validation."""
    # First compile: works for x=1 (→ v=10).
    # We'll call it with x=1 first (recorded as success).
    # Then force failure for x=99 by pushing a body that raises when x>=50.
    stub_model.extend(
        [
            # Initial compile — good for small x, raises for big x.
            {
                "reasoning": "",
                "helpers": [],
                "imports": [],
                "body": (
                    "async def solve(x: int) -> dict:\n"
                    "    if x >= 50: raise ValueError('too big')\n"
                    "    return {'v': x * 10}\n"
                ),
            },
            # Repair — passes x=99 (returns v=0) but REGRESSES x=1
            # (also returns v=0 instead of v=10). Shadow validation should
            # catch this regression and reject the revision.
            {
                "reasoning": "",
                "helpers": [],
                "imports": [],
                "body": "async def solve(x: int) -> dict:\n    return {'v': 0}\n",
            },
        ]
    )
    rt = Runtime(model=stub_model.as_model(), heal=True, max_repair_attempts=1)

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    # First call succeeds and records the prior success sample.
    assert f(1).v == 10

    # Second call triggers repair. The revision regresses x=1; shadow
    # validation rejects it → CompilationError.
    import pytest

    with pytest.raises(CompilationError) as excinfo:
        f(99)
    assert "regressed" in str(excinfo.value).lower() or "regress" in str(excinfo.value).lower()


def test_success_samples_are_bounded(stub_model) -> None:
    """Only the last N successes should be kept per (archive_key, surface)."""
    stub_model.push(_body("x + 1"))
    rt = Runtime(model=stub_model.as_model(), shadow_validate_samples=2)

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    for x in range(5):
        f(x)

    # Exactly one key for this signature → we never compile a second time.
    ((_key, samples),) = rt._success_samples.items()
    assert len(samples) == 2  # bounded
    # Most recent at the head.
    assert samples[0][0]["x"] == 4
