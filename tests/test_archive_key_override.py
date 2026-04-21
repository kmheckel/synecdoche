from __future__ import annotations

from pydantic import BaseModel

from synecdoche import Runtime


class Out(BaseModel):
    v: int


def _body(expr: str) -> dict:
    return {
        "reasoning": "r",
        "helpers": [],
        "imports": [],
        "body": f"async def solve(x: int) -> dict:\n    return {{'v': {expr}}}\n",
    }


def test_archive_key_override_isolates_archives(stub_model) -> None:
    """Two functions with the same signature but different archive_keys compile
    independently and don't collide in the archive."""
    stub_model.extend([_body("x * 10"), _body("x * 100")])
    rt = Runtime(model=stub_model.as_model())

    @rt.recursion(archive_key="branch_a")
    def fa(x: int) -> Out:
        """."""

    @rt.recursion(archive_key="branch_b")
    def fb(x: int) -> Out:
        """."""

    assert fa(1).v == 10
    assert fb(1).v == 100

    # Two separate archive entries, one per branch.
    history_a = rt.archive.history("branch_a", "empty")
    history_b = rt.archive.history("branch_b", "empty")
    assert len(history_a) == 1
    assert len(history_b) == 1


def test_archive_key_default_is_signature_hash(stub_model) -> None:
    """Without an override the key should still be the signature hash,
    and a second call should hit the cache."""
    stub_model.push(_body("x + 1"))
    rt = Runtime(model=stub_model.as_model())

    @rt.recursion
    def f(x: int) -> Out:
        """."""

    assert f(1).v == 2
    assert f(5).v == 6  # no second push needed → cache hit
