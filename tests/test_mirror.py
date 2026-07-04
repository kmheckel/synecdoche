"""Champion mirroring: the cache is readable code, not just a database."""

from __future__ import annotations

import synecdoche as syn


def test_champions_are_mirrored_as_readable_python(stub_model, tmp_path) -> None:
    body = "async def solve(x: int) -> int:\n    return x * 2\n"
    stub_model.push({"reasoning": "double it", "imports": [], "body": body})
    be = syn.Backend(model=stub_model.as_model(), archive=tmp_path / "arch")

    @syn.fn(backend=be)
    def double(x: int) -> int:
        """Double x."""

    assert double(4) == 8

    champions = list((tmp_path / "arch" / "champions").glob("*.py"))
    assert len(champions) == 1
    src = champions[0].read_text()
    assert "async def double(x: int)" in src  # renamed from solve
    assert "double it" in src  # the variant's reasoning
    assert "v1 (spawn)" in src
    compile(src, str(champions[0]), "exec")  # valid Python


def test_mirror_follows_promotion(stub_model, tmp_path) -> None:
    bad = "async def solve(x: int) -> int:\n    raise ValueError('boom')\n"
    good = "async def solve(x: int) -> int:\n    return x + 1\n"
    stub_model.extend(
        [
            {"reasoning": "first", "imports": [], "body": bad},
            {"reasoning": "healed", "imports": [], "body": good},
        ]
    )
    be = syn.Backend(model=stub_model.as_model(), archive=tmp_path / "arch")

    @syn.fn(backend=be)
    def inc(x: int) -> int:
        """Increment."""

    assert inc(1) == 2
    champions = list((tmp_path / "arch" / "champions").glob("*.py"))
    assert len(champions) == 1
    src = champions[0].read_text()
    assert "healed" in src
    assert "v2 (mutate <- v1)" in src
