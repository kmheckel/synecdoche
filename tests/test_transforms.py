"""jit seeds, gradients (feedback/descend), evolve, vmap, solidify."""

from __future__ import annotations

import pytest

import synecdoche as syn

# ---------------------------------------------------------------------------
# jit with a handwritten body: native execution, healing on failure
# ---------------------------------------------------------------------------


def test_handwritten_body_runs_natively_without_model_calls(stub_model) -> None:
    be = syn.Backend(model=stub_model.as_model())  # nothing pushed: any model call raises

    @syn.fn(backend=be)
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42
    champ = syn.champion(double)
    assert champ is not None
    assert champ.operator == "seed"
    assert "return x * 2" in champ.body.body
    assert "@syn" not in champ.body.body  # decorators stripped from the seed


def test_handwritten_body_heals_into_a_descendant(stub_model) -> None:
    # The handwritten seed crashes on negatives; the mutation fixes it.
    healed = "async def solve(x: int) -> int:\n    return abs(x) * 2\n"
    stub_model.push({"reasoning": "handle negatives", "imports": [], "body": healed})
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def double(x: int) -> int:
        if x < 0:
            raise ValueError("negative input")
        return x * 2

    assert double(3) == 6  # native, no model involved
    assert double(-4) == 8  # seed raises -> recompiled -> sandboxed descendant

    lineage = syn.lineage(double)
    assert [v.operator for v in lineage] == ["seed", "mutate"]
    assert lineage[1].parents == (lineage[0].version,)
    assert syn.champion(double).version == lineage[1].version
    assert double(5) == 10  # descendant serves subsequent calls


# ---------------------------------------------------------------------------
# feedback / descend
# ---------------------------------------------------------------------------


def test_feedback_then_descend_revises_the_program(stub_model) -> None:
    first = "async def solve(x: int) -> int:\n    return x + 1\n"
    step = "async def solve(x: int) -> int:\n    return x * 2\n"
    stub_model.extend(
        [
            {"reasoning": "first", "imports": [], "body": first},
            {"reasoning": "doubled per feedback", "imports": [], "body": step},
        ]
    )
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def grow(x: int) -> int:
        """Grow x."""

    assert grow(1) == 2
    syn.feedback(grow, 0.1, "should double, not increment")

    signals = syn.signals(grow)
    assert len(signals) == 1
    assert signals[0].kind == "feedback"
    assert signals[0].weight == pytest.approx(-0.8)

    syn.descend(grow)  # recompile under the critique, shadow-validated
    assert syn.champion(grow).operator == "mutate"
    assert grow(3) == 6


def test_descend_without_signals_is_an_error(stub_model) -> None:
    first = "async def solve(x: int) -> int:\n    return x\n"
    stub_model.push({"reasoning": "r", "imports": [], "body": first})
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def ident(x: int) -> int:
        """."""

    ident(1)
    with pytest.raises(syn.FrameworkError):
        syn.descend(ident)


def test_feedback_score_bounds() -> None:
    with pytest.raises(ValueError):
        syn.Signal.from_feedback(1.5)
    assert syn.Signal.from_feedback(0.5).weight == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------


def test_vmap_over_a_jit_function(stub_model) -> None:
    body = "async def solve(x: int) -> int:\n    return x * 10\n"
    stub_model.push({"reasoning": "r", "imports": [], "body": body})
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def tenfold(x: int) -> int:
        """Multiply by ten."""

    assert syn.vmap(tenfold)([1, 2, 3]) == [10, 20, 30]


def test_vmap_over_a_plain_function() -> None:
    assert syn.vmap(lambda x, y: x + y)([1, 2], 10) == [11, 12]


def test_vmap_respects_concurrency_bound(stub_model) -> None:
    body = "async def solve(x: int) -> int:\n    return x\n"
    stub_model.push({"reasoning": "r", "imports": [], "body": body})
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def ident(x: int) -> int:
        """."""

    ident(0)  # compile once, outside the fan-out
    assert syn.vmap(ident, concurrency=2)(list(range(5))) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------


def test_evolve_selects_the_fittest_variant(stub_model) -> None:
    stub_model.extend(
        [
            # genesis compile: off by one
            {
                "reasoning": "g",
                "helpers": [],
                "imports": [],
                "body": "async def solve(x: int) -> int:\n    return x + 1\n",
            },
            # gen 0, offspring 1 (mutate): worse
            {
                "reasoning": "m",
                "helpers": [],
                "imports": [],
                "body": "async def solve(x: int) -> int:\n    return x + 3\n",
            },
            # gen 0, offspring 2 (spawn): correct
            {
                "reasoning": "s",
                "helpers": [],
                "imports": [],
                "body": "async def solve(x: int) -> int:\n    return x * 2\n",
            },
        ]
    )
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def double(x: int) -> int:
        """Double x."""

    def score(inputs: dict, output: int) -> float:
        return 1.0 if output == inputs["x"] * 2 else 0.0

    report = syn.evolve(double, [{"x": 1}, {"x": 3}], generations=1, population=2, score=score)
    assert report.champion.fitness == pytest.approx(1.0)
    assert report.generations == 1
    assert len(report.evaluated) == 3
    # The champion now serves calls with no further compilation.
    assert double(7) == 14
    assert syn.champion(double).version == report.champion.version


# ---------------------------------------------------------------------------
# solidify
# ---------------------------------------------------------------------------


def test_solidify_renders_committable_source(stub_model, tmp_path) -> None:
    body = "async def solve(x: int) -> int:\n    return x * 2\n"
    stub_model.push({"reasoning": "double it", "imports": [], "body": body})
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def double(x: int) -> int:
        """Double x."""

    double(2)
    out = tmp_path / "double.py"
    src = syn.solidify(double, out)
    assert "async def double(x: int)" in src  # renamed to the function's own name
    assert "# Solidified by synecdoche." in src
    assert "double it" in src  # provenance: the variant's reasoning
    assert out.read_text() == src
    compile(src, str(out), "exec")  # it is valid Python


def test_solidify_of_a_seed_returns_the_handwritten_source(stub_model) -> None:
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def triple(x: int) -> int:
        return x * 3

    triple(1)
    assert "return x * 3" in syn.solidify(triple)
