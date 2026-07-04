"""The unified @rt.fn decorator: phases of matter, gradients, evolution."""

from __future__ import annotations

import pytest

from synecdoche import FrameworkError, Runtime

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def test_empty_bodies_become_synth(stub_model) -> None:
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def a(x: int) -> int:
        """Docstring only."""

    @rt.fn
    def b(x: int) -> int: ...

    @rt.fn
    def c(x: int) -> int:
        pass

    assert a.mode == b.mode == c.mode == "synth"


def test_real_bodies_become_solid(stub_model) -> None:
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def total(xs: list[float]) -> float:
        """Sum the list."""
        return sum(xs)

    assert total.mode == "solid"


# ---------------------------------------------------------------------------
# Solid: handwritten code runs natively and is the seed of the lineage
# ---------------------------------------------------------------------------


def test_solid_runs_natively_without_model_calls(stub_model) -> None:
    rt = Runtime(model=stub_model.as_model())  # nothing pushed: any model call raises

    @rt.fn
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42
    champ = double.champion
    assert champ is not None
    assert champ.operator == "seed"
    assert "return x * 2" in champ.body.body
    assert "@rt.fn" not in champ.body.body  # decorators stripped from the seed


def test_solid_heals_by_melting_into_a_descendant(stub_model) -> None:
    # The handwritten seed crashes on negatives; the mutation fixes it.
    healed = "async def solve(x: int) -> int:\n    return abs(x) * 2\n"
    stub_model.push({"reasoning": "handle negatives", "helpers": [], "imports": [], "body": healed})
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def double(x: int) -> int:
        if x < 0:
            raise ValueError("negative input")
        return x * 2

    assert double(3) == 6  # native, no model involved
    assert double(-4) == 8  # seed raises -> mutate -> sandboxed descendant

    lineage = double.lineage()
    assert [v.operator for v in lineage] == ["seed", "mutate"]
    assert lineage[1].parents == (lineage[0].version,)
    assert double.champion.version == lineage[1].version
    assert double(5) == 10  # descendant serves subsequent calls


# ---------------------------------------------------------------------------
# Gradients: feedback + backward
# ---------------------------------------------------------------------------


def test_feedback_then_backward_takes_a_gradient_step(stub_model) -> None:
    spawn = "async def solve(x: int) -> int:\n    return x + 1\n"
    step = "async def solve(x: int) -> int:\n    return x * 2\n"
    stub_model.extend(
        [
            {"reasoning": "first", "helpers": [], "imports": [], "body": spawn},
            {"reasoning": "doubled per feedback", "helpers": [], "imports": [], "body": step},
        ]
    )
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def grow(x: int) -> int:
        """Grow x."""

    assert grow(1) == 2
    grow.feedback(0.1, "should double, not increment")

    signals = grow.signals()
    assert len(signals) == 1
    assert signals[0].kind == "feedback"
    assert signals[0].weight == pytest.approx(-0.8)

    descendant = grow.backward()  # mutate under the soft signal, shadow-validated
    assert descendant.operator == "mutate"
    assert grow(3) == 6


def test_backward_without_signals_is_an_error(stub_model) -> None:
    spawn = "async def solve(x: int) -> int:\n    return x\n"
    stub_model.push({"reasoning": "r", "helpers": [], "imports": [], "body": spawn})
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def ident(x: int) -> int:
        """."""

    ident(1)
    with pytest.raises(FrameworkError):
        ident.backward()


def test_feedback_score_bounds() -> None:
    from synecdoche import Signal

    with pytest.raises(ValueError):
        Signal.from_feedback(1.5)
    assert Signal.from_feedback(0.5).weight == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Offline evolution
# ---------------------------------------------------------------------------


def test_evolve_selects_the_fittest_variant(stub_model) -> None:
    stub_model.extend(
        [
            # genesis spawn: off by one
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
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def double(x: int) -> int:
        """Double x."""

    def score(inputs: dict, output: int) -> float:
        return 1.0 if output == inputs["x"] * 2 else 0.0

    report = double.evolve([{"x": 1}, {"x": 3}], generations=1, population=2, score=score)
    assert report.champion.fitness == pytest.approx(1.0)
    assert report.generations == 1
    assert len(report.evaluated) == 3
    # The champion now serves calls with no further compilation.
    assert double(7) == 14
    assert double.champion.version == report.champion.version


# ---------------------------------------------------------------------------
# Solidify: generated code crosses back into deterministic source
# ---------------------------------------------------------------------------


def test_solidify_renders_committable_source(stub_model, tmp_path) -> None:
    spawn = "async def solve(x: int) -> int:\n    return x * 2\n"
    stub_model.push({"reasoning": "double it", "helpers": [], "imports": [], "body": spawn})
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def double(x: int) -> int:
        """Double x."""

    double(2)
    out = tmp_path / "double.py"
    src = double.solidify(out)
    assert "async def double(x: int)" in src  # renamed to the contract's name
    assert "# Solidified by synecdoche." in src
    assert "double it" in src  # provenance: the variant's reasoning
    assert out.read_text() == src
    compile(src, str(out), "exec")  # it is valid Python


def test_solidify_of_a_seed_returns_the_handwritten_source(stub_model) -> None:
    rt = Runtime(model=stub_model.as_model())

    @rt.fn
    def triple(x: int) -> int:
        return x * 3

    triple(1)
    assert "return x * 3" in triple.solidify()
