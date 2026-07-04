from __future__ import annotations

from pathlib import Path

import pytest

from synecdoche.archive import MemoryArchive, SqliteArchive, Variant, now_utc
from synecdoche.compiler import GeneratedBody
from synecdoche.evolution import Signal


def _mk_variant(version: int = 1, parents: tuple[int, ...] = (), promoted: bool = True) -> Variant:
    body = GeneratedBody(reasoning="r", helpers=[], imports=[], body="async def solve(): return 1")
    return Variant(
        signature_hash="sig1",
        surface_hash="surf1",
        version=version,
        operator="spawn" if not parents else "mutate",
        parents=parents,
        body=body,
        created_at=now_utc(),
        promoted=promoted,
    )


@pytest.fixture(params=["memory", "sqlite"])
def archive(request, tmp_path: Path):
    if request.param == "memory":
        return MemoryArchive()
    return SqliteArchive(tmp_path / "arch.sqlite")


def test_insert_then_champion(archive) -> None:
    v = _mk_variant(version=1)
    archive.insert(v)
    archive.promote("sig1", "surf1", 1)
    got = archive.champion("sig1", "surf1")
    assert got is not None
    assert got.version == 1
    assert got.body.reasoning == "r"
    assert got.operator == "spawn"


def test_next_version_increments(archive) -> None:
    assert archive.next_version("sig1", "surf1") == 1
    archive.insert(_mk_variant(version=1))
    assert archive.next_version("sig1", "surf1") == 2


def test_promote_then_rollback(archive) -> None:
    archive.insert(_mk_variant(version=1, promoted=True))
    archive.insert(_mk_variant(version=2, parents=(1,), promoted=False))
    archive.promote("sig1", "surf1", 2)

    current = archive.champion("sig1", "surf1")
    assert current is not None
    assert current.version == 2
    assert current.parents == (1,)

    rolled = archive.rollback("sig1", "surf1")
    assert rolled is not None
    assert rolled.version == 1
    assert archive.champion("sig1", "surf1").version == 1


def test_record_metrics_updates_averages(archive) -> None:
    archive.insert(_mk_variant(version=1, promoted=True))
    archive.record_metrics("sig1", "surf1", 1, success=True, latency_ms=100.0)
    archive.record_metrics("sig1", "surf1", 1, success=False, latency_ms=300.0)
    got = archive.champion("sig1", "surf1")
    assert got is not None
    assert got.metrics.invocations == 2
    assert got.metrics.successes == 1
    assert got.metrics.exceptions == 1
    assert abs(got.metrics.avg_latency_ms - 200.0) < 1e-6


def test_population_returns_all_versions(archive) -> None:
    archive.insert(_mk_variant(version=1))
    archive.insert(_mk_variant(version=2, parents=(1,)))
    archive.insert(_mk_variant(version=3, parents=(2,)))
    population = archive.population("sig1", "surf1")
    assert [v.version for v in population] == [1, 2, 3]
    assert population[2].parents == (2,)


def test_signals_round_trip(archive) -> None:
    archive.insert(_mk_variant(version=1))
    archive.record_signal("sig1", "surf1", 1, Signal.from_feedback(0.25, "too slow"))
    archive.record_signal("sig1", "surf1", 1, Signal.from_exception(ValueError("boom")))
    signals = archive.signals_for("sig1", "surf1", 1)
    assert len(signals) == 2
    kinds = {s.kind for s in signals}
    assert kinds == {"feedback", "exception"}
    feedback = next(s for s in signals if s.kind == "feedback")
    assert feedback.content == "too slow"
    assert abs(feedback.weight - (-0.5)) < 1e-9
    hard = next(s for s in signals if s.kind == "exception")
    assert hard.weight == -1.0
    assert "boom" in hard.content


def test_set_fitness_persists(archive) -> None:
    archive.insert(_mk_variant(version=1))
    archive.set_fitness("sig1", "surf1", 1, 0.875)
    got = archive.get("sig1", "surf1", 1)
    assert got is not None
    assert got.fitness == pytest.approx(0.875)
    assert got.score() == pytest.approx(0.875)


def test_score_infers_from_metrics_when_no_measured_fitness(archive) -> None:
    archive.insert(_mk_variant(version=1))
    got = archive.get("sig1", "surf1", 1)
    assert got.score() == pytest.approx(0.5)  # no evidence yet
    archive.record_metrics("sig1", "surf1", 1, success=True, latency_ms=1.0)
    archive.record_metrics("sig1", "surf1", 1, success=True, latency_ms=1.0)
    got = archive.get("sig1", "surf1", 1)
    assert got.score() == pytest.approx(1.0)
