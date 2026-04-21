from __future__ import annotations

from pathlib import Path

import pytest

from synecdoche.archive import (
    ArchiveEntry,
    MemoryArchive,
    SqliteArchive,
    now_utc,
)
from synecdoche.compiler import GeneratedBody


def _mk_entry(version: int = 1, parent: int | None = None, promoted: bool = True) -> ArchiveEntry:
    body = GeneratedBody(reasoning="r", helpers=[], imports=[], body="async def solve(): return 1")
    return ArchiveEntry(
        signature_hash="sig1",
        tool_surface_hash="surf1",
        version=version,
        parent_version=parent,
        body=body,
        created_at=now_utc(),
        trigger="initial" if parent is None else "repair",
        promoted=promoted,
    )


@pytest.fixture(params=["memory", "sqlite"])
def archive(request, tmp_path: Path):
    if request.param == "memory":
        return MemoryArchive()
    return SqliteArchive(tmp_path / "arch.sqlite")


def test_insert_then_get_current(archive) -> None:
    e = _mk_entry(version=1)
    archive.insert(e)
    archive.promote("sig1", "surf1", 1)
    got = archive.get_current("sig1", "surf1")
    assert got is not None
    assert got.version == 1
    assert got.body.reasoning == "r"


def test_next_version_increments(archive) -> None:
    assert archive.next_version("sig1", "surf1") == 1
    archive.insert(_mk_entry(version=1))
    assert archive.next_version("sig1", "surf1") == 2


def test_promote_then_rollback(archive) -> None:
    archive.insert(_mk_entry(version=1, promoted=True))
    archive.insert(_mk_entry(version=2, parent=1, promoted=False))
    archive.promote("sig1", "surf1", 2)

    current = archive.get_current("sig1", "surf1")
    assert current is not None
    assert current.version == 2

    rolled = archive.rollback("sig1", "surf1")
    assert rolled is not None
    assert rolled.version == 1
    assert archive.get_current("sig1", "surf1").version == 1


def test_record_metrics_updates_averages(archive) -> None:
    archive.insert(_mk_entry(version=1, promoted=True))
    archive.record_metrics("sig1", "surf1", 1, success=True, latency_ms=100.0)
    archive.record_metrics("sig1", "surf1", 1, success=False, latency_ms=300.0)
    got = archive.get_current("sig1", "surf1")
    assert got is not None
    assert got.metrics.invocations == 2
    assert got.metrics.successes == 1
    assert got.metrics.exceptions == 1
    assert abs(got.metrics.avg_latency_ms - 200.0) < 1e-6


def test_history_returns_all_versions(archive) -> None:
    archive.insert(_mk_entry(version=1))
    archive.insert(_mk_entry(version=2, parent=1))
    archive.insert(_mk_entry(version=3, parent=2))
    history = archive.history("sig1", "surf1")
    assert [e.version for e in history] == [1, 2, 3]
