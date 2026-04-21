from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from synecdoche.archive import ArchiveEntry, SqliteArchive
from synecdoche.archive_io import entries_from_json, entries_to_json
from synecdoche.cli import main
from synecdoche.compiler import GeneratedBody


def _mk_entry(version: int = 1, parent: int | None = None, promoted: bool = True) -> ArchiveEntry:
    body = GeneratedBody(
        reasoning="r",
        helpers=[],
        imports=[],
        body="async def solve(): return 1",
    )
    return ArchiveEntry(
        signature_hash="abc123sig",
        tool_surface_hash="surf1",
        version=version,
        parent_version=parent,
        body=body,
        created_at=datetime.now(UTC),
        trigger="initial" if parent is None else "repair",
        promoted=promoted,
    )


@pytest.fixture
def archive_path(tmp_path: Path) -> Path:
    p = tmp_path / "arch.sqlite"
    archive = SqliteArchive(p)
    archive.insert(_mk_entry(version=1, promoted=False))
    archive.insert(_mk_entry(version=2, parent=1, promoted=True))
    return p


def test_cli_ls_lists_entries(archive_path: Path, capsys) -> None:
    rc = main(["archive", "ls", "--archive", str(archive_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "abc123sig" in out
    assert "surf1" in out or "surf" in out


def test_cli_show_prints_body(archive_path: Path, capsys) -> None:
    rc = main(["archive", "show", "abc123sig", "--archive", str(archive_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "async def solve()" in out
    assert "reasoning" in out


def test_cli_diff_between_versions(archive_path: Path, capsys) -> None:
    # Both versions have identical body — diff should be empty but succeed.
    rc = main(["archive", "diff", "abc123sig", "1", "2", "--archive", str(archive_path)])
    assert rc == 0


def test_cli_rollback(archive_path: Path, capsys) -> None:
    rc = main(["archive", "rollback", "abc123sig", "--archive", str(archive_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rolled back" in out


def test_cli_export_import_round_trip(archive_path: Path, tmp_path: Path, capsys) -> None:
    out_path = tmp_path / "dump.json"
    rc = main(["archive", "export", str(out_path), "--archive", str(archive_path)])
    assert rc == 0
    assert out_path.exists()

    # Import into a fresh archive and verify entries come back.
    new_path = tmp_path / "fresh.sqlite"
    rc = main(["archive", "import", str(out_path), "--archive", str(new_path)])
    assert rc == 0

    imported = SqliteArchive(new_path)
    history = imported.history("abc123sig", "surf1")
    assert len(history) == 2
    assert history[0].body.reasoning == "r"


def test_entries_json_round_trip() -> None:
    entries = [_mk_entry(version=1), _mk_entry(version=2, parent=1)]
    data = entries_to_json(entries)
    restored = entries_from_json(data)
    assert len(restored) == 2
    assert restored[0].signature_hash == "abc123sig"
    assert restored[1].parent_version == 1
