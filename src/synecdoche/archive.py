"""Archive: persistent record of compiled bodies, keyed by signature + tool surface.

The archive is how calls after the first one skip compilation: lookup by
`(signature_hash, tool_surface_hash)` gets you the current promoted body.
Versions let repair insert a revised body as a successor while preserving
the original for rollback.

Two backends ship:
- `MemoryArchive` — dict-backed, used in tests.
- `SqliteArchive` — stdlib sqlite3, one file per project, suitable as the
  default for local development.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from .compiler import GeneratedBody

Trigger = Literal["initial", "repair", "manual"]


@dataclass
class ArchiveMetrics:
    invocations: int = 0
    successes: int = 0
    exceptions: int = 0
    validation_failures: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class ArchiveEntry:
    signature_hash: str
    tool_surface_hash: str
    version: int
    parent_version: int | None
    body: GeneratedBody
    created_at: datetime
    trigger: Trigger
    promoted: bool = False
    metrics: ArchiveMetrics = field(default_factory=ArchiveMetrics)


class Archive(Protocol):
    def get_current(self, signature_hash: str, tool_surface_hash: str) -> ArchiveEntry | None: ...
    def get_version(
        self, signature_hash: str, tool_surface_hash: str, version: int
    ) -> ArchiveEntry | None: ...
    def insert(self, entry: ArchiveEntry) -> None: ...
    def promote(self, signature_hash: str, tool_surface_hash: str, version: int) -> None: ...
    def rollback(self, signature_hash: str, tool_surface_hash: str) -> ArchiveEntry | None: ...
    def history(self, signature_hash: str, tool_surface_hash: str) -> list[ArchiveEntry]: ...
    def neighbors(self, signature_hash: str, k: int = 3) -> list[ArchiveEntry]: ...
    def next_version(self, signature_hash: str, tool_surface_hash: str) -> int: ...
    def record_metrics(
        self,
        signature_hash: str,
        tool_surface_hash: str,
        version: int,
        *,
        success: bool,
        latency_ms: float,
        validation_failure: bool = False,
    ) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class MemoryArchive:
    """Dict-backed Archive, suitable for tests and ephemeral runs."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, int], ArchiveEntry] = {}
        self._lock = threading.RLock()

    def _key_entries(self, signature_hash: str, tool_surface_hash: str) -> list[ArchiveEntry]:
        return sorted(
            [
                e
                for (sh, tsh, _), e in self._entries.items()
                if sh == signature_hash and tsh == tool_surface_hash
            ],
            key=lambda e: e.version,
        )

    def get_current(self, signature_hash: str, tool_surface_hash: str) -> ArchiveEntry | None:
        with self._lock:
            promoted = [
                e for e in self._key_entries(signature_hash, tool_surface_hash) if e.promoted
            ]
            return promoted[-1] if promoted else None

    def get_version(
        self, signature_hash: str, tool_surface_hash: str, version: int
    ) -> ArchiveEntry | None:
        with self._lock:
            return self._entries.get((signature_hash, tool_surface_hash, version))

    def insert(self, entry: ArchiveEntry) -> None:
        with self._lock:
            self._entries[(entry.signature_hash, entry.tool_surface_hash, entry.version)] = entry

    def promote(self, signature_hash: str, tool_surface_hash: str, version: int) -> None:
        with self._lock:
            for e in self._key_entries(signature_hash, tool_surface_hash):
                e.promoted = e.version == version

    def rollback(self, signature_hash: str, tool_surface_hash: str) -> ArchiveEntry | None:
        with self._lock:
            current = self.get_current(signature_hash, tool_surface_hash)
            if current is None or current.parent_version is None:
                return None
            parent = self._entries.get((signature_hash, tool_surface_hash, current.parent_version))
            if parent is None:
                return None
            for e in self._key_entries(signature_hash, tool_surface_hash):
                e.promoted = e.version == parent.version
            return parent

    def history(self, signature_hash: str, tool_surface_hash: str) -> list[ArchiveEntry]:
        with self._lock:
            return list(self._key_entries(signature_hash, tool_surface_hash))

    def neighbors(self, signature_hash: str, k: int = 3) -> list[ArchiveEntry]:
        # POC: no semantic similarity, just return k most-recent promoted entries
        # across other signatures.
        with self._lock:
            by_sig: dict[str, list[ArchiveEntry]] = {}
            for e in self._entries.values():
                if e.signature_hash == signature_hash or not e.promoted:
                    continue
                by_sig.setdefault(e.signature_hash, []).append(e)
            candidates = [max(entries, key=lambda e: e.created_at) for entries in by_sig.values()]
            candidates.sort(key=lambda e: e.created_at, reverse=True)
            return candidates[:k]

    def next_version(self, signature_hash: str, tool_surface_hash: str) -> int:
        with self._lock:
            existing = self._key_entries(signature_hash, tool_surface_hash)
            return (existing[-1].version + 1) if existing else 1

    def record_metrics(
        self,
        signature_hash: str,
        tool_surface_hash: str,
        version: int,
        *,
        success: bool,
        latency_ms: float,
        validation_failure: bool = False,
    ) -> None:
        with self._lock:
            entry = self._entries.get((signature_hash, tool_surface_hash, version))
            if entry is None:
                return
            m = entry.metrics
            m.invocations += 1
            if success:
                m.successes += 1
            else:
                m.exceptions += 1
            if validation_failure:
                m.validation_failures += 1
            m.avg_latency_ms = (m.avg_latency_ms * (m.invocations - 1) + latency_ms) / m.invocations


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    signature_hash      TEXT NOT NULL,
    tool_surface_hash   TEXT NOT NULL,
    version             INTEGER NOT NULL,
    parent_version      INTEGER,
    body_json           TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    trigger             TEXT NOT NULL,
    promoted            INTEGER NOT NULL DEFAULT 0,
    invocations         INTEGER NOT NULL DEFAULT 0,
    successes           INTEGER NOT NULL DEFAULT 0,
    exceptions          INTEGER NOT NULL DEFAULT 0,
    validation_failures INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms      REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (signature_hash, tool_surface_hash, version)
);
CREATE INDEX IF NOT EXISTS idx_entries_key
    ON entries(signature_hash, tool_surface_hash, promoted);
CREATE INDEX IF NOT EXISTS idx_entries_sig
    ON entries(signature_hash, promoted);
"""


class SqliteArchive:
    """SQLite-backed Archive. Thread-safe via a single connection + lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _row_to_entry(self, row: sqlite3.Row | tuple) -> ArchiveEntry:
        (
            sh,
            tsh,
            version,
            parent,
            body_json,
            created_at,
            trigger,
            promoted,
            inv,
            suc,
            exc,
            vf,
            lat,
        ) = row
        return ArchiveEntry(
            signature_hash=sh,
            tool_surface_hash=tsh,
            version=version,
            parent_version=parent,
            body=GeneratedBody.model_validate_json(body_json),
            created_at=datetime.fromisoformat(created_at),
            trigger=trigger,
            promoted=bool(promoted),
            metrics=ArchiveMetrics(
                invocations=inv,
                successes=suc,
                exceptions=exc,
                validation_failures=vf,
                avg_latency_ms=lat,
            ),
        )

    def _select_where(self, where: str, params: Iterable) -> list[ArchiveEntry]:
        cur = self._conn.execute(
            f"""
            SELECT signature_hash, tool_surface_hash, version, parent_version,
                   body_json, created_at, trigger, promoted,
                   invocations, successes, exceptions, validation_failures,
                   avg_latency_ms
            FROM entries WHERE {where}
            ORDER BY version ASC
            """,
            tuple(params),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def get_current(self, signature_hash: str, tool_surface_hash: str) -> ArchiveEntry | None:
        with self._lock:
            entries = self._select_where(
                "signature_hash = ? AND tool_surface_hash = ? AND promoted = 1",
                (signature_hash, tool_surface_hash),
            )
            return entries[-1] if entries else None

    def get_version(
        self, signature_hash: str, tool_surface_hash: str, version: int
    ) -> ArchiveEntry | None:
        with self._lock:
            entries = self._select_where(
                "signature_hash = ? AND tool_surface_hash = ? AND version = ?",
                (signature_hash, tool_surface_hash, version),
            )
            return entries[0] if entries else None

    def insert(self, entry: ArchiveEntry) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entries (
                    signature_hash, tool_surface_hash, version, parent_version,
                    body_json, created_at, trigger, promoted,
                    invocations, successes, exceptions, validation_failures,
                    avg_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.signature_hash,
                    entry.tool_surface_hash,
                    entry.version,
                    entry.parent_version,
                    entry.body.model_dump_json(),
                    entry.created_at.isoformat(),
                    entry.trigger,
                    int(entry.promoted),
                    entry.metrics.invocations,
                    entry.metrics.successes,
                    entry.metrics.exceptions,
                    entry.metrics.validation_failures,
                    entry.metrics.avg_latency_ms,
                ),
            )
            self._conn.commit()

    def promote(self, signature_hash: str, tool_surface_hash: str, version: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE entries SET promoted = 0 WHERE signature_hash = ? AND tool_surface_hash = ?",
                (signature_hash, tool_surface_hash),
            )
            self._conn.execute(
                "UPDATE entries SET promoted = 1 WHERE signature_hash = ? AND tool_surface_hash = ? AND version = ?",
                (signature_hash, tool_surface_hash, version),
            )
            self._conn.commit()

    def rollback(self, signature_hash: str, tool_surface_hash: str) -> ArchiveEntry | None:
        current = self.get_current(signature_hash, tool_surface_hash)
        if current is None or current.parent_version is None:
            return None
        parent = self.get_version(signature_hash, tool_surface_hash, current.parent_version)
        if parent is None:
            return None
        self.promote(signature_hash, tool_surface_hash, parent.version)
        return parent

    def history(self, signature_hash: str, tool_surface_hash: str) -> list[ArchiveEntry]:
        with self._lock:
            return self._select_where(
                "signature_hash = ? AND tool_surface_hash = ?",
                (signature_hash, tool_surface_hash),
            )

    def neighbors(self, signature_hash: str, k: int = 3) -> list[ArchiveEntry]:
        with self._lock:
            entries = self._select_where("signature_hash != ? AND promoted = 1", (signature_hash,))
            entries.sort(key=lambda e: e.created_at, reverse=True)
            return entries[:k]

    def next_version(self, signature_hash: str, tool_surface_hash: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM entries WHERE signature_hash = ? AND tool_surface_hash = ?",
                (signature_hash, tool_surface_hash),
            )
            return cur.fetchone()[0] + 1

    def record_metrics(
        self,
        signature_hash: str,
        tool_surface_hash: str,
        version: int,
        *,
        success: bool,
        latency_ms: float,
        validation_failure: bool = False,
    ) -> None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT invocations, avg_latency_ms FROM entries "
                "WHERE signature_hash = ? AND tool_surface_hash = ? AND version = ?",
                (signature_hash, tool_surface_hash, version),
            )
            row = cur.fetchone()
            if row is None:
                return
            inv, avg = row
            new_inv = inv + 1
            new_avg = (avg * inv + latency_ms) / new_inv
            self._conn.execute(
                """
                UPDATE entries
                SET invocations = ?, avg_latency_ms = ?,
                    successes = successes + ?, exceptions = exceptions + ?,
                    validation_failures = validation_failures + ?
                WHERE signature_hash = ? AND tool_surface_hash = ? AND version = ?
                """,
                (
                    new_inv,
                    new_avg,
                    1 if success else 0,
                    0 if success else 1,
                    1 if validation_failure else 0,
                    signature_hash,
                    tool_surface_hash,
                    version,
                ),
            )
            self._conn.commit()


def open_archive(spec: str | Path | Archive | None) -> Archive:
    """Convenience constructor: path → SqliteArchive, None → MemoryArchive."""
    if spec is None:
        return MemoryArchive()
    if isinstance(spec, (str, Path)):
        p = Path(spec)
        if p.suffix == "":
            p = p / "synecdoche.sqlite"
        return SqliteArchive(p)
    return spec  # already an Archive


def now_utc() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "Archive",
    "ArchiveEntry",
    "ArchiveMetrics",
    "MemoryArchive",
    "SqliteArchive",
    "now_utc",
    "open_archive",
]
