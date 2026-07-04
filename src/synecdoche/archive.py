"""The archive: a population of variants under selection, keyed by signature.

Every body the compiler has ever produced for a signature is a **variant** —
a member of that signature's population, carrying its lineage (which parents
it was varied from, by which operator), its live metrics, and the signals
recorded against it. Exactly one variant per ``(signature, surface)`` key is
*promoted*: the champion that runs on the next call.

The archive is therefore three things at once:

- a **cache** (champion lookup skips compilation),
- a **fossil record** (every variant is kept; lineage is never rewritten),
- a **gene pool** (evolution draws parents and cross-signature neighbors
  from it).

Two backends ship: ``MemoryArchive`` (dict-backed, for tests and ephemeral
runs) and ``SqliteArchive`` (stdlib sqlite3, the default for local work).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .compiler import GeneratedBody
from .evolution import Operator, Signal, default_fitness, now_utc


@dataclass
class Metrics:
    invocations: int = 0
    successes: int = 0
    exceptions: int = 0
    validation_failures: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class Variant:
    """One member of a signature's population."""

    signature_hash: str
    surface_hash: str
    version: int
    operator: Operator
    parents: tuple[int, ...]
    body: GeneratedBody
    created_at: datetime
    promoted: bool = False
    fitness: float | None = None  # measured (offline evolution); None = infer from metrics
    metrics: Metrics = field(default_factory=Metrics)

    def score(self, signals: list[Signal] | tuple[Signal, ...] = ()) -> float:
        """Measured fitness if present, else inferred from live evidence."""
        if self.fitness is not None:
            return self.fitness
        return default_fitness(
            invocations=self.metrics.invocations,
            successes=self.metrics.successes,
            signals=signals,
        )


class Archive(Protocol):
    def champion(self, signature_hash: str, surface_hash: str) -> Variant | None: ...
    def get(self, signature_hash: str, surface_hash: str, version: int) -> Variant | None: ...
    def insert(self, variant: Variant) -> None: ...
    def promote(self, signature_hash: str, surface_hash: str, version: int) -> None: ...
    def rollback(self, signature_hash: str, surface_hash: str) -> Variant | None: ...
    def population(self, signature_hash: str, surface_hash: str) -> list[Variant]: ...
    def neighbors(self, signature_hash: str, k: int = 3) -> list[Variant]: ...
    def next_version(self, signature_hash: str, surface_hash: str) -> int: ...
    def record_metrics(
        self,
        signature_hash: str,
        surface_hash: str,
        version: int,
        *,
        success: bool,
        latency_ms: float,
        validation_failure: bool = False,
    ) -> None: ...
    def record_signal(
        self, signature_hash: str, surface_hash: str, version: int, signal: Signal
    ) -> None: ...
    def signals_for(self, signature_hash: str, surface_hash: str, version: int) -> list[Signal]: ...
    def set_fitness(
        self, signature_hash: str, surface_hash: str, version: int, fitness: float
    ) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class MemoryArchive:
    """Dict-backed Archive, suitable for tests and ephemeral runs."""

    def __init__(self) -> None:
        self._variants: dict[tuple[str, str, int], Variant] = {}
        self._signals: dict[tuple[str, str, int], list[Signal]] = {}
        self._lock = threading.RLock()

    def _key_variants(self, signature_hash: str, surface_hash: str) -> list[Variant]:
        return sorted(
            [
                v
                for (sh, tsh, _), v in self._variants.items()
                if sh == signature_hash and tsh == surface_hash
            ],
            key=lambda v: v.version,
        )

    def champion(self, signature_hash: str, surface_hash: str) -> Variant | None:
        with self._lock:
            promoted = [v for v in self._key_variants(signature_hash, surface_hash) if v.promoted]
            return promoted[-1] if promoted else None

    def get(self, signature_hash: str, surface_hash: str, version: int) -> Variant | None:
        with self._lock:
            return self._variants.get((signature_hash, surface_hash, version))

    def insert(self, variant: Variant) -> None:
        with self._lock:
            key = (variant.signature_hash, variant.surface_hash, variant.version)
            self._variants[key] = variant

    def promote(self, signature_hash: str, surface_hash: str, version: int) -> None:
        with self._lock:
            for v in self._key_variants(signature_hash, surface_hash):
                v.promoted = v.version == version

    def rollback(self, signature_hash: str, surface_hash: str) -> Variant | None:
        with self._lock:
            current = self.champion(signature_hash, surface_hash)
            if current is None or not current.parents:
                return None
            parent = self._variants.get((signature_hash, surface_hash, current.parents[0]))
            if parent is None:
                return None
            self.promote(signature_hash, surface_hash, parent.version)
            return parent

    def population(self, signature_hash: str, surface_hash: str) -> list[Variant]:
        with self._lock:
            return list(self._key_variants(signature_hash, surface_hash))

    def neighbors(self, signature_hash: str, k: int = 3) -> list[Variant]:
        # POC: no semantic similarity yet — the k most recent champions of
        # other signatures, as style references for the compiler.
        with self._lock:
            by_sig: dict[str, list[Variant]] = {}
            for v in self._variants.values():
                if v.signature_hash == signature_hash or not v.promoted:
                    continue
                by_sig.setdefault(v.signature_hash, []).append(v)
            candidates = [max(vs, key=lambda v: v.created_at) for vs in by_sig.values()]
            candidates.sort(key=lambda v: v.created_at, reverse=True)
            return candidates[:k]

    def next_version(self, signature_hash: str, surface_hash: str) -> int:
        with self._lock:
            existing = self._key_variants(signature_hash, surface_hash)
            return (existing[-1].version + 1) if existing else 1

    def record_metrics(
        self,
        signature_hash: str,
        surface_hash: str,
        version: int,
        *,
        success: bool,
        latency_ms: float,
        validation_failure: bool = False,
    ) -> None:
        with self._lock:
            variant = self._variants.get((signature_hash, surface_hash, version))
            if variant is None:
                return
            m = variant.metrics
            m.invocations += 1
            if success:
                m.successes += 1
            else:
                m.exceptions += 1
            if validation_failure:
                m.validation_failures += 1
            m.avg_latency_ms = (m.avg_latency_ms * (m.invocations - 1) + latency_ms) / m.invocations

    def record_signal(
        self, signature_hash: str, surface_hash: str, version: int, signal: Signal
    ) -> None:
        with self._lock:
            self._signals.setdefault((signature_hash, surface_hash, version), []).append(signal)

    def signals_for(self, signature_hash: str, surface_hash: str, version: int) -> list[Signal]:
        with self._lock:
            return list(self._signals.get((signature_hash, surface_hash, version), []))

    def set_fitness(
        self, signature_hash: str, surface_hash: str, version: int, fitness: float
    ) -> None:
        with self._lock:
            variant = self._variants.get((signature_hash, surface_hash, version))
            if variant is not None:
                variant.fitness = fitness


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS variants (
    signature_hash      TEXT NOT NULL,
    surface_hash        TEXT NOT NULL,
    version             INTEGER NOT NULL,
    operator            TEXT NOT NULL,
    parents_json        TEXT NOT NULL DEFAULT '[]',
    body_json           TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    promoted            INTEGER NOT NULL DEFAULT 0,
    fitness             REAL,
    invocations         INTEGER NOT NULL DEFAULT 0,
    successes           INTEGER NOT NULL DEFAULT 0,
    exceptions          INTEGER NOT NULL DEFAULT 0,
    validation_failures INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms      REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (signature_hash, surface_hash, version)
);
CREATE INDEX IF NOT EXISTS idx_variants_key
    ON variants(signature_hash, surface_hash, promoted);
CREATE INDEX IF NOT EXISTS idx_variants_sig
    ON variants(signature_hash, promoted);

CREATE TABLE IF NOT EXISTS signals (
    signature_hash      TEXT NOT NULL,
    surface_hash        TEXT NOT NULL,
    version             INTEGER NOT NULL,
    kind                TEXT NOT NULL,
    content             TEXT NOT NULL,
    weight              REAL NOT NULL,
    data_json           TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_key
    ON signals(signature_hash, surface_hash, version);
"""

_VARIANT_COLS = (
    "signature_hash, surface_hash, version, operator, parents_json, body_json, "
    "created_at, promoted, fitness, invocations, successes, exceptions, "
    "validation_failures, avg_latency_ms"
)


class SqliteArchive:
    """SQLite-backed Archive. Thread-safe via a single connection + lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _row_to_variant(self, row: tuple) -> Variant:
        (
            sh,
            tsh,
            version,
            operator,
            parents_json,
            body_json,
            created_at,
            promoted,
            fitness,
            inv,
            suc,
            exc,
            vf,
            lat,
        ) = row
        return Variant(
            signature_hash=sh,
            surface_hash=tsh,
            version=version,
            operator=operator,
            parents=tuple(json.loads(parents_json)),
            body=GeneratedBody.model_validate_json(body_json),
            created_at=datetime.fromisoformat(created_at),
            promoted=bool(promoted),
            fitness=fitness,
            metrics=Metrics(
                invocations=inv,
                successes=suc,
                exceptions=exc,
                validation_failures=vf,
                avg_latency_ms=lat,
            ),
        )

    def _select_where(self, where: str, params: tuple) -> list[Variant]:
        cur = self._conn.execute(
            f"SELECT {_VARIANT_COLS} FROM variants WHERE {where} ORDER BY version ASC",
            params,
        )
        return [self._row_to_variant(r) for r in cur.fetchall()]

    def champion(self, signature_hash: str, surface_hash: str) -> Variant | None:
        with self._lock:
            variants = self._select_where(
                "signature_hash = ? AND surface_hash = ? AND promoted = 1",
                (signature_hash, surface_hash),
            )
            return variants[-1] if variants else None

    def get(self, signature_hash: str, surface_hash: str, version: int) -> Variant | None:
        with self._lock:
            variants = self._select_where(
                "signature_hash = ? AND surface_hash = ? AND version = ?",
                (signature_hash, surface_hash, version),
            )
            return variants[0] if variants else None

    def insert(self, variant: Variant) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO variants ({_VARIANT_COLS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    variant.signature_hash,
                    variant.surface_hash,
                    variant.version,
                    variant.operator,
                    json.dumps(list(variant.parents)),
                    variant.body.model_dump_json(),
                    variant.created_at.isoformat(),
                    int(variant.promoted),
                    variant.fitness,
                    variant.metrics.invocations,
                    variant.metrics.successes,
                    variant.metrics.exceptions,
                    variant.metrics.validation_failures,
                    variant.metrics.avg_latency_ms,
                ),
            )
            self._conn.commit()

    def promote(self, signature_hash: str, surface_hash: str, version: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE variants SET promoted = 0 WHERE signature_hash = ? AND surface_hash = ?",
                (signature_hash, surface_hash),
            )
            self._conn.execute(
                "UPDATE variants SET promoted = 1 "
                "WHERE signature_hash = ? AND surface_hash = ? AND version = ?",
                (signature_hash, surface_hash, version),
            )
            self._conn.commit()

    def rollback(self, signature_hash: str, surface_hash: str) -> Variant | None:
        current = self.champion(signature_hash, surface_hash)
        if current is None or not current.parents:
            return None
        parent = self.get(signature_hash, surface_hash, current.parents[0])
        if parent is None:
            return None
        self.promote(signature_hash, surface_hash, parent.version)
        return parent

    def population(self, signature_hash: str, surface_hash: str) -> list[Variant]:
        with self._lock:
            return self._select_where(
                "signature_hash = ? AND surface_hash = ?",
                (signature_hash, surface_hash),
            )

    def neighbors(self, signature_hash: str, k: int = 3) -> list[Variant]:
        with self._lock:
            variants = self._select_where("signature_hash != ? AND promoted = 1", (signature_hash,))
            variants.sort(key=lambda v: v.created_at, reverse=True)
            return variants[:k]

    def next_version(self, signature_hash: str, surface_hash: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM variants "
                "WHERE signature_hash = ? AND surface_hash = ?",
                (signature_hash, surface_hash),
            )
            return cur.fetchone()[0] + 1

    def record_metrics(
        self,
        signature_hash: str,
        surface_hash: str,
        version: int,
        *,
        success: bool,
        latency_ms: float,
        validation_failure: bool = False,
    ) -> None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT invocations, avg_latency_ms FROM variants "
                "WHERE signature_hash = ? AND surface_hash = ? AND version = ?",
                (signature_hash, surface_hash, version),
            )
            row = cur.fetchone()
            if row is None:
                return
            inv, avg = row
            new_inv = inv + 1
            new_avg = (avg * inv + latency_ms) / new_inv
            self._conn.execute(
                """
                UPDATE variants
                SET invocations = ?, avg_latency_ms = ?,
                    successes = successes + ?, exceptions = exceptions + ?,
                    validation_failures = validation_failures + ?
                WHERE signature_hash = ? AND surface_hash = ? AND version = ?
                """,
                (
                    new_inv,
                    new_avg,
                    1 if success else 0,
                    0 if success else 1,
                    1 if validation_failure else 0,
                    signature_hash,
                    surface_hash,
                    version,
                ),
            )
            self._conn.commit()

    def record_signal(
        self, signature_hash: str, surface_hash: str, version: int, signal: Signal
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO signals (signature_hash, surface_hash, version, kind, "
                "content, weight, data_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signature_hash,
                    surface_hash,
                    version,
                    signal.kind,
                    signal.content,
                    signal.weight,
                    json.dumps(signal.data, default=str),
                    signal.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def signals_for(self, signature_hash: str, surface_hash: str, version: int) -> list[Signal]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT kind, content, weight, data_json, created_at FROM signals "
                "WHERE signature_hash = ? AND surface_hash = ? AND version = ? "
                "ORDER BY created_at ASC",
                (signature_hash, surface_hash, version),
            )
            return [
                Signal(
                    kind=kind,
                    content=content,
                    weight=weight,
                    data=json.loads(data_json),
                    created_at=datetime.fromisoformat(created_at),
                )
                for kind, content, weight, data_json, created_at in cur.fetchall()
            ]

    def set_fitness(
        self, signature_hash: str, surface_hash: str, version: int, fitness: float
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE variants SET fitness = ? "
                "WHERE signature_hash = ? AND surface_hash = ? AND version = ?",
                (fitness, signature_hash, surface_hash, version),
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


__all__ = [
    "Archive",
    "MemoryArchive",
    "Metrics",
    "SqliteArchive",
    "Variant",
    "now_utc",
    "open_archive",
]
