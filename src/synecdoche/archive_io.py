"""Portable JSON serialization for ArchiveEntry lists.

This is the format used by `synecdoche archive export/import` and the
in-process helpers `Archive.export_entries()` / `Archive.import_entries()`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .archive import ArchiveEntry, ArchiveMetrics
from .compiler import GeneratedBody


def _entry_to_dict(entry: ArchiveEntry) -> dict[str, Any]:
    return {
        "signature_hash": entry.signature_hash,
        "tool_surface_hash": entry.tool_surface_hash,
        "version": entry.version,
        "parent_version": entry.parent_version,
        "body": entry.body.model_dump(),
        "created_at": entry.created_at.isoformat(),
        "trigger": entry.trigger,
        "promoted": entry.promoted,
        "metrics": {
            "invocations": entry.metrics.invocations,
            "successes": entry.metrics.successes,
            "exceptions": entry.metrics.exceptions,
            "validation_failures": entry.metrics.validation_failures,
            "avg_latency_ms": entry.metrics.avg_latency_ms,
        },
    }


def _entry_from_dict(d: dict[str, Any]) -> ArchiveEntry:
    body = GeneratedBody.model_validate(d["body"])
    m = d.get("metrics") or {}
    metrics = ArchiveMetrics(
        invocations=m.get("invocations", 0),
        successes=m.get("successes", 0),
        exceptions=m.get("exceptions", 0),
        validation_failures=m.get("validation_failures", 0),
        avg_latency_ms=m.get("avg_latency_ms", 0.0),
    )
    return ArchiveEntry(
        signature_hash=d["signature_hash"],
        tool_surface_hash=d["tool_surface_hash"],
        version=d["version"],
        parent_version=d.get("parent_version"),
        body=body,
        created_at=datetime.fromisoformat(d["created_at"]),
        trigger=d.get("trigger", "initial"),
        promoted=d.get("promoted", False),
        metrics=metrics,
    )


def entries_to_json(entries: list[ArchiveEntry]) -> str:
    return json.dumps(
        {"version": 1, "entries": [_entry_to_dict(e) for e in entries]},
        indent=2,
    )


def entries_from_json(data: str) -> list[ArchiveEntry]:
    obj = json.loads(data)
    return [_entry_from_dict(d) for d in obj.get("entries", [])]


__all__ = ["entries_from_json", "entries_to_json"]
