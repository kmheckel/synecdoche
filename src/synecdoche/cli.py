"""Minimal CLI for inspecting and manipulating a synecdoche archive.

Usage:
    synecdoche archive ls [--archive PATH]
    synecdoche archive show <sig_hash> [--version N] [--archive PATH]
    synecdoche archive diff <sig_hash> <v1> <v2> [--archive PATH]
    synecdoche archive rollback <sig_hash> [--archive PATH]
    synecdoche archive export <output.json> [--archive PATH]
    synecdoche archive import <input.json> [--archive PATH]

Default archive path: `./.archive/synecdoche.sqlite`.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from .archive import Archive, ArchiveEntry, SqliteArchive, open_archive
from .archive_io import entries_from_json, entries_to_json


def _open_archive(path: str | Path) -> Archive:
    return open_archive(str(path))


def _cmd_ls(archive: Archive) -> int:
    # Dump every row; acceptable for a POC archive that tends to be small.
    sqlite = archive if isinstance(archive, SqliteArchive) else None
    if sqlite is None:
        print("ls only supports SqliteArchive for now.", file=sys.stderr)
        return 2
    with sqlite._lock:
        cur = sqlite._conn.execute(
            "SELECT signature_hash, tool_surface_hash, version, parent_version, trigger, "
            "promoted, created_at FROM entries ORDER BY signature_hash, version"
        )
        rows = cur.fetchall()
    if not rows:
        print("(archive is empty)")
        return 0
    print(f"{'SIG':<18} {'SURFACE':<12} {'V':>3} {'PARENT':>7} {'TRIG':<8} {'CUR':<3} CREATED")
    for sig, surf, ver, parent, trig, promoted, created in rows:
        cur_marker = "*" if promoted else ""
        print(
            f"{sig[:16]:<18} {surf[:10]:<12} {ver:>3} {('' if parent is None else parent)!s:>7} "
            f"{trig:<8} {cur_marker:<3} {created}"
        )
    return 0


def _find_one_entry(archive: Archive, sig_hash: str, version: int | None) -> ArchiveEntry | None:
    if not isinstance(archive, SqliteArchive):
        print("show/diff only supports SqliteArchive for now.", file=sys.stderr)
        return None
    with archive._lock:
        if version is not None:
            cur = archive._conn.execute(
                "SELECT DISTINCT tool_surface_hash FROM entries WHERE signature_hash LIKE ?",
                (sig_hash + "%",),
            )
            hashes = [r[0] for r in cur.fetchall()]
            for h in hashes:
                entry = archive.get_version(sig_hash, h, version)
                if entry is not None:
                    return entry
            return None
        cur = archive._conn.execute(
            "SELECT tool_surface_hash FROM entries WHERE signature_hash LIKE ? "
            "AND promoted = 1 LIMIT 1",
            (sig_hash + "%",),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return archive.get_current(sig_hash, row[0])


def _cmd_show(archive: Archive, sig_hash: str, version: int | None) -> int:
    entry = _find_one_entry(archive, sig_hash, version)
    if entry is None:
        print(f"no entry found for {sig_hash}", file=sys.stderr)
        return 1
    print(f"# {entry.signature_hash} (v{entry.version}) — {entry.trigger}")
    print(f"# surface: {entry.tool_surface_hash}")
    print(f"# created: {entry.created_at.isoformat()}")
    print(f"# promoted: {entry.promoted}")
    if entry.parent_version is not None:
        print(f"# parent:   v{entry.parent_version}")
    print()
    print(f"## reasoning\n{entry.body.reasoning}")
    print("\n## imports")
    for imp in entry.body.imports:
        print(f"  {imp}")
    print("\n## helpers")
    for h in entry.body.helpers:
        print(f"  [{h.kind}] {h.name}{h.signature}")
    print("\n## body")
    print(entry.body.body)
    return 0


def _cmd_diff(archive: Archive, sig_hash: str, v1: int, v2: int) -> int:
    a = _find_one_entry(archive, sig_hash, v1)
    b = _find_one_entry(archive, sig_hash, v2)
    if a is None or b is None:
        print(f"one or both versions missing for {sig_hash}", file=sys.stderr)
        return 1
    diff = difflib.unified_diff(
        a.body.body.splitlines(keepends=True),
        b.body.body.splitlines(keepends=True),
        fromfile=f"v{a.version}",
        tofile=f"v{b.version}",
    )
    sys.stdout.writelines(diff)
    return 0


def _cmd_rollback(archive: Archive, sig_hash: str) -> int:
    if not isinstance(archive, SqliteArchive):
        print("rollback only supports SqliteArchive for now.", file=sys.stderr)
        return 2
    with archive._lock:
        cur = archive._conn.execute(
            "SELECT DISTINCT tool_surface_hash FROM entries WHERE signature_hash LIKE ?",
            (sig_hash + "%",),
        )
        hashes = [r[0] for r in cur.fetchall()]
    if not hashes:
        print(f"no entries for {sig_hash}", file=sys.stderr)
        return 1
    # Roll back each (sig, surface) pair.
    rolled_any = False
    for h in hashes:
        result = archive.rollback(sig_hash, h)
        if result is not None:
            print(f"rolled back {sig_hash}/{h[:10]} → v{result.version}")
            rolled_any = True
    if not rolled_any:
        print("nothing to roll back (no parent versions).", file=sys.stderr)
        return 1
    return 0


def _cmd_export(archive: Archive, output: Path) -> int:
    if not isinstance(archive, SqliteArchive):
        print("export only supports SqliteArchive for now.", file=sys.stderr)
        return 2
    entries = archive.history("", "")  # will return [] for empty-string keys
    # For export we want ALL entries across all keys. Query directly.
    with archive._lock:
        cur = archive._conn.execute(
            "SELECT DISTINCT signature_hash, tool_surface_hash FROM entries"
        )
        pairs = cur.fetchall()
    entries = []
    for sh, tsh in pairs:
        entries.extend(archive.history(sh, tsh))
    output.write_text(entries_to_json(entries))
    print(f"wrote {len(entries)} entries to {output}")
    return 0


def _cmd_import(archive: Archive, input_: Path) -> int:
    data = input_.read_text()
    entries = entries_from_json(data)
    for e in entries:
        archive.insert(e)
        if e.promoted:
            archive.promote(e.signature_hash, e.tool_surface_hash, e.version)
    print(f"imported {len(entries)} entries from {input_}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synecdoche")
    sub = parser.add_subparsers(dest="cmd", required=True)
    arch = sub.add_parser("archive", help="archive inspection / management")
    arch_sub = arch.add_subparsers(dest="sub", required=True)

    for cmd in ("ls", "show", "diff", "rollback", "export", "import"):
        p = arch_sub.add_parser(cmd)
        p.add_argument(
            "--archive",
            default="./.archive",
            help="archive path or directory (default: ./.archive)",
        )
        if cmd == "show":
            p.add_argument("sig_hash")
            p.add_argument("--version", type=int)
        elif cmd == "diff":
            p.add_argument("sig_hash")
            p.add_argument("v1", type=int)
            p.add_argument("v2", type=int)
        elif cmd == "rollback":
            p.add_argument("sig_hash")
        elif cmd == "export":
            p.add_argument("output", type=Path)
        elif cmd == "import":
            p.add_argument("input", type=Path)

    args = parser.parse_args(argv)

    archive = _open_archive(args.archive)
    if args.cmd != "archive":
        parser.error(f"unknown command: {args.cmd}")
    if args.sub == "ls":
        return _cmd_ls(archive)
    if args.sub == "show":
        return _cmd_show(archive, args.sig_hash, args.version)
    if args.sub == "diff":
        return _cmd_diff(archive, args.sig_hash, args.v1, args.v2)
    if args.sub == "rollback":
        return _cmd_rollback(archive, args.sig_hash)
    if args.sub == "export":
        return _cmd_export(archive, args.output)
    if args.sub == "import":
        return _cmd_import(archive, args.input)
    parser.error(f"unknown subcommand: {args.sub}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
