# Archive

The archive is how calls after the first skip compilation: lookup by
`(signature_hash, tool_surface_hash)` returns the current promoted body.

- **SQLite** by default (`archive="./.archive"`).
- **In-memory** for tests (`archive=None` → `MemoryArchive`).
- **Pluggable** — implement the `Archive` protocol.

## Versioning

Every compile inserts a new version. Repair inserts a child version pointing
at its parent. `promote()` marks the current; `rollback()` flips to the
parent for emergency reversal.

## CLI

```bash
synecdoche archive ls
synecdoche archive show <sig_hash> [--version N]
synecdoche archive diff <sig_hash> <v1> <v2>
synecdoche archive rollback <sig_hash>
synecdoche archive export dump.json
synecdoche archive import dump.json --archive ./fresh
```

## Tool-surface invalidation

On startup the runtime computes `tool_surface_hash` from the current MCP
surface. Archive entries whose hash doesn't match are not served; the next
call re-compiles. Stale entries persist for inspection.
