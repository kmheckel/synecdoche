# Repair loop

When a compiled body raises (or fails return-type validation), the runtime:

1. Captures the exception + a trace of recent tool/helper calls.
2. Builds a `RepairContext` describing the failure with structured fields.
3. Re-invokes the compiler with a repair-variant prompt.
4. **Shadow-validates** the revised body against the failing inputs *and* up
   to `shadow_validate_samples` prior successful inputs. A regression on any
   prior success rejects the revision.
5. Promotes the revised body as a new archive version.

## Rich exceptions are the API

The more structured your exceptions, the sharper the repair:

```python
class ContextWindowExceeded(FrameworkError):
    measured: int
    limit: int
    at_call: str
```

A plain `ValueError("something went wrong")` teaches the compiler nothing.
A structured exception gives it specifics.

## Framework signals

- `ContextWindowExceeded(measured, limit, at_call)`
- `ToolSurfaceDrift(tool_name, expected_schema, actual_schema)`
- `BudgetExceeded(kind, limit, measured)`

These propagate unwrapped — the repair prompt sees the typed fields and can
suggest specific defensive patterns (chunking, schema-tolerant parsing, etc.).
