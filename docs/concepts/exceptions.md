# Exceptions

All framework errors inherit from `FrameworkError`.

## User-facing

```python
class CompilationError(FrameworkError):       # compile or repair failed
class SandboxError(FrameworkError):            # body raised at execution
class ValidationError(FrameworkError):         # return value didn't validate
```

Each carries the `signature` that raised it. `SandboxError.original` is the
underlying sandbox exception; `ValidationError.pydantic_error` is the
pydantic diagnostic.

## Framework signals (typed)

```python
class ContextWindowExceeded(FrameworkError):
    measured: int
    limit: int
    at_call: str

class ToolSurfaceDrift(FrameworkError):
    tool_name: str
    expected_schema: dict
    actual_schema: dict

class BudgetExceeded(FrameworkError):
    kind: Literal["depth", "tool_calls", "wall_time"]
    limit: float
    measured: float
```

These propagate unwrapped so the repair prompt sees the typed fields
directly. User code can define its own framework-extending exceptions the
same way.

!!! tip "Design principle"
    Rich exceptions are the API between domain code and the compiler. The
    more structure you put in, the more precision the repair prompt achieves.
