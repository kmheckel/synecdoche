"""Exception-driven repair loop.

When a compiled body raises (or fails return-type validation), we re-invoke
the compiler with a `RepairContext` describing the failure. The revised
body is shadow-validated against the failing inputs before being promoted
to the current archive version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .exceptions import FrameworkError
from .jit import JITContext, RepairContext, TraceFrame

if TYPE_CHECKING:
    from .compiler import GeneratedBody


@dataclass
class RepairInput:
    """Captures a failure so the repair loop can build a RepairContext."""

    prior_body: GeneratedBody
    exception: BaseException
    trace_frames: list[TraceFrame]
    attempts_so_far: int

    def as_context(self) -> RepairContext:
        exc = self.exception
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        exc_args: dict[str, Any] = {}
        # Try to pull structured data out of the exception. Framework exceptions
        # carry typed fields; others fall back to __dict__.
        if isinstance(exc, FrameworkError):
            for k, v in vars(exc).items():
                if k.startswith("_"):
                    continue
                try:
                    exc_args[k] = v if _is_jsonable(v) else repr(v)
                except Exception:
                    exc_args[k] = repr(v)
        else:
            for k, v in vars(exc).items():
                exc_args[k] = repr(v)
        return RepairContext(
            prior_body=self.prior_body,
            exception_type=exc_type,
            exception_message=exc_msg,
            exception_args=exc_args,
            failing_line=None,  # could be derived from traceback if needed
            trace_excerpt=self.trace_frames[-8:],
            prior_attempts=self.attempts_so_far,
        )


def build_repair_ctx(base: JITContext, repair_input: RepairInput) -> JITContext:
    """Return a new JITContext identical to `base` but with repair populated."""
    return JITContext(
        signature=base.signature,
        surface=base.surface,
        inputs=base.inputs,
        archive_neighbors=base.archive_neighbors,
        remaining_depth=base.remaining_depth,
        budget=base.budget,
        repair=repair_input.as_context(),
    )


def _is_jsonable(v: Any) -> bool:
    import json

    try:
        json.dumps(v, default=str)
        return True
    except Exception:
        return False


__all__ = ["RepairInput", "build_repair_ctx"]
