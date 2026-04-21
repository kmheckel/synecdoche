"""Framework exceptions.

Rich exceptions are the API between domain code and the compiler: the more
structured the exception, the more precise the repair prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pydantic

    from .signature import CallSignature


class FrameworkError(Exception):
    """Base class for all synecdoche errors."""


@dataclass
class RepairAttempt:
    version: int
    exception_type: str
    exception_message: str
    body_excerpt: str


class CompilationError(FrameworkError):
    """The compiler failed to produce a valid body after all repair attempts."""

    def __init__(
        self,
        message: str,
        signature: CallSignature | None = None,
        attempts: list[RepairAttempt] | None = None,
    ) -> None:
        super().__init__(message)
        self.signature = signature
        self.attempts = attempts or []


class SandboxError(FrameworkError):
    """The generated body raised during execution."""

    def __init__(
        self,
        message: str,
        *,
        signature: CallSignature | None = None,
        original: BaseException | None = None,
        trace: Any = None,
    ) -> None:
        super().__init__(message)
        self.signature = signature
        self.original = original
        self.trace = trace


class ValidationError(FrameworkError):
    """The body returned a value that does not match the declared return type."""

    def __init__(
        self,
        message: str,
        *,
        signature: CallSignature | None = None,
        value: Any = None,
        pydantic_error: pydantic.ValidationError | None = None,
    ) -> None:
        super().__init__(message)
        self.signature = signature
        self.value = value
        self.pydantic_error = pydantic_error


@dataclass
class ContextWindowExceeded(FrameworkError):
    """Raised when a call would overflow an LLM context window."""

    measured: int
    limit: int
    at_call: str = ""

    def __post_init__(self) -> None:
        super().__init__(
            f"ContextWindowExceeded: measured={self.measured} limit={self.limit} at={self.at_call!r}"
        )


@dataclass
class ToolSurfaceDrift(FrameworkError):
    """Raised when a tool's actual schema doesn't match what the body expected."""

    tool_name: str
    expected_schema: dict[str, Any] = field(default_factory=dict)
    actual_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"ToolSurfaceDrift: tool={self.tool_name!r}")


@dataclass
class BudgetExceeded(FrameworkError):
    """Raised when a runtime budget is exhausted."""

    kind: Literal["depth", "tool_calls", "wall_time"]
    limit: float
    measured: float

    def __post_init__(self) -> None:
        super().__init__(
            f"BudgetExceeded: kind={self.kind} limit={self.limit} measured={self.measured}"
        )


__all__ = [
    "BudgetExceeded",
    "CompilationError",
    "ContextWindowExceeded",
    "FrameworkError",
    "RepairAttempt",
    "SandboxError",
    "ToolSurfaceDrift",
    "ValidationError",
]
