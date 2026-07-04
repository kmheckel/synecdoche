"""The decorated function: what ``@syn`` returns.

An ``Fn`` is deliberately thin — a callable with a signature. Everything
interesting about it lives elsewhere: its variants in the archive, its
machinery in the backend, and all introspection in the module-level
functions (``syn.lineage(f)``, ``syn.solidify(f)``, ...). The call site
never knows or cares whether the body it reaches was handwritten or
generated.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import inspect
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .config import current_backend
from .exceptions import FrameworkError
from .signature import CallSignature

if TYPE_CHECKING:
    from .backend import Backend


class Fn:
    """A typed spec, callable through a backend.

    A handwritten body is kept as the *seed* — the lineage's generation
    zero, run natively until it fails. An empty body means the first call
    generates one.
    """

    def __init__(
        self,
        original: Callable[..., Any],
        *,
        backend: Backend | None = None,
        model: Any = None,
        max_repairs: int | None = None,
    ) -> None:
        self._original = original
        self.signature = CallSignature.from_function(original)
        self._backend_override = backend
        self._model = model
        self._max_repairs = max_repairs
        self._seed_source: str | None = None
        if not _body_is_empty(original):
            self._seed_source = _seed_source(original)
        self._last_inputs: dict[str, Any] | None = None
        functools.update_wrapper(self, original)

    @property
    def backend(self) -> Backend:
        return self._backend_override or current_backend()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = bind_inputs(self._original, args, kwargs)
        coro = self.backend._call_fn(self, inputs)
        if inspect.iscoroutinefunction(self._original):
            return coro  # declared async — the caller awaits, as they wrote it
        return run_maybe_async(coro)

    def __repr__(self) -> str:
        return f"<synecdoche.Fn {self.signature.qualname}>"


def as_fn(f: Any, transform: str) -> Fn:
    if not isinstance(f, Fn):
        raise FrameworkError(
            f"syn.{transform}() expects a decorated function "
            f"(the result of @syn), got {type(f).__name__}."
        )
    return f


def run_maybe_async(coro: Any) -> Any:
    """Block on the coroutine from sync code; hand it back inside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return coro


# ---------------------------------------------------------------------------
# Seed detection and extraction
# ---------------------------------------------------------------------------


def _function_def(fn: Callable[..., Any]) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]:
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise FrameworkError(f"expected a plain function, got {type(node).__name__}")
    return node, src


def _body_is_empty(fn: Callable[..., Any]) -> bool:
    """True when the body is only a docstring, `pass`, and/or `...`."""
    try:
        node, _ = _function_def(fn)
    except (OSError, TypeError):
        return False  # no source (REPL, exec) — assume there is a real body
    body = node.body
    if body and _is_docstring(body[0]):
        body = body[1:]
    return all(
        isinstance(stmt, ast.Pass)
        or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        )
        for stmt in body
    )


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _seed_source(fn: Callable[..., Any]) -> str:
    """The handwritten body's source, decorators stripped — generation zero."""
    node, src = _function_def(fn)
    lines = src.splitlines()
    return "\n".join(lines[node.lineno - 1 :]) + "\n"


def bind_inputs(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


__all__ = ["Fn", "as_fn", "bind_inputs", "run_maybe_async"]
