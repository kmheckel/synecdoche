"""The transformed function: what ``syn.jit`` and ``syn.oracle`` return.

An ``Fn`` is deliberately thin — a callable with a signature and a kind.
Everything interesting about it lives elsewhere: its variants in the
archive, its machinery in the backend, and all introspection in the
module-level transforms (``syn.lineage(f)``, ``syn.solidify(f)``, ...).
The call site never knows or cares whether the body it reaches was
handwritten, synthesized, or is a direct inference.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import inspect
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from .config import current_backend
from .exceptions import FrameworkError
from .signature import CallSignature

if TYPE_CHECKING:
    from .backend import Backend

Kind = Literal["jit", "oracle"]


class Fn:
    """A typed contract, callable through a backend.

    For ``kind='jit'`` with a handwritten body, that body is kept as the
    *seed* — the lineage's generation zero, run natively until it fails.
    With an empty body, the first call compiles one. For ``kind='oracle'``
    every call is a single typed inference.
    """

    def __init__(
        self,
        original: Callable[..., Any],
        *,
        kind: Kind,
        backend: Backend | None = None,
        model: Any = None,
        max_repairs: int | None = None,
    ) -> None:
        self._original = original
        self.kind: Kind = kind
        self.signature = CallSignature.from_function(original)
        self._backend_override = backend
        self._model = model
        self._max_repairs = max_repairs
        self._seed_source: str | None = None
        if kind == "jit" and not _body_is_empty(original):
            self._seed_source = _seed_source(original)
        self._last_inputs: dict[str, Any] | None = None
        functools.update_wrapper(self, original)

    @property
    def backend(self) -> Backend:
        return self._backend_override or current_backend()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = bind_inputs(self._original, args, kwargs)
        coro = self._invoke(inputs)
        if inspect.iscoroutinefunction(self._original):
            return coro  # declared async — the caller awaits, as they wrote it
        return run_maybe_async(coro)

    async def _invoke(self, inputs: dict[str, Any]) -> Any:
        if self.kind == "oracle":
            return await self.backend._call_oracle(self, inputs)
        return await self.backend._call_jit(self, inputs)

    def __repr__(self) -> str:
        return f"<synecdoche.Fn {self.signature.qualname} kind={self.kind}>"


def as_fn(f: Any, transform: str) -> Fn:
    if not isinstance(f, Fn):
        raise FrameworkError(
            f"syn.{transform}() expects a transformed function "
            f"(the result of @syn.jit or @syn.oracle), got {type(f).__name__}."
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


__all__ = ["Fn", "Kind", "as_fn", "bind_inputs", "run_maybe_async"]
