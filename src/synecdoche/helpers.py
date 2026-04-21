"""Inline helper parsing + synthesis.

The compiler emits `GeneratedBody.helpers: list[InlineHelper]`. Each helper
has a `signature` string (e.g. ``"(path: Path, head: str) -> FileKind"``).
At runtime we need to:

1. Parse the signature into typed params + return type.
2. Synthesize a `CallSignature` so the helper looks like an ordinary
   decorated function to the recursion/infer pipelines.
3. Render an `async def` stub of the helper for Monty's type checker.

Type resolution is best-effort. We try (in order) builtins, typing, the
outer function's `__globals__`, and finally fall back to `typing.Any`.
"""

from __future__ import annotations

import ast
import typing
from dataclasses import dataclass
from typing import Any

from .compiler import InlineHelper
from .signature import CallSignature, Param


@dataclass(frozen=True)
class HelperSpec:
    """Resolved form of an `InlineHelper` — ready to hand to the runtime."""

    kind: str  # "infer" | "recursion"
    name: str
    docstring: str
    signature: CallSignature

    def render_stub(self) -> str:
        """Monty-friendly `async def` stub — enough to satisfy the typechecker."""
        params_src = ", ".join(p.render() for p in self.signature.params)
        ret = _annotation_source(self.signature.return_type)
        body_doc = self.docstring.replace('"""', "'''") or "..."
        return f'async def {self.name}({params_src}) -> {ret}:\n    """{body_doc}"""\n    ...\n'


def parse_helper(
    helper: InlineHelper,
    *,
    caller_qualname: str,
    globals_dict: dict[str, Any] | None = None,
) -> HelperSpec:
    """Turn an `InlineHelper` into a `HelperSpec` with a typed CallSignature."""
    params, return_type = _parse_signature_string(helper.signature, globals_dict or {})
    sig = CallSignature.from_synthetic(
        qualname=f"{caller_qualname}.<helper:{helper.name}>",
        docstring=helper.docstring,
        params=params,
        return_type=return_type,
    )
    return HelperSpec(
        kind=helper.kind,
        name=helper.name,
        docstring=helper.docstring,
        signature=sig,
    )


def _parse_signature_string(
    src: str, globals_dict: dict[str, Any]
) -> tuple[tuple[Param, ...], Any]:
    """Parse ``"(x: int, y: str = 'hi') -> Out"`` into Params + return type."""
    # Wrap in a dummy async def so we can rely on Python's parser.
    src = src.strip()
    if not src.startswith("("):
        raise ValueError(f"helper signature must start with '(': {src!r}")
    wrapped = f"async def _f{src}: ...\n"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as e:
        raise ValueError(f"cannot parse helper signature {src!r}: {e}") from e
    fn = tree.body[0]
    assert isinstance(fn, ast.AsyncFunctionDef)

    params: list[Param] = []
    defaults = list(fn.args.defaults)
    positional = fn.args.posonlyargs + fn.args.args
    # Align defaults (rightmost N args have the defaults).
    default_offset = len(positional) - len(defaults)
    for idx, arg in enumerate(positional):
        ann = _resolve_from_ast(arg.annotation, globals_dict) if arg.annotation else Any
        if idx >= default_offset:
            default_expr = defaults[idx - default_offset]
            default_repr = ast.unparse(default_expr)
            has_default = True
        else:
            default_repr = None
            has_default = False
        params.append(
            Param(
                name=arg.arg,
                annotation=ann,
                has_default=has_default,
                default_repr=default_repr,
            )
        )
    return_type = _resolve_from_ast(fn.returns, globals_dict) if fn.returns is not None else Any
    return tuple(params), return_type


def _resolve_from_ast(node: ast.expr | None, globals_dict: dict[str, Any]) -> Any:
    """Evaluate an annotation AST node against builtins + typing + caller globals."""
    if node is None:
        return Any
    src = ast.unparse(node)
    namespace = {**_safe_namespace(), **globals_dict}
    try:
        return eval(src, namespace)
    except Exception:
        return Any


_SAFE_BUILTINS = {
    "int",
    "float",
    "bool",
    "str",
    "bytes",
    "list",
    "dict",
    "tuple",
    "set",
    "None",
    "type",
    "object",
}


def _safe_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {name: __builtins__[name] for name in _SAFE_BUILTINS}  # type: ignore[index]
    # Expose the common typing / stdlib types helpers might reference.
    import datetime
    import pathlib

    ns.update(
        {
            "Any": Any,
            "List": list,
            "Dict": dict,
            "Tuple": tuple,
            "Set": set,
            "Optional": typing.Optional,
            "Union": typing.Union,
            "Literal": typing.Literal,
            "Path": pathlib.Path,
            "datetime": datetime.datetime,
            "date": datetime.date,
        }
    )
    return ns


def _annotation_source(t: Any) -> str:
    """Best-effort render of a type annotation as a source-level string."""
    if t is Any:
        return "object"
    if t is type(None):
        return "None"
    if hasattr(t, "__name__") and getattr(t, "__module__", "") == "builtins":
        return t.__name__
    origin = typing.get_origin(t)
    if origin is not None:
        args = ", ".join(_annotation_source(a) for a in typing.get_args(t))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    return getattr(t, "__name__", None) or "object"


__all__ = ["HelperSpec", "parse_helper"]
