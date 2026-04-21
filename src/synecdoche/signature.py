"""Call signatures: the stable, canonical identity of a decorated function.

A `CallSignature` captures everything the compiler needs to know statically:
parameter names and types, return type, docstring, and a content hash that's
used as the archive key.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import TypeAdapter


@dataclass(frozen=True)
class Param:
    name: str
    annotation: Any
    has_default: bool
    default_repr: str | None

    def render(self) -> str:
        """Render as a Python signature fragment, e.g. `x: int = 0`."""
        t = _type_name(self.annotation)
        if self.has_default:
            return f"{self.name}: {t} = {self.default_repr}"
        return f"{self.name}: {t}"


@dataclass(frozen=True)
class CallSignature:
    """Everything the compiler needs to know about a function's shape."""

    qualname: str
    docstring: str
    params: tuple[Param, ...]
    return_type: Any
    return_schema: dict[str, Any] = field(default_factory=dict)
    signature_hash: str = ""

    @classmethod
    def from_synthetic(
        cls,
        *,
        qualname: str,
        docstring: str,
        params: tuple[Param, ...],
        return_type: Any,
    ) -> CallSignature:
        """Build a CallSignature from pre-parsed pieces (no real Python function)."""
        schema = _json_schema_of(return_type)
        sig_hash = _hash_signature(qualname=qualname, params=list(params), return_type=return_type)
        return cls(
            qualname=qualname,
            docstring=docstring,
            params=params,
            return_type=return_type,
            return_schema=schema,
            signature_hash=sig_hash,
        )

    @classmethod
    def from_function(cls, fn: Callable[..., Any]) -> CallSignature:
        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}
        params: list[Param] = []
        for name, p in sig.parameters.items():
            ann = hints.get(name, p.annotation if p.annotation is not p.empty else Any)
            has_default = p.default is not p.empty
            default_repr = repr(p.default) if has_default else None
            params.append(
                Param(
                    name=name,
                    annotation=ann,
                    has_default=has_default,
                    default_repr=default_repr,
                )
            )
        return_type = hints.get(
            "return", sig.return_annotation if sig.return_annotation is not sig.empty else Any
        )
        return_schema = _json_schema_of(return_type)
        sig_hash = _hash_signature(
            qualname=fn.__qualname__,
            params=params,
            return_type=return_type,
        )
        return cls(
            qualname=fn.__qualname__,
            docstring=(inspect.getdoc(fn) or "").strip(),
            params=tuple(params),
            return_type=return_type,
            return_schema=return_schema,
            signature_hash=sig_hash,
        )

    def render(self) -> str:
        """Render the signature as a `def name(...) -> Return:` line."""
        params_src = ", ".join(p.render() for p in self.params)
        ret = _type_name(self.return_type)
        return f"def {self.qualname.split('.')[-1]}({params_src}) -> {ret}:"


def _type_name(t: Any) -> str:
    if t is type(None):
        return "None"
    if hasattr(t, "__module__") and t.__module__ == "builtins":
        return t.__name__
    return getattr(t, "__name__", None) or repr(t)


def _json_schema_of(t: Any) -> dict[str, Any]:
    if t is inspect.Parameter.empty or t is Any:
        return {}
    try:
        return TypeAdapter(t).json_schema()
    except Exception:
        return {}


def _hash_signature(
    *,
    qualname: str,
    params: list[Param],
    return_type: Any,
) -> str:
    payload = {
        "qualname": qualname,
        "params": [
            {
                "name": p.name,
                "type": _type_name(p.annotation),
                "schema": _json_schema_of(p.annotation),
                "default": p.default_repr,
            }
            for p in params
        ],
        "return_type": _type_name(return_type),
        "return_schema": _json_schema_of(return_type),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = ["CallSignature", "Param"]
