"""Sandbox abstraction and the default Monty-backed implementation.

The sandbox assembles the compiler's `GeneratedBody` into an executable
Python script, then runs it inside `pydantic-monty`, dispatching external
function calls (MCP tools + inline helpers) back through the host.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .exceptions import SandboxError

if TYPE_CHECKING:
    from .compiler import GeneratedBody
    from .signature import CallSignature
    from .surface import ToolSurface


# ---------------------------------------------------------------------------
# Public protocols
# ---------------------------------------------------------------------------


class Sandbox(Protocol):
    async def execute(
        self,
        *,
        body: GeneratedBody,
        signature: CallSignature,
        surface: ToolSurface,
        inputs: dict[str, Any],
        external_functions: dict[str, Callable[..., Awaitable[Any]]],
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Source assembly
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_IMPORTS = {
    "typing",
    "dataclasses",
    "json",
    "re",
    "math",
    "pathlib",
    "asyncio",
    "collections",
    "itertools",
    "functools",
    "pydantic",
    "datetime",
}


@dataclass
class AssembledScript:
    source: str
    stubs: str
    input_names: list[str]


def assemble_script(
    *,
    body: GeneratedBody,
    signature: CallSignature,
    surface: ToolSurface,
    allowed_imports: set[str] | None = None,
) -> AssembledScript:
    """Produce the final script the sandbox will execute.

    The script is a sequence of (filtered) imports, the body source, followed
    by a single `await solve(<args>)` expression. Monty evaluates the last
    top-level expression as the return value of the run.
    """
    allowed = allowed_imports or _DEFAULT_ALLOWED_IMPORTS
    imports = [ln for ln in body.imports if _is_allowed_import(ln, allowed)]

    input_names = [p.name for p in signature.params]
    call_args = ", ".join(input_names)

    parts: list[str] = []
    parts.extend(imports)
    if imports:
        parts.append("")
    parts.append(body.body.rstrip())
    parts.append("")
    parts.append(f"await solve({call_args})")

    # Monty's type checker needs input variable declarations in the stubs
    # alongside the tool stubs — otherwise the top-level `await solve(x, ...)`
    # references undefined names.
    input_decls = "\n".join(
        f"{p.name}: {_type_hint_for_param(p.annotation)}" for p in signature.params
    )
    tool_stubs = surface.render_stubs()
    stubs = "\n".join(s for s in (input_decls, tool_stubs) if s)

    return AssembledScript(
        source="\n".join(parts),
        stubs=stubs,
        input_names=input_names,
    )


def _type_hint_for_param(annotation: Any) -> str:
    """Render a type annotation as a stub-level hint string.

    Any non-trivial type becomes `object` — type_check_stubs is only used to
    satisfy Monty's reference resolver at the top-level call site, so coarse
    hints are fine. The body's own parameter list enforces the real types.
    """
    import typing

    if annotation is None or annotation is type(None):
        return "None"
    if annotation in (int, float, bool, str, bytes, list, dict, tuple, set, object):
        return annotation.__name__
    origin = typing.get_origin(annotation)
    if origin is not None:
        if origin in (list, set, tuple):
            return origin.__name__
        if origin is dict:
            return "dict"
    return "object"


def _is_allowed_import(line: str, allowed: set[str]) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if s.startswith("import "):
        mod = s[len("import ") :].split()[0].split(".")[0]
    elif s.startswith("from "):
        mod = s[len("from ") :].split()[0].split(".")[0]
    else:
        return False
    return mod in allowed


# ---------------------------------------------------------------------------
# Monty implementation
# ---------------------------------------------------------------------------


class MontySandbox:
    """Default sandbox: `pydantic-monty` with async external function dispatch."""

    def __init__(self, *, type_check: bool = True, allowed_imports: set[str] | None = None) -> None:
        self.type_check = type_check
        self.allowed_imports = allowed_imports or _DEFAULT_ALLOWED_IMPORTS

    async def execute(
        self,
        *,
        body: GeneratedBody,
        signature: CallSignature,
        surface: ToolSurface,
        inputs: dict[str, Any],
        external_functions: dict[str, Callable[..., Awaitable[Any]]],
    ) -> Any:
        import pydantic_monty as pm

        script = assemble_script(
            body=body, signature=signature, surface=surface, allowed_imports=self.allowed_imports
        )

        try:
            monty = pm.Monty(
                script.source,
                script_name=f"{signature.qualname.split('.')[-1]}.py",
                inputs=script.input_names,
                type_check=self.type_check,
                type_check_stubs=script.stubs or None,
            )
        except pm.MontySyntaxError as e:
            raise SandboxError(
                f"Sandbox syntax error while preparing body: {e}",
                signature=signature,
                original=e,
            ) from e
        except pm.MontyTypingError as e:
            raise SandboxError(
                f"Sandbox type check failed: {e}",
                signature=signature,
                original=e,
            ) from e

        try:
            return await monty.run_async(
                inputs=inputs,
                external_functions=external_functions,
            )
        except pm.MontyError as e:
            raise SandboxError(
                f"Sandbox execution error: {e}",
                signature=signature,
                original=e,
            ) from e
        except Exception as e:
            raise SandboxError(
                f"Body raised during execution: {type(e).__name__}: {e}",
                signature=signature,
                original=e,
            ) from e


__all__ = ["AssembledScript", "MontySandbox", "Sandbox", "assemble_script"]
