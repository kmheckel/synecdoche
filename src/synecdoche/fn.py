"""The Fn handle: a typed contract whose implementation is a population.

``@rt.fn`` returns an ``Fn`` — callable exactly like the function it wraps,
but also a reflective object. Code in synecdoche exists in three phases:

- **solid** — you wrote the body. It runs natively and is registered as the
  *seed* of the lineage; if it ever raises, the runtime melts it into a
  sandboxed descendant that fixes the defect.
- **synth** — you wrote only the contract (signature + docstring). The first
  call spawns a body; the archive's champion serves every call after.
- **oracle** — no code at all. The model is the body: one typed inference
  per call, nothing archived.

Solid and synth are the same machinery observed at different moments: a
handwritten body is generation zero, a synthesized body is generation one —
both are variants in the same archive, under the same selection pressure.
``solidify()`` runs the loop the other way, rendering the current champion
back into source you can commit, at which point it is ordinary deterministic
code again. Nothing about the call site changes as code moves between
phases; the contract is the fixed point, the implementation is fluid.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .evolution import Signal
from .exceptions import FrameworkError
from .signature import CallSignature

if TYPE_CHECKING:
    from .archive import Variant
    from .runtime import Runtime

Mode = Literal["solid", "synth", "oracle"]


@dataclass
class EvolutionReport:
    """Outcome of one offline evolution run."""

    champion: Variant
    evaluated: list[tuple[Variant, float]]
    generations: int


class Fn:
    """A callable contract bound to a runtime.

    Calling it resolves the contract through whichever phase it is in.
    Everything else on this class is reflection: inspect the lineage, grade
    the champion, take a gradient step, evolve offline, or freeze the result
    back into source.
    """

    def __init__(
        self,
        runtime: Runtime,
        original: Callable[..., Any],
        *,
        mode: Mode | Literal["auto"] = "auto",
        model: Any = None,
        max_repairs: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._original = original
        self.signature = CallSignature.from_function(original)
        self.mode: Mode = _resolve_mode(original, mode)
        self._model = model
        self._max_repairs = max_repairs
        self._seed_source = _seed_source(original) if self.mode == "solid" else None
        self._last_inputs: dict[str, Any] | None = None
        functools.update_wrapper(self, original)

    # ------------------------------------------------------------------
    # Calling
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = _bind_inputs(self._original, args, kwargs)
        coro = self._invoke(inputs)
        if inspect.iscoroutinefunction(self._original):
            return coro  # declared async — the caller awaits, as they wrote it
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro  # inside a loop — hand back an awaitable

    async def _invoke(self, inputs: dict[str, Any]) -> Any:
        if self.mode == "oracle":
            return await self._runtime._call_oracle(self, inputs)
        return await self._runtime._call_fn(self, inputs)

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    @property
    def champion(self) -> Variant | None:
        """The variant currently serving calls, or None before first call."""
        self._require_code_mode("champion")
        surface_hash = self._runtime._surface_hash_now()
        return self._runtime.archive.champion(self.signature.signature_hash, surface_hash)

    def lineage(self) -> list[Variant]:
        """Every variant ever produced for this contract, in version order."""
        self._require_code_mode("lineage")
        surface_hash = self._runtime._surface_hash_now()
        return self._runtime.archive.population(self.signature.signature_hash, surface_hash)

    def signals(self) -> list[Signal]:
        """All signals recorded against the current champion."""
        champ = self._require_champion()
        return self._runtime.archive.signals_for(
            self.signature.signature_hash, champ.surface_hash, champ.version
        )

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def feedback(self, score: float, note: str = "") -> None:
        """Record a soft gradient against the champion.

        ``score`` in [0, 1]: below 0.5 pushes away from the current body,
        above 0.5 reinforces it. The note is what the compiler actually
        reads — say *what was wrong*, not just how wrong it was.
        """
        champ = self._require_champion()
        signal = Signal.from_feedback(score, note)
        self._runtime._record_signal(self, champ, signal)

    def backward(self) -> Variant:
        """Take a textual gradient step: mutate the champion under its
        accumulated signals, shadow-validate, and promote the descendant."""
        self._require_code_mode("backward")
        return self._run(self._runtime._backward(self))

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(
        self,
        examples: list[dict[str, Any] | tuple],
        *,
        generations: int = 2,
        population: int = 4,
        score: Callable[[dict[str, Any], Any], float] | None = None,
        promote: bool = True,
    ) -> EvolutionReport:
        """Offline evolution: breed variants against examples, keep the fittest.

        Each generation produces ``population`` offspring by mutation of the
        fittest, crossover of the top two, and fresh spawns; every offspring
        is scored on all examples (validated success x optional ``score``).
        The fittest variant overall is promoted to champion.
        """
        self._require_code_mode("evolve")
        sets = [
            ex if isinstance(ex, dict) else _bind_inputs(self._original, tuple(ex), {})
            for ex in examples
        ]
        return self._run(
            self._runtime._evolve(
                self,
                sets,
                generations=generations,
                population=population,
                score=score,
                promote=promote,
            )
        )

    # ------------------------------------------------------------------
    # Phase changes
    # ------------------------------------------------------------------

    def solidify(self, path: str | Path | None = None) -> str:
        """Freeze the champion into ordinary Python source.

        Returns the source (and writes it to ``path`` if given), with the
        body renamed to the contract's own name and a provenance header. The
        loop closes when you commit it: decorated again, it becomes the seed
        of the next lineage.
        """
        champ = self._require_champion()
        name = self.signature.qualname.split(".")[-1]
        if champ.operator == "seed":
            src = champ.body.body
        else:
            signals = self.signals()
            header = (
                f"# Solidified by synecdoche.\n"
                f"# contract: {self.signature.render()}\n"
                f"# variant:  v{champ.version} ({champ.operator}"
                f"{' <- v' + ', v'.join(map(str, champ.parents)) if champ.parents else ''}), "
                f"score {champ.score(signals):.2f}\n"
                f"# reasoning: {champ.body.reasoning}\n"
            )
            imports = "\n".join(champ.body.imports)
            body = champ.body.body.replace("async def solve(", f"async def {name}(", 1)
            src = "\n\n".join(part for part in (header.rstrip(), imports, body.rstrip()) if part)
            src += "\n"
        if path is not None:
            Path(path).write_text(src)
        return src

    def rollback(self) -> Variant | None:
        """Demote the champion to its first parent. Returns the new champion."""
        champ = self._require_champion()
        return self._runtime.archive.rollback(self.signature.signature_hash, champ.surface_hash)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_code_mode(self, op: str) -> None:
        if self.mode == "oracle":
            raise FrameworkError(
                f"{op}() is meaningless for an oracle fn — there is no code, "
                f"only inference. Use mode='synth' if you want an evolvable body."
            )

    def _require_champion(self) -> Variant:
        self._require_code_mode("this operation")
        champ = self.champion
        if champ is None:
            raise FrameworkError(
                f"{self.signature.qualname} has no champion yet — call it once first."
            )
        return champ

    @staticmethod
    def _run(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro  # inside a loop — hand back an awaitable

    def __repr__(self) -> str:
        return f"<synecdoche.Fn {self.signature.qualname} mode={self.mode}>"


# ---------------------------------------------------------------------------
# Mode detection and seed extraction
# ---------------------------------------------------------------------------


def _resolve_mode(fn: Callable[..., Any], mode: Mode | Literal["auto"]) -> Mode:
    if mode != "auto":
        return mode
    return "synth" if _body_is_empty(fn) else "solid"


def _function_def(fn: Callable[..., Any]) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]:
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise FrameworkError(f"@rt.fn expects a plain function, got {type(node).__name__}")
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


def _bind_inputs(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


__all__ = ["EvolutionReport", "Fn", "Mode"]
