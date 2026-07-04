"""The public surface: one decorator, and functions that act on it.

``@syn`` (the module itself is the decorator; ``syn.fn`` is the same thing
with a name) turns a typed signature + docstring into a function whose body
is generated, cached, and improved under feedback. Everything else operates
on decorated functions:

===============  ============================================================
``fn`` / ``@syn``  generate code to meet the spec — from the handwritten
                 body if there is one, from the signature + docstring if
                 not — cache by (signature, tool surface), heal on exception
``vmap``         map a function over the leading argument, concurrently
``feedback``     record a critique against the current body
``descend``      one update step: critiques -> revised program
``evolve``       the training loop: breed variants against examples, keep
                 the fittest
``solidify``     render the current body back into committable source
``lineage`` /    inspect the learned artifact — which is always ordinary,
``champion`` /   readable Python, never opaque state
``signals`` / ``rollback``
===============  ============================================================

This is heuristic learning in the sense of Weng's "Learning Beyond
Gradients": the loop of state, action, feedback, update — where the thing
being updated is program structure, not weights. ``feedback`` accumulates
the learning signal; ``descend`` and ``evolve`` apply it; the archive keeps
the history explicit, readable, and refactorable.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from .evolution import EvolutionReport, Signal
from .exceptions import FrameworkError
from .fn import Fn, as_fn, bind_inputs, run_maybe_async

if TYPE_CHECKING:
    from .archive import Variant
    from .backend import Backend

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------


def fn(
    f: F | None = None,
    *,
    backend: Backend | None = None,
    model: Any = None,
    max_repairs: int | None = None,
) -> Any:
    """Generate code to meet the function specification. ``@syn`` is this.

    A handwritten body is kept as the lineage's generation zero and runs
    natively; when it raises, the exception becomes the compile context for
    a fixed descendant. An empty body (docstring / ``pass`` / ``...``) is
    generated from the spec alone. Either way the cache key is
    ``(signature, tool surface)`` — change the types or the tools and you
    get a fresh generation.

    Generated bodies are flat by construction: they may call mounted MCP
    tools and the built-in ``infer`` primitive (one typed judgment call),
    but can never spawn further synthesized functions. Composition happens
    in your own Python, where the call graph stays readable.
    """

    def wrap(func: F) -> F:
        return Fn(func, backend=backend, model=model, max_repairs=max_repairs)  # type: ignore[return-value]

    return wrap(f) if f is not None else wrap


# ---------------------------------------------------------------------------
# vmap — batching
# ---------------------------------------------------------------------------


def vmap(f: Callable[..., Any], *, concurrency: int | None = None) -> Callable[..., Any]:
    """Map ``f`` over its leading argument, concurrently.

    ``vmap(f)(xs, *rest)`` calls ``f(x, *rest)`` for every ``x`` in ``xs``
    and returns the results in order. Decorated functions run concurrently
    on the event loop; plain functions are mapped as-is. Set ``concurrency``
    to bound the fan-out.
    """

    def mapped(items: Any, *args: Any, **kwargs: Any) -> Any:
        semaphore = asyncio.Semaphore(concurrency) if concurrency else None

        async def one(item: Any) -> Any:
            if semaphore is None:
                return await _acall(f, item, *args, **kwargs)
            async with semaphore:
                return await _acall(f, item, *args, **kwargs)

        async def gather() -> list[Any]:
            return await asyncio.gather(*(one(item) for item in items))

        return run_maybe_async(gather())

    mapped.__name__ = f"vmap({getattr(f, '__name__', repr(f))})"
    return mapped


async def _acall(f: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = f(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


# ---------------------------------------------------------------------------
# feedback / descend — the learning signal and the update step
# ---------------------------------------------------------------------------


def feedback(f: Callable[..., Any], score: float, note: str = "") -> None:
    """Record a critique against the current body.

    ``score`` is in [0, 1]: below 0.5 pushes away from the current body,
    above 0.5 reinforces it. The note is what the compiler actually reads —
    say *what was wrong*, not just how wrong it was. Exceptions record
    themselves; this is for the failures that don't raise.
    """
    fn = as_fn(f, "feedback")
    champ = _require_champion(fn)
    fn.backend._record_signal(fn, champ, Signal.from_feedback(score, note))


def descend(f: Callable[..., Any]) -> Callable[..., Any]:
    """One update step: regenerate the current body under its accumulated
    signals, shadow-validate against the last inputs, promote the result.

    Returns ``f`` — the program *is* the parameter, and its new state lives
    in the archive.
    """
    fn = as_fn(f, "descend")
    run_maybe_async(fn.backend._descend(fn))
    return f


# ---------------------------------------------------------------------------
# evolve — the training loop
# ---------------------------------------------------------------------------


def evolve(
    f: Callable[..., Any],
    examples: list[dict[str, Any] | tuple],
    *,
    generations: int = 2,
    population: int = 4,
    score: Callable[[dict[str, Any], Any], float] | None = None,
    promote: bool = True,
) -> EvolutionReport:
    """Breed variants of ``f`` against examples and keep the fittest.

    Each generation produces ``population`` offspring — mutations of the
    fittest, crossovers of the top two, fresh generations — and scores
    every one on all examples (validated success x optional ``score``).
    The fittest variant overall is promoted and serves subsequent calls.
    This is the loop behind heuristic discovery: programs as hypotheses,
    an evaluator as the fitness function, selection over readable code.
    """
    fn = as_fn(f, "evolve")
    sets = [
        ex if isinstance(ex, dict) else bind_inputs(fn._original, tuple(ex), {}) for ex in examples
    ]
    return run_maybe_async(
        fn.backend._evolve(
            fn,
            sets,
            generations=generations,
            population=population,
            score=score,
            promote=promote,
        )
    )


# ---------------------------------------------------------------------------
# Introspection — the learned artifact is ordinary Python
# ---------------------------------------------------------------------------


def champion(f: Callable[..., Any]) -> Variant | None:
    """The variant currently serving calls, or None before the first call."""
    fn = as_fn(f, "champion")
    surface_hash = fn.backend._surface_hash_now()
    return fn.backend.archive.champion(fn.signature.signature_hash, surface_hash)


def lineage(f: Callable[..., Any]) -> list[Variant]:
    """Every variant ever produced for this signature, in version order."""
    fn = as_fn(f, "lineage")
    surface_hash = fn.backend._surface_hash_now()
    return fn.backend.archive.population(fn.signature.signature_hash, surface_hash)


def signals(f: Callable[..., Any]) -> list[Signal]:
    """All signals recorded against the current champion."""
    fn = as_fn(f, "signals")
    champ = _require_champion(fn)
    return fn.backend.archive.signals_for(
        fn.signature.signature_hash, champ.surface_hash, champ.version
    )


def solidify(f: Callable[..., Any], path: str | Path | None = None) -> str:
    """Render the current body back into ordinary Python source.

    Returns the source (and writes it to ``path`` if given), with the body
    renamed to the function's own name and a provenance header. Commit it
    and it is deterministic code; decorate it with ``@syn`` again and it
    seeds the next lineage.
    """
    fn = as_fn(f, "solidify")
    champ = _require_champion(fn)
    name = fn.signature.qualname.split(".")[-1]
    if champ.operator == "seed":
        src = champ.body.body
    else:
        sigs = fn.backend.archive.signals_for(
            fn.signature.signature_hash, champ.surface_hash, champ.version
        )
        header = (
            f"# Solidified by synecdoche.\n"
            f"# contract: {fn.signature.render()}\n"
            f"# variant:  v{champ.version} ({champ.operator}"
            f"{' <- v' + ', v'.join(map(str, champ.parents)) if champ.parents else ''}), "
            f"score {champ.score(sigs):.2f}\n"
            f"# reasoning: {champ.body.reasoning}\n"
        )
        imports = "\n".join(champ.body.imports)
        body = champ.body.body.replace("async def solve(", f"async def {name}(", 1)
        src = "\n\n".join(part for part in (header.rstrip(), imports, body.rstrip()) if part)
        src += "\n"
    if path is not None:
        Path(path).write_text(src)
    return src


def rollback(f: Callable[..., Any]) -> Variant | None:
    """Demote the champion to its first parent. Returns the new champion."""
    fn = as_fn(f, "rollback")
    champ = _require_champion(fn)
    return fn.backend.archive.rollback(fn.signature.signature_hash, champ.surface_hash)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_champion(fn: Fn) -> Variant:
    surface_hash = fn.backend._surface_hash_now()
    champ = fn.backend.archive.champion(fn.signature.signature_hash, surface_hash)
    if champ is None:
        raise FrameworkError(f"{fn.signature.qualname} has no champion yet — call it once first.")
    return champ


__all__ = [
    "champion",
    "descend",
    "evolve",
    "feedback",
    "fn",
    "lineage",
    "rollback",
    "signals",
    "solidify",
    "vmap",
]
