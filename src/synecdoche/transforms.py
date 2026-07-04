"""The transforms: the whole public surface, as composable functions.

Like an array framework, synecdoche is used through a handful of
higher-order functions that take a typed Python function and return (or act
on) a transformed one:

===============  ============================================================
``jit``          compile the function — from its source if it has a body,
                 from its signature + docstring if it doesn't — cache by
                 (signature, tool surface), heal on exception
``oracle``       treat the model itself as the function: one typed
                 inference per call, nothing compiled or cached
``vmap``         map a function over the leading argument, concurrently
``feedback``     record a critique against the current compiled body
``descend``      one optimizer step: critique -> revised program
``evolve``       a training loop: breed variants against examples, keep
                 the fittest
``solidify``     render the current body back into committable source
``lineage`` /    inspect the compiled artifact, like inspecting a lowered
``champion`` /   representation — except here it is ordinary Python
``signals`` / ``rollback``
===============  ============================================================

There is no ``grad``: programs, not tensors, are the parameters here, so
the derivative of a function with respect to a critique is another
function. ``feedback`` plays backprop (it accumulates the signal);
``descend`` plays the optimizer step (it applies them).
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
# jit / oracle — bring functions onto the backend
# ---------------------------------------------------------------------------


def jit(
    f: F | None = None,
    *,
    backend: Backend | None = None,
    model: Any = None,
    max_repairs: int | None = None,
) -> Any:
    """Compile a typed function on first call and cache the result.

    A handwritten body is kept as the lineage's generation zero and runs
    natively; when it raises, the exception becomes the compile context for
    a fixed descendant. An empty body (docstring / ``pass`` / ``...``) is
    compiled from intent alone. Either way the cache key is
    ``(signature, tool surface)`` — change the types or the tools and you
    get a fresh compilation, exactly like retracing.
    """

    def wrap(func: F) -> F:
        return Fn(func, kind="jit", backend=backend, model=model, max_repairs=max_repairs)  # type: ignore[return-value]

    return wrap(f) if f is not None else wrap


def oracle(
    f: F | None = None,
    *,
    backend: Backend | None = None,
    model: Any = None,
) -> Any:
    """Treat the model as the function: one typed inference per call.

    Nothing is compiled and nothing is archived — the neural sequence model
    is invoked as a black-box implementation of the signature, and its
    output is validated against the declared return type.
    """

    def wrap(func: F) -> F:
        return Fn(func, kind="oracle", backend=backend, model=model)  # type: ignore[return-value]

    return wrap(f) if f is not None else wrap


# ---------------------------------------------------------------------------
# vmap — batching
# ---------------------------------------------------------------------------


def vmap(f: Callable[..., Any], *, concurrency: int | None = None) -> Callable[..., Any]:
    """Map ``f`` over its leading argument, concurrently.

    ``vmap(f)(xs, *rest)`` calls ``f(x, *rest)`` for every ``x`` in ``xs``
    and returns the results in order. Transformed functions run
    concurrently on the event loop (useful: oracle calls batch into
    parallel inferences); plain functions are mapped as-is. Set
    ``concurrency`` to bound the fan-out.
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
# feedback / descend — the gradient surrogate
# ---------------------------------------------------------------------------


def feedback(f: Callable[..., Any], score: float, note: str = "") -> None:
    """Record a critique against the current compiled body.

    ``score`` is in [0, 1]: below 0.5 pushes away from the current body,
    above 0.5 reinforces it. The note is what the compiler actually reads —
    say *what was wrong*, not just how wrong it was. Exceptions record
    themselves; this is for the failures that don't raise.
    """
    fn = _jit_fn(f, "feedback")
    champ = _require_champion(fn)
    fn.backend._record_signal(fn, champ, Signal.from_feedback(score, note))


def descend(f: Callable[..., Any]) -> Callable[..., Any]:
    """One optimizer step: recompile the current body under its accumulated
    signals, shadow-validate against the last inputs, promote the result.

    Returns ``f`` — the program *is* the parameter, and its new state lives
    in the archive.
    """
    fn = _jit_fn(f, "descend")
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
    fittest, crossovers of the top two, fresh compilations — and scores
    every one on all examples (validated success x optional ``score``).
    The fittest variant overall is promoted and serves subsequent calls.
    """
    fn = _jit_fn(f, "evolve")
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
# Introspection — the compiled artifact is ordinary Python
# ---------------------------------------------------------------------------


def champion(f: Callable[..., Any]) -> Variant | None:
    """The variant currently serving calls, or None before the first call."""
    fn = _jit_fn(f, "champion")
    surface_hash = fn.backend._surface_hash_now()
    return fn.backend.archive.champion(fn.signature.signature_hash, surface_hash)


def lineage(f: Callable[..., Any]) -> list[Variant]:
    """Every variant ever produced for this signature, in version order."""
    fn = _jit_fn(f, "lineage")
    surface_hash = fn.backend._surface_hash_now()
    return fn.backend.archive.population(fn.signature.signature_hash, surface_hash)


def signals(f: Callable[..., Any]) -> list[Signal]:
    """All signals recorded against the current champion."""
    fn = _jit_fn(f, "signals")
    champ = _require_champion(fn)
    return fn.backend.archive.signals_for(
        fn.signature.signature_hash, champ.surface_hash, champ.version
    )


def solidify(f: Callable[..., Any], path: str | Path | None = None) -> str:
    """Render the current body back into ordinary Python source.

    Returns the source (and writes it to ``path`` if given), with the body
    renamed to the function's own name and a provenance header. Commit it
    and it is deterministic code; decorate it with ``@syn.jit`` again and
    it seeds the next lineage.
    """
    fn = _jit_fn(f, "solidify")
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
    fn = _jit_fn(f, "rollback")
    champ = _require_champion(fn)
    return fn.backend.archive.rollback(fn.signature.signature_hash, champ.surface_hash)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _jit_fn(f: Callable[..., Any], transform: str) -> Fn:
    fn = as_fn(f, transform)
    if fn.kind != "jit":
        raise FrameworkError(
            f"syn.{transform}() applies to @syn.jit functions — an oracle has "
            f"no compiled body to act on."
        )
    return fn


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
    "jit",
    "lineage",
    "oracle",
    "rollback",
    "signals",
    "solidify",
    "vmap",
]
