"""The Runtime: models, tools, archive, sandbox — and the selection loop.

The Runtime owns the machinery; ``Fn`` handles own the contracts. One
decorator, ``@rt.fn``, covers the whole spectrum:

    @rt.fn                      # handwritten body -> solid (seed of a lineage)
    def total(xs: list[float]) -> float:
        return sum(xs)

    @rt.fn                      # empty body -> synth (spawned at first call)
    def summarize(root: Path) -> Summary:
        \"\"\"Summarize the architecture of the codebase at root.\"\"\"

    @rt.fn(mode="oracle")       # no code at all -> one typed inference per call
    def sentiment(text: str) -> Sentiment:
        \"\"\"Classify sentiment.\"\"\"

Every call resolves the same way: find the champion variant for
``(signature, tool surface)``, execute it (natively if it is your seed,
sandboxed if it was generated), validate the result against the declared
return type, record the evidence. Failure is not an error path — it is the
selection pressure that produces the next variant.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import TypeAdapter

from .archive import Archive, Variant, now_utc, open_archive
from .compiler import Compiler, GeneratedBody, Inferencer
from .evolution import Budget, Operator, Signal, TraceFrame, Variation
from .exceptions import (
    BudgetExceeded,
    CompilationError,
    FrameworkError,
    RepairAttempt,
    ValidationError,
)
from .fn import EvolutionReport, Fn, Mode
from .sandbox import MontySandbox, Sandbox
from .signature import CallSignature
from .surface import ToolSpec, ToolSurface, build_mcp_surface
from .trace import TraceEvent, Tracer, resolve_tracer

if TYPE_CHECKING:
    from fastmcp import Client
    from pydantic_ai.models import Model

F = TypeVar("F", bound=Callable[..., Any])


class Runtime:
    """The top-level object: owns all configuration and hosts the decorator."""

    def __init__(
        self,
        *,
        # Models — either a single one or split by role.
        model: Model | None = None,
        model_code: Model | None = None,  # compiles bodies (spawn/mutate/cross)
        model_oracle: Model | None = None,  # answers oracle fns
        # External capability surface.
        mcp: list[Client] | None = None,
        # Storage.
        archive: str | Path | Archive | None = None,
        # Sandbox.
        sandbox: Sandbox | None = None,
        # Healing policy.
        heal: bool = True,
        max_repairs: int = 2,
        shadow_validate: bool = True,
        # Budgets.
        max_recursion_depth: int = 6,
        max_tool_calls_per_frame: int = 200,
        # Observability.
        trace: Any = None,
    ) -> None:
        if model is None and (model_code is None or model_oracle is None):
            raise ValueError(
                "Provide either `model=...` or both `model_code=` and `model_oracle=`."
            )
        self.model_code: Model = model_code or model  # type: ignore[assignment]
        self.model_oracle: Model = model_oracle or model  # type: ignore[assignment]

        self._mcp_clients: list[Client] = list(mcp or [])
        self.archive: Archive = open_archive(archive)
        self.sandbox: Sandbox = sandbox or MontySandbox()

        self.heal = heal
        self.max_repairs = max_repairs
        self.shadow_validate = shadow_validate
        self.budget = Budget(
            max_recursion_depth=max_recursion_depth,
            max_tool_calls_per_frame=max_tool_calls_per_frame,
        )
        self.tracer: Tracer = resolve_tracer(trace)

        # Lazy: surface is populated on first use inside an event loop.
        self._surface: ToolSurface | None = None
        self._surface_lock = asyncio.Lock()

        # Per-runtime compiler (caches system-prompt prefix inside pydantic-ai).
        self._compiler = Compiler(self.model_code)

        # Oracle agents keyed by return-type, lazy init.
        self._oracle_cache: dict[Any, Inferencer] = {}

    # ------------------------------------------------------------------
    # The decorator
    # ------------------------------------------------------------------

    @overload
    def fn(self, fn: F) -> F: ...
    @overload
    def fn(
        self,
        *,
        mode: Mode | str = "auto",
        model: Model | None = None,
        max_repairs: int | None = None,
    ) -> Callable[[F], F]: ...
    def fn(
        self,
        fn: F | None = None,
        *,
        mode: Mode | str = "auto",
        model: Model | None = None,
        max_repairs: int | None = None,
    ) -> F | Callable[[F], F]:
        """Bind a typed contract to this runtime.

        ``mode='auto'`` reads the function itself: a real body makes it
        solid, an empty body makes it synth. Pass ``mode='oracle'`` for pure
        inference, or force ``'solid'``/``'synth'`` explicitly.
        """

        def wrap(f: F) -> F:
            return Fn(self, f, mode=mode, model=model, max_repairs=max_repairs)  # type: ignore[return-value]

        if fn is not None:
            return wrap(fn)
        return wrap

    # Aliases from the previous API, kept as sugar over fn().
    def infer(self, fn: F | None = None, *, model: Model | None = None):
        """Sugar for ``fn(mode='oracle')``."""
        if fn is not None:
            return self.fn(mode="oracle")(fn)
        return self.fn(mode="oracle", model=model)

    def recursion(
        self,
        fn: F | None = None,
        *,
        model: Model | None = None,
        max_repair_attempts: int | None = None,
    ):
        """Sugar for ``fn(mode='synth')``."""
        if fn is not None:
            return self.fn(mode="synth")(fn)
        return self.fn(mode="synth", model=model, max_repairs=max_repair_attempts)

    # ------------------------------------------------------------------
    # Oracle pipeline
    # ------------------------------------------------------------------

    async def _call_oracle(self, fn: Fn, inputs: dict[str, Any]) -> Any:
        sig = fn.signature
        self.tracer.emit(TraceEvent("call_start", sig.qualname, {"mode": "oracle"}))
        model = fn._model or self.model_oracle
        key = (sig.return_type, id(model))
        oracle = self._oracle_cache.get(key)
        if oracle is None:
            oracle = Inferencer(model, sig.return_type)
            self._oracle_cache[key] = oracle
        try:
            value = await oracle.infer(
                signature_line=sig.render(),
                docstring=sig.docstring,
                inputs=inputs,
            )
        except Exception as e:
            self.tracer.emit(
                TraceEvent("call_end", sig.qualname, {"mode": "oracle", "error": type(e).__name__})
            )
            raise
        self.tracer.emit(TraceEvent("call_end", sig.qualname, {"mode": "oracle"}))
        return _validate_return(sig, value)

    # ------------------------------------------------------------------
    # Code pipeline (solid + synth): champion -> execute -> select
    # ------------------------------------------------------------------

    async def _call_fn(self, fn: Fn, inputs: dict[str, Any], *, depth: int = 0) -> Any:
        sig = fn.signature
        if depth > self.budget.max_recursion_depth:
            raise BudgetExceeded(
                kind="depth", limit=self.budget.max_recursion_depth, measured=depth
            )

        surface = await self._ensure_surface()
        self.tracer.emit(TraceEvent("call_start", sig.qualname, {"mode": fn.mode, "depth": depth}))

        champion = self.archive.champion(sig.signature_hash, surface.surface_hash)
        if champion is None:
            champion = await self._genesis(fn, surface, inputs)
        else:
            self.tracer.emit(TraceEvent("cache_hit", sig.qualname, {"version": champion.version}))

        fn._last_inputs = dict(inputs)
        max_repairs = fn._max_repairs if fn._max_repairs is not None else self.max_repairs
        attempts: list[RepairAttempt] = []

        for attempt in range(max_repairs + 1):
            t0 = time.perf_counter()
            trace_frames: list[TraceFrame] = []
            try:
                if champion.operator == "seed":
                    raw = await self._execute_native(fn, inputs)
                else:
                    external_fns = self._build_external_functions(surface, trace_frames)
                    raw = await self.sandbox.execute(
                        body=champion.body,
                        signature=sig,
                        surface=surface,
                        inputs=inputs,
                        external_functions=external_fns,
                    )
                value = _validate_return(sig, raw)
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                self.archive.record_metrics(
                    sig.signature_hash,
                    surface.surface_hash,
                    champion.version,
                    success=False,
                    latency_ms=elapsed,
                    validation_failure=isinstance(e, ValidationError),
                )
                self.archive.record_signal(
                    sig.signature_hash,
                    surface.surface_hash,
                    champion.version,
                    Signal.from_exception(e, trace_frames),
                )
                attempts.append(
                    RepairAttempt(
                        version=champion.version,
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                        body_excerpt=champion.body.body[:200],
                    )
                )
                self.tracer.emit(
                    TraceEvent(
                        "call_end",
                        sig.qualname,
                        {
                            "mode": fn.mode,
                            "ok": False,
                            "error": type(e).__name__,
                            "attempt": attempt,
                        },
                    )
                )
                if not self.heal or attempt >= max_repairs:
                    raise CompilationError(
                        f"{sig.qualname} failed after {attempt + 1} attempt(s): {e}",
                        signature=sig,
                        attempts=attempts,
                    ) from e
                champion = await self._descend(fn, surface, inputs, champion)
                continue
            else:
                elapsed = (time.perf_counter() - t0) * 1000
                self.archive.record_metrics(
                    sig.signature_hash,
                    surface.surface_hash,
                    champion.version,
                    success=True,
                    latency_ms=elapsed,
                )
                self.tracer.emit(
                    TraceEvent("call_end", sig.qualname, {"mode": fn.mode, "ok": True})
                )
                return value

        raise CompilationError(
            "Exhausted repair attempts without exception — should be unreachable.",
            signature=sig,
            attempts=attempts,
        )

    async def _execute_native(self, fn: Fn, inputs: dict[str, Any]) -> Any:
        """Run the handwritten seed as ordinary Python — it is trusted code."""
        result = fn._original(**inputs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    # ------------------------------------------------------------------
    # Variation: genesis, descent, gradient steps
    # ------------------------------------------------------------------

    async def _genesis(self, fn: Fn, surface: ToolSurface, inputs: dict[str, Any]) -> Variant:
        """First variant of a lineage: register the seed, or spawn a body."""
        if fn.mode == "solid":
            seed = GeneratedBody(
                reasoning="Handwritten seed — generation zero of this lineage.",
                imports=[],
                body=fn._seed_source or "",
            )
            return self._insert(fn.signature, surface, seed, operator="seed", parents=())
        variation = Variation(
            signature=fn.signature,
            surface=surface,
            inputs=inputs,
            neighbors=self._neighbors(fn.signature),
            budget=self.budget,
        )
        body = await self._compiler_for(fn).vary(variation)
        return self._insert(fn.signature, surface, body, operator="spawn", parents=())

    async def _descend(
        self, fn: Fn, surface: ToolSurface, inputs: dict[str, Any], parent: Variant
    ) -> Variant:
        """Mutate a failing champion under everything recorded against it."""
        signals = self.archive.signals_for(
            fn.signature.signature_hash, surface.surface_hash, parent.version
        )
        variation = Variation(
            signature=fn.signature,
            surface=surface,
            inputs=inputs,
            parents=(parent.body,),
            signals=tuple(signals),
            neighbors=self._neighbors(fn.signature),
            budget=self.budget,
        )
        body = await self._compiler_for(fn).vary(variation)
        return self._insert(
            fn.signature, surface, body, operator="mutate", parents=(parent.version,)
        )

    async def _backward(self, fn: Fn) -> Variant:
        """A textual gradient step: mutate under accumulated (soft) signals."""
        surface = await self._ensure_surface()
        sig = fn.signature
        champion = self.archive.champion(sig.signature_hash, surface.surface_hash)
        if champion is None:
            raise FrameworkError(f"{sig.qualname} has no champion yet — call it once first.")
        signals = self.archive.signals_for(
            sig.signature_hash, surface.surface_hash, champion.version
        )
        if not signals:
            raise FrameworkError(
                f"{sig.qualname} has no signals against its champion — "
                f"record feedback() before backward()."
            )
        self.tracer.emit(
            TraceEvent(
                "backward", sig.qualname, {"signals": len(signals), "version": champion.version}
            )
        )
        variation = Variation(
            signature=sig,
            surface=surface,
            inputs=fn._last_inputs or {},
            parents=(champion.body,),
            signals=tuple(signals),
            neighbors=self._neighbors(sig),
            budget=self.budget,
        )
        body = await self._compiler_for(fn).vary(variation)

        if self.shadow_validate and fn._last_inputs is not None:
            try:
                raw = await self.sandbox.execute(
                    body=body,
                    signature=sig,
                    surface=surface,
                    inputs=fn._last_inputs,
                    external_functions=self._build_external_functions(surface, []),
                )
                _validate_return(sig, raw)
            except Exception as e:
                raise CompilationError(
                    f"Gradient step failed shadow validation: {e}", signature=sig
                ) from e

        return self._insert(sig, surface, body, operator="mutate", parents=(champion.version,))

    # ------------------------------------------------------------------
    # Offline evolution
    # ------------------------------------------------------------------

    async def _evolve(
        self,
        fn: Fn,
        examples: list[dict[str, Any]],
        *,
        generations: int,
        population: int,
        score: Callable[[dict[str, Any], Any], float] | None,
        promote: bool,
    ) -> EvolutionReport:
        surface = await self._ensure_surface()
        sig = fn.signature

        champion = self.archive.champion(sig.signature_hash, surface.surface_hash)
        if champion is None:
            champion = await self._genesis(fn, surface, examples[0] if examples else {})

        pool: dict[int, tuple[Variant, float]] = {}
        fitness = await self._evaluate(fn, surface, champion, examples, score)
        self.archive.set_fitness(
            sig.signature_hash, surface.surface_hash, champion.version, fitness
        )
        pool[champion.version] = (champion, fitness)

        for gen in range(generations):
            ranked = sorted(pool.values(), key=lambda vf: vf[1], reverse=True)
            top1 = ranked[0][0]
            top2 = ranked[1][0] if len(ranked) > 1 else None
            for i in range(population):
                operator, parents = self._pick_operator(i, top1, top2)
                signals = (
                    tuple(
                        self.archive.signals_for(
                            sig.signature_hash, surface.surface_hash, parents[0].version
                        )
                    )
                    if operator == "mutate"
                    else ()
                )
                variation = Variation(
                    signature=sig,
                    surface=surface,
                    inputs=examples[i % len(examples)] if examples else {},
                    parents=tuple(p.body for p in parents),
                    signals=signals,
                    neighbors=self._neighbors(sig),
                    budget=self.budget,
                )
                try:
                    body = await self._compiler_for(fn).vary(variation)
                except Exception as e:
                    self.tracer.emit(
                        TraceEvent(
                            "evolve",
                            sig.qualname,
                            {"gen": gen, "operator": operator, "error": type(e).__name__},
                        )
                    )
                    continue
                offspring = self._insert(
                    sig,
                    surface,
                    body,
                    operator=operator,
                    parents=tuple(p.version for p in parents),
                    promoted=False,
                )
                fitness = await self._evaluate(fn, surface, offspring, examples, score)
                offspring.fitness = fitness
                self.archive.set_fitness(
                    sig.signature_hash, surface.surface_hash, offspring.version, fitness
                )
                pool[offspring.version] = (offspring, fitness)
                self.tracer.emit(
                    TraceEvent(
                        "evolve",
                        sig.qualname,
                        {
                            "gen": gen,
                            "operator": operator,
                            "version": offspring.version,
                            "fitness": round(fitness, 3),
                        },
                    )
                )

        best, best_fitness = max(pool.values(), key=lambda vf: (vf[1], vf[0].version))
        if promote:
            self.archive.promote(sig.signature_hash, surface.surface_hash, best.version)
            best.promoted = True
            self.tracer.emit(
                TraceEvent(
                    "promote",
                    sig.qualname,
                    {"version": best.version, "fitness": round(best_fitness, 3)},
                )
            )
        return EvolutionReport(
            champion=best,
            evaluated=sorted(pool.values(), key=lambda vf: vf[0].version),
            generations=generations,
        )

    @staticmethod
    def _pick_operator(
        i: int, top1: Variant, top2: Variant | None
    ) -> tuple[Operator, tuple[Variant, ...]]:
        """Round-robin the operators: mutate the best, cross the top two, spawn fresh."""
        cycle: list[tuple[Operator, tuple[Variant, ...]]] = [("mutate", (top1,))]
        if top2 is not None:
            cycle.append(("cross", (top1, top2)))
        cycle.append(("spawn", ()))
        return cycle[i % len(cycle)]

    async def _evaluate(
        self,
        fn: Fn,
        surface: ToolSurface,
        variant: Variant,
        examples: list[dict[str, Any]],
        score: Callable[[dict[str, Any], Any], float] | None,
    ) -> float:
        """Fitness of a variant over the examples: validated success x score."""
        if not examples:
            return 0.5
        total = 0.0
        for inputs in examples:
            try:
                if variant.operator == "seed":
                    raw = await self._execute_native(fn, inputs)
                else:
                    raw = await self.sandbox.execute(
                        body=variant.body,
                        signature=fn.signature,
                        surface=surface,
                        inputs=inputs,
                        external_functions=self._build_external_functions(surface, []),
                    )
                value = _validate_return(fn.signature, raw)
            except Exception:
                continue
            total += score(inputs, value) if score is not None else 1.0
        return total / len(examples)

    # ------------------------------------------------------------------
    # Archive and signal plumbing
    # ------------------------------------------------------------------

    def _insert(
        self,
        sig: CallSignature,
        surface: ToolSurface,
        body: GeneratedBody,
        *,
        operator: Operator,
        parents: tuple[int, ...],
        promoted: bool = True,
    ) -> Variant:
        version = self.archive.next_version(sig.signature_hash, surface.surface_hash)
        variant = Variant(
            signature_hash=sig.signature_hash,
            surface_hash=surface.surface_hash,
            version=version,
            operator=operator,
            parents=parents,
            body=body,
            created_at=now_utc(),
            promoted=promoted,
        )
        self.archive.insert(variant)
        if promoted:
            self.archive.promote(sig.signature_hash, surface.surface_hash, version)
        self.tracer.emit(
            TraceEvent(
                "vary",
                sig.qualname,
                {"operator": operator, "version": version, "parents": list(parents)},
            )
        )
        return variant

    def _record_signal(self, fn: Fn, champion: Variant, signal: Signal) -> None:
        self.archive.record_signal(
            fn.signature.signature_hash, champion.surface_hash, champion.version, signal
        )
        self.tracer.emit(
            TraceEvent(
                "signal",
                fn.signature.qualname,
                {"kind": signal.kind, "weight": signal.weight, "version": champion.version},
            )
        )

    def _compiler_for(self, fn: Fn) -> Compiler:
        return Compiler(fn._model) if fn._model is not None else self._compiler

    def _neighbors(self, sig: CallSignature) -> tuple[GeneratedBody, ...]:
        return tuple(v.body for v in self.archive.neighbors(sig.signature_hash, k=3))

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _build_external_functions(
        self,
        surface: ToolSurface,
        trace_frames: list[TraceFrame],
    ) -> dict[str, Callable[..., Awaitable[Any]]]:
        tool_calls = _Counter()

        def wrap_mcp(tool: ToolSpec) -> Callable[..., Awaitable[Any]]:
            client = self._mcp_clients[tool.mcp_client_id or 0]

            async def invoke(**kwargs: Any) -> Any:
                if tool_calls.value >= self.budget.max_tool_calls_per_frame:
                    raise BudgetExceeded(
                        kind="tool_calls",
                        limit=self.budget.max_tool_calls_per_frame,
                        measured=tool_calls.value,
                    )
                tool_calls.inc()
                self.tracer.emit(TraceEvent("tool_call", tool.name, {"args": _truncate(kwargs)}))
                async with client:
                    result = await client.call_tool(tool.name, kwargs)
                out = _unwrap_mcp_result(result)
                trace_frames.append(
                    TraceFrame(
                        kind="tool",
                        name=tool.name,
                        args_summary=_truncate(kwargs),
                        result_summary=_truncate(out),
                    )
                )
                return out

            return invoke

        return {t.name: wrap_mcp(t) for t in surface.tools if t.source == "mcp"}

    # ------------------------------------------------------------------
    # Tool-surface bootstrap
    # ------------------------------------------------------------------

    async def _ensure_surface(self) -> ToolSurface:
        if self._surface is not None:
            return self._surface
        async with self._surface_lock:
            if self._surface is None:
                if self._mcp_clients:
                    self._surface = await build_mcp_surface(self._mcp_clients)
                else:
                    self._surface = ToolSurface(tools=(), surface_hash="empty")
        return self._surface

    def _surface_hash_now(self) -> str:
        """Surface hash for sync reflection paths (champion, lineage, feedback)."""
        if self._surface is not None:
            return self._surface.surface_hash
        if not self._mcp_clients:
            self._surface = ToolSurface(tools=(), surface_hash="empty")
            return self._surface.surface_hash
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._ensure_surface()).surface_hash
        raise FrameworkError(
            "Tool surface not built yet and we're inside an event loop — "
            "call the function once (or await it) before reflecting on it."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Counter:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0

    def inc(self) -> None:
        self.value += 1


def _validate_return(sig: CallSignature, value: Any) -> Any:
    import inspect as _inspect

    if sig.return_type is Any or sig.return_type is _inspect.Parameter.empty:
        return value
    try:
        adapter = TypeAdapter(sig.return_type)
        return adapter.validate_python(value)
    except Exception as e:
        raise ValidationError(
            f"Return value did not validate against {sig.return_type!r}: {e}",
            signature=sig,
            value=value,
        ) from e


def _unwrap_mcp_result(result: Any) -> Any:
    """FastMCP's call_tool returns a structured object; we want the payload."""
    for attr in ("data", "structured_content", "content"):
        if hasattr(result, attr):
            v = getattr(result, attr)
            if v is not None:
                if isinstance(v, list) and v and hasattr(v[0], "text"):
                    return "\n".join(getattr(c, "text", str(c)) for c in v)
                return v
    return result


def _truncate(v: Any, n: int = 200) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


__all__ = ["Runtime"]
