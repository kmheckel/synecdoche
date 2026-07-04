"""The Backend: models, tools, archive, sandbox — the substrate transforms run on.

User code rarely talks to it directly: ``@syn`` binds functions to a
backend — explicitly via ``backend=``, or implicitly through the module
default set by ``synecdoche.configure()``.

Every call resolves the same way: find the champion variant for
``(signature, tool surface)``, execute it (natively if it is your
handwritten seed, sandboxed if it was compiled), validate the result
against the declared return type, record the evidence. Failure is not an
error path — it is the selection pressure that produces the next variant.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from .archive import Archive, Variant, now_utc, open_archive
from .compiler import Compiler, GeneratedBody, Inferencer
from .evolution import Budget, EvolutionReport, Operator, Signal, TraceFrame, Variation
from .exceptions import (
    BudgetExceeded,
    CompilationError,
    FrameworkError,
    RepairAttempt,
    ValidationError,
)
from .fn import Fn
from .sandbox import MontySandbox, Sandbox
from .signature import CallSignature
from .surface import ToolSpec, ToolSurface, build_mcp_surface, empty_surface
from .trace import TraceEvent, Tracer, resolve_tracer

if TYPE_CHECKING:
    from fastmcp import Client
    from pydantic_ai.models import Model


class Backend:
    """Owns all machinery: models, MCP surface, archive, sandbox, tracer."""

    def __init__(
        self,
        *,
        # Models — either a single one or split by role.
        model: Model | None = None,
        model_code: Model | None = None,  # compiles bodies (spawn/mutate/cross)
        model_infer: Model | None = None,  # answers the infer builtin
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
        max_tool_calls_per_frame: int = 200,
        # Observability.
        trace: Any = None,
    ) -> None:
        if model is None and (model_code is None or model_infer is None):
            raise ValueError("Provide either `model=...` or both `model_code=` and `model_infer=`.")
        self.model_code: Model = model_code or model  # type: ignore[assignment]
        self.model_infer: Model = model_infer or model  # type: ignore[assignment]

        self._mcp_clients: list[Client] = list(mcp or [])
        self.archive: Archive = open_archive(archive)
        self.sandbox: Sandbox = sandbox or MontySandbox()

        self.heal = heal
        self.max_repairs = max_repairs
        self.shadow_validate = shadow_validate
        self.budget = Budget(max_tool_calls_per_frame=max_tool_calls_per_frame)
        self.tracer: Tracer = resolve_tracer(trace)

        # Lazy: surface is populated on first use inside an event loop.
        self._surface: ToolSurface | None = None
        self._surface_lock = asyncio.Lock()

        # Single-flight genesis: concurrent first calls (e.g. under vmap)
        # must produce one lineage, not one per call.
        self._genesis_locks: dict[str, asyncio.Lock] = {}

        # Per-backend compiler (caches system-prompt prefix inside pydantic-ai).
        self._compiler = Compiler(self.model_code)

        # The `infer` builtin, lazy init.
        self._infer_agent: Inferencer | None = None

    # ------------------------------------------------------------------
    # The call pipeline: champion -> execute -> select
    # ------------------------------------------------------------------

    async def _call_fn(self, fn: Fn, inputs: dict[str, Any]) -> Any:
        sig = fn.signature
        surface = await self._ensure_surface()
        self.tracer.emit(TraceEvent("call_start", sig.qualname, {}))

        champion = self.archive.champion(sig.signature_hash, surface.surface_hash)
        if champion is None:
            lock = self._genesis_locks.setdefault(sig.signature_hash, asyncio.Lock())
            async with lock:
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
                        {"ok": False, "error": type(e).__name__, "attempt": attempt},
                    )
                )
                if not self.heal or attempt >= max_repairs:
                    raise CompilationError(
                        f"{sig.qualname} failed after {attempt + 1} attempt(s): {e}",
                        signature=sig,
                        attempts=attempts,
                    ) from e
                champion = await self._mutate_champion(fn, surface, inputs, champion)
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
                self.tracer.emit(TraceEvent("call_end", sig.qualname, {"ok": True}))
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
    # Variation: genesis, mutation, descent
    # ------------------------------------------------------------------

    async def _genesis(self, fn: Fn, surface: ToolSurface, inputs: dict[str, Any]) -> Variant:
        """First variant of a lineage: register the seed, or compile a body."""
        if fn._seed_source is not None:
            seed = GeneratedBody(
                reasoning="Handwritten seed — generation zero of this lineage.",
                imports=[],
                body=fn._seed_source,
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

    async def _mutate_champion(
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

    async def _descend(self, fn: Fn) -> Variant:
        """One optimizer step: mutate the champion under its accumulated signals."""
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
                f"record syn.feedback(...) before syn.descend(...)."
            )
        self.tracer.emit(
            TraceEvent(
                "descend", sig.qualname, {"signals": len(signals), "version": champion.version}
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
                    f"Descent step failed shadow validation: {e}", signature=sig
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
            qualname=sig.qualname,
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

        async def infer(instruction: str, data: str = "") -> Any:
            if self._infer_agent is None:
                self._infer_agent = Inferencer(self.model_infer, str)
            self.tracer.emit(TraceEvent("infer", "infer", {"instruction": _truncate(instruction)}))
            out = await self._infer_agent.judge(instruction=instruction, data=data)
            trace_frames.append(
                TraceFrame(
                    kind="infer",
                    name="infer",
                    args_summary=_truncate(instruction),
                    result_summary=_truncate(out),
                )
            )
            return out

        functions: dict[str, Callable[..., Awaitable[Any]]] = {
            t.name: wrap_mcp(t) for t in surface.tools if t.source == "mcp"
        }
        functions["infer"] = infer
        return functions

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
                    self._surface = empty_surface()
        return self._surface

    def _surface_hash_now(self) -> str:
        """Surface hash for sync introspection paths (champion, lineage, feedback)."""
        if self._surface is not None:
            return self._surface.surface_hash
        if not self._mcp_clients:
            self._surface = empty_surface()
            return self._surface.surface_hash
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._ensure_surface()).surface_hash
        raise FrameworkError(
            "Tool surface not built yet and we're inside an event loop — "
            "call the function once (or await it) before introspecting it."
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


__all__ = ["Backend"]
