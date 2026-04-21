"""The Runtime: the orchestrator a user interacts with.

Responsibilities:
- Hold model / MCP / archive / sandbox configuration.
- Provide the `@rt.infer` and `@rt.recursion` decorators.
- Dispatch each call through the right pipeline, handling caching, repair,
  and trace emission.

Decorated callables are sync by default; they detect a running event loop
and use `await`able returns when called from async code.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import TypeAdapter

from .archive import (
    Archive,
    ArchiveEntry,
    now_utc,
    open_archive,
)
from .compiler import Compiler, Inferencer
from .exceptions import (
    BudgetExceeded,
    CompilationError,
    RepairAttempt,
    ValidationError,
)
from .jit import Budget, JITContext, TraceFrame
from .repair import RepairInput, build_repair_ctx
from .sandbox import MontySandbox, Sandbox
from .signature import CallSignature
from .surface import ToolSpec, ToolSurface, build_mcp_surface
from .trace import TraceEvent, Tracer, resolve_tracer

if TYPE_CHECKING:
    from fastmcp import Client
    from pydantic_ai.models import Model

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class Runtime:
    """The top-level object: owns all configuration and hosts decorators."""

    def __init__(
        self,
        *,
        # Models — either a single one or split by role.
        model: Model | None = None,
        model_recursion: Model | None = None,
        model_infer: Model | None = None,
        # External capability surface.
        mcp: list[Client] | None = None,
        # Storage.
        archive: str | Path | Archive | None = None,
        # Sandbox.
        sandbox: Sandbox | None = None,
        # Repair policy.
        heal: bool = True,
        max_repair_attempts: int = 2,
        shadow_validate: bool = True,
        # Budgets.
        max_recursion_depth: int = 6,
        max_tool_calls_per_frame: int = 200,
        # Observability.
        trace: Any = None,
    ) -> None:
        if model is None and (model_recursion is None or model_infer is None):
            raise ValueError(
                "Provide either `model=...` or both `model_recursion=` and `model_infer=`."
            )
        self.model_recursion: Model = model_recursion or model  # type: ignore[assignment]
        self.model_infer: Model = model_infer or model  # type: ignore[assignment]

        self._mcp_clients: list[Client] = list(mcp or [])
        self.archive: Archive = open_archive(archive)
        self.sandbox: Sandbox = sandbox or MontySandbox()

        self.heal = heal
        self.max_repair_attempts = max_repair_attempts
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
        self._compiler = Compiler(self.model_recursion)

        # Inferencer agents keyed by return-type, lazy init.
        self._infer_cache: dict[Any, Inferencer] = {}

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    @overload
    def infer(self, fn: F) -> F: ...
    @overload
    def infer(self, *, model: Model | None = None) -> Callable[[F], F]: ...
    def infer(
        self,
        fn: F | None = None,
        *,
        model: Model | None = None,
    ) -> F | Callable[[F], F]:
        """Decorator: mark a function as resolved by a single typed inference."""

        def wrap(fn: F) -> F:
            sig = CallSignature.from_function(fn)

            async def invoke_async(*args: Any, **kwargs: Any) -> Any:
                inputs = _bind_inputs(fn, args, kwargs)
                return await self._call_infer(sig, inputs, model_override=model)

            return _wrap_callable(fn, invoke_async)  # type: ignore[return-value]

        if fn is not None:
            return wrap(fn)  # type: ignore[return-value]
        return wrap

    @overload
    def recursion(self, fn: F) -> F: ...
    @overload
    def recursion(
        self,
        *,
        model: Model | None = None,
        max_repair_attempts: int | None = None,
        archive_key: str | None = None,
    ) -> Callable[[F], F]: ...
    def recursion(
        self,
        fn: F | None = None,
        *,
        model: Model | None = None,
        max_repair_attempts: int | None = None,
        archive_key: str | None = None,
    ) -> F | Callable[[F], F]:
        """Decorator: mark a function whose body is compiled by an LLM at first call."""

        def wrap(fn: F) -> F:
            sig = CallSignature.from_function(fn)
            overrides = _Overrides(
                model=model,
                max_repair_attempts=max_repair_attempts,
                archive_key=archive_key,
            )

            async def invoke_async(*args: Any, **kwargs: Any) -> Any:
                inputs = _bind_inputs(fn, args, kwargs)
                return await self._call_recursion(sig, inputs, overrides=overrides, depth=0)

            return _wrap_callable(fn, invoke_async)  # type: ignore[return-value]

        if fn is not None:
            return wrap(fn)  # type: ignore[return-value]
        return wrap

    # ------------------------------------------------------------------
    # Infer pipeline
    # ------------------------------------------------------------------

    async def _call_infer(
        self,
        sig: CallSignature,
        inputs: dict[str, Any],
        *,
        model_override: Model | None,
    ) -> Any:
        self.tracer.emit(TraceEvent("call_start", sig.qualname, {"mode": "infer"}))
        model = model_override or self.model_infer
        key = (sig.return_type, id(model))
        inferencer = self._infer_cache.get(key)
        if inferencer is None:
            inferencer = Inferencer(model, sig.return_type)
            self._infer_cache[key] = inferencer
        try:
            value = await inferencer.infer(
                signature_line=sig.render(),
                docstring=sig.docstring,
                inputs=inputs,
            )
        except Exception as e:
            self.tracer.emit(
                TraceEvent("call_end", sig.qualname, {"mode": "infer", "error": type(e).__name__})
            )
            raise
        self.tracer.emit(TraceEvent("call_end", sig.qualname, {"mode": "infer"}))
        return _validate_return(sig, value)

    # ------------------------------------------------------------------
    # Recursion pipeline
    # ------------------------------------------------------------------

    async def _call_recursion(
        self,
        sig: CallSignature,
        inputs: dict[str, Any],
        *,
        overrides: _Overrides,
        depth: int,
    ) -> Any:
        if depth > self.budget.max_recursion_depth:
            raise BudgetExceeded(
                kind="depth", limit=self.budget.max_recursion_depth, measured=depth
            )

        surface = await self._ensure_surface()
        self.tracer.emit(
            TraceEvent("call_start", sig.qualname, {"mode": "recursion", "depth": depth})
        )

        # archive_key override lets users pin the archive identity independently of
        # the signature hash — useful for migrations (same function, different body
        # history) and branching.
        archive_key = overrides.archive_key or sig.signature_hash

        # Archive lookup — skip compilation on hit.
        entry = self.archive.get_current(archive_key, surface.tool_surface_hash)
        if entry is None:
            entry = await self._compile_and_archive(
                sig, surface, inputs, overrides, archive_key=archive_key
            )
            self.tracer.emit(
                TraceEvent(
                    "compile", sig.qualname, {"version": entry.version, "trigger": entry.trigger}
                )
            )
        else:
            self.tracer.emit(TraceEvent("cache_hit", sig.qualname, {"version": entry.version}))

        max_repairs = (
            overrides.max_repair_attempts
            if overrides.max_repair_attempts is not None
            else self.max_repair_attempts
        )
        attempts: list[RepairAttempt] = []
        current_entry = entry

        for attempt in range(max_repairs + 1):
            t0 = time.perf_counter()
            trace_frames: list[TraceFrame] = []
            external_fns = self._build_external_functions(
                surface, trace_frames, depth=depth, overrides=overrides
            )
            try:
                raw = await self.sandbox.execute(
                    body=current_entry.body,
                    signature=sig,
                    surface=surface,
                    inputs=inputs,
                    external_functions=external_fns,
                )
                value = _validate_return(sig, raw)
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                self.archive.record_metrics(
                    archive_key,
                    surface.tool_surface_hash,
                    current_entry.version,
                    success=False,
                    latency_ms=elapsed,
                    validation_failure=isinstance(e, ValidationError),
                )
                attempts.append(
                    RepairAttempt(
                        version=current_entry.version,
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                        body_excerpt=current_entry.body.body[:200],
                    )
                )
                self.tracer.emit(
                    TraceEvent(
                        "call_end",
                        sig.qualname,
                        {
                            "mode": "recursion",
                            "ok": False,
                            "error": type(e).__name__,
                            "attempt": attempt,
                        },
                    )
                )
                if not self.heal or attempt >= max_repairs:
                    raise CompilationError(
                        f"Recursion failed after {attempt + 1} attempt(s): {e}",
                        signature=sig,
                        attempts=attempts,
                    ) from e
                # Build a repair context and compile a revised body.
                self.tracer.emit(TraceEvent("repair", sig.qualname, {"attempt": attempt + 1}))
                current_entry = await self._repair(
                    sig=sig,
                    surface=surface,
                    inputs=inputs,
                    prior_entry=current_entry,
                    exception=e,
                    trace_frames=trace_frames,
                    prior_attempts=attempt,
                    archive_key=archive_key,
                )
                continue
            else:
                elapsed = (time.perf_counter() - t0) * 1000
                self.archive.record_metrics(
                    archive_key,
                    surface.tool_surface_hash,
                    current_entry.version,
                    success=True,
                    latency_ms=elapsed,
                )
                self.tracer.emit(
                    TraceEvent("call_end", sig.qualname, {"mode": "recursion", "ok": True})
                )
                return value

        raise CompilationError(
            "Exhausted repair attempts without exception — should be unreachable.",
            signature=sig,
            attempts=attempts,
        )

    async def _compile_and_archive(
        self,
        sig: CallSignature,
        surface: ToolSurface,
        inputs: dict[str, Any],
        overrides: _Overrides,
        archive_key: str,
    ) -> ArchiveEntry:
        compiler = Compiler(overrides.model) if overrides.model is not None else self._compiler
        neighbors = tuple(e.body for e in self.archive.neighbors(archive_key, k=3))
        ctx = JITContext(
            signature=sig,
            surface=surface,
            inputs=inputs,
            archive_neighbors=neighbors,
            remaining_depth=self.budget.max_recursion_depth,
            budget=self.budget,
        )
        body = await compiler.compile(ctx)
        version = self.archive.next_version(archive_key, surface.tool_surface_hash)
        entry = ArchiveEntry(
            signature_hash=archive_key,
            tool_surface_hash=surface.tool_surface_hash,
            version=version,
            parent_version=None,
            body=body,
            created_at=now_utc(),
            trigger="initial",
            promoted=True,
        )
        self.archive.insert(entry)
        self.archive.promote(archive_key, surface.tool_surface_hash, version)
        return entry

    async def _repair(
        self,
        *,
        sig: CallSignature,
        surface: ToolSurface,
        inputs: dict[str, Any],
        prior_entry: ArchiveEntry,
        exception: BaseException,
        trace_frames: list[TraceFrame],
        prior_attempts: int,
        archive_key: str,
    ) -> ArchiveEntry:
        repair_input = RepairInput(
            prior_body=prior_entry.body,
            exception=exception,
            trace_frames=trace_frames,
            attempts_so_far=prior_attempts,
        )
        base_ctx = JITContext(
            signature=sig,
            surface=surface,
            inputs=inputs,
            archive_neighbors=tuple(e.body for e in self.archive.neighbors(archive_key, k=3)),
            remaining_depth=self.budget.max_recursion_depth,
            budget=self.budget,
        )
        repaired_ctx = build_repair_ctx(base_ctx, repair_input)
        revised_body = await self._compiler.compile(repaired_ctx)

        # Shadow validation: re-run against the failing inputs (at minimum).
        if self.shadow_validate:
            try:
                external_fns = self._build_external_functions(
                    surface, [], depth=0, overrides=_Overrides()
                )
                raw = await self.sandbox.execute(
                    body=revised_body,
                    signature=sig,
                    surface=surface,
                    inputs=inputs,
                    external_functions=external_fns,
                )
                _validate_return(sig, raw)
            except Exception as e:
                raise CompilationError(
                    f"Revised body failed shadow validation: {e}",
                    signature=sig,
                ) from e

        version = self.archive.next_version(archive_key, surface.tool_surface_hash)
        entry = ArchiveEntry(
            signature_hash=archive_key,
            tool_surface_hash=surface.tool_surface_hash,
            version=version,
            parent_version=prior_entry.version,
            body=revised_body,
            created_at=now_utc(),
            trigger="repair",
            promoted=True,
        )
        self.archive.insert(entry)
        self.archive.promote(archive_key, surface.tool_surface_hash, version)
        return entry

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _build_external_functions(
        self,
        surface: ToolSurface,
        trace_frames: list[TraceFrame],
        *,
        depth: int,
        overrides: _Overrides,
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

        functions: dict[str, Callable[..., Awaitable[Any]]] = {
            t.name: wrap_mcp(t) for t in surface.tools if t.source == "mcp"
        }
        # Inline helpers added after initial compile, if any body declared them.
        # (Rare in POC; usually resolved via the build-time surface expansion
        # done by the recursion pipeline when it processes body.helpers.)
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
                    self._surface = ToolSurface(tools=(), tool_surface_hash="empty")
        return self._surface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Overrides:
    def __init__(
        self,
        *,
        model: Model | None = None,
        max_repair_attempts: int | None = None,
        archive_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_repair_attempts = max_repair_attempts
        self.archive_key = archive_key


class _Counter:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0

    def inc(self) -> None:
        self.value += 1


def _bind_inputs(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _wrap_callable(
    original: Callable[..., Any], invoke_async: Callable[..., Awaitable[Any]]
) -> Callable[..., Any]:
    """Create a sync-friendly wrapper that also works inside async code.

    The wrapper returns a value when called from a sync context, and an
    awaitable when called from an async context (via loop detection).
    """
    is_source_async = inspect.iscoroutinefunction(original)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — we can block with asyncio.run.
            return asyncio.run(invoke_async(*args, **kwargs))
        # Running loop exists — return a coroutine for the caller to await.
        return invoke_async(*args, **kwargs)

    # If the original was defined `async def`, preserve that marker so callers
    # that use `inspect.iscoroutinefunction` still see it as such.
    if is_source_async:
        wrapper = functools.wraps(original)(invoke_async)  # type: ignore[assignment]

    return wrapper


def _validate_return(sig: CallSignature, value: Any) -> Any:
    if sig.return_type is Any or sig.return_type is inspect.Parameter.empty:
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
    """Normalize FastMCP's CallToolResult (or similar) into a Python value.

    FastMCP >= 3 returns an object with some subset of:
    - `.structured_content` — a typed dict the server returned (most preferred).
    - `.data` — server-provided payload (may be the same as structured_content
      on some versions; empty list when the server returned text content only).
    - `.content` — a list of typed content parts (TextContent, ImageContent, ...).

    Precedence order:
      1. `structured_content` (most explicit, server-declared shape)
      2. `data` — only when it is a non-empty, non-list-of-empty value
      3. `content` — flatten any TextContent into a newline-joined string
      4. fall back to the raw object
    """
    sc = getattr(result, "structured_content", None)
    if sc is not None and sc != {} and sc != []:
        return sc

    data = getattr(result, "data", None)
    if data is not None and _is_meaningful(data):
        return data

    content = getattr(result, "content", None)
    if content is not None:
        if isinstance(content, list):
            texts = [getattr(c, "text", None) for c in content]
            texts = [t for t in texts if t is not None]
            if texts:
                return "\n".join(texts)
            # list of non-text parts — return verbatim so caller can inspect
            if content:
                return content
        else:
            return content

    return result


def _is_meaningful(v: Any) -> bool:
    """True if `v` is not None / empty dict / empty list."""
    if v is None:
        return False
    if isinstance(v, (list, dict, str)) and not v:
        return False
    return True


def _truncate(v: Any, n: int = 200) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


__all__ = ["Runtime"]
