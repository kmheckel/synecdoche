"""The evolutionary core: signals, variation, fitness, selection.

synecdoche has exactly one way to produce code: **variation**. Every compile
is a variation operator applied to zero or more parents under zero or more
signals:

- ``spawn``  — no parents. The first body for a signature, or a fresh
  exploration during offline evolution.
- ``mutate`` — one parent. "Repair" is mutation under a hard signal (an
  exception); a "gradient step" is mutation under soft signals (feedback).
- ``cross``  — two parents. Recombine the strengths of two archived bodies.

Signals are the selection pressure. A hard signal (exception, validation
failure) says *this body is wrong here, in this structured way*. A soft
signal (user feedback) says *this body is lawful but could be better, in
this direction*. Both flow into the same variation prompt — exceptions are
the sub-gradients of a non-differentiable landscape, feedback is the smooth
part. ``Fn.backward()`` is a textual gradient step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .archive import Variant
    from .compiler import GeneratedBody
    from .signature import CallSignature
    from .surface import ToolSurface

Operator = Literal["seed", "spawn", "mutate", "cross"]
SignalKind = Literal["exception", "validation", "feedback"]


def now_utc() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Budgets and trace frames
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    max_recursion_depth: int = 6
    max_tool_calls_per_frame: int = 200
    wall_time_seconds: float | None = None


@dataclass
class TraceFrame:
    kind: str  # "tool" | "helper"
    name: str
    args_summary: str
    result_summary: str


# ---------------------------------------------------------------------------
# Signals — the selection pressure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One piece of selection pressure against a specific variant.

    ``weight`` lives in [-1, 1]: -1 is a fatal defect (an exception), values
    in (-1, 1) are graded feedback, +1 is confirmation. The compiler sees the
    content; fitness sees the weight.
    """

    kind: SignalKind
    content: str
    weight: float = -1.0
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    @classmethod
    def from_exception(
        cls, exc: BaseException, trace_frames: list[TraceFrame] | None = None
    ) -> Signal:
        """Distill an exception into a hard signal.

        Structured exception attributes are the richest repair information we
        have — a ``BudgetExceeded(kind="tool_calls", limit=200, measured=214)``
        teaches the compiler exactly what to fix.
        """
        data: dict[str, Any] = {"exception_type": type(exc).__name__}
        for k, v in vars(exc).items():
            if k.startswith("_"):
                continue
            data[k] = v if _is_jsonable(v) else repr(v)
        if trace_frames:
            data["trace"] = [
                f"{t.kind}:{t.name}({t.args_summary}) -> {t.result_summary}"
                for t in trace_frames[-8:]
            ]
        return cls(kind="exception", content=f"{type(exc).__name__}: {exc}", data=data)

    @classmethod
    def from_feedback(cls, score: float, note: str = "") -> Signal:
        """Distill graded feedback into a soft signal.

        ``score`` is in [0, 1] (1 = perfect); it is mapped to weight in
        [-1, 1] so that indifferent feedback (0.5) exerts no pressure.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"feedback score must be in [0, 1], got {score}")
        return cls(
            kind="feedback",
            content=note or f"graded {score:.2f}/1.0 with no note",
            weight=2.0 * score - 1.0,
            data={"score": score},
        )


# ---------------------------------------------------------------------------
# Variation — the one compile request
# ---------------------------------------------------------------------------


@dataclass
class Variation:
    """Everything the compiler needs to produce the next body.

    The operator is implied by the shape: 0 parents = spawn, 1 = mutate,
    2 = cross. Signals accumulate against the first parent (the variant
    being varied).
    """

    signature: CallSignature
    surface: ToolSurface
    inputs: dict[str, Any] = field(default_factory=dict)
    parents: tuple[GeneratedBody, ...] = ()
    signals: tuple[Signal, ...] = ()
    neighbors: tuple[GeneratedBody, ...] = ()
    budget: Budget = field(default_factory=Budget)

    @property
    def operator(self) -> Operator:
        return ("spawn", "mutate", "cross")[min(len(self.parents), 2)]


# ---------------------------------------------------------------------------
# Fitness and selection
# ---------------------------------------------------------------------------


def default_fitness(
    *,
    invocations: int,
    successes: int,
    signals: list[Signal] | tuple[Signal, ...] = (),
) -> float:
    """Blend hard evidence (success rate) with soft evidence (feedback).

    Returns a score in [0, 1]. A variant with no history scores 0.5 —
    unknown, not unfit. Feedback shifts the success-rate estimate by up to
    ±0.25 so that soft signals steer selection without overriding crashes.
    """
    if invocations == 0:
        base = 0.5
    else:
        base = successes / invocations
    feedback = [s.weight for s in signals if s.kind == "feedback"]
    if feedback:
        base += 0.25 * (sum(feedback) / len(feedback))
    return max(0.0, min(1.0, base))


@dataclass
class EvolutionReport:
    """Outcome of one offline evolution run."""

    champion: Variant
    evaluated: list[tuple[Variant, float]]
    generations: int


def _is_jsonable(v: Any) -> bool:
    import json

    try:
        json.dumps(v)
        return True
    except Exception:
        return False


__all__ = [
    "Budget",
    "EvolutionReport",
    "Operator",
    "Signal",
    "SignalKind",
    "TraceFrame",
    "Variation",
    "default_fitness",
    "now_utc",
]
