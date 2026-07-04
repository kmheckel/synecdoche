"""Symbolic regression: evolve a closed-form expression to fit data.

Same loop as heuristic discovery, different hypothesis space: candidate
programs are closed-form expressions, the evaluator is goodness-of-fit,
and selection keeps whichever expression explains the data best. The
result is an interpretable formula, not a curve of weights.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/symbolic_regression.py
"""

from __future__ import annotations

import math
import random

from pydantic_ai.models.anthropic import AnthropicModel

import synecdoche as syn

syn.configure(model=AnthropicModel("claude-sonnet-4-6"), trace="stdout")


# Hidden ground truth (pretend these observations came from the lab).
def _ground_truth(x: float) -> float:
    return 2.5 * x * math.sin(x) + 0.5 * x


_rng = random.Random(3)
OBSERVATIONS = {
    round(x, 3): _ground_truth(x) + _rng.gauss(0.0, 0.05) for x in [i * 0.4 for i in range(1, 21)]
}


@syn
def model(x: float) -> float:
    """A candidate closed-form expression for an observed physical process.

    Keep it simple and interpretable: polynomials, trig, exp, log, and
    their compositions. This function is scored on how well it fits the
    observations, so prefer structure over table-lookup tricks.
    """


def score(inputs: dict, y_hat: float) -> float:
    y = OBSERVATIONS[round(inputs["x"], 3)]
    return 1.0 / (1.0 + (y_hat - y) ** 2)


if __name__ == "__main__":
    examples = [{"x": x} for x in OBSERVATIONS]
    report = syn.evolve(model, examples, generations=3, population=4, score=score)
    print(f"champion v{report.champion.version}, fitness {report.champion.fitness:.3f}")
    for v, fit in report.evaluated:
        print(f"  v{v.version:>2} {v.operator:<6} fitness {fit:.3f}")
    print("\n" + syn.solidify(model))
