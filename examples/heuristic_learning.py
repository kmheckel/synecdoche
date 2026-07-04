"""Heuristic learning: discover a bin-packing heuristic under selection.

The FunSearch-shaped experiment, in a few lines: the spec below is a
hypothesis space (programs that pack items into bins), the score function
is the evaluator, and `syn.evolve` runs the loop — generate, measure,
select, repeat. What comes out is not weights but a readable heuristic,
mirrored into `.synecdoche/champions/` as ordinary Python.

See Weng, "Learning Beyond Gradients" (heuristic learning: the RL loop
where the object updated is program structure), and DeepMind's FunSearch,
which discovered bin-packing heuristics exactly this way.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/heuristic_learning.py
"""

from __future__ import annotations

import math
import random

from pydantic_ai.models.anthropic import AnthropicModel

import synecdoche as syn

syn.configure(
    model=AnthropicModel("claude-sonnet-4-6"),
    archive="./.synecdoche",
    trace="stdout",
)


# The spec is the hypothesis space. The handwritten body is generation
# zero — a naive first-fit packer for evolution to beat.
@syn
def pack(items: list[float], capacity: float) -> list[int]:
    """Assign every item to a bin, returning one bin index per item.

    Bin indices start at 0 and must be dense (0, 1, 2, ...). The sum of
    the items in any bin must not exceed `capacity`. Use as few bins as
    possible.
    """
    bins: list[float] = []
    out: list[int] = []
    for item in items:
        for i, used in enumerate(bins):
            if used + item <= capacity:
                bins[i] = used + item
                out.append(i)
                break
        else:
            bins.append(item)
            out.append(len(bins) - 1)
    return out


def score(inputs: dict, assignment: list[int]) -> float:
    """The evaluator: validity is binary, efficiency is graded."""
    items, capacity = inputs["items"], inputs["capacity"]
    if len(assignment) != len(items):
        return 0.0
    fill: dict[int, float] = {}
    for item, b in zip(items, assignment, strict=True):
        fill[b] = fill.get(b, 0.0) + item
        if fill[b] > capacity + 1e-9:
            return 0.0  # overfull bin: invalid, no partial credit
    lower_bound = math.ceil(sum(items) / capacity)
    return lower_bound / len(fill)  # 1.0 = provably optimal bin count


def instances(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    return [
        {"items": [round(rng.uniform(0.1, 0.7), 2) for _ in range(24)], "capacity": 1.0}
        for _ in range(n)
    ]


if __name__ == "__main__":
    train = instances(4)

    baseline = sum(score(ex, pack(**ex)) for ex in train) / len(train)
    print(f"first-fit seed:  mean score {baseline:.3f}")

    report = syn.evolve(pack, train, generations=3, population=4, score=score)
    print(
        f"evolved champion: v{report.champion.version} "
        f"({report.champion.operator}), fitness {report.champion.fitness:.3f}"
    )

    held_out = instances(2, seed=99)
    test = sum(score(ex, pack(**ex)) for ex in held_out) / len(held_out)
    print(f"held-out:        mean score {test:.3f}")

    # The learned heuristic is code you can read, diff, and commit.
    print("\n" + syn.solidify(pack))
    print("# also mirrored at ./.synecdoche/champions/")
