"""The whole surface in one file: jit, oracle, vmap, feedback/descend, solidify.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/transforms_tour.py
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

import synecdoche as syn

syn.configure(model=AnthropicModel("claude-sonnet-4-6"), trace="stdout")


# jit over a handwritten body: runs natively, and when it raises, the
# exception becomes the compile context for a fixed replacement.


@syn.jit
def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into (major, minor, patch)."""
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)  # crashes on "2.0" or "v1.2.3"


# jit over an empty body: compiled from the signature + docstring alone.


class Haiku(BaseModel):
    lines: list[str]
    syllables: list[int]


@syn.jit
def haiku(theme: str) -> Haiku:
    """Compose a haiku about the theme with a strict 5-7-5 syllable count."""


# oracle: the model as the implementation.


@syn.oracle
def mood(text: str) -> Literal["calm", "stormy", "electric"]:
    """Judge the mood of the text."""


if __name__ == "__main__":
    print(parse_semver("1.2.3"))  # native execution — no model involved
    print(parse_semver("v2.0"))  # raises -> recompiled -> healed descendant
    print([v.operator for v in syn.lineage(parse_semver)])  # ['seed', 'mutate']

    poem = haiku("gradient descent")
    print("\n".join(poem.lines))

    print(syn.vmap(mood)([" ".join(poem.lines), "the sea is glass today"]))

    # The gradient surrogate: a critique in, a revised program out.
    syn.feedback(haiku, 0.3, "the middle line keeps landing on 8 syllables")
    syn.descend(haiku)

    # The compiled artifact is just Python — freeze it and commit it.
    print(syn.solidify(parse_semver))
