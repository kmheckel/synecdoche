"""The whole surface in one file: @syn, healing, infer, vmap, descend, solidify.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/transforms_tour.py
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

import synecdoche as syn

syn.configure(model=AnthropicModel("claude-sonnet-4-6"), trace="stdout")


# Your code, decorated: runs natively as written, and when it raises, the
# exception becomes the compile context for a fixed replacement.


@syn
def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into (major, minor, patch)."""
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)  # crashes on "2.0" or "v1.2.3"


# No body: generated from the spec. The body may use the built-in
# `infer` primitive for judgment calls (here: is a line's meter right?).


class Haiku(BaseModel):
    lines: list[str]
    syllables: list[int]


@syn
def haiku(theme: str) -> Haiku:
    """Compose a haiku about the theme with a strict 5-7-5 syllable count."""


if __name__ == "__main__":
    print(parse_semver("1.2.3"))  # native execution — no model involved
    print(parse_semver("v2.0"))  # raises -> regenerated -> healed descendant
    print([v.operator for v in syn.lineage(parse_semver)])  # ['seed', 'mutate']

    # vmap: concurrent batching over the leading argument.
    for poem in syn.vmap(haiku, concurrency=2)(["gradient descent", "the sea"]):
        print("\n".join(poem.lines), "\n")

    # A critique in, a revised program out.
    syn.feedback(haiku, 0.3, "the middle line keeps landing on 8 syllables")
    syn.descend(haiku)

    # The learned artifact is just Python — freeze it and commit it.
    print(syn.solidify(parse_semver))
