"""The whole spectrum in one file: solid, synth, oracle — and the phase changes.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/phases_of_matter.py

Demonstrates:

1. **solid**  — a handwritten body registered as the seed of a lineage. It
   crashes on malformed input, melts into a sandboxed descendant that
   handles it, and keeps working.
2. **synth**  — a contract with no body, spawned at first call.
3. **oracle** — no code at all; the model answers directly.
4. **gradients** — grade the synth champion with `feedback()`, then take a
   textual gradient step with `backward()`.
5. **solidify** — freeze the surviving champion back into source.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

from synecdoche import Runtime

rt = Runtime(model=AnthropicModel("claude-sonnet-4-6"), trace="stdout")


# 1. Solid — your code, as generation zero of a lineage. -----------------


@rt.fn
def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into (major, minor, patch)."""
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)  # crashes on "2.0" or "v1.2.3"


# 2. Synth — only the contract; the body is grown at first call. ---------


class Haiku(BaseModel):
    lines: list[str]
    syllables: list[int]


@rt.fn
def syllable_frame(theme: str) -> Haiku:
    """Compose a haiku about the theme with a strict 5-7-5 syllable count."""


# 3. Oracle — the model is the body. --------------------------------------


@rt.fn(mode="oracle")
def mood(text: str) -> Literal["calm", "stormy", "electric"]:
    """Judge the mood of the text."""


if __name__ == "__main__":
    print(parse_semver("1.2.3"))  # native execution — no model involved
    print(parse_semver("v2.0"))  # seed raises -> melts into a healed descendant
    print([v.operator for v in parse_semver.lineage()])  # ['seed', 'mutate']

    poem = syllable_frame("gradient descent")
    print("\n".join(poem.lines))

    print(mood("\n".join(poem.lines)))

    # 4. A soft gradient against the haiku champion, then a step.
    syllable_frame.feedback(0.3, "the middle line keeps landing on 8 syllables")
    syllable_frame.backward()

    # 5. Freeze what survived back into committable source.
    print(parse_semver.solidify())
