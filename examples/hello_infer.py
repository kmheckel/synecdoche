"""Minimal @rt.infer example — one typed function, no MCP needed.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/hello_infer.py

This exercises the Runtime's inference path only: the model emits a
structured value that satisfies the declared return type.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

from synecdoche import Runtime


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrase: str


rt = Runtime(
    model=AnthropicModel("claude-haiku-4-5"),
    trace="stdout",
)


@rt.infer
def classify_sentiment(text: str) -> Sentiment:
    """Classify the sentiment of the given text as positive, negative, or neutral.

    Include a 0..1 confidence and the most informative phrase from the input.
    """


if __name__ == "__main__":
    for text in [
        "This library is exactly what I needed — love how small the surface is.",
        "Ugh, the install instructions are broken and nothing works.",
        "It was fine. Does what it says.",
    ]:
        result = classify_sentiment(text)
        print(f"{result.label:>8}  conf={result.confidence:.2f}  '{result.key_phrase}'")
