"""Oracle + vmap — the model as a black-box function, batched.

Run:

    ANTHROPIC_API_KEY=sk-... uv run python examples/oracle_and_vmap.py

An oracle is the simplest transform: one typed inference per call,
validated against the return type. vmap maps it over a batch — the calls
run concurrently on the event loop.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

import synecdoche as syn

syn.configure(model=AnthropicModel("claude-haiku-4-5"), trace="stdout")


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrase: str


@syn.oracle
def classify_sentiment(text: str) -> Sentiment:
    """Classify the sentiment of the given text as positive, negative, or neutral.

    Include a 0..1 confidence and the most informative phrase from the input.
    """


if __name__ == "__main__":
    texts = [
        "This library is exactly what I needed — love how small the surface is.",
        "Ugh, the install instructions are broken and nothing works.",
        "It was fine. Does what it says.",
    ]
    for result in syn.vmap(classify_sentiment, concurrency=3)(texts):
        print(f"{result.label:>8}  conf={result.confidence:.2f}  '{result.key_phrase}'")
