"""Ollama-backed @rt.infer demo — no cloud API key required.

Prerequisites:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen2.5:7b-instruct    # or any model you prefer
    ollama serve                        # if not already running

Then:
    uv run python examples/ollama_infer.py

How it works: pydantic-ai's `OpenAIChatModel` accepts a custom provider whose
`base_url` points at Ollama's OpenAI-compatible endpoint. synecdoche doesn't
care which provider it is — the model abstraction is pydantic-ai's.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from synecdoche import Runtime


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


ollama = OpenAIChatModel(
    "qwen2.5:7b-instruct",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
)

rt = Runtime(model=ollama, trace="tree")


@rt.infer
def classify_sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral, with a 0..1 confidence."""


if __name__ == "__main__":
    for text in [
        "I love this tiny framework.",
        "Absolutely terrible UX.",
        "It's fine, I guess.",
    ]:
        result = classify_sentiment(text)
        print(f"{text!r} → {result.label} (conf={result.confidence:.2f})")
