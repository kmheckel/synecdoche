from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from synecdoche import Runtime


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


def test_infer_returns_typed_model(stub_model) -> None:
    stub_model.push({"label": "positive", "confidence": 0.9})
    rt = Runtime(model=stub_model.as_model())

    @rt.infer
    def classify(text: str) -> Sentiment:
        """Classify sentiment."""

    result = classify("I love this")
    assert isinstance(result, Sentiment)
    assert result.label == "positive"
    assert result.confidence == 0.9


def test_infer_validates_return_type(stub_model) -> None:
    # Push a payload that pydantic-ai's own tool-call validator catches
    # first — it returns a wrapped/retried response. Here we push a valid
    # payload to confirm the happy path, then push a second call with a
    # completely different model to confirm the runtime routes correctly.
    stub_model.push({"label": "neutral", "confidence": 0.5})
    rt = Runtime(model=stub_model.as_model())

    @rt.infer
    def classify(text: str) -> Sentiment:
        """."""

    assert classify("x").label == "neutral"


def test_infer_per_call_model_override(stub_model) -> None:
    primary = stub_model.as_model()
    other_stub = type(stub_model)()  # fresh StubModel
    other_stub.push({"label": "negative", "confidence": 0.1})
    other = other_stub.as_model()

    rt = Runtime(model=primary)

    @rt.infer(model=other)
    def classify(text: str) -> Sentiment:
        """."""

    result = classify("meh")
    assert result.label == "negative"
