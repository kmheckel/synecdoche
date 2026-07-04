from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

import synecdoche as syn


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


def test_oracle_returns_typed_model(stub_model) -> None:
    stub_model.push({"label": "positive", "confidence": 0.9})
    be = syn.Backend(model=stub_model.as_model())

    @syn.oracle(backend=be)
    def classify(text: str) -> Sentiment:
        """Classify sentiment."""

    result = classify("I love this")
    assert isinstance(result, Sentiment)
    assert result.label == "positive"
    assert result.confidence == 0.9


def test_oracle_per_fn_model_override(stub_model) -> None:
    primary = stub_model.as_model()
    other_stub = type(stub_model)()  # fresh StubModel
    other_stub.push({"label": "negative", "confidence": 0.1})
    other = other_stub.as_model()

    be = syn.Backend(model=primary)

    @syn.oracle(backend=be, model=other)
    def classify(text: str) -> Sentiment:
        """."""

    result = classify("meh")
    assert result.label == "negative"


def test_oracle_has_no_compiled_body_to_introspect(stub_model) -> None:
    be = syn.Backend(model=stub_model.as_model())

    @syn.oracle(backend=be)
    def classify(text: str) -> Sentiment:
        """."""

    with pytest.raises(syn.FrameworkError):
        syn.lineage(classify)
    with pytest.raises(syn.FrameworkError):
        syn.feedback(classify, 0.5)


def test_transforms_reject_untransformed_functions() -> None:
    def plain(x: int) -> int:
        return x

    with pytest.raises(syn.FrameworkError):
        syn.lineage(plain)
