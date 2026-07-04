from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from synecdoche import FrameworkError, Runtime


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


def test_oracle_returns_typed_model(stub_model) -> None:
    stub_model.push({"label": "positive", "confidence": 0.9})
    rt = Runtime(model=stub_model.as_model())

    @rt.fn(mode="oracle")
    def classify(text: str) -> Sentiment:
        """Classify sentiment."""

    result = classify("I love this")
    assert isinstance(result, Sentiment)
    assert result.label == "positive"
    assert result.confidence == 0.9


def test_infer_alias_still_works(stub_model) -> None:
    stub_model.push({"label": "neutral", "confidence": 0.5})
    rt = Runtime(model=stub_model.as_model())

    @rt.infer
    def classify(text: str) -> Sentiment:
        """."""

    assert classify("x").label == "neutral"


def test_oracle_per_fn_model_override(stub_model) -> None:
    primary = stub_model.as_model()
    other_stub = type(stub_model)()  # fresh StubModel
    other_stub.push({"label": "negative", "confidence": 0.1})
    other = other_stub.as_model()

    rt = Runtime(model=primary)

    @rt.fn(mode="oracle", model=other)
    def classify(text: str) -> Sentiment:
        """."""

    result = classify("meh")
    assert result.label == "negative"


def test_oracle_has_no_lineage(stub_model) -> None:
    rt = Runtime(model=stub_model.as_model())

    @rt.fn(mode="oracle")
    def classify(text: str) -> Sentiment:
        """."""

    with pytest.raises(FrameworkError):
        classify.lineage()
    with pytest.raises(FrameworkError):
        classify.feedback(0.5)
