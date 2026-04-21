"""Shared test fixtures — a stub pydantic-ai model that emits canned outputs.

We build a minimal `FunctionModel` that replies with the first output tool
call, filled in from a programmable queue. This lets us exercise the
Runtime end-to-end without touching a real provider.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


class StubModel:
    """Queue-based stub of a pydantic-ai Model.

    `push(value)` adds a typed value (a dict matching the expected structured
    output) to the reply queue. The next Agent.run call will return it.
    """

    def __init__(self) -> None:
        self._queue: list[Any] = []

    def push(self, value: Any) -> None:
        self._queue.append(value)

    def extend(self, values: Iterable[Any]) -> None:
        self._queue.extend(values)

    def as_model(self) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not self._queue:
                raise RuntimeError("StubModel: no queued responses; test forgot to push()")
            value = self._queue.pop(0)
            if not info.output_tools:
                raise RuntimeError("StubModel expects an Agent configured with output_type")
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args=json.dumps(value))])

        return FunctionModel(fn, model_name="stub")


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()
