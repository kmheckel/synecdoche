from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synecdoche.runtime import _unwrap_mcp_result


@dataclass
class FakeTextPart:
    text: str


@dataclass
class FakeImagePart:
    mime: str = "image/png"


@dataclass
class FakeResult:
    structured_content: Any = None
    data: Any = None
    content: Any = field(default_factory=list)


def test_structured_content_wins() -> None:
    r = FakeResult(structured_content={"a": 1}, data=[], content=[FakeTextPart("ignored")])
    assert _unwrap_mcp_result(r) == {"a": 1}


def test_data_preferred_over_content_when_meaningful() -> None:
    r = FakeResult(data={"k": 1}, content=[FakeTextPart("x")])
    assert _unwrap_mcp_result(r) == {"k": 1}


def test_empty_data_falls_through_to_content() -> None:
    r = FakeResult(data=[], content=[FakeTextPart("hello"), FakeTextPart("world")])
    assert _unwrap_mcp_result(r) == "hello\nworld"


def test_empty_dict_structured_content_falls_through() -> None:
    r = FakeResult(structured_content={}, data={"k": 1})
    assert _unwrap_mcp_result(r) == {"k": 1}


def test_non_text_content_parts_returned_verbatim() -> None:
    parts = [FakeImagePart()]
    r = FakeResult(content=parts)
    assert _unwrap_mcp_result(r) is parts


def test_fallback_to_raw_object() -> None:
    r = FakeResult()  # everything empty
    assert _unwrap_mcp_result(r) is r


def test_plain_value_passthrough() -> None:
    assert _unwrap_mcp_result("just a string") == "just a string"
    assert _unwrap_mcp_result(42) == 42
