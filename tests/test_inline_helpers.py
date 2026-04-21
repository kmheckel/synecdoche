"""End-to-end test of inline helper dispatch.

We push a GeneratedBody that declares an `@ai.infer` helper named
`classify_letter`, then calls it from the body. The runtime must register a
dispatcher that routes the helper call to an Inferencer, and Monty must
typecheck the body against the helper stub.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from synecdoche import Runtime
from synecdoche.compiler import InlineHelper
from synecdoche.helpers import parse_helper


class Kind(BaseModel):
    category: Literal["vowel", "consonant"]


class Out(BaseModel):
    summary: str


def test_parse_helper_with_builtin_types() -> None:
    helper = InlineHelper(
        kind="infer",
        name="add",
        signature="(a: int, b: int) -> int",
        docstring="Add two integers.",
    )
    spec = parse_helper(helper, caller_qualname="outer")
    assert spec.kind == "infer"
    assert spec.name == "add"
    assert spec.signature.return_type is int
    assert [p.name for p in spec.signature.params] == ["a", "b"]


def test_parse_helper_with_globals_resolution() -> None:
    helper = InlineHelper(
        kind="recursion",
        name="classify",
        signature="(letter: str) -> Kind",
        docstring="Classify a letter.",
    )
    # Pretend Kind is in the caller's globals.
    spec = parse_helper(helper, caller_qualname="outer", globals_dict={"Kind": Kind})
    assert spec.signature.return_type is Kind


def test_inline_infer_helper_dispatches_to_inferencer(stub_model) -> None:
    """A body that declares an `@ai.infer` helper and calls it should route
    the helper call to an Inferencer and get back a typed value.

    Uses a helper with a plain-int return type so the sandbox body can use
    the value directly without going through BaseModel attribute access
    (Monty's sandbox is restrictive about that for POC purposes)."""
    body_src = (
        "async def solve(letter: str) -> dict:\n"
        "    n = await score_letter(letter=letter)\n"
        "    return {'summary': letter + ':' + str(n)}\n"
    )
    stub_model.extend(
        [
            {
                "reasoning": "r",
                "helpers": [
                    {
                        "kind": "infer",
                        "name": "score_letter",
                        "signature": "(letter: str) -> int",
                        "docstring": "Return a numeric score for the letter.",
                    }
                ],
                "imports": [],
                "body": body_src,
            },
            # pydantic-ai wraps a non-BaseModel output in {"response": <value>}.
            {"response": 42},
        ]
    )

    rt = Runtime(model=stub_model.as_model(), heal=False)

    @rt.recursion
    def analyze(letter: str) -> Out:
        """Analyze one letter by delegating to an inline helper."""

    result = analyze("a")
    assert isinstance(result, Out)
    assert result.summary == "a:42"
