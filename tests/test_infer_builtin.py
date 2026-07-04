"""The `infer` builtin: the one nesting primitive generated code gets."""

from __future__ import annotations

import pytest

import synecdoche as syn


def test_generated_code_can_call_infer(stub_model) -> None:
    body = (
        "async def solve(text: str) -> str:\n"
        "    label = await infer(instruction='classify sentiment as positive, "
        "negative, or neutral', data=text)\n"
        "    return label.strip()\n"
    )
    stub_model.push({"reasoning": "judge via infer", "imports": [], "body": body})
    stub_model.push("positive")  # the infer builtin's plain-text reply
    be = syn.Backend(model=stub_model.as_model())

    @syn.fn(backend=be)
    def classify(text: str) -> str:
        """Classify sentiment."""

    assert classify("I love this") == "positive"


def test_infer_spec_is_always_on_the_surface(stub_model) -> None:
    from synecdoche.surface import empty_surface

    surface = empty_surface()
    assert surface.by_name("infer") is not None
    assert surface.surface_hash == "empty"  # builtins don't perturb the cache key
    stub = surface.render_stubs()
    assert "async def infer(" in stub


def test_transforms_reject_undecorated_functions() -> None:
    def plain(x: int) -> int:
        return x

    with pytest.raises(syn.FrameworkError):
        syn.lineage(plain)
