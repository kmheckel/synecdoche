"""The compiler: prompts, structured output schema, and model invocation.

There is exactly one code-producing entrypoint, ``Compiler.vary``. It takes a
``Variation`` — signature, tool surface, parents, signals — and emits a
``GeneratedBody``. Spawning, repairing, gradient steps, and crossover are all
the same call with different parents/signals; the prompt template renders
whichever sections the variation carries.

We use ``pydantic_ai.Agent`` as an internal implementation detail — it gives
us provider-agnostic structured output without reimplementing the
tool-emulation fallback for providers that lack native JSON-schema output.
Agent never appears in our public API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Template
from pydantic import BaseModel, Field
from pydantic_ai import Agent

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from .evolution import Variation


class GeneratedBody(BaseModel):
    """Structured output of the compiler: the genome of one variant."""

    reasoning: str = Field(description="One paragraph explaining the approach. Archived for audit.")
    imports: list[str] = Field(
        default_factory=list,
        description="Python import statements, one per line, from the sandbox allow-list only.",
    )
    body: str = Field(
        description=(
            "Source of a single `async def solve(...)` function matching the target "
            "signature, plus zero or more inline helper definitions called by solve."
        )
    )


_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"
_USER_PROMPT_PATH = Path(__file__).parent / "prompts" / "user.md.jinja"
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text()
_USER_TEMPLATE = Template(_USER_PROMPT_PATH.read_text())


def render_user_prompt(variation: Variation) -> str:
    """Render the call-specific user prompt from a Variation."""
    return _USER_TEMPLATE.render(
        operator=variation.operator,
        signature_line=variation.signature.render(),
        docstring=variation.signature.docstring,
        return_schema_json=json.dumps(variation.signature.return_schema, indent=2),
        tools_descriptions=variation.surface.render_descriptions(),
        inputs_json=_safe_inputs_json(variation.inputs),
        parents=[
            {"index": i + 1, "reasoning": p.reasoning, "body": p.body}
            for i, p in enumerate(variation.parents)
        ],
        signals=[
            {
                "kind": s.kind,
                "content": s.content,
                "weight": s.weight,
                "data_json": json.dumps(s.data, default=str),
            }
            for s in variation.signals
        ],
        neighbors=[{"reasoning": b.reasoning, "body": b.body} for b in variation.neighbors],
    )


def _safe_inputs_json(inputs: dict[str, Any]) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump()
        return repr(o)

    return json.dumps(inputs, indent=2, default=default)


class Compiler:
    """Wraps a pydantic-ai Agent for emitting ``GeneratedBody``."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self._agent: Agent[None, GeneratedBody] = Agent(
            model,
            output_type=GeneratedBody,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def vary(self, variation: Variation) -> GeneratedBody:
        """Apply a variation operator: spawn, mutate, or cross, per the parents given."""
        prompt = render_user_prompt(variation)
        result = await self._agent.run(prompt)
        return result.output


class Inferencer:
    """Backs the `infer` builtin available inside generated bodies: one
    judgment call per use, nothing archived."""

    def __init__(self, model: Model, output_type: Any = str) -> None:
        self._model = model
        self._agent: Agent[None, Any] = Agent(
            model,
            output_type=output_type,
            system_prompt=_INFER_SYSTEM_PROMPT,
        )

    async def judge(self, *, instruction: str, data: str = "") -> Any:
        prompt = (
            f"# Instruction\n\n{instruction}\n\n"
            f"# Data\n\n{data}\n\n"
            f"Answer with the result only — no preamble, no explanation."
        )
        result = await self._agent.run(prompt)
        return result.output


_INFER_SYSTEM_PROMPT = (
    "You are the `infer` primitive inside a running program: one judgment "
    "call, invoked from generated code. The instruction says what to decide; "
    "the data is what to decide about. Answer with the result only — the "
    "caller is code, not a person, and will parse your reply verbatim."
)


__all__ = ["Compiler", "GeneratedBody", "Inferencer", "render_user_prompt"]
