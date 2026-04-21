"""The compiler: prompts, structured output schema, and model invocation.

We use `pydantic_ai.Agent` as an internal implementation detail — it gives
us provider-agnostic structured output (what the user actually cares about)
without us having to reimplement the tool-emulation fallback for providers
that lack native JSON-schema output. Agent never appears in our public API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from jinja2 import Template
from pydantic import BaseModel, Field
from pydantic_ai import Agent

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from .jit import JITContext


class InlineHelper(BaseModel):
    """A helper function declared inline by a recursion body."""

    kind: Literal["recursion", "infer"]
    name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    signature: str = Field(
        description="Python signature line with types, e.g. `(path: Path, head: str) -> FileKind`"
    )
    docstring: str = Field(default="")


class GeneratedBody(BaseModel):
    """Structured output of the recursion compiler."""

    reasoning: str = Field(description="One paragraph explaining the approach. Archived for audit.")
    helpers: list[InlineHelper] = Field(default_factory=list)
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


def render_user_prompt(ctx: JITContext) -> str:
    """Render the call-specific user prompt from the JIT context."""
    repair = None
    if ctx.repair:
        repair = {
            "prior_body": ctx.repair.prior_body.body,
            "exception_type": ctx.repair.exception_type,
            "exception_message": ctx.repair.exception_message,
            "exception_args": ctx.repair.exception_args,
            "exception_args_json": json.dumps(ctx.repair.exception_args, default=str),
            "failing_line": ctx.repair.failing_line,
            "trace_excerpt": [
                f"{t.kind}:{t.name}({t.args_summary}) -> {t.result_summary}"
                for t in ctx.repair.trace_excerpt
            ],
            "prior_attempts": ctx.repair.prior_attempts,
        }
    return _USER_TEMPLATE.render(
        signature_line=ctx.signature.render(),
        docstring=ctx.signature.docstring,
        return_schema_json=json.dumps(ctx.signature.return_schema, indent=2),
        tools_descriptions=ctx.surface.render_descriptions(),
        inputs_json=_safe_inputs_json(ctx.inputs),
        archive_neighbors=[
            {
                "qualname": "archived",
                "reasoning": b.reasoning,
                "body": b.body,
            }
            for b in ctx.archive_neighbors
        ],
        repair=repair,
    )


def _safe_inputs_json(inputs: dict[str, Any]) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump()
        return repr(o)

    return json.dumps(inputs, indent=2, default=default)


class Compiler:
    """Wraps a pydantic-ai Agent for emitting `GeneratedBody`."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self._agent: Agent[None, GeneratedBody] = Agent(
            model,
            output_type=GeneratedBody,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def compile(self, ctx: JITContext) -> GeneratedBody:
        prompt = render_user_prompt(ctx)
        result = await self._agent.run(prompt)
        return result.output


class Inferencer:
    """Wraps a pydantic-ai Agent to emit a typed value of a given return type."""

    def __init__(self, model: Model, output_type: Any, system_prompt: str | None = None) -> None:
        self._model = model
        self._output_type = output_type
        self._agent: Agent[None, Any] = Agent(
            model,
            output_type=output_type,
            system_prompt=system_prompt or _INFER_SYSTEM_PROMPT,
        )

    async def infer(self, *, signature_line: str, docstring: str, inputs: dict[str, Any]) -> Any:
        prompt = (
            f"# Target signature\n\n"
            f"```python\n{signature_line}\n"
            f'    """{docstring}"""\n'
            f"```\n\n"
            f"# Inputs\n\n"
            f"```json\n{_safe_inputs_json(inputs)}\n```\n\n"
            f"Produce the typed return value directly. Your output will be validated "
            f"against the declared return type."
        )
        result = await self._agent.run(prompt)
        return result.output


_INFER_SYSTEM_PROMPT = (
    "You are resolving a typed function via a single inference. The user will "
    "provide a Python signature, a docstring describing the task, and input "
    "values. Return the typed result directly — no explanation, no prose, just "
    "the structured value. If the task cannot be completed, raise a clearly "
    "structured error using the tool mechanism provided to you."
)


__all__ = ["Compiler", "GeneratedBody", "Inferencer", "InlineHelper", "render_user_prompt"]
