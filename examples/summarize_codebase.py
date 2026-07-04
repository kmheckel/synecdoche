"""Synthesis example — the model compiles a body that calls an MCP server.

The contract below has no body, so the runtime spawns one at first call:
the model writes a `solve` body that reads files through a FastMCP
filesystem server and assembles a typed `Summary`. The body joins the
archive as that signature's champion; subsequent runs skip compilation.
If it ever raises, the exception becomes a hard signal and the champion
is mutated into a descendant that fixes the defect.

Prerequisites:

    uv add --dev fastmcp-server-filesystem  # or install your preferred FS server
    export ANTHROPIC_API_KEY=sk-...

Run:

    uv run python examples/summarize_codebase.py <path-to-repo>
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import Client
from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

from synecdoche import Runtime


class ModuleSummary(BaseModel):
    path: str
    purpose: str
    public_api: list[str]


class Summary(BaseModel):
    top_level_purpose: str
    entry_points: list[str]
    major_modules: list[ModuleSummary]


rt = Runtime(
    model_code=AnthropicModel("claude-opus-4-7"),
    model_oracle=AnthropicModel("claude-haiku-4-5"),
    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.archive",
    heal=True,
    trace="stdout",
)


@rt.fn
def summarize_codebase(root: Path) -> Summary:
    """Summarize the architecture of a codebase at the given root.

    Walk the tree via the filesystem tools, sample a few representative
    files per directory, and produce a typed summary describing the
    top-level purpose, entry points, and the major modules with their
    public APIs.
    """


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    result = summarize_codebase(target)
    print(result.model_dump_json(indent=2))

    # Reflection: the function is an object you can interrogate.
    champ = summarize_codebase.champion
    print(f"\n# champion: v{champ.version} ({champ.operator}), score {champ.score():.2f}")
    print("# solidify() would render it back into committable source.")
