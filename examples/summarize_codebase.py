"""@syn + tools — generate a body that calls an MCP server.

The function below has no body, so the first call compiles one: the model
writes a `solve` body that reads files through a FastMCP filesystem server
and assembles a typed `Summary`. The compiled body is cached against
(signature, tool surface); subsequent runs skip compilation. If it ever
raises, the exception becomes the compile context for a fixed descendant.

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

import synecdoche as syn


class ModuleSummary(BaseModel):
    path: str
    purpose: str
    public_api: list[str]


class Summary(BaseModel):
    top_level_purpose: str
    entry_points: list[str]
    major_modules: list[ModuleSummary]


syn.configure(
    model_code=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),
    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.synecdoche",
    heal=True,
    trace="stdout",
)


@syn
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

    champ = syn.champion(summarize_codebase)
    print(f"\n# champion: v{champ.version} ({champ.operator}), score {champ.score():.2f}")
    print("# The champion is also mirrored at ./.synecdoche/champions/ as readable source.")
