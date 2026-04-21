"""Full @rt.recursion example — LLM compiles a body that calls an MCP server.

This mirrors the design spec's canonical example: the model writes a
`solve` body that reads files through a FastMCP filesystem server and
assembles a typed `Summary` of the target codebase. The body is archived
after the first successful call; subsequent runs hit the cache.

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
    model_recursion=AnthropicModel("claude-opus-4-7"),
    model_infer=AnthropicModel("claude-haiku-4-5"),
    mcp=[Client("stdio://mcp-server-filesystem")],
    archive="./.archive",
    heal=True,
    trace="stdout",
)


@rt.recursion
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
