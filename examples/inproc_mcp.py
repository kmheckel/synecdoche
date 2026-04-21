"""In-process FastMCP server example — no external MCP binaries needed.

This defines a tiny FastMCP server with a couple of tools, mounts it into a
Runtime via an in-memory transport, and lets the LLM compile a body that uses
them. A single file, runnable end-to-end.

Prerequisites:
    export ANTHROPIC_API_KEY=sk-...
    # or swap in Ollama as examples/ollama_infer.py shows

Then:
    uv run python examples/inproc_mcp.py
"""

from __future__ import annotations

import os

from fastmcp import Client, FastMCP
from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel

from synecdoche import Runtime

# --- Define a tiny MCP server (in-process) ----------------------------------

server = FastMCP("in-proc-demo")

_WORDS = [
    ("dream", "noun/verb — a sequence of images, ideas, or sensations"),
    ("lexicon", "noun — the vocabulary of a person, language, or branch of knowledge"),
    ("quixotic", "adj — exceedingly idealistic, unrealistic, or impractical"),
    ("ephemeral", "adj — lasting for a very short time"),
    ("zeitgeist", "noun — the defining spirit or mood of a particular period"),
]


@server.tool
def list_words() -> list[str]:
    """Return the list of words in our tiny dictionary."""
    return [w for w, _ in _WORDS]


@server.tool
def define(word: str) -> str:
    """Look up the definition of a single word. Raises KeyError if absent."""
    for w, d in _WORDS:
        if w.lower() == word.lower():
            return d
    raise KeyError(f"unknown word: {word!r}")


# --- Build a Runtime that talks to the server over an in-memory transport ----


class Glossary(BaseModel):
    entries: dict[str, str]


rt = Runtime(
    model=AnthropicModel("claude-haiku-4-5") if os.environ.get("ANTHROPIC_API_KEY") else None,  # type: ignore[arg-type]
    mcp=[Client(server)],
    archive="./.archive",
    trace="tree",
)


@rt.recursion
def build_glossary() -> Glossary:
    """Read every word from list_words, look up each definition via `define`,
    and return a Glossary with the full map."""


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Set ANTHROPIC_API_KEY (or edit this file to use Ollama — see "
            "examples/ollama_infer.py)."
        )
    result = build_glossary()
    for word, defn in result.entries.items():
        print(f"{word:>12}  {defn}")
