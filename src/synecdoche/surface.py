"""Tool surface: the set of external capabilities the sandbox can yield to.

The surface has two kinds of entries:

- **mcp**: tools discovered from a mounted FastMCP server. The backend
  dispatches them by calling `client.call_tool(...)`.
- **builtin**: primitives every generated body gets, currently just
  `infer` — a single typed judgment call to the model. This is the only
  nesting generated code is allowed: bodies are flat by construction and
  can never spawn further synthesized functions.

We compute a `surface_hash` over the MCP portion only — builtins are
constant and don't invalidate archive entries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastmcp import Client


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params_schema: dict[str, Any]
    return_schema: dict[str, Any]
    source: Literal["mcp", "builtin"]
    mcp_client_id: int | None = None  # index into Backend._mcp_clients

    def render_stub(self) -> str:
        """Render a typed async stub for sandbox typechecking."""
        params = _params_from_schema(self.params_schema)
        params_src = ", ".join(params) if params else ""
        doc = (self.description or "").replace('"""', "'''")
        return (
            f"async def {self.name}({params_src}) -> {_ret_hint(self.return_schema)}:\n"
            f'    """{doc}"""\n'
            f"    ...\n"
        )

    def render_description(self) -> str:
        """Render a compact description for the compiler user prompt."""
        return (
            f"### {self.name}  [source: {self.source}]\n"
            f"{self.description}\n\n"
            f"params: {json.dumps(self.params_schema, sort_keys=True)}\n"
            f"returns: {json.dumps(self.return_schema, sort_keys=True)}\n"
        )


@dataclass
class ToolSurface:
    tools: tuple[ToolSpec, ...] = field(default_factory=tuple)
    surface_hash: str = ""

    def render_stubs(self) -> str:
        """Render all stubs as a single Python source fragment."""
        return "\n".join(t.render_stub() for t in self.tools)

    def render_descriptions(self) -> str:
        return "\n".join(t.render_description() for t in self.tools)

    def by_name(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None


INFER_SPEC = ToolSpec(
    name="infer",
    description=(
        "Ask the model for a single judgment: classification, extraction, "
        "phrasing — anything that is perception rather than algorithm. "
        "`instruction` says what to decide; `data` is the text to decide "
        "about. Returns the model's answer as a string (ask for JSON in the "
        "instruction if you need structure). This is a real inference call: "
        "use it for judgment, not for computation you can express as code."
    ),
    params_schema={
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "data": {"type": "string"},
        },
        "required": ["instruction"],
    },
    return_schema={"type": "string"},
    source="builtin",
)


def empty_surface() -> ToolSurface:
    """The surface with no MCP tools mounted — builtins only."""
    return ToolSurface(tools=(INFER_SPEC,), surface_hash="empty")


async def build_mcp_surface(clients: list[Client]) -> ToolSurface:
    """Connect to every MCP client, enumerate its tools, and build a ToolSurface."""
    tools: list[ToolSpec] = [INFER_SPEC]
    for idx, client in enumerate(clients):
        async with client:
            raw_tools = await client.list_tools()
            for t in raw_tools:
                tools.append(
                    ToolSpec(
                        name=t.name,
                        description=t.description or "",
                        params_schema=dict(getattr(t, "inputSchema", {}) or {}),
                        return_schema=dict(getattr(t, "outputSchema", {}) or {}),
                        source="mcp",
                        mcp_client_id=idx,
                    )
                )
    return ToolSurface(
        tools=tuple(tools),
        surface_hash=_hash_surface(tools),
    )


def _hash_surface(tools: list[ToolSpec]) -> str:
    payload = [
        {
            "name": t.name,
            "description": t.description,
            "params": t.params_schema,
            "returns": t.return_schema,
            "source": t.source,
        }
        for t in sorted(tools, key=lambda t: t.name)
        if t.source == "mcp"
    ]
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _params_from_schema(schema: dict[str, Any]) -> list[str]:
    """Derive a stub parameter list from a JSON schema."""
    if not schema:
        return []
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out: list[str] = []
    for name, prop in props.items():
        hint = _ret_hint(prop)
        if name in required:
            out.append(f"{name}: {hint}")
        else:
            out.append(f"{name}: {hint} | None = None")
    return out


def _ret_hint(schema: dict[str, Any]) -> str:
    """Map a JSON schema type to a rough Python type hint string."""
    if not schema:
        return "object"
    t = schema.get("type")
    mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list[object]",
        "object": "dict[str, object]",
        "null": "None",
    }
    if isinstance(t, list):
        return " | ".join(mapping.get(x, "object") for x in t)
    return mapping.get(t, "object")


__all__ = ["INFER_SPEC", "ToolSpec", "ToolSurface", "build_mcp_surface", "empty_surface"]
