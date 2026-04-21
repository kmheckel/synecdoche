# MCP tools

All external capabilities — filesystems, databases, domain-specific tools —
are mounted as [FastMCP](https://gofastmcp.com/) servers on the Runtime.
There is no `@tool` decorator in synecdoche itself.

## Mounting

```python
from fastmcp import Client

rt = Runtime(
    model=...,
    mcp=[
        Client("stdio://mcp-server-filesystem"),
        Client("https://tools.example.com/mcp"),
    ],
)
```

## In-process servers

For a self-contained example:

```python
from fastmcp import Client, FastMCP

server = FastMCP("my-tools")

@server.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

rt = Runtime(model=..., mcp=[Client(server)])
```

See [`examples/inproc_mcp.py`](https://github.com/kmheckel/synecdoche/blob/main/examples/inproc_mcp.py)
for a runnable version.

## Tool-surface hash

The runtime hashes the enumerated tool schemas at startup. The hash becomes
part of the archive key: swapping servers (or changing a tool's schema)
invalidates the cache and triggers re-compilation on the next call.
