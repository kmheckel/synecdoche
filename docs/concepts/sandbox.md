# Sandbox

Generated bodies run inside [`pydantic-monty`](https://pydantic.dev/articles/pydantic-monty) —
a Rust-backed sandboxed Python interpreter. Every external call (MCP tool,
inline helper) yields back to the host, which dispatches it and resumes
execution.

## What's allowed

- `typing`, `dataclasses`, `json`, `re`, `math`, `pathlib`, `asyncio`,
  `collections`, `itertools`, `functools`, `datetime`
- Imported `pydantic` BaseModels that are part of the call's signature

## What's NOT allowed

- `os`, `sys`, `subprocess`, `socket`
- Any filesystem / network / env access
- Class definitions inside the generated body

Effects cross the boundary through **MCP tools** or **inline helpers** —
never directly.

## Why Monty

- Sub-microsecond startup.
- Snapshot/resume at every yield.
- Strict sandbox guarantees (Pydantic runs a bounty program).
- Pydantic-native — the same stack that backs our structured output.

## Swapping sandboxes

`Sandbox` is a protocol. Implement `execute(body, signature, surface,
inputs, external_functions)` and pass it via `Runtime(sandbox=...)`.
Pyodide, containerized Python, or a custom RPython subset all fit.
