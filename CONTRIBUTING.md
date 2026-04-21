# Contributing to synecdoche

Thanks for your interest! This is a small POC library — PRs welcome.

## Dev setup

```bash
git clone https://github.com/kmheckel/synecdoche.git
cd synecdoche
uv sync --dev
```

That's it. `uv sync` creates `.venv/` and installs the package in editable mode.

## Running the test suite

```bash
uv run pytest -v
```

All tests are stub-model-backed — they don't call any real LLM providers and don't need
API keys. Should run in well under a second.

Live API tests are marked `@pytest.mark.live` and skipped by default; run them with
`uv run pytest -m live` (requires `ANTHROPIC_API_KEY`).

## Linting and formatting

```bash
uv run ruff check          # lint
uv run ruff format         # format
uv run ruff check --fix    # autofix lint issues
```

CI runs `ruff check`, `ruff format --check`, and `pytest`. Please match that locally
before opening a PR.

## Branch + PR conventions

- Branch off `main`. Name it `feature/<short-description>`, `fix/<short-description>`,
  or `docs/<short-description>`.
- Commits: imperative mood, ≤72 chars for the subject. Link an issue with
  `Closes #N` / `Fixes #N` in the body when applicable.
- PRs should describe what changed and why, and ideally include a test.
- Keep PRs focused — one concern per PR.

## Project layout

```
src/synecdoche/          # the package
  runtime.py             # the Runtime orchestrator + decorators
  compiler.py            # prompt rendering + GeneratedBody emission
  sandbox.py             # Monty wrapping + script assembly
  archive.py             # SQLite + in-memory backends
  repair.py              # exception-driven repair loop
  surface.py             # FastMCP tool surface ingestion
  signature.py           # CallSignature + hashing
  jit.py, trace.py, exceptions.py
  prompts/               # system + user jinja templates
tests/                   # stub-model tests
examples/                # runnable demos
docs/                    # design doc + mkdocs sources (once set up)
```

## Docs

Docs live in `docs/`. The mkdocs site (when set up — issue #6) will build via
`uv run mkdocs serve` / `uv run mkdocs build`.

## Releasing

Version lives in `src/synecdoche/__init__.py`. Bumping:

1. Update `__version__` and `pyproject.toml`.
2. `git tag vX.Y.Z && git push --tags`.
3. The `Publish` workflow builds sdist + wheel and publishes to PyPI via trusted
   publishing.

## Getting help

Open an issue with your question — we'll tag it `question`.
