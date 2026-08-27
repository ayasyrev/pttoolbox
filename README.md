# pttoolbox

Toolbox for PyTorch.

## Development

The project requires Python 3.12 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency management, running tools, and
building distributions.

```bash
uv sync --locked
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pre-commit run --all-files
```

Build the wheel and source distribution with:

```bash
uv build --no-sources
```

After inspecting the artifacts, a maintainer can publish them with:

```bash
uv publish dist/*
```
