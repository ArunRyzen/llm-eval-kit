# Contributing

## Setup
```bash
uv sync --extra dev
uv run pre-commit install
```

## Checks (CI enforces these)
```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
```

## Conventions
- Type hints everywhere; mypy clean. Tests for new logic, fully offline (FakeJudge / scorers).
- New scorer or guard: implement the `Scorer` / `Guard` protocol — it composes automatically.
- Secrets via `.env` (never committed); update `.env.example` when adding a variable.
- Conventional-commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
