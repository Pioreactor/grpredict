# Repository Guidelines

## Project Structure & Module Organization
This is a small Python package with a `src/` layout:
- `src/grpredict/__init__.py`: core implementation (`CultureGrowthEKF`, moving statistics helpers).
- `src/grpredict/__init__.pyi` and `src/grpredict/py.typed`: typing surface for consumers.
- `tests/`: pytest suite (`test_ekf.py`, `test_moving_accumulators.py`).
- `build/` and `dist/`: packaging artifacts; treat as generated output.

Keep new code close to existing modules unless there is a clear boundary that justifies a new file.

## Build, Test, and Development Commands
Use the local virtual environment (`.venv`) when available.

```bash
source .venv/bin/activate
pip install -e ".[dev]"      # editable install + pytest/black/mypy
pytest -v                      # full test suite (matches CI)
pytest tests/test_ekf.py -q    # run one file during iteration
python -m mypy src             # type-check package code
python -m build                # build sdist/wheel into dist/
```

CI (`.github/workflows/ci.yaml`) runs tests on Python 3.11 with `pytest -v`.

## Coding Style & Naming Conventions
- Follow standard Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Preserve explicit type hints on public functions and tests.
- Keep functions focused and readable; prefer straightforward logic over clever abstractions.
- Use `black` formatting conventions (configured via default behavior unless specified otherwise).

## Testing Guidelines
- Framework: `pytest`.
- Test files must be named `test_*.py`; test functions should start with `test_`.
- Add targeted tests for behavior changes before broad refactors.
- During development, run individual tests first, then the full suite before opening a PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects (for example: `adding ci`, `Update README.md`).
- Keep commit titles concise and action-oriented.
- In PRs, include: problem statement, implementation summary, and test evidence (`pytest` output or equivalent).
- Link related tickets/issues when applicable and call out any API/behavior changes explicitly.

## Security & Configuration Tips
- Never commit secrets or local environment files (`.env`, `.envrc`).
- Validate changes against typed interfaces (`.pyi`/`py.typed`) when modifying public APIs.
