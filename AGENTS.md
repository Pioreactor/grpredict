# Repository Guidelines

## Project Structure & Module Organization
This is a small Python package with a `src/` layout:
- `src/grpredict/__init__.py`: core implementation (`CultureGrowthEKF`, moving statistics helpers).
- `src/grpredict/__init__.pyi` and `src/grpredict/py.typed`: typing surface for consumers.
- `tests/`: pytest suite (`test_ekf.py`, `test_moving_accumulators.py`).
- `build/` and `dist/`: packaging artifacts; treat as generated output.

Keep new code close to existing modules unless there is a clear boundary that justifies a new file.

## Build, Test, and Development Commands
Use the local virtual environment (`.venv`) when available. Use `python3.14`.

```bash
pytest -v                      # full test suite (matches CI)
pytest tests/test_ekf.py -q    # run one file during iteration
python -m mypy src             # type-check package code
```


## Testing Guidelines
- Framework: `pytest`.
- Test files must be named `test_*.py`; test functions should start with `test_`.
- Add targeted tests for behavior changes before broad refactors.
- During development, run individual tests first, then the full suite before opening a PR.

## Security & Configuration Tips
- Never commit secrets or local environment files (`.env`, `.envrc`).
- Validate changes against typed interfaces (`.pyi`/`py.typed`) when modifying public APIs.
