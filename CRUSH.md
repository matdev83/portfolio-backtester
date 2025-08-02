# CRUSH.md

## Build/Lint/Test Commands
- **Setup**: `./.venv/Scripts/python.exe -m venv .venv && ./.venv/Scripts/python.exe -m pip install -e .[dev]`
- **Lint**: `./.venv/Scripts/python.exe -m ruff check src tests`
- **Typecheck**: `./.venv/Scripts/python.exe -m mypy src`
- **Test All**: `./.venv/Scripts/python.exe -m pytest tests/ -v`
- **Single Test**: `./.venv/Scripts/python.exe -m pytest tests/path/to/test.py::test_name -v`

## Code Style
- PEP8-compliant with Ruff formatting
- Type hints mandatory for all functions
- Google docstrings for public interfaces
- snake_case variables, PascalCase classes
- Enums for constants/magic numbers
- DRY principles and SOLID architecture
- Error handling via exceptions + logging

## Project Notes
- Always use Windows Python in WSL (`.venv/Scripts/python.exe`)
- Edit YAML in `config/`, validate syntax before committing
- **Critical**: After changes, run optimizer to verify realistic metrics and check for NaNs/inf (per `.cursor/rules/run-optimizer-to-confirm.mdc`)
- API stability: Update `api_stable_signatures.json` via `update_protected_signatures.py --quiet` when modifying `@api_stable` methods