# AGENTS.md

## Purpose

This file provides instructions and best practices for coding agents (AI assistants, automation tools, or code-generation bots) contributing to or operating on this repository.

---

## Essential Commands

- **Setup**: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- **Linting**: `ruff check src tests`
- **Type checking**: `mypy src`
- **Run all tests**: `python -m pytest tests/ -v`
- **Run single test**: `python -m pytest tests/path/to/test_file.py::test_function_name -v`
- **Run test category**: `python -m pytest tests/unit/ -v` (or integration, system)

---

## Coding Standards

- **Language:** Python 3.10+
- **Formatting:** Ruff with Google docstring convention
- **Imports:** Absolute imports within `src/portfolio_backtester/` package
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Type Hints:** Required for all function signatures and class attributes
- **Docstrings:** Google style for all public functions and classes
- **Logging:** Use `logging` module, not print statements
- **Error Handling:** Use exceptions and logging for error reporting

---

## Project Structure Guidelines

- Source code: `src/portfolio_backtester/`
- Configuration: `config/` (YAML files)
- Tests: `tests/` (mirrors src structure)
- Documentation: `docs/`
- Virtual Environment: `.venv/` (MUST activate before running commands)

---

## Agent Development Principles

- **Virtual Environment**: The project's virtual environment is located in the `.venv/` directory. It **MUST** be activated (`source .venv/bin/activate`) before executing any Python commands.
- **Dependency Management**: Agents are **NOT ALLOWED** to install packages directly using `pip`, `npm`, or any other package manager. All dependencies must be managed by editing the `pyproject.toml` file. After editing, the project must be re-installed in editable mode using `pip install -e .`. This is the only permitted use of `pip`.
- **Verification**: Before marking a task as complete, an agent **MUST** verify its work. This includes running specific tests related to the changes and executing the full test suite to ensure no regressions were introduced.
- **Architectural Principles**: Adhere to the following software design principles:
    - **TDD (Test-Driven Development)**
    - **SOLID**
    - **DRY (Don't Repeat Yourself)**
    - **KISS (Keep It Simple, Stupid)**
    - **Convention over Configuration**

---

## Pull Request Workflow

1. Make atomic commits with clear messages
2. Add/update tests for new/modified logic
3. Run linter and type checker before committing
4. Update `README.md` and docs for user-facing changes
5. Do not introduce new abstractions unless explicitly requested

---

## Configuration Management

- Edit YAML files in `config/` for scenarios and parameters
- Validate YAML syntax before committing
- Do not hardcode configuration in Python files

---

## API Stability

After introducing a new `@api_stable` method run:
```bash
python scripts/update_protected_signatures.py --quiet
```
Commit the updated `api_stable_signatures.json` file.