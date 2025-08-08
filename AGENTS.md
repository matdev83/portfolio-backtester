# AGENTS.md

## Purpose

This file provides instructions and best practices for coding agents (AI assistants, automation tools, or code-generation bots) contributing to or operating on this repository.

---

## CRITICAL: Python Execution in WSL Environment

**WINDOWS VIRTUAL ENVIRONMENT DETECTED**: This project uses a Windows Python virtual environment (.venv) inside WSL. **ALWAYS** use the Windows Python executable for all Python commands:

- **Correct**: `./.venv/Scripts/python.exe -c "print('hello')"`
- **Correct**: `./.venv/Scripts/python.exe script.py`
- **Correct**: `./.venv/Scripts/python.exe -m pytest tests/`
- **Correct**: `./.venv/Scripts/python.exe -m pip install package`
- **Incorrect**: `python script.py` (system Python)
- **Incorrect**: `source .venv/bin/activate` (Linux virtual environment activation)

This approach works because WSL can execute Windows binaries directly. The virtual environment contains Windows .exe files that must be used instead of trying to activate the virtual environment in the traditional Linux way.

## Essential Commands

- **Setup**: `./.venv/Scripts/python.exe -m venv .venv && ./.venv/Scripts/python.exe -m pip install -e .`
- **Linting**: `./.venv/Scripts/python.exe -m ruff check src tests`
- **Type checking**: `./.venv/Scripts/python.exe -m mypy src`
- **Run all tests**: `./.venv/Scripts/python.exe -m pytest tests/ -v`
- **Run single test**: `./.venv/Scripts/python.exe -m pytest tests/path/to/test_file.py::test_function_name -v`
- **Run test category**: `./.venv/Scripts/python.exe -m pytest tests/unit/ -v` (or integration, system)

---

## Coding Standards

- **Language:** Python 3.10+
- **Formatting:** Ruff with Google docstring convention
- **Architecture**: Modular, layered, object-oriented design with a focus on SOLID, DRY, and high testability through separation of concerns.
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

- **Virtual Environment**: The project's virtual environment is located in the `.venv/` directory. **DO NOT** try to activate it using `source .venv/bin/activate`. Instead, **ALWAYS** prepend `./.venv/Scripts/python.exe` to all Python commands.
- **Dependency Management**: Agents are **NOT ALLOWED** to install packages directly using `pip`, `npm`, or any other package manager. All dependencies must be managed by editing the `pyproject.toml` file. After editing, the project must be re-installed in editable mode using `./.venv/Scripts/python.exe -m pip install -e .[dev]`. This is the only permitted use of `pip`.
- **Verification**: Before marking a task as complete, an agent **MUST** verify its work. This includes running specific tests related to the changes and executing the full test suite to ensure no regressions were introduced.
- **Codebase Integrity**: Agents are expected to only make changes that improve the codebase. This includes adding new functions/methods, improving existing ones, performing maintenance tasks (improving the shape of the code), and adding new functionalities. Agents are **NOT ALLOWED** to degrade the project's shape by removing functions, functionalities, files, or features, unless **EXPLICITLY** requested by the user.
- **Architectural Principles**: Adhere to the following software design principles:
  - **TDD (Test-Driven Development)**
  - **SOLID**
  - **DRY (Don't Repeat Yourself)**
  - **KISS (Keep It Simple, Stupid)**
  - **Convention over Configuration**

---

## Running the Optimizer

To run the optimizer for a specific strategy, you can use the `--scenario-filename` argument to point to the scenario file. For example, to run the optimizer for the dummy strategy, use the following command:

```bash
./.venv/Scripts/python.exe -m src.portfolio_backtester.backtester --mode optimize --scenario-filename config/scenarios/signal/dummy_strategy/dummy_strategy_test.yaml
```

## Proper Python interpreter file

To run all Python commands inside this project use the `.venv/Scripts/python.exe` file.

---

## Actions AFTER Each File Edit

After each completed file Python (*.py) edit, agents MUST run the following quality-assurance command. This applies only to Python files:

```bash
./.venv/Scripts/python.exe -m black <modified_filename> && ./.venv/Scripts/python.exe -m ruff check --fix <modified_filename> && ./.venv/Scripts/python.exe -m mypy <modified_filename>
```

Notes:

- Always use the Windows venv interpreter path shown above.
- Replace `<modified_filename>` with the exact path to the changed file.
- Run these before proceeding to additional edits or committing.

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

### When to Update Signatures

The `@api_stable` decorator protects critical methods from breaking changes by validating their signatures. You should ONLY run the signature update command in these specific situations:

1. **After adding a new `@api_stable` decorated method**
2. **After changing the signature of an existing `@api_stable` method** (parameter names, types, or return types)
3. **After removing an `@api_stable` decorated method**

### Update Command

When one of the above situations occurs, run:

```bash
./.venv/Scripts/python.exe scripts/update_protected_signatures.py --quiet
```

Then commit the updated `api_stable_signatures.json` file.

### Test-Driven Updates

The system is designed to fail tests when signatures change unexpectedly. If tests fail with messages like "No reference signature stored for <method>" or "Signature mismatch", then you know it's time to update the signatures.

DO NOT run the update command routinely - only when the API stability protection system requires it.
