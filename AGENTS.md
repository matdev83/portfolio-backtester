# AGENTS.md

## Purpose

This file provides instructions and best practices for coding agents (AI assistants, automation tools, or code-generation bots) contributing to or operating on this repository.

---

## General Guidelines

- **Respect Project Structure:**
  - Source code is in `src/portfolio_backtester/`.
  - Configuration files are in `config/` (YAML format).
  - Tests are in `tests/`.
  - Documentation is in `docs/`.
  - Virtual Environment is in `.venv/` and you MUST activate it before running any command.
- **Do Not Invent New Concepts:**
  - This project does not use the term "agent" in its codebase. Do not introduce new abstractions unless explicitly requested.
- **Follow Existing Patterns:**
  - When adding strategies, optimizers, or modules, follow the structure and style of existing files.
- **Preserve YAML and Config Conventions:**
  - All scenario and parameter configuration is managed via YAML files in `config/`.
- **Testing:**
  - Add or update tests in `tests/` for any new or modified logic.
- **Documentation:**
  - Update `README.md` and/or relevant files in `docs/` if you add features or change usage.

---

## File and Directory Conventions

- **Strategies:**
  - Implemented as classes in `src/portfolio_backtester/strategies/`.
  - Inherit from `BaseStrategy`.
- **Optimization:**
  - Optimizer logic is in `src/portfolio_backtester/optimization/`.
- **Data Handling:**
  - Data sources and loaders are in `src/portfolio_backtester/data/`.
- **Reporting:**
  - Reporting and analytics modules are in `src/portfolio_backtester/reporting/`.
- **Configuration:**
  - All scenario and parameter configuration is managed via YAML files in `config/`.
- **Tests:**
  - Place all tests in `tests/` and follow the naming conventions of existing test files.

---

## Coding Standards

- **Language:** Python 3.10+
- **Formatting:** Follow PEP8 and the style of existing code.
- **Imports:** Use absolute imports within the `src/portfolio_backtester/` package.
- **Logging:** Use the `logging` module, not print statements.
- **Type Hints:** Use type hints for all function signatures and class attributes.
- **Error Handling:** Use exceptions and logging for error reporting.

---

## Pull Request/Change Instructions

- **Atomic Commits:** Make small, focused commits for each logical change.
- **Describe Changes:** Clearly describe the purpose and scope of each change in commit messages or PR descriptions.
- **No Unnecessary Files:** Do not add temporary, debug, or unrelated files.
- **Respect Existing Functionality:** Do not break or remove existing features unless explicitly requested.

---

## How to Add a New Strategy (Example)

1. Create a new file in `src/portfolio_backtester/strategies/`.
2. Inherit from `BaseStrategy` and follow the pattern of existing strategies.
3. Add default parameters to `config/parameters.yaml` if needed.
4. Add or update tests in `tests/`.
5. Update documentation if the new strategy is user-facing.

---

## How to Update Configuration

- Edit YAML files in `config/`.
- Validate YAML syntax and ensure all required fields are present.
- Do not hardcode configuration in Python files.

---

For further details, see `README.md` and the `docs/` directory.
