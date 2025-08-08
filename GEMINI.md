# Project-specific notes for GEMINI CLI AGENT and GEMINI LLM model

## Coding guidelines

- PEP8 formatting
- Use of f-strings for string formatting
- TDD (Test-Driven Development) for all code
- Respect SOLID principles (Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, Dependency Inversion)
- Respect DRY (Don't Repeat Yourself) principle
- Respect KISS (Keep It Simple, Stupid) principle
- Use of type hints
- Use of docstrings
- Use of enums for constants
- Use of constants for magic numbers
- Use of constants for configuration parameters

## Running tests

Remember to always activate the virtual environment from .venv/Scripts/activate before running tests.

## Running code

Remember to always activate the virtual environment from .venv/Scripts/activate before running code.

## Always Run Q&A Tools After File Edits

## Actions AFTER Each File Edit

After each completed file Python (*.py) edit, agents MUST run the following quality-assurance command. This applies only to Python files:

```bash
./.venv/Scripts/python.exe -m black <modified_filename> && ./.venv/Scripts/python.exe -m ruff check --fix <modified_filename> && ./.venv/Scripts/python.exe -m mypy <modified_filename>
```

## Code editing/management rules

All changes you make in this project need to be additive and ensure incremental enhancements, not reductions. You are NOT ALLOWED to perform any action that could result in the reduced functionality of the project (class/file/function deletion)

## Before submitting a pull request or finishing a task

You MUST run all tests and ensure that they pass. You MUST also run the code and ensure that it runs without errors. You MUST ensure that all code is formatted according to PEP8 and that the code is documented according to docstrings

## Running the Optimizer

To run the optimizer for a specific strategy, you can use the `--scenario-filename` argument to point to the scenario file. For example, to run the optimizer for the dummy strategy, use the following command:

```bash
.venv\Scripts\python -m src.portfolio_backtester.backtester --mode optimize --scenario-filename config/scenarios/signal/dummy_strategy/dummy_strategy_test.yaml
```

## Desktop-Commander MCP Usage Prompt

You are connected to the **Desktop-Commander MCP** server (`desktop-commander`).  
**Always use its tools instead of your built-in ones.**

| When you need to  | Call this MCP tool |
|--------------------|--------------------|
| list directories   | `list_directory` |
| find files by name | `search_files` |
| grep / recursive search inside files | `search_code` (ripgrep-powered) |
| read / show a file | `read_file` |
| edit / replace text | `edit_block` (surgical edits) or `write_file` |

### Why prefer Desktop-Commander

1. **Speed** - ripgrep search and diff editing handle huge codebases instantly.  
2. **Auditability** - every action is logged and undo-friendly.  
4. **Safety** - tail gigantic files via safe line limits without memory blow-ups.

**Rule:** Route **all** file access, search, editing, through these MCP tools unless explicitly instructed otherwise.

**Anti-Rule:** Don't use `desktop-commander` for shell commands, use your built-in tools instead.

## Remember

Always prefer the newest and most advanced available set of tools to access files and run shell commands, which currently is the one provided by `desktop-commander`, over your original and now obsolete tools


