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

## Code editing/management rules

All changes you make in this project need to be additive and ensure incremental enhancements, not reductions. You are NOT ALLOWED to perform any action that could result in the reduced functionality of the project (class/file/function deletion)

## Before submitting a pull request or finishing a task

You MUST run all tests and ensure that they pass. You MUST also run the code and ensure that it runs without errors. You MUST ensure that all code is formatted according to PEP8 and that the code is documented according to docstrings

## Running the Optimizer

To run the optimizer for a specific strategy, you can use the `--scenario-filename` argument to point to the scenario file. For example, to run the optimizer for the dummy strategy, use the following command:

```bash
.venv\Scripts\python -m src.portfolio_backtester.backtester --mode optimize --scenario-filename config/scenarios/signal/dummy_strategy/dummy_strategy_test.yaml
```

## Custom Workflow: review-branch-rebase-main
