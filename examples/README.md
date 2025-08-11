# Examples

These are example scenarios for running the CLI backtest/optimization with built-in strategies.

- Built-in strategies are part of the framework (`strategies/builtins/**`) and are safe to use in scenarios and tests.
- User strategies should be added under `strategies/user/**` and will be discovered automatically by the framework.
- Examples should not be imported by tests and are not part of the Python package discovery.

## Run a backtest

Use the project CLI backtester (example):

```
./.venv/Scripts/python.exe -m src.portfolio_backtester.backtester \
  --mode backtest \
  --scenario-filename examples/scenarios/portfolio/fixed_weight_example.yaml
```

## Run an optimization

```
./.venv/Scripts/python.exe -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-filename examples/scenarios/portfolio/fixed_weight_example.yaml
```

## Extend strategies

- Extend framework strategies by creating new classes under `strategies/user/{portfolio,signal,meta}`.
- Keep user code and examples separate from framework core; tests rely only on built-ins and testing fixtures.
