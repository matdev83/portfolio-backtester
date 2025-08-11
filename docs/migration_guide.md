# Migration Guide: Strategy Layout Refactor

This guide summarizes changes introduced by the strategy layout refactor and how to update your code and configs.

## New Directory Layout
- Core internals: `src/portfolio_backtester/strategies/_core/**`
- Built-in strategies: `src/portfolio_backtester/strategies/builtins/{portfolio,signal,meta}/...`
- User strategies: `src/portfolio_backtester/strategies/user/{portfolio,signal,meta}/...`
- Examples (not auto-discovered): `src/portfolio_backtester/strategies/examples/**`

## Config Scenarios
- Built-ins: `config/scenarios/builtins/<category>/<strategy>/default.yaml`
- User: `config/scenarios/user/<category>/<strategy>/default.yaml`

Examples:
- `config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml`
- `config/scenarios/builtins/signal/ema_crossover_signal_strategy/default.yaml`
- `config/scenarios/user/signal/hello_world_signal_strategy/default.yaml`

## Imports (Tests/Tooling)
- Base types: `from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy`
- Registry: `from portfolio_backtester.strategies._core.registry import get_strategy_registry`
- Strategy factory: `from portfolio_backtester.strategies._core.strategy_factory import StrategyFactory`

Avoid legacy `src.portfolio_backtester.*` imports.

## Discovery Rules
- Class names must end with: `PortfolioStrategy`, `SignalStrategy`, or `MetaStrategy`.
- Filenames must end with: `_portfolio_strategy.py`, `_signal_strategy.py`, `_meta_strategy.py`.
- Place strategies only under `builtins/` or `user/` trees.

## CLI Usage
- Backtest: `python -m src.portfolio_backtester.backtester --mode backtest --scenario-filename <path.yaml>`
- Optimize: `python -m src.portfolio_backtester.backtester --mode optimize --scenario-filename <path.yaml>`

## Validation
- Scenarios and defaults are validated before backtests/optimizations. Fix reported parameter/type issues in YAML to proceed.

## Notes
- Examples under `strategies/examples/**` and `config/scenarios/examples/**` are excluded from discovery/validation.
- Diagnostic strategies have been retired; use dedicated scripts/tests instead.


