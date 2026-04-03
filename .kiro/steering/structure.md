# Project Structure

Updated: 2026-01-31

## Organization Philosophy

- Layered, modular Python package under `src/portfolio_backtester/`.
- Configuration drives behavior: scenarios define *what to run*; code defines *how it runs*.
- Keep strategy logic isolated from data fetching; data access is handled by data source adapters.
- Prefer dependency inversion via factories/interfaces; avoid directly instantiating concrete implementations in core flows.

## Execution Flow (Mental Model)

1. CLI loads global config + scenario(s) (`config_loader.load_config()`).
2. Data is fetched through the data source abstraction (default: MDMP).
3. Backtest execution uses the modular facade (`Backtester`) composed of:
   - `DataFetcher` (fetch/prep)
   - `BacktestRunner` (core backtest loop)
   - `EvaluationEngine` (metrics/objectives)
   - `OptimizationOrchestrator` (WFO + parameter search)
4. Outputs (reports, studies, plots) are written under `data/`.

This split exists so that future specs can change one subsystem (e.g. timing or optimization) without rewriting the pipeline.

## Directory Patterns

### Core Package

Location: `src/portfolio_backtester/`

Purpose:

- CLI entry and orchestration (`backtester.py`, `core.py`).
- Configuration loading/validation (`config_loader.py`, `scenario_validator.py`, schema/validation helpers).

### Backtesting Pipeline

Location: `src/portfolio_backtester/backtester_logic/` and `src/portfolio_backtester/backtesting/`

Purpose:

- `backtester_logic/` holds the operational functions used by the backtest loop (signal generation, sizing, portfolio returns).
- `backtesting/` contains the newer “StrategyBacktester” architecture used by some paths (especially evaluation/optimization).

Key invariant: strategies are executed through shared pipeline code; don’t fork per-strategy execution logic.

### Strategies

Location: `src/portfolio_backtester/strategies/`

Purpose:

- Canonical strategy discovery paths:
  - `strategies/builtins/{portfolio,signal,meta}` (in-repo strategies)
  - `strategies/user/{portfolio,signal,meta}` (local/custom strategies)

Important:

- Automatic discovery only. Do not manually register strategies or hardcode imports.
- Avoid adding new strategies to legacy folders (`strategies/portfolio`, `strategies/signal`, `strategies/meta`) unless you also update discovery rules.

The codebase actively defends this rule: strategy registry APIs refuse manual registration and will raise if called.

### Provider Interfaces

Location: `src/portfolio_backtester/interfaces/`

Purpose:

- Pluggable providers for cross-cutting concerns:
  - universe selection
  - position sizing
  - stop loss / take profit
  - risk-off signals
  - data sources

In practice: strategies should call provider accessors (e.g. `get_universe_provider()`) rather than constructing components directly.

### Configuration

Location: `config/`

Purpose:

- `config/parameters.yaml` provides `GLOBAL_CONFIG` and shared defaults.
- `config/scenarios/**/*.yaml` defines runnable experiments.

Scenario files are validated at load time (YAML syntax + semantic validation). Semantic validation checks that `strategy_params` and `optimize` parameters match the strategy’s declared tunables.

### Tests

Location: `tests/`

Purpose:

- Unit/integration/system tests plus reusable base classes.
- Use base test classes in `tests/base/` when adding new strategy tests.

## Scenario Schema (Practical)

Scenarios are intentionally flat; common keys include:

```yaml
name: simple_momentum_strategy
strategy: SimpleMomentumPortfolioStrategy

timing_config:
  mode: time_based
  rebalance_frequency: ME

train_window_months: 60
test_window_months: 12

universe_config:
  type: named
  universe_name: sp500_top50

optimization_metric: Sharpe

optimize:
  - parameter: num_holdings
    min_value: 10
    max_value: 30
    step: 1

strategy_params:
  lookback_months: 12
  num_holdings: 10

  # Optional risk management (providers read these keys)
  stop_loss_config:
    type: AtrBasedStopLoss
    atr_length: 14
    atr_multiple: 2.0
  take_profit_config:
    type: NoTakeProfit
```

Universe configuration types are validated and typically look like:

```yaml
universe_config:
  type: named
  universe_name: sp500_top50

# or
universe_config:
  type: fixed
  tickers: [SPY, QQQ, IWM]

# or
universe_config:
  type: method
  method_name: get_top_weight_sp500_components
  n_holdings: 20
```

Patterns:

- `strategy_params` is the strategy-owned parameter bag.
- Optimization parameters must be either strategy tunables (from `tunable_parameters()`) or shared optimizer defaults.

Compatibility note:

- Semantic validation tolerates strategy-qualified params (`<strategy>.param`) and strips the prefix.
- Meta strategies cannot define `universe_config`/`universe` at the top level; they inherit universe from sub-strategies.

## Strategy Implementation Pattern (Actionable)

When adding a new strategy:

1. Create the strategy class under the canonical discovery directory:
   - Portfolio: `src/portfolio_backtester/strategies/builtins/portfolio/<name>.py`
   - Signal: `src/portfolio_backtester/strategies/builtins/signal/<name>.py`
   - Meta: `src/portfolio_backtester/strategies/builtins/meta/<name>.py`
2. Ensure the class name is the canonical name used in scenario YAML (`strategy: <ClassName>`).
3. Implement/override `generate_signals(...)` to return a weights DataFrame.
   - It is called per rebalance date with historical slices (not the full future).
   - Prefer returning a single-row DataFrame for `current_date`.
   - If you need extra non-traded inputs (e.g. benchmark constituents, macro series), override `get_non_universe_data_requirements()` and accept `non_universe_historical_data` in the signature.
4. Declare tunables via `tunable_parameters()` so scenarios and optimization validation can reason about the strategy.
5. Add a scenario YAML under `config/scenarios/builtins/...` and a unit test.

Signal generation call pattern to keep in mind:

- `generate_signals` is called with keyword args:
  - `all_historical_data`, `benchmark_historical_data`, optional `non_universe_historical_data`
  - `current_date`, optional `start_date`/`end_date` (WFO window bounds)

## Data Shape Conventions

- Daily OHLCV data is typically a DataFrame with MultiIndex columns: `(Ticker, Field)`.
- Many parts of the pipeline expect a `Close` field and will extract closes via `.xs("Close", level="Field", axis=1)`.
- Monthly closes are derived by resampling daily closes to business month end.

If you are writing tests or fixtures, prefer producing data in this shape to avoid false negatives caused by incompatible mock formats.

If you change data plumbing, keep these invariants intact or update the corresponding validators and tests.

## Naming Conventions

- Python modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Scenario files: `default.yaml` is the conventional entry scenario.

## Import Organization

Prefer absolute imports from the package.

```python
from portfolio_backtester.core import Backtester
from portfolio_backtester.config_loader import load_scenario_from_file
```

## Code Organization Principles

- Keep configuration in YAML and validate it early.
- Keep market data access behind data source adapters (MDMP is the default).
- Add features as cohesive changesets: code + config + validation + tests.
- If you touch a method decorated with `@api_stable`, update protected signatures.

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
