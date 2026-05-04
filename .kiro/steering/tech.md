# Technology Stack

Updated: 2026-01-31

## Architecture

- CLI-driven application with configuration-first execution.
- Configuration is split into:
  - Global parameters (`config/parameters.yaml`)
  - Scenario files (`config/scenarios/**/*.yaml`)
- Core orchestration uses a facade-style `Backtester` composed of specialized components (data fetch, backtest execution, evaluation, optimization).
- Data access uses Dependency Inversion: components depend on interfaces/factories, not concrete downloaders.

## Configuration Model

- `config/parameters.yaml` must define:
  - `GLOBAL_CONFIG`: run-wide defaults (benchmark, dates, costs, data source settings)
  - `OPTIMIZER_PARAMETER_DEFAULTS`: shared tunables used by optimizers (and referenced by scenario validation)
- Additional top-level config blocks (e.g. Monte Carlo / robustness) are merged into `GLOBAL_CONFIG` at load time.
- Scenario YAMLs are validated semantically:
  - `strategy` must resolve to a discovered strategy class
  - `strategy_params` keys must be declared in the strategy’s `tunable_parameters()`
  - optimization parameters must be strategy tunables or present in optimizer defaults

## Core Technologies

- Language: Python (>= 3.10)
- Packaging: `src/` layout (`portfolio_backtester` package)
- CLI: `argparse`
- Configuration: YAML (`pyyaml`) validated at load time (syntax + semantics)

## Key Libraries

- Data/compute: `pandas`, `numpy`, `scipy`
- Optimization: `optuna` + an in-repo genetic optimizer
- Performance: optional Numba acceleration and joblib parallelism in optimization paths
- Reporting/plots: `matplotlib`, `seaborn`, `plotly` (reports saved under `data/`)
- Console UX: `rich`
- Market data: `market-data-multi-provider` (MDMP) via `MarketDataMultiProviderDataSource`

## Data Access (Hard Constraint)

- Do not add direct market data downloads (no new `requests`/`yfinance.download`/etc for price series).
- Use the data source abstraction: `portfolio_backtester.interfaces.create_data_source()`.
- Default data source is MDMP; it supports:
  - canonical symbol mapping (e.g. local symbols mapped to canonical IDs)
  - cache-only runs for offline/reproducible execution

Ticker convention note:

- `GLOBAL_CONFIG.benchmark` is often a canonical MDMP symbol (e.g. `AMEX:SPY`).
- Many scenario universes use local tickers; the MDMP adapter maps them to canonical IDs internally.

Cache-only knobs:

- CLI: `--mdmp-cache-only` (sets `GLOBAL_CONFIG.data_source_config.cache_only = true`)
- Config: `GLOBAL_CONFIG.data_source_config.cache_only: true`

## Development Standards

### Type Safety

- Type hints on all public function signatures.
- mypy is configured with stricter rules for critical modules (see `pyproject.toml`).

### Code Quality

- Formatting: Black (line length 100)
- Linting: Ruff
- Type checking: mypy
- Security scanning: Bandit

### Testing

- Test runner: pytest
- Markers distinguish unit/integration/system and fast/slow tests.
- Coverage minimum is enforced via pytest config in `pyproject.toml`.

## Development Environment

### Required Tools

- Windows virtual environment at `.venv/`.
- Always run Python via `.venv\Scripts\python.exe` (do not rely on activation).

### Common Commands

```bash
# Install (editable + dev tools)
.venv\Scripts\python.exe -m pip install -e .[dev]

# Run backtest (scenario file)
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester --mode backtest --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml"

# Run optimization
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester --mode optimize --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml" --optimizer optuna --optuna-trials 200 --n-jobs 4

# QA (after editing .py files)
.venv\Scripts\python.exe -m ruff check --fix <file> && .venv\Scripts\python.exe -m black <file> && .venv\Scripts\python.exe -m mypy <file>

# Tests
.venv\Scripts\python.exe -m pytest tests/ -v

# Validate configuration (syntax + semantic checks)
.venv\Scripts\python.exe -m src.portfolio_backtester.config_loader --validate
```

## Performance & Debugging Notes

- Optimization writes a debug log file (`optimizer_debug.log`) for deep diagnostics.
- Thread caps for numerical libs can be applied from config (see `backtester.py` performance tweaks).

## API Stability

- Some methods are decorated with `@api_stable` and are treated as signature-stable.
- If you change a protected signature, update `api_stable_signatures.json` via:

```bash
.venv\Scripts\python.exe scripts/update_protected_signatures.py --quiet
```

---
_Document standards and patterns, not every dependency_
