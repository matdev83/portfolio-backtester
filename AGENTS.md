# AGENTS.md

## Purpose

This file provides instructions and best practices for coding agents (AI assistants, automation tools, or code-generation bots) contributing to or operating on this repository.

### Cursor: project skills (local only)

Optional Cursor project skills live under `.cursor/skills/<skill-name>/SKILL.md`. That tree is **gitignored** so local additions do not appear as repository changes. This repo does not vendor third-party skill collections; clone or copy skills from elsewhere (for example [wshobson/agents](https://github.com/wshobson/agents)) if you want them locally.

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

## MDMP cache, data directory, and `--mdmp-cache-only`

Portfolio-backtester reads OHLC through **market-data-multi-provider** (MDMP). For reproducible backtests and full fixed universes (e.g. country/regional ETFs), agents should understand the following.

### Same on-disk tree as the warmer

- **Default in `config/parameters.yaml`:** `mdmp_data_dir: ../market-data-multi-provider/data` (resolved from this repo’s root) so OHLC is read from the **sibling** `market-data-multi-provider` checkout — portfolio-backtester does **not** keep its own MDMP parquet tree.
- Override with **`MDMP_DATA_DIR`** (absolute path recommended) or `data_source_config.data_dir` / `mdmp_data_dir` if your layout differs.
- The factory passes the resolved path to `MarketDataClient(data_dir=...)`. If the backtester and a warm/fetch script use different roots, `--mdmp-cache-only` will miss rows that exist elsewhere.

### What `--mdmp-cache-only` implies

- MDMP returns a **subset** of requested symbols; missing keys mean **no usable parquet** (or empty slice) for that canonical id under the configured tree—not a portfolio-backtester “fake” fix.
- Recent MDMP versions align **NYSE:*** vs **AMEX:*** aliases on read; portfolio-backtester also **re-keys** `fetch_many` / `fetch_many_cached_only` results onto the requested canonical id by **bare ticker** when dict keys differ (`src/portfolio_backtester/data_sources/mdmp_data_source.py`).

### Warming symbols (run inside the **MDMP** repo checkout)

Use the MDMP maintainer script so canonical files exist before cache-only backtests, with **the same** `MDMP_DATA_DIR`:

```bash
cd <path-to-market-data-multi-provider>
set MDMP_DATA_DIR=<same path portfolio-backtester uses>
.venv\Scripts\python.exe scripts/warm_country_etf_symbols.py --refresh
```

Subset example (five symbols that were still empty on cache-only until warmed):

```bash
.venv\Scripts\python.exe scripts/warm_country_etf_symbols.py --refresh AMEX:EWI AMEX:EWL AMEX:EWM AMEX:EWU AMEX:EWW
```

Omit `--refresh` if a first-time materialization without forcing a provider re-pull is enough.

### Non–cache-only runs and TradingView

- If a symbol only resolves via **TradingView**, runs without `--mdmp-cache-only` can still fail on machines where that provider is not installed/configured; prefer Stooq/Yahoo materialization or warm those ids into parquet first.

## Research Validation Protocol

The backtester includes an opt-in `research_validate` mode for double out-of-sample WFO validation. This is a protocol layer above normal optimization: it searches WFO architecture choices on `global_train_period`, writes protocol artifacts/lock files, then validates the locked setup on `unseen_test_period`.

Run it with:

```bash
./.venv/Scripts/python.exe -m src.portfolio_backtester.backtester --mode research_validate --scenario-filename <scenario.yaml> --protocol double_oos_wfo

Optional: `--research-artifact-base-dir <path>` writes artifacts under `<path>/<scenario>/research_protocol/...` instead of `data/reports/`.
```

Scenario YAML must include a top-level `research_protocol` block, for example:

```yaml
research_protocol:
  enabled: true
  type: double_oos_wfo
  execution:
    max_grid_cells: 100   # cap on unique expanded WFO architecture cells (raises ResearchProtocolConfigError if exceeded)
    fail_fast: true       # reserved for future use; default true
  global_train_period:
    start_date: "2005-01-01"
    end_date: "2018-12-31"
  unseen_test_period:
    start_date: "2019-01-01"
    end_date: "2025-12-31"
  wfo_window_grid:
    train_window_months: [24, 36, 60]
    test_window_months: [3, 6, 12]
    wfo_step_months: [3, 6]
    walk_forward_type: [rolling]
  selection:
    top_n: 3
    metric: RobustComposite
  scoring:
    type: composite_rank
    weights:
      Calmar: 0.35
      Sortino: 0.25
      "Total Return": 0.20
      "Max Drawdown": 0.10
      Turnover: 0.10
    directions:
      Turnover: lower
  final_unseen_mode: reoptimize_with_locked_architecture
  constraints:
    - metric: "Max Drawdown"
      min_value: -0.35
    - metric: "Turnover"
      max_value: 60
    - metric: "Years Positive %"
      min_value: 0.55
  robust_selection:
    enabled: false
    weights:
      cell: 0.50
      neighbor_median: 0.30
      neighbor_min: 0.20
```

Omit `selection.metric` to get the same RobustComposite default. Single-metric selection (e.g. `metric: Calmar`) remains supported; do not set `scoring` in that case. Optional `constraints` gate architecture rows before ranking: each row needs all bounds satisfied (metric names use the same canonicalization as `scoring`); if none pass, the run fails with `ResearchConstraintError`. Optional `robust_selection` uses adjacent cells on the WFO train/test grid (same step and walk-forward type, eligible rows only) to build a `robust_score`; when `enabled`, ranking uses `robust_score` instead of the raw score. Example scenarios live under `config/scenarios/examples/research/` (small grids, suitable for smoke/CI).

Implementation notes for agents:
- Keep research protocol code under `src/portfolio_backtester/research/` unless it is only CLI/facade wiring.
- Preserve the separation between normal `optimize` and `research_validate`; do not change existing optimizer semantics to satisfy protocol behavior.
- Use TDD for protocol changes. Update tests under `tests/unit/research/` and `tests/integration/research/` before changing implementation.
- Verify artifact-producing changes with focused research tests and the full suite: `./.venv/Scripts/python.exe -m pytest tests/unit/research/ tests/integration/research/ -v` and `./.venv/Scripts/python.exe -m pytest tests/ -v`.
- Research artifacts are written under `data/reports/<scenario>/research_protocol/<run_id>/` and should include grid CSV, selected protocol YAML, lock YAML, unseen outputs, optional cost-sensitivity and bootstrap summaries, and a markdown report. Optional `research_protocol.reporting.generate_html: true` also writes `research_validation_report.html` (derivative of the same structured outputs).
- Optional `bootstrap` (under `research_protocol`) runs after unseen validation and after `cost_sensitivity` when enabled; it writes `bootstrap_significance.csv` and `bootstrap_summary.yaml` and does not affect selection.

## Proper Python interpreter file

To run all Python commands inside this project use the `.venv/Scripts/python.exe` file.

---

## Actions To Take After EACH File Edit

After each completed file Python (*.py) edit, agents MUST run the following quality-assurance command. This applies only to Python files:

```bash
./.venv/Scripts/python.exe -m ruff check --fix <modified_filename> && ./.venv/Scripts/python.exe -m black <modified_filename> && ./.venv/Scripts/python.exe -m mypy <modified_filename>
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

---

## Trade Execution Timing

The backtester supports configurable trade execution timing via `timing_config.trade_execution_timing`. This applies to **both** signal-based and portfolio/rebalancing strategies.

### Supported Values

- **`bar_close`** (default): Target weights take effect on the close of the signal/decision bar. This preserves existing backtest behavior where returns start affecting the portfolio from the next close-to-close period.
- **`next_bar_open`**: Target weights are deferred to the next trading session. The backtester remaps sparse target-weight events forward by one calendar day, which delays exposure and turnover.

### How It Works

1. **Signal generation** (`strategy_logic.py`): For signal-based strategies, skipped scan dates produce all-NaN rows (not zeros) so they are dropped by the mapper instead of becoming fake flatten events.
2. **Portfolio simulation** (`portfolio_logic.py`): 
   - For `time_based` mode: `rebalance_to_first_event_per_period()` collapses dense signals to the first observation per period while preserving the original timestamp.
   - `map_sparse_target_weights_to_execution_dates()` applies the execution timing remap.
   - `_sized_signals_to_weights_daily()` column-``ffill()``s event targets then ``fillna(0)``, then reindexes to the full session calendar with ``ffill`` so skipped signal rows do not zero out prior targets.
3. **Optimizer parity** (`evaluation_engine.py`): The same remap is applied in both fast-path blocks.

### Resolution Order

Strategy subclasses may override `get_trade_execution_timing()`. The base implementation resolves in this order:
1. `canonical_config.timing_config["trade_execution_timing"]`
2. `strategy_params["timing_config"]["trade_execution_timing"]`
3. Legacy `strategy_params["trade_execution_timing"]`
4. Default `"bar_close"`

### Adding to Scenario YAML

```yaml
timing_config:
  mode: time_based  # or signal_based
  rebalance_frequency: ME
  trade_execution_timing: bar_close  # or next_bar_open
```

When writing code that touches portfolio returns, signal generation, or optimizer evaluation paths, ensure the execution timing mapper is applied consistently.

### Simulation engines (canonical vs meta)

Standard strategies: net returns and costs come from `calculate_portfolio_returns` → `simulate_portfolio` → `canonical_portfolio_simulation_kernel` (share/cash, `rebalance_mask`, execution vs close prices). Meta strategies resolved to `MetaExecutionMode.TRADE_AGGREGATION` use a **separate** trade-aggregation path documented as an intentional alternate execution model until unified with the canonical engine; do not assume the same invariants without explicit tests (see the Meta section in [docs/simulation_execution_paths.md](docs/simulation_execution_paths.md)).

Legacy Numba helpers `drifting_weights_returns_kernel` / `detailed_commission_slippage_kernel` are **not** imported by production optimizer modules; they remain for tests and verification. See [docs/simulation_execution_paths.md](docs/simulation_execution_paths.md).

## Risk-free metrics (Sharpe / Sortino / DSR)

After `load_config()`, `GLOBAL_CONFIG` includes `risk_free_metrics_enabled` (default true) and `risk_free_yield_ticker` (default `^IRX`) unless disabled in `config/parameters.yaml`. **Sharpe**, **Sortino**, and **Deflated Sharpe** then use excess returns over implied per-bar rf from that yield index when data exists; if the yield series is missing or all-NaN, metrics fall back to the legacy CAGR/vol Sharpe path.

**Per-scenario opt-out:** in scenario `extras`, set `risk_free_metrics_enabled: false`, or set `risk_free_yield_ticker` to `null`, `""`, or the sentinel strings `legacy` / `none` / `off` (with the key present) so the scenario does not inherit the global ticker. Tests comparing legacy numbers should use one of these patterns.

**Programmatic tests** with `global_config={}` do not apply loader defaults; pass an explicit ticker or `risk_free_metrics_enabled: false` when Sharpe parity must match legacy-only behavior.

**Reference:** Full definitions (Sharpe RF-on vs legacy path, Sortino, Tail Ratio, “Deflated Sharpe” vs PSR-style output, drawdown units) are in `docs/performance_metrics.md`. Rich console row labels are controlled by `metrics_display_profile` (`legacy` | `platform_standard` | `verbose`) in `GLOBAL_CONFIG` or scenario `extras`; CSV rows keep canonical metric keys.
