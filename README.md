# Portfolio Backtester

A powerful Python tool for backtesting portfolio strategies with walk-forward optimization, Monte Carlo stress testing, and comprehensive performance analysis.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Setup](#setup)
- [Available Strategies](#available-strategies)
- [Usage](#usage)
  - [Running a Backtest](#running-a-backtest)
  - [Running Optimization](#running-optimization)
  - [Monte Carlo Stress Testing](#monte-carlo-stress-testing)
- [Configuration](#configuration)
  - [Universe Configuration](#universe-configuration)
  - [Risk Management](#risk-management)
  - [Commission and Slippage](#commission-and-slippage)
  - [Capital Allocation Modes](#capital-allocation-modes)
- [Rebalance Frequencies](#rebalance-frequencies)
- [CLI Reference](#cli-reference)
- [Example Outputs](#example-outputs)
  - [Dual Momentum Top-N Proof Runs](#dual-momentum-top-n-proof-runs)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

**1. Install the package:**

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install
pip install -e .
```

**2. Run your first backtest:**

```bash
python -m src.portfolio_backtester.backtester --mode backtest \
  --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml"
```

**3. Check the results** in `data/reports/`

### Optimization Shortcut

Use the convenience script for quick optimization runs:

```bash
./optimize.py config/scenarios/builtins/portfolio/calmar_momentum_portfolio_strategy/default.yaml \
  --optuna-trials 100 --n-jobs 4
```

---

## Key Features

### 🎯 Strategy Backtesting

- **11+ Built-in Strategies** — Momentum, VAMS, Calmar, Sortino, EMA Crossover, and more
- **Realistic Cost Modeling** — IBKR-style commissions and slippage
- **40+ Rebalancing Frequencies** — From hourly to bi-annual

### 📈 Walk-Forward Optimization

- **True Walk-Forward (Default)** — Re-optimize each window and stitch OOS results
- **CV Mode Option** — Fixed-parameter time-series cross-validation when desired
- **Double-OOS Research Protocol** — Optimize WFO architecture on a global-train period, lock the selected setup, then validate on an unseen period
- **Dual Engines** — Optuna (Bayesian) or Genetic Algorithm
- **Early Stopping** — Automatically stops unpromising optimization runs
- **Resumable Studies** — Continue optimization sessions anytime

### 🎲 Monte Carlo Stress Testing

- **Block-Bootstrap Projection** — Preserves short-term return structure
- **Two-Stage Testing** — Projection (Stage 1) and post-optimization robustness (Stage 2)
- **Synthetic OHLC Injection** — Stress test with asset-level synthetic data

### 📊 Risk Management

- **ATR-Based Stop Loss** — Volatility-adjusted exit levels
- **ATR-Based Take Profit** — Dynamic profit targets
- **Daily Monitoring** — Independent of rebalancing schedule

### 📋 Comprehensive Reporting

- **Performance Metrics** — By default (after loading `config/parameters.yaml`), **Sharpe**, **Sortino**, and **Deflated Sharpe** use **excess returns** over the implied per-bar risk-free rate from the configured Treasury yield index (`risk_free_yield_ticker`, default `^IRX`), using simple split: annual yield % divided by `steps_per_year` per bar. Set `risk_free_metrics_enabled: false` in `GLOBAL_CONFIG` for legacy **annualized return / annualized volatility** Sharpe everywhere, or override per scenario in `extras` (see AGENTS.md). If yield data are missing or all-NaN for the window, those metrics fall back to the legacy path. (Deflated Sharpe stays NaN on single non-optimized runs when trial count is 1.) Rich console row naming is controlled by `metrics_display_profile` (`legacy` default, or `platform_standard` / `verbose`); CSV keeps canonical metric keys. See [docs/performance_metrics.md](docs/performance_metrics.md) for formulas, legacy vs excess Sharpe, **Tail Ratio** vs **Gain/Loss Mean Ratio**, ADF equity vs returns, and how the **Deflated Sharpe** column relates to PSR-style statistics.
- **Drawdown diagnostics** — Average drawdown episode length (peak to trough) and average recovery (trough back to prior high, or trough to last bar when still underwater)
- **Visual Reports** — Equity curves, drawdown charts, metric distributions
- **Parameter Analysis** — Sensitivity and importance rankings

---

## Setup

### Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/matdev83/portfolio-backtester.git
cd portfolio-backtester

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install the package
pip install -e .

# AutoGluon models are cached under .strategy_cache/autogluon_models/
```

**Market data (MDMP):** `market-data-multi-provider` is listed in `pyproject.toml` and installs with the project when published on the index your pip uses. If you develop against a **local clone** of MDMP (sibling directory), install it in the same environment first, then reinstall PB:

```bash
pip install -e ../market-data-multi-provider
pip install -e .
```

---

## Available Strategies

Portfolio Backtester includes 11+ production-ready strategies:

### Portfolio Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **Simple Momentum** | Ranks assets by recent returns, holds top performers | `lookback_period`, `hold_period`, `n_holdings` |
| **Sharpe Momentum** | Ranks by risk-adjusted momentum (Sharpe ratio) | `lookback_period`, `min_periods` |
| **Calmar Momentum** | Ranks by return-to-drawdown ratio | `lookback_period`, `hold_period` |
| **Sortino Momentum** | Ranks by downside-risk-adjusted returns | `lookback_period`, `target_return` |
| **AutoGluon Sortino ML** | ML model predicts long-only weights from rolling Sortino/relative/correlation features | `rebalance_frequency`, `label_lookback_days` |
| **VAMS (Volatility-Adjusted)** | Momentum with volatility scaling | `lookback_period`, `vol_lookback` |
| **Low Volatility Factor** | Selects lowest volatility assets | `lookback_period`, `n_holdings` |
| **Static Allocation** | Fixed-weight portfolio | `weights`, `rebalance_frequency` |
| **Volatility Targeted** | Maintains target portfolio volatility | `target_volatility`, `lookback` |

### Signal Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **EMA Crossover** | Classic moving average crossover signals | `fast_period`, `slow_period` |
| **Dummy Signal** | Template for custom strategy development | — |

### Meta Strategies

| Strategy | Description |
|----------|-------------|
| **Simple Meta** | Combines multiple strategies with weights |

**Finding Strategy Configurations:**

```
config/scenarios/builtins/
├── portfolio/           # Portfolio strategies
│   ├── simple_momentum_strategy/
│   ├── sharpe_momentum_portfolio_strategy/
│   ├── calmar_momentum_portfolio_strategy/
│   ├── autogluon_sortino_ml_portfolio_strategy/
│   └── ...
└── signal/              # Signal strategies
    ├── ema_crossover_signal_strategy/
    └── ...
```

---

## Usage

### Running a Backtest

```bash
python -m src.portfolio_backtester.backtester \
  --mode backtest \
  --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml"
```

For the AutoGluon Sortino ML strategy:

```bash
python -m src.portfolio_backtester.backtester \
  --mode backtest \
  --scenario-filename "config/scenarios/builtins/portfolio/autogluon_sortino_ml_portfolio_strategy/default.yaml"
```

Note: the default AutoGluon scenario starts at the first date when all
universe tickers have data (`min_universe_coverage: 1.0`). Add explicit
`start_date`/`end_date` if you want shorter runs.

AutoGluon models are pre-trained on a 5-year schedule by default
(`pretrain_models: true`, `retrain_interval_years: 5`) and reused across
rebalance dates to keep backtests and optimization runs practical. The
default feature set omits pairwise correlation features to reduce model
complexity and training time.

Labels for the AutoGluon model are generated from forward Sortino-optimized
weights. Use `label_horizons_days` and `label_horizon_weights` in the scenario
to adjust horizon selection. The current default scenario uses a single
3-month horizon and normalizes labels to 100% exposure. Volatility targeting
can then scale weights up or down (gross exposure can drift). Exposure penalty
is disabled by default.

### Universe Search (Approximate)

For quick universe experiments without retraining the model for each
combination, you can sample random subsets using the baseline model weights:

```bash
.venv\Scripts\python.exe scripts/universe_search.py \
  --scenario-filename "config/scenarios/builtins/portfolio/autogluon_sortino_ml_portfolio_strategy/default.yaml" \
  --subset-size 12 \
  --n-samples 25 \
  --metric Sortino
```

Results are saved under `data/reports/universe_search_<timestamp>/`.

The backtest will:

1. Load historical data for the configured universe
2. Generate signals according to the strategy
3. Simulate trades with realistic costs
4. Output a performance report to `data/reports/`

### Running Optimization

Find optimal strategy parameters using walk-forward optimization:

```bash
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-filename "config/scenarios/builtins/portfolio/sharpe_momentum_portfolio_strategy/default.yaml" \
  --optimizer optuna \
  --optuna-trials 200 \
  --n-jobs 4 \
  --study-name "sharpe_momentum_v1"
```

**Key Options:**

- `--optimizer optuna` — Bayesian optimization (recommended)
- `--optimizer genetic` — Genetic algorithm
- `--optuna-trials 200` — Number of parameter combinations to try
- `--n-jobs 4` — Parallel workers (use `-1` for all CPU cores)
- `--study-name "name"` — Save/resume optimization (stored in `data/optuna_studies.db`)

### Research Validation: Double-OOS WFO

Methodology (periods, grids, artifacts, CV/resume, limitations): see [`docs/research_validation.md`](docs/research_validation.md).

Use `research_validate` when you want a stricter research workflow than a normal optimization run. The protocol searches over walk-forward architecture choices on a global-training period, writes a lock file for the selected protocol, then validates once on a separate unseen period.

```bash
python -m src.portfolio_backtester.backtester \
  --mode research_validate \
  --scenario-filename "config/scenarios/builtins/portfolio/sharpe_momentum_portfolio_strategy/default.yaml" \
  --protocol double_oos_wfo \
  --optimizer optuna \
  --optuna-trials 100
```

Add a `research_protocol` block to the scenario YAML:

```yaml
research_protocol:
  enabled: true
  type: double_oos_wfo

  execution:
    max_grid_cells: 100  # unique expanded grid cells; raises ResearchProtocolConfigError if exceeded
    fail_fast: true      # reserved for future behavior; stored on the parsed config

  global_train_period:
    start_date: "2005-01-01"
    end_date: "2018-12-31"

  unseen_test_period:
    start_date: "2019-01-01"
    end_date: "2025-12-31"

  wfo_window_grid:
    train_window_months: [24, 36, 60, 84, 120]
    test_window_months: [3, 6, 12]
    wfo_step_months: [3, 6, 12]
    walk_forward_type: [rolling]

  selection:
    top_n: 3
    # Preferred: omit `metric` or set `metric: RobustComposite` for weighted rank selection.
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

  constraints:
    - metric: "Max Drawdown"
      min_value: -0.35
    - metric: "Turnover"
      max_value: 60
    - metric: "Years Positive %"
      min_value: 0.55

  final_unseen_mode: reoptimize_with_locked_architecture

  lock:
    enabled: true
    refuse_overwrite: true

  reporting:
    enabled: true

  # Optional: neighbor-smoothed robust score for ranking (eligible cells only).
  robust_selection:
    enabled: false
    weights:
      cell: 0.50
      neighbor_median: 0.30
      neighbor_min: 0.20
```

When `metric` is omitted or `RobustComposite`, a default `scoring` block matching the weights above is used if you omit `scoring`. For a single-metric selection (e.g. `metric: Calmar`), omit `scoring` entirely. Optional `constraints` require each grid architecture to satisfy metric bounds (inclusive min/max) before ranking; metric labels accept the same aliases as composite scoring. If every row fails constraints, the run raises `ResearchConstraintError`.

The Cartesian-expansion grid is deduplicated after expansion; when the number of unique architecture cells exceeds `research_protocol.execution.max_grid_cells` (default 100), the run raises `ResearchProtocolConfigError` before optimization starts (`fail_fast` is parsed and defaults to true but is only documented for future use). Small worked examples live under `config/scenarios/examples/research/`.

Optional `robust_selection` adjusts ranking using adjacent grid neighbors (same `walk_forward_type` and `wfo_step_months`, neighboring `train_window_months` / `test_window_months` on the sorted unique axes). Only constraint-passing rows participate in neighbor lookup; when `enabled` is false, `robust_score` in CSV/YAML still defaults to the raw cell score for convenience.

```yaml
  selection:
    top_n: 3
    metric: "Calmar"
```

Supported `final_unseen_mode` values:

| Mode | Behavior |
|------|----------|
| `reoptimize_with_locked_architecture` | Re-run WFO on the unseen period using the selected WFO architecture |
| `fixed_selected_params` | Run a single unseen backtest with selected parameters merged into `strategy_params` |

Research artifacts are written under `data/reports/<scenario>/research_protocol/<run_id>/` by default. Override the root with `--research-artifact-base-dir <path>` (the scenario-safe subdirectory layout is unchanged).

- `wfo_architecture_grid.csv` — Metrics and scores for each WFO architecture cell
- `selected_protocols.yaml` — Ranked selected architectures and parameters
- `protocol_lock.yaml` — Scenario, global config, and protocol hashes plus selected architecture
- `unseen_test_returns.csv` and `unseen_test_metrics.yaml` — Final unseen validation outputs
- `research_validation_report.md` — Human-readable protocol summary (canonical)
- `research_validation_report.html` — Optional HTML derivative when `research_protocol.reporting.generate_html: true`
- `bootstrap_significance.csv` and `bootstrap_summary.yaml` — Optional post-selection bootstrap summaries (see below)

Optional `bootstrap` runs **after** unseen validation and **after** cost sensitivity (when enabled). It does not affect architecture selection or ranking; if unseen validation is skipped or absent, bootstrap is skipped.

```yaml
  bootstrap:
    enabled: true
    n_samples: 200
    random_seed: 42
    random_wfo_architecture:
      enabled: true
    block_shuffled_returns:
      enabled: true
      block_size_days: 20
```

Use `--research-skip-unseen` to generate the grid/lock artifacts without running final unseen validation. Use `--force-new-research-run` to allow lock overwrite behavior for a fresh run.

### Monte Carlo Stress Testing

Test strategy robustness under synthetic market conditions:

```bash
python -m src.portfolio_backtester.backtester \
  --mode monte_carlo \
  --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml" \
  --mc-simulations 1000 \
  --mc-years 10 \
  --interactive
```

Stage 1 projections use block-bootstrap sampling of returns. Stage 2 stress tests
inject synthetic OHLC into the backtest inputs; tune block size via
`monte_carlo_config.stage2_block_size_days` in `config/parameters.yaml`.

---

## Configuration

All configuration is done through YAML files in the `config/` directory.

### Universe Configuration

Define which assets to include in your backtest:

**Single Symbol:**

```yaml
universe_config:
  type: single_symbol
  ticker: SPY
```

**Fixed List:**

```yaml
universe_config:
  type: fixed
  tickers: [SPY, QQQ, IWM, GLD, TLT]
```

**Named Universe:**

```yaml
universe_config:
  type: named
  universe_name: sp500_top50
```

**Dynamic (Method-Based):**

```yaml
universe_config:
  type: method
  method_name: get_top_weight_sp500_components
  n_holdings: 20
```

### Walk-Forward Optimization Settings

Control how the optimizer runs across windows:

```yaml
# True walk-forward (default)
wfo_mode: "reoptimize"  # or "cv" for fixed-parameter cross-validation
train_window_months: 60
test_window_months: 12
wfo_step_months: 12      # step between window starts (months)
wfo_embargo_bdays: 5     # optional embargo between train_end and test_start
walk_forward_type: "rolling"  # or "expanding"
```

For scientific validation of WFO architecture choices, use `--mode research_validate` with a scenario-level `research_protocol` block. This opt-in mode tries a grid of `train_window_months`, `test_window_months`, `wfo_step_months`, and `walk_forward_type` values on `global_train_period`, locks the selected protocol, then validates on `unseen_test_period`.

### Risk Management

Configure stop loss and take profit levels:

**ATR-Based Stop Loss & Take Profit:**

```yaml
strategy_params:
  stop_loss_config:
    type: "AtrBasedStopLoss"
    atr_length: 14
    atr_multiple: 2.0    # Exit at 2x ATR below entry

  take_profit_config:
    type: "AtrBasedTakeProfit"
    atr_length: 21
    atr_multiple: 3.0    # Exit at 3x ATR above entry
```

**Disable Risk Management:**

```yaml
strategy_params:
  stop_loss_config:
    type: "NoStopLoss"
  take_profit_config:
    type: "NoTakeProfit"
```

Risk management runs **daily**, independent of your rebalancing schedule.

### Commission and Slippage

Configure transaction costs in `config/parameters.yaml`:

```yaml
# IBKR-style commission model
commission_per_share: 0.005
commission_min_per_order: 1.0
commission_max_percent_of_trade: 0.005

# Slippage (1 bps = 0.01%)
slippage_bps: 2.5
```

### Capital Allocation Modes

Control how position sizes are calculated over time:

| Mode | Effect | Use Case |
|------|--------|----------|
| `reinvestment` (default) | Profits/losses affect future position sizes | Realistic trading simulation |
| `fixed_fractional` | Position sizes based on initial capital only | Fair strategy comparison |

```yaml
strategy_config:
  allocation_mode: "reinvestment"  # or "fixed_fractional"
```

### Timing Configuration

Control when signals are generated and when trades are executed:

**Time-Based Rebalancing (default):**
```yaml
timing_config:
  mode: time_based
  rebalance_frequency: ME  # Monthly end (see Rebalance Frequencies below)
  trade_execution_timing: bar_close  # or next_bar_open
```

**Signal-Based Scanning:**
```yaml
timing_config:
  mode: signal_based
  scan_frequency: D        # Daily scan for signals
  min_holding_period: 1    # Minimum bars to hold
  trade_execution_timing: bar_close  # or next_bar_open
```

**Trade Execution Timing:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `bar_close` (default) | Target weights take effect at the close of the signal bar | Standard backtesting behavior |
| `next_bar_open` | Target weights are deferred to the next trading session's open | More conservative execution modeling |

The `trade_execution_timing` setting works for **both** time-based portfolio strategies and signal-based strategies. When using `next_bar_open`, the backtester shifts the effective target weight date by one trading session, which means:
- The portfolio does not immediately gain exposure to the signal
- Turnover and transaction costs are calculated on the deferred execution date
- Returns from the signal bar are not captured until the next session

This is useful when you want to model realistic execution delays where you cannot trade on the same bar that generates the signal.

---

## Rebalance Frequencies

The backtester supports 40+ rebalancing frequencies:

| Category | Frequencies |
|----------|-------------|
| **Sub-daily** | `1h`, `2h`, `4h` |
| **Daily** | `daily`, `1d` |
| **Weekly** | `weekly`, `1w`, `2w` |
| **Monthly** | `monthly`, `1m`, `2m`, `quarterly`, `3m` |
| **Annual** | `semi-annual`, `6m`, `annual`, `yearly`, `1y` |

Configure in your scenario YAML:

```yaml
strategy_params:
  rebalance_frequency: "monthly"
```

---

## CLI Reference

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | `backtest`, `optimize`, or `research_validate` | Required |
| `--scenario-filename` | Path to scenario YAML | Required |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--mdmp-cache-only` | Use MDMP cached data only; skip downloads | Off |

### Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--optimizer` | `optuna` or `genetic` | `optuna` |
| `--optuna-trials` | Max trials per optimization | `200` |
| `--n-jobs` | Parallel workers (`-1` = all cores) | `8` |
| `--timeout` | Time budget in seconds | None |
| `--random-seed` | For reproducibility | None |
| `--study-name` | Name for resumable study | None |
| `--pruning-enabled` | Early stopping of bad trials | Off |
| `--early-stop-patience` | Stop after N non-improving trials | `10` |

### Research Validation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--protocol` | Research protocol for `research_validate` mode | `double_oos_wfo` |
| `--research-artifact-base-dir` | Root directory for research artifacts (instead of `data/reports`) | None |
| `--research-skip-unseen` | Skip final unseen validation after grid selection/lock artifacts | Off |
| `--force-new-research-run` | Allow a fresh research run when lock artifacts would otherwise block overwrite | Off |

### Monte Carlo Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mc-simulations` | Number of simulations | `1000` |
| `--mc-years` | Years to project | `10` |
| `--interactive` | Show plots interactively | Off |

---

## Example Outputs

### Backtest Report

After running a backtest, you'll find a report in `data/reports/` containing:

- **Performance Summary** — Total return, Sharpe ratio, max drawdown
- **Equity Curve** — Portfolio value over time
- **Drawdown Chart** — Underwater equity visualization  
- **Monthly Returns Heatmap** — Performance by month and year
- **Trade Log** — Detailed transaction history

### Optimization Results

Optimization produces:

- **Best Parameters** — Consensus parameters (or CV best)
- **Per-Window Parameters** — When `wfo_mode: reoptimize`
- **Trial History** — Performance of each tested combination
- **Parameter Importance** — Which parameters matter most
- **Walk-Forward Validation** — Stitched out-of-sample performance estimates

### Research Validation Outputs

`research_validate` produces a reproducible protocol bundle under `data/reports/<scenario>/research_protocol/<run_id>/`:

- **WFO Architecture Grid** — `wfo_architecture_grid.csv` with metrics/scores for each window architecture
- **Selected Protocols** — `selected_protocols.yaml` with ranked architectures and selected parameters
- **Protocol Lock** — `protocol_lock.yaml` with scenario/config/protocol hashes for reproducibility
- **Unseen Validation** — Returns and metrics for the locked final validation period
- **Research Report** — `research_validation_report.md` summarizing the protocol and unseen result (optional `research_validation_report.html` via `reporting.generate_html`)

### Dual Momentum Top-N Proof Runs

The `DualMomentumLaggedPortfolioStrategy` has two useful S&P 500 Top-N scenarios with different goals:

| Scenario | Purpose | Key Settings | Cache-only Proof Result |
|----------|---------|--------------|-------------------------|
| `config/scenarios/builtins/portfolio/dual_momentum_lagged_portfolio_strategy/default.yaml` | Risk-managed production default | 12-month lookback, 1-month entry confirmation lag, 200SMA market filter, residual momentum ranking, volatility targeting | Backtest total return `195.82%` vs SPX `324.29%` |
| `config/scenarios/builtins/portfolio/dual_momentum_lagged_portfolio_strategy/canonical_topn_momentum.yaml` | Aggressive canonical-style momentum sleeve | 6-month lookback, no entry lag, no 200SMA gate, no volatility target, excess total-return ranking | Backtest total return `688.77%` vs SPX `324.29%` |
| `canonical_topn_momentum` with `momentum_skip_months: 1` | Academic-style skip-month momentum | 6-month lookback, skip 1 month in formation window, no entry lag, excess total-return ranking | Backtest total return `559.45%` vs SPX `324.29%` |

Both proof runs used MDMP cache-only data with `cache_max_age_seconds: 604800`, fetched `48/49` requested symbols, and derived `47` dynamic universe tickers from prefetched OHLC columns. This validates the dynamic point-in-time S&P 500 Top-N flow without relying on live downloads.

Run the canonical-style proof backtest:

```bash
python -m src.portfolio_backtester.backtester   --mode backtest   --scenario-filename config/scenarios/builtins/portfolio/dual_momentum_lagged_portfolio_strategy/canonical_topn_momentum.yaml   --mdmp-cache-only   --log-level INFO
```

Latest cache-only canonical proof metrics:

| Metric | Strategy | SPX |
|--------|----------|-----|
| Total Return | `688.77%` | `324.29%` |
| Ann. Return | `18.02%` | `12.29%` |
| Sharpe | `0.8939` | `0.7129` |
| Sortino | `1.3205` | `1.0656` |
| Max Drawdown | `-31.22%` | `-33.92%` |
| Beta | `0.9656` | `1.0000` |

A 4-trial optimization smoke run on the canonical-style scenario completed without errors and produced stitched out-of-sample total return `357.09%` vs SPX `324.29%`:

```bash
python -m src.portfolio_backtester.backtester   --mode optimize   --scenario-filename config/scenarios/builtins/portfolio/dual_momentum_lagged_portfolio_strategy/canonical_topn_momentum.yaml   --mdmp-cache-only   --log-level INFO   --optuna-trials 4   --n-jobs 1   --random-seed 123
```

Notes:

- `lag_months` is an entry-confirmation lag, not the same as an academic skip-month momentum definition.
- `momentum_skip_months` skips the most recent months in the formation window (e.g. skip 1 = use prices ending one month before the rebalance date). This is closer to Jegadeesh & Titman-style momentum.
- These proof runs use the available MDMP `Close` series, not dividend-adjusted or total-return constituent data, so paper-to-paper comparisons should account for data construction differences.
- The canonical-style scenario is higher-return and less risk-managed than the production default by design.

---

## Documentation

For MDMP boundaries and canonical data rules, see [Data Sources](#data-sources) below.

Example configurations are available in:

- `config/scenarios/examples/` — Risk management examples, allocation modes
- `config/scenarios/builtins/` — Built-in strategy defaults

---

## Data Sources

OHLCV and SPY holdings history are fetched through **market-data-multi-provider (MDMP)**. Provider selection, failover, and on-disk market-data caching live in MDMP; Portfolio Backtester does not write canonical OHLCV or holdings parquet itself.

Architectural tests under `tests/unit/architecture/` block new direct vendor imports (for example `yfinance`) and any `market_data_multi_provider` references outside `src/portfolio_backtester/data_sources/mdmp_facade.py`.

**Strict local-storage checklist (PB repo)**

| Area | Rule |
|------|------|
| Canonical OHLCV / SPY holdings history | MDMP only; no `to_parquet` / `read_parquet` under `src/portfolio_backtester/` |
| Direct vendor fetch clients | Forbidden imports (see `tests/unit/architecture/test_mdmp_boundary.py`) |
| `data/reports`, `data/pnl_charts`, `data/optuna/` | Run outputs / Optuna DB defaults only (gitignored); not canonical market-data roots |
| `data/config_validation_cache.json` | Config validation cache only (not market series) |
| `.gitignore` `data/*.parquet`, `data/yfinance/`, etc. | Safety net so local MDMP mirrors or legacy folders are never committed |

---

## Synthetic Data Quality Testing

Validate that Monte Carlo simulations use realistic synthetic data:

```bash
python scripts/test_synthetic_data.py AAPL --paths 5 --period 2y
```

This compares statistical properties (volatility, skewness, kurtosis) between real and synthetic data.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No data for ticker | Check if the ticker symbol is correct; try a major ETF like SPY first |
| Optimization returns all zeros | Check `--early-stop-zero-trials 5` to detect data issues faster |
| Slow optimization | Increase `--n-jobs` or reduce `--optuna-trials` |
| Can't resume study | Ensure you use the same `--study-name` |

### Getting Help

1. Check the example scenarios in `config/scenarios/builtins/`
2. Review the [Contributing Guide](CONTRIBUTING.md)
3. Open an issue on GitHub

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

For AI agents and automated contributors, see [AGENTS.md](AGENTS.md).

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

See [LICENSE.md](LICENSE.md) for full details.

---

*Built for traders and quants who want rigorous backtesting without the complexity.*
