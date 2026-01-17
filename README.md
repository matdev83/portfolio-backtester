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

- **Robust Parameter Finding** — Time-series cross-validation prevents overfitting
- **Dual Engines** — Optuna (Bayesian) or Genetic Algorithm
- **Early Stopping** — Automatically stops unpromising optimization runs
- **Resumable Studies** — Continue optimization sessions anytime

### 🎲 Monte Carlo Stress Testing

- **GARCH-Based Synthetic Data** — Realistic market simulations
- **Two-Stage Testing** — During optimization and post-optimization
- **Robustness Analysis** — Test your strategy under various market conditions

### 📊 Risk Management

- **ATR-Based Stop Loss** — Volatility-adjusted exit levels
- **ATR-Based Take Profit** — Dynamic profit targets
- **Daily Monitoring** — Independent of rebalancing schedule

### 📋 Comprehensive Reporting

- **Performance Metrics** — Sharpe, Sortino, Calmar ratios, drawdowns
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
| `--mode` | `backtest`, `optimize`, or `monte_carlo` | Required |
| `--scenario-filename` | Path to scenario YAML | Required |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

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

- **Best Parameters** — Optimal values found
- **Trial History** — Performance of each tested combination
- **Parameter Importance** — Which parameters matter most
- **Walk-Forward Validation** — Out-of-sample performance estimates

---

## Documentation

For detailed configuration guides, see:

- [S&P 500 Universe Data Management](docs/sp500_universe_management.md)

Example configurations are available in:

- `config/scenarios/examples/` — Risk management examples, allocation modes
- `config/scenarios/builtins/` — Built-in strategy defaults

---

## Data Sources

Portfolio Backtester automatically fetches market data from multiple sources with failover:

1. **Stooq** (primary) — Free historical data
2. **yfinance** (fallback) — Yahoo Finance data

Data is cached locally to speed up repeated backtests.

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
