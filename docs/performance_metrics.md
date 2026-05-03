# Strategy performance metrics

This document describes how the backtester and reports compute the metrics in the performance summary tables. The canonical implementation is `calculate_metrics` in `src/portfolio_backtester/reporting/performance_metrics.py`. Optimizer code may use a NumPy mirror in `fast_objective_metrics.py` for parity and speed.

Configuration for risk-free–based metrics is covered in [AGENTS.md](../AGENTS.md) (Risk-free metrics). Global defaults come from `config/parameters.yaml` after `load_config()`.

## Sharpe ratio: risk-free path vs legacy path

**Risk-free path (default when enabled and yield data exist)**  
The series of **excess returns** (strategy return minus implied per-bar risk-free return, aligned to the strategy index) is used. Sharpe is:

- Annualized mean excess: `mean(excess) * steps_per_year`
- Annualized volatility of excess: `std(excess, ddof=1) * sqrt(steps_per_year)`
- **Sharpe** = (annualized mean excess) / (annualized vol of excess)

This matches the usual textbook **excess-return Sharpe** (per-period means scaled to a year, paired with the same series’ sample volatility).

**Legacy path (risk-free disabled, missing yield, or all-NaN risk-free series)**  
**Ann. Return** is computed with a **geometric** annualization (`prod(1+r)^(steps/n) - 1`), and **Ann. Vol** uses the sample standard deviation of returns annualized. **Sharpe** is **Ann. Return / Ann. Vol** — i.e. **geometric annualized return divided by annualized volatility**, not `(mean * steps - rf) / (vol)`.  

When comparing to Bloomberg, vendor terminals, or papers that use arithmetic mean excess over volatility, reconcile definitions explicitly; the legacy Sharpe is **not** identical to those formulas.

**Annualized return vs Sortino numerator**  
**Ann. Return** uses geometric annualization. **Sortino** uses an **arithmetic** annualized mean (`mean(r) * steps_per_year`) in the numerator. That is intentional: Sortino’s denominator (downside deviation) is defined from the same per-period returns; the table can show geometric **Ann. Return** next to a Sortino that is not derived from that same geometric number.

## Sortino ratio

- **Minimum acceptable return (MAR)** is **0** (or 0 on excess returns when the risk-free path is active).
- **Downside deviation** uses the root mean of squared negative deviations from MAR: `sqrt(mean(min(0, r - MAR)^2))` over all periods in the window, then annualized as `downside * sqrt(steps_per_year)`.
- **Sortino** = (annualized arithmetic mean of `r`) / (annualized downside deviation).

## Tail ratio (custom definition)

The label **Tail Ratio** is **not** the common “average gain / average loss” ratio.

Implementation: take only **strictly positive** daily returns and their **95th percentile**; take only **strictly negative** returns and their **5th percentile** (lower tail of losses). **Tail Ratio** = (95th percentile of positives) / abs(5th percentile of negatives).

If either subset is empty, or the lower tail is effectively zero, the ratio can be NaN or infinite per guards in code.

## “Deflated Sharpe” vs Probabilistic Sharpe (PSR)

The report column **Deflated Sharpe** is computed from:

- The observed Sharpe (excess path when risk-free metrics apply),
- Skew and excess kurtosis of the return series used for the statistic,
- An adjustment that depends on **`num_trials`** (optimization trials),
- A normal CDF applied to a z-style statistic.

That output is **PSR-like** (a probability that the true Sharpe exceeds a hurdle under a particular statistical model). It is **not** the same object as the **Deflated Sharpe Ratio (DSR)** in Bailey & López de Prado’s paper, which deflates for selection bias in a different way.

**Practical notes:**

- If **`num_trials <= 1`** or the series is shorter than the internal minimum length, the value is **NaN**. Benchmark-only columns in Rich tables often pass **`num_trials=1`**, so the benchmark’s **Deflated Sharpe** is typically **NaN** even when the strategy column has a value after optimization.
- For apples-to-apples comparison with external **DSR** write-ups, treat this column as a **probabilistic / adjusted significance** style metric unless you replace the implementation.

## Other interpretation notes

| Topic | Note |
|--------|------|
| **Avg DD Duration / Avg Recovery Time** | Computed in **bars** (periods) from the equity curve, not calendar days, unless your bar spacing is one trading day with no gaps. |
| **VaR / CVaR (5%)** | Historical simulation on the **empirical** return distribution; VaR is the 5th **percentile** (signed like returns). |

## Metrics display profile (Rich console labels)

`GLOBAL_CONFIG.metrics_display_profile` (default **`legacy`**) and optional override in scenario **`extras.metrics_display_profile`** control **row labels in Rich performance tables only**. CSV `performance_metrics.csv` rows keep **canonical metric keys** (e.g. `Sharpe`, `ADF Statistic (equity)`) for stable parsing.

| Profile | Behavior |
|-----------|-----------|
| **`legacy`** | Original row names; equity ADF keys display as `ADF Statistic` / `ADF p-value`. |
| **`platform_standard`** | Clearer labels (e.g. `Sharpe (CAGR/vol)` when the legacy Sharpe path is used, `Tail Ratio (pctile)`, `PSR (trial-adjusted)`, drawdown durations annotated as bars or calendar days). |
| **`verbose`** | Same labels as **`platform_standard`** (reserved for future extra rows). |

## Tail ratios: percentile vs mean

- **`Tail Ratio`** (percentile): same as the historic **Tail Ratio** definition above (95th pct of wins vs 5th pct of losses among positives/negatives).
- **`Tail Ratio (mean)`** / **`Gain/Loss Mean Ratio`** (display): `mean(positive days) / abs(mean(negative days))`, a common platform-style gain/loss ratio.

## ADF tests

- **`ADF Statistic (equity)` / `ADF p-value (equity)`**: ADF on the cumulative equity curve (levels), same idea as the former single **ADF Statistic** row.
- **`ADF Statistic (returns)` / `ADF p-value (returns)`**: ADF on per-period **returns** (stationarity of the return series).

## Max drawdown recovery

- **`Max DD Recovery Time (days)`**: longest recovery span measured in **calendar days** between index timestamps (trading-day-only calendars shorten gaps versus real time).
- **`Max DD Recovery Time (bars)`**: same episodes counted in **observation bars**, comparable to **Avg DD Duration** / **Avg Recovery Time**.

## Secondary evaluator metrics

`optimization/evaluator.py` includes a lightweight `_calculate_metrics` used in some evaluator flows; it uses simplified annualization (e.g. `(1 + mean)^252 - 1` style for daily assumptions). **Reporting tables** use `calculate_metrics` in `performance_metrics.py`, not that helper.
