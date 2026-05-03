import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ..numba_optimized import compensated_kurtosis_fast, compensated_skew_fast

try:
    _HAS_NUMBA_MOMENTS = True
except Exception:
    _HAS_NUMBA_MOMENTS = False


logger = logging.getLogger(__name__)

EPSILON = 1e-9

__all__ = (
    "calculate_metrics",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_max_drawdown",
    "compensated_skew_fast",
    "compensated_kurtosis_fast",
)


def _infer_steps_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 252
    freq = pd.infer_freq(index)
    if freq:
        if freq.startswith("B") or freq == "D":
            return 252
        if freq.startswith("W"):
            return 52
        if freq.startswith("M"):
            return 12
    return 252


def _calculate_capm(rets_aligned, bench_aligned, bench_ticker_name, steps_per_year):
    # Add defensive checks for empty Series before length comparisons
    if rets_aligned.empty or bench_aligned.empty or len(rets_aligned) < 2:
        return np.nan, np.nan, np.nan

    # Use proper Series handling for standard deviation comparison
    bench_std = bench_aligned.std()
    # Handle case where bench_std might be a Series - convert to scalar safely
    if isinstance(bench_std, pd.Series):
        if bench_std.empty:
            return np.nan, np.nan, np.nan
        # Use the first value if it's a Series
        bench_std_scalar = bench_std.iloc[0] if len(bench_std) > 0 else np.nan
    else:
        bench_std_scalar = bench_std

    if pd.isna(bench_std_scalar) or abs(bench_std_scalar) < EPSILON:
        return np.nan, np.nan, np.nan

    X = sm.add_constant(bench_aligned)
    try:
        model = sm.OLS(rets_aligned, X).fit()
        alpha = model.params.get("const", np.nan) * steps_per_year
        beta = model.params.get(bench_ticker_name, np.nan)
        r_squared = model.rsquared
        return alpha, beta, r_squared
    except Exception:
        return np.nan, np.nan, np.nan


def calculate_sharpe(rets, steps_per_year):
    if rets.empty:
        return np.nan
    ann_return = (1 + rets).prod() ** (steps_per_year / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(steps_per_year)
    return ann_return / ann_vol if ann_vol > EPSILON else np.inf if ann_return > 0 else -np.inf


def calculate_sortino(rets, steps_per_year, target=0):
    if rets.empty:
        return np.nan
    target_returns = rets - target
    downside_risk = np.sqrt(np.mean(np.minimum(0, target_returns) ** 2))
    ann_return = rets.mean() * steps_per_year
    ann_downside_risk = downside_risk * np.sqrt(steps_per_year)
    return (
        ann_return / ann_downside_risk
        if ann_downside_risk > EPSILON
        else np.inf if ann_return > 0 else -np.inf
    )


def calculate_max_drawdown(equity_curve):
    if equity_curve.empty:
        return np.nan
    return (equity_curve / equity_curve.cummax() - 1).min()


def calculate_metrics(
    rets,
    bench_rets,
    bench_ticker_name,
    name="Strategy",
    num_trials=1,
    risk_free_rets=None,
):
    """Delegate to :mod:`performance_metrics` for a single canonical implementation."""
    from .performance_metrics import calculate_metrics as full_calculate_metrics

    return full_calculate_metrics(
        rets,
        bench_rets,
        bench_ticker_name,
        name=name,
        num_trials=num_trials,
        trade_stats=None,
        risk_free_rets=risk_free_rets,
    )
