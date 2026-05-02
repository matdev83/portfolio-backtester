"""NumPy-forward objective metrics for optimizer aggregation (parity-focused)."""

from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, linregress, norm, skew
from statsmodels.tsa.stattools import adfuller

from ..numba_optimized import drawdown_duration_and_recovery_fast

from .performance_metrics import (
    EPSILON_FOR_DIVISION,
    _capm_on_zeros,
    _default_zero_activity_metrics,
    _ensure_expected_keys,
    _infer_steps_per_year,
    _is_all_zero,
    _safe_moment,
    calculate_max_dd_recovery_time,
    calculate_max_flat_period,
)


def _total_ret_np(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.prod(1.0 + x) - 1.0)


def _ann_np(x: np.ndarray, steps_per_year: int) -> float:
    if x.size == 0:
        return float("nan")
    prod = float(np.prod(1.0 + x))
    if prod < 0:
        return -1.0
    return float(prod ** (steps_per_year / float(x.size)) - 1.0)


def _ann_vol_np(x: np.ndarray, steps_per_year: int) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.std(x, ddof=1) * np.sqrt(float(steps_per_year)))


def _sortino_np(x: np.ndarray, steps_per_year: int) -> float:
    if x.size == 0:
        return float("nan")
    target_returns = x - 0.0
    downside_risk = float(np.sqrt(np.mean(np.minimum(0.0, target_returns) ** 2)))
    annualized_mean_return = float(np.mean(x) * float(steps_per_year))
    if np.isnan(annualized_mean_return) or np.isnan(downside_risk):
        return float("nan")
    if downside_risk < EPSILON_FOR_DIVISION:
        if abs(annualized_mean_return) < EPSILON_FOR_DIVISION:
            return 0.0
        if annualized_mean_return > 0:
            return float("inf")
        return float("-inf")
    annualized_downside_risk = downside_risk * np.sqrt(float(steps_per_year))
    if annualized_downside_risk < EPSILON_FOR_DIVISION:
        if abs(annualized_mean_return) < EPSILON_FOR_DIVISION:
            return 0.0
        if annualized_mean_return > 0:
            return float("inf")
        return float("-inf")
    return float(annualized_mean_return / annualized_downside_risk)


def _sharpe_np(x: np.ndarray, steps_per_year: int) -> float:
    if x.size == 0:
        return float("nan")
    annualized_return = _ann_np(x, steps_per_year)
    annualized_vol = _ann_vol_np(x, steps_per_year)
    if np.isnan(annualized_return) or np.isnan(annualized_vol):
        return float("nan")
    if annualized_vol < EPSILON_FOR_DIVISION:
        if abs(annualized_return) < EPSILON_FOR_DIVISION:
            return 0.0
        if annualized_return > 0:
            return float("inf")
        return float("-inf")
    return float(annualized_return / annualized_vol)


def _mdd_equity(equity: pd.Series) -> float:
    if equity.empty or equity.isnull().all():
        return float("nan")
    cummax_series = equity.cummax()
    drawdown_series = equity / cummax_series - 1.0
    return float(drawdown_series.min())


def _calmar_np(active: np.ndarray, full_rets: pd.Series, steps_per_year: int) -> float:
    if active.size == 0:
        return float("nan")
    max_dd = _mdd_equity((1 + full_rets).cumprod())
    annualized_return = _ann_np(active, steps_per_year)
    if np.isnan(annualized_return) or np.isnan(max_dd):
        return float("nan")
    if abs(max_dd) < EPSILON_FOR_DIVISION:
        if abs(annualized_return) < EPSILON_FOR_DIVISION:
            return 0.0
        if annualized_return > 0:
            return float("inf")
        return float("-inf")
    return float(annualized_return / abs(max_dd))


def _stationarity_np(active: pd.Series) -> tuple[float, float]:
    if len(active) < 40:
        return float("nan"), float("nan")
    cumulative_pnl: pd.Series = (1 + active).cumprod()
    try:
        result = adfuller(cumulative_pnl)
        return float(result[0]), float(result[1])
    except Exception:
        return float("nan"), float("nan")


def _deflated_sharpe_np(active: pd.Series, num_trials: int, steps_per_year: int) -> float:
    if num_trials <= 1 or len(active) < 100:
        return float("nan")
    arr = active.to_numpy(dtype=np.float64, copy=False)
    sr = _sharpe_np(arr, steps_per_year)
    if np.isnan(sr):
        return float("nan")
    std_rets = float(active.std(ddof=0)) if not pd.isna(active.std(ddof=0)) else 0.0
    if std_rets <= EPSILON_FOR_DIVISION:
        return float("nan")
    sk = _safe_moment(skew, active)
    k_excess = _safe_moment(kurtosis, active)
    n = len(active)
    emc = 0.5772156649
    max_z = (1 - emc) * norm.ppf(1 - 1 / num_trials) + emc * norm.ppf(1 - 1 / (num_trials * np.e))
    var_sr = (1 / (n - 1)) * (1 - sk * sr + (k_excess / 4) * sr**2)
    if var_sr < 0 or n - 1 <= 0:
        return float("nan")
    std_sr_h0 = np.sqrt(1 / (n - 1))
    expected_max_sr_h0 = std_sr_h0 * max_z
    if var_sr <= 0:
        return float("nan")
    std_dev_sr_observed = np.sqrt(var_sr)
    if std_dev_sr_observed <= EPSILON_FOR_DIVISION or pd.isna(std_dev_sr_observed):
        return float("nan")
    dsr_z = (sr - expected_max_sr_h0) / std_dev_sr_observed
    return float(norm.cdf(dsr_z))


def _var_np(active: pd.Series, confidence_level: float = 0.05) -> float:
    if active.empty or active.isnull().all():
        return float("nan")
    return float(np.percentile(active.to_numpy(dtype=float, copy=False), confidence_level * 100))


def _cvar_np(active: pd.Series, confidence_level: float = 0.05) -> float:
    if active.empty or active.isnull().all():
        return float("nan")
    var = _var_np(active, confidence_level)
    if np.isnan(var):
        return float("nan")
    tail = active[active <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def _k_ratio_np(active: pd.Series) -> float:
    if len(active) < 2:
        return float("nan")
    equity_for_log = (1 + active.astype(float)).cumprod()
    equity_positive = equity_for_log.clip(lower=float(np.finfo(float).tiny))
    with np.errstate(invalid="ignore", divide="ignore"):
        log_equity = np.log(equity_positive.to_numpy(dtype=float, copy=False))
    log_equity_std_val = np.nanstd(log_equity)
    log_equity_std = float(log_equity_std_val) if np.isfinite(log_equity_std_val) else 0.0
    if log_equity_std < EPSILON_FOR_DIVISION:
        return float("nan")
    idx = np.arange(len(log_equity))
    slope, _intercept, _r, _p, std_err = linregress(idx, log_equity)

    def safe_float(val: object) -> float:
        try:
            if isinstance(val, (tuple, list, np.ndarray)):
                val = val[0]
            return float(cast(Any, val))
        except Exception:
            return float("nan")

    std_err_val = safe_float(std_err)
    slope_val = safe_float(slope)
    if std_err_val < EPSILON_FOR_DIVISION:
        return float("nan")
    return float((slope_val / std_err_val) * np.sqrt(len(log_equity)))


def calculate_optimizer_metrics_fast(
    rets: pd.Series,
    bench_rets: pd.Series,
    bench_ticker_name: str,
    name: str = "Strategy",
    num_trials: int = 1,
    trade_stats: Mapping[str, object] | None = None,
) -> pd.Series:
    """Match :func:`calculate_metrics` outputs while using NumPy for the objective core."""

    is_all_zero_returns = _is_all_zero(rets)
    active_rets = rets.dropna()
    active_bench_rets = bench_rets.dropna()

    if active_rets.empty:
        metrics = _default_zero_activity_metrics(is_all_zero_returns)
        if not active_bench_rets.empty and is_all_zero_returns:
            metrics.update(_capm_on_zeros(rets, active_bench_rets, bench_ticker_name))
        return cast(pd.Series, pd.Series(_ensure_expected_keys(metrics), name=name))

    common_index = pd.DatetimeIndex(active_rets.index).intersection(
        pd.DatetimeIndex(active_bench_rets.index)
    )
    rets_aligned = active_rets.loc[common_index]
    bench_aligned = active_bench_rets.loc[common_index]
    steps_per_year = _infer_steps_per_year(common_index)

    try:
        bench_std_val = bench_aligned.std()
        bench_std = float(bench_std_val) if pd.notna(bench_std_val) else 0.0
    except Exception:
        bench_std = 0.0

    try:
        rets_std_val = rets_aligned.std()
        rets_std = float(rets_std_val) if pd.notna(rets_std_val) else 0.0
    except Exception:
        rets_std = 0.0

    if (
        len(rets_aligned) < 2
        or len(bench_aligned) < 2
        or abs(bench_std) < EPSILON_FOR_DIVISION
        or abs(rets_std) < EPSILON_FOR_DIVISION
    ):
        alpha = float("nan")
        beta = float("nan")
        r_squared = float("nan")
    else:
        bench_aligned_named = bench_aligned.copy()
        bench_aligned_named.name = bench_ticker_name
        x = sm.add_constant(bench_aligned_named)
        try:
            capm = sm.OLS(rets_aligned, x).fit()
            alpha = float(capm.params.get("const", np.nan) * steps_per_year)
            beta = float(capm.params.get(bench_ticker_name, np.nan))
            r_squared = float(capm.rsquared)
        except Exception:
            alpha = float("nan")
            beta = float("nan")
            r_squared = float("nan")

    active_vals = active_rets.to_numpy(dtype=np.float64, copy=False)
    total_return = _total_ret_np(active_vals)
    ann_ret = _ann_np(active_vals, steps_per_year)
    ann_vol = _ann_vol_np(active_vals, steps_per_year)
    sharpe_v = _sharpe_np(active_vals, steps_per_year)
    sortino_v = _sortino_np(active_vals, steps_per_year)
    calmar_v = _calmar_np(active_vals, rets, steps_per_year)
    max_dd_v = _mdd_equity((1 + rets).cumprod())

    adf_stat, adf_p = _stationarity_np(active_rets)
    var_5 = _var_np(active_rets, 0.05)
    cvar_5 = _cvar_np(active_rets, 0.05)

    def tail_ratio(rets_s: pd.Series, percentile: int = 95) -> float:
        if rets_s.empty or rets_s.isnull().all():
            return float("nan")
        positive_rets = rets_s[rets_s > 0]
        negative_rets = rets_s[rets_s < 0]
        if positive_rets.empty or negative_rets.empty:
            return float("nan")
        upper_tail = float(np.percentile(positive_rets, percentile))
        lower_tail = float(np.percentile(negative_rets, 100 - percentile))
        if abs(lower_tail) < EPSILON_FOR_DIVISION:
            return float("inf") if upper_tail > 0 else float("nan")
        return float(upper_tail / abs(lower_tail))

    tail_ratio_95 = tail_ratio(active_rets, 95)

    equity_curve = (1 + rets).cumprod()
    try:
        avg_dd_duration, avg_recovery_time = drawdown_duration_and_recovery_fast(
            equity_curve.values.astype(np.float64)
        )
    except Exception:
        avg_dd_duration, avg_recovery_time = float("nan"), float("nan")

    base_metrics: dict[str, float | int] = {
        "Total Return": total_return,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe_v,
        "Sortino": sortino_v,
        "Calmar": calmar_v,
        "Alpha (ann)": alpha,
        "Beta": beta,
        "Max Drawdown": max_dd_v,
        "VaR (5%)": var_5,
        "CVaR (5%)": cvar_5,
        "Tail Ratio": tail_ratio_95,
        "Avg DD Duration": float(avg_dd_duration),
        "Avg Recovery Time": float(avg_recovery_time),
        "Skew": float(_safe_moment(skew, active_rets)),
        "Kurtosis": float(_safe_moment(kurtosis, active_rets)),
        "R^2": r_squared,
        "K-Ratio": float(_k_ratio_np(active_rets)),
        "ADF Statistic": adf_stat,
        "ADF p-value": adf_p,
        "Deflated Sharpe": float(_deflated_sharpe_np(active_rets, num_trials, steps_per_year)),
    }

    if trade_stats is not None:
        from .performance_metrics import calculate_metrics

        trade_augmented = calculate_metrics(
            rets,
            bench_rets,
            bench_ticker_name,
            name=name,
            num_trials=num_trials,
            trade_stats=trade_stats,
        )
        for k, v in trade_augmented.items():
            if k not in base_metrics:
                try:
                    base_metrics[k] = float(v) if not pd.isna(v) else float("nan")
                except Exception:
                    base_metrics[k] = float("nan")

    base_metrics["Max DD Recovery Time (days)"] = float(calculate_max_dd_recovery_time(rets))
    base_metrics["Max Flat Period (days)"] = float(calculate_max_flat_period(rets))

    return cast(pd.Series, pd.Series(_ensure_expected_keys(base_metrics), name=name))
