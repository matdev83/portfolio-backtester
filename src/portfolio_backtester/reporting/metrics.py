import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, linregress, norm, skew
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

EPSILON = 1e-9

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
    if len(rets_aligned) < 2 or abs(bench_aligned.std()) < EPSILON:
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
    return ann_return / ann_downside_risk if ann_downside_risk > EPSILON else np.inf if ann_return > 0 else -np.inf

def calculate_max_drawdown(equity_curve):
    if equity_curve.empty:
        return np.nan
    return (equity_curve / equity_curve.cummax() - 1).min()

def calculate_metrics(rets, bench_rets, bench_ticker_name, name="Strategy", num_trials=1):
    if rets.empty or rets.abs().max() < EPSILON:
        return pd.Series(np.nan, name=name)

    steps_per_year = _infer_steps_per_year(rets.index)
    equity_curve = (1 + rets).cumprod()
    
    common_index = rets.index.intersection(bench_rets.index)
    rets_aligned, bench_aligned = rets.loc[common_index], bench_rets.loc[common_index]
    
    alpha, beta, r_squared = _calculate_capm(rets_aligned, bench_aligned, bench_ticker_name, steps_per_year)

    metrics = {
        "Total Return": (1 + rets).prod() - 1,
        "Ann. Return": (1 + rets).prod() ** (steps_per_year / len(rets)) - 1,
        "Ann. Vol": rets.std() * np.sqrt(steps_per_year),
        "Sharpe": calculate_sharpe(rets, steps_per_year),
        "Sortino": calculate_sortino(rets, steps_per_year),
        "Calmar": ((1 + rets).prod() ** (steps_per_year / len(rets)) - 1) / abs(calculate_max_drawdown(equity_curve)),
        "Alpha (ann)": alpha,
        "Beta": beta,
        "R^2": r_squared,
        "Max Drawdown": calculate_max_drawdown(equity_curve),
        "Skew": skew(rets),
        "Kurtosis": kurtosis(rets),
    }
    return pd.Series(metrics, name=name)