import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm, linregress
import scipy._lib._util as sp_util

if not hasattr(sp_util, "_lazywhere"):
    def _lazywhere(cond, arrays, func, fillvalue=np.nan, out=None):
        arrays = tuple(np.asarray(a) for a in arrays)
        if out is None:
            out = np.full_like(arrays[0], fillvalue, dtype=float)
        mask = cond(*arrays)
        if mask.any():
            out[mask] = func(*[a[mask] for a in arrays])
        return out

    sp_util._lazywhere = _lazywhere

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# -----------------------------------------------------------------------------
# Helper to derive the appropriate annualisation factor (steps per year) from
# the return series' timestamp index.  Falls back to a daily assumption if the
# frequency cannot be determined automatically.
# -----------------------------------------------------------------------------


def _infer_steps_per_year(idxs: pd.DatetimeIndex) -> int:
    """Return 252 for daily, 52 for weekly, 12 for monthly data."""
    if len(idxs) < 2:
        return 252  # default – avoid division-by-zero downstream

    freq = pd.infer_freq(idxs)
    if freq is None:
        # Heuristic based on median spacing in days
        spacing = (idxs[-1] - idxs[0]).days / max(len(idxs) - 1, 1)
        if spacing <= 3:
            return 252  # ~daily
        if spacing <= 10:
            return 52   # ~weekly
        return 12       # monthly or sparser

    if freq.startswith("B") or freq in {"D"}:
        return 252
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    return 252


EPSILON_FOR_DIVISION = 1e-9  # Small epsilon to prevent division by zero or near-zero

def calculate_metrics(rets, bench_rets, bench_ticker_name, name="Strategy", num_trials=1):
    """Calculates performance metrics for a given returns series."""

    # Filter out zero returns to get active trading periods
    active_rets = rets[rets != 0].dropna()
    active_bench_rets = bench_rets[bench_rets != 0].dropna()

    # If no active returns, return very bad (but finite) values for common optimization targets
    # and NaN for others. This allows the optimizer to penalize such parameter sets.
    if active_rets.empty:
        VERY_BAD_METRIC_VALUE = -9999.0 # For metrics that are typically maximized
        NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY = 0.0 # For metrics like Beta, Alpha when no activity

        # Metrics that are commonly targets for maximization
        maximization_targets = [
            "Total Return", "Ann. Return", "Sharpe", "Sortino", "Calmar", "Deflated Sharpe", "K-Ratio"
        ]
        # Metrics that might be near zero or NaN and can be set to a neutral value
        neutral_metrics = [
            "Alpha (ann)", "Beta", "R^2"
        ]
        # Metrics where NaN is acceptable or interpretation is complex for zero activity
        # (e.g. Max DD is technically 0 if no returns, but often expected to be negative)
        # Volatility is 0, Skew/Kurtosis undefined. ADF test also not meaningful.

        metrics_dict = {
            "Total Return": VERY_BAD_METRIC_VALUE,
            "Ann. Return": VERY_BAD_METRIC_VALUE,
            "Ann. Vol": 0.0, # Volatility is indeed 0 if no active returns
            "Sharpe": VERY_BAD_METRIC_VALUE,
            "Sortino": VERY_BAD_METRIC_VALUE,
            "Calmar": VERY_BAD_METRIC_VALUE,
            "Alpha (ann)": NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY,
            "Beta": NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY,
            "Max DD": 0.0, # Max drawdown is 0 if returns are flat
            "Skew": np.nan, # Skew is undefined for constant series
            "Kurtosis": np.nan, # Kurtosis is undefined for constant series
            "R^2": NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY,
            "K-Ratio": VERY_BAD_METRIC_VALUE,
            "ADF Statistic": np.nan,
            "ADF p-value": np.nan,
            "Deflated Sharpe": VERY_BAD_METRIC_VALUE
        }
        # Ensure all expected metrics are present, even if some new ones were added to the main calculation
        # This list should match the one in the main body of the function.
        expected_metric_keys = [
            "Total Return", "Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "Calmar",
            "Alpha (ann)", "Beta", "Max DD", "Skew", "Kurtosis", "R^2", "K-Ratio",
            "ADF Statistic", "ADF p-value", "Deflated Sharpe"
        ]
        for key in expected_metric_keys:
            if key not in metrics_dict:
                # Default to NaN if a new metric was added and not covered here
                metrics_dict[key] = np.nan

        return pd.Series(metrics_dict, name=name)

    steps_per_year = _infer_steps_per_year(active_rets.index)

    def sortino_ratio(r, target=0):
        if r.empty or r.isnull().all():
            return np.nan
        target_returns = r - target
        downside_risk = np.sqrt(np.mean(np.minimum(0, target_returns) ** 2))
        if downside_risk < EPSILON_FOR_DIVISION:
            return np.nan
        return (r.mean() * steps_per_year) / (
            downside_risk * np.sqrt(steps_per_year)
        )

    def total_ret(x): return (1 + x).prod() - 1 if len(x) > 0 else np.nan
    def ann(x):
        if x.empty or x.isnull().all():
            return np.nan
        prod = (1 + x).prod()
        if prod < 0:
            return -1.0
        return prod ** (steps_per_year / len(x)) - 1
    def ann_vol(x):
        if x.empty or x.isnull().all():
            return np.nan
        return x.std() * np.sqrt(steps_per_year)
    def sharpe(x):
        if x.empty or x.isnull().all():
            return np.nan
        annualized_vol = ann_vol(x)
        return (ann(x) / annualized_vol) if annualized_vol > EPSILON_FOR_DIVISION else np.nan
    def mdd(series): 
        if series.empty or series.isnull().all(): return np.nan
        return (series / series.cummax() - 1).min()
    def calmar(x):
        if x.empty or x.isnull().all():
            return np.nan
        max_dd = mdd((1 + x).cumprod())
        annualized_return = ann(x)
        if abs(max_dd) < EPSILON_FOR_DIVISION:
            return np.nan
        return annualized_return / abs(max_dd)

    def stationarity_test(series):
        """Performs ADF test on the cumulative P&L (equity curve)."""
        if len(series) < 40:
            return np.nan, np.nan
        
        # The ADF test should be run on the price series (cumulative returns)
        cumulative_pnl = (1 + series).cumprod()
        
        try:
            result = adfuller(cumulative_pnl)
            return result[0], result[1]  # ADF Statistic, p-value
        except Exception:
            return np.nan, np.nan

    def deflated_sharpe_ratio(rets, num_trials):
        """
        Calculates the Deflated Sharpe Ratio (DSR).
        See: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
        """
        if num_trials <= 0:
            return np.nan

        if len(rets) < 100: # DSR requires a longer series
            return np.nan

        sr = sharpe(rets)
        if pd.isna(sr):
            return np.nan

        sk = skew(rets)
        k_excess = kurtosis(rets, fisher=True)
        n = len(rets)

        # The expected SR of random strategies is not necessarily zero.
        # We use the formula for the expected maximum SR from a set of trials.
        # This is a more robust way to deflate the SR.
        emc = 0.5772156649 # Euler-Mascheroni constant
        max_z = (1 - emc) * norm.ppf(1 - 1/num_trials) + emc * norm.ppf(1 - 1/(num_trials * np.e))
        
        # Variance of the Sharpe Ratio estimator
        var_sr = (1 / (n - 1)) * (1 - sk * sr + (k_excess / 4) * sr**2)
        if var_sr < 0:
            return np.nan
        
        # The expected value of the maximum Sharpe ratio from N trials
        expected_max_sr = sr + np.sqrt(var_sr) * max_z

        # Deflated Sharpe Ratio is the probability of the observed SR being lower than the expected max SR
        dsr = norm.cdf(sr, loc=expected_max_sr, scale=np.sqrt(var_sr))
        return dsr

    common_index = active_rets.index.intersection(active_bench_rets.index)
    rets_aligned, bench_aligned = active_rets.loc[common_index], active_bench_rets.loc[common_index]

    # Ensure enough data for CAPM regression
    if len(rets_aligned) < 2 or len(bench_aligned) < 2 or bench_aligned.std() == 0:
        alpha = np.nan
        beta = np.nan
        r_squared = np.nan
    else:
        X = sm.add_constant(bench_aligned)
        try:
            capm = sm.OLS(rets_aligned, X).fit()
            alpha = capm.params.get("const", np.nan) * steps_per_year
            beta = capm.params.get(bench_ticker_name, np.nan)
            r_squared = capm.rsquared
        except Exception:
            alpha = np.nan
            beta = np.nan
            r_squared = np.nan

    adf_stat, adf_p_value = stationarity_test(active_rets)

    # Coefficient of determination (R^2)
    r_squared = r_squared if 'r_squared' in locals() else np.nan

    # K-Ratio: slope of log equity curve scaled by its standard error
    if len(active_rets) >= 2:
        log_equity = np.log((1 + active_rets).cumprod())
        idx = np.arange(len(log_equity))
        reg = linregress(idx, log_equity)
        if reg.stderr < EPSILON_FOR_DIVISION:
            k_ratio = np.nan
        else:
            k_ratio = (reg.slope / reg.stderr) * np.sqrt(len(log_equity))
    else:
        k_ratio = np.nan

    metrics = pd.Series({
        "Total Return": total_ret(active_rets),
        "Ann. Return": ann(active_rets),
        "Ann. Vol": ann_vol(active_rets),
        "Sharpe": sharpe(active_rets),
        "Sortino": sortino_ratio(active_rets),
        "Calmar": calmar(active_rets),
        "Alpha (ann)": alpha,
        "Beta": beta,
        "Max DD": mdd((1 + active_rets).cumprod()),
        "Skew": skew(active_rets),
        "Kurtosis": kurtosis(active_rets),
        "R^2": r_squared,
        "K-Ratio": k_ratio,
        "ADF Statistic": adf_stat,
        "ADF p-value": adf_p_value,
        "Deflated Sharpe": deflated_sharpe_ratio(active_rets, num_trials)
    }, name=name)
    return metrics