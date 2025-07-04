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
    # However, first check if the original 'rets' series is effectively all zeros.
    is_all_zero_returns = rets.abs().max() < EPSILON_FOR_DIVISION if not rets.empty else False

    active_rets = rets[rets != 0].dropna()
    active_bench_rets = bench_rets[bench_rets != 0].dropna() # Benchmark can still be active

    # If no active returns, or if all returns were effectively zero.
    if active_rets.empty:
        VERY_BAD_METRIC_VALUE = -9999.0 # For metrics that are typically maximized
        ZERO_EQUIVALENT_METRIC_VALUE = 0.0 # For metrics that should be 0 if returns are all zero
        NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY = 0.0 # For metrics like Beta, Alpha when no activity

        # Default values for when active_rets is empty
        metrics_dict = {
            "Total Return": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE,
            "Ann. Return": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE,
            "Ann. Vol": ZERO_EQUIVALENT_METRIC_VALUE, # Volatility is indeed 0 if no active returns or all zero returns
            "Sharpe": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE,
            "Sortino": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE,
            "Calmar": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE,
            "Alpha (ann)": NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY, # Alpha might be calculable against bench even if strat is flat
            "Beta": NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY,    # Beta might be calculable
            "Max DD": ZERO_EQUIVALENT_METRIC_VALUE, # Max drawdown is 0 if returns are flat or no active returns
            "Skew": np.nan, # Skew is undefined for constant series
            "Kurtosis": np.nan, # Kurtosis is undefined for constant series
            "R^2": NEUTRAL_METRIC_VALUE_FOR_ZERO_ACTIVITY, # R^2 might be calculable
            "K-Ratio": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE,
            "ADF Statistic": np.nan,
            "ADF p-value": np.nan,
            "Deflated Sharpe": ZERO_EQUIVALENT_METRIC_VALUE if is_all_zero_returns else VERY_BAD_METRIC_VALUE
        }

        # Recalculate Alpha, Beta, R^2 if benchmark is active, even if strategy returns are all zero
        if not active_bench_rets.empty and is_all_zero_returns:
            # Align original (all-zero) rets with active_bench_rets for CAPM
            # Ensure rets_for_capm has same length as active_bench_rets and represents zero returns
            common_idx_for_capm = rets.index.intersection(active_bench_rets.index)
            if not common_idx_for_capm.empty:
                rets_for_capm = pd.Series(0.0, index=common_idx_for_capm)
                bench_for_capm = active_bench_rets.loc[common_idx_for_capm]

                if len(rets_for_capm) >= 2 and len(bench_for_capm) >=2 and bench_for_capm.std() != 0:
                    steps_per_year_for_capm = _infer_steps_per_year(common_idx_for_capm)
                    X_capm = sm.add_constant(bench_for_capm)
                    try:
                        capm_model = sm.OLS(rets_for_capm, X_capm).fit()
                        metrics_dict["Alpha (ann)"] = capm_model.params.get("const", np.nan) * steps_per_year_for_capm
                        metrics_dict["Beta"] = capm_model.params.get(bench_ticker_name, np.nan)
                        metrics_dict["R^2"] = capm_model.rsquared
                    except Exception:
                        # Keep default NaNs or neutral values if CAPM fails
                        pass


        # Ensure all expected metrics are present
        expected_metric_keys = [
            "Total Return", "Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "Calmar",
            "Alpha (ann)", "Beta", "Max DD", "Skew", "Kurtosis", "R^2", "K-Ratio",
            "ADF Statistic", "ADF p-value", "Deflated Sharpe"
        ]
        for key in expected_metric_keys:
            if key not in metrics_dict:
                metrics_dict[key] = np.nan # Default for any newly added metrics not covered

        return pd.Series(metrics_dict, name=name)

    # If active_rets is not empty, proceed with normal calculation.
    # Note: _infer_steps_per_year should ideally use rets.index if active_rets is empty
    # but the functions below (sharpe, calmar etc) use active_rets.
    # The current structure means if active_rets is empty, we don't reach here.
    # If it's not empty, then active_rets.index is valid.
    steps_per_year = _infer_steps_per_year(active_rets.index)

    def sortino_ratio(r, target=0):
        if r.empty or r.isnull().all():
            return np.nan
        target_returns = r - target
        downside_risk = np.sqrt(np.mean(np.minimum(0, target_returns) ** 2))

        annualized_mean_return = r.mean() * steps_per_year
        if pd.isna(annualized_mean_return) or pd.isna(downside_risk): # If either is NaN, Sortino is NaN
            return np.nan

        if downside_risk < EPSILON_FOR_DIVISION:
            # Downside risk is effectively zero
            if abs(annualized_mean_return) < EPSILON_FOR_DIVISION:
                return 0.0  # Mean return is also zero
            elif annualized_mean_return > 0:
                return np.inf  # Positive mean return, zero downside risk
            else:
                return -np.inf # Negative mean return, zero downside risk (less common, but possible if all returns are 0 or positive but mean is negative due to target)

        # The original formula simplifies to mean_return / downside_risk_per_period, then annualize.
        # Or, (annualized_mean_return) / (annualized_downside_risk).
        # Annualized downside risk = downside_risk * np.sqrt(steps_per_year)
        annualized_downside_risk = downside_risk * np.sqrt(steps_per_year)
        if annualized_downside_risk < EPSILON_FOR_DIVISION: # Check again after annualization, though less likely to change category
             if abs(annualized_mean_return) < EPSILON_FOR_DIVISION:
                return 0.0
             elif annualized_mean_return > 0:
                return np.inf
             else:
                return -np.inf
        return annualized_mean_return / annualized_downside_risk

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
        annualized_return = ann(x)
        annualized_vol = ann_vol(x)
        if pd.isna(annualized_return) or pd.isna(annualized_vol): # If ann_return or ann_vol is NaN, Sharpe is NaN
            return np.nan
        if annualized_vol < EPSILON_FOR_DIVISION:
            if abs(annualized_return) < EPSILON_FOR_DIVISION:
                return 0.0 # Both are zero
            elif annualized_return > 0:
                return np.inf # Positive return, zero volatility
            else:
                return -np.inf # Negative return, zero volatility
        return annualized_return / annualized_vol
    def mdd(series): 
        if series.empty or series.isnull().all(): return np.nan
        return (series / series.cummax() - 1).min()
    def calmar(x):
        if x.empty or x.isnull().all():
            return np.nan
        max_dd = mdd((1 + x).cumprod())
        annualized_return = ann(x)
        if pd.isna(annualized_return) or pd.isna(max_dd): # If ann_return or max_dd is NaN, Calmar is NaN
            return np.nan
        if abs(max_dd) < EPSILON_FOR_DIVISION:
            if abs(annualized_return) < EPSILON_FOR_DIVISION:
                return 0.0  # Both are zero
            elif annualized_return > 0:
                return np.inf # Positive return, zero drawdown
            else:
                return -np.inf # Negative return, zero drawdown
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