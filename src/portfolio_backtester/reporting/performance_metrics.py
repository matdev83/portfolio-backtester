import pandas as pd
import numpy as np
import logging
from scipy.stats import skew, kurtosis, norm, linregress
from typing import Tuple # Import Tuple for type hinting
import scipy._lib._util as sp_util

logger = logging.getLogger(__name__)

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

# Import Numba optimization with fallback
try:
    from ..numba_optimized import sortino_ratio_fast, mdd_fast, drawdown_duration_and_recovery_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


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
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"calculate_metrics called for {name}")
        logger.debug(f"rets length: {len(rets)}, first few: {rets.head().tolist() if not rets.empty else 'empty'}")
        logger.debug(f"rets has any NaN: {rets.isnull().any() if not rets.empty else 'empty'}")
        logger.debug(f"rets all zero: {(rets == 0).all() if not rets.empty else 'empty'}")

    # Filter out zero returns to get active trading periods
    # However, first check if the original 'rets' series is effectively all zeros.
    is_all_zero_returns = rets.abs().max() < EPSILON_FOR_DIVISION if not rets.empty else False
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"is_all_zero_returns: {is_all_zero_returns}")

    active_rets = rets[rets != 0].dropna()
    active_bench_rets = bench_rets[bench_rets != 0].dropna() # Benchmark can still be active
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"active_rets length: {len(active_rets)}")

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
            "Max Drawdown": ZERO_EQUIVALENT_METRIC_VALUE, # Max drawdown is 0 if returns are flat or no active returns
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

                # Add check for rets_for_capm.std() == 0
                rets_std = rets_for_capm.std()
                bench_std = bench_for_capm.std()
                # Ensure rets_std and bench_std are scalar values for comparison
                def ensure_scalar(value):
                    """Convert any pandas Series, numpy array, or other sequence to a scalar float."""
                    if value is None:
                        return 0.0
                    
                    # If it's already a scalar number, return it
                    if isinstance(value, (int, float)) and not hasattr(value, '__len__'):
                        return float(value)
                    
                    # Handle pandas Series/DataFrame
                    if hasattr(value, 'iloc'):
                        if len(value) == 0:
                            return 0.0
                        return float(value.iloc[0])
                    
                    # Handle numpy arrays and other sequences
                    if hasattr(value, '__len__'):
                        if len(value) == 0:
                            return 0.0
                        return float(value[0])
                    
                    # Handle numpy scalars
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return 0.0
                
                # Convert both to scalars
                rets_std_scalar = ensure_scalar(rets_std)
                bench_std_scalar = ensure_scalar(bench_std)
                
                # Now we can safely compare scalars
                if (len(rets_for_capm) >= 2 and len(bench_for_capm) >= 2 and 
                    bench_std_scalar != 0 and rets_std_scalar != 0):
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
            "Alpha (ann)", "Beta", "Max Drawdown", "VaR (5%)", "CVaR (5%)", "Tail Ratio",
            "Avg DD Duration", "Avg Recovery Time", "Skew", "Kurtosis", "R^2", "K-Ratio",
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
        if series.empty or series.isnull().all(): 
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"mdd: series is empty or all null. Length: {len(series)}, all null: {series.isnull().all() if not series.empty else 'N/A'}")
            return np.nan
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"mdd: series length: {len(series)}, first few values: {series.head().tolist()}")
            logger.debug(f"mdd: series has any NaN: {series.isnull().any()}, series min: {series.min()}, series max: {series.max()}")
        
        cummax_series = series.cummax()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"mdd: cummax min: {cummax_series.min()}, cummax max: {cummax_series.max()}")
        
        drawdown_series = (series / cummax_series - 1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"mdd: drawdown min: {drawdown_series.min()}, drawdown max: {drawdown_series.max()}")
            logger.debug(f"mdd: drawdown has any NaN: {drawdown_series.isnull().any()}")
        
        result = drawdown_series.min()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"mdd: final result: {result}")
        return result
    def calmar(x):
        if x.empty or x.isnull().all():
            return np.nan
        # Use original rets for max drawdown calculation, not active_rets
        max_dd = mdd((1 + rets).cumprod())  # Zero returns are valid for drawdown calculation
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
        
        # The expected SR of random strategies is often assumed to be 0.
        # E[max SR_random] = E[SR_random] + StDev[SR_random] * max_z
        # Assuming E[SR_random] = 0, and StDev[SR_random] for non-autocorrelated returns
        # with 0 skew and 0 excess kurtosis is approx. sqrt(1/n).
        # More precisely, for SR estimator, it's sqrt(1/(n-1)) under H0 (true SR=0, sk=0, k_exc=0).
        if n - 1 <= 0: # Should be caught by len(rets) < 100, but defensive
            return np.nan

        std_sr_h0 = np.sqrt(1 / (n - 1)) # Standard deviation of SR under H0 (true SR=0, sk=0, k_exc=0)
        expected_max_sr_h0 = std_sr_h0 * max_z

        # Variance of the observed Sharpe Ratio estimator (var_sr) is already calculated:
        # var_sr = (1 / (n - 1)) * (1 - sk * sr + (k_excess / 4) * sr**2)
        if var_sr <= 0: # Variance must be positive for a valid standard deviation
            return np.nan

        std_dev_sr_observed = np.sqrt(var_sr)

        # Deflated Sharpe Ratio Z-statistic
        # This measures how many standard deviations the observed SR is from the expected max SR under H0.
        dsr_z_statistic = (sr - expected_max_sr_h0) / std_dev_sr_observed

        # The Deflated Sharpe Ratio is often presented as a probability (PSR - Probabilistic Sharpe Ratio)
        # P( SR_true > 0 | taking into account selection bias from num_trials)
        # Or, more directly, P(observed SR > expected_max_SR_under_H0)
        dsr_probability = norm.cdf(dsr_z_statistic)

        return dsr_probability

    def value_at_risk(rets, confidence_level=0.05):
        """Calculate Value at Risk (VaR) at given confidence level."""
        if rets.empty or rets.isnull().all():
            return np.nan
        return np.percentile(rets, confidence_level * 100)

    def conditional_value_at_risk(rets, confidence_level=0.05):
        """Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR."""
        if rets.empty or rets.isnull().all():
            return np.nan
        var = value_at_risk(rets, confidence_level)
        if pd.isna(var):
            return np.nan
        tail_losses = rets[rets <= var]
        if tail_losses.empty:
            return var  # If no losses beyond VaR, return VaR itself
        return tail_losses.mean()

    def tail_ratio(rets, percentile=95):
        """Calculate Tail Ratio - ratio of average positive returns to average negative returns."""
        if rets.empty or rets.isnull().all():
            return np.nan
        
        positive_rets = rets[rets > 0]
        negative_rets = rets[rets < 0]
        
        if positive_rets.empty or negative_rets.empty:
            return np.nan
            
        upper_tail = np.percentile(positive_rets, percentile)
        lower_tail = np.percentile(negative_rets, 100 - percentile)
        
        if abs(lower_tail) < EPSILON_FOR_DIVISION:
            return np.inf if upper_tail > 0 else np.nan
            
        return upper_tail / abs(lower_tail)

    def drawdown_duration_and_recovery(equity_curve):
        """Calculate average drawdown duration and recovery time."""
        if equity_curve.empty or equity_curve.isnull().all():
            return np.nan, np.nan
        
        if NUMBA_AVAILABLE:
            return drawdown_duration_and_recovery_fast(equity_curve.values)
        else:
            # Calculate running maximum and drawdown
            running_max = equity_curve.cummax()
            drawdown = (equity_curve / running_max - 1)
            
            # Identify drawdown periods
            is_drawdown = drawdown < -EPSILON_FOR_DIVISION
            
            if not is_drawdown.any():
                return 0.0, 0.0  # No drawdowns
            
            # Find drawdown periods
            drawdown_periods = []
            recovery_periods = []
            
            in_drawdown = False
            drawdown_start = None
            drawdown_end = None
            
            for i, (date, dd_val) in enumerate(drawdown.items()):
                if dd_val < -EPSILON_FOR_DIVISION and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    drawdown_start = i
                elif dd_val >= -EPSILON_FOR_DIVISION and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    drawdown_end = i - 1
                    
                    if drawdown_start is not None and drawdown_end is not None:
                        duration = drawdown_end - drawdown_start + 1
                        drawdown_periods.append(duration)
                        
                        # Find recovery period (time to reach new high)
                        peak_before_dd = running_max.iloc[drawdown_start]
                        recovery_start = drawdown_end + 1
                        
                        if recovery_start < len(equity_curve):
                            recovery_found = False
                            for j in range(recovery_start, len(equity_curve)):
                                if equity_curve.iloc[j] >= peak_before_dd:
                                    recovery_time = j - drawdown_end
                                    recovery_periods.append(recovery_time)
                                    recovery_found = True
                                    break
                            
                            if not recovery_found:
                                # Still in recovery at end of period
                                recovery_time = len(equity_curve) - 1 - drawdown_end
                                recovery_periods.append(recovery_time)
            
            # Handle case where period ends in drawdown
            if in_drawdown and drawdown_start is not None:
                duration = len(drawdown) - drawdown_start
                drawdown_periods.append(duration)
            
            avg_dd_duration = np.mean(drawdown_periods) if drawdown_periods else 0.0
            avg_recovery_time = np.mean(recovery_periods) if recovery_periods else np.nan
            
            return avg_dd_duration, avg_recovery_time

    common_index = active_rets.index.intersection(active_bench_rets.index)
    rets_aligned, bench_aligned = active_rets.loc[common_index], active_bench_rets.loc[common_index]
    
    # Debug logging
    if len(rets_aligned) == 0 or len(bench_aligned) == 0:
        logger.debug(f"Empty aligned series: rets_len={len(rets_aligned)}, bench_len={len(bench_aligned)}")
    if not isinstance(rets_aligned, pd.Series) or not isinstance(bench_aligned, pd.Series):
        logger.debug(f"Unexpected types: rets_type={type(rets_aligned)}, bench_type={type(bench_aligned)}")

    # Ensure enough data for CAPM regression and that strategy returns have variance
    # Handle std() calculation safely
    try:
        bench_std_val = bench_aligned.std()
        bench_std = float(bench_std_val) if pd.notna(bench_std_val) else 0.0
    except:
        bench_std = 0.0
        
    try:
        rets_std_val = rets_aligned.std()
        rets_std = float(rets_std_val) if pd.notna(rets_std_val) else 0.0
    except:
        rets_std = 0.0
    
    if len(rets_aligned) < 2 or len(bench_aligned) < 2 or abs(bench_std) < EPSILON_FOR_DIVISION or abs(rets_std) < EPSILON_FOR_DIVISION:
        alpha = np.nan
        beta = np.nan
        r_squared = np.nan
    else:
        # Set the name of the benchmark series to ensure proper column naming in regression
        bench_aligned_named = bench_aligned.copy()
        bench_aligned_named.name = bench_ticker_name
        
        X = sm.add_constant(bench_aligned_named)
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
        # Ensure log_equity has variance before regression
        log_equity_std = float(log_equity.std()) if not pd.isna(log_equity.std()) else 0.0
        if log_equity_std < EPSILON_FOR_DIVISION:
            k_ratio = np.nan
        else:
            idx = np.arange(len(log_equity))
            slope, intercept, r_value, p_value, std_err = linregress(idx, log_equity)
            # Ensure std_err and slope are floats (not tuple or object)
            def safe_float(val):
                try:
                    if isinstance(val, (tuple, list, np.ndarray)):
                        val = val[0]
                    return float(val)
                except Exception:
                    return np.nan
            std_err_val = safe_float(std_err)
            slope_val = safe_float(slope)
            if std_err_val < EPSILON_FOR_DIVISION:
                k_ratio = np.nan
            else:
                k_ratio = (slope_val / std_err_val) * np.sqrt(len(log_equity))
    else:
        k_ratio = np.nan

    # Calculate new risk metrics
    var_5pct = value_at_risk(active_rets, 0.05)
    cvar_5pct = conditional_value_at_risk(active_rets, 0.05)
    tail_ratio_95 = tail_ratio(active_rets, 95)
    
    # Calculate drawdown metrics using full equity curve
    equity_curve = (1 + rets).cumprod()
    avg_dd_duration, avg_recovery_time = drawdown_duration_and_recovery(equity_curve)

    metrics = pd.Series({
        "Total Return": total_ret(active_rets),
        "Ann. Return": ann(active_rets),
        "Ann. Vol": ann_vol(active_rets),
        "Sharpe": sharpe(active_rets),
        "Sortino": sortino_ratio(active_rets),
        "Calmar": calmar(active_rets),
        "Alpha (ann)": alpha,
        "Beta": beta,
        "Max Drawdown": mdd((1 + rets).cumprod()),  # Use original rets, not active_rets - zero returns are valid for drawdown
        "VaR (5%)": var_5pct,
        "CVaR (5%)": cvar_5pct,
        "Tail Ratio": tail_ratio_95,
        "Avg DD Duration": avg_dd_duration,
        "Avg Recovery Time": avg_recovery_time,
        "Skew": skew(active_rets),
        "Kurtosis": kurtosis(active_rets),
        "R^2": r_squared,
        "K-Ratio": k_ratio,
        "ADF Statistic": adf_stat,
        "ADF p-value": adf_p_value,
        "Deflated Sharpe": deflated_sharpe_ratio(active_rets, num_trials)
    }, name=name)
    return metrics
