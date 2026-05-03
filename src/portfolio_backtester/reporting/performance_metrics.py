"""Performance metrics for reporting (Rich tables, CSV, optimizer parity).

Canonical entry point: :func:`calculate_metrics`.

**Sharpe:** With risk-free series aligned and non-empty, Sharpe uses annualized
mean and volatility of **excess** returns. Otherwise it uses **geometric**
annualized return divided by annualized volatility (legacy path). See
``docs/performance_metrics.md``.

**Tail Ratio:** Custom percentile ratio (95th pct of wins vs 5th pct of losses),
not average gain / average loss.

**Deflated Sharpe:** Column name in reports; implementation returns a
PSR-like probability (normal CDF of a z-statistic), not the paper DSR formula.
Requires ``num_trials > 1`` and sufficient history for a defined value.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, linregress, norm, skew
from statsmodels.tsa.stattools import adfuller
from ..numba_optimized import drawdown_duration_and_recovery_fast
from .metric_display import SHARPE_PATH_EXCESS, SHARPE_PATH_LEGACY_CAGR_VOL

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper to derive the appropriate annualisation factor (steps per year) from
# the return series' timestamp index.  Falls back to a daily assumption if the
# frequency cannot be determined automatically.
# -----------------------------------------------------------------------------


def _infer_steps_per_year(idxs: pd.DatetimeIndex) -> int:
    """Return 252 for daily, 52 for weekly, 12 for monthly data."""
    if len(idxs) < 3:
        return 252  # default – avoid infer_freq error and division-by-zero downstream

    try:
        freq = pd.infer_freq(idxs)
    except (ValueError, TypeError):
        # Handle potential errors in frequency inference
        freq = None

    if freq is None:
        # Heuristic based on median spacing in days
        spacing = (idxs[-1] - idxs[0]).days / max(len(idxs) - 1, 1)
        if spacing <= 3:
            return 252  # ~daily
        if spacing <= 10:
            return 52  # ~weekly
        return 12  # monthly or sparser

    if freq.startswith("B") or freq in {"D"}:
        return 252
    if freq.startswith("W"):
        return 52
    if freq.startswith("M") or freq.startswith("ME") or freq.startswith("MS"):
        return 12
    return 252


EPSILON_FOR_DIVISION = 1e-15  # Small epsilon to prevent division by zero or near-zero


def _is_all_zero(series: pd.Series) -> bool:
    return False if series.empty else bool(series.abs().max() < EPSILON_FOR_DIVISION)


def _safe_moment(func, series: pd.Series) -> float:
    """Compute a statistical moment (skew/kurtosis) with stability guards.

    - If the series is too short or has effectively zero variance, return 0.0
      without invoking the underlying routine which may trigger runtime
      warnings due to catastrophic cancellation.
    - Suppress RuntimeWarning coming from the underlying implementation and
      coerce the result to float. If anything goes wrong, return 0.0.
    """
    # If series is empty or has effectively zero variance, emit a RuntimeWarning
    # to preserve previous behaviour expected by tests, then return 0.0.
    # Warn only when the series is empty or effectively constant (all values equal)
    # Use a variance-based guard instead of relying solely on unique counts.
    # This avoids emitting warnings for series with small but meaningful variance
    # while still warning for effectively-constant data that would cause
    # catastrophic cancellation in moment calculations.
    # Use a stricter condition for emitting the catastrophic-cancellation warning.
    # Only warn when the series is empty or effectively constant (one unique non-NA value).
    try:
        non_na = series.dropna()
        unique_vals = non_na.nunique()
        std_val = (
            float(non_na.std(ddof=0)) if not non_na.empty and pd.notna(non_na.std(ddof=0)) else 0.0
        )
    except Exception:
        non_na = series
        unique_vals = 0
        std_val = 0.0

    # Warn when series is empty, or truly constant (one unique value), or has
    # effectively zero variance. This keeps the expected warning for constant
    # inputs (tests) while reducing false positives via a tiny EPSILON.
    # Do not emit a RuntimeWarning here; return a stable 0.0 for degenerate inputs.
    if non_na.empty or unique_vals <= 1 or std_val <= EPSILON_FOR_DIVISION:
        return 0.0

    # Otherwise compute the moment while suppressing any low-level RuntimeWarnings
    # from the underlying library to avoid noisy output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            # Use compensated summation-based implementations for higher precision
            # where appropriate (skew/kurtosis). Fall back to the provided func
            # for other statistical functions.
            if func is skew:
                try:
                    # Try to use Numba-optimized implementation if available
                    from ..numba_optimized import compensated_skew_fast

                    arr = np.asarray(series.values, dtype=np.float64)
                    return float(compensated_skew_fast(arr))
                except Exception:
                    return float(_compensated_skew(series))
            if func is kurtosis:
                try:
                    from ..numba_optimized import compensated_kurtosis_fast

                    arr = np.asarray(series.values, dtype=np.float64)
                    return float(compensated_kurtosis_fast(arr))
                except Exception:
                    return float(_compensated_kurtosis(series))

            val = func(series)
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0


def _compensated_skew(series: pd.Series) -> float:
    """Compute skew using compensated summation (math.fsum) for improved accuracy."""
    import math

    x = [float(v) for v in series]
    n = len(x)
    if n == 0:
        return 0.0
    mean = math.fsum(x) / n
    # central moments using compensated sums
    m2 = math.fsum((xi - mean) ** 2 for xi in x) / n
    if m2 <= 0.0:
        return 0.0
    m3 = math.fsum((xi - mean) ** 3 for xi in x)
    # Normalize using float operations to avoid Any return type for mypy
    denom = (m2**1.5) * n
    if denom == 0.0:
        return 0.0
    return float(m3 / denom)


def _compensated_kurtosis(series: pd.Series) -> float:
    """Compute excess kurtosis using compensated summation (math.fsum)."""
    import math

    x = [float(v) for v in series]
    n = len(x)
    if n == 0:
        return 0.0
    mean = math.fsum(x) / n
    m2 = math.fsum((xi - mean) ** 2 for xi in x) / n
    if m2 <= 0.0:
        return 0.0
    m4 = math.fsum((xi - mean) ** 4 for xi in x) / n
    return m4 / (m2**2) - 3.0


def _default_zero_activity_metrics(is_all_zero_returns: bool) -> dict:
    VERY_BAD = -9999.0
    ZERO = 0.0
    NEUTRAL = 0.0
    m = {
        "Total Return": ZERO if is_all_zero_returns else VERY_BAD,
        "Ann. Return": ZERO if is_all_zero_returns else VERY_BAD,
        "Ann. Vol": ZERO,
        "Sharpe": ZERO if is_all_zero_returns else VERY_BAD,
        "Sortino": ZERO if is_all_zero_returns else VERY_BAD,
        "Calmar": ZERO if is_all_zero_returns else VERY_BAD,
        "Alpha (ann)": NEUTRAL,
        "Beta": NEUTRAL,
        "Max Drawdown": ZERO,
        "Skew": np.nan,
        "Kurtosis": np.nan,
        "R^2": NEUTRAL,
        "K-Ratio": ZERO if is_all_zero_returns else VERY_BAD,
        "ADF Statistic (equity)": np.nan,
        "ADF p-value (equity)": np.nan,
        "ADF Statistic (returns)": np.nan,
        "ADF p-value (returns)": np.nan,
        "Deflated Sharpe": ZERO if is_all_zero_returns else VERY_BAD,
    }
    return m


def _ensure_expected_keys(metrics_dict: dict) -> dict:
    expected = [
        "Total Return",
        "Ann. Return",
        "Ann. Vol",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Alpha (ann)",
        "Beta",
        "Max Drawdown",
        "VaR (5%)",
        "CVaR (5%)",
        "Tail Ratio",
        "Avg DD Duration",
        "Avg Recovery Time",
        "Skew",
        "Kurtosis",
        "R^2",
        "K-Ratio",
        "ADF Statistic (equity)",
        "ADF p-value (equity)",
        "ADF Statistic (returns)",
        "ADF p-value (returns)",
        "Tail Ratio (mean)",
        "Deflated Sharpe",
        "Max DD Recovery Time (bars)",
    ]
    for k in expected:
        metrics_dict.setdefault(k, np.nan)
    return metrics_dict


def _capm_on_zeros(rets: pd.Series, active_bench_rets: pd.Series, bench_ticker_name: str) -> dict:
    """Compute CAPM metrics for the degenerate case of zero strategy returns.

    Handles Series/DataFrame/std edge cases robustly to avoid ambiguous truth checks.
    """
    out = {"Alpha (ann)": 0.0, "Beta": 0.0, "R^2": 0.0}
    common_idx = rets.index.intersection(active_bench_rets.index)
    if common_idx.empty:
        return out

    # Zero strategy returns on the common index
    rets_zero = pd.Series(0.0, index=common_idx)

    # Align benchmark
    bench = active_bench_rets.loc[common_idx]

    # Require at least 2 observations
    if len(rets_zero) < 2 or len(bench) < 2:
        return out

    # Robust std checks: convert to scalar when possible; if Series, reduce to a scalar guard
    def _std_is_zero(x) -> bool:
        try:
            s = x.std(ddof=0)
            if np.isscalar(s):
                return bool(abs(float(str(s))) < EPSILON_FOR_DIVISION)
            # If s is a Series/array-like, consider zero variance only if all components are ~0
            if hasattr(s, "abs"):
                return bool(s.abs().lt(EPSILON_FOR_DIVISION).all())
            # Fallback: convert to numpy array
            s_arr = np.asarray(s, dtype=float)
            return bool(np.all(np.abs(s_arr) < EPSILON_FOR_DIVISION))
        except Exception:
            # Be conservative – treat as zero-variance to skip regression
            return True

    if _std_is_zero(bench) or _std_is_zero(rets_zero):
        return out

    steps = _infer_steps_per_year(pd.DatetimeIndex(common_idx))

    # Ensure the regressor has a consistent name for coefficient extraction
    bench_named = bench.copy()
    if isinstance(bench_named, pd.Series):
        bench_named.name = bench_ticker_name

    X = sm.add_constant(bench_named)
    try:
        capm = sm.OLS(rets_zero, X).fit()
        out["Alpha (ann)"] = capm.params.get("const", np.nan) * steps
        out["Beta"] = capm.params.get(bench_ticker_name, np.nan)
        out["R^2"] = capm.rsquared
    except Exception:
        # Leave defaults on failure
        pass
    return out


def calculate_metrics(
    rets,
    bench_rets,
    bench_ticker_name,
    name="Strategy",
    num_trials=1,
    trade_stats=None,
    risk_free_rets: Optional[pd.Series] = None,
):
    is_all_zero_returns = _is_all_zero(rets)
    # IMPORTANT:
    # Treat 0.0 returns as valid "in-cash / no-position" days. Filtering them out
    # artificially shortens the effective time horizon for strategies that spend
    # time in cash, which can inflate annualized return / Sharpe / Sortino and
    # lead to inconsistent comparisons vs the benchmark.
    active_rets = rets.dropna()
    active_bench_rets = bench_rets.dropna()
    if isinstance(active_rets, pd.DataFrame):
        if active_rets.empty or len(active_rets.columns) == 0:
            active_rets = pd.Series(dtype=float)
        else:
            active_rets = active_rets.iloc[:, 0]
    if isinstance(active_bench_rets, pd.DataFrame):
        if active_bench_rets.empty or len(active_bench_rets.columns) == 0:
            active_bench_rets = pd.Series(dtype=float)
        else:
            active_bench_rets = active_bench_rets.iloc[:, 0]

    if active_rets.empty:
        metrics = _default_zero_activity_metrics(is_all_zero_returns)
        if not active_bench_rets.empty and is_all_zero_returns:
            metrics.update(_capm_on_zeros(rets, active_bench_rets, bench_ticker_name))
        out = pd.Series(_ensure_expected_keys(metrics), name=name)
        out.attrs["sharpe_path"] = SHARPE_PATH_LEGACY_CAGR_VOL
        return out

    # If active_rets is not empty, proceed with normal calculation.
    # Note: _infer_steps_per_year should ideally use rets.index if active_rets is empty
    # but the functions below (sharpe, calmar etc) use active_rets.
    # The current structure means if active_rets is empty, we don't reach here.
    # If it's not empty, then active_rets.index is valid.
    steps_per_year = _infer_steps_per_year(pd.DatetimeIndex(active_rets.index))

    excess_active: pd.Series | None = None
    use_rf_effective = False
    if risk_free_rets is not None and not risk_free_rets.empty:
        rf_aligned = risk_free_rets.reindex(active_rets.index).ffill().bfill()
        excess_active = (active_rets.astype(float) - rf_aligned.astype(float)).dropna()
        use_rf_effective = excess_active is not None and not excess_active.empty

    def sharpe_traditional_excess(excess: pd.Series) -> float:
        """Annualized mean excess return / annualized volatility of excess (sample std)."""
        if excess.empty or excess.isnull().all():
            return np.nan
        steps_ex = _infer_steps_per_year(pd.DatetimeIndex(excess.index))
        mean_excess = float(excess.mean() * steps_ex)
        vol_excess = float(excess.std(ddof=1) * np.sqrt(steps_ex))
        if pd.isna(mean_excess) or pd.isna(vol_excess):
            return np.nan
        if vol_excess < EPSILON_FOR_DIVISION:
            if abs(mean_excess) < EPSILON_FOR_DIVISION:
                return 0.0
            return np.inf if mean_excess > 0 else -np.inf
        return mean_excess / vol_excess

    def sortino_ratio(r, target=0):
        if r.empty or r.isnull().all():
            return np.nan
        steps_loc = (
            _infer_steps_per_year(pd.DatetimeIndex(r.index)) if len(r) >= 1 else steps_per_year
        )
        target_returns = r - target
        downside_risk = np.sqrt(np.mean(np.minimum(0, target_returns) ** 2))

        annualized_mean_return = r.mean() * steps_loc
        if pd.isna(annualized_mean_return) or pd.isna(
            downside_risk
        ):  # If either is NaN, Sortino is NaN
            return np.nan

        if downside_risk < EPSILON_FOR_DIVISION:
            # Downside risk is effectively zero
            if abs(annualized_mean_return) < EPSILON_FOR_DIVISION:
                return 0.0  # Mean return is also zero
            elif annualized_mean_return > 0:
                return np.inf  # Positive mean return, zero downside risk
            else:
                return (
                    -np.inf
                )  # Negative mean return, zero downside risk (less common, but possible if all returns are 0 or positive but mean is negative due to target)

        # The original formula simplifies to mean_return / downside_risk_per_period, then annualize.
        # Or, (annualized_mean_return) / (annualized_downside_risk).
        # Annualized downside risk = downside_risk * np.sqrt(steps_per_year)
        annualized_downside_risk = downside_risk * np.sqrt(steps_loc)
        if (
            annualized_downside_risk < EPSILON_FOR_DIVISION
        ):  # Check again after annualization, though less likely to change category
            if abs(annualized_mean_return) < EPSILON_FOR_DIVISION:
                return 0.0
            elif annualized_mean_return > 0:
                return np.inf
            else:
                return -np.inf
        return annualized_mean_return / annualized_downside_risk

    def total_ret(x):
        return (1 + x).prod() - 1 if len(x) > 0 else np.nan

    def ann(x):
        if x.empty or x.isnull().all():
            return np.nan
        steps_a = _infer_steps_per_year(pd.DatetimeIndex(x.index))
        prod = (1 + x).prod()
        if prod < 0:
            return -1.0
        return prod ** (steps_a / len(x)) - 1

    def ann_vol(x):
        if x.empty or x.isnull().all():
            return np.nan
        steps_v = _infer_steps_per_year(pd.DatetimeIndex(x.index))
        return x.std() * np.sqrt(steps_v)

    def sharpe(x):
        if use_rf_effective and excess_active is not None:
            return sharpe_traditional_excess(excess_active)
        if x.empty or x.isnull().all():
            return np.nan
        annualized_return = ann(x)
        annualized_vol = ann_vol(x)
        if pd.isna(annualized_return) or pd.isna(
            annualized_vol
        ):  # If ann_return or ann_vol is NaN, Sharpe is NaN
            return np.nan
        if annualized_vol < EPSILON_FOR_DIVISION:
            if abs(annualized_return) < EPSILON_FOR_DIVISION:
                return 0.0  # Both are zero
            elif annualized_return > 0:
                return np.inf  # Positive return, zero volatility
            else:
                return -np.inf  # Negative return, zero volatility
        return annualized_return / annualized_vol

    def mdd(series):
        if series.empty or series.isnull().all():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"mdd: series is empty or all null. Length: {len(series)}, all null: {series.isnull().all() if not series.empty else 'N/A'}"
                )
            return np.nan

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"mdd: series length: {len(series)}, first few values: {series.head().tolist()}"
            )
            logger.debug(
                f"mdd: series has any NaN: {series.isnull().any()}, series min: {series.min()}, series max: {series.max()}"
            )

        cummax_series = series.cummax()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"mdd: cummax min: {cummax_series.min()}, cummax max: {cummax_series.max()}"
            )

        drawdown_series = series / cummax_series - 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"mdd: drawdown min: {drawdown_series.min()}, drawdown max: {drawdown_series.max()}"
            )
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
        if pd.isna(annualized_return) or pd.isna(
            max_dd
        ):  # If ann_return or max_dd is NaN, Calmar is NaN
            return np.nan
        if abs(max_dd) < EPSILON_FOR_DIVISION:
            if abs(annualized_return) < EPSILON_FOR_DIVISION:
                return 0.0  # Both are zero
            elif annualized_return > 0:
                return np.inf  # Positive return, zero drawdown
            else:
                return -np.inf  # Negative return, zero drawdown
        return annualized_return / abs(max_dd)

    def stationarity_test_equity(series):
        """ADF test on cumulative equity (price-like series)."""
        if len(series) < 40:
            return np.nan, np.nan

        cumulative_pnl = (1 + series).cumprod()

        try:
            result = adfuller(cumulative_pnl)
            return result[0], result[1]  # ADF Statistic, p-value
        except Exception:
            return np.nan, np.nan

    def stationarity_test_returns(series):
        """ADF test on per-period returns (stationarity of return process)."""
        if len(series) < 40:
            return np.nan, np.nan
        clean = series.dropna()
        if len(clean) < 40:
            return np.nan, np.nan
        try:
            result = adfuller(clean)
            return result[0], result[1]
        except Exception:
            return np.nan, np.nan

    def deflated_sharpe_ratio(rets, num_trials):
        """
        Calculates the Deflated Sharpe Ratio (DSR).
        See: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
        """
        if num_trials <= 0:
            return np.nan

        if num_trials <= 1:
            return np.nan

        rser = excess_active if use_rf_effective and excess_active is not None else rets

        if len(rser) < 100:  # DSR requires a longer series
            return np.nan

        sr = sharpe(rets)
        if pd.isna(sr):
            return np.nan

        # Guard against precision loss on near-constant returns
        std_rets_val = rser.std(ddof=1)
        std_rets = float(std_rets_val) if pd.notna(std_rets_val) else 0.0
        if std_rets <= EPSILON_FOR_DIVISION:
            # Near-constant returns – DSR undefined without emitting warnings.
            return np.nan
        # Use numerically-stable implementations for skew and kurtosis
        sk = _safe_moment(skew, rser)
        k_excess = _safe_moment(kurtosis, rser)
        n = len(rser)

        # The expected SR of random strategies is not necessarily zero.
        # We use the formula for the expected maximum SR from a set of trials.
        # This is a more robust way to deflate the SR.
        emc = 0.5772156649  # Euler-Mascheroni constant
        max_z = (1 - emc) * norm.ppf(1 - 1 / num_trials) + emc * norm.ppf(
            1 - 1 / (num_trials * np.e)
        )

        # Variance of the Sharpe Ratio estimator
        var_sr = (1 / (n - 1)) * (1 - sk * sr + (k_excess / 4) * sr**2)
        if var_sr < 0:
            return np.nan

        # The expected SR of random strategies is often assumed to be 0.
        # E[max SR_random] = E[SR_random] + StDev[SR_random] * max_z
        # Assuming E[SR_random] = 0, and StDev[SR_random] for non-autocorrelated returns
        # with 0 skew and 0 excess kurtosis is approx. sqrt(1/n).
        # More precisely, for SR estimator, it's sqrt(1/(n-1)) under H0 (true SR=0, sk=0, k_exc=0).
        if n - 1 <= 0:  # Should be caught by len(rets) < 100, but defensive
            return np.nan

        std_sr_h0 = np.sqrt(
            1 / (n - 1)
        )  # Standard deviation of SR under H0 (true SR=0, sk=0, k_exc=0)
        expected_max_sr_h0 = std_sr_h0 * max_z

        # Variance of the observed Sharpe Ratio estimator (var_sr) is already calculated:
        # var_sr = (1 / (n - 1)) * (1 - sk * sr + (k_excess / 4) * sr**2)
        if var_sr <= 0:  # Variance must be positive for a valid standard deviation
            return np.nan

        std_dev_sr_observed = np.sqrt(var_sr)

        # Deflated Sharpe Ratio Z-statistic
        # This measures how many standard deviations the observed SR is from the expected max SR under H0.
        # Guard division by near-zero variance
        if std_dev_sr_observed <= EPSILON_FOR_DIVISION or pd.isna(std_dev_sr_observed):
            return np.nan
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

    def tail_ratio_mean(rets):
        """Ratio of mean positive return to mean absolute negative return (common platform style)."""
        if rets.empty or rets.isnull().all():
            return np.nan
        positive_rets = rets[rets > 0]
        negative_rets = rets[rets < 0]
        if positive_rets.empty or negative_rets.empty:
            return np.nan
        pos_mean = float(positive_rets.mean())
        neg_mean = float(negative_rets.mean())
        if abs(neg_mean) < EPSILON_FOR_DIVISION:
            return np.inf if pos_mean > 0 else np.nan
        return pos_mean / abs(neg_mean)

    def drawdown_duration_and_recovery(equity_curve):
        """Calculate average drawdown duration and recovery time."""
        if equity_curve.empty or equity_curve.isnull().all():
            return np.nan, np.nan

        # Use optimized drawdown calculation with proper edge case handling
        try:
            return drawdown_duration_and_recovery_fast(equity_curve.values.astype(np.float64))
        except Exception:
            # Handle edge cases with insufficient data
            return np.nan, np.nan

    # Use Pandas Index intersection; ensure both are Index to avoid pylance type issue
    common_index = pd.DatetimeIndex(active_rets.index).intersection(
        pd.DatetimeIndex(active_bench_rets.index)
    )
    rets_aligned, bench_aligned = (
        active_rets.loc[common_index],
        active_bench_rets.loc[common_index],
    )
    # Infer steps per year from a DatetimeIndex explicitly to satisfy type checker
    steps_per_year = _infer_steps_per_year(common_index)

    # Debug logging
    if len(rets_aligned) == 0 or len(bench_aligned) == 0:
        logger.debug(
            f"Empty aligned series: rets_len={len(rets_aligned)}, bench_len={len(bench_aligned)}"
        )
    if not isinstance(rets_aligned, pd.Series) or not isinstance(bench_aligned, pd.Series):
        logger.debug(
            f"Unexpected types: rets_type={type(rets_aligned)}, bench_type={type(bench_aligned)}"
        )

    # Ensure enough data for CAPM regression and that strategy returns have variance
    # Handle std() calculation safely
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

    adf_eq_stat, adf_eq_p = stationarity_test_equity(active_rets)
    adf_ret_stat, adf_ret_p = stationarity_test_returns(active_rets)

    # Coefficient of determination (R^2)
    r_squared = r_squared if "r_squared" in locals() else np.nan

    # K-Ratio: slope of log equity curve scaled by its standard error
    if len(active_rets) >= 2:
        equity_for_log = (1 + active_rets.astype(float)).cumprod()
        equity_positive = equity_for_log.clip(lower=np.finfo(float).tiny)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_equity = np.log(equity_positive.to_numpy(dtype=float, copy=False))
        log_equity_std_val = np.nanstd(log_equity)
        log_equity_std = float(log_equity_std_val) if np.isfinite(log_equity_std_val) else 0.0
        if log_equity_std < EPSILON_FOR_DIVISION:
            k_ratio = np.nan
        else:
            idx = np.arange(len(log_equity))
            slope, intercept, r_value, p_value, std_err = linregress(idx, log_equity)

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
    tail_ratio_mean_v = tail_ratio_mean(active_rets)

    # Calculate drawdown metrics using full equity curve
    equity_curve = (1 + rets).cumprod()
    avg_dd_duration, avg_recovery_time = drawdown_duration_and_recovery(equity_curve)

    # Base metrics
    vol_series = excess_active if use_rf_effective and excess_active is not None else active_rets
    sortino_series = (
        excess_active if use_rf_effective and excess_active is not None else active_rets
    )
    base_metrics = {
        "Total Return": total_ret(active_rets),
        "Ann. Return": ann(active_rets),
        "Ann. Vol": ann_vol(vol_series),
        "Sharpe": sharpe(active_rets),
        "Sortino": sortino_ratio(sortino_series),
        "Calmar": calmar(active_rets),
        "Alpha (ann)": alpha,
        "Beta": beta,
        "Max Drawdown": mdd(
            (1 + rets).cumprod()
        ),  # Use original rets, not active_rets - zero returns are valid for drawdown
        "VaR (5%)": var_5pct,
        "CVaR (5%)": cvar_5pct,
        "Tail Ratio": tail_ratio_95,
        "Tail Ratio (mean)": tail_ratio_mean_v,
        "Avg DD Duration": avg_dd_duration,
        "Avg Recovery Time": avg_recovery_time,
        # For summary metrics, emit a RuntimeWarning on near-constant data to match tests' expectation,
        # while still returning 0.0 for Skew/Kurtosis values to avoid NaNs in tables.
        "Skew": _safe_moment(skew, active_rets),
        "Kurtosis": _safe_moment(kurtosis, active_rets),
        "R^2": r_squared,
        "K-Ratio": k_ratio,
        "ADF Statistic (equity)": adf_eq_stat,
        "ADF p-value (equity)": adf_eq_p,
        "ADF Statistic (returns)": adf_ret_stat,
        "ADF p-value (returns)": adf_ret_p,
        "Deflated Sharpe": deflated_sharpe_ratio(active_rets, num_trials),
    }

    # Add trade-based metrics if available
    if trade_stats is not None:
        trade_metrics = {}

        # Add directional trade statistics (All/Long/Short)
        for direction in ["all", "long", "short"]:
            direction_title = direction.title()
            prefix = f"{direction}_"

            trade_metrics.update(
                {
                    f"Number of Trades ({direction_title})": trade_stats.get(
                        f"{prefix}num_trades", 0
                    ),
                    f"Win Rate % ({direction_title})": trade_stats.get(
                        f"{prefix}win_rate_pct", 0.0
                    ),
                    f"Number of Winners ({direction_title})": trade_stats.get(
                        f"{prefix}num_winners", 0
                    ),
                    f"Number of Losers ({direction_title})": trade_stats.get(
                        f"{prefix}num_losers", 0
                    ),
                    f"Total P&L Net ({direction_title})": trade_stats.get(
                        f"{prefix}total_pnl_net", 0.0
                    ),
                    f"Largest Single Profit ({direction_title})": trade_stats.get(
                        f"{prefix}largest_profit", 0.0
                    ),
                    f"Largest Single Loss ({direction_title})": trade_stats.get(
                        f"{prefix}largest_loss", 0.0
                    ),
                    f"Mean Profit ({direction_title})": trade_stats.get(
                        f"{prefix}mean_profit", 0.0
                    ),
                    f"Mean Loss ({direction_title})": trade_stats.get(f"{prefix}mean_loss", 0.0),
                    f"Mean Trade P&L ({direction_title})": trade_stats.get(
                        f"{prefix}mean_trade_pnl", 0.0
                    ),
                    f"Reward/Risk Ratio ({direction_title})": trade_stats.get(
                        f"{prefix}reward_risk_ratio", 0.0
                    ),
                    f"Commissions Paid ({direction_title})": trade_stats.get(
                        f"{prefix}total_commissions_paid", 0.0
                    ),
                    f"Avg MFE ({direction_title})": trade_stats.get(f"{prefix}avg_mfe", 0.0),
                    f"Avg MAE ({direction_title})": trade_stats.get(f"{prefix}avg_mae", 0.0),
                    f"Information Score ({direction_title})": trade_stats.get(
                        f"{prefix}information_score", 0.0
                    ),
                    f"Min Trade Duration Days ({direction_title})": trade_stats.get(
                        f"{prefix}min_trade_duration_days", 0
                    ),
                    f"Max Trade Duration Days ({direction_title})": trade_stats.get(
                        f"{prefix}max_trade_duration_days", 0
                    ),
                    f"Mean Trade Duration Days ({direction_title})": trade_stats.get(
                        f"{prefix}mean_trade_duration_days", 0.0
                    ),
                    f"Trades per Month ({direction_title})": trade_stats.get(
                        f"{prefix}trades_per_month", 0.0
                    ),
                }
            )

        # Add portfolio-level metrics (not direction-specific)
        trade_metrics.update(
            {
                "Max Margin Load": trade_stats.get("max_margin_load", 0.0),
                "Mean Margin Load": trade_stats.get("mean_margin_load", 0.0),
            }
        )

        base_metrics.update(trade_metrics)

    # Calculate additional derived metrics
    max_dd_recovery_time = calculate_max_dd_recovery_time(rets)
    max_dd_recovery_bars = calculate_max_dd_recovery_bars(rets)
    max_flat_period = calculate_max_flat_period(rets)

    base_metrics.update(
        {
            "Max DD Recovery Time (days)": max_dd_recovery_time,
            "Max DD Recovery Time (bars)": max_dd_recovery_bars,
            "Max Flat Period (days)": max_flat_period,
        }
    )

    metrics = pd.Series(base_metrics, name=name)
    metrics.attrs["sharpe_path"] = (
        SHARPE_PATH_EXCESS if use_rf_effective else SHARPE_PATH_LEGACY_CAGR_VOL
    )
    return metrics


def calculate_max_dd_recovery_time(rets):
    """Calculate maximum drawdown recovery time in days."""
    if rets.empty or not isinstance(rets.index, pd.DatetimeIndex):
        logger.warning(
            "Cannot calculate max_dd_recovery_time: rets is empty or index is not a DatetimeIndex."
        )
        return 0

    logger.debug(
        f"Calculating max_dd_recovery_time for rets with length {len(rets)} and index from {rets.index.min()} to {rets.index.max()}"
    )

    try:
        # Calculate cumulative returns
        cumulative = (1 + rets).cumprod()

        # Calculate running maximum and drawdown
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1

        # Find recovery periods
        max_recovery_time = 0
        in_drawdown = False
        drawdown_start = None

        for i, (date, dd_value) in enumerate(drawdown.items()):
            if dd_value < -1e-6 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = date
            elif dd_value >= -1e-6 and in_drawdown:
                # End of drawdown (recovery)
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_time = (date - drawdown_start).days
                    max_recovery_time = max(max_recovery_time, recovery_time)

        logger.debug(f"Successfully calculated max_dd_recovery_time: {max_recovery_time}")
        return max_recovery_time
    except Exception as e:
        logger.warning(f"Could not calculate max drawdown recovery time: {e}")
        logger.debug(f"rets that caused the error:\n{rets.to_string()}")
        return np.nan


def calculate_max_dd_recovery_bars(rets):
    """Maximum drawdown recovery length in observation bars (episode logic aligned with calendar-days helper).

    Counts bars from first observation strictly underwater until recovery to prior peak watermark,
    using the same threshold heuristic as :func:`calculate_max_dd_recovery_time`.
    """
    if rets.empty or not isinstance(rets.index, pd.DatetimeIndex):
        return 0
    try:
        cumulative = (1 + rets).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        max_recovery_bars = 0
        in_drawdown = False
        start_i: int | None = None
        for i, dd_value in enumerate(drawdown):
            if dd_value < -1e-6 and not in_drawdown:
                in_drawdown = True
                start_i = i
            elif dd_value >= -1e-6 and in_drawdown:
                in_drawdown = False
                if start_i is not None:
                    max_recovery_bars = max(max_recovery_bars, i - start_i)
        return max_recovery_bars
    except Exception:
        return np.nan


def calculate_max_flat_period(rets):
    """Calculate maximum flat period (consecutive days with zero returns)."""
    if rets.empty:
        return 0

    # Check if returns are all non-zero
    all_nonzero = True
    for ret in rets:
        if abs(ret) < EPSILON_FOR_DIVISION:
            all_nonzero = False
            break

    if all_nonzero:
        return 0  # No flat periods if all returns are non-zero

    max_flat = 0
    current_flat = 0

    for ret in rets:
        if abs(ret) < EPSILON_FOR_DIVISION:  # Use the same threshold as elsewhere
            current_flat += 1
            max_flat = max(max_flat, current_flat)
        else:
            current_flat = 0

    return max_flat
