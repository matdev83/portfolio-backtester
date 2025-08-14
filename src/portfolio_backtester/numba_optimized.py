"""
Numba-optimized functions for high-performance backtesting.

This module contains JIT-compiled functions that provide significant speedups
for computationally intensive operations in the backtesting pipeline.

The functions are organized into the following categories:
1. Basic Mathematical Operations
2. Rolling Window Calculations
3. Technical Indicators
4. Risk Metrics (Sharpe, Sortino, Beta, Correlation)
5. Advanced Risk Metrics (VAMS, Calmar, Drawdown)
6. Monte Carlo Simulation
7. Batch Processing Functions

All functions use Numba's JIT compilation for optimal performance and are
designed to match pandas behavior exactly, particularly for statistical
calculations using sample standard deviation (ddof=1).
"""

import logging
import os

import numpy as np

# Single-path architecture: Numba is a hard dependency
import numba
from numba import prange, set_num_threads

# Configure Numba threading
threads_env = os.environ.get("NUMBA_NUM_THREADS")
try:
    threads = int(threads_env) if threads_env else (os.cpu_count() or 1)
except Exception:
    threads = os.cpu_count() or 1
if threads < 1:
    threads = 1
set_num_threads(threads)

logger = logging.getLogger(__name__)
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Numba configured with {threads} threads")


# =============================================================================
# 1. BASIC MATHEMATICAL OPERATIONS
# =============================================================================


@numba.jit(nopython=True, cache=True)
def calculate_returns_fast(prices):
    """
    Calculate percentage returns from price series.

    Computes the percentage change between consecutive elements in a NumPy array,
    handling NaN values and zero/negative prices appropriately.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of price values.

    Returns
    -------
    np.ndarray
        1-D array of returns, with the first element as NaN.

    Notes
    -----
    - Returns are calculated as (price[t] / price[t-1]) - 1
    - NaN values are propagated appropriately
    - Zero or negative prices result in NaN returns
    """
    returns = np.full_like(prices, np.nan)
    for i in range(1, len(prices)):
        if (
            not np.isnan(prices[i - 1])
            and prices[i - 1] > 0
            and not np.isnan(prices[i])
        ):
            returns[i] = (prices[i] / prices[i - 1]) - 1.0
    return returns


@numba.jit(nopython=True, cache=True)
def momentum_scores_fast_vectorized(prices_now, prices_then):
    """
    Calculate vectorized momentum scores for multiple assets.

    Computes momentum as (current_price / past_price) - 1 for all assets
    simultaneously, with proper handling of invalid values.

    Parameters
    ----------
    prices_now : np.ndarray
        Current prices for all assets.
    prices_then : np.ndarray
        Historical prices for comparison.

    Returns
    -------
    np.ndarray
        Momentum scores with same shape as input arrays.

    Notes
    -----
    - Momentum = (price_now / price_then) - 1
    - Invalid prices (NaN, zero, negative) result in NaN momentum
    """
    valid_mask = (prices_then > 0) & (~np.isnan(prices_then)) & (~np.isnan(prices_now))
    result = np.full_like(prices_now, np.nan)
    result[valid_mask] = (prices_now[valid_mask] / prices_then[valid_mask]) - 1.0
    return result


@numba.jit(nopython=True, cache=True)
def returns_to_prices_fast(returns: np.ndarray, initial_price: float) -> np.ndarray:
    """
    Convert returns series to price levels with safeguards.

    Transforms a series of returns into cumulative price levels, starting
    from an initial price. Includes safeguards against extreme values that
    could cause mathematical issues.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns in decimal form (e.g., 0.05 for 5%).
    initial_price : float
        Starting price level.

    Returns
    -------
    np.ndarray
        Array of price levels.

    Notes
    -----
    - Extreme returns are clipped to prevent negative prices
    - Maximum negative return is capped at -50%
    - Maximum positive return is capped at 500%
    """
    # Clip extreme outliers that would cause mathematical issues
    max_negative_return = -0.50  # Maximum -50% return to prevent negative prices
    clipped_returns = np.clip(returns, max_negative_return, 5.0)  # Cap at 500% gain too

    # Convert to price relatives
    price_relatives = 1.0 + clipped_returns

    # Ensure positive price relatives
    price_relatives = np.maximum(price_relatives, 0.05)

    # Calculate cumulative product for prices
    prices = np.zeros(len(price_relatives))
    prices[0] = initial_price * price_relatives[0]

    for i in range(1, len(price_relatives)):
        prices[i] = prices[i - 1] * price_relatives[i]

    return prices


@numba.jit(nopython=True, cache=True)
def mdd_fast(series):
    """
    Calculate maximum drawdown from a price/equity series.

    Computes the maximum peak-to-trough decline in a price or equity curve,
    which is a key risk metric for trading strategies.

    Parameters
    ----------
    series : np.ndarray
        Price or equity curve values.

    Returns
    -------
    float
        Maximum drawdown as a decimal (negative value).

    Notes
    -----
    - Drawdown is calculated as (current_value / running_maximum) - 1
    - Result is always <= 0, with more negative values indicating larger drawdowns
    """
    cummax = np.maximum.accumulate(series)
    drawdown = (series / cummax) - 1
    return np.min(drawdown)


# =============================================================================
# 2. ROLLING WINDOW CALCULATIONS
# =============================================================================


def create_jitted_rolling_fn(stat_func, func_name, annualization_factor=1.0):
    """
    Higher-order function to create JIT-compiled rolling window functions.

    Creates optimized rolling window functions for various statistical operations,
    with proper handling of NaN values and configurable annualization.

    Parameters
    ----------
    stat_func : callable
        The Numba-compatible statistical function to apply (e.g., np.mean, np.std).
    func_name : str
        Name to assign to the generated function for debugging.
    annualization_factor : float, optional
        Factor to multiply the result by (e.g., sqrt(252) for volatility).

    Returns
    -------
    callable
        JIT-compiled function that performs the rolling calculation.

    Notes
    -----
    - Handles NaN values by requiring at least window//2 valid observations
    - Uses efficient sliding window approach
    - Results match pandas rolling() behavior
    """

    @numba.jit(nopython=True, cache=True)
    def rolling_fn(data, window):
        n = len(data)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            window_data = data[i - window + 1 : i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= window // 2:
                stat = stat_func(valid_data)
                result[i] = stat * annualization_factor
        return result

    rolling_fn.__name__ = func_name
    return rolling_fn


# Create commonly used rolling functions
rolling_mean_fast = create_jitted_rolling_fn(np.mean, "rolling_mean_fast")


@numba.jit(nopython=True, cache=True)
def rolling_std_fast(data, window):
    """
    Calculate rolling standard deviation with pandas-compatible behavior.

    Computes rolling standard deviation using sample standard deviation (ddof=1)
    to match pandas rolling().std() behavior exactly.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling standard deviation values.

    Notes
    -----
    - Uses ddof=1 (sample standard deviation) to match pandas
    - Requires at least window//2 valid observations
    - Single value windows return NaN (consistent with pandas)
    """
    n = len(data)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= window // 2 and len(valid_data) > 1:
            # Use ddof=1 to match pandas behavior (sample standard deviation)
            mean_val = np.mean(valid_data)
            variance = np.sum((valid_data - mean_val) ** 2) / (
                len(valid_data) - 1
            )  # ddof=1
            result[i] = np.sqrt(variance)
        elif len(valid_data) == 1:
            # Single value cannot have sample standard deviation (ddof=1) - return NaN to match pandas
            result[i] = np.nan

    return result


# Backward compatibility alias expected by tests
rolling_std_fixed = rolling_std_fast


@numba.jit(nopython=True, cache=True)
def rolling_cumprod_fast(data, window):
    """
    Calculate rolling cumulative product for momentum calculations.

    Computes rolling momentum as the cumulative product of (1 + returns) - 1
    over a specified window, which is equivalent to compound returns.

    Parameters
    ----------
    data : np.ndarray
        Input data array (typically returns).
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling momentum values (cumulative product - 1).

    Notes
    -----
    - Calculates (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    - Any NaN in the window results in NaN output
    - Equivalent to pandas (1 + returns).rolling(window).apply(np.prod) - 1
    """
    n = len(data)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]

        # Check if any NaN in window - if so, result is NaN
        has_nan = False
        for j in range(len(window_data)):
            if np.isnan(window_data[j]):
                has_nan = True
                break

        if has_nan:
            result[i] = np.nan
        else:
            # Calculate cumulative product: (1 + r1) * (1 + r2) * ... - 1
            prod = 1.0
            for j in range(len(window_data)):
                prod *= 1.0 + window_data[j]
            result[i] = prod - 1.0

    return result


# Backward compatibility alias expected by tests
rolling_cumprod_fixed = rolling_cumprod_fast


@numba.jit(nopython=True, cache=True)
def rolling_downside_volatility_fast(data, window):
    """
    Calculate rolling downside volatility using only negative returns.

    Computes volatility based only on negative returns, which is useful
    for downside risk assessment and Sortino ratio calculations.

    Parameters
    ----------
    data : np.ndarray
        Input price data.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling downside volatility values.

    Notes
    -----
    - Only considers negative returns for volatility calculation
    - Uses sample standard deviation (ddof=1) for consistency
    - Returns 0.0 when no negative returns are present
    """
    n = len(data)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= window // 2:
            # Calculate returns from prices
            if len(valid_data) > 1:
                returns = np.diff(valid_data) / valid_data[:-1]
                # Only use negative returns for downside volatility
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 1:
                    # Use ddof=1 to match pandas behavior (sample standard deviation)
                    mean_val = np.mean(downside_returns)
                    variance = np.sum((downside_returns - mean_val) ** 2) / (
                        len(downside_returns) - 1
                    )
                    result[i] = np.sqrt(variance)
                elif len(downside_returns) == 1:
                    result[i] = 0.0
                else:
                    result[i] = 0.0
            else:
                result[i] = 0.0

    return result


# =============================================================================
# 3. TECHNICAL INDICATORS
# =============================================================================


@numba.jit(nopython=True, cache=True)
def ema_fast(data, window):
    """
    Calculate exponential moving average with proper NaN handling.

    Computes exponential moving average using the standard EMA formula
    with alpha = 2/(window+1), handling NaN values appropriately.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    window : int
        EMA window parameter (used to calculate alpha).

    Returns
    -------
    np.ndarray
        Exponential moving average values.

    Notes
    -----
    - Alpha = 2 / (window + 1)
    - Handles NaN values by carrying forward last valid EMA
    - First valid data point initializes the EMA
    """
    n = len(data)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (window + 1.0)

    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return ema

    ema[first_valid_idx] = data[first_valid_idx]

    for i in range(first_valid_idx + 1, n):
        if not np.isnan(data[i]):
            prev_ema = ema[i - 1]
            if np.isnan(prev_ema):
                last_valid_ema_idx = -1
                for j in range(i - 1, -1, -1):
                    if not np.isnan(ema[j]):
                        last_valid_ema_idx = j
                        break
                if last_valid_ema_idx != -1:
                    prev_ema = ema[last_valid_ema_idx]
                else:
                    prev_ema = data[i]

            ema[i] = alpha * data[i] + (1 - alpha) * prev_ema
        else:
            if i > 0:
                ema[i] = ema[i - 1]

    return ema


@numba.jit(nopython=True, cache=True)
def true_range_fast(high, low, close_prev):
    """
    Calculate True Range for ATR calculations.

    Computes True Range as the maximum of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close_prev : np.ndarray
        Previous period's closing prices.

    Returns
    -------
    np.ndarray
        True Range values.

    Notes
    -----
    - Vectorized implementation for optimal performance
    - Handles all three True Range components simultaneously
    """
    return np.maximum(
        high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev))
    )


@numba.jit(nopython=True, cache=True)
def atr_fast(high, low, close, window):
    """
    Calculate Average True Range using simple moving average.

    Computes ATR as the simple moving average of True Range values
    over the specified window period.

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Closing prices.
    window : int
        ATR calculation window.

    Returns
    -------
    np.ndarray
        ATR values.

    Notes
    -----
    - Uses simple moving average of True Range
    - First period's close is set to NaN for True Range calculation
    """
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    tr = true_range_fast(high, low, close_prev)
    return rolling_mean_fast(tr, window)


@numba.jit(nopython=True, cache=True)
def atr_exponential_fast(high, low, close, window):
    """
    Calculate Average True Range using exponential moving average.

    Computes ATR as the exponential moving average of True Range values,
    which gives more weight to recent observations.

    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Closing prices.
    window : int
        EMA window parameter for ATR calculation.

    Returns
    -------
    np.ndarray
        Exponential ATR values.

    Notes
    -----
    - Uses exponential moving average for smoother ATR
    - More responsive to recent price action than simple ATR
    """
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    tr = true_range_fast(high, low, close_prev)
    return ema_fast(tr, window)


@numba.jit(nopython=True, cache=True)
def atr_fast_fixed(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """
    Calculate Average True Range with comprehensive edge case handling.

    Robust ATR implementation that matches pandas behavior exactly
    and handles all edge cases properly.

    Parameters
    ----------
    high : np.ndarray
        High prices array.
    low : np.ndarray
        Low prices array.
    close : np.ndarray
        Close prices array.
    window : int
        Rolling window length for ATR calculation.

    Returns
    -------
    np.ndarray
        ATR values with same length as input arrays.

    Notes
    -----
    - Handles NaN values appropriately
    - Ensures logical price ordering (high >= low)
    - Uses simple moving average of True Range
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n < 2 or window < 1:
        return result

    # Calculate True Range
    tr = np.full(n, np.nan)

    for i in range(1, n):
        if not (
            np.isnan(high[i])
            or np.isnan(low[i])
            or np.isnan(close[i])
            or np.isnan(close[i - 1])
        ):
            # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, max(hc, lc))

    # Calculate rolling mean of True Range (ATR)
    for i in range(window - 1, n):
        window_tr = tr[i - window + 1 : i + 1]

        # Count valid (non-NaN) values
        valid_count = 0
        tr_sum = 0.0
        for j in range(len(window_tr)):
            if not np.isnan(window_tr[j]):
                valid_count += 1
                tr_sum += window_tr[j]

        if valid_count > 0:
            result[i] = tr_sum / valid_count

    return result


# =============================================================================
# 4. RISK METRICS (Sharpe, Sortino, Beta, Correlation)
# =============================================================================


@numba.jit(nopython=True, cache=True)
def rolling_sharpe_fast(returns, window, annualization_factor=1.0):
    """
    Calculate rolling Sharpe ratio with pandas-compatible behavior.

    Computes rolling Sharpe ratio using sample standard deviation (ddof=1)
    to ensure consistency with pandas calculations.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    window : int
        Rolling window size.
    annualization_factor : float, optional
        Annualization factor (e.g., 252 for daily data, 12 for monthly).

    Returns
    -------
    np.ndarray
        Rolling Sharpe ratio values.

    Notes
    -----
    - Uses sample standard deviation (ddof=1) for consistency with pandas
    - Sharpe = (mean_return / std_return) * sqrt(annualization_factor)
    - Returns 0.0 for very low volatility periods
    """
    n = len(returns)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_returns = returns[i - window + 1 : i + 1]
        valid_returns = window_returns[~np.isnan(window_returns)]

        if len(valid_returns) >= window // 2 and len(valid_returns) > 1:
            mean_ret = np.mean(valid_returns)
            # Use ddof=1 for sample standard deviation to match pandas
            variance = np.sum((valid_returns - mean_ret) ** 2) / (
                len(valid_returns) - 1
            )
            std_ret = np.sqrt(variance)
            if std_ret > 1e-10:
                sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor)
                result[i] = sharpe
            else:
                result[i] = 0.0
        elif len(valid_returns) == 1:
            result[i] = 0.0
    return result


@numba.jit(nopython=True, cache=True)
def rolling_sortino_fast(returns, window, target_return=0.0, annualization_factor=1.0):
    """
    Calculate rolling Sortino ratio with proper downside deviation.

    Computes Sortino ratio using only negative returns for the denominator,
    providing a better measure of downside risk than standard Sharpe ratio.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    window : int
        Rolling window size.
    target_return : float, optional
        Target return for downside calculation (default 0.0).
    annualization_factor : float, optional
        Annualization factor for scaling.

    Returns
    -------
    np.ndarray
        Rolling Sortino ratio values.

    Notes
    -----
    - Uses sample standard deviation (ddof=1) for downside deviation
    - Only considers returns below target_return for denominator
    - Returns NaN when insufficient downside observations
    """
    n = len(returns)
    result = np.full(n, np.nan)

    for i in range(n):
        if i >= window - 1:
            window_returns = returns[i - window + 1 : i + 1]
            valid_returns = window_returns[~np.isnan(window_returns)]

            if len(valid_returns) >= window:
                mean_ret = np.mean(valid_returns)

                downside_returns = valid_returns[valid_returns < target_return]
                if len(downside_returns) > 1:
                    # Use ddof=1 for sample standard deviation to match pandas
                    mean_downside = np.mean(downside_returns)
                    downside_variance = np.sum(
                        (downside_returns - mean_downside) ** 2
                    ) / (len(downside_returns) - 1)
                    downside_dev = np.sqrt(downside_variance)
                    if downside_dev > 1e-10:
                        sortino = ((mean_ret - target_return) / downside_dev) * np.sqrt(
                            annualization_factor
                        )
                        result[i] = sortino
                    else:
                        result[i] = np.nan  # Match pandas behavior for zero downside
                elif len(downside_returns) == 1:
                    # Single downside return has zero deviation
                    result[i] = np.nan
                else:
                    # No downside returns - pandas returns NaN in this case
                    result[i] = np.nan
    return result


@numba.jit(nopython=True, cache=True)
def rolling_beta_fast(asset_returns, benchmark_returns, window):
    """
    Calculate rolling beta coefficient between asset and benchmark.

    Computes rolling beta as the ratio of covariance to benchmark variance,
    measuring the asset's sensitivity to benchmark movements.

    Parameters
    ----------
    asset_returns : np.ndarray
        Asset return series.
    benchmark_returns : np.ndarray
        Benchmark return series.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling beta values.

    Notes
    -----
    - Beta = Covariance(asset, benchmark) / Variance(benchmark)
    - Uses sample statistics for consistency with pandas
    - Returns 0.0 for very low benchmark volatility
    """
    n = len(asset_returns)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        asset_window = asset_returns[i - window + 1 : i + 1]
        bench_window = benchmark_returns[i - window + 1 : i + 1]

        valid_mask = (~np.isnan(asset_window)) & (~np.isnan(bench_window))
        valid_asset = asset_window[valid_mask]
        valid_bench = bench_window[valid_mask]

        if len(valid_asset) >= window // 2:
            bench_var = np.var(valid_bench)
            if bench_var > 1e-10:
                covariance = np.cov(valid_asset, valid_bench)[0, 1]
                result[i] = covariance / bench_var
            else:
                result[i] = 0.0
    return result


@numba.jit(nopython=True, cache=True)
def rolling_correlation_fast(asset_returns, benchmark_returns, window):
    """
    Calculate rolling correlation between asset and benchmark returns.

    Computes Pearson correlation coefficient over a rolling window,
    measuring the linear relationship strength between two return series.

    Parameters
    ----------
    asset_returns : np.ndarray
        Asset return series.
    benchmark_returns : np.ndarray
        Benchmark return series.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling correlation values.

    Notes
    -----
    - Correlation = Covariance / (std_asset * std_benchmark)
    - Uses sample standard deviation (ddof=1) for consistency
    - Returns 0.0 for very low volatility periods
    """
    n = len(asset_returns)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        asset_window = asset_returns[i - window + 1 : i + 1]
        bench_window = benchmark_returns[i - window + 1 : i + 1]

        valid_mask = (~np.isnan(asset_window)) & (~np.isnan(bench_window))
        valid_asset = asset_window[valid_mask]
        valid_bench = bench_window[valid_mask]

        if len(valid_asset) >= window // 2 and len(valid_asset) > 1:
            # Use ddof=1 for sample standard deviation to match pandas
            asset_mean = np.mean(valid_asset)
            bench_mean = np.mean(valid_bench)
            asset_variance = np.sum((valid_asset - asset_mean) ** 2) / (
                len(valid_asset) - 1
            )
            bench_variance = np.sum((valid_bench - bench_mean) ** 2) / (
                len(valid_bench) - 1
            )
            asset_std = np.sqrt(asset_variance)
            bench_std = np.sqrt(bench_variance)

            if asset_std > 1e-10 and bench_std > 1e-10:
                covariance = np.sum(
                    (valid_asset - asset_mean) * (valid_bench - bench_mean)
                ) / (len(valid_asset) - 1)
                result[i] = covariance / (asset_std * bench_std)
            else:
                result[i] = 0.0
        elif len(valid_asset) == 1:
            result[i] = 0.0
    return result


@numba.jit(nopython=True, cache=True)
def rolling_beta_fast_portfolio(port_prices, mkt_prices, lookback_months):
    """
    Calculate trailing beta for portfolio vs market over specified lookback.

    Computes the most recent beta estimate using the specified lookback period,
    typically used for portfolio risk assessment and hedging decisions.

    Parameters
    ----------
    port_prices : np.ndarray
        Portfolio price series (monthly closes).
    mkt_prices : np.ndarray
        Market/benchmark price series (monthly closes).
    lookback_months : int
        Lookback window in months for beta estimation.

    Returns
    -------
    float
        Latest beta estimate (1.0 if insufficient data).

    Notes
    -----
    - Returns single beta value (not time series)
    - Uses most recent lookback_months of data
    - Fallback to 1.0 beta when insufficient data or zero market variance
    """
    n = len(port_prices)
    m = len(mkt_prices)
    if n == 0 or m == 0:
        return 1.0

    # Use common length
    L = n if n < m else m
    start_idx = 1  # need previous price for return

    # Compute returns arrays aligned to last L observations
    port_rets = np.full(L - start_idx, np.nan)
    mkt_rets = np.full(L - start_idx, np.nan)
    for i in range(start_idx, L):
        p_prev = port_prices[i - 1]
        p_cur = port_prices[i]
        m_prev = mkt_prices[i - 1]
        m_cur = mkt_prices[i]
        if (
            p_prev > 0
            and m_prev > 0
            and not (
                np.isnan(p_prev)
                or np.isnan(p_cur)
                or np.isnan(m_prev)
                or np.isnan(m_cur)
            )
        ):
            port_rets[i - start_idx] = (p_cur / p_prev) - 1.0
            mkt_rets[i - start_idx] = (m_cur / m_prev) - 1.0

    # Determine window
    window = lookback_months
    if window > len(port_rets):
        window = len(port_rets)
    if window < 3:
        return 1.0

    # Slice last *window* observations
    port_window = port_rets[-window:]
    mkt_window = mkt_rets[-window:]

    # Remove NaNs synchronously
    valid_count = 0
    sum_port = 0.0
    sum_mkt = 0.0
    for i in range(window):
        if not (np.isnan(port_window[i]) or np.isnan(mkt_window[i])):
            valid_count += 1
            sum_port += port_window[i]
            sum_mkt += mkt_window[i]
    if valid_count < window // 2:
        return 1.0

    mean_port = sum_port / valid_count
    mean_mkt = sum_mkt / valid_count

    cov = 0.0
    var_mkt = 0.0
    for i in range(window):
        pr = port_window[i]
        mr = mkt_window[i]
        if not (np.isnan(pr) or np.isnan(mr)):
            cov += (pr - mean_port) * (mr - mean_mkt)
            var_mkt += (mr - mean_mkt) * (mr - mean_mkt)

    if var_mkt > 1e-10:
        return cov / var_mkt
    else:
        return 1.0


@numba.jit(nopython=True, cache=True)
def rolling_sharpe_fast_portfolio(prices, window_months, annualization_factor=12.0):
    """
    Calculate rolling Sharpe ratio for single asset using price data.

    Computes Sharpe ratio from price series by first calculating returns,
    then applying rolling Sharpe calculation with proper annualization.

    Parameters
    ----------
    prices : np.ndarray
        Price series (daily prices).
    window_months : int
        Window length in months (converted to days using 21 days/month).
    annualization_factor : float, optional
        Annualization factor (default 12.0 for monthly).

    Returns
    -------
    np.ndarray
        Rolling Sharpe ratio time series.

    Notes
    -----
    - Converts monthly window to daily using 21 trading days/month
    - Uses sample standard deviation for consistency
    - Handles insufficient data gracefully
    """
    n = len(prices)
    out = np.full(n, np.nan)
    if n == 0 or window_months <= 0:
        return out

    # Compute daily returns
    rets = np.full(n, np.nan)
    for i in range(1, n):
        prev = prices[i - 1]
        cur = prices[i]
        if prev > 0 and not (np.isnan(prev) or np.isnan(cur)):
            rets[i] = (cur / prev) - 1.0

    window_days = window_months * 21  # Approximate trading days per month

    for i in range(window_days, n):
        # Accumulate statistics for the window
        sum_ret = 0.0
        sum_sq_ret = 0.0
        count = 0
        for j in range(i - window_days + 1, i + 1):
            r = rets[j]
            if not np.isnan(r):
                sum_ret += r
                sum_sq_ret += r * r
                count += 1
        if count >= window_days // 2 and count > 1:
            mean = sum_ret / count
            var = (sum_sq_ret / count) - (mean * mean)
            if var > 1e-10:
                std = np.sqrt(var)
                out[i] = (mean / std) * np.sqrt(annualization_factor)
            else:
                out[i] = 0.0
    return out


# =============================================================================
# 5. ADVANCED RISK METRICS (VAMS, Calmar, Drawdown)
# =============================================================================


@numba.jit(nopython=True, cache=True)
def vams_fast(prices, lookback_months, alpha=0.5):
    """
    Calculate Volatility Adjusted Momentum Score (DPVAMS) for single asset.

    Computes DPVAMS as momentum minus alpha times downside volatility,
    providing a risk-adjusted momentum measure that penalizes downside risk.

    Parameters
    ----------
    prices : np.ndarray
        Monthly price series.
    lookback_months : int
        Lookback window length in months.
    alpha : float, optional
        Downside penalty coefficient (default 0.5).

    Returns
    -------
    np.ndarray
        DPVAMS values (NaN for insufficient history periods).

    Notes
    -----
    - DPVAMS = momentum - alpha * downside_volatility
    - Momentum = (price_now / price_then) - 1
    - Downside volatility uses only negative returns
    - Uses population variance for downside volatility calculation
    """
    n = len(prices)
    out = np.full(n, np.nan)

    if n == 0 or lookback_months <= 0:
        return out

    # Pre-compute simple returns (monthly) once
    rets = np.full(n, np.nan)
    for i in range(1, n):
        p_prev = prices[i - 1]
        p_now = prices[i]
        if p_prev > 0 and not (np.isnan(p_prev) or np.isnan(p_now)):
            rets[i] = (p_now / p_prev) - 1.0

    # Rolling loop
    for i in range(lookback_months, n):
        p_now = prices[i]
        p_then = prices[i - lookback_months]
        if p_then <= 0 or np.isnan(p_now) or np.isnan(p_then):
            continue

        # Momentum component
        momentum = (p_now / p_then) - 1.0

        # Downside volatility over window [i-lookback_months+1, i]
        neg_sum = 0.0
        neg_sumsq = 0.0
        count = 0
        for j in range(i - lookback_months + 1, i + 1):
            r = rets[j]
            if not np.isnan(r) and r < 0.0:
                neg_sum += r
                neg_sumsq += r * r
                count += 1
        if count > 0:
            mean_neg = neg_sum / count
            var_neg = (neg_sumsq / count) - (mean_neg * mean_neg)
            if var_neg < 0.0:
                var_neg = 0.0
            downside_vol = np.sqrt(var_neg)
        else:
            downside_vol = 0.0

        out[i] = momentum - alpha * downside_vol
    return out


@numba.jit(nopython=True, cache=True)
def drawdown_duration_and_recovery_fast(equity_curve: np.ndarray) -> tuple:
    """
    Calculate average drawdown duration and recovery time from equity curve.

    Analyzes equity curve to identify drawdown periods and their recovery times,
    providing insights into the temporal characteristics of portfolio losses.

    Parameters
    ----------
    equity_curve : np.ndarray
        Equity curve values (cumulative returns).

    Returns
    -------
    tuple
        (average_drawdown_duration, average_recovery_time) in periods.

    Notes
    -----
    - Drawdown period: time from peak to trough
    - Recovery period: time from trough back to previous peak
    - Uses epsilon threshold to identify significant drawdowns
    - Handles ongoing drawdowns at period end
    """
    n = len(equity_curve)
    if n < 2:
        return np.nan, np.nan

    # Calculate running maximum
    running_max = np.full(n, np.nan)
    running_max[0] = equity_curve[0]
    for i in range(1, n):
        running_max[i] = max(running_max[i - 1], equity_curve[i])

    # Calculate drawdown
    drawdown = np.full(n, np.nan)
    for i in range(n):
        if running_max[i] > 0:
            drawdown[i] = (equity_curve[i] / running_max[i]) - 1.0
        else:
            drawdown[i] = 0.0

    # Find drawdown periods
    epsilon = 1e-9
    drawdown_periods = []
    recovery_periods = []

    in_drawdown = False
    drawdown_start = -1

    for i in range(n):
        if drawdown[i] < -epsilon and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            drawdown_start = i
        elif drawdown[i] >= -epsilon and in_drawdown:
            # End of drawdown
            in_drawdown = False
            if drawdown_start >= 0:
                duration = i - drawdown_start
                drawdown_periods.append(duration)

                # Find recovery period (time to reach new high)
                peak_before_dd = running_max[drawdown_start]
                recovery_start = i

                recovery_found = False
                for j in range(recovery_start, n):
                    if equity_curve[j] >= peak_before_dd:
                        recovery_time = j - i
                        recovery_periods.append(recovery_time)
                        recovery_found = True
                        break

                if not recovery_found:
                    # Still in recovery at end of period
                    recovery_time = n - 1 - i
                    recovery_periods.append(recovery_time)

    # Handle case where period ends in drawdown
    if in_drawdown and drawdown_start >= 0:
        duration = n - drawdown_start
        drawdown_periods.append(duration)

    # Calculate averages
    if len(drawdown_periods) > 0:
        avg_dd_duration: float = float(np.mean(np.array(drawdown_periods)))
    else:
        avg_dd_duration = 0.0

    if len(recovery_periods) > 0:
        avg_recovery_time: float = float(np.mean(np.array(recovery_periods)))
    else:
        avg_recovery_time = np.nan

    return avg_dd_duration, avg_recovery_time


# =============================================================================
# 6. MONTE CARLO SIMULATION
# =============================================================================


@numba.jit(nopython=True, parallel=True, cache=True)
def simulate_garch_process_fast(
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    nu: float,
    target_volatility: float,
    mean_return: float,
    length: int,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Simulate GJR-GARCH(1,1,1) process with Student-t innovations.

    High-performance simulation of asymmetric GARCH process with leverage effects,
    commonly used for modeling financial return volatility clustering.

    Parameters
    ----------
    omega : float
        GARCH omega parameter (long-term variance component).
    alpha : float
        GARCH alpha parameter (ARCH effect).
    beta : float
        GARCH beta parameter (persistence).
    gamma : float
        GARCH gamma parameter (leverage/asymmetry effect).
    nu : float
        Degrees of freedom for Student-t distribution (>2 for t-dist, else Normal).
    target_volatility : float
        Target unconditional volatility.
    mean_return : float
        Mean return level.
    length : int
        Number of periods to simulate.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated return series with GJR-GARCH dynamics.

    Notes
    -----
    - GJR-GARCH allows for leverage effects (asymmetric volatility response)
    - Student-t innovations provide fat tails typical of financial returns
    - Variance bounds are enforced to prevent numerical instability
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Initialize arrays
    returns = np.zeros(length)
    variances = np.zeros(length)

    # Initialize variance to the unconditional target variance
    initial_variance = target_volatility**2
    variances[0] = initial_variance

    # Generate innovations
    # Note: nu must be > 0 to use Student-t, otherwise use Normal
    if nu > 2.0:
        # Use Student-t approximation with Box-Muller + scaling
        # This is a simplified approach for Numba compatibility
        df = max(nu, 3.0)
        innovations = np.random.standard_normal(length)
        # Scale to approximate Student-t distribution
        t_scale = np.sqrt(df / (df - 2.0))
        innovations *= t_scale
    else:
        innovations = np.random.standard_normal(length)

    # Simulate GJR-GARCH process
    for t in range(length):
        # Generate return for the current period
        current_std_dev = np.sqrt(max(1e-12, variances[t]))
        returns[t] = mean_return + innovations[t] * current_std_dev

        # Update variance for the next period
        if t < length - 1:
            error_term = returns[t] - mean_return
            error_term_sq = error_term**2

            # Asymmetry term: I(e_{t-1} < 0)
            indicator = 1.0 if error_term < 0 else 0.0

            variances[t + 1] = (
                omega
                + alpha * error_term_sq
                + gamma * error_term_sq * indicator
                + beta * variances[t]
            )

            # Ensure variance stays positive and reasonable
            variances[t + 1] = max(variances[t + 1], 1e-12)
            variances[t + 1] = min(variances[t + 1], initial_variance * 100.0)

    return returns


@numba.jit(nopython=True, parallel=True, cache=True)
def generate_ohlc_from_prices_fast(
    prices: np.ndarray, random_seed: int = 42
) -> np.ndarray:
    """
    Generate realistic OHLC data from closing price series.

    Creates plausible Open, High, Low, Close data from a closing price series
    by modeling realistic intraday price movements and gaps.

    Parameters
    ----------
    prices : np.ndarray
        Closing price series.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, 4) containing [Open, High, Low, Close] data.

    Notes
    -----
    - Opening prices include small random gaps from previous close
    - High/Low ranges are based on daily volatility and direction
    - Ensures logical OHLC relationships (High >= Open,Close; Low <= Open,Close)
    - Prevents negative prices through minimum price bounds
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    n = len(prices)
    ohlc = np.zeros((n, 4))

    for i in range(n):
        close = prices[i]

        if i == 0:
            # First day: open at close price
            open_price = close
        else:
            # Subsequent days: open near previous close with small gap
            prev_close = prices[i - 1]
            gap_factor = 1.0 + np.random.normal(0.0, 0.005)  # Small random gap
            open_price = prev_close * gap_factor

        # Generate realistic intraday range
        if open_price != 0:
            daily_volatility = abs(close - open_price) / open_price
        else:
            daily_volatility = 0.01
        daily_volatility = max(daily_volatility, 0.005)  # Minimum volatility

        # High and low based on realistic intraday movements
        intraday_range = open_price * daily_volatility * np.random.uniform(1.5, 4.0)

        # Determine high and low
        if close >= open_price:
            # Up day
            high = max(open_price, close) + intraday_range * np.random.uniform(0.2, 0.8)
            low = min(open_price, close) - intraday_range * np.random.uniform(0.1, 0.5)
        else:
            # Down day
            high = max(open_price, close) + intraday_range * np.random.uniform(0.1, 0.5)
            low = min(open_price, close) - intraday_range * np.random.uniform(0.2, 0.8)

        # Ensure low > 0 and logical ordering
        low = max(low, close * 0.5, open_price * 0.5)
        high = max(high, open_price, close)

        ohlc[i, 0] = open_price  # Open
        ohlc[i, 1] = high  # High
        ohlc[i, 2] = low  # Low
        ohlc[i, 3] = close  # Close

    return ohlc


@numba.jit(nopython=True, cache=True)
def garch_simulation_fast(
    length: int,
    omega: float,
    alpha: float,
    beta: float,
    mean_return: float,
    initial_variance: float,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Simulate standard GARCH(1,1) process (simplified version).

    Basic GARCH simulation without leverage effects, useful for
    simpler volatility modeling scenarios.

    Parameters
    ----------
    length : int
        Number of periods to simulate.
    omega : float
        GARCH omega parameter (long-term variance).
    alpha : float
        GARCH alpha parameter (ARCH effect).
    beta : float
        GARCH beta parameter (persistence).
    mean_return : float
        Mean return for the series.
    initial_variance : float
        Initial variance value.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated return series.

    Notes
    -----
    - Standard GARCH(1,1) without leverage effects
    - Simpler than GJR-GARCH but still captures volatility clustering
    - Variance equation: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
    """
    np.random.seed(random_seed)

    returns = np.zeros(length)
    variances = np.zeros(length)

    # Set initial variance
    variances[0] = initial_variance

    for t in range(1, length):
        # GARCH variance equation
        variances[t] = (
            omega
            + alpha * (returns[t - 1] - mean_return) ** 2
            + beta * variances[t - 1]
        )

        # Ensure variance is positive
        variances[t] = max(variances[t], 1e-12)

        # Generate return with conditional variance
        returns[t] = np.random.normal(mean_return, np.sqrt(variances[t]))

    return returns


@numba.jit(nopython=True, parallel=True, cache=True)
def generate_synthetic_returns_batch(
    omega_array: np.ndarray,
    alpha_array: np.ndarray,
    beta_array: np.ndarray,
    gamma_array: np.ndarray,
    nu_array: np.ndarray,
    target_vol_array: np.ndarray,
    mean_return_array: np.ndarray,
    length: int,
    n_assets: int,
    base_seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic returns for multiple assets in parallel.

    High-performance batch generation of synthetic return series using
    GJR-GARCH models with different parameters for each asset.

    Parameters
    ----------
    omega_array : np.ndarray
        Omega parameters for each asset.
    alpha_array : np.ndarray
        Alpha parameters for each asset.
    beta_array : np.ndarray
        Beta parameters for each asset.
    gamma_array : np.ndarray
        Gamma parameters for each asset.
    nu_array : np.ndarray
        Nu parameters for each asset.
    target_vol_array : np.ndarray
        Target volatilities for each asset.
    mean_return_array : np.ndarray
        Mean returns for each asset.
    length : int
        Number of time periods to generate.
    n_assets : int
        Number of assets.
    base_seed : int, optional
        Base random seed.

    Returns
    -------
    np.ndarray
        Array of shape (length, n_assets) with synthetic returns.

    Notes
    -----
    - Each asset uses different random seed for independence
    - Post-simulation scaling ensures target statistics are met
    - Parallel execution for optimal performance
    """
    result = np.zeros((length, n_assets))

    # Generate returns for each asset in parallel
    for asset_idx in numba.prange(n_assets):
        # Use different seed for each asset
        asset_seed = base_seed + asset_idx

        asset_returns = simulate_garch_process_fast(
            omega_array[asset_idx],
            alpha_array[asset_idx],
            beta_array[asset_idx],
            gamma_array[asset_idx],
            nu_array[asset_idx],
            target_vol_array[asset_idx],
            mean_return_array[asset_idx],
            length,
            asset_seed,
        )

        # Post-simulation scaling to match target statistics
        generated_std = np.std(asset_returns)
        if generated_std > 1e-12:
            scale_factor = target_vol_array[asset_idx] / generated_std
            asset_returns *= scale_factor

        # Adjust mean after scaling
        asset_returns -= np.mean(asset_returns)
        asset_returns += mean_return_array[asset_idx]

        result[:, asset_idx] = asset_returns

    return result


# =============================================================================
# 7. BATCH PROCESSING FUNCTIONS
# =============================================================================


@numba.jit(nopython=True, parallel=True, cache=True)
def sharpe_fast(returns_matrix, window, annualization_factor=1.0):
    """
    Calculate rolling Sharpe ratios for multiple assets in parallel.

    Vectorized computation of Sharpe ratios across a 2-D returns matrix,
    providing significant performance improvements for multi-asset analysis.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array with shape (time, assets).
    window : int
        Rolling window length.
    annualization_factor : float, optional
        Annualization factor for scaling.

    Returns
    -------
    np.ndarray
        Matrix of shape (time, assets) with Sharpe ratios.

    Notes
    -----
    - Parallel processing across assets for optimal performance
    - Uses sample standard deviation (ddof=1) for consistency
    - Handles insufficient data gracefully with NaN values
    """
    n_time, n_assets = returns_matrix.shape
    out = np.full((n_time, n_assets), np.nan)

    for a in prange(n_assets):
        series = returns_matrix[:, a]
        for t in range(window - 1, n_time):
            # Collect window data
            val_count = 0
            sum_ret = 0.0
            sum_sq = 0.0
            for k in range(t - window + 1, t + 1):
                r = series[k]
                if not np.isnan(r):
                    val_count += 1
                    sum_ret += r
                    sum_sq += r * r
            if val_count >= window // 2 and val_count > 1:
                mean = sum_ret / val_count
                variance = (sum_sq / val_count) - (mean * mean)
                if variance > 1e-10:
                    std = np.sqrt(variance)
                    out[t, a] = (mean / std) * np.sqrt(annualization_factor)
                else:
                    out[t, a] = 0.0
    return out


# Alias for backward compatibility
sharpe_fast_fixed = sharpe_fast


@numba.jit(nopython=True, parallel=True, cache=True)
def sortino_fast(returns_matrix, window, target_return=0.0, annualization_factor=1.0):
    """
    Calculate rolling Sortino ratios for multiple assets in parallel.

    Vectorized computation of Sortino ratios using only downside deviations,
    providing better risk-adjusted performance measurement than Sharpe ratios.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array with shape (time, assets).
    window : int
        Rolling window length.
    target_return : float, optional
        Target return for downside calculation.
    annualization_factor : float, optional
        Annualization factor for scaling.

    Returns
    -------
    np.ndarray
        Matrix of shape (time, assets) with Sortino ratios.

    Notes
    -----
    - Only considers returns below target for denominator calculation
    - Parallel processing for optimal performance
    - Returns NaN when insufficient downside observations
    """
    n_time, n_assets = returns_matrix.shape
    out = np.full((n_time, n_assets), np.nan)

    for a in prange(n_assets):
        series = returns_matrix[:, a]
        for t in range(window - 1, n_time):
            count = 0
            sum_ret = 0.0
            down_sum = 0.0
            down_sq = 0.0
            for k in range(t - window + 1, t + 1):
                r = series[k]
                if not np.isnan(r):
                    count += 1
                    sum_ret += r
                    if r < target_return:
                        down_sum += r - target_return
                        diff = r - target_return
                        down_sq += diff * diff
            if count >= window // 2 and count > 1:
                mean_ret = sum_ret / count
                if down_sq > 1e-10:
                    downside_std = np.sqrt(down_sq / count)
                    out[t, a] = ((mean_ret - target_return) / downside_std) * np.sqrt(
                        annualization_factor
                    )
                else:
                    out[t, a] = np.nan
    return out


# Alias for backward compatibility
sortino_fast_fixed = sortino_fast


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_sharpe_batch(
    returns_matrix: np.ndarray, window: int, annualization_factor: float = 1.0
) -> np.ndarray:
    """
    Batch rolling Sharpe ratio calculation with pandas-compatible behavior.

    High-performance batch calculation of rolling Sharpe ratios for
    multiple assets, using sample standard deviation for consistency.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets).
    window : int
        Rolling window length.
    annualization_factor : float, optional
        Annualization factor (e.g., 12 for monthly data).

    Returns
    -------
    np.ndarray
        2-D array of rolling Sharpe ratios with same shape as input.

    Notes
    -----
    - Uses ddof=1 for sample standard deviation
    - Parallel processing across assets
    - Proper annualization of both mean and volatility
    """
    n_periods, n_assets = returns_matrix.shape
    result: np.ndarray = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            window_returns = returns_matrix[t - window + 1 : t + 1, asset_idx]

            mean_ret = np.mean(window_returns)
            # Use ddof=1 for sample standard deviation to match pandas
            if len(window_returns) > 1:
                variance = np.sum((window_returns - mean_ret) ** 2) / (
                    len(window_returns) - 1
                )
                std_ret = np.sqrt(variance)

                if std_ret > 1e-12:
                    sharpe = (mean_ret * annualization_factor) / (
                        std_ret * np.sqrt(annualization_factor)
                    )
                    result[t, asset_idx] = sharpe
            else:
                result[t, asset_idx] = 0.0

    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_sortino_batch(
    returns_matrix: np.ndarray,
    window: int,
    target_return: float = 0.0,
    annualization_factor: float = 1.0,
) -> np.ndarray:
    """
    Batch rolling Sortino ratio calculation with proper downside deviation.

    High-performance batch calculation of rolling Sortino ratios using
    only negative returns for the denominator calculation.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets).
    window : int
        Rolling window length.
    target_return : float, optional
        Target return for downside deviation calculation.
    annualization_factor : float, optional
        Annualization factor (e.g., 12 for monthly data).

    Returns
    -------
    np.ndarray
        2-D array of rolling Sortino ratios with same shape as input.

    Notes
    -----
    - Uses only returns below target_return for denominator
    - Parallel processing for optimal performance
    - Returns NaN when no downside returns exist
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            window_returns = returns_matrix[t - window + 1 : t + 1, asset_idx]

            mean_ret = np.mean(window_returns)

            # Calculate downside deviation
            downside_returns = window_returns[window_returns < target_return]
            if len(downside_returns) > 0:
                downside_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
                if downside_dev > 1e-12:
                    sortino = ((mean_ret - target_return) * annualization_factor) / (
                        downside_dev * np.sqrt(annualization_factor)
                    )
                    result[t, asset_idx] = sortino
                else:
                    result[t, asset_idx] = np.nan
            else:
                # No downside returns - match pandas behavior
                result[t, asset_idx] = np.nan

    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_beta_batch(
    returns_matrix: np.ndarray, benchmark_returns: np.ndarray, window: int
) -> np.ndarray:
    """
    Batch rolling beta calculation against benchmark.

    High-performance calculation of rolling beta coefficients for
    multiple assets against a common benchmark.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of asset returns with shape (time, assets).
    benchmark_returns : np.ndarray
        1-D array of benchmark returns.
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray
        2-D array of rolling betas with same shape as returns_matrix.

    Notes
    -----
    - Beta = Covariance(asset, benchmark) / Variance(benchmark)
    - Parallel processing across assets
    - Handles low benchmark volatility gracefully
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            asset_window = returns_matrix[t - window + 1 : t + 1, asset_idx]
            bench_window = benchmark_returns[t - window + 1 : t + 1]

            # Calculate covariance and variance
            asset_mean = np.mean(asset_window)
            bench_mean = np.mean(bench_window)

            covariance = np.mean(
                (asset_window - asset_mean) * (bench_window - bench_mean)
            )
            bench_variance = np.mean((bench_window - bench_mean) ** 2)

            if bench_variance > 1e-12:
                beta = covariance / bench_variance
                result[t, asset_idx] = beta

    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_correlation_batch(
    returns_matrix: np.ndarray, benchmark_returns: np.ndarray, window: int
) -> np.ndarray:
    """
    Batch rolling correlation calculation against benchmark.

    High-performance calculation of rolling correlation coefficients
    between multiple assets and a common benchmark.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of asset returns with shape (time, assets).
    benchmark_returns : np.ndarray
        1-D array of benchmark returns.
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray
        2-D array of rolling correlations with same shape as returns_matrix.

    Notes
    -----
    - Correlation = Covariance / (std_asset * std_benchmark)
    - Parallel processing for optimal performance
    - Handles low volatility periods appropriately
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            asset_window = returns_matrix[t - window + 1 : t + 1, asset_idx]
            bench_window = benchmark_returns[t - window + 1 : t + 1]

            # Calculate correlation
            asset_mean = np.mean(asset_window)
            bench_mean = np.mean(bench_window)

            asset_std = np.sqrt(np.mean((asset_window - asset_mean) ** 2))
            bench_std = np.sqrt(np.mean((bench_window - bench_mean) ** 2))
            covariance = np.mean(
                (asset_window - asset_mean) * (bench_window - bench_mean)
            )

            if asset_std > 1e-12 and bench_std > 1e-12:
                correlation = covariance / (asset_std * bench_std)
                result[t, asset_idx] = correlation

    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_downside_volatility_batch(
    returns_matrix: np.ndarray, window: int
) -> np.ndarray:
    """
    Batch rolling downside volatility calculation.

    High-performance calculation of rolling downside volatility for
    multiple assets, considering only negative returns.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets).
    window : int
        Rolling window length.

    Returns
    -------
    np.ndarray
        2-D array of rolling downside volatilities with same shape as input.

    Notes
    -----
    - Only considers negative returns for volatility calculation
    - Parallel processing across assets
    - Small positive value when no negative returns exist
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            window_returns = returns_matrix[t - window + 1 : t + 1, asset_idx]

            # Calculate downside volatility (only negative returns)
            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) > 0:
                downside_vol = np.sqrt(np.mean(downside_returns**2))
            else:
                downside_vol = 1e-9  # Small positive value

            result[t, asset_idx] = downside_vol

    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def vams_batch_fast(returns_matrix: np.ndarray, window: int) -> np.ndarray:
    """
    Batch VAMS calculation for multiple assets.

    High-performance calculation of Volatility Adjusted Momentum Scores
    for multiple assets simultaneously.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets).
    window : int
        Rolling window length (lookback months).

    Returns
    -------
    np.ndarray
        2-D array of VAMS values with same shape as input.

    Notes
    -----
    - VAMS = momentum / volatility
    - Momentum from rolling cumulative product
    - Uses standard deviation for volatility
    - Parallel processing for optimal performance
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        asset_returns = returns_matrix[:, asset_idx]

        # Calculate momentum using rolling cumulative product
        momentum = rolling_cumprod_fast(asset_returns, window)

        # Calculate rolling volatility
        volatility = rolling_std_fast(asset_returns, window)

        # Calculate VAMS = momentum / volatility
        for t in range(len(momentum)):
            if (
                not np.isnan(momentum[t])
                and not np.isnan(volatility[t])
                and volatility[t] > 1e-9
            ):
                result[t, asset_idx] = momentum[t] / volatility[t]

    return result


# Alias for backward compatibility
vams_batch_fixed = vams_batch_fast


@numba.jit(nopython=True, parallel=True, cache=True)
def dp_vams_batch_fast(
    returns_matrix: np.ndarray, window: int, alpha: float
) -> np.ndarray:
    """
    Batch DP-VAMS calculation for multiple assets.

    High-performance calculation of Downside Penalized VAMS, which
    adjusts momentum scores based on downside risk characteristics.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets).
    window : int
        Rolling window length (lookback months).
    alpha : float
        Weighting factor for downside deviation (0 to 1).

    Returns
    -------
    np.ndarray
        2-D array of DP-VAMS values with same shape as input.

    Notes
    -----
    - DP-VAMS = momentum / (alpha * downside_dev + (1 - alpha) * total_vol)
    - Uses sample standard deviation (ddof=1) for consistency
    - Parallel processing for optimal performance
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        asset_returns = returns_matrix[:, asset_idx]

        # Calculate momentum using rolling cumulative product (fixed)
        momentum = rolling_cumprod_fast(asset_returns, window)

        # Calculate total volatility using fixed std (ddof=1)
        total_vol = rolling_std_fast(asset_returns, window)

        # Calculate downside volatility with ddof=1
        downside_vol = np.full(n_periods, np.nan)
        for t in range(window - 1, n_periods):
            window_returns = asset_returns[t - window + 1 : t + 1]

            # Get only negative returns for downside calculation
            downside_returns = []
            for j in range(len(window_returns)):
                if not np.isnan(window_returns[j]) and window_returns[j] < 0:
                    downside_returns.append(window_returns[j])

            if len(downside_returns) > 1:
                # Use ddof=1 for sample standard deviation
                mean_downside = np.mean(np.array(downside_returns))
                variance = np.sum((np.array(downside_returns) - mean_downside) ** 2) / (
                    len(downside_returns) - 1
                )
                downside_vol[t] = np.sqrt(variance)
            elif len(downside_returns) == 1:
                # Single downside return has zero deviation
                downside_vol[t] = 0.0
            else:
                # No downside returns
                downside_vol[t] = 0.0

        # Calculate DP-VAMS
        for t in range(len(momentum)):
            if (
                not np.isnan(momentum[t])
                and not np.isnan(total_vol[t])
                and not np.isnan(downside_vol[t])
            ):
                denominator = alpha * downside_vol[t] + (1.0 - alpha) * total_vol[t]
                if denominator > 1e-9:
                    result[t, asset_idx] = momentum[t] / denominator

    return result


# Alias for backward compatibility
dp_vams_batch_fixed = dp_vams_batch_fast


@numba.jit(nopython=True, parallel=True, cache=True)
def calmar_batch_fixed(
    returns_matrix: np.ndarray, window: int, cal_factor: float = 12.0
) -> np.ndarray:
    """
    Batch Calmar ratio calculation with proper maximum drawdown.

    High-performance calculation of Calmar ratios (annualized return / max drawdown)
    for multiple assets with accurate drawdown measurement.

    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets).
    window : int
        Rolling window length.
    cal_factor : float, optional
        Annualization factor (default 12.0 for monthly data).

    Returns
    -------
    np.ndarray
        2-D array of Calmar ratios with same shape as input.

    Notes
    -----
    - Calmar = Annualized Return / Maximum Drawdown
    - Uses proper cumulative return calculation for drawdown
    - Handles zero drawdown cases appropriately
    - Parallel processing for optimal performance
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)

    for asset_idx in numba.prange(n_assets):
        asset_returns = returns_matrix[:, asset_idx]

        for t in range(window - 1, n_periods):
            window_returns = asset_returns[t - window + 1 : t + 1]

            # Count valid returns
            valid_count = 0
            for j in range(len(window_returns)):
                if not np.isnan(window_returns[j]):
                    valid_count += 1

            if valid_count < window // 2:  # Need at least half the window
                result[t, asset_idx] = np.nan
                continue

            # Calculate annualized return
            valid_sum = 0.0
            for j in range(len(window_returns)):
                if not np.isnan(window_returns[j]):
                    valid_sum += window_returns[j]

            annualized_return = (valid_sum / valid_count) * cal_factor

            # Calculate maximum drawdown
            cumulative_returns = np.full(len(window_returns), np.nan)
            cum_ret = 1.0

            for j in range(len(window_returns)):
                if not np.isnan(window_returns[j]):
                    cum_ret *= 1.0 + window_returns[j]
                    cumulative_returns[j] = cum_ret

            # Find maximum drawdown
            max_dd = 0.0
            peak = 1.0

            for j in range(len(cumulative_returns)):
                if not np.isnan(cumulative_returns[j]):
                    if cumulative_returns[j] > peak:
                        peak = cumulative_returns[j]

                    drawdown = (peak - cumulative_returns[j]) / peak
                    if drawdown > max_dd:
                        max_dd = drawdown

            # Calculate Calmar ratio
            if max_dd > 1e-6:  # Avoid division by zero
                result[t, asset_idx] = annualized_return / max_dd
            else:
                # If no drawdown, use a large positive value (capped)
                if annualized_return > 0:
                    result[t, asset_idx] = 10.0
                elif annualized_return < 0:
                    result[t, asset_idx] = -10.0
                else:
                    result[t, asset_idx] = 0.0

    return result
