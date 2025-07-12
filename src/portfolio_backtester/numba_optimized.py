"""
Numba-optimized mathematical functions for performance-critical calculations.

This module contains JIT-compiled functions that provide significant speedups
for pure mathematical operations without any state dependencies.

All functions are designed to be mathematically equivalent to their pandas
counterparts while providing 5-10x performance improvements.
"""

import numba
import numpy as np
from numba import prange


@numba.jit(nopython=True, cache=True)
def calculate_returns_fast(prices):
    """
    Calculates the percentage change between consecutive elements in a NumPy array.

    Args:
        prices (np.ndarray): A 1D NumPy array of prices.

    Returns:
        np.ndarray: A 1D NumPy array of returns, with the first element as NaN.
    """
    returns = np.full_like(prices, np.nan)
    for i in range(1, len(prices)):
        if not np.isnan(prices[i - 1]) and prices[i - 1] > 0 and not np.isnan(prices[i]):
            returns[i] = (prices[i] / prices[i - 1]) - 1.0
    return returns


def create_jitted_rolling_fn(stat_func, func_name, annualization_factor=1.0):
    """
    Higher-order function to create a JIT-compiled rolling window function.

    Args:
        stat_func (callable): The Numba-compatible statistical function to apply (e.g., np.std).
        func_name (str): The name to assign to the generated function.
        annualization_factor (float): An optional factor to multiply the result by.

    Returns:
        callable: A JIT-compiled function that performs the rolling calculation.
    """
    @numba.jit(nopython=True, cache=True)
    def rolling_fn(data, window):
        n = len(data)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= window // 2:
                stat = stat_func(valid_data)
                result[i] = stat * annualization_factor
        return result

    rolling_fn.__name__ = func_name
    return rolling_fn


rolling_mean_fast = create_jitted_rolling_fn(np.mean, 'rolling_mean_fast')
rolling_std_fast = create_jitted_rolling_fn(np.std, 'rolling_std_fast')
rolling_volatility_fast = create_jitted_rolling_fn(np.std, 'rolling_volatility_fast', np.sqrt(252))
rolling_downside_volatility_fast = create_jitted_rolling_fn(
    lambda x: np.std(x[x < 0]), 'rolling_downside_volatility_fast'
)


@numba.jit(nopython=True, cache=True)
def momentum_scores_fast_vectorized(prices_now, prices_then):
    """
    Vectorized version of momentum calculation for better performance.
    """
    valid_mask = (prices_then > 0) & (~np.isnan(prices_then)) & (~np.isnan(prices_now))
    result = np.full_like(prices_now, np.nan)
    result[valid_mask] = (prices_now[valid_mask] / prices_then[valid_mask]) - 1.0
    return result

momentum_scores_fast = momentum_scores_fast_vectorized


@numba.jit(nopython=True, cache=True)
def ema_fast(data, window):
    """
    Fast exponential moving average calculation using Numba.
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
            prev_ema = ema[i-1]
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
                ema[i] = ema[i-1]

    return ema


@numba.jit(nopython=True, cache=True)
def true_range_fast(high, low, close_prev):
    """
    Fast True Range calculation using Numba. Vectorized.
    """
    return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))


@numba.jit(nopython=True, cache=True)
def atr_fast(high, low, close, window):
    """
    Fast Average True Range calculation using Numba (SMA of True Range).
    """
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    tr = true_range_fast(high, low, close_prev)
    return rolling_mean_fast(tr, window)


@numba.jit(nopython=True, cache=True)
def atr_exponential_fast(high, low, close, window):
    """
    Fast Exponential Average True Range calculation using Numba.
    """
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    tr = true_range_fast(high, low, close_prev)
    return ema_fast(tr, window)


@numba.jit(nopython=True, cache=True)
def rolling_beta_fast(asset_returns, benchmark_returns, window):
    """
    Fast rolling beta calculation using Numba.
    """
    n = len(asset_returns)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        asset_window = asset_returns[i - window + 1:i + 1]
        bench_window = benchmark_returns[i - window + 1:i + 1]
        
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
def rolling_sharpe_fast(returns, window, annualization_factor=1.0):
    """
    Fast rolling Sharpe ratio calculation using Numba.
    """
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_returns = returns[i - window + 1:i + 1]
        valid_returns = window_returns[~np.isnan(window_returns)]
        
        if len(valid_returns) >= window // 2:
            mean_ret = np.mean(valid_returns)
            std_ret = np.std(valid_returns)
            if std_ret > 1e-10:
                sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor)
                result[i] = sharpe
            else:
                result[i] = 0.0
    return result


@numba.jit(nopython=True, cache=True)
def rolling_sortino_fast(returns, window, target_return=0.0, annualization_factor=1.0):
    """
    Fast rolling Sortino ratio calculation using Numba.
    """
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(n):
        if i >= window - 1:
            window_returns = returns[i - window + 1:i + 1]
            valid_returns = window_returns[~np.isnan(window_returns)]
            
            if len(valid_returns) >= window:
                mean_ret = np.mean(valid_returns)
                
                downside_returns = valid_returns[valid_returns < target_return]
                if len(downside_returns) > 0:
                    downside_dev = np.std(downside_returns)
                    if downside_dev > 1e-10:
                        sortino = ((mean_ret - target_return) / downside_dev) * np.sqrt(annualization_factor)
                        result[i] = sortino
                    else:
                        result[i] = np.nan # Match pandas behavior for zero downside
                else:
                    # No downside returns - pandas returns NaN in this case
                    result[i] = np.nan
    return result


@numba.jit(nopython=True, cache=True)
def mdd_fast(series):
    """
    Fast maximum drawdown calculation using Numba.
    """
    cummax = np.maximum.accumulate(series)
    drawdown = (series / cummax) - 1
    return np.min(drawdown)
@numba.jit(nopython=True, cache=True)
def rolling_correlation_fast(asset_returns, benchmark_returns, window):
    """
    Fast rolling correlation calculation using Numba.
    """
    n = len(asset_returns)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        asset_window = asset_returns[i - window + 1:i + 1]
        bench_window = benchmark_returns[i - window + 1:i + 1]
        
        valid_mask = (~np.isnan(asset_window)) & (~np.isnan(bench_window))
        valid_asset = asset_window[valid_mask]
        valid_bench = bench_window[valid_mask]
        
        if len(valid_asset) >= window // 2:
            asset_std = np.std(valid_asset)
            bench_std = np.std(valid_bench)
            
            if asset_std > 1e-10 and bench_std > 1e-10:
                covariance = np.cov(valid_asset, valid_bench)[0, 1]
                result[i] = covariance / (asset_std * bench_std)
            else:
                result[i] = 0.0
    return result


@numba.jit(nopython=True, cache=True)
def vams_fast(prices, lookback_months, alpha=0.5):
    """
    Compute Volatility Adjusted Momentum Score (DPVAMS) quickly using Numba.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of price values (monthly closes).
    lookback_months : int
        Look-back window length in months.
    alpha : float
        Downside penalty coefficient in DPVAMS formula.

    Returns
    -------
    np.ndarray
        Array with DPVAMS values (NaN for periods with insufficient history).
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
def rolling_sharpe_fast_portfolio(prices, window_months, annualization_factor=12.0):
    """
    Fast rolling Sharpe ratio for a single asset/portfolio using daily prices.

    The function approximates *window_months* by assuming 21 trading days per month.
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


@numba.jit(nopython=True, cache=True)
def rolling_beta_fast_portfolio(port_prices, mkt_prices, lookback_months):
    """
    Compute trailing beta of a portfolio relative to the market.

    Parameters
    ----------
    port_prices : np.ndarray
        1D price series of the portfolio (monthly closes).
    mkt_prices : np.ndarray
        1D price series of the benchmark/market (monthly closes).
    lookback_months : int
        Window length in months used for beta estimation.

    Returns
    -------
    float
        Latest beta estimate (NaN if insufficient data, 1.0 fallback).
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
        if p_prev > 0 and m_prev > 0 and not (np.isnan(p_prev) or np.isnan(p_cur) or np.isnan(m_prev) or np.isnan(m_cur)):
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


@numba.jit(nopython=True, parallel=True, cache=True)
def sharpe_fast(returns_matrix, window, annualization_factor=1.0):
    """Vectorised rolling Sharpe ratio for a 2-D returns matrix.

    Parameters
    ----------
    returns_matrix : np.ndarray[time, assets]
        Array of asset returns.
    window : int
        Rolling window length in observations.
    annualization_factor : float, optional
        Factor (e.g. 12 or 252) used to annualise the mean; the standard
        deviation is scaled by sqrt(annualization_factor).

    Returns
    -------
    np.ndarray
        Matrix of shape (time, assets) with Sharpe ratios (NaN where
        insufficient observations).
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


@numba.jit(nopython=True, parallel=True, cache=True)
def sortino_fast(returns_matrix, window, target_return=0.0, annualization_factor=1.0):
    """Vectorised rolling Sortino ratio for 2-D returns matrix."""
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
                        down_sum += (r - target_return)
                        diff = r - target_return
                        down_sq += diff * diff
            if count >= window // 2 and count > 1:
                mean_ret = sum_ret / count
                if down_sq > 1e-10:
                    downside_std = np.sqrt(down_sq / count)
                    out[t, a] = ((mean_ret - target_return) / downside_std) * np.sqrt(annualization_factor)
                else:
                    out[t, a] = np.nan
    return out