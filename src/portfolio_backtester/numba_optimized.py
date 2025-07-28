"""
Numba-optimized functions for high-performance backtesting.

This module contains JIT-compiled functions that provide significant speedups
for computationally intensive operations in the backtesting pipeline.
"""

import logging
import os

import numpy as np

try:
    import numba
    from numba import prange
    NUMBA_AVAILABLE = True
    
    # Configure Numba threading for optimal performance
    # Default to using all available cores, but allow override via environment variable
    num_threads = int(os.environ.get('NUMBA_NUM_THREADS', os.cpu_count()))
    numba.set_num_threads(num_threads)
    
    # Log the configuration for debugging
    import logging
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Numba configured with {num_threads} threads")
    
except ImportError:
    NUMBA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.WARNING):
        logger.warning("Numba not available - falling back to pure Python implementations")


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

@numba.jit(nopython=True, cache=True)
def rolling_downside_volatility_fast(data, window):
    """Calculate rolling downside volatility using only negative returns."""
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        
        if len(valid_data) >= window // 2:
            # Calculate returns from prices
            if len(valid_data) > 1:
                returns = np.diff(valid_data) / valid_data[:-1]
                # Only use negative returns for downside volatility
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    result[i] = np.std(downside_returns)
                else:
                    result[i] = 0.0
            else:
                result[i] = 0.0
    
    return result


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
    random_seed: int = 42
) -> np.ndarray:
    """
    Numba-jitted GJR-GARCH(1,1,1) process simulation for 15-30x speedup.
    
    Parameters
    ----------
    omega : float
        GARCH omega parameter (decimal scale)
    alpha : float
        GARCH alpha parameter
    beta : float
        GARCH beta parameter
    gamma : float
        GARCH gamma (asymmetry) parameter
    nu : float
        Degrees of freedom for Student-t distribution (if > 2, else Normal)
        NOTE: Must be a float, not None. Use 0.0 for Normal distribution.
    target_volatility : float
        Target unconditional volatility
    mean_return : float
        Mean return
    length : int
        Number of periods to simulate
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of simulated returns with GJR-GARCH dynamics
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Initialize arrays
    returns = np.zeros(length)
    variances = np.zeros(length)
    
    # Initialize variance to the unconditional target variance
    initial_variance = target_volatility ** 2
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
            error_term_sq = error_term ** 2
            
            # Asymmetry term: I(e_{t-1} < 0)
            indicator = 1.0 if error_term < 0 else 0.0
            
            variances[t + 1] = omega + alpha * error_term_sq + gamma * error_term_sq * indicator + beta * variances[t]
            
            # Ensure variance stays positive and reasonable
            variances[t + 1] = max(variances[t + 1], 1e-12)
            variances[t + 1] = min(variances[t + 1], initial_variance * 100.0)
    
    return returns


@numba.jit(nopython=True, parallel=True, cache=True)
def generate_ohlc_from_prices_fast(prices: np.ndarray, random_seed: int = 42) -> np.ndarray:
    """
    Numba-jitted OHLC generation from price series for 15-30x speedup.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of closing prices
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of shape (n, 4) with OHLC data [Open, High, Low, Close]
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
            prev_close = prices[i-1]
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
        ohlc[i, 1] = high        # High
        ohlc[i, 2] = low         # Low
        ohlc[i, 3] = close       # Close
    
    return ohlc


@numba.jit(nopython=True, cache=True)
def returns_to_prices_fast(returns: np.ndarray, initial_price: float) -> np.ndarray:
    """
    Numba-jitted conversion of returns to price levels for improved performance.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (decimal form)
    initial_price : float
        Starting price level
        
    Returns
    -------
    np.ndarray
        Array of price levels
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
        prices[i] = prices[i-1] * price_relatives[i]
    
    return prices


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
    base_seed: int = 42
) -> np.ndarray:
    """
    Numba-jitted batch generation of synthetic returns for multiple assets in parallel.
    
    Parameters
    ----------
    omega_array : np.ndarray
        Array of omega parameters for each asset
    alpha_array : np.ndarray
        Array of alpha parameters for each asset
    beta_array : np.ndarray
        Array of beta parameters for each asset
    gamma_array : np.ndarray
        Array of gamma parameters for each asset
    nu_array : np.ndarray
        Array of nu parameters for each asset
    target_vol_array : np.ndarray
        Array of target volatilities for each asset
    mean_return_array : np.ndarray
        Array of mean returns for each asset
    length : int
        Number of time periods to generate
    n_assets : int
        Number of assets
    base_seed : int
        Base random seed
        
    Returns
    -------
    np.ndarray
        Array of shape (length, n_assets) with synthetic returns
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
            asset_seed
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


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_sharpe_batch(returns_matrix: np.ndarray, window: int, annualization_factor: float = 1.0) -> np.ndarray:
    """
    Batched rolling Sharpe ratio calculation for 2-D returns matrix.
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length
    annualization_factor : float
        Annualization factor (e.g., 12 for monthly data)
        
    Returns
    -------
    np.ndarray
        2-D array of rolling Sharpe ratios with same shape as input
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            window_returns = returns_matrix[t - window + 1:t + 1, asset_idx]
            
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            
            if std_ret > 1e-12:
                sharpe = (mean_ret * annualization_factor) / (std_ret * np.sqrt(annualization_factor))
                result[t, asset_idx] = sharpe
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_sortino_batch(returns_matrix: np.ndarray, window: int, target_return: float = 0.0, annualization_factor: float = 1.0) -> np.ndarray:
    """
    Batched rolling Sortino ratio calculation for 2-D returns matrix.
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length
    target_return : float
        Target return for downside deviation calculation
    annualization_factor : float
        Annualization factor (e.g., 12 for monthly data)
        
    Returns
    -------
    np.ndarray
        2-D array of rolling Sortino ratios with same shape as input
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            window_returns = returns_matrix[t - window + 1:t + 1, asset_idx]
            
            mean_ret = np.mean(window_returns)
            
            # Calculate downside deviation
            downside_returns = window_returns[window_returns < target_return]
            if len(downside_returns) > 0:
                downside_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
                if downside_dev > 1e-12:
                    sortino = ((mean_ret - target_return) * annualization_factor) / (downside_dev * np.sqrt(annualization_factor))
                    result[t, asset_idx] = sortino
                else:
                    result[t, asset_idx] = np.nan
            else:
                # No downside returns - match pandas behavior
                result[t, asset_idx] = np.nan
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_beta_batch(returns_matrix: np.ndarray, benchmark_returns: np.ndarray, window: int) -> np.ndarray:
    """
    Batched rolling beta calculation for 2-D returns matrix against benchmark.
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of asset returns with shape (time, assets)
    benchmark_returns : np.ndarray
        1-D array of benchmark returns
    window : int
        Rolling window length
        
    Returns
    -------
    np.ndarray
        2-D array of rolling betas with same shape as returns_matrix
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            asset_window = returns_matrix[t - window + 1:t + 1, asset_idx]
            bench_window = benchmark_returns[t - window + 1:t + 1]
            
            # Calculate covariance and variance
            asset_mean = np.mean(asset_window)
            bench_mean = np.mean(bench_window)
            
            covariance = np.mean((asset_window - asset_mean) * (bench_window - bench_mean))
            bench_variance = np.mean((bench_window - bench_mean) ** 2)
            
            if bench_variance > 1e-12:
                beta = covariance / bench_variance
                result[t, asset_idx] = beta
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_correlation_batch(returns_matrix: np.ndarray, benchmark_returns: np.ndarray, window: int) -> np.ndarray:
    """
    Batched rolling correlation calculation for 2-D returns matrix against benchmark.
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of asset returns with shape (time, assets)
    benchmark_returns : np.ndarray
        1-D array of benchmark returns
    window : int
        Rolling window length
        
    Returns
    -------
    np.ndarray
        2-D array of rolling correlations with same shape as returns_matrix
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            asset_window = returns_matrix[t - window + 1:t + 1, asset_idx]
            bench_window = benchmark_returns[t - window + 1:t + 1]
            
            # Calculate correlation
            asset_mean = np.mean(asset_window)
            bench_mean = np.mean(bench_window)
            
            asset_std = np.sqrt(np.mean((asset_window - asset_mean) ** 2))
            bench_std = np.sqrt(np.mean((bench_window - bench_mean) ** 2))
            covariance = np.mean((asset_window - asset_mean) * (bench_window - bench_mean))
            
            if asset_std > 1e-12 and bench_std > 1e-12:
                correlation = covariance / (asset_std * bench_std)
                result[t, asset_idx] = correlation
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def rolling_downside_volatility_batch(returns_matrix: np.ndarray, window: int) -> np.ndarray:
    """
    Batched rolling downside volatility calculation for 2-D returns matrix.
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length
        
    Returns
    -------
    np.ndarray
        2-D array of rolling downside volatilities with same shape as input
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        for t in range(window - 1, n_periods):
            window_returns = returns_matrix[t - window + 1:t + 1, asset_idx]
            
            # Calculate downside volatility (only negative returns)
            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) > 0:
                downside_vol = np.sqrt(np.mean(downside_returns ** 2))
            else:
                downside_vol = 1e-9  # Small positive value
            
            result[t, asset_idx] = downside_vol
    
    return result


@numba.jit(nopython=True, cache=True)
def rolling_cumprod_fast(data: np.ndarray, window: int) -> np.ndarray:
    """
    Fast rolling cumulative product calculation for momentum calculations.
    
    Replaces (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    
    Parameters
    ----------
    data : np.ndarray
        1-D array of returns data
    window : int
        Rolling window length
        
    Returns
    -------
    np.ndarray
        1-D array of rolling momentum values (cumulative product - 1)
    """
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        
        # Check for NaN values in window
        has_nan = False
        for j in range(len(window_data)):
            if np.isnan(window_data[j]):
                has_nan = True
                break
        
        if not has_nan:
            # Calculate cumulative product: (1 + r1) * (1 + r2) * ... - 1
            prod = 1.0
            for j in range(len(window_data)):
                prod *= (1.0 + window_data[j])
            result[i] = prod - 1.0
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def vams_batch_fast(returns_matrix: np.ndarray, window: int) -> np.ndarray:
    """
    Batch VAMS (Volatility Adjusted Momentum Scores) calculation for multiple assets.
    
    VAMS = momentum / volatility
    where momentum = rolling cumulative product - 1
    and volatility = rolling standard deviation
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length (lookback months)
        
    Returns
    -------
    np.ndarray
        2-D array of VAMS values with same shape as input
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
            if (not np.isnan(momentum[t]) and not np.isnan(volatility[t]) and 
                volatility[t] > 1e-9):
                result[t, asset_idx] = momentum[t] / volatility[t]
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def dp_vams_batch_fast(returns_matrix: np.ndarray, window: int, alpha: float) -> np.ndarray:
    """
    Batch DP-VAMS (Downside Penalized VAMS) calculation for multiple assets.
    
    DP-VAMS = momentum / (alpha * downside_dev + (1 - alpha) * total_vol)
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length (lookback months)
    alpha : float
        Weighting factor for downside deviation (0 to 1)
        
    Returns
    -------
    np.ndarray
        2-D array of DP-VAMS values with same shape as input
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        asset_returns = returns_matrix[:, asset_idx]
        
        # Calculate momentum using rolling cumulative product
        momentum = rolling_cumprod_fast(asset_returns, window)
        
        # Calculate total volatility
        total_vol = rolling_std_fast(asset_returns, window)
        
        # Calculate downside volatility
        downside_vol = np.full(n_periods, np.nan)
        for t in range(window - 1, n_periods):
            window_returns = asset_returns[t - window + 1:t + 1]
            
            # Get only negative returns for downside calculation
            downside_count = 0
            downside_sum_sq = 0.0
            for j in range(len(window_returns)):
                if not np.isnan(window_returns[j]) and window_returns[j] < 0:
                    downside_sum_sq += window_returns[j] ** 2
                    downside_count += 1
            
            if downside_count > 0:
                downside_vol[t] = np.sqrt(downside_sum_sq / downside_count)
            else:
                downside_vol[t] = 0.0
        
        # Calculate DP-VAMS
        for t in range(len(momentum)):
            if (not np.isnan(momentum[t]) and not np.isnan(total_vol[t]) and 
                not np.isnan(downside_vol[t])):
                denominator = alpha * downside_vol[t] + (1.0 - alpha) * total_vol[t]
                if denominator > 1e-9:
                    result[t, asset_idx] = momentum[t] / denominator
    
    return result


@numba.jit(nopython=True, cache=True)
def garch_simulation_fast(length: int, omega: float, alpha: float, beta: float, 
                         mean_return: float, initial_variance: float, 
                         random_seed: int = 42) -> np.ndarray:
    """
    Fast GARCH(1,1) simulation using Numba.
    
    Replaces the Python loop in synthetic data generation for significant speedup.
    
    Parameters
    ----------
    length : int
        Number of periods to simulate
    omega : float
        GARCH omega parameter (long-term variance)
    alpha : float
        GARCH alpha parameter (ARCH effect)
    beta : float
        GARCH beta parameter (persistence)
    mean_return : float
        Mean return for the series
    initial_variance : float
        Initial variance value
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of simulated returns
    """
    np.random.seed(random_seed)
    
    returns = np.zeros(length)
    variances = np.zeros(length)
    
    # Set initial variance
    variances[0] = initial_variance
    
    for t in range(1, length):
        # GARCH variance equation
        variances[t] = (omega + 
                       alpha * (returns[t-1] - mean_return)**2 + 
                       beta * variances[t-1])
        
        # Ensure variance is positive
        variances[t] = max(variances[t], 1e-12)
        
        # Generate return with conditional variance
        returns[t] = np.random.normal(mean_return, np.sqrt(variances[t]))
    
    return returns


@numba.jit(nopython=True, parallel=True, cache=True)
def calmar_batch_fast(returns_matrix: np.ndarray, window: int, cal_factor: float = 12.0) -> np.ndarray:
    """
    Batch Calmar ratio calculation for multiple assets.
    
    Calmar = annualized_return / approximated_max_drawdown
    where approximated_max_drawdown = rolling_std * 2.5
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length
    cal_factor : float
        Annualization factor (default 12 for monthly data)
        
    Returns
    -------
    np.ndarray
        2-D array of Calmar ratios with same shape as input
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        asset_returns = returns_matrix[:, asset_idx]
        
        # Calculate rolling mean
        rolling_mean = rolling_mean_fast(asset_returns, window) * cal_factor
        
        # Calculate rolling std as proxy for max drawdown
        rolling_std = rolling_std_fast(asset_returns, window)
        approx_max_dd = rolling_std * 2.5
        
        # Calculate Calmar ratio
        for t in range(len(rolling_mean)):
            if (not np.isnan(rolling_mean[t]) and not np.isnan(approx_max_dd[t]) and 
                approx_max_dd[t] > 1e-6):
                calmar_val = rolling_mean[t] / approx_max_dd[t]
                # Clip extreme values
                if calmar_val > 10.0:
                    result[t, asset_idx] = 10.0
                elif calmar_val < -10.0:
                    result[t, asset_idx] = -10.0
                else:
                    result[t, asset_idx] = calmar_val
    
    return result


@numba.jit(nopython=True, parallel=True, cache=True)
def calmar_batch_fast(returns_matrix: np.ndarray, window: int, cal_factor: float = 12.0) -> np.ndarray:
    """
    Batch Calmar ratio calculation for multiple assets.
    
    Calmar = annualized_return / approximated_max_drawdown
    where approximated_max_drawdown = rolling_std * 2.5
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2-D array of returns with shape (time, assets)
    window : int
        Rolling window length
    cal_factor : float
        Annualization factor (default 12 for monthly data)
        
    Returns
    -------
    np.ndarray
        2-D array of Calmar ratios with same shape as input
    """
    n_periods, n_assets = returns_matrix.shape
    result = np.full((n_periods, n_assets), np.nan)
    
    for asset_idx in numba.prange(n_assets):
        asset_returns = returns_matrix[:, asset_idx]
        
        # Calculate rolling mean
        rolling_mean = rolling_mean_fast(asset_returns, window) * cal_factor
        
        # Calculate rolling std as proxy for max drawdown
        rolling_std = rolling_std_fast(asset_returns, window)
        approx_max_dd = rolling_std * 2.5
        
        # Calculate Calmar ratio
        for t in range(len(rolling_mean)):
            if (not np.isnan(rolling_mean[t]) and not np.isnan(approx_max_dd[t]) and 
                approx_max_dd[t] > 1e-6):
                calmar_val = rolling_mean[t] / approx_max_dd[t]
                # Clip extreme values
                if calmar_val > 10.0:
                    result[t, asset_idx] = 10.0
                elif calmar_val < -10.0:
                    result[t, asset_idx] = -10.0
                else:
                    result[t, asset_idx] = calmar_val
    
    return result