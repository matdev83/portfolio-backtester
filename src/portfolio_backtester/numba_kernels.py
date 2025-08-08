from typing import Tuple

import numpy as np

# Direct import of Numba - no fallback needed since Numba is a hard dependency
from numba import njit


@njit(cache=True, fastmath=False)
def _weights_diff_abs(weights: np.ndarray) -> np.ndarray:
    """
    Compute absolute day-over-day change in weights per asset.
    weights: [T, N]
    returns: [T, N] with first row = abs(weights[0])
    """
    T, N = weights.shape
    out: np.ndarray = np.empty((T, N), dtype=weights.dtype)
    # first day: treat as full rebalance to current weights
    for j in range(N):
        out[0, j] = abs(weights[0, j])
    for i in range(1, T):
        for j in range(N):
            out[i, j] = abs(weights[i, j] - weights[i - 1, j])
    return out


@njit(cache=True, fastmath=False)
def _masked_weighted_sum(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Vectorized masked sum across columns per row.
    values: [T, N], weights: [T, N], mask: [T, N] (True = valid)
    returns: [T]
    """
    T, N = values.shape
    out: np.ndarray = np.zeros(T, dtype=values.dtype)
    for i in range(T):
        acc = 0.0
        for j in range(N):
            if mask[i, j]:
                acc += values[i, j] * weights[i, j]
        out[i] = acc
    return out


@njit(cache=True, fastmath=False)
def _cumprod_1p(x: np.ndarray) -> np.ndarray:
    """
    Cumulative product of (1 + x) in a numerically stable loop.
    x: [T]
    """
    T = x.shape[0]
    out: np.ndarray = np.empty(T, dtype=x.dtype)
    c = 1.0
    for i in range(T):
        c = c * (1.0 + x[i])
        out[i] = c
    return out


@njit(cache=True, fastmath=False)
def position_and_pnl_kernel(
    weights_for_returns: np.ndarray,  # [T, N]
    rets: np.ndarray,  # [T, N]
    mask: np.ndarray,  # [T, N], True where rets valid
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute daily gross portfolio return (using weights_for_returns),
    equity curve (cumprod of 1+ret), and per-day turnover.

    Returns:
      daily_gross: [T]
      equity_curve: [T]
      turnover: [T] (sum of per-asset abs weight changes)
    """
    # Daily gross portfolio returns
    daily_gross = _masked_weighted_sum(rets, weights_for_returns, mask)

    # Equity curve
    equity_curve = _cumprod_1p(daily_gross)

    # Turnover from weights_for_turnover = current weights
    weights_diff = _weights_diff_abs(weights_for_returns)  # same shape
    T, N = weights_for_returns.shape
    turnover = np.zeros(T, dtype=weights_for_returns.dtype)
    for i in range(T):
        s = 0.0
        for j in range(N):
            s += weights_diff[i, j]
        turnover[i] = s

    return daily_gross, equity_curve, turnover


# NumPy fallback function removed - using only optimized Numba implementation


@njit(cache=True, fastmath=False)
def detailed_commission_slippage_kernel(
    weights_current: np.ndarray,  # [T, N] weights at day t (for turnover calc)
    close_prices: np.ndarray,  # [T, N] close prices
    portfolio_value: float,
    commission_per_share: float,
    commission_min_per_order: float,
    commission_max_percent: float,
    slippage_bps: float,
    price_mask: np.ndarray,  # [T, N] True where price valid (>0) and finite
) -> np.ndarray:
    """
    Compute per-day total transaction cost fraction of portfolio value using an IBKR-style model:
      trade_value = abs(delta_weight) * portfolio_value
      shares = trade_value / close_price
      commission_per_trade = max(shares * commission_per_share, commission_min_per_order) if shares>0 else 0
      commission_amount = min(commission_per_trade, trade_value * commission_max_percent)
      slippage_amount = trade_value * (slippage_bps / 10000)
      total_cost_day = sum_j(commission_amount_j + slippage_amount_j)
      per_day_fraction = total_cost_day / portfolio_value

    Returns:
      per_day_cost_fraction: [T]
    """
    T, N = weights_current.shape
    out: np.ndarray = np.zeros(T, dtype=weights_current.dtype)

    # first day turnover = abs(weights_current[0])
    for i in range(T):
        day_cost = 0.0
        if i == 0:
            # No transaction costs on the first day (no prior holdings).
            out[i] = 0.0
            continue
        for j in range(N):
            dw = abs(weights_current[i, j] - (weights_current[i - 1, j] if i > 0 else 0.0))
            if dw <= 0.0:
                continue
            if not price_mask[i, j]:
                continue
            price = close_prices[i, j]
            if price <= 0.0 or not np.isfinite(price):
                continue

            trade_value = dw * portfolio_value
            shares = trade_value / price

            # Commission per trade with min and max caps
            commission_trade = 0.0
            if (i > 0) and (shares > 0.0):
                commission_trade = shares * commission_per_share
                if commission_trade < commission_min_per_order:
                    commission_trade = commission_min_per_order
                max_commission = trade_value * commission_max_percent
                if commission_trade > max_commission:
                    commission_trade = max_commission

            # Slippage
            slippage_amount = trade_value * (slippage_bps / 10000.0)

            day_cost += commission_trade + slippage_amount

        out[i] = day_cost / portfolio_value if portfolio_value > 0.0 else 0.0

    return out


@njit(cache=True, fastmath=False)
def run_backtest_numba(
    prices: np.ndarray,  # [T, N] price matrix
    signals: np.ndarray,  # [T, N] signal matrix  
    start_indices: np.ndarray,  # [W] start indices for each window
    end_indices: np.ndarray,  # [W] end indices for each window
) -> np.ndarray:
    """
    Run backtest across multiple WFO windows using Numba for optimal performance.
    
    Args:
        prices: Price matrix [time, assets]
        signals: Signal matrix [time, assets] 
        start_indices: Start index for each window
        end_indices: End index for each window
        
    Returns:
        Array of portfolio returns for each window
    """
    n_windows = len(start_indices)
    window_returns: np.ndarray = np.zeros(n_windows, dtype=np.float32)
    
    for w in range(n_windows):
        start_idx = start_indices[w]
        end_idx = end_indices[w]
        
        if start_idx >= end_idx or end_idx > len(prices):
            window_returns[w] = np.nan
            continue
            
        # Extract window data
        window_prices = prices[start_idx:end_idx]
        window_signals = signals[start_idx:end_idx]
        
        # Calculate returns for this window
        window_return = _calculate_window_return_numba(window_prices, window_signals)
        window_returns[w] = window_return
        
    return window_returns


@njit(cache=True, fastmath=False)
def _calculate_window_return_numba(prices: np.ndarray, signals: np.ndarray) -> np.float32:
    """Calculate portfolio return for a single window."""
    n_days, n_assets = prices.shape
    
    if n_days <= 1:
        return np.float32(np.nan)
        
    portfolio_return = 0.0
    
    for day in range(1, n_days):
        daily_return = 0.0
        total_weight = 0.0
        
        # Calculate weighted return for this day
        for asset in range(n_assets):
            if (not np.isnan(signals[day-1, asset]) and 
                not np.isnan(prices[day-1, asset]) and 
                not np.isnan(prices[day, asset]) and
                prices[day-1, asset] > 0):
                
                weight = signals[day-1, asset]
                asset_return = (prices[day, asset] - prices[day-1, asset]) / prices[day-1, asset]
                daily_return += weight * asset_return
                total_weight += abs(weight)
        
        # Normalize if needed and accumulate
        if total_weight > 0:
            portfolio_return += daily_return
            
    return np.float32(portfolio_return / max(1, n_days - 1))  # Average daily return


# NumPy fallback function removed - using only optimized Numba implementation
