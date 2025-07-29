import numpy as np
from numba import njit, prange

@njit
def window_mean_std(data, starts, ends):
    """
    Calculates the mean and standard deviation of a 1D array over multiple windows.

    Args:
        data (np.ndarray): The input data.
        starts (np.ndarray): The start indices of the windows.
        ends (np.ndarray): The end indices of the windows.

    Returns:
        np.ndarray: A 2D array where each row contains the mean and standard deviation of a window.
    """
    n_windows = len(starts)
    results = np.empty((n_windows, 2), dtype=np.float32)
    for i in range(n_windows):
        window = data[starts[i]:ends[i]]
        if window.size == 0:
            results[i, 0] = np.nan
            results[i, 1] = np.nan
        else:
            results[i, 0] = np.mean(window)
            results[i, 1] = np.std(window)
    return results

@njit
def run_backtest_fast(daily_returns, test_starts, test_ends, strategy_func):
    """
    Runs a simplified backtest using Numba.

    Args:
        daily_returns (np.ndarray): The daily returns of the assets.
        test_starts (np.ndarray): The start indices of the test windows.
        test_ends (np.ndarray): The end indices of the test windows.
        strategy_func (callable): A user-defined strategy function.

    Returns:
        np.ndarray: An array of portfolio returns for each window.
    """
    num_windows = len(test_starts)
    portfolio_returns = np.empty(num_windows, dtype=np.float32)

    for i in range(num_windows):
        window_returns = daily_returns[test_starts[i]:test_ends[i]]
        if window_returns.size > 0:
            portfolio_returns[i] = strategy_func(window_returns)
        else:
            portfolio_returns[i] = np.nan

    return portfolio_returns

@njit(parallel=True)
def run_backtest_numba(
    prices: np.ndarray,
    signals: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
) -> np.ndarray:
    """
    Run a backtest using Numba for performance.

    Parameters
    ----------
    prices : np.ndarray
        A 2D array of prices with shape (time, assets).
    signals : np.ndarray
        A 2D array of signals with shape (time, assets).
    start_indices : np.ndarray
        An array of start indices for each backtest window.
    end_indices : np.ndarray
        An array of end indices for each backtest window.

    Returns
    -------
    np.ndarray
        An array of portfolio returns for each window.
    """
    num_windows = len(start_indices)
    portfolio_returns = np.empty(num_windows, dtype=np.float64)

    for i in prange(num_windows):
        start = start_indices[i]
        end = end_indices[i]
        
        window_prices = prices[start:end]
        window_signals = signals[start:end]
        
        if window_prices.shape[0] > 0:
            # Calculate daily returns for the window
            window_returns = np.full_like(window_prices, np.nan)
            for t in range(1, window_prices.shape[0]):
                for asset in range(window_prices.shape[1]):
                    if window_prices[t - 1, asset] > 0:
                        window_returns[t, asset] = (window_prices[t, asset] / window_prices[t - 1, asset]) - 1.0

            # Replace remaining NaNs in returns with 0.0 so they don't propagate
            for r in range(window_returns.shape[0]):
                for c in range(window_returns.shape[1]):
                    if np.isnan(window_returns[r, c]):
                        window_returns[r, c] = 0.0

            # ------------------------------------------------------------------
            # Manual one–day lag of signals (avoid np.roll keyword limitation in
            # Numba). First day has no previous signal ⇒ use 0.
            # ------------------------------------------------------------------
            shifted_signals = np.empty_like(window_signals)
            shifted_signals[0, :] = 0.0
            for t in range(1, window_signals.shape[0]):
                shifted_signals[t, :] = window_signals[t - 1, :]
 
            # ------------------------------------------------------------------
            # Skip assets that have no finite returns inside this window.  They
            # typically appear when a price series starts after the window
            # begins.  Zero-out their signals so they contribute nothing but do
            # not invalidate the whole window.
            # ------------------------------------------------------------------
            has_valid_asset = False
            for asset in range(window_returns.shape[1]):
                asset_has_data = False
                for t in range(window_returns.shape[0]):
                    if not np.isnan(window_returns[t, asset]):
                        asset_has_data = True
                        break
                if not asset_has_data:
                    shifted_signals[:, asset] = 0.0
                else:
                    has_valid_asset = True

            # If no asset had data, this window is invalid
            if not has_valid_asset:
                portfolio_returns[i] = np.nan
                continue

            # If all returns are NaN (e.g., asset starts mid-window) or all signals zero, skip.
            if np.isnan(window_returns).all() or np.all(shifted_signals == 0):
                portfolio_returns[i] = np.nan
                continue

            # Calculate portfolio returns
            portfolio_daily_returns = np.sum(window_returns * shifted_signals, axis=1)
            if np.isnan(portfolio_daily_returns).all():
                portfolio_returns[i] = np.nan
                continue
            portfolio_returns[i] = np.prod(1.0 + portfolio_daily_returns[np.isfinite(portfolio_daily_returns)]) - 1.0
        else:
            portfolio_returns[i] = np.nan

    return portfolio_returns
