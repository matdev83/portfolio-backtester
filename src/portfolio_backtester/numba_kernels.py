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
                    if window_prices[t-1, asset] > 0:
                        window_returns[t, asset] = (window_prices[t, asset] / window_prices[t-1, asset]) - 1.0
            
            # Calculate portfolio returns
            # Lag signals by one day to avoid lookahead bias
            portfolio_daily_returns = np.sum(window_returns * np.roll(window_signals, 1, axis=0), axis=1)
            portfolio_returns[i] = np.prod(1 + portfolio_daily_returns) - 1
        else:
            portfolio_returns[i] = np.nan

    return portfolio_returns
