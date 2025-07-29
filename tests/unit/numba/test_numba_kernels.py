
import numpy as np

from src.portfolio_backtester.numba_kernels import run_backtest_numba


def _python_reference(prices: np.ndarray, signals: np.ndarray) -> float:
    """Reference Python version of the toy back-test used for validation."""
    returns = (prices[1:] / prices[:-1]) - 1.0
    shifted = np.zeros_like(signals)
    shifted[1:] = signals[:-1]
    daily_port = np.sum(returns * shifted[1:], axis=1)  # skip first day (no returns)
    cum = np.prod(1.0 + daily_port) - 1.0
    return cum


def test_run_backtest_numba_one_window():
    # Toy data: 3 days × 2 assets
    prices = np.array(
        [[100.0, 100.0],
         [102.0, 100.0],
         [104.0, 100.0]],
        dtype=np.float32,
    )

    signals = np.array(
        [[0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5]],
        dtype=np.float32,
    )

    start_idx = np.array([0], dtype=np.int64)
    end_idx = np.array([3], dtype=np.int64)

    result = run_backtest_numba(prices, signals, start_idx, end_idx)
    expected = _python_reference(prices, signals)

    assert result.shape == (1,)
    assert np.allclose(result[0], expected, atol=1e-6)


def test_empty_returns_window():
    # Price constant -> zero valid returns
    prices = np.array([[100.0]], dtype=np.float32)
    signals = np.array([[1.0]], dtype=np.float32)
    res = run_backtest_numba(prices, signals, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))
    assert np.isnan(res[0])


def test_skip_asset_with_no_data():
    # Two assets: asset0 has data, asset1 starts late (returns NaN)
    prices = np.array([[100.0, 0.0],
                       [101.0, 0.0],
                       [102.0, 105.0]], dtype=np.float32)
    # second column price zero first two rows gives NaN returns
    signals = np.ones_like(prices, dtype=np.float32) * 0.5

    res = run_backtest_numba(prices, signals,
                             np.array([0], dtype=np.int64),
                             np.array([3], dtype=np.int64))
    # Expect finite return (asset0 contributes)
    assert np.isfinite(res[0])