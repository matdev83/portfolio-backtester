import numpy as np
import pandas as pd
import pytest
from portfolio_backtester.numba_optimized import (
    rolling_std_fast,
    rolling_sharpe_fast,
    rolling_sortino_fast,
    rolling_beta_fast,
    vams_batch_fast,
    drawdown_duration_and_recovery_fast,
    simulate_garch_process_fast,
)


@pytest.fixture
def returns_series():
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 100)


def test_rolling_std_fast(returns_series):
    window = 20
    # Calculate using Numba
    result_numba = rolling_std_fast(returns_series, window)

    # Calculate using Pandas
    result_pandas = pd.Series(returns_series).rolling(window).std(ddof=1).values

    # Compare (handle NaNs at start)
    np.testing.assert_allclose(result_numba[window - 1 :], result_pandas[window - 1 :], atol=1e-10)

    # Check that first window-1 are NaN
    assert np.isnan(result_numba[: window - 1]).all()


def test_rolling_std_min_periods():
    # Numba implementation says: if len(valid) >= window // 2
    data = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    window = 4
    # Window 0-3: [1, 2, 3, nan] -> Valid: 3. Window size 4. 3 >= 2. Calc std of [1,2,3].

    result = rolling_std_fast(data, window)

    # Index 3 (4th element): Window is [1, 2, 3, nan]
    valid_data = np.array([1.0, 2.0, 3.0])
    expected_std = np.std(valid_data, ddof=1)

    np.testing.assert_allclose(result[3], expected_std, atol=1e-10)


def test_rolling_sharpe_fast(returns_series):
    window = 20
    annual_factor = 252

    result_numba = rolling_sharpe_fast(returns_series, window, annual_factor)

    # Pandas equivalent
    s = pd.Series(returns_series)
    rolling = s.rolling(window)
    # Sharpe = mean / std * sqrt(252)
    expected = (rolling.mean() / rolling.std(ddof=1)) * np.sqrt(annual_factor)

    np.testing.assert_allclose(
        result_numba[window - 1 :], expected.values[window - 1 :], atol=1e-10
    )


def test_rolling_sortino_fast():
    # Create series with known positive and negative values
    data = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    window = 5
    target = 0.0
    annual = 1.0

    result = rolling_sortino_fast(data, window, target, annual)

    # Last point (index 4) covers full array
    valid_data = data
    downside = valid_data[valid_data < target]  # [-0.02, -0.01]

    mean_ret = np.mean(valid_data)
    # Downside dev with ddof=1
    downside_dev = np.std(downside, ddof=1)

    expected_sortino = (mean_ret - target) / downside_dev * np.sqrt(annual)

    np.testing.assert_allclose(result[4], expected_sortino, atol=1e-10)


def test_rolling_beta_fast():
    asset = np.array([0.01, 0.02, 0.01, 0.03, 0.02])
    bench = np.array([0.005, 0.01, 0.005, 0.015, 0.01])  # Perfect 2x correlation roughly
    window = 5

    result = rolling_beta_fast(asset, bench, window)

    # Manual calc
    cov = np.cov(asset, bench, ddof=1)[0, 1]
    var_bench = np.var(bench, ddof=1)
    expected_beta = cov / var_bench

    np.testing.assert_allclose(result[4], expected_beta, atol=1e-10)


def test_drawdown_duration_and_recovery_fast():
    equity = np.array([100.0, 90.0, 95.0, 100.0])
    avg_dd, avg_rec = drawdown_duration_and_recovery_fast(equity)
    np.testing.assert_allclose(avg_dd, 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(avg_rec, 2.0, rtol=0, atol=1e-9)

    equity_deep = np.array([100.0, 98.0, 90.0, 88.0, 95.0, 100.0])
    avg_dd_d, avg_rec_d = drawdown_duration_and_recovery_fast(equity_deep)
    np.testing.assert_allclose(avg_dd_d, 3.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(avg_rec_d, 2.0, rtol=0, atol=1e-9)

    equity_open = np.array([100.0, 90.0, 85.0, 87.0])
    avg_dd_o, avg_rec_o = drawdown_duration_and_recovery_fast(equity_open)
    np.testing.assert_allclose(avg_dd_o, 2.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(avg_rec_o, 1.0, rtol=0, atol=1e-9)


def test_simulate_garch_process_fast():
    # Smoke test for GARCH simulation
    returns = simulate_garch_process_fast(
        omega=0.000001,
        alpha=0.05,
        beta=0.9,
        gamma=0.05,
        nu=5.0,
        target_volatility=0.01,
        mean_return=0.0,
        length=1000,
    )
    assert len(returns) == 1000
    assert np.isfinite(returns).all()
    # Volatility checking is stochastic and complex due to params.
    # Just check it's positive and not exploding completely (e.g. < 1.0)
    std = np.std(returns)
    assert std > 0.0
    assert std < 1.0


def test_vams_batch_fast():
    # 3 periods, 2 assets
    # Asset 0: Small noise (stable)
    # Asset 1: Volatile

    returns = np.array([[0.0101, 0.01], [0.0099, -0.01], [0.0100, 0.01]])
    window = 3

    result = vams_batch_fast(returns, window)

    # Asset 0: Momentum positive. Vol very small. VAMS large.
    # Asset 1: Momentum small (compound return). Vol large. VAMS small.

    assert result[2, 0] > result[2, 1]
