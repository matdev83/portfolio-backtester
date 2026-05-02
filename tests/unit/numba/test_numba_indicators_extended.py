import numpy as np
from portfolio_backtester.numba_optimized import (
    ema_fast,
    atr_fast_fixed,
    calmar_batch_fixed,
    dp_vams_batch_fast,
    rolling_sharpe_fast_portfolio,
    rolling_beta_fast_portfolio,
    drawdown_duration_and_recovery_fast,
    true_range_fast,
)


def test_ema_fast():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 2
    # alpha = 2 / (2 + 1) = 2/3
    # EMA[0] = 1.0
    # EMA[1] = (2/3)*2.0 + (1/3)*1.0 = 1.333 + 0.333 = 1.666
    # EMA[2] = (2/3)*3.0 + (1/3)*1.666 = 2.0 + 0.555 = 2.555

    result = ema_fast(data, window)
    assert result[0] == 1.0
    np.testing.assert_allclose(result[1], 1.6666666666666665, atol=1e-10)
    np.testing.assert_allclose(result[2], 2.5555555555555554, atol=1e-10)


def test_ema_fast_with_nans():
    # NaN at start
    data = np.array([np.nan, 1.0, 2.0])
    window = 2
    result = ema_fast(data, window)

    assert np.isnan(result[0])
    assert result[1] == 1.0  # First valid initializes EMA
    # EMA[2] = alpha*2.0 + (1-alpha)*1.0 = 1.666...
    np.testing.assert_allclose(result[2], 1.6666666666666665, atol=1e-10)


def test_atr_fast_fixed_basic():
    high = np.array([10.0, 11.0, 12.0])
    low = np.array([9.0, 10.0, 11.0])
    close = np.array([9.5, 10.5, 11.5])
    window = 2

    # TR[1]: max(11-10, |11-9.5|, |10-9.5|) = max(1, 1.5, 0.5) = 1.5
    # TR[2]: max(12-11, |12-10.5|, |11-10.5|) = max(1, 1.5, 0.5) = 1.5

    # ATR is rolling mean of TR over window.
    # ATR[1]: mean([TR[0]?, TR[1]])
    # Wait, implementation:
    # for i in range(window - 1, n):
    #   window_tr = tr[i - window + 1 : i + 1]
    #   valid_count = ...
    #   result[i] = tr_sum / valid_count

    # TR[0] calculation logic:
    # for i in range(1, n):
    #   ... tr[i] = ...
    # TR[0] stays NaN (initialized).

    # So for window=2 at index 1:
    # window_tr = [tr[0], tr[1]] = [NaN, 1.5]
    # valid_count = 1
    # result[1] = 1.5 / 1 = 1.5

    result = atr_fast_fixed(high, low, close, window)

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 1.5, atol=1e-10)


def test_calmar_batch_fixed():
    # 2 assets, 5 periods
    # Asset 0: 10% returns consistently (No DD)
    # Asset 1: 10%, -50%, 100% (Huge DD)

    returns = np.array([[0.1, 0.1], [0.1, -0.5], [0.1, 1.0], [0.1, 0.1], [0.1, 0.1]])
    window = 5
    cal_factor = 1.0  # Simplify annualization

    result = calmar_batch_fixed(returns, window, cal_factor)

    # Check index 4 (end of window)
    # Asset 0: Avg Ret 0.1. Max DD 0.0.
    # Code handles Max DD < 1e-6 -> returns 10.0 (capped)
    assert result[4, 0] == 10.0

    # Asset 1:
    # Cum Ret:
    # 0: 1.1
    # 1: 1.1 * 0.5 = 0.55. Peak 1.1. DD (1.1-0.55)/1.1 = 0.5
    # 2: 0.55 * 2.0 = 1.1. Peak 1.1. DD 0.
    # ...
    # Max DD is 0.5.
    # Avg Ret = (0.1 - 0.5 + 1.0 + 0.1 + 0.1) / 5 = 0.8 / 5 = 0.16.
    # Calmar = 0.16 / 0.5 = 0.32

    np.testing.assert_allclose(result[4, 1], 0.32, atol=1e-10)


def test_dp_vams_batch_fast():
    # DP VAMS = Momentum / (alpha * DownsideVol + (1-alpha) * TotalVol)
    # Test alpha=0 (Pure Momentum / TotalVol) vs alpha=1 (Momentum / DownsideVol)

    # Asset with high volatility but no downside (always positive returns)
    # Asset: [0.1, 0.2, 0.1]
    # Momentum > 0. TotalVol > 0. DownsideVol = 0.

    returns = np.array([[0.1, 0.2, 0.1]]).T  # [3, 1]

    window = 3

    # Case 1: Alpha = 0.0 -> Denominator = Total Vol
    res_0 = dp_vams_batch_fast(returns, window, 0.0)
    # Case 2: Alpha = 1.0 -> Denominator = Downside Vol (which is 0 -> small epsilon?)
    # Code:
    # downside_vol[t] = 0.0
    # denominator = 1.0 * 0.0 + 0.0 * total_vol = 0.0
    # if denominator > 1e-9: ... else result is NaN (initially)

    res_1 = dp_vams_batch_fast(returns, window, 1.0)

    # So res_0 should be valid
    assert np.isfinite(res_0[2, 0])

    # res_1 should be NaN (division by zero protection)
    assert np.isnan(res_1[2, 0])


def test_rolling_sharpe_fast_portfolio():
    # Prices: 100, 101, 102, 103...
    # Constant positive return. Sharpe should be massive (or inf if std is 0?)
    # Code uses sample std.
    # Returns: 0.01, 0.0099, 0.0098... (decreasing slightly)
    # So std > 0.

    prices = np.linspace(100, 120, 30)  # 21 days
    window_months = 1  # 21 days

    result = rolling_sharpe_fast_portfolio(prices, window_months)

    # Should have values after 21 days
    assert np.isnan(result[20])  # Window is 21 days. Index 21 (22nd day) uses indices 1..21.
    # Loop: for i in range(window_days, n):
    # if window_days=21. i starts at 21.
    # range(21 - 21 + 1, 21 + 1) -> range(1, 22). returns[1]...returns[21].
    # So index 21 should be populated.

    assert np.isfinite(result[21])
    assert result[21] > 0.0


def test_rolling_beta_fast_portfolio():
    # Port and Market move together perfectly
    port = np.array([100.0, 101.0, 102.0])
    mkt = np.array([1000.0, 1010.0, 1020.0])
    lookback = 3

    beta = rolling_beta_fast_portfolio(port, mkt, lookback)

    # Returns are identical (1%). Cov / Var should be 1.0.
    np.testing.assert_allclose(beta, 1.0, atol=1e-10)


def test_drawdown_recovery_logic():
    equity = np.array([100.0, 50.0, 75.0, 100.0])
    avg_dd, avg_rec = drawdown_duration_and_recovery_fast(equity)
    np.testing.assert_allclose(avg_dd, 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(avg_rec, 2.0, rtol=0, atol=1e-9)


def test_true_range_fast():
    h = np.array([10.0, 12.0])
    lows = np.array([8.0, 9.0])
    cp = np.array([np.nan, 11.0])  # Prev close

    # TR[1] = max(12-9, |12-11|, |9-11|) = max(3, 1, 2) = 3.0

    tr = true_range_fast(h, lows, cp)
    assert np.isnan(tr[0])  # Because cp[0] is nan -> result nan
    np.testing.assert_allclose(tr[1], 3.0, atol=1e-10)
