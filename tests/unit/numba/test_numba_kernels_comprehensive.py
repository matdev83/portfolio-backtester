import numpy as np
import pytest
from portfolio_backtester.numba_kernels import (
    detailed_commission_slippage_kernel,
    trade_tracking_kernel,
    trade_lifecycle_kernel,
    position_and_pnl_kernel,
)


def test_position_and_pnl_kernel_comprehensive():
    T, N = 5, 2
    # 5 days, 2 assets
    weights = np.array([[0.5, 0.5], [0.6, 0.4], [0.6, 0.4], [0.0, 1.0], [0.0, 0.0]])
    rets = np.array([[0.01, -0.01], [0.02, 0.00], [-0.01, 0.01], [0.00, 0.05], [0.00, 0.00]])
    mask = np.ones((T, N), dtype=bool)

    daily_gross, equity_curve, turnover = position_and_pnl_kernel(weights, rets, mask)

    # daily_gross calculation:
    # Day 0: 0.01*0.5 + -0.01*0.5 = 0.0
    # Day 1: 0.02*0.6 + 0.00*0.4 = 0.012
    # Day 2: -0.01*0.6 + 0.01*0.4 = -0.006 + 0.004 = -0.002
    # Day 3: 0.00*0.0 + 0.05*1.0 = 0.05
    # Day 4: 0.0
    expected_daily = [0.0, 0.012, -0.002, 0.05, 0.0]
    np.testing.assert_allclose(daily_gross, expected_daily, atol=1e-10)

    # equity_curve: cumprod(1+daily_gross)
    # 0: 1.0
    # 1: 1.012
    # 2: 1.012 * 0.998 = 1.009976
    # 3: 1.009976 * 1.05 = 1.0604748
    expected_equity = np.cumprod(1 + np.array(expected_daily))
    np.testing.assert_allclose(equity_curve, expected_equity, atol=1e-10)

    # turnover calculation (sum of abs diff):
    # Day 0: 0.5 + 0.5 = 1.0 (initial)
    # Day 1: |0.6-0.5| + |0.4-0.5| = 0.1 + 0.1 = 0.2
    # Day 2: 0.0
    # Day 3: |0.0-0.6| + |1.0-0.4| = 0.6 + 0.6 = 1.2
    # Day 4: |0.0-0.0| + |0.0-1.0| = 1.0
    expected_turnover = [1.0, 0.2, 0.0, 1.2, 1.0]
    np.testing.assert_allclose(turnover, expected_turnover, atol=1e-10)


def test_detailed_commission_slippage_kernel_caps():
    T, N = 2, 1
    # weights: day 0: 0.0, day 1: 1.0 -> change 1.0
    weights = np.array([[0.0], [1.0]])
    close_prices = np.array([[100.0], [100.0]])
    portfolio_value = 1000.0

    # Scenario: Max Commission Cap
    # Trade Value = 1000. Comm Max = 1% = 10.0.
    # Comm per share = 100.0. Shares = 10. Raw Comm = 1000.
    # Cap should limit it to 10.0.
    cost_frac, _ = detailed_commission_slippage_kernel(
        weights, close_prices, portfolio_value, 100.0, 1.0, 0.01, 0.0, np.ones((T, N), dtype=bool)
    )
    # Day 1 cost fraction: 10.0 / 1000.0 = 0.01
    assert cost_frac[1] == pytest.approx(0.01)

    # Scenario: Min Commission Cap
    # Trade Value = 1000. Shares = 10.
    # Comm per share = 0.001. Raw Comm = 0.01.
    # Min Comm = 5.0.
    cost_frac_min, _ = detailed_commission_slippage_kernel(
        weights, close_prices, portfolio_value, 0.001, 5.0, 0.1, 0.0, np.ones((T, N), dtype=bool)
    )
    # Day 1 cost fraction: 5.0 / 1000.0 = 0.005
    assert cost_frac_min[1] == pytest.approx(0.005)


def test_trade_tracking_kernel_fixed_allocation():
    # Allocation Mode 1 (Fixed)
    # Capital base remains initial_pv for all end-of-day rebalances
    initial_pv = 1000.0
    weights = np.array([[1.0], [1.0]])
    prices = np.array([[100.0], [200.0]])
    mask = np.ones((2, 1), dtype=bool)
    commissions = np.zeros((2, 1))

    pvals, cash, pos = trade_tracking_kernel(initial_pv, 1, weights, prices, mask, commissions)

    # Day 0: 1000. Buy 10 shares. Cash 0.
    assert pos[0, 0] == 10.0
    assert pvals[0] == 1000.0

    # Day 1: Start. Price 200. Value: 10*200 = 2000.
    assert pvals[1] == 2000.0

    # No rebalance when weights are unchanged; shares remain constant.
    assert pos[1, 0] == pytest.approx(pos[0, 0])
    assert cash[1] == pytest.approx(cash[0])


def test_trade_lifecycle_kernel_flip():
    # Flip from Long 10 to Short 10
    N = 1
    positions = np.array([[0.0], [10.0], [-10.0]])
    prices = np.array([[100.0], [100.0], [100.0]])
    dates = np.array([0, 1, 2])
    commissions = np.array([[0.0], [0.0], [2.0]])  # $2 commission on flip

    trade_dtype = [
        ("ticker_idx", "i4"),
        ("entry_date", "i8"),
        ("exit_date", "i8"),
        ("entry_price", "f8"),
        ("exit_price", "f8"),
        ("quantity", "f8"),
        ("pnl", "f8"),
        ("commission", "f8"),
    ]
    out_trades = np.zeros(10, dtype=trade_dtype)
    open_pos_dtype = [
        ("is_open", "bool"),
        ("entry_date", "i8"),
        ("entry_price", "f8"),
        ("quantity", "f8"),
        ("total_commission", "f8"),
    ]
    out_open_pos = np.zeros(N, dtype=open_pos_dtype)

    count = trade_lifecycle_kernel(
        positions, prices, dates, commissions, 1000.0, out_trades, out_open_pos
    )

    # One trade should be completed (the long one)
    assert count == 1
    assert out_trades[0]["quantity"] == 10.0
    # Commission split: abs(prev)=10, abs(curr)=10. Total 20.
    # Close part (10/20 * 2.0) = 1.0
    assert out_trades[0]["commission"] == 1.0

    # Short position should be open
    assert out_open_pos[0]["is_open"]
    assert out_open_pos[0]["quantity"] == -10.0
    assert out_open_pos[0]["total_commission"] == 1.0


def test_detailed_commission_slippage_robustness():
    # Test with zero prices, empty mask
    T, N = 3, 2
    weights = np.ones((T, N))
    prices = np.zeros((T, N))  # All zero prices
    mask = np.zeros((T, N), dtype=bool)  # Nothing valid

    cost_frac, detailed = detailed_commission_slippage_kernel(
        weights, prices, 1000.0, 0.01, 1.0, 0.01, 5.0, mask
    )

    assert (cost_frac == 0.0).all()
    assert (detailed == 0.0).all()
