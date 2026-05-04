import numpy as np
from portfolio_backtester.numba_kernels import (
    _weights_diff_abs,
    _masked_weighted_sum,
    _cumprod_1p,
    detailed_commission_slippage_kernel,
    trade_tracking_kernel,
    trade_lifecycle_kernel,
)


def test_weights_diff_abs():
    weights = np.array([[0.1, 0.2], [0.1, 0.3], [0.0, 0.3], [-0.1, 0.2]], dtype=np.float64)

    # Expected:
    # Day 0: abs(weights[0]) -> [0.1, 0.2]
    # Day 1: abs([0.1-0.1, 0.3-0.2]) -> [0.0, 0.1]
    # Day 2: abs([0.0-0.1, 0.3-0.3]) -> [0.1, 0.0]
    # Day 3: abs([-0.1-0.0, 0.2-0.3]) -> [0.1, 0.1]

    expected = np.array([[0.1, 0.2], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1]], dtype=np.float64)

    result = _weights_diff_abs(weights)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_masked_weighted_sum():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([[0.5, 0.5], [0.2, 0.8]])
    mask = np.array([[True, True], [False, True]])

    # Expected:
    # Row 0: 1.0*0.5 + 2.0*0.5 = 1.5
    # Row 1: 3.0*0.2 (skipped due to mask) + 4.0*0.8 = 3.2

    expected = np.array([1.5, 3.2])
    result = _masked_weighted_sum(values, weights, mask)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_cumprod_1p():
    x = np.array([0.1, -0.1, 0.2])
    # Expected:
    # 0: 1 * 1.1 = 1.1
    # 1: 1.1 * 0.9 = 0.99
    # 2: 0.99 * 1.2 = 1.188

    expected = np.array([1.1, 0.99, 1.188])
    result = _cumprod_1p(x)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_detailed_commission_slippage_kernel():
    # Setup small scenario
    T, N = 3, 2
    weights = np.array(
        [
            [0.5, 0.5],  # Day 0: initial entry turnover from flat to 50/50
            [0.6, 0.4],  # Day 1: Asset 0 chg 0.1, Asset 1 chg 0.1. Total chg 0.2
            [0.6, 0.4],  # Day 2: No change
        ],
        dtype=np.float64,
    )

    close_prices = np.full((T, N), 100.0)
    portfolio_value = 10000.0
    commission_per_share = 0.005
    commission_min = 1.0
    commission_max_pct = 0.01  # 1%
    slippage_bps = 5.0  # 5 bps = 0.0005
    price_mask = np.ones((T, N), dtype=bool)

    cost_frac, detailed_cost_frac = detailed_commission_slippage_kernel(
        weights,
        close_prices,
        portfolio_value,
        commission_per_share,
        commission_min,
        commission_max_pct,
        slippage_bps,
        price_mask,
    )

    # Day 0:
    # Asset 0: diff 0.5 -> Trade Value 5000. Shares 50. Comm: max(50*0.005, 1.0) = 1.0. Slippage: 5000*0.0005 = 2.5. Total: 3.5
    # Asset 1: diff 0.5 -> Trade Value 5000. Shares 50. Comm: 1.0. Slippage: 2.5. Total: 3.5
    # Total Day 0: 7.0 -> Fraction: 0.0007
    np.testing.assert_allclose(cost_frac[0], 0.0007, atol=1e-10)

    # Day 1:
    # Asset 0: diff 0.1 -> Trade Value 1000. Shares 10. Comm: 10*0.005=0.05 < 1.0 -> 1.0. Slippage: 1000*0.0005 = 0.5. Total: 1.5
    # Asset 1: diff 0.1 -> Trade Value 1000. Shares 10. Comm: 1.0 (min). Slippage: 0.5. Total: 1.5
    # Total Day 1: 3.0
    # Fraction: 3.0 / 10000.0 = 0.0003

    np.testing.assert_allclose(cost_frac[1], 0.0003, atol=1e-10)
    np.testing.assert_allclose(detailed_cost_frac[1, 0], 1.5 / 10000.0, atol=1e-10)

    # Day 2: 0 cost (no weight change)
    assert cost_frac[2] == 0.0


def test_trade_tracking_kernel_reinvestment():
    # 2 Days, 2 Assets
    # Allocation Mode 0 (Reinvestment)
    # Day 0: Start 1000. Weights [0.5, 0.5]. Prices [100, 100]. Comm 0.
    #   Buy 5 shares A (500), 5 shares B (500). Cash 0.
    # Day 1: Start. Prices [110, 90].
    #   Value: 5*110 + 5*90 = 550 + 450 = 1000. Capital Base -> 1000.
    #   Target Weights [1.0, 0.0].
    #   Rebalance: Target A: 1000. Target B: 0.
    #   Pos A: 1000/110 = 9.0909. Pos B: 0.
    #   Comm: say 0 for simplicity.
    #   Cash: 1000 - 9.0909*110 - 0 = 0.

    initial_pv = 1000.0
    weights = np.array([[0.5, 0.5], [1.0, 0.0]])
    prices = np.array([[100.0, 100.0], [110.0, 90.0]])
    mask = np.ones((2, 2), dtype=bool)
    commissions = np.zeros((2, 2))  # fraction

    pvals, cash, pos = trade_tracking_kernel(initial_pv, 0, weights, prices, mask, commissions)

    # Check Day 0
    assert pvals[0] == 1000.0
    np.testing.assert_allclose(pos[0], [5.0, 5.0])

    # Check Day 1
    # Day 1 PV is calculated at Start of Day (before rebalance) using Day 1 prices and Day 0 positions
    # PV[1] = Cash[0] (0) + 5*110 + 5*90 = 1000.0
    assert pvals[1] == 1000.0

    # End of Day 1 Rebalance
    # Target A: 1000 * 1.0 / 110 = 9.090909
    np.testing.assert_allclose(pos[1, 0], 1000.0 / 110.0)
    assert pos[1, 1] == 0.0


def test_trade_lifecycle_kernel_basic():
    # Simulating a simple buy and sell
    # 3 Days
    # Day 0: Pos 0
    # Day 1: Pos 10 (Buy) @ 100. Comm 1.0.
    # Day 2: Pos 0 (Sell) @ 110. Comm 1.0.

    T, N = 3, 1
    positions = np.zeros((T, N))
    positions[1, 0] = 10.0
    positions[2, 0] = 0.0

    prices = np.array([[100.0], [100.0], [110.0]])
    dates = np.array([0, 1, 2])  # Mock dates
    commissions = np.array(
        [
            [0.0],
            [1.0],
            [1.0],  # Absolute commission values here?
            # trade_lifecycle_kernel receives 'commissions'.
            # In trade_tracking_kernel, commissions input is 'fraction of portfolio'.
            # But wait, trade_lifecycle_kernel logic sums it up: 'total_commission += commission'.
            # If the input 'commissions' array holds absolute dollar values, then it works.
            # Let's assume for this test it holds absolute values.
        ]
    )

    # Define structured array dtype for trades
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

    # Define dtype for open positions
    open_pos_dtype = [
        ("is_open", "bool"),
        ("entry_date", "i8"),
        ("entry_price", "f8"),
        ("quantity", "f8"),
        ("total_commission", "f8"),
    ]
    out_open_pos = np.zeros(N, dtype=open_pos_dtype)

    initial_capital = 10000.0

    count = trade_lifecycle_kernel(
        positions, prices, dates, commissions, initial_capital, out_trades, out_open_pos
    )

    assert count == 1
    trade = out_trades[0]
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 110.0
    assert trade["quantity"] == 10.0
    assert trade["pnl"] == (110 - 100) * 10.0  # 100.0
    # Commission: Open (Day 1) + Close (Day 2) = 1.0 + 1.0 = 2.0
    assert trade["commission"] == 2.0
