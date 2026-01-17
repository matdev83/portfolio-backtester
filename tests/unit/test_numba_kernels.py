import numpy as np
from portfolio_backtester.numba_kernels import (
    trade_tracking_kernel,
    trade_lifecycle_kernel,
    run_backtest_numba,
    _calculate_window_return_numba,
)


class TestNumbaKernelsExtended:
    def test_trade_tracking_kernel_reinvestment(self):
        # Setup: 3 days, 2 assets
        T, N = 3, 2
        initial_capital = 10000.0
        allocation_mode = 0  # Reinvestment

        # Prices: Asset 0 doubles, Asset 1 stays constant
        prices = np.array(
            [
                [100.0, 50.0],
                [110.0, 50.0],  # Asset 0 +10%
                [120.0, 50.0],  # Asset 0 +~9%
            ],
            dtype=np.float64,
        )

        # Weights: 50/50 allocation
        weights = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )

        # Price mask: all valid
        price_mask = np.ones((T, N), dtype=bool)

        # Commissions: simplified, say 0.01% per day per asset
        commissions = np.zeros((T, N), dtype=np.float64)

        # Day 0: Buy
        # Capital: 10000
        # Target: 5000 per asset
        # Pos 0: 50 shares @ 100
        # Pos 1: 100 shares @ 50
        # Cost: 0 (simplified)

        # Day 1: No rebalance (weights are unchanged), shares are held constant.
        # Portfolio Value before any rebalance:
        # Pos 0: 50 * 110 = 5500
        # Pos 1: 100 * 50 = 5000
        # Total: 10500

        portfolio_values, cash_values, positions = trade_tracking_kernel(
            initial_capital, allocation_mode, weights, prices, price_mask, commissions
        )

        # Checks
        assert portfolio_values[0] == 10000.0
        assert np.isclose(portfolio_values[1], 10500.0)

        # Check positions at day 0
        assert np.isclose(positions[0, 0], 50.0)
        assert np.isclose(positions[0, 1], 100.0)

        # Check positions at day 1 (no rebalance)
        assert np.isclose(positions[1, 0], positions[0, 0])
        assert np.isclose(positions[1, 1], positions[0, 1])

    def test_trade_tracking_kernel_fixed(self):
        # Setup: 3 days, 2 assets
        T, N = 3, 2
        initial_capital = 10000.0
        allocation_mode = 1  # Fixed capital base

        prices = np.array(
            [
                [100.0, 50.0],
                [200.0, 50.0],
                [100.0, 50.0],
            ],
            dtype=np.float64,
        )

        weights = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )

        price_mask = np.ones((T, N), dtype=bool)
        commissions = np.zeros((T, N), dtype=np.float64)

        portfolio_values, cash_values, positions = trade_tracking_kernel(
            initial_capital, allocation_mode, weights, prices, price_mask, commissions
        )

        # Day 0:
        # Target: 5000 per asset (based on 10000 fixed)
        # Pos 0: 50
        # Pos 1: 100

        # Day 1:
        # PV: 50 * 200 + 100 * 50 = 10000 + 5000 = 15000
        # No rebalance when weights are unchanged; shares remain constant.

        assert np.isclose(portfolio_values[1], 15000.0)
        assert np.isclose(positions[1, 0], positions[0, 0])

    def test_trade_lifecycle_kernel_long_short(self):
        # 5 days, 1 asset
        # Day 0: Pos 0
        # Day 1: Pos 10 (Buy)
        # Day 2: Pos 10 (Hold)
        # Day 3: Pos 0 (Sell)
        # Day 4: Pos -10 (Short)
        # Day 5: Pos 0 (Cover)

        positions = np.array([[0.0], [10.0], [10.0], [0.0], [-10.0], [0.0]], dtype=np.float64)

        prices = np.array(
            [
                [100.0],
                [100.0],  # Entry Long
                [110.0],
                [120.0],  # Exit Long, Entry Short (conceptual, but here we go to 0 then -10)
                [115.0],
                [105.0],  # Exit Short
            ],
            dtype=np.float64,
        )

        dates = np.arange(6, dtype=np.int64)
        commissions = np.zeros((6, 1), dtype=np.float64)
        initial_capital = 10000.0

        n_days, n_assets = positions.shape
        max_trades = n_days * n_assets
        out_trades = np.zeros(
            max_trades,
            dtype=[
                ("ticker_idx", "i8"),
                ("entry_date", "i8"),
                ("exit_date", "i8"),
                ("entry_price", "f8"),
                ("exit_price", "f8"),
                ("quantity", "f8"),
                ("pnl", "f8"),
                ("commission", "f8"),
            ],
        )
        out_open_pos = np.zeros(
            n_assets,
            dtype=[
                ("is_open", "?"),
                ("entry_date", "i8"),
                ("entry_price", "f8"),
                ("quantity", "f8"),
                ("total_commission", "f8"),
            ],
        )

        trade_count = trade_lifecycle_kernel(
            positions, prices, dates, commissions, initial_capital, out_trades, out_open_pos
        )

        trades = out_trades[:trade_count]

        # Expected: 2 trades
        # Trade 1: Long. Entry 100, Exit 120. Qty 10. PnL (120-100)*10 = 200.
        # Trade 2: Short. Entry 115 (wait, logic check), Exit 105.

        # Logic check on `trade_lifecycle_kernel`:
        # Day 1: Pos 0 -> 10. Open Long. Entry Price 100.
        # Day 3: Pos 10 -> 0. Close Long. Exit Price 120. PnL 200.
        # Day 4: Pos 0 -> -10. Open Short. Entry Price 115.
        # Day 5: Pos -10 -> 0. Close Short. Exit Price 105. PnL (105-115)*(-10) = -10 * -10 = 100.

        assert len(trades) == 2

        # Trade 1
        t1 = trades[0]
        assert t1["entry_price"] == 100.0
        assert t1["exit_price"] == 120.0
        assert t1["quantity"] == 10.0
        assert t1["pnl"] == 200.0

        # Trade 2
        t2 = trades[1]
        assert t2["entry_price"] == 115.0
        assert t2["exit_price"] == 105.0
        assert t2["quantity"] == -10.0
        assert t2["pnl"] == 100.0

    def test_run_backtest_numba_simple(self):
        # 3 Days, 1 Asset
        prices = np.array([[100.0], [110.0], [121.0]], dtype=np.float64)  # +10%  # +10%

        signals = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)  # Full long

        # Window: Day 0 to Day 3 (exclusive)
        start_indices = np.array([0], dtype=np.int64)
        end_indices = np.array([3], dtype=np.int64)

        returns = run_backtest_numba(prices, signals, start_indices, end_indices)

        # _calculate_window_return_numba logic:
        # Day 1 (from 0 to 1): Return (110-100)/100 = 0.1. Signal[0] = 1.0. Daily = 0.1 * 1.0 = 0.1.
        # Day 2 (from 1 to 2): Return (121-110)/110 = 0.1. Signal[1] = 1.0. Daily = 0.1 * 1.0 = 0.1.
        # Avg = (0.1 + 0.1) / (3-1) = 0.2 / 2 = 0.1

        assert len(returns) == 1
        assert np.isclose(returns[0], 0.1)

    def test_calculate_window_return_numba_nans(self):
        prices = np.array([[100.0], [np.nan], [100.0]], dtype=np.float64)

        signals = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)

        ret = _calculate_window_return_numba(prices, signals)

        # Should skip NaN days and divide by total days - 1
        # Day 1: NaN price -> skip
        # Day 2: NaN prev price -> skip
        # Result 0.0
        assert ret == 0.0
