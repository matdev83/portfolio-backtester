import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns


class TestPortfolioLogic:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        daily = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
        rets = daily.pct_change().fillna(0.0)
        return daily, rets

    def test_calculate_portfolio_returns_ndarray_path(self, sample_data):
        daily, rets = sample_data

        # Signals: 50/50 allocation
        sized_signals = pd.DataFrame(0.5, index=daily.index, columns=["A", "B"])

        scenario_config = {
            "timing_config": {"rebalance_frequency": "D"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config = {"feature_flags": {"ndarray_simulation": True}, "portfolio_value": 10000.0}

        # Returns should be 0 since price is constant 100.0
        returns, tracker = calculate_portfolio_returns(
            sized_signals,
            scenario_config,
            daily,
            rets,
            ["A", "B"],
            global_config,
            track_trades=False,
        )

        assert isinstance(returns, pd.Series)
        assert (returns == 0.0).all()

    def test_calculate_portfolio_returns_with_returns(self):
        dates = pd.date_range("2023-01-01", periods=3)
        # Prices: 100 -> 110 -> 121 (+10% each day)
        daily = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)
        rets = daily.pct_change().fillna(0.0)

        sized_signals = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)

        scenario_config = {
            "timing_config": {"rebalance_frequency": "D"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config = {"feature_flags": {"ndarray_simulation": True}}

        # calculate_portfolio_returns uses shifted weights (previous day's weight)
        # Day 0: Weight NaN (shift) -> 0.0. Ret 0.0. Port Ret 0.0.
        # Day 1: Weight 1.0 (from Day 0). Ret 0.1. Port Ret 0.1.
        # Day 2: Weight 1.0 (from Day 1). Ret 0.1. Port Ret 0.1.

        returns, _ = calculate_portfolio_returns(
            sized_signals, scenario_config, daily, rets, ["A"], global_config
        )

        assert returns.iloc[0] == 0.0
        assert returns.iloc[1] == pytest.approx(0.1)
        assert returns.iloc[2] == pytest.approx(0.1)

    def test_calculate_portfolio_returns_disabled_ndarray_error(self, sample_data):
        daily, rets = sample_data
        sized_signals = pd.DataFrame(0.5, index=daily.index, columns=["A", "B"])

        # Explicitly disable ndarray simulation to trigger error
        global_config = {"feature_flags": {"ndarray_simulation": False}}

        with pytest.raises(
            RuntimeError, match="legacy Pandas-based portfolio simulation has been removed"
        ):
            calculate_portfolio_returns(sized_signals, {}, daily, rets, ["A", "B"], global_config)

    @patch("portfolio_backtester.backtester_logic.portfolio_logic.TradeTracker")
    def test_calculate_portfolio_returns_track_trades(self, mock_tracker_cls, sample_data):
        daily, rets = sample_data
        sized_signals = pd.DataFrame(0.5, index=daily.index, columns=["A", "B"])

        mock_tracker = MagicMock()
        # Numba needs a concrete float type, not a Mock object
        mock_tracker.initial_portfolio_value = 1000.0
        mock_tracker.allocation_mode = "fixed_capital"  # Add allocation mode
        mock_tracker.portfolio_value_tracker.daily_portfolio_value = pd.Series(
            1000.0, index=daily.index
        )
        mock_tracker_cls.return_value = mock_tracker

        returns, tracker = calculate_portfolio_returns(
            sized_signals,
            {"allocation_mode": "fixed_capital"},
            daily,
            rets,
            ["A", "B"],
            {"portfolio_value": 1000.0},
            track_trades=True,
        )

        assert tracker is not None
        # Check if populate_from_kernel_results was called on the tracker
        mock_tracker.populate_from_kernel_results.assert_called_once()
