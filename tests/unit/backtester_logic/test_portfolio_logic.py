import numpy as np
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from portfolio_backtester.backtester_logic import portfolio_logic
from portfolio_backtester.backtester_logic.portfolio_logic import (
    _sized_signals_to_weights_daily,
    calculate_portfolio_returns,
)
from portfolio_backtester.portfolio.rebalancing import rebalance_to_first_event_per_period


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
        global_config = {"portfolio_value": 10000.0}

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
        global_config = {}

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

    def test_calculate_portfolio_returns_none_rets_daily_matches_canonical(self):
        dates = pd.date_range("2023-01-01", periods=3)
        daily = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)
        rets = daily.pct_change(fill_method=None).fillna(0.0)
        sized_signals = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
        scenario_config = {
            "timing_config": {"rebalance_frequency": "D"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config: dict = {}
        kwargs = dict(
            sized_signals=sized_signals,
            scenario_config=scenario_config,
            price_data_daily_ohlc=daily,
            universe_tickers=["A"],
            global_config=global_config,
        )
        r_with, _ = calculate_portfolio_returns(**kwargs, rets_daily=rets)
        r_none, _ = calculate_portfolio_returns(**kwargs, rets_daily=None)
        pd.testing.assert_series_equal(r_with, r_none, check_names=False)

    def test_time_based_pipeline_calls_rebalance_before_execution_map(self, sample_data):
        daily, rets = sample_data
        sized_signals = pd.DataFrame(0.5, index=daily.index, columns=["A", "B"])
        scenario_config = {
            "timing_config": {"mode": "time_based", "rebalance_frequency": "D"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config = {"portfolio_value": 10000.0}
        order: list[str] = []
        real_rebalance = portfolio_logic.rebalance_to_first_event_per_period
        real_map = portfolio_logic.map_sparse_target_weights_to_execution_dates

        def rebalance_tracked(w, f):
            order.append("rebalance")
            return real_rebalance(w, f)

        def map_tracked(w, **kwargs):
            order.append("map")
            return real_map(w, **kwargs)

        with patch.object(
            portfolio_logic, "rebalance_to_first_event_per_period", side_effect=rebalance_tracked
        ):
            with patch.object(
                portfolio_logic,
                "map_sparse_target_weights_to_execution_dates",
                side_effect=map_tracked,
            ):
                calculate_portfolio_returns(
                    sized_signals, scenario_config, daily, rets, ["A", "B"], global_config
                )
        assert order == ["rebalance", "map"]

    def test_signal_based_skips_rebalance_before_execution_map(self, sample_data):
        daily, rets = sample_data
        sized_signals = pd.DataFrame(0.5, index=daily.index, columns=["A", "B"])
        scenario_config = {
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config = {"portfolio_value": 10000.0}
        order: list[str] = []
        real_map = portfolio_logic.map_sparse_target_weights_to_execution_dates

        def map_tracked(w, **kwargs):
            order.append("map")
            return real_map(w, **kwargs)

        with patch.object(portfolio_logic, "rebalance_to_first_event_per_period") as mock_rebalance:
            with patch.object(
                portfolio_logic,
                "map_sparse_target_weights_to_execution_dates",
                side_effect=map_tracked,
            ):
                calculate_portfolio_returns(
                    sized_signals, scenario_config, daily, rets, ["A", "B"], global_config
                )
            mock_rebalance.assert_not_called()
        assert order == ["map"]

    def test_missing_timing_config_dict_skips_rebalance(self, sample_data):
        daily, rets = sample_data
        sized_signals = pd.DataFrame(0.5, index=daily.index, columns=["A", "B"])
        scenario_config = {"costs_config": {"transaction_costs_bps": 0.0}}
        global_config = {"portfolio_value": 10000.0}
        with patch.object(portfolio_logic, "rebalance_to_first_event_per_period") as mock_rebalance:
            calculate_portfolio_returns(
                sized_signals, scenario_config, daily, rets, ["A", "B"], global_config
            )
            mock_rebalance.assert_not_called()

    def test_rebalance_to_first_event_per_period_keeps_month_end_signal_on_actual_day(self):
        fri = pd.Timestamp("2023-09-29")
        sized = pd.DataFrame({"A": [1.0]}, index=[fri])

        out = rebalance_to_first_event_per_period(sized, "ME")
        assert len(out) == 1
        assert out.index[0] == fri
        assert float(out.iloc[0]["A"]) == pytest.approx(1.0)

    def test_time_based_me_collapses_intramonth_dense_weights_to_first_row(self, sample_data):
        daily, _rets_unused = sample_data
        alt = daily.astype(float).copy()
        alt["A"] = np.linspace(100.0, 110.0, len(daily))
        alt["B"] = np.linspace(100.0, 95.0, len(daily))
        rets_alt = alt.pct_change(fill_method=None).fillna(0.0)
        sized_up = pd.DataFrame(
            {"A": [0.7, 0.3], "B": [0.3, 0.7]},
            index=[daily.index[0], daily.index[1]],
        )
        sized_up = sized_up.reindex(daily.index, method="ffill")
        scenario_time = {
            "timing_config": {"mode": "time_based", "rebalance_frequency": "ME"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        scenario_sig = {
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config = {"portfolio_value": 10000.0}
        r_time, _ = calculate_portfolio_returns(
            sized_up, scenario_time, alt, rets_alt, ["A", "B"], global_config
        )
        r_sig, _ = calculate_portfolio_returns(
            sized_up, scenario_sig, alt, rets_alt, ["A", "B"], global_config
        )
        assert not r_time.equals(r_sig)

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

    def test_sized_signals_to_weights_daily_month_end_friday_preserves_target_on_signal_day(self):
        thu = pd.Timestamp("2023-09-28")
        fri = pd.Timestamp("2023-09-29")
        mon = pd.Timestamp("2023-10-02")
        idx = pd.DatetimeIndex([thu, fri, mon])
        sized = pd.DataFrame({"A": [1.0]}, index=[fri])
        weights_daily = _sized_signals_to_weights_daily(sized, ["A"], idx)
        assert weights_daily.loc[fri, "A"] == pytest.approx(1.0)
        assert weights_daily.loc[mon, "A"] == pytest.approx(1.0)

    def test_sized_signals_ffill_carries_targets_over_all_nan_row(self):
        """Column ffill before fillna(0) keeps exposure across skipped scan rows."""
        d0 = pd.Timestamp("2023-01-02")
        d1 = pd.Timestamp("2023-01-03")
        d2 = pd.Timestamp("2023-01-04")
        idx = pd.DatetimeIndex([d0, d1, d2])
        sized = pd.DataFrame(
            [[0.6, 0.4], [np.nan, np.nan], [0.25, 0.75]],
            index=idx,
            columns=["A", "B"],
        )
        weights_daily = _sized_signals_to_weights_daily(sized, ["A", "B"], idx)
        assert weights_daily.loc[d1, "A"] == pytest.approx(0.6)
        assert weights_daily.loc[d1, "B"] == pytest.approx(0.4)
        assert weights_daily.loc[d2, "A"] == pytest.approx(0.25)
        assert weights_daily.loc[d2, "B"] == pytest.approx(0.75)

    def test_time_based_me_sparse_month_end_friday_shifted_returns_start_next_session(self):
        thu = pd.Timestamp("2023-09-28")
        fri = pd.Timestamp("2023-09-29")
        mon = pd.Timestamp("2023-10-02")
        idx = pd.DatetimeIndex([thu, fri, mon])
        daily = pd.DataFrame(100.0, index=idx, columns=["A"])
        daily.loc[mon, "A"] = 110.0
        rets = pd.DataFrame(0.0, index=idx, columns=["A"])
        rets.loc[mon, "A"] = 0.1
        sized = pd.DataFrame({"A": [1.0]}, index=[fri])
        scenario_config = {
            "timing_config": {"mode": "time_based", "rebalance_frequency": "ME"},
            "costs_config": {"transaction_costs_bps": 0.0},
        }
        global_config = {"portfolio_value": 10000.0}
        returns, _ = calculate_portfolio_returns(
            sized, scenario_config, daily, rets, ["A"], global_config
        )
        assert returns.loc[thu] == pytest.approx(0.0)
        assert returns.loc[fri] == pytest.approx(0.0)
        assert returns.loc[mon] == pytest.approx(0.1)

    def test_close_nan_yields_invalid_price_mask_for_simulation_arrays(self):
        dates = pd.date_range("2023-01-01", periods=4)
        daily = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 55.0, 60.5, 66.55]},
            index=dates,
        )
        daily.loc[dates[2], "B"] = np.nan

        from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
            build_close_and_mask_from_dataframe,
        )

        close_arr, mask = build_close_and_mask_from_dataframe(daily, dates, ["A", "B"])
        assert mask.shape == (4, 2)
        assert not mask[2, 1]
        assert close_arr[2, 1] == pytest.approx(0.0)
