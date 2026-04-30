import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import (
    calculate_portfolio_returns,
)


def test_trade_tracking_matches_untracked_path_without_costs() -> None:
    dates = pd.bdate_range("2023-01-02", periods=4)
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 121.0],
            "B": [100.0, 100.0, 100.0, 110.0],
        },
        index=dates,
    )
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    sized_signals = pd.DataFrame(
        {
            "A": [1.0, 1.0, 0.0, 0.0],
            "B": [0.0, 0.0, 1.0, 1.0],
        },
        index=dates,
    )
    scenario_config = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "portfolio_value": 1000.0,
    }

    untracked_returns, _ = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        prices,
        returns,
        ["A", "B"],
        global_config,
        track_trades=False,
    )
    tracked_returns, _ = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        prices,
        returns,
        ["A", "B"],
        global_config,
        track_trades=True,
    )

    pd.testing.assert_series_equal(tracked_returns, untracked_returns, check_freq=False)


def test_trade_tracking_charges_initial_entry_costs_on_day_zero() -> None:
    dates = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    sized_signals = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    scenario_config = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 100.0},
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "portfolio_value": 1000.0,
    }

    _, tracker = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        prices,
        returns,
        ["A"],
        global_config,
        track_trades=True,
    )

    assert tracker is not None
    portfolio_values = tracker.portfolio_value_tracker.daily_portfolio_value
    assert portfolio_values.iloc[0] == pytest.approx(990.0)


def test_trade_tracking_missing_held_price_matches_untracked_path() -> None:
    dates = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, np.nan, 100.0]}, index=dates)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    sized_signals = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    scenario_config = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "portfolio_value": 1000.0,
    }

    untracked_returns, _ = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        prices,
        returns,
        ["A"],
        global_config,
        track_trades=False,
    )
    tracked_returns, _ = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        prices,
        returns,
        ["A"],
        global_config,
        track_trades=True,
    )

    assert np.isfinite(tracked_returns.to_numpy()).all()
    pd.testing.assert_series_equal(tracked_returns, untracked_returns)


def test_trade_tracking_reports_commissions_in_dollars_not_fractions() -> None:
    dates = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    sized_signals = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, index=dates)
    scenario_config = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 100.0},
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "portfolio_value": 1000.0,
    }

    _, tracker = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        prices,
        returns,
        ["A"],
        global_config,
        track_trades=True,
    )

    assert tracker is not None
    completed_trades = tracker.trade_lifecycle_manager.get_completed_trades()
    assert len(completed_trades) == 1

    total_commission = completed_trades[0].commission_entry + completed_trades[0].commission_exit
    assert total_commission == pytest.approx(20.0)
