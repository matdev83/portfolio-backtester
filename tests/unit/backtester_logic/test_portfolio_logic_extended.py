import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=5)
    sized_signals = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
    price_data = pd.DataFrame(
        index=dates,
        columns=pd.MultiIndex.from_product([["A"], ["Close"]], names=["Ticker", "Field"]),
    )
    price_data.loc[:, ("A", "Close")] = [100.0, 101.0, 102.0, 103.0, 104.0]
    rets_daily = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01, 0.01]}, index=dates)
    return sized_signals, price_data, rets_daily


def test_calculate_portfolio_returns_standard(sample_data):
    sized_signals, price_data, rets_daily = sample_data
    scenario_config = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    global_config = {"feature_flags": {"ndarray_simulation": True}}

    returns, tracker = calculate_portfolio_returns(
        sized_signals, scenario_config, price_data, rets_daily, ["A"], global_config
    )

    assert isinstance(returns, pd.Series)
    assert tracker is None
    # Check that returns are calculated (first day 0 due to shift)
    assert returns.iloc[0] == 0.0
    assert returns.iloc[1] > 0


def test_calculate_portfolio_returns_with_trade_tracking(sample_data):
    sized_signals, price_data, rets_daily = sample_data
    scenario_config = {
        "timing_config": {"rebalance_frequency": "D"},
        "allocation_mode": "fixed_capital",
    }
    global_config = {"feature_flags": {"ndarray_simulation": True}, "portfolio_value": 10000.0}

    returns, tracker = calculate_portfolio_returns(
        sized_signals,
        scenario_config,
        price_data,
        rets_daily,
        ["A"],
        global_config,
        track_trades=True,
    )

    assert tracker is not None
    assert tracker.initial_portfolio_value == 10000.0
    assert tracker.allocation_mode == "fixed_capital"


@patch("portfolio_backtester.backtester_logic.portfolio_logic.StrategyResolverFactory.create")
def test_calculate_meta_strategy_portfolio_returns(mock_factory, sample_data):
    sized_signals, price_data, rets_daily = sample_data

    # Setup mock strategy and resolver
    mock_resolver = mock_factory.return_value
    mock_resolver.is_meta_strategy.return_value = True

    mock_strategy = MagicMock()
    mock_aggregator = mock_strategy.get_trade_aggregator.return_value
    mock_aggregator.get_aggregated_trades.return_value = [MagicMock()]
    mock_aggregator.get_portfolio_timeline.return_value = pd.DataFrame(
        {"returns": [0.01] * 5}, index=sized_signals.index
    )

    returns, tracker = calculate_portfolio_returns(
        sized_signals, {}, price_data, rets_daily, ["A"], {}, strategy=mock_strategy
    )

    assert (returns == 0.01).all()
