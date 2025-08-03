import pandas as pd
import numpy as np
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import (
    calculate_portfolio_returns, _calculate_position_weights
)


class DummyTxCostModel:
    """Simple stub that returns commission costs in various shapes."""

    def __init__(self, return_type: str):
        self.return_type = return_type

    def calculate(self, *args, **kwargs):  # pragma: no cover – args ignored for stub
        """Mimic the real `.calculate` signature accepting arbitrary kwargs."""
        turnover = kwargs.get("turnover")
        weights_daily = kwargs.get("weights_daily")

        if self.return_type == "scalar":
            return 0.001, {}
        elif self.return_type == "series":
            # Produce Series aligned to turnover index
            if turnover is None:
                raise ValueError("Turnover must be supplied for series output")
            return pd.Series(0.001, index=turnover.index), {}
        elif self.return_type == "dataframe":
            if turnover is None or weights_daily is None:
                raise ValueError("Turnover and weights_daily must be supplied for dataframe output")
            df = pd.DataFrame(0.001, index=turnover.index, columns=weights_daily.columns)
            return df, {}
        else:
            raise ValueError(f"Unsupported return_type {self.return_type}")


@pytest.mark.parametrize("return_type", ["scalar", "series", "dataframe"])
def test_calculate_portfolio_returns_handles_commission_shapes(monkeypatch, return_type):
    """calculate_portfolio_returns should correctly broadcast / align commission
    outputs of different shapes (scalar, Series, DataFrame) without raising and
    produce sensible net returns.
    """
    # Patch transaction-cost factory to return our dummy stub
    monkeypatch.setattr(
        "portfolio_backtester.trading.get_transaction_cost_model",
        lambda cfg: DummyTxCostModel(return_type),
    )

    # Create simple 5-day price and return series for two assets
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    tickers = ["AAA", "BBB"]

    # Price DataFrame (single-level columns is fine for the function)
    prices = pd.DataFrame({t: 100 + np.arange(5) for t in tickers}, index=dates)

    # Daily returns – constant 1 % and 2 % for simplicity
    rets_daily = pd.DataFrame({
        "AAA": np.full(5, 0.01),
        "BBB": np.full(5, 0.02),
    }, index=dates)

    # Sized signals – constant weights (will be rebalanced daily)
    sized_signals = pd.DataFrame({
        "AAA": np.full(5, 0.6),
        "BBB": np.full(5, 0.4),
    }, index=dates)

    scenario_config = {"timing_config": {"rebalance_frequency": "D"}}
    global_config = {"portfolio_value": 100_000}

    net_returns, trade_tracker = calculate_portfolio_returns(
        sized_signals=sized_signals,
        scenario_config=scenario_config,
        price_data_daily_ohlc=prices,
        rets_daily=rets_daily,
        universe_tickers=tickers,
        global_config=global_config,
        track_trades=False,
    )

    # Verify trade tracker is None when track_trades = False
    assert trade_tracker is None

    # Build expected gross returns: first day 0 (weights are shifted), then 0.014
    expected_gross = pd.Series([0.0] + [0.014] * 4, index=dates)

    # Commission – always 0.001 per day
    if return_type == "scalar":
        expected_net = expected_gross - 0.001
        pd.testing.assert_series_equal(net_returns, expected_net)
    elif return_type == "series":
        expected_net = expected_gross - 0.001
        pd.testing.assert_series_equal(net_returns, expected_net)
    else:  # DataFrame commission output aggregated across assets
        expected_net = expected_gross - 0.001 * len(tickers)
        pd.testing.assert_series_equal(net_returns, expected_net)


def test_calculate_position_weights_basic():
    """_calculate_position_weights should convert position quantities into weights."""
    current_positions = {"AAA": 10, "BBB": 5}  # quantities
    prices = pd.Series({"AAA": 100, "BBB": 200})
    base_portfolio_value = 10 * 100 + 5 * 200  # 1000 + 1000 = 2000

    weights = _calculate_position_weights(current_positions, prices, base_portfolio_value)

    # Expected weights
    expected_weights = pd.Series({
        "AAA": (10 * 100) / base_portfolio_value,
        "BBB": (5 * 200) / base_portfolio_value,
    }, index=["AAA", "BBB"])

    pd.testing.assert_series_equal(weights, expected_weights)

    # Assets with zero quantity or missing price should produce zero weight
    current_positions_extra = {"AAA": 10, "BBB": 5, "CCC": 0}
    prices_extra = pd.Series({"AAA": 100, "BBB": 200, "CCC": 50})
    weights_extra = _calculate_position_weights(current_positions_extra, prices_extra, base_portfolio_value)

    # Weight for CCC should be zero
    assert weights_extra["CCC"] == 0.0
