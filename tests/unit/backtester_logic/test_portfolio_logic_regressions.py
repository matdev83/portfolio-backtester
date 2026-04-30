import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import (
    calculate_portfolio_returns,
)


def test_signal_based_intramonth_weights_are_not_rebucketed_to_month_end() -> None:
    dates = pd.bdate_range("2023-01-02", "2023-01-31")
    prices = pd.DataFrame({"A": 100.0}, index=dates)
    prices.loc[pd.Timestamp("2023-01-23") :, "A"] = 120.0
    returns = prices.pct_change(fill_method=None).fillna(0.0)

    sized_signals = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2023-01-10")])

    scenario_config = {
        "timing_config": {"mode": "signal_based", "rebalance_frequency": "ME"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "portfolio_value": 10000.0,
    }

    portfolio_returns, _ = calculate_portfolio_returns(
        sized_signals=sized_signals,
        scenario_config=scenario_config,
        price_data_daily_ohlc=prices,
        rets_daily=returns,
        universe_tickers=["A"],
        global_config=global_config,
        track_trades=False,
    )

    assert portfolio_returns.loc[pd.Timestamp("2023-01-23")] == pytest.approx(0.20)
