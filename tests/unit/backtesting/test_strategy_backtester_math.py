import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester


def _make_returns(vals):
    dates = pd.date_range(start="2020-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=dates)


def test_calculate_max_drawdown_simple():
    # synthetic equity curve: up then down 50% => max DD -0.5
    rets = _make_returns([0.1, 0.1, -0.5])
    bt = StrategyBacktester(global_config={"benchmark": "SPY"}, data_source=None)
    dd = bt._calculate_max_drawdown(rets)
    assert np.isclose(dd, -0.5, atol=1e-6)


def test_drawdown_series_monotonic():
    rets = _make_returns([0.05]*10 + [-0.1]*5)
    bt = StrategyBacktester(global_config={"benchmark": "SPY"}, data_source=None)
    dd_series = bt._calculate_drawdown_series(rets.cumsum())
    # Drawdown series should never exceed 0 and end negative after loss
    assert (dd_series <= 0).all()
    assert dd_series.iloc[-1] < 0


def test_rolling_sharpe_basic():
    # constant positive return gives infinite sharpe (std 0 -> result 0 per implementation)
    rets = _make_returns([0.01]*300)
    bt = StrategyBacktester(global_config={"benchmark": "SPY"}, data_source=None)
    sharpe = bt._calculate_rolling_sharpe(rets, window=252)
    # With zero rolling std the Sharpe formula yields inf
    assert np.isinf(sharpe.iloc[-1])