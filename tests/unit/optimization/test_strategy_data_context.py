"""Strategy data context injection into :func:`generate_signals`."""

from __future__ import annotations

from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pandas as pd

from portfolio_backtester.backtester_logic.strategy_logic import generate_signals
from portfolio_backtester.optimization.strategy_data_context import StrategyDataContext
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy


def _make_mi_ohlc(
    dates: pd.DatetimeIndex, universe_tickers: list[str], benchmark_ticker: str
) -> pd.DataFrame:
    dfs = []
    for t in universe_tickers + [benchmark_ticker]:
        df = pd.DataFrame(
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.0,
                "Volume": 1e6,
            },
            index=dates,
        )
        df.columns = pd.MultiIndex.from_product([[t], df.columns], names=["Ticker", "Field"])
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def _timing(dates: pd.DatetimeIndex) -> MagicMock:
    timing_controller = MagicMock()
    timing_controller.reset_state.return_value = None
    timing_controller.get_rebalance_dates.return_value = dates
    timing_controller.should_generate_signal.return_value = True
    return timing_controller


class OldKwStrategy(BaseStrategy):
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame([[1.0]], index=[current_date], columns=["X"])


class DataCtxStrategy(BaseStrategy):
    last_ctx: StrategyDataContext | None = None

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        data_context: StrategyDataContext | None = None,
    ) -> pd.DataFrame:
        DataCtxStrategy.last_ctx = data_context
        return pd.DataFrame([[1.0]], index=[current_date], columns=["X"])


class KwargsStrategy(BaseStrategy):
    last_kwargs: dict

    def __init__(self) -> None:
        super().__init__({})
        self.last_kwargs = {}

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.last_kwargs = {
            "all_historical_data": all_historical_data,
            "benchmark_historical_data": benchmark_historical_data,
            "non_universe_historical_data": non_universe_historical_data,
            "current_date": current_date,
            "start_date": start_date,
            "end_date": end_date,
            **kwargs,
        }
        cd = current_date
        cols = list(all_historical_data.columns.get_level_values("Ticker").unique())
        w = {c: 0.0 for c in cols}
        if cols:
            w[cols[0]] = 1.0
        return pd.DataFrame([w], index=[cd])


def test_flag_off_no_data_context_passed_to_kwargs_strategy() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = KwargsStrategy()
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}

    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=MagicMock(return_value=False),
        global_config={"feature_flags": {"strategy_data_context": False}},
    )
    assert "data_context" not in strat.last_kwargs


def test_flag_on_kwargs_strategy_receives_data_context() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = KwargsStrategy()
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}

    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=MagicMock(return_value=False),
        global_config={"feature_flags": {"strategy_data_context": True}},
    )
    assert "data_context" in strat.last_kwargs
    ctx = strat.last_kwargs["data_context"]
    assert isinstance(ctx, StrategyDataContext)
    assert ctx.benchmark_ticker == "SPY"


def test_old_signature_strategy_flag_on_does_not_receive_data_context_kwarg() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = OldKwStrategy({})
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}

    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=MagicMock(return_value=False),
        global_config={"feature_flags": {"strategy_data_context": True}},
    )


def test_explicit_data_context_strategy_receives_context_when_flag_on() -> None:
    DataCtxStrategy.last_ctx = None
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = DataCtxStrategy({})
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}

    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=MagicMock(return_value=False),
        global_config={"feature_flags": {"strategy_data_context": True}},
    )
    assert DataCtxStrategy.last_ctx is not None
    assert DataCtxStrategy.last_ctx.panel is not None
