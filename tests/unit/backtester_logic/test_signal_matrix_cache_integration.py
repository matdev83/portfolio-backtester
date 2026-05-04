"""Integration tests for signal matrix cache in :func:`generate_signals`."""

from __future__ import annotations

from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pandas as pd

from portfolio_backtester.backtester_logic.strategy_logic import generate_signals
from portfolio_backtester.optimization.signal_cache import SignalCache, default_never_timed_out
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
from portfolio_backtester.strategies._core.target_generation import StrategyContext


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


class CacheableSignalStrategy(BaseStrategy):
    signal_matrix_cache_deterministic = True
    calls = 0

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        CacheableSignalStrategy.calls += 1
        cols = list(all_historical_data.columns.get_level_values("Ticker").unique())
        w = {c: 0.0 for c in cols}
        if cols:
            w[cols[0]] = 1.0
        return pd.DataFrame([w], index=[current_date])

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        cols = list(context.universe_tickers)
        rows: list[dict[str, float]] = []
        for _d in context.rebalance_dates:
            CacheableSignalStrategy.calls += 1
            w = {c: 0.0 for c in cols}
            if cols:
                w[cols[0]] = 1.0
            rows.append(w)
        return pd.DataFrame(rows, index=context.rebalance_dates, columns=cols)


class NotEligibleStrategy(BaseStrategy):
    calls = 0

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        NotEligibleStrategy.calls += 1
        cols = list(all_historical_data.columns.get_level_values("Ticker").unique())
        w = {c: 0.0 for c in cols}
        if cols:
            w[cols[0]] = 1.0
        return pd.DataFrame([w], index=[current_date])

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        cols = list(context.universe_tickers)
        rows: list[dict[str, float]] = []
        for _d in context.rebalance_dates:
            NotEligibleStrategy.calls += 1
            w = {c: 0.0 for c in cols}
            if cols:
                w[cols[0]] = 1.0
            rows.append(w)
        return pd.DataFrame(rows, index=context.rebalance_dates, columns=cols)


def test_feature_flag_off_no_cache_repeated_work() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = CacheableSignalStrategy({})
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}
    gc = {"feature_flags": {"signal_cache": False}}
    CacheableSignalStrategy.calls = 0
    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    a = CacheableSignalStrategy.calls
    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    b = CacheableSignalStrategy.calls
    assert a == 3
    assert b == 6


def test_cache_hit_skips_strategy_calls() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = CacheableSignalStrategy({})
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}
    gc: dict[str, Any] = {
        "feature_flags": {"signal_cache": True},
        "_signal_matrix_cache": SignalCache(),
    }
    CacheableSignalStrategy.calls = 0
    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    assert CacheableSignalStrategy.calls == 3
    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    assert CacheableSignalStrategy.calls == 3
    st = gc["_signal_matrix_cache"].stats()
    assert st["hits"] >= 1


def test_cached_returns_do_not_mutate_store() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = CacheableSignalStrategy({})
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}
    gc = {"feature_flags": {"signal_cache": True}, "_signal_matrix_cache": SignalCache()}
    r1 = generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    r2 = generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    orig = float(r2.iloc[0, 0])
    r1.iloc[0, 0] = 999.0
    r3 = generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    assert float(r3.iloc[0, 0]) == orig


def test_non_eligible_strategy_never_caches() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    ohlc = _make_mi_ohlc(dates, ["X"], "SPY")
    strat = NotEligibleStrategy({})
    s = cast(Any, strat)
    s.get_timing_controller = MagicMock(return_value=_timing(dates))
    s._cached_close_prices = {}
    s._cached_universe_prices = {}
    gc = {"feature_flags": {"signal_cache": True}, "_signal_matrix_cache": SignalCache()}
    NotEligibleStrategy.calls = 0
    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    generate_signals(
        strategy=strat,
        scenario_config={"timing_config": {"rebalance_frequency": "D"}},
        price_data_daily_ohlc=ohlc,
        universe_tickers=["X"],
        benchmark_ticker="SPY",
        has_timed_out=default_never_timed_out,
        global_config=gc,
    )
    assert NotEligibleStrategy.calls == 4
