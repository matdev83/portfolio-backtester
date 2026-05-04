"""signal_based sparse targets: skipped scans must not become zero-weight events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import _sized_signals_to_weights_daily
from portfolio_backtester.backtester_logic.strategy_logic import (
    LegacyGenerateSignalsAdapter,
    generate_signals,
)
from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
from portfolio_backtester.timing.trade_execution_timing import (
    map_sparse_target_weights_to_execution_dates,
)


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


@pytest.mark.parametrize("trade_execution_timing", ["bar_close", "next_bar_open"])
def test_skipped_signal_generation_does_not_flatten_after_portfolio_remap(
    trade_execution_timing: str,
) -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    universe_tickers = ["AAA", "BBB"]
    benchmark_ticker = "SPY"
    ohlc = _make_mi_ohlc(dates, universe_tickers, benchmark_ticker)
    scan_dates = dates[:3]

    timing_controller = MagicMock()
    timing_controller.reset_state.return_value = None
    timing_controller.get_rebalance_dates.return_value = scan_dates

    should_flags = iter([True, False, True])

    def _should_generate(*_args: object, **_kwargs: object) -> bool:
        return next(should_flags)

    timing_controller.should_generate_signal.side_effect = _should_generate

    triggered: list[pd.Timestamp] = []

    def _mock_generate_signals(
        all_historical_data: object,
        benchmark_historical_data: object,
        non_universe_historical_data: object = None,
        current_date: pd.Timestamp | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        assert current_date is not None
        triggered.append(pd.Timestamp(current_date))
        return pd.DataFrame([{"AAA": 1.0, "BBB": 0.0}], index=[current_date])

    strategy = MagicMock(spec=BaseStrategy)
    strategy.get_timing_controller.return_value = timing_controller
    strategy.generate_signals.side_effect = _mock_generate_signals
    strategy.get_non_universe_data_requirements.return_value = []
    strategy.get_trade_execution_timing.return_value = trade_execution_timing
    strategy._cached_close_prices = {}
    strategy._cached_universe_prices = {}

    scenario_config = CanonicalScenarioConfig.from_dict(
        {
            "name": "sparse_skip",
            "strategy": "dummy",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )

    signals = generate_signals(
        strategy=LegacyGenerateSignalsAdapter(strategy),
        scenario_config=scenario_config,
        price_data_daily_ohlc=ohlc,
        universe_tickers=universe_tickers,
        benchmark_ticker=benchmark_ticker,
        has_timed_out=MagicMock(return_value=False),
    )

    assert triggered == [scan_dates[0], scan_dates[2]]
    assert signals.loc[scan_dates[1]].isna().all()

    mapped = map_sparse_target_weights_to_execution_dates(
        signals,
        trade_execution_timing=trade_execution_timing,
        calendar=pd.DatetimeIndex(ohlc.index),
    )

    weights_daily = _sized_signals_to_weights_daily(mapped, universe_tickers, ohlc.index)
    mid = weights_daily.loc[scan_dates[1]]

    assert float(mid["AAA"]) == pytest.approx(1.0)
    assert float(mid["BBB"]) == pytest.approx(0.0)


def test_signal_based_explicit_all_zero_row_is_kept() -> None:
    dates = pd.date_range("2020-03-02", periods=5, freq="B")
    universe_tickers = ["AAA", "BBB"]
    benchmark_ticker = "SPY"
    ohlc = _make_mi_ohlc(dates, universe_tickers, benchmark_ticker)

    timing_controller = MagicMock()
    timing_controller.reset_state.return_value = None
    timing_controller.get_rebalance_dates.return_value = dates[:3]
    timing_controller.should_generate_signal.return_value = True

    strategy = MagicMock(spec=BaseStrategy)
    strategy.get_timing_controller.return_value = timing_controller
    strategy.generate_signals.side_effect = [
        pd.DataFrame([{"AAA": 1.0, "BBB": 0.0}], index=[dates[0]]),
        pd.DataFrame([{"AAA": 0.0, "BBB": 0.0}], index=[dates[1]]),
        pd.DataFrame([{"AAA": 1.0, "BBB": 0.0}], index=[dates[2]]),
    ]
    strategy.get_non_universe_data_requirements.return_value = []
    strategy._cached_close_prices = {}
    strategy._cached_universe_prices = {}

    scenario_config = CanonicalScenarioConfig.from_dict(
        {
            "name": "sparse_zero",
            "strategy": "dummy",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )

    signals = generate_signals(
        strategy=LegacyGenerateSignalsAdapter(strategy),
        scenario_config=scenario_config,
        price_data_daily_ohlc=ohlc,
        universe_tickers=universe_tickers,
        benchmark_ticker=benchmark_ticker,
        has_timed_out=MagicMock(return_value=False),
    )

    flat = signals.loc[dates[1]]
    assert (flat == 0.0).all()
    mapped = map_sparse_target_weights_to_execution_dates(
        signals,
        trade_execution_timing="bar_close",
        calendar=pd.DatetimeIndex(ohlc.index),
    )
    assert dates[1] in mapped.index
