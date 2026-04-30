"""Regression: method universes resolve point-in-time during signal generation."""

from unittest.mock import MagicMock

import pandas as pd

from portfolio_backtester.backtester_logic.strategy_logic import generate_signals
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


def test_generate_signals_point_in_time_method_universe_membership_changes_signals():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    universe_tickers = ["AAA", "BBB", "CCC"]
    benchmark_ticker = "SPY"
    ohlc = _make_mi_ohlc(dates, universe_tickers, benchmark_ticker)

    timing_controller = MagicMock()
    timing_controller.reset_state.return_value = None
    timing_controller.get_rebalance_dates.return_value = dates
    timing_controller.should_generate_signal.return_value = True

    universe_provider = MagicMock()
    universe_provider.supports_dynamic_universe.return_value = True

    pivot_date = pd.Timestamp("2020-01-03")

    def mock_get_universe_method_with_date(_gc, current_date):
        if current_date <= pivot_date:
            return [("AAA", 1.0)]
        return [("BBB", 1.0)]

    def mock_generate_signals(
        all_historical_data,
        benchmark_historical_data,
        non_universe_historical_data=None,
        current_date=None,
        **kwargs,
    ):
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            cols = list(all_historical_data.columns.get_level_values("Ticker").unique())
        else:
            cols = list(all_historical_data.columns)
        weights = {c: 0.0 for c in cols}
        if cols:
            weights[cols[0]] = 1.0
        return pd.DataFrame([weights], index=[current_date])

    strategy = MagicMock(spec=BaseStrategy)
    strategy.get_timing_controller.return_value = timing_controller
    strategy.get_universe_provider.return_value = universe_provider
    strategy.get_universe_method_with_date.side_effect = mock_get_universe_method_with_date
    strategy.generate_signals.side_effect = mock_generate_signals
    strategy.get_non_universe_data_requirements.return_value = []
    strategy._cached_close_prices = {}
    strategy._cached_universe_prices = {}

    scenario_config = {"timing_config": {"rebalance_frequency": "D"}}

    signals = generate_signals(
        strategy=strategy,
        scenario_config=scenario_config,
        price_data_daily_ohlc=ohlc,
        universe_tickers=universe_tickers,
        benchmark_ticker=benchmark_ticker,
        has_timed_out=MagicMock(return_value=False),
        global_config={"universe": universe_tickers},
    )

    assert list(signals.columns) == universe_tickers

    early_row = signals.loc[dates[0]]
    assert early_row["AAA"] == 1.0
    assert early_row["BBB"] == 0.0

    late_row = signals.loc[dates[-1]]
    assert late_row["BBB"] == 1.0
    assert late_row["AAA"] == 0.0
