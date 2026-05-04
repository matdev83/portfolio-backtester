"""Preloaded OHLC/price frame data source for in-process backtests."""

from __future__ import annotations

import pandas as pd

from portfolio_backtester.interfaces.data_source_interface import IDataSource


class PreloadedFrameDataSource(IDataSource):
    """Returns a fixed DataFrame for every ``get_data`` call.

    Used when daily OHLC is already materialized (e.g. ``BacktestRunner.run_backtest_mode``)
    and ``StrategyBacktester`` only needs a ``BaseDataSource``-shaped dependency. Tickers and
    date bounds are ignored; callers are responsible for passing aligned frames into
    ``backtest_strategy`` separately.
    """

    def __init__(self, daily_frame: pd.DataFrame) -> None:
        self._daily_frame = daily_frame

    def get_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self._daily_frame
