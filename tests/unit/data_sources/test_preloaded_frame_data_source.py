"""Tests for PreloadedFrameDataSource."""

import pandas as pd

from portfolio_backtester.data_sources.preloaded_frame_data_source import PreloadedFrameDataSource


def test_get_data_returns_same_frame_ignoring_tickers_and_dates() -> None:
    frame = pd.DataFrame({"A": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    src = PreloadedFrameDataSource(frame)
    out = src.get_data(["OTHER"], "1999-01-01", "2099-12-31")
    assert out is frame
    assert len(out) == 2
