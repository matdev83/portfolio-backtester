import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.cache.window_processor import WindowProcessor


def test_compute_returns_preserves_nan_no_fill_zero() -> None:
    idx = pd.date_range("2023-01-02", periods=4, freq="B")
    prices = pd.DataFrame({"x": [100.0, 101.0, np.nan, 103.0]}, index=idx)
    out = WindowProcessor.compute_returns(prices)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(101.0 / 100.0 - 1.0)
    assert pd.isna(out.iloc[2, 0])
    assert pd.isna(out.iloc[3, 0])


def test_compute_returns_leading_row_nan() -> None:
    idx = pd.date_range("2023-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"x": [100.0, 101.0, 102.0]}, index=idx)
    out = WindowProcessor.compute_returns(prices)
    assert pd.isna(out.iloc[0, 0])
