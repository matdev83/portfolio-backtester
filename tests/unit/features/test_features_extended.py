import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from portfolio_backtester.features.atr import ATRFeature
from portfolio_backtester.features.vams import VAMS
from portfolio_backtester.features.dp_vams import DPVAMS


@pytest.fixture
def ohlc_data():
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    tickers = ["A", "B"]

    # Create MultiIndex OHLC
    data: dict[tuple[str, str], np.ndarray] = {}
    for t in tickers:
        data[(t, "Open")] = np.linspace(100, 110, 20)
        data[(t, "High")] = np.linspace(101, 111, 20)
        data[(t, "Low")] = np.linspace(99, 109, 20)
        data[(t, "Close")] = np.linspace(100.5, 110.5, 20)

    df = pd.DataFrame(data, index=dates)
    # Explicit MultiIndex construction (type-safe for mypy)
    df.columns = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close"]],
        names=["Ticker", "Field"],
    )
    return df


@pytest.fixture
def close_data():
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    data = pd.DataFrame({"A": np.linspace(100, 110, 20), "B": np.linspace(50, 60, 20)}, index=dates)
    return data


def test_atr_compute_standard(ohlc_data):
    atr = ATRFeature(atr_period=14)
    result = atr.compute(ohlc_data)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "A" in result.columns
    assert "B" in result.columns
    # First 13 values should be NaN (period-1)
    assert result["A"].iloc[:13].isna().all()
    # 14th value onwards should be calculated
    assert not result["A"].iloc[14:].isna().all()


def test_atr_compute_empty_data():
    atr = ATRFeature(atr_period=14)
    result = atr.compute(pd.DataFrame())
    assert result.empty


def test_atr_compute_invalid_format(close_data):
    atr = ATRFeature(atr_period=14)
    # Passed simple close data instead of MultiIndex OHLC
    with pytest.raises(ValueError, match="ATR calculation requires OHLC data"):
        atr.compute(close_data)


def test_atr_compute_missing_ticker_data(ohlc_data):
    # Add a ticker that is missing full OHLC
    # Actually, MultiIndex structure enforces columns.
    # But if we access a ticker that doesn't have all fields?
    # The code iterates over `data.columns.get_level_values("Ticker").unique()`
    # If a ticker is in that list, we try to access High/Low/Close.
    # If partial data is missing, pandas might raise KeyError.

    # Let's create data where ticker C has only Close
    dates = ohlc_data.index
    c_close = pd.DataFrame({("C", "Close"): np.ones(len(dates))}, index=dates)
    c_close.columns.names = ["Ticker", "Field"]

    mixed_data = pd.concat([ohlc_data, c_close], axis=1)

    atr = ATRFeature(atr_period=14)
    result = atr.compute(mixed_data)

    assert "C" in result.columns
    assert result["C"].isna().all()  # Should handle KeyError and return NaN


def test_vams_compute_standard(close_data):
    vams = VAMS(lookback_months=3)  # Need >1 for std calculation
    result = vams.compute(close_data)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape == close_data.shape
    assert not result.isna().all().all()


def test_vams_compute_empty(close_data):
    vams = VAMS(lookback_months=12)
    result = vams.compute(pd.DataFrame())
    assert result.empty


def test_dp_vams_compute_standard(close_data):
    dp_vams = DPVAMS(lookback_months=1, alpha=2.0)
    result = dp_vams.compute(close_data)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape == close_data.shape

    # DPVAMS should handle NaNs by filling 0
    assert not result.isna().any().any()


def test_dp_vams_compute_empty(close_data):
    dp = DPVAMS(lookback_months=12, alpha=2.0)
    result = dp.compute(pd.DataFrame())
    assert result.empty


def test_atr_numba_integration(ohlc_data):
    # Verify Numba function is called
    atr = ATRFeature(atr_period=14)

    with patch("portfolio_backtester.features.atr.atr_fast_fixed") as mock_fast:
        mock_fast.return_value = np.zeros(len(ohlc_data))
        atr.compute(ohlc_data)
        assert mock_fast.call_count == 2  # Once for A, once for B


def test_atr_compute_single_row_multiindex_returns_nan() -> None:
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    cols = pd.MultiIndex.from_product(
        [["A"], ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
    )
    df = pd.DataFrame([[100.0, 101.0, 99.0, 100.0]], index=dates, columns=cols)

    atr = ATRFeature(atr_period=14)
    out = atr.compute(df)

    assert list(out.index) == list(dates)
    assert list(out.columns) == ["A"]
    assert out.shape == (1, 1)
    assert out["A"].isna().all()


def test_atr_compute_all_nan_ohlc_returns_nan_column() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = pd.MultiIndex.from_product(
        [["A"], ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
    )
    df = pd.DataFrame(np.nan, index=dates, columns=cols)

    atr = ATRFeature(atr_period=14)
    out = atr.compute(df)

    assert out.shape == (len(dates), 1)
    assert out["A"].isna().all()


def test_dp_vams_compute_single_row_preserves_shape_and_no_nans() -> None:
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    close = pd.DataFrame({"A": [100.0], "B": [50.0]}, index=dates)

    dp = DPVAMS(lookback_months=12, alpha=2.0)
    out = dp.compute(close)

    assert out.shape == close.shape
    pd.testing.assert_index_equal(out.index, close.index)
    pd.testing.assert_index_equal(out.columns, close.columns)
    assert not out.isna().any().any()


def test_dp_vams_name_is_stable_and_parameterized() -> None:
    dp = DPVAMS(lookback_months=3, alpha=2.0)
    assert dp.name == "dp_vams_3m_2.00a"
