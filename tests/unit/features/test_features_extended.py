import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from portfolio_backtester.features.atr import ATRFeature
from portfolio_backtester.features.vams import VAMS
from portfolio_backtester.features.dp_vams import DPVAMS

@pytest.fixture
def ohlc_data():
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    tickers = ["A", "B"]
    
    # Create MultiIndex OHLC
    data = {}
    for t in tickers:
        data[(t, "Open")] = np.linspace(100, 110, 20)
        data[(t, "High")] = np.linspace(101, 111, 20)
        data[(t, "Low")] = np.linspace(99, 109, 20)
        data[(t, "Close")] = np.linspace(100.5, 110.5, 20)
        
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Ticker", "Field"])
    return df

@pytest.fixture
def close_data():
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    data = pd.DataFrame({
        "A": np.linspace(100, 110, 20),
        "B": np.linspace(50, 60, 20)
    }, index=dates)
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
    assert result["C"].isna().all() # Should handle KeyError and return NaN

def test_vams_compute_standard(close_data):
    vams = VAMS(lookback_months=3) # Need >1 for std calculation
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
        assert mock_fast.call_count == 2 # Once for A, once for B
