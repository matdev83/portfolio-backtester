import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import MagicMock, patch
from portfolio_backtester.data_sources.mdmp_data_source import MarketDataMultiProviderDataSource

@pytest.fixture
def mock_mdmp_client():
    with patch("market_data_multi_provider.MarketDataClient") as mock:
        yield mock

def test_mdmp_data_source_init():
    # Test initialization with and without data_dir
    with patch("market_data_multi_provider.MarketDataClient") as mock_client:
        ds = MarketDataMultiProviderDataSource(data_dir="test_dir")
        assert ds.data_dir.name == "test_dir"
        mock_client.assert_called_once_with(data_dir="test_dir")

def test_mdmp_data_source_import_error():
    # Test behavior when MDMP package is missing
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="market-data-multi-provider is not installed"):
            MarketDataMultiProviderDataSource()

def test_mdmp_get_data_success(mock_mdmp_client):
    # Setup mock client behavior
    client_instance = mock_mdmp_client.return_value
    
    # Mock data for SPY
    spy_data = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [1000, 1100]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
    
    # MDMP returns Dict[canonical_id, DataFrame]
    # to_canonical_id("SPY") -> "AMEX:SPY" (assumed mapping)
    client_instance.fetch_many.return_value = {
        "AMEX:SPY": spy_data
    }
    
    with patch("portfolio_backtester.data_sources.mdmp_data_source.to_canonical_id", return_value="AMEX:SPY"):
        ds = MarketDataMultiProviderDataSource()
        result = ds.get_data(["SPY"], "2023-01-01", "2023-01-02")
    
    assert isinstance(result.columns, pd.MultiIndex)
    assert ("SPY", "Close") in result.columns
    assert len(result) == 2
    assert result[("SPY", "Close")].iloc[0] == 101.0

def test_mdmp_get_data_empty_input():
    ds = MarketDataMultiProviderDataSource()
    result = ds.get_data([], "2023-01-01", "2023-01-02")
    assert result.empty

def test_mdmp_normalize_ohlcv():
    ds = MarketDataMultiProviderDataSource()
    
    # Test with lowercase columns
    raw_df = pd.DataFrame({
        "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]
    })
    
    normalized = ds._normalize_ohlcv(raw_df, "TEST")
    assert ("TEST", "Open") in normalized.columns
    assert ("TEST", "Close") in normalized.columns
    
    # Test with missing Close
    bad_df = pd.DataFrame({"Open": [1.0]})
    assert ds._normalize_ohlcv(bad_df, "TEST") is None

def test_mdmp_get_data_partial_failure(mock_mdmp_client):
    client_instance = mock_mdmp_client.return_value
    
    spy_data = pd.DataFrame({
        "Close": [100.0]
    }, index=pd.to_datetime(["2023-01-01"]))
    
    # SPY success, AAPL None
    client_instance.fetch_many.return_value = {
        "AMEX:SPY": spy_data,
        "NASDAQ:AAPL": None
    }
    
    def side_effect(ticker):
        return "AMEX:SPY" if ticker == "SPY" else "NASDAQ:AAPL"
        
    with patch("portfolio_backtester.data_sources.mdmp_data_source.to_canonical_id", side_effect=side_effect):
        ds = MarketDataMultiProviderDataSource()
        result = ds.get_data(["SPY", "AAPL"], "2023-01-01", "2023-01-01")
    
    assert ("SPY", "Close") in result.columns
    assert ("AAPL", "Close") not in result.columns
    assert len(result) == 1
