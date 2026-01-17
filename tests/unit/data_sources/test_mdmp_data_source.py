import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from portfolio_backtester.data_sources.mdmp_data_source import MarketDataMultiProviderDataSource


class TestMarketDataMultiProviderDataSource:
    @pytest.fixture
    def mock_client(self):
        with patch(
            "portfolio_backtester.data_sources.mdmp_data_source.MarketDataMultiProviderDataSource.__init__",
            return_value=None,
        ):
            # Manually instantiate and set client
            source = MarketDataMultiProviderDataSource()
            source.client = MagicMock()
            source.data_dir = None
            source._min_coverage_ratio = None
            return source

    def test_get_data_success(self, mock_client):
        # Setup mock return data
        dates = pd.date_range("2020-01-01", periods=2)
        df_spy = pd.DataFrame(
            {"Open": [100, 101], "Close": [102, 103], "Volume": [1000, 2000]}, index=dates
        )

        # MDMP returns a dict of DataFrames
        # Mock client.fetch_many
        mock_client.client.fetch_many.return_value = {"US:SPY": df_spy}

        # Manually set the helper method _parse_date since we skipped __init__
        # Actually __init__ doesn't set _parse_date, it's a method on the class

        result = mock_client.get_data(["SPY"], "2020-01-01", "2020-01-02")

        assert not result.empty
        assert isinstance(result.columns, pd.MultiIndex)
        assert ("SPY", "Close") in result.columns
        assert result.loc["2020-01-01", ("SPY", "Close")] == 102

    def test_get_data_empty(self, mock_client):
        mock_client.client.fetch_many.return_value = {}

        result = mock_client.get_data(["SPY"], "2020-01-01", "2020-01-02")

        assert result.empty

    def test_get_data_partial_failure(self, mock_client):
        dates = pd.date_range("2020-01-01", periods=2)
        df_valid = pd.DataFrame({"Close": [10, 11]}, index=dates)

        mock_client.client.fetch_many.return_value = {
            "US:VALID": df_valid,
            "US:INVALID": pd.DataFrame(),  # Empty
        }

        result = mock_client.get_data(["VALID", "INVALID"], "2020-01-01", "2020-01-02")

        assert "VALID" in result.columns.get_level_values("Ticker")
        assert "INVALID" not in result.columns.get_level_values("Ticker")

    def test_normalize_ohlcv_missing_columns(self, mock_client):
        df_bad = pd.DataFrame({"Foo": [1, 2]})

        result = mock_client._normalize_ohlcv(df_bad, "BAD")
        assert result is None

    def test_normalize_ohlcv_case_insensitive(self, mock_client):
        dates = pd.date_range("2020-01-01", periods=1)
        df = pd.DataFrame({"open": [1], "CLOSE": [2]}, index=dates)

        result = mock_client._normalize_ohlcv(df, "TEST")

        assert result is not None
        assert ("TEST", "Open") in result.columns
        assert ("TEST", "Close") in result.columns
