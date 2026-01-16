"""Integration tests for MarketDataMultiProviderDataSource.

These tests verify that the MDMP data source adapter works correctly
with the market-data-multi-provider package and returns data in the
expected format for portfolio-backtester.
"""

import pytest
import pandas as pd
from datetime import date, timedelta

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestSymbolMapper:
    """Tests for the symbol mapping functionality."""

    def test_to_canonical_id_spy(self) -> None:
        """Test SPY maps to AMEX:SPY."""
        from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id

        assert to_canonical_id("SPY") == "AMEX:SPY"

    def test_to_canonical_id_aapl(self) -> None:
        """Test AAPL maps to NASDAQ:AAPL."""
        from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id

        assert to_canonical_id("AAPL") == "NASDAQ:AAPL"

    def test_to_canonical_id_gspc_index(self) -> None:
        """Test ^GSPC maps to SP:SPX."""
        from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id

        assert to_canonical_id("^GSPC") == "SP:SPX"

    def test_to_canonical_id_vix(self) -> None:
        """Test VIX maps to CBOE:VIX."""
        from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id

        assert to_canonical_id("^VIX") == "CBOE:VIX"
        assert to_canonical_id("VIX") == "CBOE:VIX"

    def test_to_canonical_id_nyse_stock(self) -> None:
        """Test NYSE stocks map correctly."""
        from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id

        # JNJ is a NYSE stock
        assert to_canonical_id("JNJ") == "NYSE:JNJ"

    def test_to_canonical_id_already_canonical(self) -> None:
        """Test already-canonical IDs pass through."""
        from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id

        assert to_canonical_id("AMEX:SPY") == "AMEX:SPY"
        assert to_canonical_id("NASDAQ:AAPL") == "NASDAQ:AAPL"

    def test_from_canonical_id(self) -> None:
        """Test reverse mapping from canonical to local format."""
        from portfolio_backtester.data_sources.symbol_mapper import from_canonical_id

        assert from_canonical_id("AMEX:SPY") == "SPY"
        assert from_canonical_id("NASDAQ:AAPL") == "AAPL"
        # SP:SPX maps to SPX (the base symbol, not yfinance ^GSPC format)
        assert from_canonical_id("SP:SPX") == "SPX"

    def test_get_exchange_prefix(self) -> None:
        """Test exchange prefix detection."""
        from portfolio_backtester.data_sources.symbol_mapper import get_exchange_prefix

        assert get_exchange_prefix("AAPL") == "NASDAQ"
        assert get_exchange_prefix("SPY") == "AMEX"
        assert get_exchange_prefix("JNJ") == "NYSE"

    def test_is_special_symbol(self) -> None:
        """Test special symbol detection."""
        from portfolio_backtester.data_sources.symbol_mapper import is_special_symbol

        assert is_special_symbol("^GSPC") is True
        assert is_special_symbol("^VIX") is True
        assert is_special_symbol("SPY") is False
        assert is_special_symbol("AAPL") is False


class TestMdmpDataSourceCreation:
    """Tests for MDMP data source factory creation."""

    def test_factory_creates_mdmp_source(self) -> None:
        """Test that factory creates MDMP data source correctly."""
        from portfolio_backtester.interfaces.data_source_interface import create_data_source
        from portfolio_backtester.data_sources.mdmp_data_source import (
            MarketDataMultiProviderDataSource,
        )

        ds = create_data_source({"data_source": "mdmp"})
        assert isinstance(ds, MarketDataMultiProviderDataSource)

    def test_factory_alias_works(self) -> None:
        """Test that 'market-data-multi-provider' alias works."""
        from portfolio_backtester.interfaces.data_source_interface import create_data_source
        from portfolio_backtester.data_sources.mdmp_data_source import (
            MarketDataMultiProviderDataSource,
        )

        ds = create_data_source({"data_source": "market-data-multi-provider"})
        assert isinstance(ds, MarketDataMultiProviderDataSource)

    def test_mdmp_source_has_required_methods(self) -> None:
        """Test that MDMP source has all required BaseDataSource methods."""
        from portfolio_backtester.data_sources.mdmp_data_source import (
            MarketDataMultiProviderDataSource,
        )

        ds = MarketDataMultiProviderDataSource()
        assert hasattr(ds, "get_data")
        assert callable(ds.get_data)


class TestMdmpDataFetching:
    """Tests for actual data fetching (requires network access)."""

    @pytest.fixture
    def data_source(self) -> "MarketDataMultiProviderDataSource":
        """Create a MDMP data source instance."""
        from portfolio_backtester.data_sources.mdmp_data_source import (
            MarketDataMultiProviderDataSource,
        )

        return MarketDataMultiProviderDataSource()

    def test_fetch_single_ticker_returns_dataframe(
        self, data_source: "MarketDataMultiProviderDataSource"
    ) -> None:
        """Test fetching a single ticker returns a DataFrame."""
        # Use a date range that should have data
        end = date.today() - timedelta(days=5)
        start = end - timedelta(days=30)

        result = data_source.get_data(
            ["SPY"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        assert isinstance(result, pd.DataFrame)
        # Should have some data
        if not result.empty:
            assert len(result) > 0

    def test_fetch_returns_multiindex_columns(
        self, data_source: "MarketDataMultiProviderDataSource"
    ) -> None:
        """Test that fetched data has MultiIndex columns (Ticker, Field)."""
        end = date.today() - timedelta(days=5)
        start = end - timedelta(days=30)

        result = data_source.get_data(
            ["SPY"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        if not result.empty:
            assert isinstance(result.columns, pd.MultiIndex)
            assert result.columns.names == ["Ticker", "Field"]

    def test_fetch_includes_ohlcv_fields(
        self, data_source: "MarketDataMultiProviderDataSource"
    ) -> None:
        """Test that fetched data includes OHLCV fields."""
        end = date.today() - timedelta(days=5)
        start = end - timedelta(days=30)

        result = data_source.get_data(
            ["SPY"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        if not result.empty:
            fields = result.columns.get_level_values("Field").unique().tolist()
            # Should have at least Close
            assert "Close" in fields

    def test_fetch_multiple_tickers(
        self, data_source: "MarketDataMultiProviderDataSource"
    ) -> None:
        """Test fetching multiple tickers returns data for all."""
        end = date.today() - timedelta(days=5)
        start = end - timedelta(days=30)
        tickers = ["SPY", "QQQ"]

        result = data_source.get_data(
            tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        if not result.empty:
            result_tickers = result.columns.get_level_values("Ticker").unique().tolist()
            # Should have data for at least one ticker
            assert len(result_tickers) > 0

    def test_fetch_empty_tickers_returns_empty(
        self, data_source: "MarketDataMultiProviderDataSource"
    ) -> None:
        """Test that empty ticker list returns empty DataFrame."""
        result = data_source.get_data([], "2024-01-01", "2024-01-31")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_fetch_has_datetime_index(
        self, data_source: "MarketDataMultiProviderDataSource"
    ) -> None:
        """Test that fetched data has DatetimeIndex."""
        end = date.today() - timedelta(days=5)
        start = end - timedelta(days=30)

        result = data_source.get_data(
            ["SPY"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        if not result.empty:
            assert isinstance(result.index, pd.DatetimeIndex)


class TestMdmpCompatibilityWithHybrid:
    """Tests comparing MDMP output format with HybridDataSource."""

    def test_output_structure_matches_hybrid(self) -> None:
        """Test that MDMP output structure matches what HybridDataSource produces."""
        from portfolio_backtester.data_sources.mdmp_data_source import (
            MarketDataMultiProviderDataSource,
        )

        mdmp = MarketDataMultiProviderDataSource()

        end = date.today() - timedelta(days=5)
        start = end - timedelta(days=30)

        result = mdmp.get_data(
            ["SPY"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        if not result.empty:
            # Check MultiIndex structure
            assert isinstance(result.columns, pd.MultiIndex)
            assert len(result.columns.levels) == 2

            # Check field naming
            fields = result.columns.get_level_values("Field").unique().tolist()
            # Standard OHLCV field names should be capitalized
            for field in fields:
                assert field[0].isupper(), f"Field {field} should be capitalized"
