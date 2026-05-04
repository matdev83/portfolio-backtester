import pytest
import pandas as pd
from unittest.mock import patch
from portfolio_backtester.data_sources.mdmp_data_source import (
    MarketDataMultiProviderDataSource,
    align_mdmp_results_to_requested,
    bare_ticker_from_canonical_id,
)


@pytest.fixture
def mock_mdmp_client():
    with patch("portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient") as mock:
        yield mock


def test_mdmp_data_source_init():
    # Test initialization with and without data_dir
    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"
    ) as mock_client:
        ds = MarketDataMultiProviderDataSource(data_dir="test_dir")
        assert ds.data_dir.name == "test_dir"
        mock_client.assert_called_once_with(data_dir="test_dir")


def test_mdmp_data_source_import_error():
    with patch("portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient", None):
        with pytest.raises(ImportError, match="market-data-multi-provider is not installed"):
            MarketDataMultiProviderDataSource()


def test_mdmp_get_data_deduplicates_canonical_aliases(mock_mdmp_client) -> None:
    """``AMEX:SPY`` and ``SPY`` must not double-count against MDMP fetch results."""
    client_instance = mock_mdmp_client.return_value
    spy_data = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )
    client_instance.fetch_many.return_value = {"AMEX:SPY": spy_data}

    def canon(ticker: str) -> str:
        t = ticker.strip().upper()
        if t in ("SPY", "AMEX:SPY"):
            return "AMEX:SPY"
        return ticker

    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.to_canonical_id",
        side_effect=canon,
    ):
        ds = MarketDataMultiProviderDataSource()
        result = ds.get_data(["AMEX:SPY", "SPY"], "2023-01-01", "2023-01-02")

    client_instance.fetch_many.assert_called_once()
    args, _kwargs = client_instance.fetch_many.call_args
    assert list(args[0]) == ["AMEX:SPY"]
    assert len(result.columns.get_level_values(0).unique()) == 1


def test_mdmp_get_data_success(mock_mdmp_client):
    # Setup mock client behavior
    client_instance = mock_mdmp_client.return_value

    # Mock data for SPY
    spy_data = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    # MDMP returns Dict[canonical_id, DataFrame]
    # to_canonical_id("SPY") -> "AMEX:SPY" (assumed mapping)
    client_instance.fetch_many.return_value = {"AMEX:SPY": spy_data}

    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.to_canonical_id",
        return_value="AMEX:SPY",
    ):
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
    raw_df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]}
    )

    normalized = ds._normalize_ohlcv(raw_df, "TEST")
    assert ("TEST", "Open") in normalized.columns
    assert ("TEST", "Close") in normalized.columns

    # Test with missing Close
    bad_df = pd.DataFrame({"Open": [1.0]})
    assert ds._normalize_ohlcv(bad_df, "TEST") is None


def test_mdmp_get_data_partial_failure(mock_mdmp_client):
    client_instance = mock_mdmp_client.return_value

    spy_data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2023-01-01"]))

    # SPY success, AAPL None
    client_instance.fetch_many.return_value = {"AMEX:SPY": spy_data, "NASDAQ:AAPL": None}

    def side_effect(ticker):
        return "AMEX:SPY" if ticker == "SPY" else "NASDAQ:AAPL"

    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.to_canonical_id",
        side_effect=side_effect,
    ):
        ds = MarketDataMultiProviderDataSource()
        result = ds.get_data(["SPY", "AAPL"], "2023-01-01", "2023-01-01")

    assert ("SPY", "Close") in result.columns
    assert ("AAPL", "Close") not in result.columns
    assert len(result) == 1


def test_bare_ticker_from_canonical_id() -> None:
    assert bare_ticker_from_canonical_id("NYSE:EWI") == "EWI"
    assert bare_ticker_from_canonical_id("AMEX:SPY") == "SPY"


def test_align_mdmp_results_exchange_alias() -> None:
    df = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2023-01-01"]))
    aligned = align_mdmp_results_to_requested(["AMEX:EWI"], {"NYSE:EWI": df})
    assert aligned["AMEX:EWI"] is df


def test_align_mdmp_second_request_same_bare_gets_none_after_pool_exhausted() -> None:
    """When two canonical ids share a bare symbol but MDMP returns one frame, only one request wins."""

    df = pd.DataFrame({"Close": [42.0]}, index=pd.to_datetime(["2023-01-01"]))
    aligned = align_mdmp_results_to_requested(["AMEX:DUPE", "NYSE:DUPE"], {"NYSE:DUPE": df})
    assert aligned["AMEX:DUPE"] is df
    assert aligned["NYSE:DUPE"] is None


def test_align_mdmp_exact_key_preferred_when_exact_and_bare_alias_exist() -> None:
    df_requested = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2023-01-01"]))
    df_other = pd.DataFrame({"Close": [99.0]}, index=pd.to_datetime(["2023-01-01"]))
    aligned = align_mdmp_results_to_requested(
        ["AMEX:ZZZ"],
        {"AMEX:ZZZ": df_requested, "NYSE:ZZZ": df_other},
    )
    assert aligned["AMEX:ZZZ"] is df_requested


def test_mdmp_get_data_exchange_key_alias(mock_mdmp_client) -> None:
    """MDMP may return on-disk keys (e.g. NYSE:*) while PB requested another canonical."""
    client_instance = mock_mdmp_client.return_value
    ewi = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [100],
        },
        index=pd.to_datetime(["2023-01-01"]),
    )
    client_instance.fetch_many.return_value = {"NYSE:EWI": ewi}

    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.to_canonical_id",
        return_value="AMEX:EWI",
    ):
        ds = MarketDataMultiProviderDataSource()
        result = ds.get_data(["EWI"], "2023-01-01", "2023-01-01")

    assert ("EWI", "Close") in result.columns
    assert len(result) == 1
