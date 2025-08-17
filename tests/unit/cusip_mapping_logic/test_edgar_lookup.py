"""
Unit tests for the edgar_lookup module.
"""
import pytest
from unittest.mock import Mock, patch
import requests

from portfolio_backtester.cusip_mapping_logic.edgar_lookup import (
    _company_tickers,
    _cik_for_ticker,
    lookup_edgar,
)


class TestEdgarLookup:
    """Test suite for the edgar_lookup module."""

    @patch("portfolio_backtester.cusip_mapping_logic.edgar_lookup.requests.get")
    def test_company_tickers_network_error(self, mock_get):
        """Test handling of network errors when fetching company tickers."""
        mock_get.side_effect = requests.exceptions.RequestException
        with pytest.raises(requests.exceptions.RequestException):
            _company_tickers()

    @patch(
        "portfolio_backtester.cusip_mapping_logic.edgar_lookup._company_tickers",
        return_value={"1": {"ticker": "AAPL", "cik": "320193"}},
    )
    def test_cik_for_ticker_from_json(self, mock_company_tickers):
        """Test successful CIK lookup from the company_tickers.json file."""
        cik = _cik_for_ticker("AAPL")
        assert cik == "0000320193"

    @patch("portfolio_backtester.cusip_mapping_logic.edgar_lookup._company_tickers", return_value={})
    @patch("portfolio_backtester.cusip_mapping_logic.edgar_lookup.requests.get")
    def test_cik_for_ticker_from_api(self, mock_get, mock_company_tickers):
        """Test successful CIK lookup from the EDGAR API."""
        mock_get.return_value.json.return_value = {"cik": "320193"}
        cik = _cik_for_ticker("AAPL")
        assert cik == "0000320193"

    @patch("portfolio_backtester.cusip_mapping_logic.edgar_lookup._cik_for_ticker", return_value=None)
    def test_lookup_edgar_no_cik(self, mock_cik_for_ticker):
        """Test EDGAR lookup when CIK is not found."""
        cusip, name = lookup_edgar("INVALID")
        assert cusip is None
        assert name is None
