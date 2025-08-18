"""
Unit tests for the openfigi_lookup module.
"""
import pytest
from unittest.mock import Mock, patch
import requests

from portfolio_backtester.cusip_mapping_logic.openfigi_lookup import lookup_openfigi, _query_openfigi_api


class TestOpenFigiLookup:
    """Test suite for the openfigi_lookup module."""

    @patch("portfolio_backtester.cusip_mapping_logic.openfigi_lookup._query_openfigi_api")
    def test_lookup_openfigi_success(self, mock_query):
        """Test a successful CUSIP lookup."""
        mock_query.return_value = [
            {
                "data": [
                    {
                        "cusip": "037833100",
                        "securityDescription": "APPLE INC",
                    }
                ]
            }
        ]
        cusip, name = lookup_openfigi("test_api_key", "AAPL")
        assert cusip == "037833100"
        assert name == "APPLE INC"

    def test_lookup_openfigi_no_api_key(self):
        """Test CUSIP lookup with no API key."""
        cusip, name = lookup_openfigi("", "AAPL")
        assert cusip is None
        assert name is None

    @patch("portfolio_backtester.cusip_mapping_logic.openfigi_lookup._query_openfigi_api")
    def test_lookup_openfigi_not_found(self, mock_query):
        """Test CUSIP lookup for a ticker that is not found."""
        mock_query.return_value = [{"warning": "No matching results found"}]
        cusip, name = lookup_openfigi("test_api_key", "INVALID")
        assert cusip is None
        assert name is None

    @patch("portfolio_backtester.cusip_mapping_logic.openfigi_lookup.requests.post")
    def test_query_openfigi_api_request_exception(self, mock_post):
        """Test handling of API request exceptions."""
        mock_post.side_effect = requests.exceptions.RequestException
        result = _query_openfigi_api("test_api_key", "test_url", [])
        assert result is None



