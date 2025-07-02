import os
import pytest

from portfolio_backtester.cusip_mapping import CusipMappingDB, OPENFIGI_API_KEY

# Skip the test automatically if no API key present (e.g. in CI without secrets)
pytestmark = [pytest.mark.network, pytest.mark.skipif(not OPENFIGI_API_KEY, reason="OpenFIGI key missing – live test skipped")]


def test_openfigi_resolves_aapl(tmp_path, monkeypatch):
    """Live integration test – resolves AAPL -> 037833100 using the real API."""
    db = CusipMappingDB()
    cusip, name = db.resolve("AAPL", throttle=0)  # no sleep for single call
    assert cusip == "037833100"
    assert "APPLE" in name.upper() or name == "BBG000B9XRY4" 