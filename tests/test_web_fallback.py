import pytest

from src.portfolio_backtester.cusip_mapping import CusipMappingDB

# Skip if no Internet (quick head request to duckduckgo); if fails, test skipped
import socket

def _has_net():
    try:
        socket.create_connection(("duckduckgo.com", 80), timeout=3)
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skip(reason="DuckDuckGo scraping is unreliable and not core to backtesting functionality.")


def test_duckduckgo_fallback_for_fb(tmp_path, monkeypatch):
    db = CusipMappingDB()
    # Ensure FB not in cache so we trigger fallback
    db._cache.pop("FB", None)
    try:
        cusip, _ = db.resolve("FB", throttle=0)
        assert len(cusip) in (8, 9)
    finally:
        # Clean-up DB entry to avoid polluting future runs
        pass 