"""Tests for SPY holdings ticker aliases."""

from portfolio_backtester.universe import normalize_spy_holding_ticker


def test_normalize_spy_holding_ticker_maps_legacy_symbols() -> None:
    assert normalize_spy_holding_ticker("brkb") == "BRK.B"
    assert normalize_spy_holding_ticker("VISA") == "V"
    assert normalize_spy_holding_ticker("SLBA") == "SLB"
    assert normalize_spy_holding_ticker("gec") == "GE"


def test_normalize_spy_holding_ticker_passes_through_unknown() -> None:
    assert normalize_spy_holding_ticker("MSFT") == "MSFT"
