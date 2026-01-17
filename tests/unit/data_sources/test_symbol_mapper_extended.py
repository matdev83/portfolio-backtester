import pytest
from portfolio_backtester.data_sources.symbol_mapper import (
    to_canonical_id,
    from_canonical_id,
    get_exchange_prefix,
    is_special_symbol,
    clear_cache,
    get_stooq_symbol
)

def setup_function():
    clear_cache()

def test_to_canonical_id_basic():
    assert to_canonical_id("AAPL") == "NASDAQ:AAPL"
    assert to_canonical_id("SPY") == "AMEX:SPY"
    assert to_canonical_id("IBM") == "NYSE:IBM" # Default

def test_to_canonical_id_special():
    assert to_canonical_id("^GSPC") == "SP:SPX"
    assert to_canonical_id("SPX") == "SP:SPX"
    assert to_canonical_id("^VIX") == "CBOE:VIX"
    assert to_canonical_id("^TNX") == "TVC:TNX"

def test_to_canonical_id_normalization():
    assert to_canonical_id("  aapl  ") == "NASDAQ:AAPL"
    assert to_canonical_id("NYSE:TSLA") == "NYSE:TSLA" # Pass-through

def test_from_canonical_id():
    assert from_canonical_id("NASDAQ:AAPL") == "AAPL"
    assert from_canonical_id("AMEX:SPY") == "SPY"
    assert from_canonical_id("SP:SPX") == "SPX" # Preferred local form
    assert from_canonical_id("UNKNOWN:FOO") == "FOO"

def test_get_exchange_prefix():
    assert get_exchange_prefix("AAPL") == "NASDAQ"
    assert get_exchange_prefix("SPY") == "AMEX"
    assert get_exchange_prefix("MSFT") == "NASDAQ"
    assert get_exchange_prefix("IBM") == "NYSE"
    assert get_exchange_prefix("^GSPC") == "SP"

def test_is_special_symbol():
    assert is_special_symbol("^GSPC") is True
    assert is_special_symbol("AAPL") is False
    assert is_special_symbol("^VIX") is True

def test_get_stooq_symbol():
    assert get_stooq_symbol("AAPL") == "aapl.us"
    assert get_stooq_symbol("SPY") == "spy.us"
    assert get_stooq_symbol("^GSPC") == "^spx"
    assert get_stooq_symbol("^DJI") == "dji" # Logic: ticker_upper[1:].lower()
