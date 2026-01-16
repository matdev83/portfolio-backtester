import pytest
import shutil
from pathlib import Path
from unittest.mock import patch
from portfolio_backtester.universe_loader import (
    load_named_universe,
    load_multiple_named_universes,
    list_available_universes,
    validate_universe_exists,
    UniverseLoaderError,
    clear_universe_cache,
    _validate_ticker
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def temp_universes_dir(tmp_path):
    """Create a temporary directory for universe files."""
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    return universes_dir

@pytest.fixture
def patch_universes_dir(temp_universes_dir):
    """Patch the UNIVERSES_DIR constant in the module."""
    with patch("portfolio_backtester.universe_loader.UNIVERSES_DIR", temp_universes_dir):
        # Clear cache before and after to ensure isolation
        clear_universe_cache()
        yield temp_universes_dir
        clear_universe_cache()

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_validate_ticker():
    assert _validate_ticker("AAPL") is True
    assert _validate_ticker("BRK.B") is True
    assert _validate_ticker("T") is True
    assert _validate_ticker("123") is True # Some tickers can be numeric
    
    assert _validate_ticker("") is False
    assert _validate_ticker("toolongtickername") is False # > 10 chars
    assert _validate_ticker("INVALID$") is False # Invalid char
    assert _validate_ticker("low") is False # Regex enforces uppercase A-Z0-9.- 

def test_load_named_universe_valid(patch_universes_dir):
    # Create a valid universe file
    u_file = patch_universes_dir / "test_univ.txt"
    u_file.write_text("AAPL\nMSFT\nGOOGL # Comment\n\nNVDA", encoding="utf-8")
    
    tickers = load_named_universe("test_univ")
    
    assert len(tickers) == 4
    assert tickers == ["AAPL", "MSFT", "GOOGL", "NVDA"]

def test_load_named_universe_duplicates(patch_universes_dir):
    u_file = patch_universes_dir / "dups.txt"
    u_file.write_text("AAPL\nAAPL\nMSFT", encoding="utf-8")
    
    tickers = load_named_universe("dups")
    
    assert len(tickers) == 2
    assert tickers == ["AAPL", "MSFT"] # Order preserved

def test_load_named_universe_not_found(patch_universes_dir):
    with pytest.raises(UniverseLoaderError, match="Universe file not found"):
        load_named_universe("non_existent")

def test_load_named_universe_invalid_format(patch_universes_dir):
    u_file = patch_universes_dir / "invalid.txt"
    u_file.write_text("AAPL\nINV@LID\nMSFT", encoding="utf-8")
    
    with pytest.raises(UniverseLoaderError, match="Invalid tickers found"):
        load_named_universe("invalid")

def test_load_multiple_named_universes(patch_universes_dir):
    (patch_universes_dir / "u1.txt").write_text("A\nB")
    (patch_universes_dir / "u2.txt").write_text("B\nC")
    
    tickers = load_multiple_named_universes(["u1", "u2"])
    
    # Union of A,B and B,C -> A,B,C
    assert len(tickers) == 3
    assert set(tickers) == {"A", "B", "C"}
    # Order should be preserved: A, B (from u1), C (from u2)
    assert tickers == ["A", "B", "C"]

def test_list_available_universes(patch_universes_dir):
    (patch_universes_dir / "alpha.txt").touch()
    (patch_universes_dir / "beta.txt").touch()
    (patch_universes_dir / "gamma.csv").touch() # Should be ignored
    
    available = list_available_universes()
    
    assert len(available) == 2
    assert "alpha" in available
    assert "beta" in available
    assert "gamma" not in available

def test_validate_universe_exists(patch_universes_dir):
    (patch_universes_dir / "exists.txt").touch()
    
    assert validate_universe_exists("exists") is True
    assert validate_universe_exists("non_exists") is False
    assert validate_universe_exists("") is False

def test_caching(patch_universes_dir):
    u_file = patch_universes_dir / "cache_test.txt"
    u_file.write_text("A", encoding="utf-8")
    
    # First load
    t1 = load_named_universe("cache_test")
    assert t1 == ["A"]
    
    # Modify file
    u_file.write_text("B", encoding="utf-8")
    
    # Second load (should be cached 'A')
    t2 = load_named_universe("cache_test")
    assert t2 == ["A"]
    
    # Clear cache
    clear_universe_cache()
    
    # Third load (should be 'B')
    t3 = load_named_universe("cache_test")
    assert t3 == ["B"]