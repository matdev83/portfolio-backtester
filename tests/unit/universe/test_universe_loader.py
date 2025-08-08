"""Unit tests for universe_loader module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from portfolio_backtester.universe_loader import (
    load_named_universe,
    load_multiple_named_universes,
    list_available_universes,
    validate_universe_exists,
    clear_universe_cache,
    get_universe_info,
    UniverseLoaderError,
    _validate_ticker,
    _parse_universe_file,
)


class TestTickerValidation:
    """Test ticker validation functionality."""

    def test_valid_tickers(self):
        """Test that valid tickers pass validation."""
        valid_tickers = [
            "AAPL",
            "MSFT",
            "BRK.B",
            "BF.B",
            "GOOGL",
            "META",
            "A",
            "AA",
            "AAA",
            "AAAA",
            "12345",
            "ABC123",
            "A-B",
            "A.B-C",
        ]

        for ticker in valid_tickers:
            assert _validate_ticker(ticker), f"Ticker '{ticker}' should be valid"

    def test_invalid_tickers(self):
        """Test that invalid tickers fail validation."""
        invalid_tickers = [
            "",  # Empty
            "a",  # Lowercase
            "aapl",  # Lowercase
            "AAPL ",  # Trailing space (should be stripped before validation)
            " AAPL",  # Leading space (should be stripped before validation)
            "AA PL",  # Space in middle
            "AAPL@",  # Invalid character
            "AAPL#",  # Invalid character
            "AAPL$",  # Invalid character
            "AAPL%",  # Invalid character
            "AAPL&",  # Invalid character
            "AAPL*",  # Invalid character
            "AAPL+",  # Invalid character
            "AAPL=",  # Invalid character
            "AAPL!",  # Invalid character
            "AAPL?",  # Invalid character
            "AAPL/",  # Invalid character
            "AAPL\\",  # Invalid character
            "AAPL|",  # Invalid character
            "AAPL<",  # Invalid character
            "AAPL>",  # Invalid character
            "AAPL[",  # Invalid character
            "AAPL]",  # Invalid character
            "AAPL{",  # Invalid character
            "AAPL}",  # Invalid character
            "AAPL(",  # Invalid character
            "AAPL)",  # Invalid character
            'AAPL"',  # Invalid character
            "AAPL'",  # Invalid character
            "AAPL`",  # Invalid character
            "AAPL~",  # Invalid character
            "AAPL^",  # Invalid character
            "A" * 11,  # Too long (11 characters)
        ]

        for ticker in invalid_tickers:
            assert not _validate_ticker(ticker), f"Ticker '{ticker}' should be invalid"


class TestParseUniverseFile:
    """Test universe file parsing functionality."""

    def test_parse_simple_file(self):
        """Test parsing a simple universe file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\nMSFT\nGOOGL\n")
            f.flush()

            file_path = Path(f.name)

        try:
            tickers = _parse_universe_file(file_path)
            assert tickers == ["AAPL", "MSFT", "GOOGL"]
        finally:
            file_path.unlink()

    def test_parse_file_with_comments(self):
        """Test parsing file with comments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# This is a comment\nAAPL\nMSFT  # Inline comment\n# Another comment\nGOOGL\n")
            f.flush()

            file_path = Path(f.name)

        try:
            tickers = _parse_universe_file(file_path)
            assert tickers == ["AAPL", "MSFT", "GOOGL"]
        finally:
            file_path.unlink()

    def test_parse_file_with_empty_lines(self):
        """Test parsing file with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\n\nMSFT\n\n\nGOOGL\n\n")
            f.flush()

            file_path = Path(f.name)

        try:
            tickers = _parse_universe_file(file_path)
            assert tickers == ["AAPL", "MSFT", "GOOGL"]
        finally:
            file_path.unlink()

    def test_parse_file_with_whitespace(self):
        """Test parsing file with various whitespace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  AAPL  \n\t MSFT \t\n   GOOGL   \n")
            f.flush()

            file_path = Path(f.name)

        try:
            tickers = _parse_universe_file(file_path)
            assert tickers == ["AAPL", "MSFT", "GOOGL"]
        finally:
            file_path.unlink()

    def test_parse_file_with_duplicates(self):
        """Test parsing file with duplicate tickers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\nMSFT\nAAPL\nGOOGL\nMSFT\n")
            f.flush()

            file_path = Path(f.name)

        try:
            tickers = _parse_universe_file(file_path)
            # Should preserve order and remove duplicates
            assert tickers == ["AAPL", "MSFT", "GOOGL"]
        finally:
            file_path.unlink()

    def test_parse_file_case_normalization(self):
        """Test that tickers are normalized to uppercase."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aapl\nMsft\nGOOGL\n")
            f.flush()

            file_path = Path(f.name)

        try:
            tickers = _parse_universe_file(file_path)
            assert tickers == ["AAPL", "MSFT", "GOOGL"]
        finally:
            file_path.unlink()

    def test_parse_file_with_invalid_tickers(self):
        """Test parsing file with invalid tickers raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\nINVALID@TICKER\nMSFT\n")
            f.flush()

            file_path = Path(f.name)

        try:
            with pytest.raises(UniverseLoaderError) as exc_info:
                _parse_universe_file(file_path)

            assert "Invalid tickers found" in str(exc_info.value)
            assert "Line 2: 'INVALID@TICKER'" in str(exc_info.value)
        finally:
            file_path.unlink()

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file raises error."""
        nonexistent_path = Path("/nonexistent/file.txt")

        with pytest.raises(UniverseLoaderError) as exc_info:
            _parse_universe_file(nonexistent_path)

        assert "Universe file not found" in str(exc_info.value)

    def test_parse_directory_instead_of_file(self):
        """Test parsing directory instead of file raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)

            with pytest.raises(UniverseLoaderError) as exc_info:
                _parse_universe_file(dir_path)

            assert "Universe path is not a file" in str(exc_info.value)


class TestLoadNamedUniverse:
    """Test named universe loading functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def create_test_universe(self, name: str, content: str):
        """Helper to create a test universe file."""
        universe_file = self.temp_dir / f"{name}.txt"
        universe_file.write_text(content)
        return universe_file

    def test_load_existing_universe(self):
        """Test loading an existing universe."""
        # Create test universe file
        self.create_test_universe("test_universe", "AAPL\nMSFT\nGOOGL\n")

        tickers = load_named_universe("test_universe")
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_load_nonexistent_universe(self):
        """Test loading nonexistent universe raises error."""
        with pytest.raises(UniverseLoaderError) as exc_info:
            load_named_universe("nonexistent_universe")

        assert "Universe file not found" in str(exc_info.value)

    def test_load_empty_universe_name(self):
        """Test loading universe with empty name raises error."""
        with pytest.raises(UniverseLoaderError) as exc_info:
            load_named_universe("")

        assert "Universe name cannot be empty" in str(exc_info.value)

    def test_load_universe_with_path_traversal(self):
        """Test that path traversal attempts are sanitized."""
        # Create test universe file
        self.create_test_universe("safe_universe", "AAPL\nMSFT\n")

        # Try to load with path traversal - should be sanitized
        tickers = load_named_universe("../safe_universe")
        assert tickers == ["AAPL", "MSFT"]

    def test_universe_caching(self):
        """Test that universe loading is cached."""
        # Create test universe file
        self.create_test_universe("cached_universe", "AAPL\nMSFT\n")

        # Load universe twice
        tickers1 = load_named_universe("cached_universe")
        tickers2 = load_named_universe("cached_universe")

        assert tickers1 == tickers2 == ["AAPL", "MSFT"]

        # Verify cache info shows hits
        cache_info = load_named_universe.cache_info()
        assert cache_info.hits >= 1


class TestLoadMultipleNamedUniverses:
    """Test loading multiple named universes."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def create_test_universe(self, name: str, content: str):
        """Helper to create a test universe file."""
        universe_file = self.temp_dir / f"{name}.txt"
        universe_file.write_text(content)
        return universe_file

    def test_load_multiple_universes(self):
        """Test loading multiple universes and getting their union."""
        # Create test universe files
        self.create_test_universe("universe1", "AAPL\nMSFT\n")
        self.create_test_universe("universe2", "GOOGL\nAMZN\n")

        tickers = load_multiple_named_universes(["universe1", "universe2"])
        assert set(tickers) == {"AAPL", "MSFT", "GOOGL", "AMZN"}

    def test_load_multiple_universes_with_overlap(self):
        """Test loading multiple universes with overlapping tickers."""
        # Create test universe files with overlap
        self.create_test_universe("universe1", "AAPL\nMSFT\nGOOGL\n")
        self.create_test_universe("universe2", "GOOGL\nAMZN\nNVDA\n")

        tickers = load_multiple_named_universes(["universe1", "universe2"])
        # Should remove duplicates while preserving order
        assert tickers == ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    def test_load_empty_universe_list(self):
        """Test loading empty list of universes returns empty list."""
        tickers = load_multiple_named_universes([])
        assert tickers == []


class TestListAvailableUniverses:
    """Test listing available universes."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def test_list_universes(self):
        """Test listing available universes."""
        # Create test universe files
        for i in range(1, 4):
            (self.temp_dir / f"universe{i}.txt").touch()

        universes = list_available_universes()
        assert universes == ["universe1", "universe2", "universe3"]

    def test_list_universes_nonexistent_dir(self):
        """Test listing universes when directory doesn't exist."""
        # Remove the temp directory to simulate nonexistent directory
        shutil.rmtree(self.temp_dir)

        universes = list_available_universes()
        assert universes == []


class TestValidateUniverseExists:
    """Test universe existence validation."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def test_validate_existing_universe(self):
        """Test validating existing universe."""
        # Create test universe file
        universe_file = self.temp_dir / "existing_universe.txt"
        universe_file.write_text("AAPL\n")

        assert validate_universe_exists("existing_universe") is True

    def test_validate_nonexistent_universe(self):
        """Test validating nonexistent universe."""
        assert validate_universe_exists("nonexistent_universe") is False

    def test_validate_empty_universe_name(self):
        """Test validating empty universe name."""
        assert validate_universe_exists("") is False


class TestGetUniverseInfo:
    """Test getting universe information."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def test_get_universe_info(self):
        """Test getting information about a universe."""
        # Create test universe file
        universe_file = self.temp_dir / "info_universe.txt"
        universe_file.write_text("AAPL\nMSFT\nGOOGL\n")

        info = get_universe_info("info_universe")

        assert info["name"] == "info_universe"
        assert info["ticker_count"] == 3
        assert info["tickers"] == ["AAPL", "MSFT", "GOOGL"]
        assert info["file_exists"] is True
        assert info["file_size_bytes"] > 0


class TestClearUniverseCache:
    """Test cache clearing functionality."""

    def test_clear_cache(self):
        """Test that cache clearing works."""
        # Clear cache and check that it doesn't raise an error
        clear_universe_cache()

        # Verify cache is cleared by checking cache info
        cache_info = load_named_universe.cache_info()
        assert cache_info.currsize == 0


class TestIntegrationWithFixtures:
    """Integration tests using test fixtures."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def test_load_test_fixture_universe(self):
        """Test loading universe from test fixtures."""
        # Create test fixture content in temp directory
        test_universe_content = """# Test universe for unit tests
AAPL
MSFT
GOOGL
# This is a comment
AMZN
NVDA

# Empty lines and comments should be ignored
TSLA"""

        universe_file = self.temp_dir / "test_universe.txt"
        universe_file.write_text(test_universe_content)

        tickers = load_named_universe("test_universe")
        expected_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
        assert tickers == expected_tickers

    def test_load_small_test_universe(self):
        """Test loading small test universe."""
        # Create small test fixture content in temp directory
        small_universe_content = "AAPL\nMSFT"

        universe_file = self.temp_dir / "test_universe_small.txt"
        universe_file.write_text(small_universe_content)

        tickers = load_named_universe("test_universe_small")
        expected_tickers = ["AAPL", "MSFT"]
        assert tickers == expected_tickers
