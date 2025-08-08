"""Integration tests for universe configuration fixes."""

import pytest
import pandas as pd
from unittest.mock import patch

from tests.fixtures.universe_test_fixtures import create_universe_fixture
from portfolio_backtester.strategies.base.base_strategy import BaseStrategy
from portfolio_backtester.universe_loader import (
    load_named_universe,
    load_multiple_named_universes,
    list_available_universes,
    validate_universe_exists,
    get_universe_info,
    UniverseLoaderError,
)


class MockStrategy(BaseStrategy):
    """Test strategy for universe configuration testing."""

    def generate_signals(
        self,
        all_historical_data,
        benchmark_historical_data,
        current_date,
        start_date=None,
        end_date=None,
    ):
        """Dummy implementation for testing."""
        return pd.DataFrame()


class TestUniverseConfigurationIntegration:
    """Integration tests for universe configuration fixes."""

    def setup_method(self):
        """Set up test environment."""
        self.fixture = create_universe_fixture()
        self.fixture.setup()

    def teardown_method(self):
        """Clean up test environment."""
        self.fixture.teardown()

    def test_end_to_end_universe_loading(self):
        """Test complete universe loading workflow."""
        # Create test universes
        self.fixture.create_standard_universes()

        # Test individual universe loading
        tech_tickers = load_named_universe("tech")
        assert tech_tickers == ["AAPL", "MSFT", "GOOGL", "NVDA"]

        finance_tickers = load_named_universe("finance")
        assert finance_tickers == ["JPM", "BAC", "WFC", "GS"]

        # Test multiple universe loading
        combined_tickers = load_multiple_named_universes(["tech", "finance"])
        expected_combined = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BAC", "WFC", "GS"]
        assert combined_tickers == expected_combined

        # Test universe listing
        available_universes = list_available_universes()
        assert set(available_universes) >= {"tech", "finance", "small", "formatted"}

        # Test universe validation
        assert validate_universe_exists("tech") is True
        assert validate_universe_exists("nonexistent") is False

        # Test universe info
        tech_info = get_universe_info("tech")
        assert tech_info["name"] == "tech"
        assert tech_info["ticker_count"] == 4
        assert tech_info["tickers"] == ["AAPL", "MSFT", "GOOGL", "NVDA"]
        assert tech_info["file_exists"] is True

    def test_strategy_universe_configuration_integration(self):
        """Test strategy universe configuration with all types."""
        # Create test universes
        self.fixture.create_standard_universes()

        global_config = {"universe": ["DEFAULT1", "DEFAULT2"]}

        # Test fixed universe configuration
        strategy_config = {
            "universe_config": {"type": "fixed", "tickers": ["AAPL", "MSFT", "GOOGL"]}
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe == expected

        # Test named universe configuration
        strategy_config = {"universe_config": {"type": "named", "universe_name": "tech"}}
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("NVDA", 1.0)]
        assert universe == expected

        # Test multiple named universes configuration
        strategy_config = {
            "universe_config": {"type": "named", "universe_names": ["tech", "finance"]}
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should contain all tickers from both universes
        universe_tickers = [ticker for ticker, weight in universe]
        assert "AAPL" in universe_tickers
        assert "JPM" in universe_tickers
        assert len(universe) == 8  # 4 tech + 4 finance

    def test_error_handling_and_fallback(self):
        """Test error handling and fallback behavior."""
        global_config = {"universe": ["FALLBACK1", "FALLBACK2"]}

        # Test fallback for nonexistent named universe
        strategy_config = {
            "universe_config": {"type": "named", "universe_name": "nonexistent_universe"}
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)

        # Should fallback to global config
        expected = [("FALLBACK1", 1.0), ("FALLBACK2", 1.0)]
        assert universe == expected

        # Test fallback for invalid universe configuration
        strategy_config = {"universe_config": {"type": "invalid_type"}}
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)

        # Should fallback to global config
        expected = [("FALLBACK1", 1.0), ("FALLBACK2", 1.0)]
        assert universe == expected

    def test_universe_file_parsing_edge_cases(self):
        """Test universe file parsing with various edge cases."""
        # Test universe with comments and empty lines
        formatted_content = """# This is a comment
AAPL
MSFT  # Inline comment

# Another comment
GOOGL
"""
        self.fixture.create_universe("edge_case", formatted_content)

        tickers = load_named_universe("edge_case")
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

        # Test universe with duplicates
        duplicate_content = "AAPL\nMSFT\nAAPL\nGOOGL\nMSFT\n"
        self.fixture.create_universe("duplicates", duplicate_content)

        tickers = load_named_universe("duplicates")
        assert tickers == ["AAPL", "MSFT", "GOOGL"]  # Duplicates removed, order preserved

        # Test universe with case normalization
        case_content = "aapl\nMsft\nGOOGL\n"
        self.fixture.create_universe("case_test", case_content)

        tickers = load_named_universe("case_test")
        assert tickers == ["AAPL", "MSFT", "GOOGL"]  # All normalized to uppercase

    def test_universe_caching_behavior(self):
        """Test that universe caching works correctly."""
        # Create test universe
        self.fixture.create_universe("cache_test", "AAPL\nMSFT\n")

        # Load universe multiple times
        tickers1 = load_named_universe("cache_test")
        tickers2 = load_named_universe("cache_test")
        tickers3 = load_named_universe("cache_test")

        # All should return the same result
        assert tickers1 == tickers2 == tickers3 == ["AAPL", "MSFT"]

        # Check cache statistics
        cache_info = load_named_universe.cache_info()
        assert cache_info.hits >= 2  # At least 2 cache hits

    @patch("portfolio_backtester.universe.get_top_weight_sp500_components")
    def test_method_based_universe_configuration(self, mock_sp500_func):
        """Test method-based universe configuration."""
        mock_sp500_func.return_value = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 5,
                "exact": False,
            }
        }

        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)

        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("AMZN", 1.0), ("NVDA", 1.0)]
        assert universe == expected

        # Verify the function was called with correct parameters
        mock_sp500_func.assert_called_once()
        call_args = mock_sp500_func.call_args
        assert call_args[1]["n"] == 5
        assert call_args[1]["exact"] is False

    def test_universe_validation_errors(self):
        """Test universe validation error handling."""
        # Test invalid ticker symbols
        invalid_content = "AAPL\nINVALID@TICKER\nMSFT\n"
        self.fixture.create_universe("invalid", invalid_content)

        with pytest.raises(UniverseLoaderError) as exc_info:
            load_named_universe("invalid")

        assert "Invalid tickers found" in str(exc_info.value)
        assert "INVALID@TICKER" in str(exc_info.value)

    def test_isolated_test_execution(self):
        """Test that tests are properly isolated and don't interfere with each other."""
        # This test verifies that the fixture setup/teardown works correctly
        # and tests don't interfere with each other

        # Create a universe in this test
        self.fixture.create_universe("isolation_test", "TEST1\nTEST2\n")

        # Verify it exists
        assert validate_universe_exists("isolation_test") is True
        tickers = load_named_universe("isolation_test")
        assert tickers == ["TEST1", "TEST2"]

        # The teardown_method should clean this up automatically
        # so it won't interfere with other tests
