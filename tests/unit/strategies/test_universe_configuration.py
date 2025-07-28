"""Unit tests for universe configuration in strategies."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.portfolio_backtester.strategies.base_strategy import BaseStrategy
from src.portfolio_backtester.universe_loader import UniverseLoaderError, clear_universe_cache


class MockStrategy(BaseStrategy):
    """Test strategy for universe configuration testing."""
    
    def generate_signals(self, all_historical_data, benchmark_historical_data, current_date, start_date=None, end_date=None):
        """Dummy implementation for testing."""
        return pd.DataFrame()


class TestUniverseConfiguration:
    """Test universe configuration in BaseStrategy."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch('src.portfolio_backtester.universe_loader.UNIVERSES_DIR', self.temp_dir)
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
    
    def test_default_universe_fallback(self):
        """Test fallback to global config universe when no universe_config."""
        global_config = {"universe": ["AAPL", "MSFT", "GOOGL"]}
        strategy_config = {}
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe == expected
    
    def test_fixed_universe_configuration(self):
        """Test fixed universe configuration."""
        global_config = {"universe": ["DEFAULT1", "DEFAULT2"]}
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"]
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("AMZN", 1.0)]
        assert universe == expected
    
    def test_fixed_universe_case_normalization(self):
        """Test that fixed universe tickers are normalized to uppercase."""
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": ["aapl", "Msft", "GOOGL"]
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe == expected
    
    def test_fixed_universe_validation_errors(self):
        """Test validation errors for fixed universe configuration."""
        global_config = {"universe": ["DEFAULT1"]}
        
        # Test missing tickers
        strategy_config = {
            "universe_config": {
                "type": "fixed"
                # Missing tickers
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test empty tickers list
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": []
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test non-list tickers
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": "AAPL,MSFT"  # String instead of list
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test non-string ticker
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": ["AAPL", 123, "MSFT"]  # Number in list
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
    
    def test_named_universe_configuration(self):
        """Test named universe configuration."""
        # Create test universe file
        self.create_test_universe("tech_stocks", "AAPL\nMSFT\nGOOGL\nNVDA\n")
        
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "tech_stocks"
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("NVDA", 1.0)]
        assert universe == expected
    
    def test_multiple_named_universes_configuration(self):
        """Test multiple named universes configuration."""
        # Create test universe files
        self.create_test_universe("tech", "AAPL\nMSFT\n")
        self.create_test_universe("finance", "JPM\nBAC\n")
        
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_names": ["tech", "finance"]
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("JPM", 1.0), ("BAC", 1.0)]
        assert universe == expected
    
    def test_named_universe_validation_errors(self):
        """Test validation errors for named universe configuration."""
        global_config = {"universe": ["DEFAULT1"]}
        
        # Test both universe_name and universe_names specified
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "tech",
                "universe_names": ["finance"]
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test neither universe_name nor universe_names specified
        strategy_config = {
            "universe_config": {
                "type": "named"
                # Missing both universe_name and universe_names
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test nonexistent universe
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "nonexistent_universe"
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test empty universe_names list
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_names": []
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
        
        # Test non-list universe_names
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_names": "tech,finance"  # String instead of list
            }
        }
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
    
    @patch('src.portfolio_backtester.universe.get_top_weight_sp500_components')
    def test_method_universe_configuration(self, mock_sp500_func):
        """Test method-based universe configuration."""
        # Mock the S&P 500 function
        mock_sp500_func.return_value = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 5,
                "exact": False
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("AMZN", 1.0), ("NVDA", 1.0)]
        assert universe == expected
        
        # Verify the function was called with correct parameters
        mock_sp500_func.assert_called_once()
        call_args = mock_sp500_func.call_args
        assert call_args[1]['n'] == 5
        assert call_args[1]['exact'] is False
    
    @patch('src.portfolio_backtester.universe.get_top_weight_sp500_components')
    def test_method_universe_default_parameters(self, mock_sp500_func):
        """Test method-based universe with default parameters."""
        mock_sp500_func.return_value = ["AAPL", "MSFT"]
        
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "method"
                # Using defaults: method_name="get_top_weight_sp500_components", n_holdings=50, exact=False
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0)]
        assert universe == expected
        
        # Verify default parameters were used
        call_args = mock_sp500_func.call_args
        assert call_args[1]['n'] == 50  # Default
        assert call_args[1]['exact'] is False  # Default
    
    def test_method_universe_unknown_method(self):
        """Test method-based universe with unknown method."""
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "method",
                "method_name": "unknown_method"
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
    
    def test_unknown_universe_type(self):
        """Test unknown universe type."""
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "unknown_type"
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
    
    def test_missing_universe_type(self):
        """Test missing universe type."""
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                # Missing type
                "tickers": ["AAPL", "MSFT"]
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        # Should fallback to global config
        assert universe == [("DEFAULT1", 1.0)]
    
    @patch('src.portfolio_backtester.universe.get_top_weight_sp500_components')
    def test_get_universe_method_with_date(self, mock_sp500_func):
        """Test get_universe_method_with_date functionality."""
        mock_sp500_func.return_value = ["AAPL", "MSFT", "GOOGL"]
        
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 3
            }
        }
        
        strategy = MockStrategy(strategy_config)
        current_date = pd.Timestamp("2024-01-15")
        universe = strategy.get_universe_method_with_date(global_config, current_date)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe == expected
        
        # Verify the function was called with the specific date
        call_args = mock_sp500_func.call_args
        assert call_args[1]['date'] == current_date
    
    def test_get_universe_method_with_date_non_method_type(self):
        """Test get_universe_method_with_date with non-method universe types."""
        # Create test universe file
        self.create_test_universe("test", "AAPL\nMSFT\n")
        
        global_config = {"universe": ["DEFAULT1"]}
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "test"
            }
        }
        
        strategy = MockStrategy(strategy_config)
        current_date = pd.Timestamp("2024-01-15")
        universe = strategy.get_universe_method_with_date(global_config, current_date)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0)]
        assert universe == expected


class TestBackwardCompatibility:
    """Test backward compatibility with existing strategy configurations."""
    
    def test_no_universe_config_uses_global(self):
        """Test that strategies without universe_config use global universe."""
        global_config = {"universe": ["AAPL", "MSFT", "GOOGL"]}
        strategy_config = {}  # No universe_config
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe == expected
    
    def test_empty_global_universe(self):
        """Test behavior with empty global universe."""
        global_config = {"universe": []}
        strategy_config = {}
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        assert universe == []
    
    def test_missing_global_universe(self):
        """Test behavior when global config has no universe key."""
        global_config = {}  # No universe key
        strategy_config = {}
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        assert universe == []


class TestErrorHandlingAndLogging:
    """Test error handling and logging behavior."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch('src.portfolio_backtester.universe_loader.UNIVERSES_DIR', self.temp_dir)
        self.universes_dir_patcher.start()
        clear_universe_cache()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.universes_dir_patcher.stop()
        shutil.rmtree(self.temp_dir)
        clear_universe_cache()
    
    @patch('src.portfolio_backtester.strategies.base_strategy.logger')
    def test_error_logging_and_fallback(self, mock_logger):
        """Test that errors are logged and fallback works."""
        global_config = {"universe": ["FALLBACK1", "FALLBACK2"]}
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "nonexistent_universe"
            }
        }
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        # Should fallback to global config
        expected = [("FALLBACK1", 1.0), ("FALLBACK2", 1.0)]
        assert universe == expected
        
        # Should have logged an error
        mock_logger.error.assert_called_once()
        mock_logger.info.assert_called_once_with("Falling back to global universe")
    
    @patch('src.portfolio_backtester.strategies.base_strategy.logger')
    def test_universe_loader_error_handling(self, mock_logger):
        """Test handling of UniverseLoaderError."""
        global_config = {"universe": ["FALLBACK"]}
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "test"
            }
        }
        
        # Create invalid universe file
        universe_file = self.temp_dir / "test.txt"
        universe_file.write_text("AAPL\nINVALID@TICKER\nMSFT\n")
        
        strategy = MockStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        # Should fallback to global config
        assert universe == [("FALLBACK", 1.0)]
        
        # Should have logged error and fallback message
        mock_logger.error.assert_called_once()
        mock_logger.info.assert_called_once_with("Falling back to global universe")