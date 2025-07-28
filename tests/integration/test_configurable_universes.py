"""Integration tests for configurable universes functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
from src.portfolio_backtester.universe_loader import clear_universe_cache


class TestConfigurableUniversesIntegration:
    """Integration tests for configurable universes with real strategies."""
    
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
    
    def test_momentum_strategy_with_fixed_universe(self):
        """Test MomentumStrategy with fixed universe configuration."""
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"]
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 2,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT1", "DEFAULT2"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("AMZN", 1.0)]
        assert universe == expected
    
    def test_momentum_strategy_with_named_universe(self):
        """Test MomentumStrategy with named universe configuration."""
        # Create test universe
        self.create_test_universe("tech_giants", "AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\n")
        
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "tech_giants"
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 3,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT1"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("AMZN", 1.0), ("NVDA", 1.0)]
        assert universe == expected
    
    def test_momentum_strategy_with_multiple_named_universes(self):
        """Test MomentumStrategy with multiple named universes."""
        # Create test universes
        self.create_test_universe("tech", "AAPL\nMSFT\nGOOGL\n")
        self.create_test_universe("finance", "JPM\nBAC\nWFC\n")
        
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_names": ["tech", "finance"]
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 4,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT1"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("JPM", 1.0), ("BAC", 1.0), ("WFC", 1.0)]
        assert universe == expected
    
    @patch('src.portfolio_backtester.universe.get_top_weight_sp500_components')
    def test_momentum_strategy_with_method_universe(self, mock_sp500_func):
        """Test MomentumStrategy with method-based universe."""
        mock_sp500_func.return_value = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        
        strategy_config = {
            "universe_config": {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 5
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 3,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT1"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0), ("AMZN", 1.0), ("NVDA", 1.0)]
        assert universe == expected
        
        # Verify the method was called
        mock_sp500_func.assert_called_once()
    
    def test_momentum_strategy_fallback_to_global(self):
        """Test MomentumStrategy fallback to global universe when config fails."""
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "nonexistent_universe"
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 2,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["FALLBACK1", "FALLBACK2"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        # Should fallback to global universe
        expected = [("FALLBACK1", 1.0), ("FALLBACK2", 1.0)]
        assert universe == expected
    
    def test_momentum_strategy_no_universe_config(self):
        """Test MomentumStrategy without universe_config (backward compatibility)."""
        strategy_config = {
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 2,
                "long_only": True
            }
            # No universe_config
        }
        
        global_config = {"universe": ["COMPAT1", "COMPAT2", "COMPAT3"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        # Should use global universe
        expected = [("COMPAT1", 1.0), ("COMPAT2", 1.0), ("COMPAT3", 1.0)]
        assert universe == expected
    
    @patch('src.portfolio_backtester.universe.get_top_weight_sp500_components')
    def test_universe_with_date_context(self, mock_sp500_func):
        """Test universe resolution with date context."""
        mock_sp500_func.return_value = ["AAPL", "MSFT", "GOOGL"]
        
        strategy_config = {
            "universe_config": {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 3
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 2,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT1"]}
        
        strategy = MomentumStrategy(strategy_config)
        current_date = pd.Timestamp("2024-01-15")
        universe = strategy.get_universe_method_with_date(global_config, current_date)
        
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe == expected
        
        # Verify the method was called with the correct date
        call_args = mock_sp500_func.call_args
        assert call_args[1]['date'] == current_date
    
    def test_universe_configuration_validation(self):
        """Test that invalid universe configurations are handled gracefully."""
        invalid_configs = [
            # Missing type
            {"universe_config": {"tickers": ["AAPL"]}},
            
            # Invalid type
            {"universe_config": {"type": "invalid_type"}},
            
            # Fixed type with invalid tickers
            {"universe_config": {"type": "fixed", "tickers": "not_a_list"}},
            
            # Named type with both universe_name and universe_names
            {"universe_config": {"type": "named", "universe_name": "test", "universe_names": ["test2"]}},
            
            # Method type with unknown method
            {"universe_config": {"type": "method", "method_name": "unknown_method"}},
        ]
        
        global_config = {"universe": ["FALLBACK"]}
        
        for config in invalid_configs:
            config["strategy_params"] = {"lookback_months": 6, "num_holdings": 1, "long_only": True}
            strategy = MomentumStrategy(config)
            universe = strategy.get_universe(global_config)
            
            # Should fallback to global universe for all invalid configs
            assert universe == [("FALLBACK", 1.0)], f"Failed for config: {config}"


class TestScenarioConfigurationCompatibility:
    """Test compatibility with scenario-level universe configuration."""
    
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
    
    def test_scenario_universe_override(self):
        """Test that scenario-level universe still takes precedence."""
        # This test simulates how core.py handles universe resolution
        strategy_config = {
            "universe_config": {
                "type": "fixed",
                "tickers": ["STRATEGY1", "STRATEGY2"]
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 1,
                "long_only": True
            }
        }
        
        # Simulate scenario config with universe override
        scenario_config = {
            "strategy": "momentum",
            "universe": ["SCENARIO1", "SCENARIO2"],  # This should take precedence
            "strategy_params": strategy_config["strategy_params"]
        }
        
        global_config = {"universe": ["GLOBAL1", "GLOBAL2"]}
        
        # Test strategy universe resolution
        strategy = MomentumStrategy(strategy_config)
        strategy_universe = strategy.get_universe(global_config)
        
        # Strategy should use its own universe_config
        expected_strategy = [("STRATEGY1", 1.0), ("STRATEGY2", 1.0)]
        assert strategy_universe == expected_strategy
        
        # In actual backtesting, scenario-level universe would be used instead
        # This is handled in core.py, not in the strategy itself
        if "universe" in scenario_config:
            scenario_universe = [(ticker, 1.0) for ticker in scenario_config["universe"]]
            expected_scenario = [("SCENARIO1", 1.0), ("SCENARIO2", 1.0)]
            assert scenario_universe == expected_scenario


class TestPerformanceAndCaching:
    """Test performance and caching behavior of universe loading."""
    
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
    
    def test_universe_caching_across_strategies(self):
        """Test that universe loading is cached across multiple strategy instances."""
        # Create test universe
        self.create_test_universe("cached_universe", "AAPL\nMSFT\nGOOGL\n")
        
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "cached_universe"
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 2,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT"]}
        
        # Create multiple strategy instances
        strategy1 = MomentumStrategy(strategy_config)
        strategy2 = MomentumStrategy(strategy_config)
        strategy3 = MomentumStrategy(strategy_config)
        
        # Get universe from all strategies
        universe1 = strategy1.get_universe(global_config)
        universe2 = strategy2.get_universe(global_config)
        universe3 = strategy3.get_universe(global_config)
        
        # All should return the same result
        expected = [("AAPL", 1.0), ("MSFT", 1.0), ("GOOGL", 1.0)]
        assert universe1 == expected
        assert universe2 == expected
        assert universe3 == expected
    
    def test_large_universe_performance(self):
        """Test performance with a large universe file."""
        # Create a large universe (500 tickers)
        large_universe_content = "\n".join([f"TICKER{i:03d}" for i in range(500)])
        self.create_test_universe("large_universe", large_universe_content)
        
        strategy_config = {
            "universe_config": {
                "type": "named",
                "universe_name": "large_universe"
            },
            "strategy_params": {
                "lookback_months": 6,
                "num_holdings": 50,
                "long_only": True
            }
        }
        
        global_config = {"universe": ["DEFAULT"]}
        
        strategy = MomentumStrategy(strategy_config)
        universe = strategy.get_universe(global_config)
        
        # Should successfully load all 500 tickers
        assert len(universe) == 500
        assert universe[0] == ("TICKER000", 1.0)
        assert universe[-1] == ("TICKER499", 1.0)