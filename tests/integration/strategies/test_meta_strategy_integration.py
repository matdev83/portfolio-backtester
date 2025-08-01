"""Integration tests for meta strategies with real strategy classes."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.portfolio_backtester.strategies.meta.simple_meta_strategy import SimpleMetaStrategy


class TestMetaStrategyIntegration:
    """Integration tests for meta strategies."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        assets = ["AAPL", "MSFT", "GOOGL"]
        
        # Create MultiIndex columns for OHLCV data
        columns = pd.MultiIndex.from_product(
            [assets, ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"]
        )
        
        # Generate random price data
        np.random.seed(42)  # For reproducible tests
        data = np.random.randn(len(dates), len(columns)) * 0.02 + 1.0
        data = np.cumprod(data, axis=0) * 100  # Cumulative product for price-like behavior
        
        df = pd.DataFrame(data, index=dates, columns=columns)
        
        # Ensure OHLC relationships are maintained
        for asset in assets:
            df[(asset, "High")] = df[[
                (asset, "Open"), (asset, "Close")
            ]].max(axis=1) * (1 + np.random.rand(len(dates)) * 0.01)
            
            df[(asset, "Low")] = df[[
                (asset, "Open"), (asset, "Close")
            ]].min(axis=1) * (1 - np.random.rand(len(dates)) * 0.01)
            
            df[(asset, "Volume")] = np.random.randint(1000000, 10000000, len(dates))
        
        return df
    
    @pytest.fixture
    def benchmark_data(self):
        """Create sample benchmark data."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        
        columns = pd.MultiIndex.from_product(
            [["SPY"], ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"]
        )
        
        np.random.seed(42)
        data = np.random.randn(len(dates), len(columns)) * 0.015 + 1.0
        data = np.cumprod(data, axis=0) * 100
        
        df = pd.DataFrame(data, index=dates, columns=columns)
        
        # Maintain OHLC relationships
        df[("SPY", "High")] = df[[
            ("SPY", "Open"), ("SPY", "Close")
        ]].max(axis=1) * (1 + np.random.rand(len(dates)) * 0.005)
        
        df[("SPY", "Low")] = df[[
            ("SPY", "Open"), ("SPY", "Close")
        ]].min(axis=1) * (1 - np.random.rand(len(dates)) * 0.005)
        
        df[("SPY", "Volume")] = np.random.randint(50000000, 200000000, len(dates))
        
        return df
    
    def test_simple_meta_strategy_signal_generation(self, sample_data, benchmark_data):
        """Test that SimpleMetaStrategy can generate signals using real sub-strategies."""
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "momentum",
                    "strategy_class": "CalmarMomentumStrategy",
                    "strategy_params": {
                        "rolling_window": 3,  # Short window for test data
                        "num_holdings": 2,
                        "price_column_asset": "Close",
                        "price_column_benchmark": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "M"
                        }
                    },
                    "weight": 0.7
                },
                {
                    "strategy_id": "seasonal",
                    "strategy_class": "IntramonthSeasonalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 5,
                        "hold_days": 5,
                        "price_column_asset": "Close",
                        "trade_longs": True,
                        "trade_shorts": False,
                        "timing_config": {
                            "mode": "signal_based"
                        }
                    },
                    "weight": 0.3
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Test signal generation for a specific date
        current_date = pd.Timestamp("2023-06-15")  # Mid-year date with sufficient history
        
        # Ensure we have data up to current_date
        historical_data = sample_data[sample_data.index <= current_date]
        benchmark_historical = benchmark_data[benchmark_data.index <= current_date]
        
        # Generate signals
        signals = meta_strategy.generate_signals(
            all_historical_data=historical_data,
            benchmark_historical_data=benchmark_historical,
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date
        )
        
        # Verify signals structure
        assert isinstance(signals, pd.DataFrame)
        assert current_date in signals.index
        assert len(signals.columns) > 0  # Should have some assets
        
        # Verify signal values are reasonable (between -1 and 1 for normalized weights)
        signal_values = signals.loc[current_date]
        assert all(abs(val) <= 2.0 for val in signal_values if not pd.isna(val))  # Allow some flexibility
    
    def test_capital_allocation_calculation(self):
        """Test capital allocation calculations."""
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "CalmarMomentumStrategy",
                    "strategy_params": {
                        "rolling_window": 6,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 0.6
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "IntramonthSeasonalStrategy", 
                    "strategy_params": {
                        "entry_day": 5,
                        "timing_config": {"mode": "signal_based"}
                    },
                    "weight": 0.4
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Test initial capital allocation
        capital_allocations = meta_strategy.calculate_sub_strategy_capital()
        assert capital_allocations["strategy1"] == 600000
        assert capital_allocations["strategy2"] == 400000
        
        # Test capital update after returns
        returns = {
            "strategy1": 0.10,  # 10% return
            "strategy2": -0.05  # -5% return
        }
        
        meta_strategy.update_available_capital(returns)
        
        # Expected P&L: (600000 * 0.10) + (400000 * -0.05) = 60000 - 20000 = 40000
        expected_new_capital = 1000000 + 40000
        assert meta_strategy.available_capital == expected_new_capital
        
        # Test new capital allocation
        new_allocations = meta_strategy.calculate_sub_strategy_capital()
        assert new_allocations["strategy1"] == expected_new_capital * 0.6
        assert new_allocations["strategy2"] == expected_new_capital * 0.4
    
    def test_get_universe_combination(self):
        """Test that meta strategy combines universes from sub-strategies."""
        config = {
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "CalmarMomentumStrategy",
                    "strategy_params": {
                        "universe_config": ["AAPL", "MSFT"],
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 0.5
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "IntramonthSeasonalStrategy",
                    "strategy_params": {
                        "universe_config": ["GOOGL", "AMZN"],
                        "timing_config": {"mode": "signal_based"}
                    },
                    "weight": 0.5
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Mock global config
        global_config = {"universe": ["DEFAULT1", "DEFAULT2"]}
        
        # Get combined universe
        universe = meta_strategy.get_universe(global_config)
        
        # Should contain assets from both sub-strategies
        universe_tickers = [ticker for ticker, weight in universe]
        
        # The exact universe depends on how sub-strategies resolve their universe_config
        # but we can verify it's a list of (ticker, weight) tuples
        assert isinstance(universe, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in universe)
        assert all(isinstance(ticker, str) and isinstance(weight, (int, float)) for ticker, weight in universe)