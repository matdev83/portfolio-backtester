"""
Tests for the simplified synthetic data generator.
"""

import pandas as pd
import numpy as np
from src.portfolio_backtester.monte_carlo.synthetic_data_generator import (
    SyntheticDataGenerator, GARCHParameters, AssetStatistics
)

class TestSyntheticDataGenerator:
    """Test the simplified synthetic data generator."""

    def setup_method(self):
        """Set up test data and generator."""
        self.config = {
            'random_seed': 42
        }
        self.generator = SyntheticDataGenerator(self.config)
        self.ohlc_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=pd.date_range('2020-01-01', periods=5))

    def test_analyze_asset_statistics(self):
        """Test that asset statistics are calculated correctly."""
        stats = self.generator.analyze_asset_statistics(self.ohlc_data)
        assert isinstance(stats, AssetStatistics)
        assert isinstance(stats.garch_params, GARCHParameters)

    def test_generate_synthetic_returns(self):
        """Test that synthetic returns are generated correctly."""
        stats = self.generator.analyze_asset_statistics(self.ohlc_data)
        returns = self.generator.generate_synthetic_returns(stats, 100)
        assert isinstance(returns, np.ndarray)
        assert len(returns) == 100

    def test_generate_synthetic_prices(self):
        """Test that synthetic prices are generated correctly."""
        prices = self.generator.generate_synthetic_prices(self.ohlc_data, 100)
        assert isinstance(prices, pd.DataFrame)
        assert len(prices) == 100
        assert list(prices.columns) == ['Open', 'High', 'Low', 'Close']
