"""
Unit tests for the asset_replacement module.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from portfolio_backtester.monte_carlo.asset_replacement import AssetReplacementManager


class TestAssetReplacementManager:
    """Test suite for the AssetReplacementManager class."""

    def setup_method(self):
        """Set up the test environment."""
        self.config = {"replacement_percentage": 0.5}
        self.manager = AssetReplacementManager(self.config)
        self.universe = ["AAPL", "GOOG", "MSFT", "AMZN"]

    def test_select_assets_for_replacement(self):
        """Test the selection of assets for replacement."""
        selected_assets = self.manager.select_assets_for_replacement(self.universe, random_seed=42)
        assert isinstance(selected_assets, set)
        assert len(selected_assets) == 2
        assert all(asset in self.universe for asset in selected_assets)

    def test_get_replacement_statistics(self):
        """Test the calculation of replacement statistics."""
        self.manager.select_assets_for_replacement(self.universe, random_seed=42)
        self.manager.select_assets_for_replacement(self.universe, random_seed=43)
        stats = self.manager.get_replacement_statistics()
        assert stats["total_runs"] == 2
        assert stats["total_assets_replaced"] == 4
        assert stats["avg_replacement_percentage"] == 0.5
        assert stats["avg_assets_per_run"] == 2.0
        assert sum(stats["asset_replacement_counts"].values()) == 4
        assert len(stats["asset_replacement_counts"]) <= 4

    def test_clear_cache(self):
        """Test clearing the cache."""
        self.manager._asset_stats_cache["test"] = "data"
        self.manager.clear_cache()
        assert not self.manager._asset_stats_cache

    def test_reset_history(self):
        """Test resetting the history."""
        self.manager.select_assets_for_replacement(self.universe)
        self.manager.reset_history()
        assert not self.manager.replacement_history

    def test_get_replacement_info(self):
        """Test getting replacement info."""
        selected_assets = self.manager.select_assets_for_replacement(self.universe, random_seed=42)
        info = self.manager.get_replacement_info()
        assert info.selected_assets == selected_assets
