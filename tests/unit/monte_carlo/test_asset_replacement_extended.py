import pytest
import pandas as pd
import numpy as np
from portfolio_backtester.monte_carlo.asset_replacement import AssetReplacementManager

@pytest.fixture
def config():
    return {
        "replacement_percentage": 0.5,
        "enable_synthetic_data": True,
        "min_historical_observations": 5
    }

def test_select_assets_for_replacement(config):
    mgr = AssetReplacementManager(config)
    universe = ["A", "B", "C", "D"]
    
    selected = mgr.select_assets_for_replacement(universe, random_seed=42)
    
    # 50% of 4 = 2
    assert len(selected) == 2
    assert selected.issubset(set(universe))

def test_replace_asset_data_test_phase(config):
    mgr = AssetReplacementManager(config)
    
    dates = pd.date_range("2023-01-01", periods=10)
    data = {
        "A": pd.DataFrame({
            "Open": np.ones(10), "High": np.ones(10), "Low": np.ones(10), "Close": np.ones(10)
        }, index=dates)
    }
    
    # Replace last 5 days
    start, end = dates[5], dates[9]
    
    # We need some historical data before start to avoid warnings/failures
    # Our data has 5 days before start.
    
    replaced = mgr.replace_asset_data(data, {"A"}, start, end, phase="test")
    
    # First 5 days should be original (all ones)
    assert (replaced["A"].iloc[:5]["Close"] == 1.0).all()
    # Last 5 days should be synthetic (not ones)
    assert not (replaced["A"].iloc[5:]["Close"] == 1.0).all()

def test_replace_asset_data_train_phase_skip(config):
    mgr = AssetReplacementManager(config)
    data = {"A": pd.DataFrame()}
    
    # Should return original data in train phase
    replaced = mgr.replace_asset_data(data, {"A"}, None, None, phase="train")
    assert replaced is data

def test_create_monte_carlo_dataset(config):
    mgr = AssetReplacementManager(config)
    universe = ["A"]
    dates = pd.date_range("2023-01-01", periods=10)
    data = {
        "A": pd.DataFrame({
            "Open": np.ones(10), "High": np.ones(10), "Low": np.ones(10), "Close": np.ones(10)
        }, index=dates)
    }
    
    mod_data, info = mgr.create_monte_carlo_dataset(
        data, universe, dates[5], dates[9], random_seed=42
    )
    
    assert "A" in info.selected_assets
    assert not (mod_data["A"].iloc[5:]["Close"] == 1.0).all()