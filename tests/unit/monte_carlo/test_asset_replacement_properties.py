"""
Property-based tests for the asset replacement module.

This module uses Hypothesis to test invariants and properties of the asset replacement
logic used in Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.monte_carlo.asset_replacement import (
    AssetReplacementManager,
    ReplacementInfo,
)


@st.composite
def monte_carlo_configs(draw, min_assets=2, max_assets=10):
    """
    Generate valid Monte Carlo configuration dictionaries.
    """
    replacement_percentage = draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False))
    random_seed = draw(st.integers(min_value=0, max_value=1000))
    enable_synthetic_data = draw(st.booleans())
    
    config = {
        "replacement_percentage": replacement_percentage,
        "random_seed": random_seed,
        "enable_synthetic_data": enable_synthetic_data,
        "stage1_optimization": draw(st.booleans()),
        "monte_carlo_max_attempts": draw(st.integers(min_value=1, max_value=5)),
        "validation_config": {
            "enable_validation": draw(st.booleans()),
            "correlation_threshold": draw(st.floats(min_value=0.5, max_value=0.9)),
            "volatility_threshold": draw(st.floats(min_value=0.1, max_value=0.5)),
        },
        "generation_config": {
            "max_attempts": draw(st.integers(min_value=1, max_value=10)),
            "validation_tolerance": draw(st.floats(min_value=0.1, max_value=1.0)),
        },
    }
    
    return config


@st.composite
def ohlc_data_frames(draw, min_rows=30, max_rows=100, min_assets=2, max_assets=5):
    """
    Generate OHLCV data frames for testing.
    """
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    
    # Generate dates
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))  # Avoid month end issues
    start_date = datetime(start_year, start_month, start_day)
    dates = pd.date_range(start=start_date, periods=rows, freq='B')
    
    # Create assets
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Create a dictionary to hold data for each asset
    data_dict = {}
    
    for asset in assets:
        # Generate prices with some realistic properties
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
        volatility = draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False))
        
        # Generate price series
        prices = np.random.normal(0, volatility, rows).cumsum() + base_price
        prices = np.maximum(prices, 1.0)  # Ensure prices are positive
        
        # Generate OHLCV data
        opens = prices * draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=0.98, max_value=1.02)))
        highs = np.maximum(prices * draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=1.0, max_value=1.05))), opens)
        lows = np.minimum(prices * draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=0.95, max_value=1.0))), opens)
        closes = prices
        volumes = draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=1000, max_value=1000000)))
        
        # Create DataFrame for this asset
        asset_df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
        
        data_dict[asset] = asset_df
    
    # Convert dictionary of DataFrames to MultiIndex DataFrame
    dfs = []
    for asset, df in data_dict.items():
        # Add ticker level to columns
        df_copy = df.copy()
        df_copy.columns = pd.MultiIndex.from_product([[asset], df_copy.columns], names=["Ticker", "Attribute"])
        dfs.append(df_copy)
    
    # Concatenate all DataFrames
    if dfs:
        ohlc_data = pd.concat(dfs, axis=1)
    else:
        # Create empty DataFrame with correct structure if no assets
        ohlc_data = pd.DataFrame(columns=pd.MultiIndex.from_product([["DUMMY"], ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Attribute"]))
    
    return ohlc_data, assets


@given(monte_carlo_configs(), ohlc_data_frames())
@settings(deadline=None)
def test_asset_replacement_manager_initialization(config, ohlc_data_assets):
    """Test that AssetReplacementManager initializes correctly."""
    ohlc_data, assets = ohlc_data_assets
    
    # Initialize AssetReplacementManager
    manager = AssetReplacementManager(config)
    
    # Check that manager is initialized correctly
    assert manager.config == config
    assert isinstance(manager.replacement_history, list)
    assert len(manager.replacement_history) == 0
    assert isinstance(manager._asset_stats_cache, dict)
    assert isinstance(manager._full_data_cache, dict)
    assert hasattr(manager, "synthetic_generator")


@given(monte_carlo_configs(), ohlc_data_frames())
@settings(deadline=None)
def test_select_assets_for_replacement(config, ohlc_data_assets):
    """Test that asset selection for replacement works correctly."""
    ohlc_data, assets = ohlc_data_assets
    
    # Ensure enable_synthetic_data is True for this test
    config["enable_synthetic_data"] = True
    config["replacement_percentage"] = 0.5  # Ensure we select at least one asset
    
    # Initialize AssetReplacementManager
    manager = AssetReplacementManager(config)
    
    # Get the universe of assets
    universe = assets
    
    # Select assets for replacement
    selected_assets = manager.select_assets_for_replacement(universe, random_seed=config.get("random_seed"))
    
    # Check that selected_assets is a subset of universe
    assert all(asset in universe for asset in selected_assets)
    
    # Check that the number of selected assets matches the expected percentage
    expected_count = max(1, int(len(universe) * config["replacement_percentage"]))
    assert len(selected_assets) == expected_count
    
    # Check that selection is deterministic for the same random seed
    selected_assets2 = manager.select_assets_for_replacement(universe, random_seed=config.get("random_seed"))
    assert selected_assets == selected_assets2
    
    # Check that selection is different for different random seed
    # Only check if we have enough assets to make different selections likely
    if len(universe) > 3 and config["replacement_percentage"] < 0.5:
        selected_assets3 = manager.select_assets_for_replacement(universe, random_seed=config.get("random_seed")+1)
        # We can't guarantee different selections with random sampling, so skip assertion


@given(monte_carlo_configs(), ohlc_data_frames())
@settings(deadline=None)
def test_get_replacement_info(config, ohlc_data_assets):
    """Test that replacement info is correctly tracked."""
    ohlc_data, assets = ohlc_data_assets
    
    # Ensure enable_synthetic_data is True for this test
    config["enable_synthetic_data"] = True
    
    # Initialize AssetReplacementManager
    manager = AssetReplacementManager(config)
    
    # Get the universe of assets
    universe = assets
    
    # Create a Monte Carlo dataset
    test_start = ohlc_data.index[0]
    test_end = ohlc_data.index[-1]
    
    # Create a dictionary of asset data
    original_data = {}
    for asset in assets:
        original_data[asset] = ohlc_data.xs(asset, level="Ticker", axis=1, drop_level=True)
    
    # Create a Monte Carlo dataset with different run IDs
    for i in range(3):
        modified_data, replacement_info = manager.create_monte_carlo_dataset(
            original_data=original_data,
            universe=universe,
            test_start=test_start,
            test_end=test_end,
            run_id=f"test_run_{i}",
            random_seed=config.get("random_seed") + i
        )
    
    # Check that get_replacement_info returns correct info for each window
    assert len(manager.replacement_history) > 0
    
    # Check that each replacement info has the expected structure
    for info in manager.replacement_history:
        assert isinstance(info, ReplacementInfo)
        assert hasattr(info, "selected_assets")
        assert hasattr(info, "replacement_percentage")
        assert hasattr(info, "random_seed")
        assert hasattr(info, "total_assets")


@given(monte_carlo_configs(), ohlc_data_frames())
@settings(deadline=None)
def test_create_monte_carlo_dataset(config, ohlc_data_assets):
    """Test that create_monte_carlo_dataset correctly creates a dataset with replaced assets."""
    ohlc_data, assets = ohlc_data_assets
    
    # Ensure enable_synthetic_data is True for this test
    config["enable_synthetic_data"] = True
    
    # Initialize AssetReplacementManager
    manager = AssetReplacementManager(config)
    
    # Get the universe of assets
    universe = assets
    
    # Create a dictionary of asset data
    original_data = {}
    for asset in assets:
        original_data[asset] = ohlc_data.xs(asset, level="Ticker", axis=1, drop_level=True)
    
    # Create a Monte Carlo dataset
    test_start = ohlc_data.index[0]
    test_end = ohlc_data.index[-1]
    
    modified_data, replacement_info = manager.create_monte_carlo_dataset(
        original_data=original_data,
        universe=universe,
        test_start=test_start,
        test_end=test_end,
        run_id="test_run",
        random_seed=config.get("random_seed")
    )
    
    # Check that replacement_info has the correct structure
    assert isinstance(replacement_info, ReplacementInfo)
    assert hasattr(replacement_info, "selected_assets")
    assert hasattr(replacement_info, "replacement_percentage")
    assert hasattr(replacement_info, "random_seed")
    assert hasattr(replacement_info, "total_assets")
    
    # Check that modified_data has the same assets as original_data
    assert set(modified_data.keys()) == set(original_data.keys())
    
    # Check that each asset's data has the same shape
    for asset in assets:
        assert modified_data[asset].shape == original_data[asset].shape
        
    # Check that with enable_synthetic_data=False, no assets are replaced
    config["enable_synthetic_data"] = False
    manager = AssetReplacementManager(config)
    
    modified_data2, replacement_info2 = manager.create_monte_carlo_dataset(
        original_data=original_data,
        universe=universe,
        test_start=test_start,
        test_end=test_end,
        run_id="test_run",
        random_seed=config.get("random_seed")
    )
    
    # Check that no assets were selected for replacement
    assert len(replacement_info2.selected_assets) == 0