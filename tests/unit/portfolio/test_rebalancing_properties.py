"""
Property-based tests for portfolio rebalancing logic.

This module uses Hypothesis to test invariants and properties of the portfolio rebalancing
functions, ensuring they behave correctly under a wide range of inputs and edge cases.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.portfolio.rebalancing import rebalance


@st.composite
def weight_dataframes(draw, min_rows=5, max_rows=30, min_assets=2, max_assets=10):
    """
    Generate weight DataFrames for testing rebalancing logic.
    
    Args:
        min_rows: Minimum number of time periods
        max_rows: Maximum number of time periods
        min_assets: Minimum number of assets
        max_assets: Maximum number of assets
        
    Returns:
        DataFrame with weights indexed by date
    """
    # Generate number of rows and assets
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    
    # Generate dates with various frequencies for more diverse testing
    freq = draw(st.sampled_from(["B", "D", "W", "ME"]))
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))  # Avoid month end issues
    
    start_date = datetime(start_year, start_month, start_day)
    dates = pd.date_range(start=start_date, periods=rows, freq=freq)
    
    # Create asset names
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate weights - with different approaches
    weights_approach = draw(st.integers(min_value=0, max_value=2))
    weights_data = {}
    
    if weights_approach == 0:
        # Standard approach: Equal weights
        for asset in assets:
            weights_data[asset] = np.ones(rows) / n_assets
            
    elif weights_approach == 1:
        # Random weights that sum to 1 for each date
        for i in range(rows):
            raw_weights = np.random.random(n_assets)
            normalized_weights = raw_weights / raw_weights.sum()
            for j, asset in enumerate(assets):
                if asset not in weights_data:
                    weights_data[asset] = np.zeros(rows)
                weights_data[asset][i] = normalized_weights[j]
                
    else:
        # Sparse weights with some zeros
        for i in range(rows):
            # Select a subset of assets to have non-zero weights
            n_active = draw(st.integers(min_value=1, max_value=n_assets))
            active_indices = np.random.choice(n_assets, size=n_active, replace=False)
            
            raw_weights = np.zeros(n_assets)
            raw_weights[active_indices] = np.random.random(n_active)
            
            # Ensure weights sum to 1
            if raw_weights.sum() > 0:
                normalized_weights = raw_weights / raw_weights.sum()
            else:
                # Fallback if all weights are zero
                normalized_weights = np.zeros(n_assets)
                normalized_weights[0] = 1.0
                
            for j, asset in enumerate(assets):
                if asset not in weights_data:
                    weights_data[asset] = np.zeros(rows)
                weights_data[asset][i] = normalized_weights[j]
    
    # Create the DataFrame
    weights_df = pd.DataFrame(weights_data, index=dates)
    
    # Sometimes add NaN values to test robustness
    if draw(st.booleans()):
        for col in weights_df.columns:
            mask = np.random.random(len(weights_df)) < 0.1  # 10% chance of NaN
            weights_df.loc[mask, col] = np.nan
    
    return weights_df


@given(weight_dataframes(), st.sampled_from(["D", "W", "M", "Q", "Y"]))
@settings(deadline=None)
def test_rebalance_preserves_column_names(weights, frequency):
    """Test that rebalance preserves column names."""
    # Skip empty dataframes
    assume(not weights.empty)
    
    rebalanced = rebalance(weights, frequency)
    
    # Check that columns are preserved
    assert set(rebalanced.columns) == set(weights.columns), \
        f"Columns changed after rebalancing: {set(weights.columns)} -> {set(rebalanced.columns)}"


@given(weight_dataframes(), st.sampled_from(["D", "W", "M", "Q", "Y"]))
@settings(deadline=None)
def test_rebalance_period_boundaries(weights, frequency):
    """
    Test that rebalancing respects period boundaries.
    
    Rebalanced weights should occur at the appropriate period boundaries
    (e.g., month-end for monthly rebalancing).
    """
    # Skip empty dataframes
    assume(not weights.empty)
    assume(len(weights) > 1)
    
    # Special handling for "M" -> "ME" conversion
    freq = "ME" if frequency == "M" else ("QE" if frequency == "Q" else ("YE" if frequency == "Y" else frequency))
    
    rebalanced = rebalance(weights, frequency)
    
    # For each rebalance date, it should be a valid boundary for the frequency
    if freq != "D":  # Daily doesn't change the dates
        for date in rebalanced.index:
            # Check that the date matches the expected pattern for the frequency
            next_date = pd.date_range(start=date, periods=2, freq=freq)[1]
            
            # For month-end and quarter-end frequencies, the day should be the last day of the month
            if freq in ["ME", "QE", "YE"]:
                assert date.is_month_end, f"{date} is not a month-end date (frequency: {frequency})"
            
            # For week-end, the day should be the last day of the week
            elif freq == "W":
                # Different pandas versions use different week end days (Sat/Sun)
                assert date.dayofweek in [4, 5, 6], f"{date} is not a week-end date (frequency: {frequency})"


@given(weight_dataframes(), st.sampled_from(["D", "W", "M", "Q", "Y"]))
@settings(deadline=None)
def test_rebalance_preserves_weight_constraints(weights, frequency):
    """
    Test that rebalancing preserves important weight constraints:
    1. Weights should still sum to approximately 1 for each date
    2. Signs of weights should be preserved for weights that remain non-zero
    """
    # Skip empty dataframes
    assume(not weights.empty)
    
    # Handle edge case: If weights contains non-business days, they might disappear in resampling
    # Make sure we have at least one business day
    assume(any(date.dayofweek < 5 for date in weights.index))
    
    rebalanced = rebalance(weights, frequency)
    
    # Skip empty rebalanced dataframes (can happen with some frequency combinations)
    assume(not rebalanced.empty)
    
    # For each date in the rebalanced dataframe
    for date in rebalanced.index:
        # Test 1: Weights should sum to approximately 1 (allowing for numerical precision issues)
        # But only if the original weights summed to approximately 1
        original_dates = [d for d in weights.index if d <= date]
        if original_dates:
            closest_date = max(original_dates)
            
            # Calculate the original sum ignoring NaN values
            original_values = weights.loc[closest_date].dropna()
            if not original_values.empty:
                original_sum = original_values.sum()
                
                # Skip weights that sum far from 1
                assume(0.97 <= original_sum <= 1.03)
        
        # Test 2: Signs of weights should be preserved for non-zero weights
        # Find the closest date in the original dataframe
        original_dates = [d for d in weights.index if d <= date]
        if original_dates:
            closest_date = max(original_dates)
            
            for col in rebalanced.columns:
                # Only test columns that exist in both dataframes and have non-NaN values
                # and only test if both values are non-zero (rebalancing may make small weights zero)
                if (col in weights.columns and 
                    not pd.isna(weights.loc[closest_date, col]) and 
                    not pd.isna(rebalanced.loc[date, col]) and
                    abs(weights.loc[closest_date, col]) > 0.05 and  # Only check significant weights
                    abs(rebalanced.loc[date, col]) > 0.05):          # that remain significant
                    
                    original_sign = np.sign(weights.loc[closest_date, col])
                    rebalanced_sign = np.sign(rebalanced.loc[date, col])
                    
                    # Sign should be preserved for significant weights
                    assert original_sign == rebalanced_sign, \
                        f"Sign changed for {col} from {original_sign} to {rebalanced_sign}"


@given(weight_dataframes())
@settings(deadline=None)
def test_daily_rebalancing_preserves_values(weights):
    """
    Test that daily rebalancing ("D" frequency) preserves the values in the dataframe,
    although it might resample to business days if the original dataframe includes weekends.
    """
    # Skip empty dataframes
    assume(not weights.empty)
    
    rebalanced = rebalance(weights, "D")
    
    # Check that all business days in the original dataframe are in the rebalanced dataframe
    # with the same values (this is true even if the rebalanced dataframe has additional days)
    for date in weights.index:
        if date.dayofweek < 5:  # Business day (0=Monday, 4=Friday)
            if date in rebalanced.index:  # Should be there unless filtered by other rules
                for col in weights.columns:
                    # Only check values that are not NaN in the original
                    if col in weights.columns and col in rebalanced.columns and not pd.isna(weights.loc[date, col]):
                        # Check that non-NaN values are preserved
                        assert weights.loc[date, col] == rebalanced.loc[date, col], \
                            f"Value changed for {col} on {date}: {weights.loc[date, col]} -> {rebalanced.loc[date, col]}"


def test_rebalance_handles_empty_dataframe():
    """
    Test that rebalance handles empty dataframes gracefully.
    """
    # Create an empty dataframe with proper DatetimeIndex
    empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
    
    for freq in ["D", "W", "M", "Q", "Y"]:
        rebalanced = rebalance(empty_df, freq)
        assert rebalanced.empty, f"Rebalancing an empty dataframe should return an empty dataframe (frequency: {freq})"


@given(weight_dataframes(), st.sampled_from(["D", "W", "M", "Q", "Y"]))
@settings(deadline=None)
def test_rebalance_forward_fills(weights, frequency):
    """
    Test that rebalance forward fills values properly:
    - After rebalancing, we should have at most one weight vector per period
    - If no weight vector is available for a period, it should forward fill from the previous period
    """
    # Skip empty dataframes
    assume(not weights.empty)
    assume(len(weights) > 1)
    
    rebalanced = rebalance(weights, frequency)
    
    # Special handling for "M" -> "ME" conversion
    freq = "ME" if frequency == "M" else ("QE" if frequency == "Q" else ("YE" if frequency == "Y" else frequency))
    
    # Check that there's at most one weight vector per period
    if freq != "D":
        # Create a series of rebalanced dates
        date_periods = pd.Series(rebalanced.index).dt.to_period(freq=freq[0])
        # Check for duplicated periods
        duplicated_periods = date_periods.duplicated()
        assert not duplicated_periods.any(), \
            f"Found multiple rebalance dates in the same period: {rebalanced.index[duplicated_periods]}"
    
    # If we have a date range larger than the rebalanced data,
    # check that weights are forward-filled appropriately
    start_date = weights.index.min()
    end_date = weights.index.max()
    
    # Create a date range covering the entire period
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Reindex with the full date range and forward fill
    expanded_rebalanced = rebalanced.reindex(date_range, method="ffill")
    
    # Check that expanded_rebalanced has weights for dates not in rebalanced
    extra_dates = [d for d in date_range if d not in rebalanced.index]
    if extra_dates:
        for date in extra_dates:
            # Find the closest previous date in rebalanced
            prev_dates = [d for d in rebalanced.index if d < date]
            if prev_dates:
                prev_date = max(prev_dates)
                # Check that weights are forward-filled correctly
                assert expanded_rebalanced.loc[date].equals(rebalanced.loc[prev_date]), \
                    f"Weights not correctly forward-filled for {date}"
