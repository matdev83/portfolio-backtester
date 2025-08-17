"""
Property-based tests for portfolio utilities.

This module uses Hypothesis to test invariants and properties of the portfolio utility
functions in the utils/portfolio_utils.py module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from hypothesis import given, settings, strategies as st, assume, example
from hypothesis.extra import numpy as hnp

from portfolio_backtester.utils.portfolio_utils import default_candidate_weights
from portfolio_backtester.portfolio.position_sizer import _normalize_weights

from tests.strategies import weights_and_leverage


@st.composite
def score_series_and_params(draw):
    """Generate score series and parameters for default_candidate_weights."""
    n_assets = draw(st.integers(min_value=3, max_value=20))
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate scores - both positive and negative
    score_elements = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    scores_array = draw(hnp.arrays(dtype=float, shape=n_assets, elements=score_elements))
    
    # Ensure at least one positive score for trade_shorts=False cases
    if draw(st.booleans()):
        scores_array[0] = abs(scores_array[0]) + 0.1  # Ensure positive
    
    scores = pd.Series(scores_array, index=assets)
    
    # Generate parameters
    num_holdings = draw(st.one_of(
        st.none(),
        st.integers(min_value=1, max_value=n_assets)
    ))
    top_decile_fraction = draw(st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False))
    trade_shorts = draw(st.booleans())
    
    params = {
        "num_holdings": num_holdings,
        "top_decile_fraction": top_decile_fraction,
        "trade_shorts": trade_shorts
    }
    
    return scores, params


@given(score_series_and_params())
@settings(deadline=None)
def test_default_candidate_weights_properties(data):
    """Test properties of default_candidate_weights."""
    scores, params = data
    
    # Calculate candidate weights
    weights = default_candidate_weights(scores, params)
    
    # Check that weights have the same index as scores
    assert weights.index.equals(scores.index)
    
    # Check that weights are finite
    assert np.all(np.isfinite(weights))
    
    # Count actual non-zero weights
    non_zero_count = (weights != 0).sum()
    
    # Check that weights sum to 1.0 or 0.0
    assert np.isclose(weights.sum(), 1.0) or np.isclose(weights.sum(), 0.0)
    
    # If weights sum to 1.0, check that they are properly distributed
    if np.isclose(weights.sum(), 1.0):
        # All weights should be non-negative
        assert (weights >= 0).all()
        
        # For single non-zero weight, it should be 1.0
        if non_zero_count == 1:
            assert weights.max() == 1.0
        # For multiple non-zero weights, they should be equal
        elif non_zero_count > 1:
            non_zero_weights = weights[weights > 0]
            assert np.allclose(non_zero_weights, non_zero_weights.iloc[0])


@st.composite
def non_zero_weights_and_leverage(draw):
    """Generate weights with at least one non-zero value and leverage."""
    n_assets = draw(st.integers(min_value=2, max_value=10))
    n_rows = draw(st.integers(min_value=1, max_value=5))
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate weights with at least one non-zero value per row
    weights_list = []
    for _ in range(n_rows):
        # Generate random weights
        row_weights = draw(
            hnp.arrays(
                dtype=float,
                shape=n_assets,
                elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        
        # Ensure at least one non-zero weight
        if np.all(np.isclose(row_weights, 0)):
            row_weights[0] = 1.0
            
        weights_list.append(row_weights)
    
    weights_df = pd.DataFrame(weights_list, columns=assets)
    leverage = draw(st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False))
    
    return weights_df, leverage


@given(non_zero_weights_and_leverage())
@settings(deadline=None)
def test_normalize_weights_properties(data):
    """Test properties of _normalize_weights."""
    weights_df, leverage = data
    
    # Calculate normalized weights
    normalized = _normalize_weights(weights_df, leverage)
    
    # Check that normalized weights have the same shape as input
    assert normalized.shape == weights_df.shape
    assert normalized.index.equals(weights_df.index)
    assert normalized.columns.equals(weights_df.columns)
    
    # Check that normalized weights are finite
    assert not normalized.isna().any().any()
    
    # For each row, check normalization properties
    for i in range(len(weights_df)):
        row = weights_df.iloc[i]
        norm_row = normalized.iloc[i]
        
        # If row sum is zero, normalized row should be all zeros
        if np.isclose(row.abs().sum(), 0):
            assert np.allclose(norm_row, 0)
        # Otherwise, normalized row absolute sum should equal leverage
        else:
            assert np.isclose(norm_row.abs().sum(), leverage, rtol=1e-6, atol=1e-9)
            
            # Check that signs are preserved
            for j in range(len(row)):
                if not np.isclose(row.iloc[j], 0):
                    assert np.sign(norm_row.iloc[j]) == np.sign(row.iloc[j])


@given(weights_and_leverage(allow_negative=True))
@settings(deadline=None)
def test_normalize_weights_preserves_direction(data):
    """Test that _normalize_weights preserves the direction of weights."""
    weights_df, leverage = data
    
    # Calculate normalized weights
    normalized = _normalize_weights(weights_df, leverage)
    
    # Check that signs are preserved for all non-zero weights
    for i in range(len(weights_df)):
        row = weights_df.iloc[i]
        norm_row = normalized.iloc[i]
        
        for j in range(len(row)):
            if not np.isclose(row.iloc[j], 0):
                assert np.sign(norm_row.iloc[j]) == np.sign(row.iloc[j])


@given(non_zero_weights_and_leverage())
@settings(deadline=None)
def test_normalize_weights_scales_with_leverage(data):
    """Test that _normalize_weights scales correctly with leverage."""
    weights_df, leverage = data
    
    # Calculate normalized weights with leverage 1.0
    normalized_base = _normalize_weights(weights_df, 1.0)
    
    # Calculate normalized weights with specified leverage
    normalized = _normalize_weights(weights_df, leverage)
    
    # Check that normalized weights scale with leverage
    for i in range(len(weights_df)):
        row = weights_df.iloc[i]
        
        # Only check rows with non-zero weights
        if not np.isclose(row.abs().sum(), 0):
            assert np.allclose(normalized.iloc[i], normalized_base.iloc[i] * leverage, rtol=1e-6, atol=1e-9)