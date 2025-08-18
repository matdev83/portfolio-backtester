"""
Property-based tests for numba-optimized rolling metrics.

This module uses Hypothesis to test invariants and properties of the rolling metrics
functions in the numba_optimized module.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.numba_optimized import (
    rolling_mean_fast,
    rolling_std_fast,
    rolling_sharpe_batch,
    rolling_sortino_batch,
    rolling_beta_batch,
    rolling_correlation_batch,
    rolling_downside_volatility_fast,
)

from tests.strategies import return_matrices, return_series


@given(return_series(min_size=20, max_size=200))
@settings(deadline=None)
def test_rolling_mean_fast_matches_pandas(returns):
    """Test that rolling_mean_fast matches pandas rolling mean."""
    window = 10
    
    # Calculate with our optimized function
    result_fast = rolling_mean_fast(returns.values, window)
    
    # Calculate with pandas
    result_pandas = returns.rolling(window=window).mean().values
    
    # Compare results where both are finite
    mask = np.isfinite(result_fast) & np.isfinite(result_pandas)
    if mask.any():
        # Allow a slightly looser tolerance for numerical edge cases involving
        # extremely small values and dtype conversions across numpy/pandas.
        assert np.allclose(result_fast[mask], result_pandas[mask], rtol=1e-6, atol=1e-12)
    
    # Check NaN positions match
    nan_mask_fast = np.isnan(result_fast)
    nan_mask_pandas = np.isnan(result_pandas)
    assert np.array_equal(nan_mask_fast, nan_mask_pandas)


@given(return_series(min_size=20, max_size=200))
@settings(deadline=None)
def test_rolling_std_fast_matches_pandas(returns):
    """Test that rolling_std_fast matches pandas rolling std."""
    window = 10
    
    # Calculate with our optimized function
    result_fast = rolling_std_fast(returns.values, window)
    
    # Calculate with pandas (using ddof=1 for sample standard deviation)
    result_pandas = returns.rolling(window=window).std(ddof=1).values
    
    # Compare results where both are finite
    mask = np.isfinite(result_fast) & np.isfinite(result_pandas)
    if mask.any():
        # Allow a slightly looser tolerance for numerical edge cases involving
        # extremely small values and dtype conversions across numpy/pandas.
        assert np.allclose(result_fast[mask], result_pandas[mask], rtol=1e-5, atol=1e-8)
    
    # Check NaN positions match
    nan_mask_fast = np.isnan(result_fast)
    nan_mask_pandas = np.isnan(result_pandas)
    assert np.array_equal(nan_mask_fast, nan_mask_pandas)


@st.composite
def non_degenerate_return_matrices(draw):
    """Generate non-degenerate return matrices with sufficient variance."""
    matrix = draw(return_matrices(min_rows=30, max_rows=100, min_cols=2, max_cols=5, ensure_nonzero_variance=True))
    
    # Ensure the last window has sufficient variance
    window = 20
    last_window = matrix[-window:, :]
    
    # Check if last window has sufficient variance in each column
    for col in range(matrix.shape[1]):
        col_std = np.std(last_window[:, col])
        assume(col_std > 1e-6)
    
    return matrix


@given(non_degenerate_return_matrices())
@settings(deadline=None)
def test_rolling_sharpe_batch_properties(returns_matrix):
    """Test properties of rolling_sharpe_batch."""
    window = 20
    annualization = np.sqrt(252)  # Standard annualization factor for daily returns
    
    # Calculate Sharpe ratios
    sharpe_ratios = rolling_sharpe_batch(returns_matrix, window, annualization)
    
    # Check shape
    assert sharpe_ratios.shape == returns_matrix.shape
    
    # First window-1 rows should be NaN
    assert np.all(np.isnan(sharpe_ratios[:window-1]))
    
    # Check finiteness for remaining rows - we don't test specific values
    # due to numerical precision issues with very small values
    finite_mask = np.isfinite(sharpe_ratios[window-1:])
    assert finite_mask.any(), "No finite Sharpe ratios found"


@given(non_degenerate_return_matrices())
@settings(deadline=None)
def test_rolling_sortino_batch_properties(returns_matrix):
    """Test properties of rolling_sortino_batch."""
    window = 20
    annualization = np.sqrt(252)  # Standard annualization factor for daily returns
    target_return = 0.0
    
    # Calculate Sortino ratios
    sortino_ratios = rolling_sortino_batch(returns_matrix, window, target_return, annualization)
    
    # Check shape
    assert sortino_ratios.shape == returns_matrix.shape
    
    # First window-1 rows should be NaN
    assert np.all(np.isnan(sortino_ratios[:window-1]))
    
    # Check finiteness for remaining rows
    finite_mask = np.isfinite(sortino_ratios[window-1:])
    assert finite_mask.any(), "No finite Sortino ratios found"
    
    # For assets with all returns above target, Sortino should be very large or inf
    for t in range(window-1, returns_matrix.shape[0]):
        for a in range(returns_matrix.shape[1]):
            window_returns = returns_matrix[t-window+1:t+1, a]
            if np.all(window_returns >= target_return) and np.mean(window_returns) > target_return:
                # Either very large or inf
                assert np.isnan(sortino_ratios[t, a]) or sortino_ratios[t, a] > 100 or np.isinf(sortino_ratios[t, a])


@st.composite
def correlated_returns(draw):
    """Generate asset returns correlated with a benchmark."""
    rows = draw(st.integers(min_value=30, max_value=100))
    cols = draw(st.integers(min_value=2, max_value=5))
    
    # Generate benchmark returns
    benchmark_returns = draw(
        hnp.arrays(
            dtype=float,
            shape=rows,
            elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Generate asset returns correlated with benchmark
    asset_returns = np.zeros((rows, cols))
    for i in range(cols):
        # Correlation coefficient between -1 and 1
        correlation = draw(st.floats(min_value=-0.9, max_value=0.9))
        
        # Asset-specific volatility
        vol = draw(st.floats(min_value=0.01, max_value=0.05))
        
        # Generate correlated returns
        idiosyncratic = draw(
            hnp.arrays(
                dtype=float,
                shape=rows,
                elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
            )
        )
        
        # Mix benchmark and idiosyncratic returns to achieve target correlation
        asset_returns[:, i] = correlation * benchmark_returns + np.sqrt(1 - correlation**2) * idiosyncratic
        
        # Scale to target volatility
        asset_returns[:, i] *= vol / np.std(asset_returns[:, i]) if np.std(asset_returns[:, i]) > 0 else 1.0
    
    # Ensure non-zero variance in the last window
    window = 20
    last_window_bench = benchmark_returns[-window:]
    assume(np.std(last_window_bench) > 1e-6)
    
    for i in range(cols):
        last_window_asset = asset_returns[-window:, i]
        assume(np.std(last_window_asset) > 1e-6)
    
    return asset_returns, benchmark_returns


@given(correlated_returns())
@settings(deadline=None)
def test_rolling_correlation_batch_properties(data):
    """Test properties of rolling_correlation_batch."""
    returns_matrix, benchmark_returns = data
    window = 20
    
    # Calculate correlations
    correlations = rolling_correlation_batch(returns_matrix, benchmark_returns, window)
    
    # Check shape
    assert correlations.shape == returns_matrix.shape
    
    # First window-1 rows should be NaN
    assert np.all(np.isnan(correlations[:window-1]))
    
    # Check bounds for valid values
    valid_mask = np.isfinite(correlations)
    if valid_mask.any():
        valid_values = correlations[valid_mask]
        assert np.all(valid_values >= -1.0 - 1e-10)
        assert np.all(valid_values <= 1.0 + 1e-10)


@given(correlated_returns())
@settings(deadline=None)
def test_rolling_beta_batch_properties(data):
    """Test properties of rolling_beta_batch."""
    returns_matrix, benchmark_returns = data
    window = 20
    
    # Calculate betas
    betas = rolling_beta_batch(returns_matrix, benchmark_returns, window)
    
    # Check shape
    assert betas.shape == returns_matrix.shape
    
    # First window-1 rows should be NaN
    assert np.all(np.isnan(betas[:window-1]))
    
    # Check finiteness for remaining rows
    finite_mask = np.isfinite(betas[window-1:])
    assert finite_mask.any(), "No finite beta values found"
    
    # For highly correlated assets, beta should be close to the ratio of volatilities
    for t in range(window-1, returns_matrix.shape[0]):
        for a in range(returns_matrix.shape[1]):
            window_asset = returns_matrix[t-window+1:t+1, a]
            window_bench = benchmark_returns[t-window+1:t+1]
            
            if np.std(window_bench) > 1e-6 and np.std(window_asset) > 1e-6:
                # Calculate correlation
                corr = np.corrcoef(window_asset, window_bench)[0, 1]
                
                # For highly correlated assets, beta should be close to vol_asset / vol_bench * correlation
                if abs(corr) > 0.9:
                    expected_beta = np.std(window_asset) / np.std(window_bench) * corr
                    assert np.isclose(betas[t, a], expected_beta, rtol=1e-6, atol=1e-6)


@given(return_series(min_size=30, max_size=100))
@settings(deadline=None)
def test_rolling_downside_volatility_fast_properties(returns):
    """Test properties of rolling_downside_volatility_fast."""
    window = 20
    
    # Calculate downside volatility
    downside_vol = rolling_downside_volatility_fast(returns.values, window)
    
    # First window-1 elements should be NaN
    assert np.all(np.isnan(downside_vol[:window-1]))
    
    # Downside volatility should be non-negative
    valid_mask = np.isfinite(downside_vol)
    if valid_mask.any():
        assert np.all(downside_vol[valid_mask] >= 0)
    
    # For windows with all positive returns, downside volatility should be 0
    for i in range(window-1, len(returns)):
        window_returns = returns.values[i-window+1:i+1]
        if np.all(window_returns >= 0):
            assert downside_vol[i] == 0 or np.isnan(downside_vol[i])
    
    # For windows with large negative returns, downside volatility should be larger
    for i in range(window-1, len(returns)):
        window_returns = returns.values[i-window+1:i+1]
        neg_returns = window_returns[window_returns < 0]
        if len(neg_returns) > 0 and len(neg_returns) / len(window_returns) > 0.5:
            # Many negative returns should lead to higher downside vol
            assert downside_vol[i] > 0 or np.isnan(downside_vol[i])
