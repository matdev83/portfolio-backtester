"""
Tests for Numba-optimized mathematical functions.

These tests ensure mathematical equivalence between Numba-optimized functions
and their pandas counterparts, with comprehensive edge case coverage.
"""

import pytest
import numpy as np
import pandas as pd
from src.portfolio_backtester.numba_optimized import (
    momentum_scores_fast, momentum_scores_fast_vectorized,
    rolling_mean_fast, rolling_std_fast, rolling_sharpe_fast,
    rolling_sortino_fast, rolling_beta_fast, rolling_correlation_fast
)


class TestMomentumScoresFast:
    """Test Numba-optimized momentum score calculations."""
    
    def test_basic_momentum_calculation(self):
        """Test basic momentum calculation matches pandas."""
        # Test data
        prices_now = np.array([110.0, 95.0, 105.0, 120.0])
        prices_then = np.array([100.0, 100.0, 100.0, 100.0])
        
        # Expected results (manual calculation)
        expected = np.array([0.1, -0.05, 0.05, 0.2])  # (110/100-1, 95/100-1, etc.)
        
        # Numba calculation
        result = momentum_scores_fast(prices_now, prices_then)
        
        # Pandas equivalent
        pandas_result = (prices_now / prices_then) - 1.0
        
        # Verify mathematical equivalence
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
        np.testing.assert_array_almost_equal(result, pandas_result, decimal=10)
    
    def test_vectorized_momentum_calculation(self):
        """Test vectorized version matches loop version."""
        prices_now = np.array([110.0, 95.0, 105.0, 120.0])
        prices_then = np.array([100.0, 100.0, 100.0, 100.0])
        
        result_loop = momentum_scores_fast(prices_now, prices_then)
        result_vectorized = momentum_scores_fast_vectorized(prices_now, prices_then)
        
        np.testing.assert_array_almost_equal(result_loop, result_vectorized, decimal=10)
    
    def test_zero_prices_handling(self):
        """Test handling of zero prices (should return NaN)."""
        prices_now = np.array([110.0, 95.0])
        prices_then = np.array([0.0, 100.0])  # Zero price
        
        result = momentum_scores_fast(prices_now, prices_then)
        
        # First element should be NaN due to division by zero
        assert np.isnan(result[0])
        # Second element should be valid
        np.testing.assert_almost_equal(result[1], -0.05, decimal=10)
    
    def test_negative_prices_handling(self):
        """Test handling of negative prices (should return NaN)."""
        prices_now = np.array([110.0, 95.0])
        prices_then = np.array([-100.0, 100.0])  # Negative price
        
        result = momentum_scores_fast(prices_now, prices_then)
        
        # First element should be NaN due to negative price
        assert np.isnan(result[0])
        # Second element should be valid
        np.testing.assert_almost_equal(result[1], -0.05, decimal=10)
    
    def test_nan_prices_handling(self):
        """Test handling of NaN prices."""
        prices_now = np.array([110.0, np.nan, 105.0])
        prices_then = np.array([100.0, 100.0, np.nan])
        
        result = momentum_scores_fast(prices_now, prices_then)
        
        # First element should be valid
        np.testing.assert_almost_equal(result[0], 0.1, decimal=10)
        # Second and third elements should be NaN
        assert np.isnan(result[1])
        assert np.isnan(result[2])


class TestPositionSizerOptimizations:
    """Test Numba-optimized position sizer rolling statistics."""
    
    def test_rolling_mean_fast(self):
        """Test fast rolling mean calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        window = 3
        
        # Numba calculation
        result_numba = rolling_mean_fast(data, window)
        
        # Manual verification for specific windows
        # Window at index 2: [1,2,3] -> mean = 2.0
        # Window at index 3: [2,3,4] -> mean = 3.0
        np.testing.assert_almost_equal(result_numba[2], 2.0, decimal=10)
        np.testing.assert_almost_equal(result_numba[3], 3.0, decimal=10)
        
        # First two elements should be NaN (not enough data)
        assert np.isnan(result_numba[0])
        assert np.isnan(result_numba[1])
    
    def test_rolling_std_fast(self):
        """Test fast rolling standard deviation calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        window = 3
        
        # Numba calculation
        result_numba = rolling_std_fast(data, window)
        
        # Manual verification for window [1,2,3] at index 2
        expected_std = np.std([1.0, 2.0, 3.0])
        np.testing.assert_almost_equal(result_numba[2], expected_std, decimal=10)
        
        # First two elements should be NaN
        assert np.isnan(result_numba[0])
        assert np.isnan(result_numba[1])
    
    def test_rolling_sharpe_fast(self):
        """Test fast rolling Sharpe ratio calculation."""
        # Create simple test data
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.025, 0.02])
        window = 3
        
        # Numba calculation
        result_numba = rolling_sharpe_fast(returns, window)
        
        # Manual verification for window [0.01, 0.02, 0.015] at index 2
        window_data = returns[0:3]
        expected_mean = np.mean(window_data)
        expected_std = np.std(window_data)
        expected_sharpe = expected_mean / expected_std
        
        np.testing.assert_almost_equal(result_numba[2], expected_sharpe, decimal=8)
        
        # First two elements should be NaN
        assert np.isnan(result_numba[0])
        assert np.isnan(result_numba[1])
    
    def test_rolling_beta_fast(self):
        """Test fast rolling beta calculation."""
        # Create simple correlated data
        benchmark_returns = np.array([0.01, 0.02, 0.015, 0.01, 0.025])
        asset_returns = np.array([0.012, 0.024, 0.018, 0.012, 0.030])  # Beta should be ~1.2
        window = 3
        
        # Numba calculation
        result_numba = rolling_beta_fast(asset_returns, benchmark_returns, window)
        
        # Check that we get reasonable beta values
        assert not np.isnan(result_numba[2])  # Should have valid result at index 2
        assert 0.5 < result_numba[2] < 2.0  # Reasonable beta range
    
    def test_rolling_functions_with_nan_data(self):
        """Test rolling functions handle NaN data correctly."""
        data = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0])
        window = 3
        
        # Test all rolling functions with NaN data
        mean_result = rolling_mean_fast(data, window)
        std_result = rolling_std_fast(data, window)
        sharpe_result = rolling_sharpe_fast(data, window)
        
        # Should not crash and should handle NaN appropriately
        assert len(mean_result) == len(data)
        assert len(std_result) == len(data)
        assert len(sharpe_result) == len(data)
        
        # Results should be finite where there's enough valid data
        assert np.isfinite(mean_result[-1])  # Last window should have enough data
    
    def test_rolling_functions_edge_cases(self):
        """Test rolling functions with edge cases."""
        # Empty array
        empty_data = np.array([])
        result_empty = rolling_mean_fast(empty_data, 3)
        assert len(result_empty) == 0
        
        # Single element
        single_data = np.array([5.0])
        result_single = rolling_mean_fast(single_data, 3)
        assert len(result_single) == 1
        assert np.isnan(result_single[0])  # Not enough data for window
        
        # All NaN
        nan_data = np.array([np.nan, np.nan, np.nan, np.nan])
        result_nan = rolling_mean_fast(nan_data, 2)
        assert len(result_nan) == 4
        assert np.isnan(result_nan).all()
    
    def test_rolling_functions_mathematical_properties(self):
        """Test mathematical properties of rolling functions."""
        # Constant data should have zero standard deviation
        constant_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        std_result = rolling_std_fast(constant_data, 3)
        
        # Standard deviation of constant data should be 0
        for i in range(2, len(constant_data)):  # After window-1
            if not np.isnan(std_result[i]):
                np.testing.assert_almost_equal(std_result[i], 0.0, decimal=10)
        
        # Mean of constant data should equal the constant
        mean_result = rolling_mean_fast(constant_data, 3)
        for i in range(2, len(constant_data)):
            if not np.isnan(mean_result[i]):
                np.testing.assert_almost_equal(mean_result[i], 5.0, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])