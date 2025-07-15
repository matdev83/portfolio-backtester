"""
Tests for Numba-optimized mathematical functions.

These tests ensure mathematical equivalence between Numba-optimized functions
and their pandas counterparts, with comprehensive edge case coverage.
"""

import numpy as np
import pytest

from src.portfolio_backtester.numba_optimized import (
    atr_exponential_fast,
    atr_fast,
    momentum_scores_fast,
    momentum_scores_fast_vectorized,
    rolling_beta_fast,
    rolling_mean_fast,
    rolling_sharpe_fast,
    rolling_std_fast,
    true_range_fast,
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


class TestATROptimizations:
    """Test Numba-optimized ATR calculations."""
    
    def test_true_range_fast(self):
        """Test fast True Range calculation."""
        # Create test OHLC data
        high = np.array([102.0, 105.0, 103.0, 107.0, 104.0])
        low = np.array([98.0, 101.0, 99.0, 103.0, 100.0])
        close_prev = np.array([100.0, 102.0, 105.0, 103.0, 107.0])
        
        # Numba calculation
        tr_numba = true_range_fast(high, low, close_prev)
        
        # Manual calculation for verification
        expected_tr = np.array([
            max(102.0 - 98.0, abs(102.0 - 100.0), abs(98.0 - 100.0)),  # max(4, 2, 2) = 4
            max(105.0 - 101.0, abs(105.0 - 102.0), abs(101.0 - 102.0)),  # max(4, 3, 1) = 4
            max(103.0 - 99.0, abs(103.0 - 105.0), abs(99.0 - 105.0)),  # max(4, 2, 6) = 6
            max(107.0 - 103.0, abs(107.0 - 103.0), abs(103.0 - 103.0)),  # max(4, 4, 0) = 4
            max(104.0 - 100.0, abs(104.0 - 107.0), abs(100.0 - 107.0))   # max(4, 3, 7) = 7
        ])
        
        np.testing.assert_array_almost_equal(tr_numba, expected_tr, decimal=10)
    
    def test_atr_fast(self):
        """Test fast ATR calculation."""
        # Create test OHLC data
        high = np.array([102.0, 105.0, 103.0, 107.0, 104.0, 106.0, 108.0])
        low = np.array([98.0, 101.0, 99.0, 103.0, 100.0, 102.0, 104.0])
        close = np.array([100.0, 102.0, 105.0, 103.0, 107.0, 104.0, 106.0])
        window = 3
        
        # Numba calculation
        atr_numba = atr_fast(high, low, close, window)
        
        # Verify structure
        assert len(atr_numba) == len(high)
        assert np.isnan(atr_numba[0])  # First value should be NaN
        assert np.isnan(atr_numba[1])  # Second value should be NaN (not enough data)
        
        # Check that we get valid ATR values after sufficient data
        assert not np.isnan(atr_numba[-1])  # Last value should be valid
        assert atr_numba[-1] > 0  # ATR should be positive
    
    def test_atr_exponential_fast(self):
        """Test fast exponential ATR calculation."""
        # Create test OHLC data
        high = np.array([102.0, 105.0, 103.0, 107.0, 104.0, 106.0, 108.0, 105.0])
        low = np.array([98.0, 101.0, 99.0, 103.0, 100.0, 102.0, 104.0, 101.0])
        close = np.array([100.0, 102.0, 105.0, 103.0, 107.0, 104.0, 106.0, 105.0])
        window = 14
        
        # Numba calculation
        atr_exp_numba = atr_exponential_fast(high, low, close, window)
        
        # Verify structure
        assert len(atr_exp_numba) == len(high)
        assert np.isnan(atr_exp_numba[0])  # First value should be NaN
        assert not np.isnan(atr_exp_numba[1])  # Second value should be valid (first TR)
        
        # Check exponential smoothing property (later values influenced by earlier)
        assert atr_exp_numba[-1] > 0  # ATR should be positive
        
        # Verify exponential smoothing is working (values should be related)
        valid_values = atr_exp_numba[~np.isnan(atr_exp_numba)]
        assert len(valid_values) >= 2
    
    
    def test_atr_with_nan_data(self):
        """Test ATR functions handle NaN data correctly."""
        # Create data with NaN values
        high = np.array([102.0, np.nan, 103.0, 107.0, 104.0])
        low = np.array([98.0, 101.0, np.nan, 103.0, 100.0])
        close = np.array([100.0, 102.0, 105.0, np.nan, 107.0])
        close_prev = np.array([99.0, 100.0, 102.0, 105.0, 103.0])  # Previous close for TR
        
        # Test True Range with NaN
        tr_result = true_range_fast(high, low, close_prev)
        assert len(tr_result) == 5
        assert np.isnan(tr_result[1])  # Should be NaN due to NaN in high
        assert np.isnan(tr_result[2])  # Should be NaN due to NaN in low
        
        # Test ATR with NaN
        atr_result = atr_fast(high, low, close, 3)
        assert len(atr_result) == len(high)
        # Should handle NaN gracefully without crashing
    
    def test_atr_edge_cases(self):
        """Test ATR functions with edge cases."""
        # Empty arrays
        empty_array = np.array([])
        atr_empty = atr_fast(empty_array, empty_array, empty_array, 3)
        assert len(atr_empty) == 0
        
        # Single element
        single_high = np.array([100.0])
        single_low = np.array([95.0])
        single_close = np.array([98.0])
        atr_single = atr_fast(single_high, single_low, single_close, 3)
        assert len(atr_single) == 1
        assert np.isnan(atr_single[0])  # Not enough data
        
        # Two elements (minimum for TR calculation)
        two_high = np.array([100.0, 102.0])
        two_low = np.array([95.0, 98.0])
        two_close = np.array([98.0, 101.0])
        atr_two = atr_fast(two_high, two_low, two_close, 2)
        assert len(atr_two) == 2
        assert np.isnan(atr_two[0])  # First value always NaN
        # Second value might still be NaN due to window requirement
        # Just check that function doesn't crash
    
    def test_atr_mathematical_properties(self):
        """Test mathematical properties of ATR calculations."""
        # ATR should always be positive for valid data
        high = np.array([102.0, 105.0, 103.0, 107.0, 104.0])
        low = np.array([98.0, 101.0, 99.0, 103.0, 100.0])
        close = np.array([100.0, 102.0, 105.0, 103.0, 107.0])
        
        atr_result = atr_fast(high, low, close, 3)
        
        # All valid ATR values should be positive
        valid_atr = atr_result[~np.isnan(atr_result)]
        assert (valid_atr >= 0).all()
        
        # ATR should be reasonable relative to price ranges
        price_range = np.max(high) - np.min(low)
        max_atr = np.max(valid_atr)
        assert max_atr <= price_range  # ATR shouldn't exceed total price range
    
    def test_atr_vs_pandas_equivalent(self):
        """Test ATR calculation against pandas equivalent where possible."""
        # Create test data
        high = np.array([102.0, 105.0, 103.0, 107.0, 104.0, 106.0, 108.0])
        low = np.array([98.0, 101.0, 99.0, 103.0, 100.0, 102.0, 104.0])
        close = np.array([100.0, 102.0, 105.0, 103.0, 107.0, 104.0, 106.0])
        
        # Calculate True Range manually for comparison
        tr_manual = np.full(len(high), np.nan)
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_manual[i] = max(tr1, tr2, tr3)
        
        # Compare with Numba True Range
        tr_numba = true_range_fast(high[1:], low[1:], close[:-1])
        
        # Should match manual calculation
        np.testing.assert_array_almost_equal(tr_numba, tr_manual[1:], decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])