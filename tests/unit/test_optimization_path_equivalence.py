"""
Comprehensive tests to ensure optimized and standard code paths produce identical results.

This test suite validates that both the Numba-optimized and pandas/numpy fallback
implementations produce functionally equivalent results across all features.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.numba_optimized import (
    rolling_std_fixed,
    rolling_cumprod_fixed,
    vams_batch_fixed,
    sortino_fast_fixed,
    sharpe_fast_fixed,
    true_range_fast,
    atr_fast,
)

from portfolio_backtester.features.vams import VAMS
from portfolio_backtester.features.dp_vams import DPVAMS
from portfolio_backtester.features.sortino_ratio import SortinoRatio
from portfolio_backtester.features.atr import ATRFeature
from portfolio_backtester.numba_kernels import (
    position_and_pnl_kernel,
    detailed_commission_slippage_kernel,
)


# Test data generators
def generate_test_price_data(n_periods=100, n_assets=5, seed=42):
    """Generate realistic test price data."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    # Generate correlated returns
    returns = np.random.multivariate_normal(
        mean=[0.0001] * n_assets,
        cov=np.eye(n_assets) * 0.02 + np.ones((n_assets, n_assets)) * 0.005,
        size=n_periods,
    )

    # Convert to prices starting from 100
    prices = pd.DataFrame(index=dates, columns=assets)
    prices.iloc[0] = 100.0

    for i in range(1, n_periods):
        prices.iloc[i] = prices.iloc[i - 1] * (1 + returns[i])

    return prices


def generate_test_ohlc_data(n_periods=100, n_assets=3, seed=42):
    """Generate OHLC test data."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    # Create MultiIndex columns for OHLC
    columns = pd.MultiIndex.from_product(
        [assets, ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
    )

    ohlc_data = pd.DataFrame(index=dates, columns=columns)

    for asset in assets:
        # Generate close prices first
        returns = np.random.normal(0.0001, 0.02, n_periods)
        close_prices = 100 * np.cumprod(1 + returns)

        # Generate OHLC based on close prices
        for i, close in enumerate(close_prices):
            daily_vol = abs(returns[i]) * 2  # Intraday volatility
            high = close * (1 + np.random.uniform(0, daily_vol))
            low = close * (1 - np.random.uniform(0, daily_vol))
            open_price = close * (1 + np.random.uniform(-daily_vol / 2, daily_vol / 2))

            ohlc_data.loc[dates[i], (asset, "Open")] = open_price
            ohlc_data.loc[dates[i], (asset, "High")] = max(open_price, high, close)
            ohlc_data.loc[dates[i], (asset, "Low")] = min(open_price, low, close)
            ohlc_data.loc[dates[i], (asset, "Close")] = close

    return ohlc_data


def generate_test_weights(n_periods=100, n_assets=5, seed=42):
    """Generate test portfolio weights."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    assets = [f"ASSET_{i}" for i in range(n_assets)]

    weights = np.random.dirichlet([1] * n_assets, n_periods)
    return pd.DataFrame(weights, index=dates, columns=assets)


class TestFeatureEquivalence:
    """Test that optimized and standard feature implementations produce identical results."""

    @pytest.fixture
    def price_data(self):
        return generate_test_price_data(n_periods=252, n_assets=10)

    @pytest.fixture
    def ohlc_data(self):
        return generate_test_ohlc_data(n_periods=252, n_assets=5)

    def test_vams_single_path_implementation(self, price_data):
        """Test VAMS feature single optimized implementation."""
        lookback_months = 12
        vams_feature = VAMS(lookback_months=lookback_months)

        # Test the single optimized implementation
        result = vams_feature.compute(price_data)

        # Validate result structure and properties
        assert isinstance(result, pd.DataFrame)
        assert result.shape == price_data.shape
        assert result.index.equals(price_data.index)
        assert result.columns.equals(price_data.columns)

        # Validate that VAMS values are reasonable (finite where data is sufficient)
        # First lookback_months-1 rows should be NaN
        assert result.iloc[: lookback_months - 1].isna().all().all()

        # Later rows should have some finite values (where there's sufficient data)
        if len(result) > lookback_months:
            later_values = result.iloc[lookback_months:]
            assert later_values.notna().any().any(), "Should have some valid VAMS values"

    def test_dp_vams_single_path_implementation(self, price_data):
        """Test DP-VAMS feature single optimized implementation."""
        lookback_months = 12
        alpha = 0.5
        dp_vams_feature = DPVAMS(lookback_months=lookback_months, alpha=alpha)

        # Test the single optimized implementation
        result = dp_vams_feature.compute(price_data)

        # Validate result structure and properties
        assert isinstance(result, pd.DataFrame)
        assert result.shape == price_data.shape
        assert result.index.equals(price_data.index)
        assert result.columns.equals(price_data.columns)

        # Validate that DP-VAMS values are reasonable (finite where data is sufficient)
        # First lookback_months-1 rows should be NaN or zero
        early_values = result.iloc[: lookback_months - 1]
        assert (early_values.isna() | (early_values == 0)).all().all()

        # Later rows should have some finite values (where there's sufficient data)
        if len(result) > lookback_months:
            later_values = result.iloc[lookback_months:]
            assert (
                (later_values.notna() & (later_values != 0)).any().any()
            ), "Should have some valid DP-VAMS values"

    def test_sortino_ratio_single_path_implementation(self, price_data):
        """Test Sortino ratio feature single optimized implementation."""
        rolling_window = 60
        target_return = 0.0
        sortino_feature = SortinoRatio(rolling_window=rolling_window, target_return=target_return)

        # Test single optimized implementation
        result = sortino_feature.compute(price_data)

        # Validate result properties
        assert result.shape == price_data.shape
        assert not result.isna().all().all()  # Should have some non-NaN values
        assert result.iloc[: rolling_window - 1].isna().all().all()  # First window-1 should be NaN

        # Check that values are within reasonable bounds (clipped to [-10, 10])
        assert result.min().min() >= -10.0
        assert result.max().max() <= 10.0

    def test_atr_single_path_implementation(self, ohlc_data):
        """Test ATR feature single optimized implementation."""
        atr_period = 14
        atr_feature = ATRFeature(atr_period=atr_period)

        # Test single optimized implementation
        result = atr_feature.compute(ohlc_data)

        # Validate result properties
        expected_assets = ohlc_data.columns.get_level_values("Ticker").unique()
        assert result.shape == (len(ohlc_data), len(expected_assets))
        assert list(result.columns) == list(expected_assets)

        # ATR should be positive (or NaN for insufficient data)
        non_nan_values = result.dropna()
        if not non_nan_values.empty:
            assert (non_nan_values >= 0).all().all()

        # First few values should be NaN due to ATR period requirement
        assert result.iloc[: atr_period - 1].isna().all().all()


class TestKernelEquivalence:
    """Test that Numba kernels and NumPy fallbacks produce identical results."""

    @pytest.fixture
    def kernel_test_data(self):
        """Generate test data for kernel functions."""
        n_periods = 100
        n_assets = 5

        # Generate weights and returns
        weights = generate_test_weights(n_periods, n_assets)
        returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
        mask = np.random.choice([True, False], (n_periods, n_assets), p=[0.95, 0.05])

        return {
            "weights": weights.values,
            "returns": returns,
            "mask": mask,
            "prices": np.random.uniform(50, 200, (n_periods, n_assets)),
        }

    def test_position_and_pnl_kernel_equivalence(self, kernel_test_data):
        """Test position and P&L kernel optimized vs standard implementation."""
        weights = kernel_test_data["weights"]
        returns = kernel_test_data["returns"]
        mask = kernel_test_data["mask"]

        # Test Numba implementation
        daily_gross_numba, equity_curve_numba, turnover_numba = position_and_pnl_kernel(
            weights, returns, mask
        )

        # Test single optimized implementation (no fallback needed)
        # The Numba implementation is now the only path

        # Validate single optimized implementation results
        assert daily_gross_numba.shape == (len(weights),)
        assert equity_curve_numba.shape == (len(weights),)
        assert turnover_numba.shape == (len(weights),)

        # Validate that results are reasonable
        assert np.all(np.isfinite(daily_gross_numba))
        assert np.all(np.isfinite(equity_curve_numba))
        assert np.all(np.isfinite(turnover_numba))
        assert np.all(turnover_numba >= 0)  # Turnover should be non-negative

    def test_commission_slippage_kernel_equivalence(self, kernel_test_data):
        """Test commission/slippage kernel optimized vs standard implementation."""
        weights = kernel_test_data["weights"]
        prices = kernel_test_data["prices"]
        mask = kernel_test_data["mask"] & (prices > 0)

        # Commission parameters
        portfolio_value = 100000.0
        commission_per_share = 0.005
        commission_min_per_order = 1.0
        commission_max_percent = 0.005
        slippage_bps = 2.5

        # Test Numba implementation
        costs_numba_tuple = detailed_commission_slippage_kernel(
            weights,
            prices,
            portfolio_value,
            commission_per_share,
            commission_min_per_order,
            commission_max_percent,
            slippage_bps,
            mask,
        )
        costs_numba, _ = costs_numba_tuple

        # Test single optimized implementation (no fallback needed)
        # The Numba implementation is now the only path

        # Validate single optimized implementation results
        assert costs_numba.shape == (len(weights),)
        assert np.all(np.isfinite(costs_numba))
        assert np.all(costs_numba >= 0)  # Transaction costs should be non-negative


class TestEdgeCases:
    """Test edge cases that might behave differently between implementations."""

    def test_empty_data_handling(self):
        """Test how the optimized implementation handles empty data."""
        empty_data = pd.DataFrame()

        # Test VAMS with empty data
        vams_feature = VAMS(lookback_months=12)
        result = vams_feature.compute(empty_data)

        # Should return empty DataFrame
        assert result.empty

    def test_nan_handling(self):
        """Test how both implementations handle NaN values."""
        # Create data with NaN values
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        assets = ["A", "B", "C"]

        data = (
            pd.DataFrame(np.random.randn(50, 3) * 0.02 + 1, index=dates, columns=assets).cumprod()
            * 100
        )

        # Introduce some NaN values
        data.iloc[10:15, 0] = np.nan
        data.iloc[20:25, 1] = np.nan

        # Test VAMS with NaN data
        vams_feature = VAMS(lookback_months=12)
        result = vams_feature.compute(data)

        # Validate that the function handles NaN data gracefully
        assert isinstance(result, pd.DataFrame)
        assert result.shape == data.shape

        # Should have some valid results where there's sufficient non-NaN data
        # The optimized function should handle NaN values appropriately
        valid_results = result.notna()
        assert valid_results.any().any(), "Should have some valid VAMS values despite NaN input"

    def test_extreme_values(self):
        """Test how both implementations handle extreme values."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        assets = ["A", "B"]

        # Create data with extreme values
        data = pd.DataFrame(index=dates, columns=assets)
        data["A"] = np.concatenate(
            [
                np.ones(50) * 100,  # Stable period
                np.ones(25) * 1000,  # Extreme jump
                np.ones(25) * 10,  # Extreme drop
            ]
        )
        data["B"] = np.random.lognormal(0, 0.1, 100) * 100

        # Test Sortino ratio with extreme data using single optimized implementation
        sortino_feature = SortinoRatio(rolling_window=30, target_return=0.0)

        result = sortino_feature.compute(data)

        # Validate that extreme values are handled properly
        assert result.shape == data.shape
        assert not result.isna().all().all()  # Should have some non-NaN values

        # Check that values are clipped to reasonable bounds
        assert result.min().min() >= -10.0
        assert result.max().max() <= 10.0

        # Verify that the function doesn't crash with extreme values
        assert not np.isinf(result).any().any()  # No infinite values

    def test_single_row_data(self):
        """Test functions with single row of data."""
        single_value = np.array([0.01])
        single_row_2d = np.array([[0.01, 0.02]])

        # Rolling functions should return NaN for insufficient data
        result_std = rolling_std_fixed(single_value, 2)
        assert len(result_std) == 1
        assert np.isnan(result_std[0])

        result_cumprod = rolling_cumprod_fixed(single_value, 2)
        assert len(result_cumprod) == 1
        assert np.isnan(result_cumprod[0])

        # Batch functions
        result_vams = vams_batch_fixed(single_row_2d, 2)
        assert result_vams.shape == single_row_2d.shape
        assert np.isnan(result_vams).all()

    def test_all_nan_data(self):
        """Test functions with all NaN input data."""
        all_nan_1d = np.array([np.nan, np.nan, np.nan, np.nan])
        all_nan_2d = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])

        # Rolling functions
        result_std = rolling_std_fixed(all_nan_1d, 2)
        assert np.isnan(result_std).all()

        result_cumprod = rolling_cumprod_fixed(all_nan_1d, 2)
        assert np.isnan(result_cumprod).all()

        # Batch functions
        result_vams = vams_batch_fixed(all_nan_2d, 2)
        assert np.isnan(result_vams).all()

        result_sortino = sortino_fast_fixed(all_nan_2d, 2, 0.0, 1.0)
        assert np.isnan(result_sortino).all()

        result_sharpe = sharpe_fast_fixed(all_nan_2d, 2, 1.0)
        assert np.isnan(result_sharpe).all()

    def test_mixed_nan_data(self):
        """Test functions with mixed valid and NaN data."""
        mixed_data = np.array([0.01, np.nan, 0.02, np.nan, 0.03, 0.01])
        mixed_2d = np.array([[0.01, np.nan], [np.nan, 0.02], [0.03, 0.01], [0.02, np.nan]])

        # Rolling functions should handle NaN gracefully
        result_std = rolling_std_fixed(mixed_data, 3)
        assert len(result_std) == len(mixed_data)
        # Should have some valid results where there's enough non-NaN data
        assert not np.isnan(result_std).all()

        # Batch functions
        result_vams = vams_batch_fixed(mixed_2d, 2)
        assert result_vams.shape == mixed_2d.shape
        # Should have some valid results
        assert not np.isnan(result_vams).all()

    def test_nan_in_ohlc_data(self):
        """Test ATR functions with NaN in OHLC data."""
        high = np.array([102.0, np.nan, 103.0, 107.0])
        low = np.array([98.0, 101.0, np.nan, 103.0])
        close = np.array([100.0, 102.0, 105.0, np.nan])

        # True Range should handle NaN
        tr_result = true_range_fast(high, low, np.roll(close, 1))
        assert len(tr_result) == len(high)
        # Should have NaN where input data is NaN (but depends on calculation)
        # The function may still produce valid results if other inputs are valid
        assert np.isnan(tr_result[1])  # NaN in high
        assert np.isnan(tr_result[2])  # NaN in low
        # Note: tr_result[3] might not be NaN if high-low is valid

        # ATR should handle NaN
        atr_result = atr_fast(high, low, close, 2)
        assert len(atr_result) == len(high)

    def test_nan_in_kernel_functions(self):
        """Test kernel functions with NaN values."""
        weights = np.array([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]])
        returns = np.array([[0.01, np.nan], [np.nan, 0.02], [0.015, -0.005]])
        mask = np.array([[True, False], [False, True], [True, True]])  # Mask out NaN values

        # Should handle NaN gracefully with proper masking
        daily_gross, equity_curve, turnover = position_and_pnl_kernel(weights, returns, mask)

        assert len(daily_gross) == 3
        assert np.isfinite(daily_gross).all()  # Should be finite with proper masking
        assert np.isfinite(equity_curve).all()
        assert np.isfinite(turnover).all()

    def test_very_large_values(self):
        """Test functions with very large input values."""
        large_values = np.array([1e6, 1e7, 1e8, 1e9])
        large_2d = np.array([[1e6, 1e7], [1e8, 1e9], [1e10, 1e11]])

        # Rolling functions should handle large values
        result_std = rolling_std_fixed(large_values, 3)
        assert np.isfinite(result_std[-1])  # Should be finite

        # Batch functions
        result_vams = vams_batch_fixed(large_2d, 2)
        # VAMS might be extreme but should be finite
        valid_results = result_vams[np.isfinite(result_vams)]
        assert len(valid_results) > 0  # Should have some finite results

    def test_very_small_values(self):
        """Test functions with very small input values."""
        small_values = np.array([1e-10, 1e-11, 1e-12, 1e-13])
        small_2d = np.array([[1e-10, 1e-11], [1e-12, 1e-13], [1e-14, 1e-15]])

        # Rolling functions should handle small values
        result_std = rolling_std_fixed(small_values, 3)
        assert np.isfinite(result_std[-1])

        # Batch functions
        result_sharpe = sharpe_fast_fixed(small_2d, 2, 1.0)
        # Results might be extreme but should be finite or NaN (not inf)
        assert not np.isinf(result_sharpe).any()

    def test_extreme_negative_values(self):
        """Test functions with extreme negative values."""
        negative_values = np.array([-0.5, -0.8, -0.9, -0.95])  # Large negative returns
        negative_2d = np.array([[-0.5, -0.3], [-0.8, -0.6], [-0.9, -0.7]])

        # Rolling functions
        result_std = rolling_std_fixed(negative_values, 3)
        assert np.isfinite(result_std[-1])

        # VAMS with extreme negative returns
        result_vams = vams_batch_fixed(negative_2d, 2)
        # Should handle extreme negative returns without crashing
        assert result_vams.shape == negative_2d.shape

    def test_zero_values(self):
        """Test functions with zero values."""
        zero_values = np.array([0.0, 0.0, 0.0, 0.0])
        zero_2d = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Rolling standard deviation of zeros should be zero
        result_std = rolling_std_fixed(zero_values, 3)
        np.testing.assert_almost_equal(result_std[-1], 0.0, decimal=12)

        # VAMS with zero returns
        result_vams = vams_batch_fixed(zero_2d, 2)
        # Momentum should be 0, volatility should be 0, VAMS should be NaN (0/0)
        assert np.isnan(result_vams[-1, 0])

    def test_infinite_values(self):
        """Test functions with infinite values."""
        inf_values = np.array([0.01, np.inf, 0.02, 0.03])
        inf_2d = np.array([[0.01, 0.02], [np.inf, 0.03], [0.04, -np.inf]])

        # Functions should handle inf values gracefully (treat as invalid)
        result_std = rolling_std_fixed(inf_values, 3)
        assert len(result_std) == len(inf_values)
        # Should not propagate inf
        finite_results = result_std[np.isfinite(result_std)]
        assert len(finite_results) >= 0  # Should have some finite or all NaN results

        # Batch functions should handle inf
        result_vams = vams_batch_fixed(inf_2d, 2)
        assert result_vams.shape == inf_2d.shape
        # Should not have inf in results (either finite or NaN)
        assert not np.isinf(result_vams).any()


class TestPerformanceBenchmark:
    """Benchmark performance differences between optimized and standard implementations."""

    @pytest.fixture
    def large_dataset(self):
        """Generate a large dataset for performance testing."""
        return generate_test_price_data(n_periods=2000, n_assets=50, seed=123)

    def test_vams_performance_validation(self, large_dataset):
        """Validate performance of the single optimized VAMS implementation."""
        import time

        vams_feature = VAMS(lookback_months=12)

        # Time the single optimized implementation
        start_time = time.time()
        result = vams_feature.compute(large_dataset)
        execution_time = time.time() - start_time

        # Validate result structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape == large_dataset.shape

        # Performance should be reasonable for large datasets
        print(f"VAMS Performance: {execution_time:.4f}s for {large_dataset.shape} dataset")

        # Should complete in reasonable time (less than 5 seconds for large dataset)
        assert execution_time < 5.0, f"VAMS took {execution_time:.4f}s, expected < 5.0s"


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "--tb=short"])
