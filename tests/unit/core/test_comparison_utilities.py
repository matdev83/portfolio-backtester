"""
Side-by-side comparison utilities for testing equivalence.

This module provides utilities to compare Numba-optimized implementations
with their pandas/numpy equivalents, ensuring mathematical consistency
and identifying any discrepancies.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Callable, Dict, Any, Tuple, Optional, List
import warnings
from dataclasses import dataclass

from portfolio_backtester.numba_optimized import (
    rolling_std_fixed,
    rolling_cumprod_fixed,
    vams_batch_fixed,
    sortino_fast_fixed,
    sharpe_fast_fixed,
)


@dataclass
class ComparisonResult:
    """Result of a side-by-side comparison."""

    function_name: str
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    num_compared_values: int
    num_nan_mismatches: int
    tolerance_used: float
    details: Dict[str, Any]


class EquivalenceComparator:
    """Utility for comparing optimized and reference implementations."""

    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize comparator.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.comparison_results: List[ComparisonResult] = []

    def compare_arrays(
        self,
        optimized_result: np.ndarray,
        reference_result: np.ndarray,
        function_name: str,
        tolerance: Optional[float] = None,
    ) -> ComparisonResult:
        """
        Compare two arrays element-wise.

        Args:
            optimized_result: Result from optimized implementation
            reference_result: Result from reference implementation
            function_name: Name of function being tested
            tolerance: Override default tolerance

        Returns:
            ComparisonResult with detailed comparison information
        """
        if tolerance is None:
            tolerance = self.tolerance

        # Ensure arrays have same shape
        if optimized_result.shape != reference_result.shape:
            return ComparisonResult(
                function_name=function_name,
                passed=False,
                max_absolute_error=np.inf,
                max_relative_error=np.inf,
                num_compared_values=0,
                num_nan_mismatches=0,
                tolerance_used=tolerance,
                details={
                    "error": f"Shape mismatch: {optimized_result.shape} vs {reference_result.shape}"
                },
            )

        # Handle NaN values
        opt_nan_mask = np.isnan(optimized_result)
        ref_nan_mask = np.isnan(reference_result)

        # Count NaN mismatches
        nan_mismatches = np.sum(opt_nan_mask != ref_nan_mask)

        # Get finite values for comparison
        both_finite = ~opt_nan_mask & ~ref_nan_mask

        if not np.any(both_finite):
            # All values are NaN - check if NaN patterns match
            passed = nan_mismatches == 0
            return ComparisonResult(
                function_name=function_name,
                passed=passed,
                max_absolute_error=0.0 if passed else np.inf,
                max_relative_error=0.0 if passed else np.inf,
                num_compared_values=0,
                num_nan_mismatches=nan_mismatches,
                tolerance_used=tolerance,
                details={"note": "All values are NaN"},
            )

        # Compare finite values
        opt_finite = optimized_result[both_finite]
        ref_finite = reference_result[both_finite]

        # Calculate errors
        absolute_errors = np.abs(opt_finite - ref_finite)
        max_absolute_error = np.max(absolute_errors)

        # Relative errors (avoid division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            relative_errors = np.abs(absolute_errors / np.maximum(np.abs(ref_finite), 1e-15))
            max_relative_error = np.max(relative_errors)

        # Check if comparison passes
        passed = max_absolute_error <= tolerance and nan_mismatches == 0

        result = ComparisonResult(
            function_name=function_name,
            passed=passed,
            max_absolute_error=max_absolute_error,
            max_relative_error=max_relative_error,
            num_compared_values=len(opt_finite),
            num_nan_mismatches=nan_mismatches,
            tolerance_used=tolerance,
            details={
                "mean_absolute_error": np.mean(absolute_errors),
                "std_absolute_error": np.std(absolute_errors),
                "mean_relative_error": np.mean(relative_errors),
                "worst_absolute_indices": np.argsort(absolute_errors)[-5:].tolist(),
                "worst_relative_indices": np.argsort(relative_errors)[-5:].tolist(),
            },
        )

        self.comparison_results.append(result)
        return result

    def compare_rolling_std(
        self, data: np.ndarray, window: int, tolerance: Optional[float] = None
    ) -> ComparisonResult:
        """Compare rolling_std_fixed with pandas equivalent."""
        # Optimized implementation
        opt_result = rolling_std_fixed(data, window)

        # Pandas reference
        df = pd.DataFrame(data)
        ref_result = df.rolling(window).std().iloc[:, 0].values

        return self.compare_arrays(opt_result, ref_result, "rolling_std_fixed", tolerance)

    def compare_rolling_cumprod(
        self, data: np.ndarray, window: int, tolerance: Optional[float] = None
    ) -> ComparisonResult:
        """Compare rolling_cumprod_fixed with pandas equivalent."""
        # Optimized implementation
        opt_result = rolling_cumprod_fixed(data, window)

        # Pandas reference
        df = pd.DataFrame(1.0 + data)
        ref_result = df.rolling(window).apply(np.prod, raw=True).iloc[:, 0].values - 1.0

        return self.compare_arrays(opt_result, ref_result, "rolling_cumprod_fixed", tolerance)

    def compare_vams_batch(
        self, returns_matrix: np.ndarray, window: int, tolerance: Optional[float] = None
    ) -> ComparisonResult:
        """Compare vams_batch_fixed with pandas equivalent."""
        # Optimized implementation
        opt_result = vams_batch_fixed(returns_matrix, window)

        # Pandas reference
        df = pd.DataFrame(returns_matrix)
        momentum = (1 + df).rolling(window).apply(np.prod, raw=True) - 1
        volatility = df.rolling(window).std()  # Uses ddof=1
        ref_result = (momentum / volatility).values

        return self.compare_arrays(opt_result, ref_result, "vams_batch_fixed", tolerance)

    def compare_sharpe_batch(
        self,
        returns_matrix: np.ndarray,
        window: int,
        annualization_factor: float = 1.0,
        tolerance: Optional[float] = None,
    ) -> ComparisonResult:
        """Compare sharpe_fast_fixed with pandas equivalent."""
        # Optimized implementation
        opt_result = sharpe_fast_fixed(returns_matrix, window, annualization_factor)

        # Pandas reference
        df = pd.DataFrame(returns_matrix)
        rolling_mean = df.rolling(window).mean()
        rolling_std = df.rolling(window).std()  # Uses ddof=1
        ref_result = ((rolling_mean / rolling_std) * np.sqrt(annualization_factor)).values

        return self.compare_arrays(opt_result, ref_result, "sharpe_fast_fixed", tolerance)

    def compare_sortino_batch(
        self,
        returns_matrix: np.ndarray,
        window: int,
        target_return: float = 0.0,
        annualization_factor: float = 1.0,
        tolerance: Optional[float] = None,
    ) -> ComparisonResult:
        """Compare sortino_fast_fixed with manual pandas calculation."""
        # Optimized implementation
        opt_result = sortino_fast_fixed(returns_matrix, window, target_return, annualization_factor)

        # Manual pandas reference (more complex due to downside deviation)
        n_periods, n_assets = returns_matrix.shape
        ref_result = np.full_like(returns_matrix, np.nan)

        for asset in range(n_assets):
            for t in range(window - 1, n_periods):
                window_returns = returns_matrix[t - window + 1 : t + 1, asset]
                valid_returns = window_returns[~np.isnan(window_returns)]

                if len(valid_returns) >= window // 2:
                    mean_ret = np.mean(valid_returns)

                    downside_returns = valid_returns[valid_returns < target_return]
                    if len(downside_returns) > 1:
                        # Use ddof=1 for sample standard deviation
                        downside_dev = np.std(downside_returns, ddof=1)
                        if downside_dev > 1e-10:
                            sortino = ((mean_ret - target_return) / downside_dev) * np.sqrt(
                                annualization_factor
                            )
                            ref_result[t, asset] = sortino

        return self.compare_arrays(opt_result, ref_result, "sortino_fast_fixed", tolerance)

    def run_comprehensive_comparison(
        self, test_data: Dict[str, np.ndarray], window: int = 12
    ) -> Dict[str, ComparisonResult]:
        """
        Run comprehensive comparison across all functions.

        Args:
            test_data: Dictionary with test data arrays
            window: Window size for rolling calculations

        Returns:
            Dictionary of comparison results by function name
        """
        results = {}

        # Single-asset tests
        if "returns_1d" in test_data:
            data_1d = test_data["returns_1d"]

            results["rolling_std"] = self.compare_rolling_std(data_1d, window)
            results["rolling_cumprod"] = self.compare_rolling_cumprod(data_1d, window)

        # Multi-asset tests
        if "returns_2d" in test_data:
            data_2d = test_data["returns_2d"]

            results["vams_batch"] = self.compare_vams_batch(data_2d, window)
            results["sharpe_batch"] = self.compare_sharpe_batch(data_2d, window, 12.0)
            results["sortino_batch"] = self.compare_sortino_batch(data_2d, window, 0.0, 12.0)

        return results

    def generate_comparison_report(self) -> str:
        """Generate a detailed comparison report."""
        if not self.comparison_results:
            return "No comparisons performed."

        report_lines = [
            "=== EQUIVALENCE COMPARISON REPORT ===",
            f"Total comparisons: {len(self.comparison_results)}",
            f"Tolerance used: {self.tolerance}",
            "",
        ]

        passed_count = sum(1 for r in self.comparison_results if r.passed)
        failed_count = len(self.comparison_results) - passed_count

        report_lines.extend([f"PASSED: {passed_count}", f"FAILED: {failed_count}", ""])

        # Detailed results
        for result in self.comparison_results:
            status = "PASS" if result.passed else "FAIL"
            report_lines.extend(
                [
                    f"Function: {result.function_name} [{status}]",
                    f"  Max absolute error: {result.max_absolute_error:.2e}",
                    f"  Max relative error: {result.max_relative_error:.2e}",
                    f"  Values compared: {result.num_compared_values}",
                    f"  NaN mismatches: {result.num_nan_mismatches}",
                    "",
                ]
            )

            if not result.passed:
                report_lines.extend(
                    [
                        "  FAILURE DETAILS:",
                        f"    Mean absolute error: {result.details.get('mean_absolute_error', 'N/A'):.2e}",
                        f"    Std absolute error: {result.details.get('std_absolute_error', 'N/A'):.2e}",
                        "",
                    ]
                )

        return "\n".join(report_lines)

    def assert_all_passed(self):
        """Assert that all comparisons passed."""
        failed_results = [r for r in self.comparison_results if not r.passed]

        if failed_results:
            report = self.generate_comparison_report()
            pytest.fail(f"Equivalence tests failed:\n{report}")


class RegressionTester:
    """Utility for regression testing against known good results."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.baseline_results: Dict[str, np.ndarray] = {}

    def record_baseline(self, function_name: str, result: np.ndarray):
        """Record a baseline result for future regression testing."""
        self.baseline_results[function_name] = result.copy()

    def check_regression(self, function_name: str, current_result: np.ndarray) -> bool:
        """Check if current result matches recorded baseline."""
        if function_name not in self.baseline_results:
            raise ValueError(f"No baseline recorded for {function_name}")

        baseline = self.baseline_results[function_name]

        if baseline.shape != current_result.shape:
            return False

        # Handle NaN values
        baseline_nan = np.isnan(baseline)
        current_nan = np.isnan(current_result)

        # Check NaN patterns match
        if not np.array_equal(baseline_nan, current_nan):
            return False

        # Compare finite values
        both_finite = ~baseline_nan & ~current_nan
        if np.any(both_finite):
            max_error = np.max(np.abs(baseline[both_finite] - current_result[both_finite]))
            return max_error <= self.tolerance

        return True  # All NaN case

    def run_regression_suite(self, test_functions: Dict[str, Callable]) -> Dict[str, bool]:
        """Run regression tests for multiple functions."""
        results = {}

        for func_name, test_func in test_functions.items():
            try:
                current_result = test_func()
                results[func_name] = self.check_regression(func_name, current_result)
            except Exception as e:
                results[func_name] = False
                print(f"Regression test failed for {func_name}: {e}")

        return results


class PerformanceMonitor:
    """Monitor performance of optimized functions."""

    def __init__(self):
        self.performance_data: Dict[str, List[float]] = {}

    def time_function(
        self,
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any] = None,
        function_name: str = None,
        num_runs: int = 5,
    ) -> Tuple[Any, float]:
        """
        Time a function execution.

        Args:
            func: Function to time
            args: Function arguments
            kwargs: Function keyword arguments
            function_name: Name for recording
            num_runs: Number of runs for averaging

        Returns:
            Tuple of (result, average_time)
        """
        import time

        if kwargs is None:
            kwargs = {}

        if function_name is None:
            function_name = func.__name__

        times = []
        result = None

        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = np.mean(times)

        # Record performance data
        if function_name not in self.performance_data:
            self.performance_data[function_name] = []
        self.performance_data[function_name].append(avg_time)

        return result, avg_time

    def compare_performance(
        self,
        optimized_func: Callable,
        reference_func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any] = None,
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """
        Compare performance between optimized and reference implementations.

        Returns:
            Dictionary with timing results and speedup factor
        """
        if kwargs is None:
            kwargs = {}

        # Time optimized function
        _, opt_time = self.time_function(optimized_func, args, kwargs, "optimized", num_runs)

        # Time reference function
        _, ref_time = self.time_function(reference_func, args, kwargs, "reference", num_runs)

        speedup = ref_time / opt_time if opt_time > 0 else float("inf")

        return {"optimized_time": opt_time, "reference_time": ref_time, "speedup_factor": speedup}

    def generate_performance_report(self) -> str:
        """Generate performance monitoring report."""
        if not self.performance_data:
            return "No performance data collected."

        report_lines = ["=== PERFORMANCE MONITORING REPORT ===", ""]

        for func_name, times in self.performance_data.items():
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            report_lines.extend(
                [
                    f"Function: {func_name}",
                    f"  Average time: {avg_time:.6f}s",
                    f"  Std deviation: {std_time:.6f}s",
                    f"  Min time: {min_time:.6f}s",
                    f"  Max time: {max_time:.6f}s",
                    f"  Runs: {len(times)}",
                    "",
                ]
            )

        return "\n".join(report_lines)


class TestComparisonUtilities:
    """Test the comparison utilities themselves."""

    def test_equivalence_comparator(self):
        """Test EquivalenceComparator functionality."""
        comparator = EquivalenceComparator(tolerance=1e-10)

        # Test identical arrays
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])

        result = comparator.compare_arrays(arr1, arr2, "test_identical")
        assert result.passed
        assert result.max_absolute_error == 0.0
        assert result.num_compared_values == 3

        # Test arrays with small differences
        arr3 = np.array([1.0, 2.0, 3.0 + 1e-11])
        result2 = comparator.compare_arrays(arr1, arr3, "test_small_diff")
        assert result2.passed  # Within tolerance

        # Test arrays with large differences
        arr4 = np.array([1.0, 2.0, 4.0])
        result3 = comparator.compare_arrays(arr1, arr4, "test_large_diff")
        assert not result3.passed
        assert result3.max_absolute_error == 1.0

    def test_nan_handling(self):
        """Test NaN handling in comparisons."""
        comparator = EquivalenceComparator()

        # Arrays with matching NaN patterns
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([1.0, np.nan, 3.0])

        result = comparator.compare_arrays(arr1, arr2, "test_matching_nan")
        assert result.passed
        assert result.num_nan_mismatches == 0
        assert result.num_compared_values == 2  # Only finite values compared

        # Arrays with mismatched NaN patterns
        arr3 = np.array([1.0, 2.0, 3.0])
        result2 = comparator.compare_arrays(arr1, arr3, "test_mismatched_nan")
        assert not result2.passed
        assert result2.num_nan_mismatches == 1

    def test_rolling_std_comparison(self):
        """Test rolling standard deviation comparison."""
        comparator = EquivalenceComparator()

        # Test data
        data = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        window = 3

        result = comparator.compare_rolling_std(data, window)
        assert result.passed
        assert result.function_name == "rolling_std_fixed"

    def test_performance_monitor(self):
        """Test PerformanceMonitor functionality."""
        monitor = PerformanceMonitor()

        # Simple test function
        def test_func(x):
            return np.sum(x**2)

        test_data = np.random.random(1000)
        result, avg_time = monitor.time_function(test_func, (test_data,), num_runs=3)

        assert result is not None
        assert avg_time > 0
        assert "test_func" in monitor.performance_data
        assert len(monitor.performance_data["test_func"]) == 1

    def test_regression_tester(self):
        """Test RegressionTester functionality."""
        tester = RegressionTester(tolerance=1e-12)

        # Record baseline
        baseline = np.array([1.0, 2.0, 3.0])
        tester.record_baseline("test_function", baseline)

        # Test identical result
        identical = np.array([1.0, 2.0, 3.0])
        assert tester.check_regression("test_function", identical)

        # Test slightly different result (within tolerance)
        close = np.array([1.0, 2.0, 3.0 + 1e-13])
        assert tester.check_regression("test_function", close)

        # Test significantly different result
        different = np.array([1.0, 2.0, 4.0])
        assert not tester.check_regression("test_function", different)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
