"""
Benefit estimation for parallel WFO processing.

This module provides calculations for estimating the performance benefits
of parallel processing vs sequential processing for WFO windows.

This implementation now supports Dependency Inversion Principle (DIP) by
accepting interfaces for mathematical operations dependencies.
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ParallelBenefitEstimator:
    """
    Estimates the potential performance benefits of parallel processing.

    This class is responsible for:
    - Calculating timing estimates for sequential vs parallel execution
    - Estimating speedup ratios and time savings
    - Providing overhead estimates for process management
    - Helping determine when parallel processing is beneficial
    """

    def __init__(
        self,
        max_workers: int,
        parallel_overhead: float = 0.1,
        math_operations: Optional[Any] = None,
    ):
        """
        Initialize the benefit estimator.

        Args:
            max_workers: Maximum number of worker processes available
            parallel_overhead: Estimated overhead for parallel processing (default: 10%)
            math_operations: Optional math operations interface (DIP)
        """
        self.max_workers = max_workers
        self.parallel_overhead = parallel_overhead

        # Initialize math operations dependency (DIP)
        if math_operations is not None:
            self._math_operations = math_operations
        else:
            # Import here to avoid circular imports
            from ...interfaces.math_operations_interface import create_math_operations

            self._math_operations = create_math_operations()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"ParallelBenefitEstimator initialized: max_workers={max_workers}, overhead={parallel_overhead}"
            )

    def estimate_parallel_benefit(
        self, num_windows: int, avg_window_time: float
    ) -> Dict[str, float]:
        """
        Estimate the potential benefit of parallel processing.

        Args:
            num_windows: Number of WFO windows to process
            avg_window_time: Average time per window in seconds

        Returns:
            Dictionary with comprehensive timing estimates including:
            - sequential_time: Total time for sequential processing
            - estimated_parallel_time: Estimated time for parallel processing
            - estimated_speedup: Speedup ratio (sequential/parallel)
            - estimated_time_saved: Absolute time savings in seconds
            - max_workers: Number of worker processes used
            - parallel_overhead: Overhead factor applied
            - efficiency: Parallel efficiency (speedup / max_workers)
        """
        sequential_time = num_windows * avg_window_time

        # Calculate ideal parallel time (no overhead)
        ideal_parallel_time = sequential_time / self.max_workers

        # Apply overhead for process management, communication, and coordination
        estimated_parallel_time = ideal_parallel_time * (1 + self.parallel_overhead)

        # Calculate performance metrics
        speedup = sequential_time / estimated_parallel_time if estimated_parallel_time > 0 else 1.0
        time_saved = sequential_time - estimated_parallel_time
        efficiency = speedup / self.max_workers if self.max_workers > 0 else 0.0

        # Calculate break-even analysis
        overhead_time = estimated_parallel_time - ideal_parallel_time
        break_even_windows = self._calculate_break_even_windows(avg_window_time)

        result = {
            "sequential_time": sequential_time,
            "estimated_parallel_time": estimated_parallel_time,
            "ideal_parallel_time": ideal_parallel_time,
            "estimated_speedup": speedup,
            "estimated_time_saved": time_saved,
            "parallel_efficiency": efficiency,
            "max_workers": self.max_workers,
            "parallel_overhead": self.parallel_overhead,
            "overhead_time": overhead_time,
            "break_even_windows": break_even_windows,
            "num_windows": num_windows,
            "avg_window_time": avg_window_time,
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Benefit estimation: {num_windows} windows, "
                f"{speedup:.2f}x speedup, {time_saved:.1f}s saved, "
                f"{efficiency:.2f} efficiency"
            )

        return result

    def is_parallel_beneficial(
        self,
        num_windows: int,
        avg_window_time: float,
        min_speedup: float = 1.5,
        min_time_saved: float = 5.0,
    ) -> bool:
        """
        Determine if parallel processing is beneficial based on estimates.

        Args:
            num_windows: Number of windows to process
            avg_window_time: Average time per window in seconds
            min_speedup: Minimum speedup required to be considered beneficial
            min_time_saved: Minimum time savings in seconds to be considered beneficial

        Returns:
            True if parallel processing is estimated to be beneficial
        """
        estimates = self.estimate_parallel_benefit(num_windows, avg_window_time)

        speedup_beneficial = estimates["estimated_speedup"] >= min_speedup
        time_saved_beneficial = estimates["estimated_time_saved"] >= min_time_saved
        above_break_even = num_windows >= estimates["break_even_windows"]

        is_beneficial = speedup_beneficial and time_saved_beneficial and above_break_even

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Parallel processing beneficial: {is_beneficial} "
                f"(speedup: {speedup_beneficial}, time_saved: {time_saved_beneficial}, "
                f"break_even: {above_break_even})"
            )

        return is_beneficial

    def _calculate_break_even_windows(self, avg_window_time: float) -> int:
        """
        Calculate the minimum number of windows where parallel processing breaks even.

        Args:
            avg_window_time: Average time per window in seconds

        Returns:
            Minimum number of windows for break-even point
        """
        if avg_window_time <= 0:
            return 1

        # Break-even occurs when speedup overcomes overhead
        # Sequential time = N * avg_time
        # Parallel time = (N * avg_time / workers) * (1 + overhead)
        # Break-even when sequential >= parallel
        # N * avg_time >= (N * avg_time / workers) * (1 + overhead)
        # Simplifying: workers >= (1 + overhead)
        # For typical overhead values, this is usually satisfied
        # But we also need enough work to justify process startup costs

        # Empirical formula: overhead cost divided by per-window savings
        per_window_savings = avg_window_time * (1 - 1 / self.max_workers)
        overhead_cost = avg_window_time * self.parallel_overhead

        if per_window_savings <= 0:
            return 999999  # Never beneficial - return very large int

        break_even = self._math_operations.max_value(2, int(overhead_cost / per_window_savings) + 1)
        return int(break_even)

    def generate_performance_report(self, num_windows: int, avg_window_time: float) -> str:
        """
        Generate a human-readable performance report.

        Args:
            num_windows: Number of windows to process
            avg_window_time: Average time per window in seconds

        Returns:
            Formatted performance report string
        """
        estimates = self.estimate_parallel_benefit(num_windows, avg_window_time)

        report_lines = [
            "Parallel Processing Performance Estimate:",
            "========================================",
            f"Windows to process: {num_windows}",
            f"Average window time: {avg_window_time:.2f}s",
            f"Max workers: {self.max_workers}",
            "",
            "Timing Estimates:",
            f"  Sequential time: {estimates['sequential_time']:.1f}s ({estimates['sequential_time']/60:.1f}m)",
            f"  Parallel time: {estimates['estimated_parallel_time']:.1f}s ({estimates['estimated_parallel_time']/60:.1f}m)",
            f"  Ideal parallel time: {estimates['ideal_parallel_time']:.1f}s",
            f"  Overhead time: {estimates['overhead_time']:.1f}s",
            "",
            "Performance Metrics:",
            f"  Estimated speedup: {estimates['estimated_speedup']:.2f}x",
            f"  Time saved: {estimates['estimated_time_saved']:.1f}s ({estimates['estimated_time_saved']/60:.1f}m)",
            f"  Parallel efficiency: {estimates['parallel_efficiency']:.1%}",
            f"  Break-even windows: {estimates['break_even_windows']}",
            "",
            f"Recommendation: {'Use parallel processing' if self.is_parallel_beneficial(num_windows, avg_window_time) else 'Use sequential processing'}",
        ]

        return "\n".join(report_lines)
