"""
Adaptive batch sizing for genetic algorithm optimization.

This module provides utilities to dynamically adjust batch sizes for parallel
evaluation based on parameter space complexity and population diversity.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from loguru import logger


class AdaptiveBatchSizer:
    """
    Adaptively determines optimal batch size for parallel population evaluation.

    This class analyzes parameter space complexity, population diversity,
    and system capabilities to recommend optimal batch sizes for joblib.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 50,
        target_cpu_utilization: float = 0.85,
        adaptation_rate: float = 0.2,
        n_jobs: int = 1,
    ):
        """
        Initialize the adaptive batch sizer.

        Args:
            min_batch_size: Minimum batch size to recommend
            max_batch_size: Maximum batch size to recommend
            target_cpu_utilization: Target CPU utilization (0.0-1.0)
            adaptation_rate: Rate at which to adapt batch size (0.0-1.0)
            n_jobs: Number of parallel workers
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_cpu_utilization = min(1.0, max(0.1, target_cpu_utilization))
        self.adaptation_rate = min(1.0, max(0.01, adaptation_rate))
        self.n_jobs = max(1, n_jobs)

        # Internal state
        self._current_batch_size: int = 1
        self._last_population_complexity: float = 0.0
        self._complexity_history: List[float] = []
        self._batch_history: List[int] = []
        self._performance_history: List[float] = []

        # Initialize with reasonable defaults based on n_jobs
        if self.n_jobs <= 0:  # Auto mode (-1)
            import os

            cpu_count = os.cpu_count() or 1
            self._current_batch_size = min(max(1, cpu_count // 4), self.max_batch_size)
        else:
            self._current_batch_size = min(max(1, self.n_jobs // 2), self.max_batch_size)

        logger.debug(f"AdaptiveBatchSizer initialized with batch size: {self._current_batch_size}")

    def estimate_parameter_space_complexity(self, parameter_space: Dict[str, Any]) -> float:
        """
        Estimate the complexity of the parameter space.

        Higher complexity means more diverse individuals are possible.

        Args:
            parameter_space: Parameter space configuration

        Returns:
            Complexity score (higher means more complex)
        """
        if not parameter_space:
            return 1.0

        complexity_score = 0.0

        for param_name, config in parameter_space.items():
            param_type = config.get("type", "")

            if param_type == "int":
                # Integer range complexity
                low = config.get("low", 0)
                high = config.get("high", 1)
                step = config.get("step", 1)
                possible_values = max(1, (high - low) // step)
                complexity_score += min(100, possible_values)  # Cap at 100 to avoid overweighting

            elif param_type == "float":
                # Float range complexity (estimate possible distinct values)
                low = config.get("low", 0.0)
                high = config.get("high", 1.0)
                # Assume typical float precision gives us about 1000 distinct values in reasonable ranges
                possible_values = min(1000, max(100, int((high - low) * 100)))
                complexity_score += min(200, possible_values)  # Cap at 200

            elif param_type in ("categorical", "multi-categorical"):
                # Categorical complexity based on number of choices
                choices = config.get("choices", []) or config.get("values", [])
                complexity_score += len(choices) * (5 if param_type == "multi-categorical" else 1)

        # Normalize complexity score to 0.1-10.0 range
        normalized_score = max(0.1, min(10.0, complexity_score / max(1, len(parameter_space))))

        logger.debug(f"Parameter space complexity estimated at {normalized_score:.2f}")
        return normalized_score

    def analyze_population_diversity(self, population: List[Dict[str, Any]]) -> float:
        """
        Analyze population diversity to inform batch sizing.

        More diverse populations benefit from different batch sizing strategies.

        Args:
            population: Current GA population

        Returns:
            Diversity score (0.0-1.0)
        """
        if not population:
            return 0.0

        # Calculate parameter-wise variance as a simple diversity metric
        param_sets: Dict[str, List[Any]] = {}
        for individual in population:
            for param, value in individual.items():
                if param not in param_sets:
                    param_sets[param] = []
                param_sets[param].append(value)

        # Calculate normalized variance for each parameter
        variances = []
        for param, values in param_sets.items():
            if len(values) <= 1:
                continue

            # Handle numeric parameters
            if all(isinstance(v, (int, float)) for v in values):
                try:
                    variance = np.var(values)
                    # Normalize by parameter range
                    min_val = min(values)
                    max_val = max(values)
                    range_size = max(max_val - min_val, 1e-10)
                    normalized_variance = min(1.0, variance / (range_size**2))
                    variances.append(normalized_variance)
                except Exception:
                    pass
            # Handle categorical parameters
            else:
                # Use unique value ratio as diversity measure
                unique_values = len(set(str(v) for v in values))
                unique_ratio = unique_values / len(values)
                variances.append(unique_ratio)

                # Overall diversity is average of parameter variances
        diversity_score = sum(variances) / max(len(variances), 1)

        logger.debug(f"Population diversity score: {diversity_score:.4f}")
        return float(diversity_score)

    def update_batch_size(
        self,
        parameter_space: Dict[str, Any],
        population: List[Dict[str, Any]],
        execution_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update the recommended batch size based on current conditions.

        Args:
            parameter_space: Parameter space configuration
            population: Current population
            execution_time_ms: Last execution time in milliseconds (optional)

        Returns:
            Dictionary with batch size recommendations and metrics
        """
        # Calculate complexity and diversity metrics
        complexity = self.estimate_parameter_space_complexity(parameter_space)
        diversity = self.analyze_population_diversity(population)

        # Track history
        self._complexity_history.append(complexity)
        if len(self._complexity_history) > 10:
            self._complexity_history.pop(0)

        # Current complexity is smoothed over recent history
        current_complexity = sum(self._complexity_history) / len(self._complexity_history)

        # Factor in execution time if provided
        time_factor = 1.0
        if execution_time_ms is not None:
            self._performance_history.append(execution_time_ms)
            if len(self._performance_history) > 5:
                self._performance_history.pop(0)

            # If we have at least two performance measurements, check trend
            if len(self._performance_history) >= 2:
                prev_time = self._performance_history[-2]
                curr_time = self._performance_history[-1]

                # Adjust time factor based on performance trend
                if curr_time > prev_time * 1.2:  # 20% slower
                    time_factor = 0.8  # Reduce batch size
                elif curr_time < prev_time * 0.8:  # 20% faster
                    time_factor = 1.2  # Increase batch size

        # Calculate raw batch size recommendation based on complexity and diversity
        # Higher complexity → smaller batches
        # Higher diversity → smaller batches (more computation per individual)
        complexity_factor = 1.0 / max(0.1, min(10.0, current_complexity))
        diversity_factor = 1.0 / max(0.1, 1.0 + diversity * 2)

        # Base size scaled by complexity, diversity and time factors
        population_size = max(1, len(population))
        ideal_batch_count = min(self.n_jobs * 2, max(self.n_jobs, population_size // 2))
        base_size = max(1, population_size // ideal_batch_count)

        raw_batch_size = int(base_size * complexity_factor * diversity_factor * time_factor)

        # Apply constraints and smooth changes
        target_batch_size = max(self.min_batch_size, min(self.max_batch_size, raw_batch_size))

        # Smooth changes using adaptation rate
        if self._batch_history:
            prev_batch_size = self._batch_history[-1]
            self._current_batch_size = int(
                prev_batch_size * (1.0 - self.adaptation_rate)
                + target_batch_size * self.adaptation_rate
            )
        else:
            self._current_batch_size = target_batch_size

        # Ensure batch size is at least 1 and at most max_batch_size
        self._current_batch_size = max(
            self.min_batch_size, min(self.max_batch_size, self._current_batch_size)
        )

        # Track batch size history
        self._batch_history.append(self._current_batch_size)
        if len(self._batch_history) > 10:
            self._batch_history.pop(0)

        # Calculate batch count
        batch_count = max(
            1, (population_size + self._current_batch_size - 1) // self._current_batch_size
        )

        logger.debug(
            f"Adaptive batch size: {self._current_batch_size} "
            f"(complexity: {current_complexity:.2f}, diversity: {diversity:.2f})"
        )

        return {
            "batch_size": self._current_batch_size,
            "batch_count": batch_count,
            "complexity_score": current_complexity,
            "diversity_score": diversity,
            "time_factor": time_factor,
            "population_size": population_size,
        }

    def get_current_batch_size(self) -> int:
        """Get the current recommended batch size."""
        return self._current_batch_size

    def get_batch_size_as_joblib_param(self) -> Union[str, int]:
        """
        Get the current batch size formatted for joblib.

        Returns "auto" for special case, otherwise an integer.
        """
        # Special case: for low complexity, use "auto" for joblib's own heuristics
        if (
            self._complexity_history
            and sum(self._complexity_history) / len(self._complexity_history) < 0.5
        ):
            return "auto"
        return self._current_batch_size

    def reset(self) -> None:
        """Reset the internal state for a new optimization run."""
        self._complexity_history = []
        self._batch_history = []
        self._performance_history = []
