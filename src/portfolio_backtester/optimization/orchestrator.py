"""
Optimization orchestrator that coordinates the optimization process.

This module implements the OptimizationOrchestrator class that coordinates
between parameter generators and backtesting evaluators. It manages the
optimization lifecycle, progress tracking, timeout, and early stopping logic.
"""

import logging
import time
from typing import Any, Dict, Optional

from .results import OptimizationResult, OptimizationData
from .evaluator import BacktestEvaluator
from .parameter_generator import ParameterGenerator

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Unified progress tracking system for optimization.

    This class provides progress tracking, timeout handling, and early stopping
    logic that works across all optimization backends.

    Attributes:
        start_time: Time when optimization started
        timeout_seconds: Maximum time allowed for optimization
        early_stop_patience: Number of evaluations without improvement before stopping
        best_value: Best objective value seen so far
        evaluations_without_improvement: Counter for early stopping
        total_evaluations: Total number of parameter evaluations performed
        is_multi_objective: Whether this is multi-objective optimization
    """

    def __init__(
        self,
        timeout_seconds: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
        is_multi_objective: bool = False,
    ):
        """Initialize the progress tracker.

        Args:
            timeout_seconds: Maximum time allowed for optimization (None for no timeout)
            early_stop_patience: Number of evaluations without improvement before stopping
            is_multi_objective: Whether this is multi-objective optimization
        """
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds
        self.early_stop_patience = early_stop_patience
        self.is_multi_objective = is_multi_objective

        # Progress tracking state
        self.best_value = None
        self.evaluations_without_improvement = 0
        self.total_evaluations = 0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"ProgressTracker initialized: timeout={timeout_seconds}s, "
                f"patience={early_stop_patience}, multi_objective={is_multi_objective}"
            )

    def should_stop(self) -> bool:
        """Check if optimization should stop due to timeout or early stopping.

        Returns:
            True if optimization should stop, False otherwise
        """
        # Check timeout
        if self.timeout_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.timeout_seconds:
                logger.info(f"Optimization stopped due to timeout ({elapsed:.1f}s)")
                return True

        # Check early stopping
        if (
            self.early_stop_patience is not None
            and self.evaluations_without_improvement >= self.early_stop_patience
        ):
            logger.info(
                f"Optimization stopped due to early stopping "
                f"({self.evaluations_without_improvement} evaluations without improvement)"
            )
            return True

        return False

    def update_progress(self, objective_value: Any) -> None:
        """Update progress tracking with a new evaluation result.

        Args:
            objective_value: The objective value from the latest evaluation
        """
        self.total_evaluations += 1

        # Check if this is an improvement
        is_improvement = self._is_improvement(objective_value)

        if is_improvement:
            self.best_value = objective_value
            self.evaluations_without_improvement = 0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"New best value: {objective_value}")
        else:
            self.evaluations_without_improvement += 1

        # Log progress periodically
        if self.total_evaluations % 10 == 0:
            elapsed = time.time() - self.start_time
            logger.info(
                f"Evaluation {self.total_evaluations}: "
                f"best={self.best_value}, elapsed={elapsed:.1f}s"
            )

    def _is_improvement(self, objective_value: Any) -> bool:
        """Check if the given objective value is an improvement.

        Args:
            objective_value: The objective value to check

        Returns:
            True if this is an improvement over the current best
        """
        if self.best_value is None:
            return True

        if self.is_multi_objective:
            # For multi-objective, we consider it an improvement if any objective improved
            # This is a simplified approach - more sophisticated methods could be used
            if isinstance(objective_value, (list, tuple)) and isinstance(
                self.best_value, (list, tuple)
            ):
                for new_val, best_val in zip(objective_value, self.best_value):
                    if new_val > best_val:
                        return True
            return False
        else:
            # For single objective, simple comparison (assuming maximization)
            return objective_value > self.best_value

    def get_status(self) -> Dict[str, Any]:
        """Get current optimization status.

        Returns:
            Dictionary containing current optimization status
        """
        elapsed = time.time() - self.start_time
        return {
            "total_evaluations": self.total_evaluations,
            "best_value": self.best_value,
            "elapsed_seconds": elapsed,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "timeout_seconds": self.timeout_seconds,
            "early_stop_patience": self.early_stop_patience,
        }


class OptimizationOrchestrator:
    """Coordinates optimization process between parameter generators and evaluators.

    This class manages the optimization lifecycle, coordinating between parameter
    generators and backtesting evaluators. It provides unified progress tracking,
    timeout handling, and early stopping logic that works across all optimization
    backends.

    Attributes:
        parameter_generator: The parameter generator to use for optimization
        evaluator: The BacktestEvaluator for parameter evaluation
        progress_tracker: Progress tracking and early stopping logic
    """

    def __init__(
        self,
        parameter_generator: ParameterGenerator,
        evaluator: BacktestEvaluator,
        timeout_seconds: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
    ):
        """Initialize the optimization orchestrator.

        Args:
            parameter_generator: Parameter generator implementing the ParameterGenerator protocol
            evaluator: BacktestEvaluator instance for parameter evaluation
            timeout_seconds: Maximum time allowed for optimization
            early_stop_patience: Number of evaluations without improvement before stopping
        """
        self.parameter_generator = parameter_generator
        self.evaluator = evaluator

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            timeout_seconds=timeout_seconds,
            early_stop_patience=early_stop_patience,
            is_multi_objective=evaluator.is_multi_objective,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OptimizationOrchestrator initialized")

    def optimize(
        self,
        scenario_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        data: OptimizationData,
        backtester: Any,  # StrategyBacktester - avoiding circular import
    ) -> OptimizationResult:
        """Run the complete optimization process.

        This method coordinates the entire optimization process, managing the
        interaction between parameter generation and evaluation while tracking
        progress and handling timeouts and early stopping.

        Args:
            scenario_config: Scenario configuration including strategy and base parameters
            optimization_config: Optimization-specific configuration
            data: OptimizationData containing price data and walk-forward windows
            backtester: StrategyBacktester instance for evaluation

        Returns:
            OptimizationResult: Final optimization results with best parameters
        """
        logger.info("Starting optimization process")

        # Initialize the parameter generator
        self.parameter_generator.initialize(scenario_config, optimization_config)

        try:
            # Main optimization loop
            while (
                not self.parameter_generator.is_finished()
                and not self.progress_tracker.should_stop()
            ):
                # Get next parameter set to evaluate
                parameters = self.parameter_generator.suggest_parameters()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Evaluating parameters: {parameters}")

                # Evaluate the parameters
                evaluation_result = self.evaluator.evaluate_parameters(
                    parameters, scenario_config, data, backtester
                )

                # Update progress tracking
                self.progress_tracker.update_progress(evaluation_result.objective_value)

                # Report result back to parameter generator
                self.parameter_generator.report_result(parameters, evaluation_result)

            # Get final results
            final_result = self.parameter_generator.get_best_result()

            # Log final status
            status = self.progress_tracker.get_status()
            logger.info(
                f"Optimization completed: {status['total_evaluations']} evaluations "
                f"in {status['elapsed_seconds']:.1f}s, best value: {status['best_value']}"
            )

            return final_result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Try to get partial results if possible
            try:
                return self.parameter_generator.get_best_result()
            except Exception:
                # Return empty result if we can't get anything
                return OptimizationResult(
                    best_parameters={},
                    best_value=(
                        -1e9
                        if not self.evaluator.is_multi_objective
                        else [-1e9] * len(self.evaluator.metrics_to_optimize)
                    ),
                    n_evaluations=self.progress_tracker.total_evaluations,
                    optimization_history=[],
                )

    def get_progress_status(self) -> Dict[str, Any]:
        """Get current optimization progress status.

        Returns:
            Dictionary containing current optimization status
        """
        return self.progress_tracker.get_status()
