import logging
import time
from typing import Any, Dict, Optional

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
