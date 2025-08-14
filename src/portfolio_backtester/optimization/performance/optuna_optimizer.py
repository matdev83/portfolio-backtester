"""
Optuna-specific performance optimizer implementation.

This module provides performance optimizations specifically tailored for
Optuna optimization engine while maintaining the abstract interface.
"""

import logging
from typing import Dict, Any, Optional
from .base_optimizer import BasePerformanceOptimizer
from .interfaces import (
    AbstractTradeTracker,
    AbstractTrialDeduplicator,
    AbstractParallelRunner,
)

logger = logging.getLogger(__name__)


class OptunaPerformanceOptimizer(BasePerformanceOptimizer):
    """
    Performance optimizer specifically designed for Optuna optimization engine.

    This implementation provides Optuna-specific optimizations including
    vectorized trade tracking, trial deduplication, and parallel execution.
    """

    def __init__(
        self,
        enable_vectorized_tracking: bool = True,
        enable_deduplication: bool = True,
        enable_parallel_execution: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            enable_vectorized_tracking=enable_vectorized_tracking,
            enable_deduplication=enable_deduplication,
            enable_parallel_execution=enable_parallel_execution,
            n_jobs=n_jobs,
        )

        # Initialize Optuna-specific components with precise optional typing
        self._trade_tracker: Optional[AbstractTradeTracker] = None
        self._deduplicator: Optional[AbstractTrialDeduplicator] = None
        self._parallel_runner: Optional[AbstractParallelRunner] = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OptunaPerformanceOptimizer initialized")

    def get_trade_tracker(self) -> Optional[AbstractTradeTracker]:
        """Get the trade tracker implementation for Optuna."""
        if not self.enable_vectorized_tracking:
            return None

        if self._trade_tracker is None:
            try:
                from .vectorized_tracking import VectorizedTradeTracker

                tracker: AbstractTradeTracker = VectorizedTradeTracker()
                self._trade_tracker = tracker

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Vectorized trade tracker created for Optuna")
            except ImportError as e:
                logger.warning(f"Failed to create vectorized trade tracker: {e}")
                return None

        return self._trade_tracker

    def get_deduplicator(self) -> Optional[AbstractTrialDeduplicator]:
        """Get the deduplicator implementation for Optuna."""
        if not self.enable_deduplication:
            return None

        if self._deduplicator is None:
            try:
                # OptunaTrialDeduplicator implements AbstractTrialDeduplicator
                dedup = OptunaTrialDeduplicator(enable_deduplication=True)
                self._deduplicator = dedup

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Optuna trial deduplicator created")
            except Exception as e:
                logger.warning(f"Failed to create Optuna deduplicator: {e}")
                return None

        return self._deduplicator

    def get_parallel_runner(self) -> Optional[AbstractParallelRunner]:
        """Get the parallel runner implementation for Optuna."""
        if not self.enable_parallel_execution or self.n_jobs <= 1:
            return None

        if self._parallel_runner is None:
            try:
                runner: AbstractParallelRunner = OptunaParallelRunner(n_jobs=self.n_jobs)
                self._parallel_runner = runner

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Optuna parallel runner created with {self.n_jobs} workers")
            except Exception as e:
                logger.warning(f"Failed to create Optuna parallel runner: {e}")
                return None

        return self._parallel_runner

    def supports_optimizer(self, optimizer_type: str) -> bool:
        """Check if this performance optimizer supports the given optimizer type."""
        return optimizer_type.lower() == "optuna"


class OptunaTrialDeduplicator(AbstractTrialDeduplicator):
    """
    Optuna-specific trial deduplicator that integrates with Optuna's trial system.
    """

    def __init__(self, enable_deduplication: bool = True):
        from .deduplication import BaseTrialDeduplicator

        # BaseTrialDeduplicator is concrete (not abstract); hold a typed instance
        self._base_deduplicator: BaseTrialDeduplicator = BaseTrialDeduplicator(enable_deduplication)

        # Optuna-specific tracking
        self._optuna_cache_hits: int = 0

    def is_duplicate(self, params: Dict[str, Any]) -> bool:
        """Check if parameters are duplicate using base implementation."""
        return self._base_deduplicator.is_duplicate(params)

    def add_trial(self, params: Dict[str, Any], result: Optional[float] = None) -> None:
        """Add a trial to the cache using base implementation."""
        # The base deduplicator registers parameters/results; map to its API
        # It may expose register_parameters; fall back to is_duplicate semantics if absent.
        # Prefer a common add-like API; fall back gracefully if unavailable.
        if hasattr(self._base_deduplicator, "add_trial"):
            self._base_deduplicator.add_trial(params, result)  # runtime attribute check above
        elif hasattr(self._base_deduplicator, "register"):
            self._base_deduplicator.register(params, result)  # runtime attribute check above
        else:
            # Fallback: no-op if base does not expose an add/register API
            pass

    def get_cached_value(self, parameters: Dict[str, Any]) -> Optional[float]:
        """Get cached value and track Optuna-specific cache hits."""
        cached_value = self._base_deduplicator.get_cached_value(parameters)

        if cached_value is not None:
            self._optuna_cache_hits += 1
            logger.info(f"Trial: Using cached value {cached_value:.6f} for duplicate parameters")

        return cached_value

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics including Optuna-specific metrics."""
        base_stats = self._base_deduplicator.get_stats()
        base_stats["optuna_cache_hits"] = self._optuna_cache_hits
        return base_stats


class OptunaParallelRunner(AbstractParallelRunner):
    """
    Optuna-specific parallel runner that uses the existing ParallelOptimizationRunner.

    This class adapts the existing Optuna parallel implementation to the new
    abstract interface while maintaining all existing functionality.
    """

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"OptunaParallelRunner initialized with {n_jobs} workers")

    def _get_default_optuna_storage_url(self) -> str:
        """Get the default Optuna storage URL."""
        from ...constants import DEFAULT_OPTUNA_STORAGE_URL

        return DEFAULT_OPTUNA_STORAGE_URL

    def run(self, config: Dict[str, Any]) -> Any:
        """Run optimization in parallel using existing implementation via ParallelOptimizationRunner."""
        try:
            from ..parallel_optimization_runner import ParallelOptimizationRunner

            # Expecting config to carry required keys
            scenario_config: Dict[str, Any] = config["scenario_config"]
            optimization_config: Dict[str, Any] = config["optimization_config"]
            data: Any = config["data"]
            n_jobs: int = int(config.get("n_jobs") or self.n_jobs)
            runner = ParallelOptimizationRunner(
                scenario_config=scenario_config,
                optimization_config=optimization_config,
                data=data,
                n_jobs=n_jobs,
                storage_url=optimization_config.get(
                    "storage_url", self._get_default_optuna_storage_url()
                ),
                enable_deduplication=True,
            )
            return runner.run()
        except KeyError as e:
            logger.error(f"Missing configuration key for parallel run: {e}")
            raise
        except ImportError as e:
            logger.error(f"Failed to import ParallelOptimizationRunner: {e}")
            raise
        except Exception as e:
            logger.error(f"Parallel optimization failed: {e}")
            raise

    def supports_optimizer(self, optimizer_type: str) -> bool:
        """Check if this runner supports the given optimizer type."""
        return optimizer_type.lower() == "optuna"


# Wrapper class to integrate with existing Optuna objective adapter
class OptunaDedupObjectiveWrapper:
    """
    Wrapper that integrates the new deduplication system with existing Optuna objectives.

    This maintains compatibility with the existing OptunaObjectiveAdapter while
    using the new abstract deduplication interface.
    """

    def __init__(self, base_objective, deduplicator: Optional[AbstractTrialDeduplicator] = None):
        self.base_objective = base_objective
        self.deduplicator = deduplicator

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"OptunaDedupObjectiveWrapper created (deduplication: {'enabled' if deduplicator else 'disabled'})"
            )

    def __call__(self, trial):
        """Evaluate trial with deduplication if available."""
        # Extract parameters from trial
        if hasattr(self.base_objective, "_trial_to_params"):
            params = self.base_objective._trial_to_params(trial)
        else:
            params = trial.params.copy()

        # Check for cached value if deduplicator is available
        if self.deduplicator:
            cached_value = self.deduplicator.get_cached_value(params)
            if cached_value is not None:
                return cached_value

        # Evaluate using base objective
        objective_value = self.base_objective(trial)

        # Register the result if deduplicator is available
        if self.deduplicator:
            self.deduplicator.add_trial(params, objective_value)

        return objective_value
