"""Abstract interfaces for performance optimization components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class AbstractPerformanceOptimizer(ABC):
    """Abstract base class for performance optimizers."""

    @abstractmethod
    def optimize_trade_tracking(self, weights: Any, prices: Any, costs: Any) -> Dict[str, Any]:
        """Optimize trade tracking performance.

        Args:
            weights: Portfolio weights
            prices: Asset prices
            costs: Transaction costs

        Returns:
            Dictionary of optimized trade statistics
        """
        pass

    @abstractmethod
    def deduplicate_parameters(self, params: Dict[str, Any]) -> bool:
        """Check if parameters are duplicates.

        Args:
            params: Parameter dictionary to check

        Returns:
            True if parameters are duplicates, False otherwise
        """
        pass

    @abstractmethod
    def run_parallel_optimization(
        self, config: Dict[str, Any]
    ) -> Any:  # Should return OptimizationResult
        """Run optimization in parallel.

        Args:
            config: Optimization configuration

        Returns:
            Optimization result
        """
        pass


class AbstractTradeTracker(ABC):
    """Abstract base class for trade trackers."""

    @abstractmethod
    def track_trades_optimized(self, weights: Any, prices: Any, costs: Any) -> Dict[str, Any]:
        """Track trades with optimized performance.

        Args:
            weights: Portfolio weights
            prices: Asset prices
            costs: Transaction costs

        Returns:
            Dictionary of trade statistics
        """
        pass


class AbstractTrialDeduplicator(ABC):
    """Abstract base class for trial deduplication."""

    @abstractmethod
    def is_duplicate(self, params: Dict[str, Any]) -> bool:
        """Check if trial parameters are duplicates.

        Args:
            params: Parameter dictionary to check

        Returns:
            True if parameters are duplicates, False otherwise
        """
        pass

    @abstractmethod
    def add_trial(self, params: Dict[str, Any], result: Optional[float] = None) -> None:
        """Add a trial to the deduplication cache.

        Args:
            params: Parameter dictionary
            result: Optional trial result
        """
        pass

    @abstractmethod
    def get_cached_value(self, parameters: Dict[str, Any]) -> Optional[float]:
        """Get cached objective value for parameters if available.

        Args:
            parameters: Dictionary of parameter names to values

        Returns:
            Cached objective value or None if not available
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics.

        Returns:
            Dictionary with deduplication stats
        """
        pass


class AbstractParallelRunner(ABC):
    """Abstract base class for parallel runners."""

    @abstractmethod
    def run(self, config: Dict[str, Any]) -> Any:  # Should return OptimizationResult
        """Run optimization in parallel.

        Args:
            config: Runner configuration

        Returns:
            Optimization result
        """
        pass


class AbstractPerformanceOptimizerFactory(ABC):
    """Abstract factory for creating performance optimizers."""

    @abstractmethod
    def create_performance_optimizer(
        self, optimizer_type: str, config: Dict[str, Any]
    ) -> Optional[AbstractPerformanceOptimizer]:
        """Create a performance optimizer for a specific optimizer type.

        Args:
            optimizer_type: Type of optimizer ('optuna', 'genetic', etc.)
            config: Configuration dictionary

        Returns:
            Performance optimizer instance or None if not supported
        """
        pass
