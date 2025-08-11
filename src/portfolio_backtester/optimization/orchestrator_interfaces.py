from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from portfolio_backtester.backtester_logic.backtester_facade import (
        BacktesterFacade,
    )
    from portfolio_backtester.optimization.optimization_result import (
        OptimizationResult,
    )


class OptimizationOrchestrator(ABC):
    """Abstract base class for optimization orchestrators."""

    @abstractmethod
    def optimize(
        self,
        scenario_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        data: Any,
        backtester: "BacktesterFacade",
    ) -> "OptimizationResult":
        """
        Run the optimization process.

        Args:
            scenario_config: The configuration for the backtesting scenario.
            optimization_config: The configuration for the optimization.
            data: The market data for the backtest.
            backtester: The backtester facade instance.

        Returns:
            The result of the optimization.
        """
        pass
