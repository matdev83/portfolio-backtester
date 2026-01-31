from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Union

from portfolio_backtester.backtester_logic.backtester_facade import Backtester

if TYPE_CHECKING:
    from portfolio_backtester.optimization.results import (
        OptimizationResult,
    )
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

# Alias for backward compatibility
BacktesterFacade = Backtester


class OptimizationOrchestrator(ABC):
    """Abstract base class for optimization orchestrators."""

    @abstractmethod
    def optimize(
        self,
        scenario_config: Union[Dict[str, Any], "CanonicalScenarioConfig"],
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
