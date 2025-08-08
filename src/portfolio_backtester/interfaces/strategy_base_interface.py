"""
Strategy Base Interface

Provides abstract interface for strategy base functionality to eliminate direct super() calls
and enable polymorphic strategy composition instead of inheritance.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..timing.timing_controller import TimingController
    from ..risk_off_signals import IRiskOffSignalGenerator
    from ..risk_management.stop_loss_handlers import BaseStopLoss


class IStrategyBase(ABC):
    """
    Abstract interface for strategy base functionality.

    This interface defines the contract for strategy base operations,
    allowing strategies to use composition instead of inheritance hierarchies.
    """

    @abstractmethod
    def initialize_strategy(self, strategy_config: Dict[str, Any]) -> None:
        """
        Initialize the strategy with configuration parameters.

        Args:
            strategy_config: Dictionary containing strategy configuration
        """
        pass

    @abstractmethod
    def get_timing_controller(self) -> Optional["TimingController"]:
        """Get the timing controller for this strategy."""
        pass

    @abstractmethod
    def supports_daily_signals(self) -> bool:
        """
        Determine if strategy supports daily signals based on timing controller.
        """
        pass

    @abstractmethod
    def get_risk_off_signal_generator(self) -> Optional["IRiskOffSignalGenerator"]:
        """Get the risk-off signal generator instance if configured."""
        pass

    @abstractmethod
    def get_stop_loss_handler(self) -> "BaseStopLoss":
        """Get the stop loss handler instance."""
        pass

    @abstractmethod
    def get_universe(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get the universe of assets for this strategy.

        Args:
            global_config: Global configuration dictionary

        Returns:
            List of (ticker, weight) tuples
        """
        pass

    @abstractmethod
    def get_universe_method_with_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """
        Get the universe of assets for this strategy with date context.

        Args:
            global_config: Global configuration dictionary
            current_date: Current date for universe resolution

        Returns:
            List of (ticker, weight) tuples
        """
        pass

    @abstractmethod
    def get_non_universe_data_requirements(self) -> List[str]:
        """
        Returns a list of tickers that are not part of the trading universe
        but are required for the strategy's calculations.
        """
        pass

    @abstractmethod
    def get_synthetic_data_requirements(self) -> bool:
        """
        Returns a boolean indicating whether the strategy requires synthetic data generation.
        """
        pass

    @abstractmethod
    def get_minimum_required_periods(self) -> int:
        """
        Calculate the minimum number of periods (months) of historical data required
        for this strategy to function properly.

        Returns:
            int: Minimum number of months of historical data required
        """
        pass

    @abstractmethod
    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Tuple[bool, str]:
        """
        Validates that there is sufficient historical data available for the strategy
        to perform reliable calculations as of the current_date.

        Args:
            all_historical_data: DataFrame with historical data for universe assets
            benchmark_historical_data: DataFrame with historical data for benchmark
            current_date: The date for which we're checking data sufficiency

        Returns:
            tuple[bool, str]: (is_sufficient, reason_if_not)
        """
        pass

    @abstractmethod
    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ) -> List[str]:
        """
        Filter the universe to only include assets that have sufficient historical data
        as of the current date.

        Args:
            all_historical_data: DataFrame with historical data for universe assets
            current_date: The date for which we're checking data availability
            min_periods_override: Override minimum periods requirement

        Returns:
            list: List of assets that have sufficient data
        """
        pass


class StrategyBaseAdapter(IStrategyBase):
    """
    Adapter that wraps BaseStrategy functionality to implement IStrategyBase interface.

    This allows strategies to use composition instead of inheritance while maintaining
    access to all base strategy functionality.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """Initialize the adapter with strategy configuration."""
        # Import here to avoid circular imports
        from ..strategies.base.base_strategy import BaseStrategy

        # Create a concrete implementation of BaseStrategy for delegation
        class ConcreteBaseStrategy(BaseStrategy):
            def generate_signals(self, *args, **kwargs) -> pd.DataFrame:
                # This method won't be called through the adapter
                return pd.DataFrame()

        self._base_strategy = ConcreteBaseStrategy(strategy_config)

    def initialize_strategy(self, strategy_config: Dict[str, Any]) -> None:
        """Initialize the strategy with configuration parameters."""
        # Re-initialize if needed
        self._base_strategy.strategy_config = strategy_config
        self._base_strategy.strategy_params = strategy_config

    def get_timing_controller(self) -> Optional["TimingController"]:
        """Get the timing controller for this strategy."""
        return self._base_strategy.get_timing_controller()

    def supports_daily_signals(self) -> bool:
        """Determine if strategy supports daily signals based on timing controller."""
        return self._base_strategy.supports_daily_signals()

    def get_risk_off_signal_generator(self) -> Optional["IRiskOffSignalGenerator"]:
        """Get the risk-off signal generator instance if configured."""
        from ..risk_off_signals import IRiskOffSignalGenerator

        result = self._base_strategy.get_risk_off_signal_generator()
        return result if isinstance(result, IRiskOffSignalGenerator) or result is None else None

    def get_stop_loss_handler(self) -> "BaseStopLoss":
        """Get the stop loss handler instance."""
        return self._base_strategy.get_stop_loss_handler()

    def get_universe(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get the universe of assets for this strategy."""
        return self._base_strategy.get_universe(global_config)

    def get_universe_method_with_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """Get the universe of assets for this strategy with date context."""
        return self._base_strategy.get_universe_method_with_date(global_config, current_date)

    def get_non_universe_data_requirements(self) -> List[str]:
        """Returns a list of tickers required but not part of the trading universe."""
        return self._base_strategy.get_non_universe_data_requirements()

    def get_synthetic_data_requirements(self) -> bool:
        """Returns whether the strategy requires synthetic data generation."""
        return self._base_strategy.get_synthetic_data_requirements()

    def get_minimum_required_periods(self) -> int:
        """Calculate the minimum number of periods of historical data required."""
        return self._base_strategy.get_minimum_required_periods()

    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Tuple[bool, str]:
        """Validates that there is sufficient historical data available."""
        result = self._base_strategy.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        # Ensure we return the correct type
        if isinstance(result, tuple) and len(result) == 2:
            return (bool(result[0]), str(result[1]))
        return (False, "Invalid validation result")

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ) -> List[str]:
        """Filter the universe to only include assets with sufficient historical data."""
        return self._base_strategy.filter_universe_by_data_availability(
            all_historical_data, current_date, min_periods_override
        )

    @property
    def strategy_config(self) -> Dict[str, Any]:
        """Get the strategy configuration."""
        return self._base_strategy.strategy_config

    @property
    def strategy_params(self) -> Dict[str, Any]:
        """Get the strategy parameters."""
        return self._base_strategy.strategy_params


class StrategyBaseFactory:
    """
    Factory for creating strategy base implementations.

    This factory eliminates the need for strategies to know about specific
    base strategy implementations, promoting loose coupling.
    """

    @staticmethod
    def create_strategy_base(strategy_config: Dict[str, Any]) -> IStrategyBase:
        """
        Create a strategy base implementation.

        Args:
            strategy_config: Strategy configuration dictionary

        Returns:
            IStrategyBase implementation
        """
        return StrategyBaseAdapter(strategy_config)
