"""
Signal Generator Interface

Provides abstract interface for all signal generation strategies to eliminate isinstance violations
and enable polymorphic signal generation across different strategy types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
import pandas as pd


class ISignalGenerator(ABC):
    """
    Abstract interface for signal generators.

    This interface defines the contract for all signal generation implementations,
    allowing strategies to use composition instead of complex inheritance hierarchies.
    """

    @abstractmethod
    def generate_signals_for_range(
        self,
        data: Dict[str, Any],
        universe_tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate signals for a full date range using vectorized operations.

        Args:
            data: Dictionary containing all necessary data for signal generation
            universe_tickers: List of universe ticker symbols
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with signals for each date and ticker
        """
        pass

    @abstractmethod
    def generate_signal_for_date(
        self, data: Dict[str, Any], universe_tickers: List[str], current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Generate signal for a single date with state management.

        Args:
            data: Dictionary containing all necessary data for signal generation
            universe_tickers: List of universe ticker symbols
            current_date: Current trading date

        Returns:
            DataFrame with signals for current date
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal state for new backtest runs."""
        pass

    @abstractmethod
    def is_in_position(self) -> bool:
        """Check if currently in a position."""
        pass

    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        pass


class SignalGeneratorFactory:
    """
    Factory for creating signal generators.

    This factory eliminates the need for strategies to know about specific
    signal generator implementations, promoting loose coupling.
    """

    def __init__(self):
        self._generators: Dict[str, Type[ISignalGenerator]] = {}

    def register_generator(self, name: str, generator_class: Type[ISignalGenerator]) -> None:
        """Register a signal generator implementation."""
        self._generators[name] = generator_class

    def create_generator(self, name: str, config: Dict[str, Any]) -> ISignalGenerator:
        """Create a signal generator by name."""
        if name not in self._generators:
            raise ValueError(
                f"Unknown signal generator: {name}. Available: {list(self._generators.keys())}"
            )

        generator_class = self._generators[name]
        return generator_class()

    def get_available_generators(self) -> List[str]:
        """Get list of available signal generator names."""
        return list(self._generators.keys())


# Global factory instance
signal_generator_factory = SignalGeneratorFactory()

# Alias for backwards compatibility
SignalGeneratorProvider = ISignalGenerator
