"""
Risk-off Signal Generator Interface

Defines the abstract interface for risk-off signal generation following SOLID principles.
The interface provides clear abstractions for risk regime detection, supporting both simple
and complex risk-off signal implementations.

This interface follows the Dependency Inversion Principle (DIP) by:
- Depending on abstractions, not concretions
- Allowing high-level modules (strategies) to depend on abstractions
- Making low-level modules (signal generators) implement abstractions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pandas as pd


class IRiskOffSignalGenerator(ABC):
    """
    Interface for generating risk-off signals for trading strategies.

    This interface defines how strategies obtain risk regime signals,
    supporting both simple threshold-based and complex multi-factor risk-off detection.

    Signal Semantics:
    - True: Risk-off conditions detected (strategies should reduce/eliminate positions)
    - False: Risk-on conditions (normal strategy operation)

    Design Principles:
    - Single Responsibility: Only responsible for risk regime detection
    - Open/Closed: Open for extension (new implementations), closed for modification
    - Liskov Substitution: All implementations must be substitutable
    - Interface Segregation: Focused interface with minimal dependencies
    - Dependency Inversion: Depends on abstractions, not concrete implementations
    """

    @abstractmethod
    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Generate risk-off signal for the specified date.

        Args:
            all_historical_data: Historical OHLCV data for all universe assets up to current_date
            benchmark_historical_data: Historical OHLCV data for benchmark up to current_date
            non_universe_historical_data: Historical OHLCV data for non-universe assets
            current_date: Date for which to generate the signal

        Returns:
            bool: True if risk-off conditions detected, False for risk-on conditions

        Raises:
            ValueError: If required data is missing or invalid

        Note:
            Implementations should be stateless where possible, or manage state carefully
            for walk-forward optimization compatibility.
        """
        pass

    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration parameters for this signal generator.

        Returns:
            Dictionary containing configuration parameters used by this generator
        """
        pass

    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate configuration parameters for this signal generator.

        Args:
            config: Configuration parameters to validate

        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        pass

    def get_required_data_columns(self) -> List[str]:
        """
        Get list of required data columns for this signal generator.

        Returns:
            List of column names required in historical data (e.g., ['Close', 'Volume'])

        Note:
            Default implementation returns minimal requirements. Override for specific needs.
        """
        return ["Close"]

    def get_minimum_data_periods(self) -> int:
        """
        Get minimum number of historical periods required for signal generation.

        Returns:
            Minimum number of periods needed for reliable signal generation

        Note:
            Default implementation returns conservative estimate. Override for specific needs.
        """
        return 20  # Conservative default for most technical indicators

    def supports_non_universe_data(self) -> bool:
        """
        Check if this generator uses non-universe historical data.

        Returns:
            True if the generator requires non-universe data, False otherwise

        Note:
            Override this method if your implementation uses market-wide indicators
            like VIX, sector indices, or other external market data.
        """
        return False

    def get_signal_description(self) -> str:
        """
        Get human-readable description of this signal generator.

        Returns:
            String description of what this signal generator does
        """
        return f"{self.__class__.__name__}: Risk-off signal generator"
