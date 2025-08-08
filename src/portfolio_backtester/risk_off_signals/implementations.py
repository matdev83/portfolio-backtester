"""
Risk-off Signal Generator Implementations

Provides concrete implementations of the IRiskOffSignalGenerator interface.
Includes both production-ready and testing implementations following SOLID principles.
"""

from typing import Dict, Any, List
import logging

import pandas as pd

from .interface import IRiskOffSignalGenerator

logger = logging.getLogger(__name__)


class NoRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """
    Default risk-off signal generator that never signals risk-off conditions.

    This implementation follows the principle of least surprise - it never
    triggers risk-off signals, allowing strategies to operate normally.
    This is the appropriate default behavior for most strategies.

    Design Pattern: Null Object Pattern
    - Provides a default "do nothing" implementation
    - Eliminates the need for null checks in client code
    - Always returns False (risk-on conditions)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize the no-risk-off signal generator.

        Args:
            config: Configuration parameters (unused but accepted for interface compatibility)
        """
        self._config = config or {}

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Generate risk-off signal - always returns False (risk-on).

        This implementation never signals risk-off conditions, allowing
        strategies to operate without risk regime interference.

        Returns:
            bool: Always False (risk-on conditions)
        """
        return False  # Never signal risk-off

    def get_configuration(self) -> Dict[str, Any]:
        """Get configuration - returns empty dict as no parameters are used."""
        return dict(self._config)

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration - always valid as no parameters are required."""
        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        """Get required data columns - minimal requirements as signal is not data-dependent."""
        return []  # No data required for null implementation

    def get_minimum_data_periods(self) -> int:
        """Get minimum data periods - zero as no historical data is analyzed."""
        return 0  # No historical data required

    def get_signal_description(self) -> str:
        """Get description of this signal generator."""
        return "No Risk-off Signal Generator: Never signals risk-off conditions (always risk-on)"


class DummyRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """
    Dummy risk-off signal generator for testing and development purposes.

    This implementation provides configurable hardcoded risk-off windows for testing.
    It should NOT be used in production - it's designed for unit testing, integration
    testing, and strategy development where predictable risk-off periods are needed.

    Configuration Parameters:
    - risk_off_windows: List of (start_date, end_date) tuples for risk-off periods
    - default_risk_state: Default risk state when not in specified windows ('on' or 'off')

    Design Pattern: Test Double (specifically a Stub)
    - Provides predictable, controlled behavior for testing
    - Configurable via constructor parameters
    - Should be replaced with real implementation in production
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize dummy risk-off signal generator.

        Args:
            config: Configuration containing 'risk_off_windows' and 'default_risk_state'
        """
        self._config = config or {}

        # Parse risk-off windows from config
        self._risk_off_windows = self._parse_risk_off_windows()

        # Default risk state when not in specified windows
        self._default_risk_state = self._config.get("default_risk_state", "on")

        if self._default_risk_state not in ["on", "off"]:
            raise ValueError(
                f"Invalid default_risk_state: {self._default_risk_state}. Must be 'on' or 'off'"
            )

    def _parse_risk_off_windows(self) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
        """Parse risk-off windows from configuration."""
        windows_config = self._config.get("risk_off_windows", [])

        # If no windows specified, use some default test windows
        if not windows_config:
            return [
                (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31")),  # Financial crisis
                (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")),  # COVID crash
            ]

        parsed_windows = []
        for window in windows_config:
            if isinstance(window, (list, tuple)) and len(window) == 2:
                start_date = pd.Timestamp(window[0])
                end_date = pd.Timestamp(window[1])
                parsed_windows.append((start_date, end_date))
            else:
                logger.warning(
                    f"Invalid risk-off window format: {window}. Expected (start_date, end_date) tuple."
                )

        return parsed_windows

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Generate risk-off signal based on configured test windows.

        Args:
            all_historical_data: Universe data (unused in dummy implementation)
            benchmark_historical_data: Benchmark data (unused in dummy implementation)
            non_universe_historical_data: Non-universe data (unused)
            current_date: Date to generate signal for

        Returns:
            bool: True if current_date falls in configured risk-off window,
                  False otherwise (unless default_risk_state is 'off')
        """
        # Check if current date falls within any configured risk-off window
        for start_date, end_date in self._risk_off_windows:
            if start_date <= current_date <= end_date:
                return True  # Risk-off signal

        # Return default state when not in risk-off windows
        return bool(self._default_risk_state == 'off')

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        config = dict(self._config)
        # Add parsed windows for inspection
        config["_parsed_risk_off_windows"] = [
            (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            for start, end in self._risk_off_windows
        ]
        return config

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration parameters."""
        # Validate default_risk_state
        default_state = config.get("default_risk_state", "on")
        if default_state not in ["on", "off"]:
            return (False, f"Invalid default_risk_state: {default_state}. Must be 'on' or 'off'")

        # Validate risk_off_windows format
        windows = config.get("risk_off_windows", [])
        if not isinstance(windows, list):
            return (False, "risk_off_windows must be a list of (start_date, end_date) tuples")

        for i, window in enumerate(windows):
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                return (False, f"Window {i} must be a (start_date, end_date) tuple, got: {window}")

            try:
                start_date = pd.Timestamp(window[0])
                end_date = pd.Timestamp(window[1])
                if start_date >= end_date:
                    return (False, f"Window {i}: start_date must be before end_date")
            except Exception as e:
                return (False, f"Window {i}: Invalid date format - {e}")

        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        """Get required data columns - minimal as dummy doesn't analyze data."""
        return []  # Dummy implementation doesn't require data

    def get_minimum_data_periods(self) -> int:
        """Get minimum data periods - zero as dummy doesn't analyze data."""
        return 0  # No historical analysis in dummy implementation

    def get_signal_description(self) -> str:
        """Get description of this signal generator."""
        num_windows = len(self._risk_off_windows)
        return (
            f"Dummy Risk-off Signal Generator: Uses {num_windows} hardcoded test windows "
            f"(default: {self._default_risk_state}). FOR TESTING ONLY."
        )


# Example of how to implement a real risk-off signal generator:
#
# class VixBasedRiskOffSignalGenerator(IRiskOffSignalGenerator):
#     """
#     Risk-off signal based on VIX (volatility index) levels.
#
#     Signals risk-off when VIX exceeds configured threshold,
#     indicating heightened market fear and volatility.
#     """
#
#     def __init__(self, config: Dict[str, Any] | None = None):
#         self._config = config or {}
#         self._vix_threshold = self._config.get('vix_threshold', 30.0)
#         self._lookback_days = self._config.get('lookback_days', 5)
#
#     def generate_risk_off_signal(
#         self,
#         all_historical_data: pd.DataFrame,
#         benchmark_historical_data: pd.DataFrame,
#         non_universe_historical_data: pd.DataFrame,
#         current_date: pd.Timestamp,
#     ) -> bool:
#         # Implementation would:
#         # 1. Extract VIX data from non_universe_historical_data
#         # 2. Check if recent VIX levels exceed threshold
#         # 3. Return True if risk-off conditions detected
#         pass
#
#     def supports_non_universe_data(self) -> bool:
#         return True  # Requires VIX data
#
#     def get_required_data_columns(self) -> List[str]:
#         return ['Close']  # VIX close price
#
#     def get_minimum_data_periods(self) -> int:
#         return self._lookback_days
