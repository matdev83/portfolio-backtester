from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any # Replaced Set with Dict, Any for config typing

import pandas as pd

# Removed Feature import as it's no longer used by BaseRoRoSignal
# from ..features.base import Feature


class BaseRoRoSignal(ABC):
    """
    Abstract base class for Risk-on/Risk-off (RoRo) signals.
    """

    def __init__(self, roro_config: Dict[str, Any] | None = None): # Changed type hint for config
        self.roro_config = roro_config if roro_config is not None else {}

    @abstractmethod
    def generate_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> bool:
        """
        Generates the RoRo signal for the current_date.

        Parameters:
        - all_historical_data (pd.DataFrame): Historical data for all universe assets up to current_date.
        - benchmark_historical_data (pd.DataFrame): Historical data for the benchmark up to current_date.
        - current_date (pd.Timestamp): The date for which to generate the signal.

        Returns:
        - bool: True for risk-on, False for risk-off.
        """
        pass

    def get_required_features(self) -> set:
        """
        Returns a set of features required by the RoRo signal.
        By default, RoRo signals do not require any features.
        Subclasses can override this method if they need features.
        """
        return set()


class DummyRoRoSignal(BaseRoRoSignal):
    """
    A dummy RoRo signal that returns True (risk-on) for specific hardcoded date windows
    and False (risk-off) for all other periods for the given current_date.
    """

    def generate_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> bool:
        """
        Generates the dummy RoRo signal for the current_date.

        Returns True (risk-on) for:
        - 2006-01-01 to 2009-12-31
        - 2020-01-01 to 2020-04-01
        - 2022-01-01 to 2022-11-05
        And False (risk-off) for all other periods.
        """

        # Define date windows for risk-on (signal = True)
        risk_on_windows = [
            (pd.Timestamp("2006-01-01"), pd.Timestamp("2009-12-31")),
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")),
            (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-05")),
        ]

        for start_date, end_date in risk_on_windows:
            if start_date <= current_date <= end_date:
                return True  # Risk-on

        return False  # Risk-off
