from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

import pandas as pd

from ..features.base import Feature


class BaseRoRoSignal(ABC):
    """
    Abstract base class for Risk-on/Risk-off (RoRo) signals.
    """

    def __init__(self, roro_config: dict | None = None):
        self.roro_config = roro_config if roro_config is not None else {}

    @abstractmethod
    def generate_signal(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generates the RoRo signal.

        Parameters:
        - dates (pd.DatetimeIndex): The dates for which to generate the signal.

        Returns:
        - pd.Series: A series with the RoRo signal (typically 1 for risk-on, 0 for risk-off),
                     indexed by date.
        """
        pass

    def get_required_features(self) -> Set[Feature]:
        """
        Returns a set of features required by the RoRo signal.
        By default, RoRo signals do not require any features.
        Subclasses can override this method if they need features.
        """
        return set()


class DummyRoRoSignal(BaseRoRoSignal):
    """
    A dummy RoRo signal that returns 1 for specific hardcoded date windows
    and 0 for all other periods.
    """

    def generate_signal(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generates the dummy RoRo signal.

        Returns 1 for:
        - 2006-01-01 to 2009-12-31
        - 2020-01-01 to 2020-04-01
        - 2022-01-01 to 2022-11-05
        And 0 for all other periods.
        """
        signal = pd.Series(0, index=dates, dtype=int)

        # Define date windows for risk-on (signal = 1)
        risk_on_windows = [
            (pd.Timestamp("2006-01-01"), pd.Timestamp("2009-12-31")),
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")),
            (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-05")),
        ]

        for start_date, end_date in risk_on_windows:
            signal.loc[(dates >= start_date) & (dates <= end_date)] = 1

        return signal
