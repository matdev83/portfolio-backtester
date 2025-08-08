from abc import ABC, abstractmethod
import pandas as pd


class BasePositionSizer(ABC):
    """Abstract base class for all position sizers."""

    @abstractmethod
    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Calculate the target weights for each asset.

        Note: This method must return a DataFrame of positive weights.
        The direction of the trade (long/short) is determined by the strategy,
        not the sizer.

        Args:
            signals (pd.DataFrame): The trading signals.
            prices (pd.DataFrame): The price data.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The calculated weights.
        """
        pass
