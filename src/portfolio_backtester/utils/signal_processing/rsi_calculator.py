"""
RSI Calculator

Handles RSI (Relative Strength Index) calculation for trading strategies.
Separated from strategy logic to follow Single Responsibility Principle.
"""

import logging
from typing import Optional

import pandas as pd
from ta.momentum import RSIIndicator

logger = logging.getLogger(__name__)


class RSICalculator:
    """
    Calculator for RSI (Relative Strength Index) technical indicator.

    This class is responsible solely for RSI computation using the ta library.
    It handles edge cases like insufficient data and provides clean interfaces
    for strategy classes.
    """

    def __init__(self, period: int = 2):
        """
        Initialize RSI calculator.

        Args:
            period: RSI period (e.g., 2 for RSI(2), 14 for RSI(14))
        """
        if period < 1:
            raise ValueError("RSI period must be at least 1")
        self.period = period

    def calculate(self, price_series: pd.Series) -> pd.Series:
        """
        Calculate RSI for a price series using ta library.

        Args:
            price_series: Series of prices (typically close prices)

        Returns:
            Series of RSI values with same index as input
            Returns empty series if insufficient data
        """
        if len(price_series) < self.period + 1:
            logger.debug(
                f"Insufficient data for RSI calculation: {len(price_series)} data points, need at least {self.period + 1}"
            )
            return pd.Series(index=price_series.index, dtype=float)

        try:
            # Use ta library for RSI calculation
            rsi: pd.Series = RSIIndicator(close=price_series, window=self.period).rsi()
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=price_series.index, dtype=float)

    def get_current_rsi(
        self, price_series: pd.Series, current_date: pd.Timestamp
    ) -> Optional[float]:
        """
        Get RSI value for a specific date.

        Args:
            price_series: Series of prices
            current_date: Date to get RSI value for

        Returns:
            RSI value for the date, or None if not available
        """
        rsi_series = self.calculate(price_series)
        if current_date in rsi_series.index:
            rsi_value = rsi_series.loc[current_date]
            return None if pd.isna(rsi_value) else float(rsi_value)
        return None

    def is_below_threshold(
        self, price_series: pd.Series, current_date: pd.Timestamp, threshold: float
    ) -> bool:
        """
        Check if RSI is below a threshold for a specific date.

        Args:
            price_series: Series of prices
            current_date: Date to check
            threshold: RSI threshold value

        Returns:
            True if RSI is below threshold, False otherwise
        """
        current_rsi = self.get_current_rsi(price_series, current_date)
        return current_rsi is not None and current_rsi < threshold

    @property
    def minimum_data_points(self) -> int:
        """
        Get minimum number of data points needed for RSI calculation.

        Returns:
            Minimum data points required
        """
        return self.period + 1
