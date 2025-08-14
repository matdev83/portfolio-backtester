"""
Window-specific data processing for cache operations.

This module provides functionality for processing window-specific data,
including extracting close prices from various DataFrame formats and
computing returns for specific time windows.
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging

from ..interfaces.close_price_extractor_interface import ClosePriceExtractorFactory

logger = logging.getLogger(__name__)


class WindowProcessor:
    """
    Handles window-specific data processing operations.

    This class now uses polymorphic interfaces instead of isinstance violations,
    maintaining backward compatibility while improving extensibility.
    """

    def __init__(self):
        self._extractor_factory = ClosePriceExtractorFactory()

    def extract_close_prices(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract close prices from a DataFrame with various column structures.

        Args:
            data: Input DataFrame that may have MultiIndex columns or simple columns

        Returns:
            DataFrame with close prices, or None if extraction fails
        """
        extractor = self._extractor_factory.get_extractor(data)
        if extractor:
            return extractor.extract_close_prices(data)
        return None

    @staticmethod
    def compute_returns(price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute returns from price data.

        Args:
            price_data: DataFrame with price data

        Returns:
            DataFrame with computed returns (pct_change)
        """
        return price_data.pct_change(fill_method=None).fillna(0)

    def precompute_window_returns(
        self, daily_data: pd.DataFrame, windows: List[Tuple]
    ) -> Dict[str, pd.DataFrame]:
        """
        Pre-compute returns for all windows.

        Args:
            daily_data: Full daily price data
            windows: List of (tr_start, tr_end, te_start, te_end) tuples

        Returns:
            Dictionary mapping window identifiers to return DataFrames
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pre-computing returns for {len(windows)} windows")

        window_returns = {}

        for window_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            # Get the full window data (training + test)
            window_data = daily_data.loc[tr_start:te_end]

            if window_data.empty:
                continue

            # Extract close prices
            close_prices_df = self.extract_close_prices(window_data)
            if close_prices_df is None:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Could not extract close prices for window {window_idx}")
                continue

            # Compute returns for this window
            returns_df = self.compute_returns(close_prices_df)

            # Create a unique identifier for this window
            window_id = f"{tr_start}_{te_end}_{window_data.shape}"
            window_returns[window_id] = returns_df

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pre-computed returns for {len(window_returns)} windows")

        return window_returns

    def get_window_data_by_dates(
        self,
        daily_data: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """
        Get window data by date range and extract close prices.

        Args:
            daily_data: Full daily price data
            window_start: Start of the window
            window_end: End of the window

        Returns:
            DataFrame with close prices for the window, or None if extraction fails
        """
        window_data = daily_data.loc[window_start:window_end]

        if window_data.empty:
            return None

        return self.extract_close_prices(window_data)
