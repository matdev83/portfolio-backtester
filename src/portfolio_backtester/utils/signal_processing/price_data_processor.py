"""
Price Data Processor

Handles price data extraction and processing for trading strategies.
Separated from strategy logic to follow Single Responsibility Principle.

SOLID Refactoring: This class now uses polymorphic interfaces instead of isinstance checks:
- Uses ISignalPriceExtractor for polymorphic price extraction
- Uses IColumnHandler for polymorphic column operations
- Uses IClosePriceProcessor for polymorphic price processing
"""

import logging
from typing import Optional, List, Dict

import pandas as pd

from ...interfaces.signal_price_extractor_interface import (
    SignalPriceExtractorFactory,
    ISignalPriceExtractor,
)
from ...interfaces.column_handler_interface import (
    ColumnHandlerFactory,
    IColumnHandler,
)

logger = logging.getLogger(__name__)


class PriceDataProcessor:
    """
    Processor for extracting and handling price data from various DataFrame formats.

    This class is responsible solely for data extraction and preprocessing,
    handling both MultiIndex and single-level column formats commonly used
    in the backtesting framework.

    SOLID Refactoring: Uses composition with polymorphic interfaces to eliminate
    isinstance violations while maintaining full backward compatibility.
    """

    def __init__(self, price_column: str = "Close"):
        """
        Initialize price data processor.

        Args:
            price_column: Name of the price column to extract (default: "Close")
        """
        self.price_column = price_column
        # Polymorphic components will be initialized per DataFrame
        self._cached_extractors: Dict[int, ISignalPriceExtractor] = {}

    def _get_extractor(self, data: pd.DataFrame) -> ISignalPriceExtractor:
        """Get appropriate price extractor for the DataFrame structure."""
        # Cache extractors based on DataFrame structure to avoid repeated instantiation
        data_key = id(data.columns)
        if data_key not in self._cached_extractors:
            self._cached_extractors[data_key] = SignalPriceExtractorFactory.create(
                data, self.price_column
            )
        return self._cached_extractors[data_key]

    def _get_column_handler(self, data: pd.DataFrame) -> IColumnHandler:
        """Get appropriate column handler for the DataFrame structure."""
        return ColumnHandlerFactory.create(data)

    def extract_ticker_prices(
        self, data: pd.DataFrame, ticker: str, current_date: pd.Timestamp
    ) -> Optional[pd.Series]:
        """
        Extract price series for a specific ticker from historical data.

        Args:
            data: Historical data DataFrame (can be MultiIndex or single-level columns)
            ticker: Ticker symbol to extract (e.g., "SPY")
            current_date: Current date to filter data up to

        Returns:
            Series of prices up to current_date, or None if ticker not found

        SOLID: Uses polymorphic price extractor instead of isinstance checks
        """
        try:
            # Use polymorphic extractor - eliminates isinstance violations
            extractor = self._get_extractor(data)
            return extractor.extract_ticker_price_series(
                data, ticker, current_date, self.price_column
            )

        except Exception as e:
            logger.error(f"Error extracting {ticker} prices: {e}")
            return None

    def get_universe_tickers(self, data: pd.DataFrame) -> List[str]:
        """
        Extract universe ticker list from DataFrame columns.

        Args:
            data: Historical data DataFrame

        Returns:
            List of ticker symbols

        SOLID: Uses polymorphic extractor instead of isinstance checks
        """
        # Use polymorphic extractor - eliminates isinstance violations
        extractor = self._get_extractor(data)
        return extractor.get_available_tickers(data)

    def validate_data_availability(self, data: pd.DataFrame, required_tickers: List[str]) -> bool:
        """
        Check if all required tickers are available in the dataset.

        Args:
            data: Historical data DataFrame
            required_tickers: List of required ticker symbols

        Returns:
            True if all tickers are available, False otherwise

        SOLID: Uses polymorphic validation instead of direct isinstance checks
        """
        # Use polymorphic extractor for validation
        extractor = self._get_extractor(data)
        available_tickers = extractor.get_available_tickers(data)
        missing_tickers = [ticker for ticker in required_tickers if ticker not in available_tickers]

        if missing_tickers:
            logger.warning(f"Missing tickers in data: {missing_tickers}")
            return False
        return True

    def create_empty_signals_dataframe(
        self, tickers: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Create empty signals DataFrame for a date range.

        Args:
            tickers: List of ticker symbols for columns
            start_date: Start date of the range
            end_date: End date of the range

        Returns:
            DataFrame filled with zeros for the specified date range and tickers

        SOLID: Enhanced with consistent interface patterns
        """
        if start_date == end_date:
            index = [start_date]
        else:
            index = pd.date_range(start_date, end_date, freq="D").tolist()

        return pd.DataFrame(index=index, columns=tickers, dtype=float).fillna(0.0)
