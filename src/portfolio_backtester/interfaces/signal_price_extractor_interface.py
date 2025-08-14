"""
Signal Price Extractor Interface

This module provides polymorphic interfaces for extracting price data from different DataFrame formats
in signal generation strategies, replacing isinstance checks with extensible strategy pattern implementations.

Key interfaces:
- ISignalPriceExtractor: Core interface for price data extraction
- ISignalDataValidator: Interface for validating signal data availability
- IClosesPriceProcessor: Interface for processing close prices from various DataFrame formats

This eliminates isinstance violations while maintaining full backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ISignalPriceExtractor(ABC):
    """
    Interface for extracting price data for signal generation.

    Replaces isinstance checks in signal strategies with polymorphic behavior
    based on DataFrame column structure.
    """

    @abstractmethod
    def extract_ticker_price_series(
        self,
        dataframe: pd.DataFrame,
        ticker: str,
        current_date: pd.Timestamp,
        price_column: str = "Close",
    ) -> Optional[pd.Series]:
        """
        Extract price series for a specific ticker up to current date.

        Args:
            dataframe: Historical data DataFrame
            ticker: Ticker symbol to extract
            current_date: Current date to filter data up to
            price_column: Price column name to extract

        Returns:
            Series of prices up to current_date, or None if not available
        """
        pass

    @abstractmethod
    def extract_all_close_prices(
        self, dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """
        Extract close prices for all tickers in the DataFrame.

        Args:
            dataframe: Historical data DataFrame
            price_column: Price column name to extract

        Returns:
            DataFrame with close prices for all tickers
        """
        pass

    @abstractmethod
    def get_available_tickers(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Get list of available ticker symbols from DataFrame.

        Args:
            dataframe: DataFrame to extract tickers from

        Returns:
            List of available ticker symbols
        """
        pass

    @abstractmethod
    def validate_ticker_data_availability(
        self, dataframe: pd.DataFrame, ticker: str, current_date: pd.Timestamp
    ) -> bool:
        """
        Validate that ticker data is available for the current date.

        Args:
            dataframe: DataFrame to check
            ticker: Ticker symbol to validate
            current_date: Date to check availability for

        Returns:
            True if data is available, False otherwise
        """
        pass


class MultiIndexSignalPriceExtractor(ISignalPriceExtractor):
    """Price extractor for DataFrames with MultiIndex columns."""

    def __init__(self, price_column: str = "Close"):
        """
        Initialize extractor with price column specification.

        Args:
            price_column: Name of price column to extract (default: "Close")
        """
        self.price_column = price_column

    def extract_ticker_price_series(
        self,
        dataframe: pd.DataFrame,
        ticker: str,
        current_date: pd.Timestamp,
        price_column: str = "Close",
    ) -> Optional[pd.Series]:
        """Extract price series from MultiIndex DataFrame."""
        try:
            # Handle MultiIndex columns - try direct column access first
            if (ticker, price_column) in dataframe.columns:
                ticker_prices = dataframe[(ticker, price_column)]
            else:
                # Try ticker-level access if "Ticker" level exists
                if ticker in dataframe.columns.get_level_values("Ticker"):
                    ticker_data = dataframe.xs(ticker, level="Ticker", axis=1)
                    if price_column in ticker_data.columns:
                        ticker_prices = ticker_data[price_column]
                    else:
                        logger.warning(
                            f"No {price_column} column found for {ticker} in MultiIndex DataFrame"
                        )
                        return None
                else:
                    logger.warning(f"{ticker} not found in MultiIndex DataFrame columns")
                    return None

            # Filter data up to current date and remove NaN values
            ticker_prices = ticker_prices[ticker_prices.index <= current_date].dropna()
            return ticker_prices

        except Exception as e:
            logger.error(f"Error extracting {ticker} prices from MultiIndex DataFrame: {e}")
            return None

    def extract_all_close_prices(
        self, dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """Extract all close prices from MultiIndex DataFrame."""
        try:
            # Use xs to extract the price column across all tickers
            result = dataframe.xs(price_column, level="Field", axis=1)
            # Ensure we always return a DataFrame
            if isinstance(result, pd.Series):
                return result.to_frame(name=price_column)
            return result
        except KeyError:
            # If "Field" level doesn't exist, try direct access
            close_data = {}
            for ticker in self.get_available_tickers(dataframe):
                if (ticker, price_column) in dataframe.columns:
                    close_data[ticker] = dataframe[(ticker, price_column)]
            return pd.DataFrame(close_data)

    def get_available_tickers(self, dataframe: pd.DataFrame) -> List[str]:
        """Get tickers from MultiIndex columns."""
        try:
            # Try to get from "Ticker" level first
            return dataframe.columns.get_level_values("Ticker").unique().tolist()
        except KeyError:
            # If no "Ticker" level, get from level 0
            return dataframe.columns.get_level_values(0).unique().tolist()

    def validate_ticker_data_availability(
        self, dataframe: pd.DataFrame, ticker: str, current_date: pd.Timestamp
    ) -> bool:
        """Validate ticker data availability in MultiIndex DataFrame."""
        try:
            if current_date not in dataframe.index:
                return False

            # Check if ticker exists and has data for current date
            if (ticker, self.price_column) in dataframe.columns:
                current_data = dataframe.loc[current_date, (ticker, self.price_column)]
                return pd.notna(current_data)

            return False
        except Exception:
            return False


class SingleIndexSignalPriceExtractor(ISignalPriceExtractor):
    """Price extractor for DataFrames with single-level columns."""

    def __init__(self, price_column: str = "Close"):
        """
        Initialize extractor with price column specification.

        Args:
            price_column: Name of price column to extract (default: "Close")
        """
        self.price_column = price_column

    def extract_ticker_price_series(
        self,
        dataframe: pd.DataFrame,
        ticker: str,
        current_date: pd.Timestamp,
        price_column: str = "Close",
    ) -> Optional[pd.Series]:
        """Extract price series from single-level DataFrame."""
        try:
            if ticker in dataframe.columns:
                ticker_prices = dataframe[ticker]
                # Filter data up to current date and remove NaN values
                ticker_prices = ticker_prices[ticker_prices.index <= current_date].dropna()
                return ticker_prices
            else:
                logger.warning(f"{ticker} not found in single-level DataFrame columns")
                return None
        except Exception as e:
            logger.error(f"Error extracting {ticker} prices from single-level DataFrame: {e}")
            return None

    def extract_all_close_prices(
        self, dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """For single-level columns, assume columns are already close prices."""
        return dataframe

    def get_available_tickers(self, dataframe: pd.DataFrame) -> List[str]:
        """Get column names as ticker symbols."""
        return dataframe.columns.tolist()

    def validate_ticker_data_availability(
        self, dataframe: pd.DataFrame, ticker: str, current_date: pd.Timestamp
    ) -> bool:
        """Validate ticker data availability in single-level DataFrame."""
        try:
            if current_date not in dataframe.index or ticker not in dataframe.columns:
                return False

            current_data = dataframe.loc[current_date, ticker]
            return pd.notna(current_data)
        except Exception:
            return False


class ISignalDataValidator(ABC):
    """Interface for validating data sufficiency for signal generation."""

    @abstractmethod
    def validate_data_sufficiency(
        self,
        dataframe: pd.DataFrame,
        required_tickers: List[str],
        current_date: pd.Timestamp,
        min_periods: int = 1,
    ) -> Tuple[bool, str]:
        """
        Validate that sufficient data is available for signal generation.

        Args:
            dataframe: DataFrame to validate
            required_tickers: List of required ticker symbols
            current_date: Current date to validate for
            min_periods: Minimum periods of data required

        Returns:
            Tuple of (is_sufficient, error_message)
        """
        pass

    @abstractmethod
    def filter_valid_assets(
        self, dataframe: pd.DataFrame, current_date: pd.Timestamp, min_periods: int = 1
    ) -> List[str]:
        """
        Filter assets that have sufficient data for the current date.

        Args:
            dataframe: DataFrame to filter
            current_date: Current date to check
            min_periods: Minimum periods required

        Returns:
            List of assets with sufficient data
        """
        pass


class SignalDataValidator(ISignalDataValidator):
    """Standard implementation of signal data validation."""

    def __init__(self, price_extractor: ISignalPriceExtractor):
        """
        Initialize validator with price extractor.

        Args:
            price_extractor: Price extractor to use for validation
        """
        self.price_extractor = price_extractor

    def validate_data_sufficiency(
        self,
        dataframe: pd.DataFrame,
        required_tickers: List[str],
        current_date: pd.Timestamp,
        min_periods: int = 1,
    ) -> Tuple[bool, str]:
        """Validate data sufficiency using price extractor."""
        if dataframe.empty:
            return False, "DataFrame is empty"

        if current_date not in dataframe.index:
            return False, f"No data for current_date: {current_date}"

        # Check each required ticker
        missing_tickers = []
        for ticker in required_tickers:
            if not self.price_extractor.validate_ticker_data_availability(
                dataframe, ticker, current_date
            ):
                missing_tickers.append(ticker)

        if missing_tickers:
            return False, f"Missing or insufficient data for tickers: {missing_tickers}"

        return True, ""

    def filter_valid_assets(
        self, dataframe: pd.DataFrame, current_date: pd.Timestamp, min_periods: int = 1
    ) -> List[str]:
        """Filter assets with sufficient data."""
        if dataframe.empty or current_date not in dataframe.index:
            return []

        valid_assets = []
        available_tickers = self.price_extractor.get_available_tickers(dataframe)

        for ticker in available_tickers:
            if self.price_extractor.validate_ticker_data_availability(
                dataframe, ticker, current_date
            ):
                valid_assets.append(ticker)

        return valid_assets


class IClosePriceProcessor(ABC):
    """Interface for processing close prices from various DataFrame formats."""

    @abstractmethod
    def extract_close_values(
        self, dataframe: pd.DataFrame, tickers: List[str], current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Extract current close price values for specified tickers.

        Args:
            dataframe: DataFrame containing price data
            tickers: List of ticker symbols to extract
            current_date: Date to extract prices for

        Returns:
            Dictionary mapping ticker symbols to close prices
        """
        pass

    @abstractmethod
    def create_signals_dataframe(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Create empty signals DataFrame for specified tickers and date range.

        Args:
            tickers: List of ticker symbols for columns
            start_date: Start date of the range
            end_date: End date of the range (optional)

        Returns:
            DataFrame filled with zeros for the specified date range and tickers
        """
        pass


class ClosePriceProcessor(IClosePriceProcessor):
    """Standard implementation of close price processing."""

    def __init__(self, price_extractor: ISignalPriceExtractor):
        """
        Initialize processor with price extractor.

        Args:
            price_extractor: Price extractor to use for processing
        """
        self.price_extractor = price_extractor

    def extract_close_values(
        self, dataframe: pd.DataFrame, tickers: List[str], current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Extract close values using price extractor."""
        close_values = {}

        for ticker in tickers:
            try:
                if self.price_extractor.validate_ticker_data_availability(
                    dataframe, ticker, current_date
                ):
                    price_series = self.price_extractor.extract_ticker_price_series(
                        dataframe, ticker, current_date
                    )
                    if price_series is not None and not price_series.empty:
                        # Get the last available price up to current_date
                        close_values[ticker] = float(price_series.iloc[-1])
            except Exception as e:
                logger.warning(f"Failed to extract close price for {ticker}: {e}")

        return close_values

    def create_signals_dataframe(
        self,
        tickers: List[str],
        start_date: pd.Timestamp,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Create empty signals DataFrame."""
        if end_date is None or start_date == end_date:
            index = [start_date]
        else:
            index = pd.date_range(start_date, end_date, freq="D").tolist()

        return pd.DataFrame(index=index, columns=tickers, dtype=float).fillna(0.0)


class SignalPriceExtractorFactory:
    """Factory for creating appropriate signal price extractors based on DataFrame structure."""

    @staticmethod
    def create(dataframe: pd.DataFrame, price_column: str = "Close") -> ISignalPriceExtractor:
        """
        Create appropriate price extractor based on DataFrame column structure.

        Args:
            dataframe: DataFrame to analyze
            price_column: Price column name to use

        Returns:
            Appropriate price extractor instance
        """
        if isinstance(dataframe.columns, pd.MultiIndex):
            return MultiIndexSignalPriceExtractor(price_column=price_column)
        else:
            return SingleIndexSignalPriceExtractor(price_column=price_column)

    @staticmethod
    def create_with_validator(
        dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> Tuple[ISignalPriceExtractor, ISignalDataValidator]:
        """
        Create price extractor with corresponding data validator.

        Args:
            dataframe: DataFrame to analyze
            price_column: Price column name to use

        Returns:
            Tuple of (price_extractor, data_validator)
        """
        price_extractor = SignalPriceExtractorFactory.create(dataframe, price_column)
        data_validator = SignalDataValidator(price_extractor)
        return price_extractor, data_validator

    @staticmethod
    def create_full_suite(
        dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> Tuple[ISignalPriceExtractor, ISignalDataValidator, IClosePriceProcessor]:
        """
        Create complete suite of signal processing components.

        Args:
            dataframe: DataFrame to analyze
            price_column: Price column name to use

        Returns:
            Tuple of (price_extractor, data_validator, price_processor)
        """
        price_extractor = SignalPriceExtractorFactory.create(dataframe, price_column)
        data_validator = SignalDataValidator(price_extractor)
        price_processor = ClosePriceProcessor(price_extractor)
        return price_extractor, data_validator, price_processor
