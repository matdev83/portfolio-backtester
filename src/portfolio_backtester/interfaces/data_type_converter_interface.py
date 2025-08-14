"""
Data Type Converter Interface

This module provides polymorphic interfaces for handling data type conversions in signal strategies,
replacing isinstance checks with extensible strategy pattern implementations.

Key interfaces:
- IDataTypeConverter: Core interface for data type conversions
- ISeriesConverter: Interface for converting data to pandas Series
- IArrayConverter: Interface for converting data to numpy arrays
- IDataFrameConverter: Interface for converting data to pandas DataFrame

This eliminates isinstance violations while maintaining full backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IDataTypeConverter(ABC):
    """
    Interface for converting data types in signal strategies.

    Replaces isinstance checks with polymorphic behavior based on the Strategy pattern.
    """

    @abstractmethod
    def ensure_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert data to DataFrame if not already.

        Args:
            data: Input data of any type

        Returns:
            DataFrame representation of the data
        """
        pass

    @abstractmethod
    def ensure_series(self, data: Any, name: Optional[str] = None) -> pd.Series:
        """
        Convert data to Series if not already.

        Args:
            data: Input data of any type
            name: Optional name for the Series

        Returns:
            Series representation of the data
        """
        pass

    @abstractmethod
    def ensure_numpy_array(self, data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Convert data to numpy array if not already.

        Args:
            data: Input data of any type
            dtype: Optional dtype for the array

        Returns:
            Numpy array representation of the data
        """
        pass

    @abstractmethod
    def detect_data_type(self, data: Any) -> str:
        """
        Detect the type of input data.

        Args:
            data: Input data to analyze

        Returns:
            String description of data type
        """
        pass


class StandardDataTypeConverter(IDataTypeConverter):
    """Standard implementation of data type conversion."""

    def ensure_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert data to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return data.to_frame()
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        elif isinstance(data, (list, tuple)):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            # Try to convert to DataFrame directly
            try:
                return pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Failed to convert {type(data)} to DataFrame: {e}")
                raise ValueError(f"Cannot convert {type(data)} to DataFrame")

    def ensure_series(self, data: Any, name: Optional[str] = None) -> pd.Series:
        """Convert data to Series."""
        if isinstance(data, pd.Series):
            if name is not None:
                return data.rename(name)
            return data
        elif isinstance(data, pd.DataFrame):
            # If DataFrame has single column, convert to Series
            if len(data.columns) == 1:
                series = data.iloc[:, 0]
                if name is not None:
                    series.name = name
                return series
            else:
                raise ValueError(
                    "Cannot convert multi-column DataFrame to Series without specifying column"
                )
        elif isinstance(data, np.ndarray):
            return pd.Series(data, name=name)
        elif isinstance(data, (list, tuple)):
            return pd.Series(data, name=name)
        else:
            # Try to convert to Series directly
            try:
                return pd.Series(
                    data, name=name, dtype=object
                )  # Explicit dtype to avoid Any return
            except Exception as e:
                logger.error(f"Failed to convert {type(data)} to Series: {e}")
                raise ValueError(f"Cannot convert {type(data)} to Series")

    def ensure_numpy_array(self, data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, np.ndarray):
            if dtype is not None and data.dtype != dtype:
                return data.astype(dtype)
            return data
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            array = data.values
            if dtype is not None:
                return np.asarray(array, dtype=dtype)
            return np.asarray(array)
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=dtype)
        else:
            # Try to convert to array directly
            try:
                return np.array(data, dtype=dtype)
            except Exception as e:
                logger.error(f"Failed to convert {type(data)} to numpy array: {e}")
                raise ValueError(f"Cannot convert {type(data)} to numpy array")

    def detect_data_type(self, data: Any) -> str:
        """Detect the type of input data."""
        if isinstance(data, pd.DataFrame):
            return f"DataFrame ({data.shape[0]}x{data.shape[1]})"
        elif isinstance(data, pd.Series):
            return f"Series (length {len(data)})"
        elif isinstance(data, np.ndarray):
            return f"ndarray {data.shape} {data.dtype}"
        elif isinstance(data, list):
            return f"list (length {len(data)})"
        elif isinstance(data, tuple):
            return f"tuple (length {len(data)})"
        elif isinstance(data, dict):
            return f"dict ({len(data)} keys)"
        else:
            return f"{type(data).__name__}"


class ISeriesConverter(ABC):
    """Interface for converting to pandas Series."""

    @abstractmethod
    def convert_to_series(self, data: Any, name: Optional[str] = None) -> pd.Series:
        """
        Convert input data to pandas Series.

        Args:
            data: Input data
            name: Optional name for Series

        Returns:
            pandas Series
        """
        pass

    @abstractmethod
    def validate_series_conversion(self, data: Any) -> bool:
        """
        Check if data can be converted to Series.

        Args:
            data: Input data to validate

        Returns:
            True if conversion is possible, False otherwise
        """
        pass


class SeriesConverter(ISeriesConverter):
    """Standard implementation of Series conversion."""

    def __init__(self, converter: IDataTypeConverter) -> None:
        """
        Initialize with data type converter.

        Args:
            converter: Data type converter to use
        """
        self.converter = converter

    def convert_to_series(self, data: Any, name: Optional[str] = None) -> pd.Series:
        """Convert data to Series using converter."""
        return self.converter.ensure_series(data, name)

    def validate_series_conversion(self, data: Any) -> bool:
        """Validate Series conversion possibility."""
        try:
            self.converter.ensure_series(data)
            return True
        except (ValueError, TypeError):
            return False


class IArrayConverter(ABC):
    """Interface for converting to numpy arrays."""

    @abstractmethod
    def convert_to_array(self, data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Convert input data to numpy array.

        Args:
            data: Input data
            dtype: Optional dtype for array

        Returns:
            numpy array
        """
        pass

    @abstractmethod
    def ensure_boolean_array(self, data: Any) -> np.ndarray:
        """
        Convert input data to boolean numpy array.

        Args:
            data: Input data

        Returns:
            Boolean numpy array
        """
        pass

    @abstractmethod
    def validate_array_conversion(self, data: Any) -> bool:
        """
        Check if data can be converted to array.

        Args:
            data: Input data to validate

        Returns:
            True if conversion is possible, False otherwise
        """
        pass


class ArrayConverter(IArrayConverter):
    """Standard implementation of Array conversion."""

    def __init__(self, converter: IDataTypeConverter) -> None:
        """
        Initialize with data type converter.

        Args:
            converter: Data type converter to use
        """
        self.converter = converter

    def convert_to_array(self, data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Convert data to array using converter."""
        return self.converter.ensure_numpy_array(data, dtype)

    def ensure_boolean_array(self, data: Any) -> np.ndarray:
        """Convert to boolean array."""
        array = self.converter.ensure_numpy_array(data, dtype=np.dtype(bool))
        return array.astype(bool)

    def validate_array_conversion(self, data: Any) -> bool:
        """Validate array conversion possibility."""
        try:
            self.converter.ensure_numpy_array(data)
            return True
        except (ValueError, TypeError):
            return False


class IDataFrameConverter(ABC):
    """Interface for converting to pandas DataFrame."""

    @abstractmethod
    def convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert input data to pandas DataFrame.

        Args:
            data: Input data

        Returns:
            pandas DataFrame
        """
        pass

    @abstractmethod
    def handle_multiindex_result(self, data: Any) -> pd.DataFrame:
        """
        Handle conversion of MultiIndex results to DataFrame.

        Args:
            data: Input data (potentially MultiIndex result)

        Returns:
            pandas DataFrame
        """
        pass

    @abstractmethod
    def validate_dataframe_conversion(self, data: Any) -> bool:
        """
        Check if data can be converted to DataFrame.

        Args:
            data: Input data to validate

        Returns:
            True if conversion is possible, False otherwise
        """
        pass


class DataFrameConverter(IDataFrameConverter):
    """Standard implementation of DataFrame conversion."""

    def __init__(self, converter: IDataTypeConverter) -> None:
        """
        Initialize with data type converter.

        Args:
            converter: Data type converter to use
        """
        self.converter = converter

    def convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert data to DataFrame using converter."""
        return self.converter.ensure_dataframe(data)

    def handle_multiindex_result(self, data: Any) -> pd.DataFrame:
        """Handle MultiIndex results that may return Series."""
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame
            return data.to_frame()
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            # Use standard conversion
            return self.converter.ensure_dataframe(data)

    def validate_dataframe_conversion(self, data: Any) -> bool:
        """Validate DataFrame conversion possibility."""
        try:
            self.converter.ensure_dataframe(data)
            return True
        except (ValueError, TypeError):
            return False


class ISignalDataTypeProcessor(ABC):
    """Interface for processing data types in signal generation contexts."""

    @abstractmethod
    def process_close_prices_extraction(
        self, extraction_result: Any, field_name: str = "Close"
    ) -> pd.DataFrame:
        """
        Process close prices extraction results to ensure DataFrame output.

        Args:
            extraction_result: Result from price extraction operation
            field_name: Name of the field being extracted

        Returns:
            DataFrame with close prices
        """
        pass

    @abstractmethod
    def process_boolean_conditions(self, condition_data: Any) -> np.ndarray:
        """
        Process boolean condition data to ensure numpy array output.

        Args:
            condition_data: Boolean condition data

        Returns:
            Boolean numpy array
        """
        pass

    @abstractmethod
    def process_signal_weights(self, weights_data: Any, tickers: List[str]) -> pd.Series:
        """
        Process signal weights to ensure proper Series output.

        Args:
            weights_data: Signal weights data
            tickers: List of ticker symbols

        Returns:
            Series with signal weights
        """
        pass


class SignalDataTypeProcessor(ISignalDataTypeProcessor):
    """Signal data type processor implementation."""

    def __init__(self) -> None:
        """Initialize with standard converters."""
        converter = StandardDataTypeConverter()
        self.series_converter = SeriesConverter(converter)
        self.array_converter = ArrayConverter(converter)
        self.dataframe_converter = DataFrameConverter(converter)

    def process_close_prices_extraction(
        self, extraction_result: Any, field_name: str = "Close"
    ) -> pd.DataFrame:
        """Process close prices to ensure DataFrame."""
        return self.dataframe_converter.handle_multiindex_result(extraction_result)

    def process_boolean_conditions(self, condition_data: Any) -> np.ndarray:
        """Process boolean conditions to ensure array."""
        return self.array_converter.ensure_boolean_array(condition_data)

    def process_signal_weights(self, weights_data: Any, tickers: List[str]) -> pd.Series:
        """Process signal weights to ensure Series."""
        series = self.series_converter.convert_to_series(weights_data)
        # Ensure the index matches the tickers if needed
        if len(series) == len(tickers) and not series.index.equals(pd.Index(tickers)):
            return pd.Series(series.values, index=tickers, name=series.name)
        return series


class DataTypeConverterFactory:
    """Factory for creating data type converters and processors."""

    @staticmethod
    def create_standard_converter() -> IDataTypeConverter:
        """Create standard data type converter."""
        return StandardDataTypeConverter()

    @staticmethod
    def create_series_converter() -> ISeriesConverter:
        """Create Series converter with standard converter."""
        converter = StandardDataTypeConverter()
        return SeriesConverter(converter)

    @staticmethod
    def create_array_converter() -> IArrayConverter:
        """Create array converter with standard converter."""
        converter = StandardDataTypeConverter()
        return ArrayConverter(converter)

    @staticmethod
    def create_dataframe_converter() -> IDataFrameConverter:
        """Create DataFrame converter with standard converter."""
        converter = StandardDataTypeConverter()
        return DataFrameConverter(converter)

    @staticmethod
    def create_signal_processor() -> ISignalDataTypeProcessor:
        """Create signal data type processor."""
        return SignalDataTypeProcessor()

    @staticmethod
    def create_full_suite() -> tuple[
        IDataTypeConverter,
        ISeriesConverter,
        IArrayConverter,
        IDataFrameConverter,
        ISignalDataTypeProcessor,
    ]:
        """
        Create complete suite of data type conversion components.

        Returns:
            Tuple of all converter components
        """
        converter = StandardDataTypeConverter()
        series_converter = SeriesConverter(converter)
        array_converter = ArrayConverter(converter)
        dataframe_converter = DataFrameConverter(converter)
        signal_processor = SignalDataTypeProcessor()

        return (
            converter,
            series_converter,
            array_converter,
            dataframe_converter,
            signal_processor,
        )
