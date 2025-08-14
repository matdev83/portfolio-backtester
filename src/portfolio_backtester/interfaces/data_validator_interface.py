"""
Data validator interfaces for eliminating isinstance violations in data sources.

This module provides polymorphic interfaces for validating DataFrame structure
and index types without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IIndexValidator(ABC):
    """Interface for validating DataFrame index types."""

    @abstractmethod
    def can_validate(self, index: Any) -> bool:
        """Check if this validator can handle the given index type."""
        pass

    @abstractmethod
    def is_valid_datetime_index(self, index: Any) -> bool:
        """Check if index is a valid datetime index."""
        pass


class DatetimeIndexValidator(IIndexValidator):
    """Validator for datetime index structures."""

    def can_validate(self, index: Any) -> bool:
        """Check if index has datetime characteristics using duck typing."""
        return (
            hasattr(index, "to_pydatetime")
            and hasattr(index, "date")
            and hasattr(index, "normalize")
        )

    def is_valid_datetime_index(self, index: Any) -> bool:
        """Always returns True for datetime index."""
        return True


class NonDatetimeIndexValidator(IIndexValidator):
    """Validator for non-datetime index structures."""

    def can_validate(self, index: Any) -> bool:
        """Always returns True as fallback validator."""
        return True

    def is_valid_datetime_index(self, index: Any) -> bool:
        """Always returns False for non-datetime index."""
        return False


class IndexValidatorFactory:
    """Factory for creating appropriate index validators."""

    def __init__(self) -> None:
        self._validators = [
            DatetimeIndexValidator(),
            NonDatetimeIndexValidator(),  # Fallback
        ]

    def get_validator(self, index: Any) -> IIndexValidator:
        """Get the appropriate validator for the given index."""
        for validator in self._validators:
            if validator.can_validate(index):
                return validator
        # Should never reach here due to NonDatetimeIndexValidator fallback
        return self._validators[-1]


class IColumnStructureValidator(ABC):
    """Interface for validating column structures."""

    @abstractmethod
    def can_validate(self, columns: Any) -> bool:
        """Check if this validator can handle the given column type."""
        pass

    @abstractmethod
    def validate_ticker_and_field(
        self, columns: Any, ticker: str, field: str = "Close"
    ) -> tuple[bool, str]:
        """
        Validate that ticker and field exist in columns.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class MultiIndexColumnValidator(IColumnStructureValidator):
    """Validator for MultiIndex column structures."""

    def can_validate(self, columns: Any) -> bool:
        """Check if columns is a MultiIndex using duck typing."""
        return (
            hasattr(columns, "nlevels")
            and hasattr(columns, "names")
            and hasattr(columns, "get_level_values")
            and getattr(columns, "nlevels", 1) > 1
        )

    def validate_ticker_and_field(
        self, columns: Any, ticker: str, field: str = "Close"
    ) -> tuple[bool, str]:
        """Validate ticker and field exist in MultiIndex columns."""
        try:
            # Check if field exists in the Field level
            if field not in columns.get_level_values("Field"):
                return False, f"No {field} column in MultiIndex"

            # Check if ticker exists in the Ticker level
            if ticker not in columns.get_level_values("Ticker"):
                return False, "Ticker not found in MultiIndex"

            return True, ""
        except KeyError as e:
            return False, f"MultiIndex validation error: {e}"


class SimpleColumnValidator(IColumnStructureValidator):
    """Validator for simple (non-MultiIndex) column structures."""

    def can_validate(self, columns: Any) -> bool:
        """Check if columns is a simple Index using duck typing."""
        return hasattr(columns, "__contains__") and hasattr(columns, "tolist")

    def validate_ticker_and_field(
        self, columns: Any, ticker: str, field: str = "Close"
    ) -> tuple[bool, str]:
        """Validate ticker exists in simple columns."""
        if ticker not in columns:
            available = getattr(columns, "tolist", lambda: list(columns))()
            return False, f"Ticker column not found. Available: {available}"
        return True, ""


class ColumnStructureValidatorFactory:
    """Factory for creating appropriate column structure validators."""

    def __init__(self):
        self._validators = [
            MultiIndexColumnValidator(),
            SimpleColumnValidator(),  # Fallback
        ]

    def get_validator(self, columns: Any) -> IColumnStructureValidator:
        """Get the appropriate validator for the given columns."""
        for validator in self._validators:
            if validator.can_validate(columns):
                return validator
        # Should never reach here due to SimpleColumnValidator fallback
        return self._validators[-1]


class PolymorphicDataValidator:
    """Polymorphic data validator that eliminates isinstance violations."""

    def __init__(self):
        self._index_factory = IndexValidatorFactory()
        self._column_factory = ColumnStructureValidatorFactory()

    def is_valid_datetime_index(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has valid datetime index using polymorphic validation."""
        validator = self._index_factory.get_validator(df.index)
        return validator.is_valid_datetime_index(df.index)

    def validate_ticker_data_structure(
        self, df: pd.DataFrame, ticker: str, field: str = "Close"
    ) -> tuple[bool, str]:
        """
        Validate that DataFrame has required ticker and field data.

        Args:
            df: DataFrame to validate
            ticker: Ticker symbol to check for
            field: Field name to check for (default: "Close")

        Returns:
            Tuple of (is_valid, error_message)
        """
        validator = self._column_factory.get_validator(df.columns)
        return validator.validate_ticker_and_field(df.columns, ticker, field)


# Factory function for easy instantiation
def create_data_validator() -> PolymorphicDataValidator:
    """Create a new polymorphic data validator instance."""
    return PolymorphicDataValidator()
