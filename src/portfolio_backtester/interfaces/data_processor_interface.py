"""
Data processing interfaces for SSGA daily processing and data validation.

This module provides polymorphic interfaces to replace isinstance violations
in the spy_holdings.py module for better adherence to the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class IDataValidator(ABC):
    """Interface for data validation operations."""

    @abstractmethod
    def validate_dataframe(self, data: Any) -> bool:
        """
        Validate if the provided data is a valid DataFrame.

        Parameters
        ----------
        data : Any
            The data to validate

        Returns
        -------
        bool
            True if data is a valid DataFrame, False otherwise
        """
        pass

    @abstractmethod
    def is_empty_or_invalid(self, data: Any) -> bool:
        """
        Check if the data is empty or invalid.

        Parameters
        ----------
        data : Any
            The data to check

        Returns
        -------
        bool
            True if data is empty or invalid, False otherwise
        """
        pass


class ICusipValidator(ABC):
    """Interface for CUSIP format validation."""

    @abstractmethod
    def is_valid_cusip_format(self, cusip: Any) -> bool:
        """
        Validate if the provided value is a valid CUSIP format.

        Parameters
        ----------
        cusip : Any
            The value to validate as CUSIP

        Returns
        -------
        bool
            True if valid CUSIP format, False otherwise
        """
        pass


class DataFrameValidator(IDataValidator):
    """Concrete implementation for DataFrame validation."""

    def validate_dataframe(self, data: Any) -> bool:
        """Validate if the provided data is a valid DataFrame."""
        return isinstance(data, pd.DataFrame)

    def is_empty_or_invalid(self, data: Any) -> bool:
        """Check if the data is empty or invalid."""
        if not self.validate_dataframe(data):
            return True
        # Type narrowing: we know data is pd.DataFrame due to validate_dataframe check
        return bool(data.empty)


class CusipFormatValidator(ICusipValidator):
    """Concrete implementation for CUSIP format validation."""

    def is_valid_cusip_format(self, cusip: Any) -> bool:
        """Validate if the provided value is a valid CUSIP format."""
        return isinstance(cusip, str) and len(cusip) == 9 and cusip.isalnum()


class DataProcessorFactory:
    """Factory for creating data processor instances."""

    @staticmethod
    def create_dataframe_validator() -> IDataValidator:
        """Create a DataFrame validator instance."""
        return DataFrameValidator()

    @staticmethod
    def create_cusip_validator() -> ICusipValidator:
        """Create a CUSIP validator instance."""
        return CusipFormatValidator()
