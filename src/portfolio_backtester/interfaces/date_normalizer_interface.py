"""Date normalizer interfaces for polymorphic date handling.

This module provides interfaces and implementations for normalizing different
date types to pandas Timestamps. This replaces isinstance violations with
proper polymorphic design following the Open/Closed Principle.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class IDateNormalizer(ABC):
    """Interface for normalizing different date representations to Timestamp."""

    @abstractmethod
    def can_handle(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> bool:
        """Check if this normalizer can handle the given date object.

        Args:
            date_obj: The date object to check

        Returns:
            True if this normalizer can handle the date object
        """
        pass

    @abstractmethod
    def normalize(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Normalize the date object to a pandas Timestamp.

        Args:
            date_obj: The date object to normalize

        Returns:
            Normalized pandas Timestamp at 00:00:00

        Raises:
            TypeError: If the date object cannot be handled by this normalizer
        """
        pass

    @abstractmethod
    def normalize_preserve_time(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Normalize the date object to a pandas Timestamp preserving time.

        Args:
            date_obj: The date object to normalize

        Returns:
            Pandas Timestamp with original time component preserved

        Raises:
            TypeError: If the date object cannot be handled by this normalizer
        """
        pass


class TimestampDateNormalizer(IDateNormalizer):
    """Normalizer for pandas Timestamp objects."""

    def can_handle(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> bool:
        """Check if the date object is a pandas Timestamp."""
        return isinstance(date_obj, pd.Timestamp)

    def normalize(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Normalize pandas Timestamp to 00:00:00."""
        if not self.can_handle(date_obj):
            raise TypeError(f"Cannot handle date object of type {type(date_obj)}")

        # MyPy needs explicit type checking here
        assert isinstance(date_obj, pd.Timestamp)
        return date_obj.normalize()

    def normalize_preserve_time(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Return pandas Timestamp preserving original time."""
        if not self.can_handle(date_obj):
            raise TypeError(f"Cannot handle date object of type {type(date_obj)}")

        # MyPy needs explicit type checking here
        assert isinstance(date_obj, pd.Timestamp)
        return date_obj


class StringDateNormalizer(IDateNormalizer):
    """Normalizer for string date representations."""

    def can_handle(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> bool:
        """Check if the date object is a string."""
        return isinstance(date_obj, str)

    def normalize(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Convert string to normalized pandas Timestamp."""
        if not self.can_handle(date_obj):
            raise TypeError(f"Cannot handle date object of type {type(date_obj)}")

        ts = pd.Timestamp(date_obj)
        return ts.normalize()

    def normalize_preserve_time(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Convert string to pandas Timestamp preserving any time component."""
        if not self.can_handle(date_obj):
            raise TypeError(f"Cannot handle date object of type {type(date_obj)}")

        assert isinstance(date_obj, str)
        return pd.Timestamp(date_obj)


class DatetimeDateNormalizer(IDateNormalizer):
    """Normalizer for datetime.date objects."""

    def can_handle(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> bool:
        """Check if the date object is a datetime.date."""
        return isinstance(date_obj, dt.date)

    def normalize(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Convert datetime.date to normalized pandas Timestamp."""
        if not self.can_handle(date_obj):
            raise TypeError(f"Cannot handle date object of type {type(date_obj)}")

        ts = pd.Timestamp(date_obj)
        return ts.normalize()

    def normalize_preserve_time(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Convert datetime.date to pandas Timestamp (will be 00:00:00 by nature)."""
        if not self.can_handle(date_obj):
            raise TypeError(f"Cannot handle date object of type {type(date_obj)}")

        assert isinstance(date_obj, dt.date)
        return pd.Timestamp(date_obj)


class DateNormalizerFactory:
    """Factory for creating appropriate date normalizers."""

    def __init__(self):
        """Initialize factory with available normalizers."""
        self._normalizers = [
            TimestampDateNormalizer(),
            StringDateNormalizer(),
            DatetimeDateNormalizer(),
        ]

    def get_normalizer(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> IDateNormalizer:
        """Get the appropriate normalizer for the given date object.

        Args:
            date_obj: The date object to get a normalizer for

        Returns:
            The appropriate date normalizer

        Raises:
            TypeError: If no normalizer can handle the date object
        """
        for normalizer in self._normalizers:
            if normalizer.can_handle(date_obj):
                return normalizer

        raise TypeError(f"No normalizer available for date object of type {type(date_obj)}")

    def normalize_date(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
        """Normalize a date object using the appropriate normalizer.

        Args:
            date_obj: The date object to normalize

        Returns:
            Normalized pandas Timestamp at 00:00:00

        Raises:
            TypeError: If no normalizer can handle the date object
        """
        normalizer = self.get_normalizer(date_obj)
        return normalizer.normalize(date_obj)

    def normalize_date_to_string_key(self, date_obj: Union[str, dt.date, pd.Timestamp]) -> str:
        """Normalize a date object to ISO YYYY-MM-DD string format for cache keys.

        Args:
            date_obj: The date object to normalize

        Returns:
            ISO date string in YYYY-MM-DD format

        Raises:
            TypeError: If no normalizer can handle the date object
        """
        timestamp = self.normalize_date(date_obj)
        return timestamp.strftime("%Y-%m-%d")

    def normalize_date_preserve_time(
        self, date_obj: Union[str, dt.date, pd.Timestamp]
    ) -> pd.Timestamp:
        """Normalize a date object to Timestamp preserving time component.

        Args:
            date_obj: The date object to normalize

        Returns:
            Pandas Timestamp with original time preserved

        Raises:
            TypeError: If no normalizer can handle the date object
        """
        normalizer = self.get_normalizer(date_obj)
        return normalizer.normalize_preserve_time(date_obj)


# Global factory instance for convenience
_date_normalizer_factory = DateNormalizerFactory()


def normalize_date_polymorphic(
    date_obj: Union[str, dt.date, pd.Timestamp],
) -> pd.Timestamp:
    """Convenience function for polymorphic date normalization.

    This function replaces isinstance-based date normalization with polymorphic
    approach using the factory pattern.

    Args:
        date_obj: The date object to normalize

    Returns:
        Normalized pandas Timestamp at 00:00:00

    Raises:
        TypeError: If the date object type is not supported
    """
    return _date_normalizer_factory.normalize_date(date_obj)


def normalize_date_to_string_key_polymorphic(
    date_obj: Union[str, dt.date, pd.Timestamp],
) -> str:
    """Convenience function for polymorphic date-to-string normalization.

    Args:
        date_obj: The date object to normalize

    Returns:
        ISO date string in YYYY-MM-DD format

    Raises:
        TypeError: If the date object type is not supported
    """
    return _date_normalizer_factory.normalize_date_to_string_key(date_obj)


def normalize_date_preserve_time_polymorphic(
    date_obj: Union[str, dt.date, pd.Timestamp],
) -> pd.Timestamp:
    """Convenience function for polymorphic date normalization preserving time.

    Args:
        date_obj: The date object to normalize

    Returns:
        Pandas Timestamp with original time component preserved

    Raises:
        TypeError: If the date object type is not supported
    """
    return _date_normalizer_factory.normalize_date_preserve_time(date_obj)
