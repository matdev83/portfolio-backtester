"""Tests for date normalizer polymorphic interfaces."""

import datetime as dt
import pytest
import pandas as pd

from portfolio_backtester.interfaces.date_normalizer_interface import (
    TimestampDateNormalizer,
    StringDateNormalizer,
    DatetimeDateNormalizer,
    DateNormalizerFactory,
    normalize_date_polymorphic,
)


class TestTimestampDateNormalizer:
    """Test TimestampDateNormalizer implementation."""

    def test_can_handle_timestamp(self):
        """Test that normalizer correctly identifies Timestamp objects."""
        normalizer = TimestampDateNormalizer()
        ts = pd.Timestamp("2023-01-15 14:30:00")

        assert normalizer.can_handle(ts) is True
        assert normalizer.can_handle("2023-01-15") is False
        assert normalizer.can_handle(dt.date(2023, 1, 15)) is False

    def test_normalize_timestamp(self):
        """Test timestamp normalization to 00:00:00."""
        normalizer = TimestampDateNormalizer()
        ts = pd.Timestamp("2023-01-15 14:30:00")

        result = normalizer.normalize(ts)
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_normalize_invalid_type_raises_error(self):
        """Test that normalizing invalid type raises TypeError."""
        normalizer = TimestampDateNormalizer()

        with pytest.raises(TypeError):
            normalizer.normalize("2023-01-15")


class TestStringDateNormalizer:
    """Test StringDateNormalizer implementation."""

    def test_can_handle_string(self):
        """Test that normalizer correctly identifies string objects."""
        normalizer = StringDateNormalizer()

        assert normalizer.can_handle("2023-01-15") is True
        assert normalizer.can_handle("2023-01-15 14:30:00") is True
        assert normalizer.can_handle(pd.Timestamp("2023-01-15")) is False
        assert normalizer.can_handle(dt.date(2023, 1, 15)) is False

    def test_normalize_string(self):
        """Test string date normalization."""
        normalizer = StringDateNormalizer()

        result = normalizer.normalize("2023-01-15")
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected
        assert result.hour == 0

    def test_normalize_string_with_time(self):
        """Test string with time normalization."""
        normalizer = StringDateNormalizer()

        result = normalizer.normalize("2023-01-15 14:30:00")
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected

    def test_normalize_invalid_type_raises_error(self):
        """Test that normalizing invalid type raises TypeError."""
        normalizer = StringDateNormalizer()

        with pytest.raises(TypeError):
            normalizer.normalize(dt.date(2023, 1, 15))


class TestDatetimeDateNormalizer:
    """Test DatetimeDateNormalizer implementation."""

    def test_can_handle_date(self):
        """Test that normalizer correctly identifies date objects."""
        normalizer = DatetimeDateNormalizer()

        assert normalizer.can_handle(dt.date(2023, 1, 15)) is True
        assert (
            normalizer.can_handle(dt.datetime(2023, 1, 15, 14, 30)) is True
        )  # datetime is subclass of date
        assert normalizer.can_handle("2023-01-15") is False
        # Note: pd.Timestamp is also a subclass of datetime, so it can be handled by DatetimeDateNormalizer
        # However, the factory will prefer TimestampDateNormalizer which comes first

    def test_normalize_date(self):
        """Test date normalization."""
        normalizer = DatetimeDateNormalizer()
        date_obj = dt.date(2023, 1, 15)

        result = normalizer.normalize(date_obj)
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected

    def test_normalize_datetime(self):
        """Test datetime normalization."""
        normalizer = DatetimeDateNormalizer()
        datetime_obj = dt.datetime(2023, 1, 15, 14, 30, 45)

        result = normalizer.normalize(datetime_obj)
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected

    def test_normalize_invalid_type_raises_error(self):
        """Test that normalizing invalid type raises TypeError."""
        normalizer = DatetimeDateNormalizer()

        with pytest.raises(TypeError):
            normalizer.normalize("2023-01-15")


class TestDateNormalizerFactory:
    """Test DateNormalizerFactory functionality."""

    def test_get_normalizer_for_timestamp(self):
        """Test factory returns correct normalizer for Timestamp."""
        factory = DateNormalizerFactory()
        ts = pd.Timestamp("2023-01-15")

        normalizer = factory.get_normalizer(ts)
        assert isinstance(normalizer, TimestampDateNormalizer)

    def test_get_normalizer_for_string(self):
        """Test factory returns correct normalizer for string."""
        factory = DateNormalizerFactory()

        normalizer = factory.get_normalizer("2023-01-15")
        assert isinstance(normalizer, StringDateNormalizer)

    def test_get_normalizer_for_date(self):
        """Test factory returns correct normalizer for date."""
        factory = DateNormalizerFactory()
        date_obj = dt.date(2023, 1, 15)

        normalizer = factory.get_normalizer(date_obj)
        assert isinstance(normalizer, DatetimeDateNormalizer)

    def test_get_normalizer_for_invalid_type(self):
        """Test factory raises error for unsupported type."""
        factory = DateNormalizerFactory()
        
        with pytest.raises(TypeError):
            factory.get_normalizer(12345)  # type: ignore[arg-type]

    def test_normalize_date_convenience_method(self):
        """Test factory convenience method for direct normalization."""
        factory = DateNormalizerFactory()

        # Test various input types
        result1 = factory.normalize_date("2023-01-15")
        result2 = factory.normalize_date(dt.date(2023, 1, 15))
        result3 = factory.normalize_date(pd.Timestamp("2023-01-15 14:30:00"))

        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result1 == expected
        assert result2 == expected
        assert result3 == expected


class TestPolymorphicConvenienceFunction:
    """Test the global convenience function."""

    def test_normalize_date_polymorphic_with_string(self):
        """Test convenience function with string input."""
        result = normalize_date_polymorphic("2023-01-15")
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected

    def test_normalize_date_polymorphic_with_date(self):
        """Test convenience function with date input."""
        date_obj = dt.date(2023, 1, 15)
        result = normalize_date_polymorphic(date_obj)
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected

    def test_normalize_date_polymorphic_with_timestamp(self):
        """Test convenience function with timestamp input."""
        ts = pd.Timestamp("2023-01-15 14:30:00")
        result = normalize_date_polymorphic(ts)
        expected = pd.Timestamp("2023-01-15 00:00:00")

        assert result == expected

    def test_normalize_date_polymorphic_with_invalid_type(self):
        """Test convenience function with invalid type."""
        with pytest.raises(TypeError):
            normalize_date_polymorphic(12345)  # type: ignore[arg-type]
