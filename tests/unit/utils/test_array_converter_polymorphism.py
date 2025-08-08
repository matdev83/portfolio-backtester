"""
Tests for array converter polymorphic interfaces.

This module tests the polymorphic interfaces that replace isinstance violations
in DataFrame to array conversion functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.portfolio_backtester.interfaces.array_converter_interface import (
    DataFrameValidator,
    NullDataFrameValidator,
    DataFrameValidatorFactory,
    MultiIndexColumnProcessor,
    SimpleColumnProcessor,
    ColumnProcessorFactory,
    PolymorphicArrayConverter,
    create_array_converter,
)


class TestDataFrameValidator:
    """Test DataFrame validator for input validation."""

    def setup_method(self):
        self.validator = DataFrameValidator()

    def test_can_validate_dataframe(self):
        """Test that validator can handle DataFrame objects."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert self.validator.can_validate(df) is True

    def test_can_validate_dataframe_like_object(self):
        """Test that validator can handle DataFrame-like objects."""
        mock_df = Mock()
        mock_df.index = pd.Index([1, 2, 3])
        mock_df.columns = pd.Index(["A", "B"])
        mock_df.values = np.array([[1, 4], [2, 5], [3, 6]])
        mock_df.sort_index = Mock()
        assert self.validator.can_validate(mock_df) is True

    def test_cannot_validate_non_dataframe(self):
        """Test that validator cannot handle non-DataFrame objects."""
        assert self.validator.can_validate("not_a_dataframe") is False
        assert self.validator.can_validate([1, 2, 3]) is False
        assert self.validator.can_validate(None) is False

    def test_validate_valid_dataframe(self):
        """Test validating a valid DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Should not raise any exception
        self.validator.validate(df)

    def test_validate_invalid_input(self):
        """Test validating invalid input."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            self.validator.validate("not_a_dataframe")


class TestNullDataFrameValidator:
    """Test null/fallback DataFrame validator."""

    def setup_method(self):
        self.validator = NullDataFrameValidator()

    def test_can_validate_always_true(self):
        """Test that validator always returns True (fallback)."""
        assert self.validator.can_validate("anything") is True
        assert self.validator.can_validate(None) is True
        assert self.validator.can_validate(123) is True

    def test_validate_always_raises(self):
        """Test that validator always raises TypeError."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            self.validator.validate("anything")

        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            self.validator.validate(None)


class TestDataFrameValidatorFactory:
    """Test DataFrame validator factory."""

    def setup_method(self):
        self.factory = DataFrameValidatorFactory()

    def test_get_validator_for_dataframe(self):
        """Test getting validator for DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        validator = self.factory.get_validator(df)
        assert isinstance(validator, DataFrameValidator)

    def test_get_validator_for_invalid_input(self):
        """Test getting validator for invalid input."""
        validator = self.factory.get_validator("not_a_dataframe")
        assert isinstance(validator, NullDataFrameValidator)


class TestMultiIndexColumnProcessor:
    """Test MultiIndex column processor."""

    def setup_method(self):
        self.processor = MultiIndexColumnProcessor()

    def test_can_process_multiindex(self):
        """Test that processor can handle MultiIndex columns."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        assert self.processor.can_process(df.columns) is True

    def test_cannot_process_simple_index(self):
        """Test that processor cannot handle simple Index columns."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert self.processor.can_process(df.columns) is False

    def test_process_columns_with_field_level(self):
        """Test processing MultiIndex with 'Field' level."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        result = self.processor.process_columns(df, field="Close")
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["A", "B"]
        assert len(result) == 5

    def test_process_columns_with_last_level_as_field(self):
        """Test processing MultiIndex with last level as field."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Data"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        result = self.processor.process_columns(df, field="Close")
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["A", "B"]

    def test_process_columns_field_required(self):
        """Test that field parameter is required for MultiIndex."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        with pytest.raises(ValueError, match="Multi-Index DataFrame requires \\*field\\* parameter"):
            self.processor.process_columns(df, field=None)

    def test_process_columns_field_not_found(self):
        """Test handling when requested field is not found."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        with pytest.raises(KeyError, match="Field 'Open' not found in DataFrame columns"):
            self.processor.process_columns(df, field="Open")

    def test_process_columns_handles_none_level_names(self):
        """Test processing MultiIndex with None level names."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=[None, None])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        # Should use default "Field" when level name is None
        result = self.processor.process_columns(df, field="Close")
        assert isinstance(result, pd.DataFrame)


class TestSimpleColumnProcessor:
    """Test simple (non-MultiIndex) column processor."""

    def setup_method(self):
        self.processor = SimpleColumnProcessor()

    def test_can_process_simple_index(self):
        """Test that processor can handle simple Index columns."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert self.processor.can_process(df.columns) is True

    def test_cannot_process_multiindex(self):
        """Test that processor cannot handle MultiIndex columns."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        assert self.processor.can_process(df.columns) is False

    def test_process_columns_returns_copy(self):
        """Test that processing returns a copy of the DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = self.processor.process_columns(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result is not df  # Should be a copy
        assert result.equals(df)  # But content should be the same

    def test_process_columns_ignores_field_parameter(self):
        """Test that field parameter is ignored for simple columns."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = self.processor.process_columns(df, field="ignored")
        
        assert isinstance(result, pd.DataFrame)
        assert result.equals(df)


class TestColumnProcessorFactory:
    """Test column processor factory."""

    def setup_method(self):
        self.factory = ColumnProcessorFactory()

    def test_get_processor_for_multiindex(self):
        """Test getting processor for MultiIndex columns."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        processor = self.factory.get_processor(columns)
        assert isinstance(processor, MultiIndexColumnProcessor)

    def test_get_processor_for_simple_index(self):
        """Test getting processor for simple Index columns."""
        columns = pd.Index(["A", "B", "C"])
        processor = self.factory.get_processor(columns)
        assert isinstance(processor, SimpleColumnProcessor)


class TestPolymorphicArrayConverter:
    """Test polymorphic array converter that eliminates isinstance violations."""

    def setup_method(self):
        self.converter = PolymorphicArrayConverter()

    def test_convert_simple_dataframe(self):
        """Test converting simple DataFrame to array."""
        df = pd.DataFrame({
            "AAPL": [100.0, 101.0, 102.0],
            "GOOGL": [1500.0, 1510.0, 1520.0],
        }, index=pd.date_range("2023-01-01", periods=3))
        
        matrix, tickers = self.converter.convert_to_array(df)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert matrix.shape == (3, 2)
        assert tickers == ["AAPL", "GOOGL"]
        np.testing.assert_array_almost_equal(matrix[0], [100.0, 1500.0])

    def test_convert_multiindex_dataframe(self):
        """Test converting MultiIndex DataFrame to array."""
        arrays = [["AAPL", "AAPL", "GOOGL", "GOOGL"], 
                  ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(
            [[100.0, 1000, 1500.0, 2000], 
             [101.0, 1100, 1510.0, 2100]], 
            columns=columns,
            index=pd.date_range("2023-01-01", periods=2)
        )
        
        matrix, tickers = self.converter.convert_to_array(df, field="Close")
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 2)
        assert tickers == ["AAPL", "GOOGL"]
        np.testing.assert_array_almost_equal(matrix[0], [100.0, 1500.0])

    def test_convert_sorts_non_monotonic_index(self):
        """Test that converter sorts non-monotonic index."""
        df = pd.DataFrame({
            "A": [1.0, 3.0, 2.0],
            "B": [4.0, 6.0, 5.0],
        }, index=pd.date_range("2023-01-01", periods=3)[::-1])  # Reverse order
        
        matrix, tickers = self.converter.convert_to_array(df)
        
        # Should be sorted by index
        assert matrix.shape == (3, 2)
        np.testing.assert_array_almost_equal(matrix[0], [2.0, 5.0])  # Last row became first
        np.testing.assert_array_almost_equal(matrix[2], [1.0, 4.0])  # First row became last

    def test_convert_handles_nan_values(self):
        """Test that converter handles NaN values correctly."""
        df = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, np.nan],
        }, index=pd.date_range("2023-01-01", periods=3))
        
        matrix, tickers = self.converter.convert_to_array(df)
        
        assert isinstance(matrix, np.ndarray)
        assert np.isnan(matrix[1, 0])  # NaN preserved
        assert np.isnan(matrix[2, 1])  # NaN preserved

    def test_convert_invalid_input_raises_error(self):
        """Test that converter raises error for invalid input."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            self.converter.convert_to_array("not_a_dataframe")

    def test_convert_multiindex_missing_field_raises_error(self):
        """Test that converter raises error when field is missing for MultiIndex."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        with pytest.raises(ValueError, match="Multi-Index DataFrame requires \\*field\\* parameter"):
            self.converter.convert_to_array(df)

    def test_convert_multiindex_field_not_found_raises_error(self):
        """Test that converter raises error when requested field is not found."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)
        
        with pytest.raises(KeyError, match="Field 'Open' not found in DataFrame columns"):
            self.converter.convert_to_array(df, field="Open")


class TestCreateArrayConverter:
    """Test factory function for creating array converter."""

    def test_create_array_converter(self):
        """Test creating array converter."""
        converter = create_array_converter()
        assert isinstance(converter, PolymorphicArrayConverter)


class TestPolymorphicIntegration:
    """Integration tests verifying isinstance violations are eliminated."""

    def test_no_isinstance_usage_in_polymorphic_converter(self):
        """Test that polymorphic converter doesn't use isinstance internally."""
        converter = create_array_converter()
        
        # Test with various DataFrame types that would have triggered isinstance checks
        test_cases = [
            # Simple DataFrame
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            # MultiIndex DataFrame
            pd.DataFrame(
                np.random.randn(3, 4),
                columns=pd.MultiIndex.from_arrays([["A", "A", "B", "B"], 
                                                   ["Close", "Volume", "Close", "Volume"]],
                                                  names=["Ticker", "Field"])
            ),
        ]
        
        # All these should work without isinstance checks
        for df in test_cases:
            if isinstance(df.columns, pd.MultiIndex):
                matrix, tickers = converter.convert_to_array(df, field="Close")
            else:
                matrix, tickers = converter.convert_to_array(df)
            
            # Basic success test
            assert isinstance(matrix, np.ndarray)
            assert isinstance(tickers, list)

    def test_polymorphic_converter_equivalent_to_original_logic(self):
        """Test that polymorphic converter produces same results as original isinstance logic."""
        converter = create_array_converter()
        
        # Test case 1: Simple DataFrame
        df_simple = pd.DataFrame({
            "AAPL": [100.0, 101.0],
            "GOOGL": [1500.0, 1510.0],
        }, index=pd.date_range("2023-01-01", periods=2))
        
        matrix, tickers = converter.convert_to_array(df_simple)
        
        # Verify results match expected behavior
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 2)
        assert tickers == ["AAPL", "GOOGL"]
        
        # Test case 2: MultiIndex DataFrame
        arrays = [["AAPL", "AAPL", "GOOGL", "GOOGL"], 
                  ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df_multi = pd.DataFrame(
            [[100.0, 1000, 1500.0, 2000], 
             [101.0, 1100, 1510.0, 2100]], 
            columns=columns,
            index=pd.date_range("2023-01-01", periods=2)
        )
        
        matrix, tickers = converter.convert_to_array(df_multi, field="Close")
        
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 2)
        assert tickers == ["AAPL", "GOOGL"]
        np.testing.assert_array_almost_equal(matrix[0], [100.0, 1500.0])

    def test_error_handling_equivalent_to_original(self):
        """Test that error handling matches original isinstance-based logic."""
        converter = create_array_converter()
        
        # Test invalid input type
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            converter.convert_to_array("not_a_dataframe")
        
        # Test MultiIndex without field
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(2, 4), columns=columns)
        
        with pytest.raises(ValueError, match="Multi-Index DataFrame requires \\*field\\* parameter"):
            converter.convert_to_array(df)
        
        # Test field not found
        with pytest.raises(KeyError, match="Field 'Nonexistent' not found in DataFrame columns"):
            converter.convert_to_array(df, field="Nonexistent")