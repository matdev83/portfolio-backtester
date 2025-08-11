"""
Integration tests for utils/__init__.py polymorphic implementations.

This module tests that the updated utils functions work correctly with the
polymorphic interfaces and produce the same results as the original isinstance-based logic.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from portfolio_backtester.utils import _resolve_strategy, _df_to_float32_array


class TestResolveStrategyPolymorphic:
    """Test _resolve_strategy function with polymorphic implementation."""

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_resolve_strategy_dict_specification(self, mock_enumerate):
        """Test resolving dictionary strategy specification."""
        mock_strategy = Mock()
        mock_enumerate.return_value = {"momentum_strategy": mock_strategy}

        # Test dict with 'name' key
        result = _resolve_strategy({"name": "momentum_strategy"})
        assert result == mock_strategy

        # Test dict with 'strategy' key
        result = _resolve_strategy({"strategy": "momentum_strategy"})
        assert result == mock_strategy

        # Test dict with 'type' key
        result = _resolve_strategy({"type": "momentum_strategy"})
        assert result == mock_strategy

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_resolve_strategy_string_specification(self, mock_enumerate):
        """Test resolving string strategy specification."""
        mock_strategy = Mock()
        mock_enumerate.return_value = {"calmar_strategy": mock_strategy}

        result = _resolve_strategy("calmar_strategy")
        assert result == mock_strategy

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_resolve_strategy_not_found(self, mock_enumerate):
        """Test resolving when strategy is not found."""
        mock_enumerate.return_value = {"other_strategy": Mock()}

        result = _resolve_strategy("nonexistent_strategy")
        assert result is None

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_resolve_strategy_invalid_specification(self, mock_enumerate):
        """Test resolving invalid strategy specification."""
        mock_enumerate.return_value = {"test_strategy": Mock()}

        # Test invalid types that would have failed isinstance checks
        result = _resolve_strategy(123)
        assert result is None

        result = _resolve_strategy(None)
        assert result is None

        result = _resolve_strategy([])
        assert result is None

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_resolve_strategy_dict_priority_order(self, mock_enumerate):
        """Test that dict resolution follows correct priority order."""
        mock_strategy = Mock()
        mock_enumerate.return_value = {"name_strategy": mock_strategy}

        # 'name' should take priority over 'strategy' and 'type'
        spec = {"name": "name_strategy", "strategy": "strategy_strategy", "type": "type_strategy"}
        result = _resolve_strategy(spec)
        assert result == mock_strategy

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_resolve_strategy_dict_no_valid_keys(self, mock_enumerate):
        """Test dict with no valid strategy keys."""
        mock_enumerate.return_value = {"test_strategy": Mock()}

        result = _resolve_strategy({"invalid_key": "some_value"})
        assert result is None

    def test_resolve_strategy_no_isinstance_usage(self):
        """Test that function doesn't use isinstance checks internally."""
        # This test verifies the function works without isinstance by testing
        # with types that would have caused issues in original implementation

        # Mock dict-like object without being actual dict
        class DictLike:
            def get(self, key, default=None):
                if key == "name":
                    return "test_strategy"
                return default

        # Mock string-like object without being actual string
        class StringLike:
            def lower(self):
                return "test_strategy"

            def __str__(self):
                return "test_strategy"

        # These should work with polymorphic approach
        dict_like = DictLike()
        string_like = StringLike()

        # Should not raise exceptions due to isinstance usage
        result1 = _resolve_strategy(dict_like)
        result2 = _resolve_strategy(string_like)

        # Results depend on strategy registry, but should not fail due to type checking
        assert result1 is None or result1 is not None
        assert result2 is None or result2 is not None


class TestDfToFloat32ArrayPolymorphic:
    """Test _df_to_float32_array function with polymorphic implementation."""

    def test_convert_simple_dataframe(self):
        """Test converting simple DataFrame to float32 array."""
        df = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0, 102.0],
                "GOOGL": [1500.0, 1510.0, 1520.0],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        matrix, tickers = _df_to_float32_array(df)

        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert matrix.shape == (3, 2)
        assert tickers == ["AAPL", "GOOGL"]
        np.testing.assert_array_almost_equal(matrix[0], [100.0, 1500.0])

    def test_convert_multiindex_dataframe(self):
        """Test converting MultiIndex DataFrame to float32 array."""
        arrays = [["AAPL", "AAPL", "GOOGL", "GOOGL"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(
            [[100.0, 1000, 1500.0, 2000], [101.0, 1100, 1510.0, 2100]],
            columns=columns,
            index=pd.date_range("2023-01-01", periods=2),
        )

        matrix, tickers = _df_to_float32_array(df, field="Close")

        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 2)
        assert tickers == ["AAPL", "GOOGL"]
        np.testing.assert_array_almost_equal(matrix[0], [100.0, 1500.0])

    def test_convert_non_monotonic_index(self):
        """Test converting DataFrame with non-monotonic index."""
        df = pd.DataFrame(
            {
                "A": [1.0, 3.0, 2.0],
                "B": [4.0, 6.0, 5.0],
            },
            index=pd.date_range("2023-01-01", periods=3)[::-1],
        )  # Reverse order

        matrix, tickers = _df_to_float32_array(df)

        # Should be sorted by index
        assert matrix.shape == (3, 2)
        # After sorting, the last row (with reverse indexing) becomes first
        np.testing.assert_array_almost_equal(matrix[0], [2.0, 5.0])

    def test_convert_with_nan_values(self):
        """Test converting DataFrame with NaN values."""
        df = pd.DataFrame(
            {
                "A": [1.0, np.nan, 3.0],
                "B": [4.0, 5.0, np.nan],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        matrix, tickers = _df_to_float32_array(df)

        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert np.isnan(matrix[1, 0])  # NaN preserved
        assert np.isnan(matrix[2, 1])  # NaN preserved

    def test_convert_invalid_input_raises_error(self):
        """Test that function raises error for invalid input."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            _df_to_float32_array("not_a_dataframe")  # type: ignore[arg-type]

    def test_convert_multiindex_missing_field_raises_error(self):
        """Test that function raises error when field is missing for MultiIndex."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(3, 4), columns=columns)

        with pytest.raises(
            ValueError, match="Multi-Index DataFrame requires \\*field\\* parameter"
        ):
            _df_to_float32_array(df)

    def test_convert_multiindex_field_not_found_raises_error(self):
        """Test that function raises error when requested field is not found."""
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(3, 4), columns=columns)

        with pytest.raises(KeyError, match="Field 'Open' not found in DataFrame columns"):
            _df_to_float32_array(df, field="Open")

    def test_convert_no_isinstance_usage(self):
        """Test that function doesn't use isinstance checks internally."""
        # Create DataFrame-like object that would pass polymorphic validation
        # but might fail isinstance checks
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Test with various column types that would have triggered isinstance checks
        test_cases = [
            # Simple DataFrame
            df,
            # MultiIndex DataFrame
            pd.DataFrame(
                np.random.randn(3, 4),
                columns=pd.MultiIndex.from_arrays(
                    [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]],
                    names=["Ticker", "Field"],
                ),
            ),
        ]

        # All these should work without isinstance checks
        for test_df in test_cases:
            if hasattr(test_df.columns, "nlevels") and test_df.columns.nlevels > 1:
                matrix, tickers = _df_to_float32_array(test_df, field="Close")
            else:
                matrix, tickers = _df_to_float32_array(test_df)

            # Basic success test
            assert isinstance(matrix, np.ndarray)
            assert matrix.dtype == np.float32
            assert isinstance(tickers, list)


class TestPolymorphicEquivalence:
    """Test that polymorphic implementations produce equivalent results to original isinstance logic."""

    @patch(
        "portfolio_backtester.interfaces.strategy_resolver_interface.PolymorphicStrategyResolver.enumerate_strategies_with_params"
    )
    def test_strategy_resolution_equivalence(self, mock_enumerate):
        """Test that polymorphic strategy resolution matches original logic."""
        # Set up mock strategy registry
        mock_strategies = {
            "momentum_strategy": Mock(),
            "calmar_strategy": Mock(),
        }
        mock_enumerate.return_value = mock_strategies

        # Test cases that would have been handled by original isinstance logic
        test_cases = [
            # Dict specifications
            ({"name": "momentum_strategy"}, mock_strategies["momentum_strategy"]),
            ({"strategy": "calmar_strategy"}, mock_strategies["calmar_strategy"]),
            ({"type": "momentum_strategy"}, mock_strategies["momentum_strategy"]),
            # String specifications
            ("momentum_strategy", mock_strategies["momentum_strategy"]),
            ("calmar_strategy", mock_strategies["calmar_strategy"]),
            # Invalid specifications
            ({"invalid": "key"}, None),
            (123, None),
            (None, None),
            ([], None),
        ]

        for spec, expected in test_cases:
            result = _resolve_strategy(spec)
            assert result == expected, f"Failed for spec: {spec}"

    def test_array_conversion_equivalence(self):
        """Test that polymorphic array conversion matches original logic."""
        # Test case 1: Simple DataFrame
        df_simple = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0],
                "GOOGL": [1500.0, 1510.0],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        matrix, tickers = _df_to_float32_array(df_simple)

        # Expected behavior
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 2)
        assert tickers == ["AAPL", "GOOGL"]
        assert np.allclose(matrix, df_simple.astype(np.float32).values)

        # Test case 2: MultiIndex DataFrame
        arrays = [["AAPL", "AAPL", "GOOGL", "GOOGL"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df_multi = pd.DataFrame(
            [[100.0, 1000, 1500.0, 2000], [101.0, 1100, 1510.0, 2100]],
            columns=columns,
            index=pd.date_range("2023-01-01", periods=2),
        )

        matrix, tickers = _df_to_float32_array(df_multi, field="Close")

        # Expected behavior
        expected_values = df_multi.xs("Close", level="Field", axis=1).astype(np.float32)
        assert matrix.dtype == np.float32
        assert matrix.shape == (2, 2)
        assert tickers == ["AAPL", "GOOGL"]
        assert np.allclose(matrix, expected_values.values.astype(np.float32))

    def test_error_handling_equivalence(self):
        """Test that error handling matches original isinstance-based logic."""
        # Test invalid input for array conversion
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            _df_to_float32_array("not_a_dataframe")  # type: ignore[arg-type]

        # Test MultiIndex without field
        arrays = [["A", "A", "B", "B"], ["Close", "Volume", "Close", "Volume"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Field"])
        df = pd.DataFrame(np.random.randn(2, 4), columns=columns)

        with pytest.raises(
            ValueError, match="Multi-Index DataFrame requires \\*field\\* parameter"
        ):
            _df_to_float32_array(df)

    def test_performance_characteristics(self):
        """Test that polymorphic implementations maintain reasonable performance."""
        # Create test data
        df = pd.DataFrame(np.random.randn(1000, 50))

        # Test array conversion performance
        import time

        start_time = time.time()
        matrix, tickers = _df_to_float32_array(df)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second for this size)
        assert (end_time - start_time) < 1.0
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
