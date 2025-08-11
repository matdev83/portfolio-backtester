"""
Tests for price data polymorphism interfaces.

This module tests the polymorphic price extraction and series normalization
interfaces that replace isinstance violations in price_data_utils.py.
"""

import numpy as np
import pandas as pd

from portfolio_backtester.interfaces.price_extractor_interface import (
    IPriceExtractor,
    DataFrameExtractor,
    SeriesExtractor,
    ScalarExtractor,
    PriceExtractorFactory,
)
from portfolio_backtester.interfaces.series_normalizer_interface import (
    ISeriesNormalizer,
    SeriesNormalizer,
    DataFrameNormalizer,
    SeriesNormalizerFactory,
)


class TestPriceExtractorInterface:
    """Test the price extractor interface and implementations."""

    def test_dataframe_extractor_single_column(self):
        """Test DataFrameExtractor with single column DataFrame."""
        extractor = DataFrameExtractor()
        df = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]})
        tickers = pd.Index(["AAPL", "MSFT"])

        assert extractor.can_handle(df)
        result = extractor.extract(df, tickers)

        assert isinstance(result, pd.Series)
        assert result.name == "AAPL"
        pd.testing.assert_series_equal(result, df["AAPL"])

    def test_dataframe_extractor_multiple_columns_with_close(self):
        """Test DataFrameExtractor with multiple columns including 'Close'."""
        extractor = DataFrameExtractor()
        df = pd.DataFrame(
            {
                "Open": [99.0, 100.0, 101.0],
                "Close": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
            }
        )
        tickers = pd.Index(["AAPL"])

        assert extractor.can_handle(df)
        result = extractor.extract(df, tickers)

        assert isinstance(result, pd.Series)
        assert result.name == "Close"
        pd.testing.assert_series_equal(result, df["Close"])

    def test_dataframe_extractor_multiple_columns_no_close(self):
        """Test DataFrameExtractor with multiple columns without 'Close'."""
        extractor = DataFrameExtractor()
        df = pd.DataFrame(
            {
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Volume": [1000, 1100, 1200],
            }
        )
        tickers = pd.Index(["AAPL"])

        assert extractor.can_handle(df)
        result = extractor.extract(df, tickers)

        assert isinstance(result, pd.Series)
        assert result.name == "High"  # First column
        pd.testing.assert_series_equal(result, df["High"])

    def test_dataframe_extractor_non_dataframe(self):
        """Test DataFrameExtractor with non-DataFrame data."""
        extractor = DataFrameExtractor()
        series = pd.Series([100.0, 101.0])

        assert not extractor.can_handle(series)
        assert not extractor.can_handle("not a dataframe")
        assert not extractor.can_handle(42.0)

    def test_series_extractor(self):
        """Test SeriesExtractor with pandas Series."""
        extractor = SeriesExtractor()
        series = pd.Series([100.0, 101.0, 102.0], index=["2020-01-01", "2020-01-02", "2020-01-03"])
        tickers = pd.Index(["AAPL"])

        assert extractor.can_handle(series)
        result = extractor.extract(series, tickers)

        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, series)

    def test_series_extractor_non_series(self):
        """Test SeriesExtractor with non-Series data."""
        extractor = SeriesExtractor()
        df = pd.DataFrame({"AAPL": [100.0]})

        assert not extractor.can_handle(df)
        assert not extractor.can_handle("not a series")
        assert not extractor.can_handle(42.0)

    def test_scalar_extractor_with_tickers(self):
        """Test ScalarExtractor with non-empty tickers."""
        extractor = ScalarExtractor()
        scalar_value = 100.0
        tickers = pd.Index(["AAPL", "MSFT"])

        assert extractor.can_handle(scalar_value)
        result = extractor.extract(scalar_value, tickers)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.index[0] == "AAPL"
        assert result.iloc[0] == 100.0

    def test_scalar_extractor_empty_tickers(self):
        """Test ScalarExtractor with empty tickers."""
        extractor = ScalarExtractor()
        scalar_value = 100.0
        tickers = pd.Index([])

        assert extractor.can_handle(scalar_value)
        result = extractor.extract(scalar_value, tickers)

        assert isinstance(result, pd.Series)
        assert len(result) == 0
        assert result.dtype == float

    def test_scalar_extractor_can_handle_various_types(self):
        """Test ScalarExtractor can handle various non-pandas types."""
        extractor = ScalarExtractor()

        assert extractor.can_handle(42.0)
        assert extractor.can_handle(42)
        assert extractor.can_handle("string_value")
        assert extractor.can_handle(np.float64(3.14))

        # Should not handle pandas objects
        assert not extractor.can_handle(pd.Series([1, 2, 3]))
        assert not extractor.can_handle(pd.DataFrame({"col": [1, 2, 3]}))


class TestPriceExtractorFactory:
    """Test the price extractor factory."""

    def test_factory_dataframe(self):
        """Test factory returns DataFrameExtractor for DataFrame."""
        factory = PriceExtractorFactory()
        df = pd.DataFrame({"AAPL": [100.0]})
        extractor = factory.get_extractor(df)
        assert isinstance(extractor, DataFrameExtractor)

    def test_factory_series(self):
        """Test factory returns SeriesExtractor for Series."""
        factory = PriceExtractorFactory()
        series = pd.Series([100.0])
        extractor = factory.get_extractor(series)
        assert isinstance(extractor, SeriesExtractor)

    def test_factory_scalar(self):
        """Test factory returns ScalarExtractor for scalar values."""
        factory = PriceExtractorFactory()

        extractor = factory.get_extractor(42.0)
        assert isinstance(extractor, ScalarExtractor)

        extractor = factory.get_extractor("string")
        assert isinstance(extractor, ScalarExtractor)

        extractor = factory.get_extractor(np.int64(10))
        assert isinstance(extractor, ScalarExtractor)


class TestSeriesNormalizerInterface:
    """Test the series normalizer interface and implementations."""

    def test_series_normalizer_basic(self):
        """Test SeriesNormalizer with basic Series."""
        normalizer = SeriesNormalizer()
        series = pd.Series([100.0, 101.0, 102.0], name="AAPL")

        assert normalizer.can_normalize(series)
        result = normalizer.normalize(series)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)
        assert list(result.columns) == ["AAPL"]
        pd.testing.assert_series_equal(result.iloc[:, 0], series)

    def test_series_normalizer_with_target_columns(self):
        """Test SeriesNormalizer with target columns."""
        normalizer = SeriesNormalizer()
        series = pd.Series([100.0, 101.0, 102.0], name="AAPL")
        target_cols = pd.Index(["AAPL", "MSFT", "GOOGL"])

        result = normalizer.normalize(series, target_cols)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["AAPL", "MSFT", "GOOGL"]
        # Only AAPL should have values, others should be NaN
        assert not result["AAPL"].isna().any()
        assert result["MSFT"].isna().all()
        assert result["GOOGL"].isna().all()

    def test_series_normalizer_non_series(self):
        """Test SeriesNormalizer with non-Series data."""
        normalizer = SeriesNormalizer()
        df = pd.DataFrame({"AAPL": [100.0]})

        assert not normalizer.can_normalize(df)
        assert not normalizer.can_normalize("not a series")

    def test_dataframe_normalizer_basic(self):
        """Test DataFrameNormalizer with basic DataFrame."""
        normalizer = DataFrameNormalizer()
        df = pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]})

        assert normalizer.can_normalize(df)
        result = normalizer.normalize(df)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        # Should be a copy, not the same object
        assert result is not df

    def test_dataframe_normalizer_with_target_columns(self):
        """Test DataFrameNormalizer with target columns."""
        normalizer = DataFrameNormalizer()
        df = pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]})
        target_cols = pd.Index(["AAPL", "GOOGL"])

        result = normalizer.normalize(df, target_cols)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["AAPL", "GOOGL"]
        assert not result["AAPL"].isna().any()
        assert result["GOOGL"].isna().all()

    def test_dataframe_normalizer_non_dataframe(self):
        """Test DataFrameNormalizer with non-DataFrame data."""
        normalizer = DataFrameNormalizer()
        series = pd.Series([100.0])

        assert not normalizer.can_normalize(series)
        assert not normalizer.can_normalize("not a dataframe")


class TestSeriesNormalizerFactory:
    """Test the series normalizer factory."""

    def test_factory_series(self):
        """Test factory returns SeriesNormalizer for Series."""
        factory = SeriesNormalizerFactory()
        series = pd.Series([100.0, 101.0])
        normalizer = factory.get_normalizer(series)
        assert isinstance(normalizer, SeriesNormalizer)

    def test_factory_dataframe(self):
        """Test factory returns DataFrameNormalizer for DataFrame."""
        factory = SeriesNormalizerFactory()
        df = pd.DataFrame({"AAPL": [100.0, 101.0]})
        normalizer = factory.get_normalizer(df)
        assert isinstance(normalizer, DataFrameNormalizer)

    def test_factory_unsupported_type(self):
        """Test factory returns DataFrameNormalizer for unsupported types."""
        factory = SeriesNormalizerFactory()
        normalizer = factory.get_normalizer("string value")
        assert isinstance(normalizer, DataFrameNormalizer)


class TestBackwardCompatibility:
    """Test backward compatibility of the polymorphic implementation."""

    def test_extract_current_prices_compatibility(self):
        """Test that extract_current_prices maintains backward compatibility."""
        from portfolio_backtester.utils.price_data_utils import extract_current_prices

        # Test with DataFrame (single column)
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=dates)
        tickers = pd.Index(["AAPL", "MSFT"])

        result = extract_current_prices(df, dates[0], tickers)

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.loc["AAPL"] == 100.0
        assert pd.isna(result.loc["MSFT"])

    def test_normalize_price_series_to_dataframe_compatibility(self):
        """Test that normalize_price_series_to_dataframe maintains backward compatibility."""
        from portfolio_backtester.utils.price_data_utils import normalize_price_series_to_dataframe

        # Test with Series
        series = pd.Series([100.0, 101.0, 102.0], name="AAPL")
        result = normalize_price_series_to_dataframe(series)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)
        assert list(result.columns) == ["AAPL"]

        # Test with DataFrame
        df = pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]})
        result = normalize_price_series_to_dataframe(df)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)


class TestSOLIDCompliance:
    """Test SOLID principle compliance of the polymorphic implementation."""

    def test_open_closed_principle_extensibility(self):
        """Test that new extractors can be added without modifying existing code."""

        class CustomExtractor(IPriceExtractor):
            """Custom extractor for testing extensibility."""

            def can_handle(self, data):
                return isinstance(data, dict)

            def extract(self, data, universe_tickers):
                if self.can_handle(data) and "price" in data:
                    return pd.Series(
                        [data["price"]],
                        index=[universe_tickers[0]] if not universe_tickers.empty else [],
                    )
                return pd.Series(dtype=float)

        # Create custom factory with additional extractor
        class ExtendedFactory(PriceExtractorFactory):
            def __init__(self):
                self._extractors = [
                    CustomExtractor(),  # Custom extractor first
                    DataFrameExtractor(),
                    SeriesExtractor(),
                    ScalarExtractor(),
                ]

        # Test extensibility
        factory = ExtendedFactory()
        custom_data = {"price": 150.0}
        extractor = factory.get_extractor(custom_data)
        assert isinstance(extractor, CustomExtractor)

        result = extractor.extract(custom_data, pd.Index(["AAPL"]))
        assert result.loc["AAPL"] == 150.0

    def test_single_responsibility_principle(self):
        """Test that each extractor has a single, well-defined responsibility."""
        df_extractor = DataFrameExtractor()
        series_extractor = SeriesExtractor()
        scalar_extractor = ScalarExtractor()

        # DataFrameExtractor should only handle DataFrames
        df = pd.DataFrame({"AAPL": [100.0]})
        assert df_extractor.can_handle(df)
        assert not df_extractor.can_handle(pd.Series([100.0]))

        # SeriesExtractor should only handle Series
        series = pd.Series([100.0])
        assert series_extractor.can_handle(series)
        assert not series_extractor.can_handle(df)

        # ScalarExtractor should handle non-pandas objects
        assert scalar_extractor.can_handle(100.0)
        assert not scalar_extractor.can_handle(df)
        assert not scalar_extractor.can_handle(series)

    def test_interface_segregation_principle(self):
        """Test that interfaces are focused and not bloated."""
        # IPriceExtractor interface has only two methods, both essential
        interface_methods = [
            method for method in dir(IPriceExtractor) if not method.startswith("_")
        ]

        # Should have exactly the essential methods
        expected_methods = ["can_handle", "extract"]
        assert set(interface_methods) == set(expected_methods)

        # ISeriesNormalizer interface has only two methods, both essential
        normalizer_methods = [
            method for method in dir(ISeriesNormalizer) if not method.startswith("_")
        ]

        expected_normalizer_methods = ["can_normalize", "normalize"]
        assert set(normalizer_methods) == set(expected_normalizer_methods)

    def test_dependency_inversion_principle(self):
        """Test that high-level modules depend on abstractions."""
        # Factory returns interface implementations, not concrete classes
        factory = PriceExtractorFactory()
        result = factory.get_extractor(pd.DataFrame({"AAPL": [100.0]}))
        assert isinstance(result, IPriceExtractor)

        # Factory can be substituted with different implementations
        custom_factory = PriceExtractorFactory()
        extractor = custom_factory.get_extractor(42.0)
        assert isinstance(extractor, IPriceExtractor)
