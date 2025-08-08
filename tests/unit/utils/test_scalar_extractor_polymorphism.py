"""
Tests for scalar extractor polymorphism.

This module tests the polymorphic scalar extraction interfaces that replace
isinstance violations in pandas_utils.py, ensuring SOLID principle compliance.
"""

import numpy as np
import pandas as pd

from src.portfolio_backtester.interfaces.scalar_extractor_interface import (
    IScalarExtractor,
    PandasScalarExtractor,
    NumericScalarExtractor,
    NumericItemExtractor,
    NullScalarExtractor,
    ScalarExtractorFactory,
    PolymorphicScalarExtractor,
)
from src.portfolio_backtester.pandas_utils import extract_numeric_scalar


class TestScalarExtractorInterface:
    """Test the scalar extractor interface and implementations."""

    def test_pandas_scalar_extractor_series_single_element(self):
        """Test PandasScalarExtractor with single-element Series."""
        extractor = PandasScalarExtractor()
        series = pd.Series([3.14])
        
        assert extractor.can_extract(series)
        result = extractor.extract_scalar(series)
        assert result == 3.14

    def test_pandas_scalar_extractor_dataframe_single_element(self):
        """Test PandasScalarExtractor with single-element DataFrame."""
        extractor = PandasScalarExtractor()
        df = pd.DataFrame({'value': [2.718]})
        
        assert extractor.can_extract(df)
        result = extractor.extract_scalar(df)
        assert result == 2.718

    def test_pandas_scalar_extractor_series_multiple_elements(self):
        """Test PandasScalarExtractor with multi-element Series returns None."""
        extractor = PandasScalarExtractor()
        series = pd.Series([1.0, 2.0, 3.0])
        
        assert extractor.can_extract(series)
        result = extractor.extract_scalar(series)
        assert result is None

    def test_pandas_scalar_extractor_series_with_nan(self):
        """Test PandasScalarExtractor with NaN value returns None."""
        extractor = PandasScalarExtractor()
        series = pd.Series([np.nan])
        
        assert extractor.can_extract(series)
        result = extractor.extract_scalar(series)
        assert result is None

    def test_pandas_scalar_extractor_empty_series(self):
        """Test PandasScalarExtractor with empty Series returns None."""
        extractor = PandasScalarExtractor()
        series = pd.Series([], dtype=float)
        
        assert extractor.can_extract(series)
        result = extractor.extract_scalar(series)
        assert result is None

    def test_pandas_scalar_extractor_non_pandas_value(self):
        """Test PandasScalarExtractor with non-pandas value."""
        extractor = PandasScalarExtractor()
        value = 42.0
        
        assert not extractor.can_extract(value)
        result = extractor.extract_scalar(value)
        assert result is None

    def test_numeric_scalar_extractor_float(self):
        """Test NumericScalarExtractor with float value."""
        extractor = NumericScalarExtractor()
        value = 3.14159
        
        assert extractor.can_extract(value)
        result = extractor.extract_scalar(value)
        assert result == 3.14159

    def test_numeric_scalar_extractor_int(self):
        """Test NumericScalarExtractor with int value."""
        extractor = NumericScalarExtractor()
        value = 42
        
        assert extractor.can_extract(value)
        result = extractor.extract_scalar(value)
        assert result == 42.0

    def test_numeric_scalar_extractor_numpy_types(self):
        """Test NumericScalarExtractor with numpy scalar types."""
        extractor = NumericScalarExtractor()
        
        # Test numpy float
        np_float = np.float64(2.718)
        assert extractor.can_extract(np_float)
        assert extractor.extract_scalar(np_float) == 2.718
        
        # Test numpy int
        np_int = np.int32(100)
        assert extractor.can_extract(np_int)
        assert extractor.extract_scalar(np_int) == 100.0

    def test_numeric_scalar_extractor_nan_value(self):
        """Test NumericScalarExtractor with NaN value returns None."""
        extractor = NumericScalarExtractor()
        value = float('nan')
        
        assert extractor.can_extract(value)
        result = extractor.extract_scalar(value)
        assert result is None

    def test_numeric_scalar_extractor_non_numeric(self):
        """Test NumericScalarExtractor with non-numeric value."""
        extractor = NumericScalarExtractor()
        value = "not a number"
        
        assert not extractor.can_extract(value)
        result = extractor.extract_scalar(value)
        assert result is None

    def test_numeric_item_extractor(self):
        """Test NumericItemExtractor for numpy array items."""
        extractor = NumericItemExtractor()
        
        # Test with various numeric types
        assert extractor.can_extract(np.float64(1.5))
        assert extractor.extract_scalar(np.float64(1.5)) == 1.5
        
        assert extractor.can_extract(5)
        assert extractor.extract_scalar(5) == 5.0
        
        assert extractor.can_extract(3.14)
        assert extractor.extract_scalar(3.14) == 3.14

    def test_null_scalar_extractor(self):
        """Test NullScalarExtractor always returns None."""
        extractor = NullScalarExtractor()
        
        # Should accept any value
        assert extractor.can_extract("anything")
        assert extractor.can_extract(42)
        assert extractor.can_extract(pd.Series([1, 2, 3]))
        
        # But always return None
        assert extractor.extract_scalar("anything") is None
        assert extractor.extract_scalar(42) is None
        assert extractor.extract_scalar(pd.Series([1, 2, 3])) is None


class TestScalarExtractorFactory:
    """Test the scalar extractor factory."""

    def test_factory_pandas_series(self):
        """Test factory returns PandasScalarExtractor for Series."""
        factory = ScalarExtractorFactory()
        series = pd.Series([1.0])
        extractor = factory.get_extractor(series)
        assert isinstance(extractor, PandasScalarExtractor)

    def test_factory_pandas_dataframe(self):
        """Test factory returns PandasScalarExtractor for DataFrame."""
        factory = ScalarExtractorFactory()
        df = pd.DataFrame({'col': [1.0]})
        extractor = factory.get_extractor(df)
        assert isinstance(extractor, PandasScalarExtractor)

    def test_factory_numeric_scalar(self):
        """Test factory returns NumericScalarExtractor for numeric values."""
        factory = ScalarExtractorFactory()
        
        extractor = factory.get_extractor(42.0)
        assert isinstance(extractor, NumericScalarExtractor)
        
        extractor = factory.get_extractor(np.int64(10))
        assert isinstance(extractor, NumericScalarExtractor)

    def test_factory_unsupported_type(self):
        """Test factory returns NullScalarExtractor for unsupported types."""
        factory = ScalarExtractorFactory()
        extractor = factory.get_extractor("string value")
        assert isinstance(extractor, NullScalarExtractor)


class TestPolymorphicScalarExtractor:
    """Test the main polymorphic scalar extractor."""

    def test_polymorphic_extractor_pandas_series(self):
        """Test polymorphic extractor with pandas Series."""
        extractor = PolymorphicScalarExtractor()
        series = pd.Series([42.0])
        result = extractor.extract_numeric_scalar(series)
        assert result == 42.0

    def test_polymorphic_extractor_numeric_scalar(self):
        """Test polymorphic extractor with numeric scalar."""
        extractor = PolymorphicScalarExtractor()
        result = extractor.extract_numeric_scalar(3.14)
        assert result == 3.14

    def test_polymorphic_extractor_unsupported_type(self):
        """Test polymorphic extractor with unsupported type."""
        extractor = PolymorphicScalarExtractor()
        result = extractor.extract_numeric_scalar("not a number")
        assert result is None

    def test_polymorphic_extractor_custom_factory(self):
        """Test polymorphic extractor with custom factory."""
        # Create a custom factory for testing
        custom_factory = ScalarExtractorFactory()
        extractor = PolymorphicScalarExtractor(factory=custom_factory)
        
        result = extractor.extract_numeric_scalar(2.718)
        assert result == 2.718


class TestBackwardCompatibility:
    """Test backward compatibility of the updated pandas_utils.extract_numeric_scalar."""

    def test_extract_numeric_scalar_series_single_element(self):
        """Test extract_numeric_scalar maintains compatibility with Series."""
        series = pd.Series([3.14])
        result = extract_numeric_scalar(series)
        assert result == 3.14

    def test_extract_numeric_scalar_dataframe_single_element(self):
        """Test extract_numeric_scalar maintains compatibility with DataFrame."""
        df = pd.DataFrame({'value': [2.718]})
        result = extract_numeric_scalar(df)
        assert result == 2.718

    def test_extract_numeric_scalar_float(self):
        """Test extract_numeric_scalar maintains compatibility with float."""
        result = extract_numeric_scalar(42.5)
        assert result == 42.5

    def test_extract_numeric_scalar_int(self):
        """Test extract_numeric_scalar maintains compatibility with int."""
        result = extract_numeric_scalar(100)
        assert result == 100.0

    def test_extract_numeric_scalar_numpy_types(self):
        """Test extract_numeric_scalar maintains compatibility with numpy types."""
        result = extract_numeric_scalar(np.float64(1.618))
        assert result == 1.618
        
        result = extract_numeric_scalar(np.int32(50))
        assert result == 50.0

    def test_extract_numeric_scalar_nan_values(self):
        """Test extract_numeric_scalar handles NaN values correctly."""
        assert extract_numeric_scalar(float('nan')) is None
        assert extract_numeric_scalar(pd.Series([np.nan])) is None

    def test_extract_numeric_scalar_multiple_elements(self):
        """Test extract_numeric_scalar returns None for multi-element containers."""
        series = pd.Series([1.0, 2.0, 3.0])
        assert extract_numeric_scalar(series) is None
        
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert extract_numeric_scalar(df) is None

    def test_extract_numeric_scalar_empty_containers(self):
        """Test extract_numeric_scalar handles empty containers."""
        empty_series = pd.Series([], dtype=float)
        assert extract_numeric_scalar(empty_series) is None
        
        empty_df = pd.DataFrame()
        assert extract_numeric_scalar(empty_df) is None

    def test_extract_numeric_scalar_unsupported_types(self):
        """Test extract_numeric_scalar returns None for unsupported types."""
        assert extract_numeric_scalar("string") is None
        assert extract_numeric_scalar([1, 2, 3]) is None
        assert extract_numeric_scalar({'key': 'value'}) is None


class TestSOLIDCompliance:
    """Test SOLID principle compliance of the polymorphic implementation."""

    def test_open_closed_principle_extensibility(self):
        """Test that new extractors can be added without modifying existing code."""
        
        class CustomScalarExtractor(IScalarExtractor):
            """Custom extractor for testing extensibility."""
            
            def can_extract(self, value):
                return isinstance(value, str) and value.isdigit()
            
            def extract_scalar(self, value):
                if self.can_extract(value):
                    return float(value)
                return None
        
        # Create custom factory with additional extractor
        class ExtendedFactory(ScalarExtractorFactory):
            _extractors = [
                CustomScalarExtractor(),  # Custom extractor first
                PandasScalarExtractor(),
                NumericScalarExtractor(),
                NullScalarExtractor(),
            ]
        
        # Test extensibility
        extended_extractor = PolymorphicScalarExtractor(factory=ExtendedFactory())
        result = extended_extractor.extract_numeric_scalar("123")
        assert result == 123.0
        
        # Verify normal functionality still works
        result = extended_extractor.extract_numeric_scalar(45.6)
        assert result == 45.6

    def test_single_responsibility_principle(self):
        """Test that each extractor has a single, well-defined responsibility."""
        pandas_extractor = PandasScalarExtractor()
        numeric_extractor = NumericScalarExtractor()
        null_extractor = NullScalarExtractor()
        
        # PandasScalarExtractor should only handle pandas objects
        assert pandas_extractor.can_extract(pd.Series([1]))
        assert not pandas_extractor.can_extract(42.0)
        
        # NumericScalarExtractor should only handle numeric scalars
        assert numeric_extractor.can_extract(42.0)
        assert not numeric_extractor.can_extract(pd.Series([1]))
        
        # NullScalarExtractor accepts everything but always returns None
        assert null_extractor.can_extract("anything")
        assert null_extractor.extract_scalar("anything") is None

    def test_interface_segregation_principle(self):
        """Test that the interface is focused and not bloated."""
        # The IScalarExtractor interface has only two methods, both essential
        interface_methods = [method for method in dir(IScalarExtractor) 
                           if not method.startswith('_')]
        
        # Should have exactly the essential methods
        expected_methods = ['can_extract', 'extract_scalar']
        assert set(interface_methods) == set(expected_methods)

    def test_dependency_inversion_principle(self):
        """Test that high-level modules depend on abstractions."""
        # PolymorphicScalarExtractor depends on the factory interface
        extractor = PolymorphicScalarExtractor()
        
        # The factory returns interface implementations, not concrete classes
        factory = ScalarExtractorFactory()
        result = factory.get_extractor(42.0)
        assert isinstance(result, IScalarExtractor)
        
        # Can substitute different factory implementations
        custom_factory = ScalarExtractorFactory()
        custom_extractor = PolymorphicScalarExtractor(factory=custom_factory)
        assert custom_extractor.extract_numeric_scalar(1.5) == 1.5