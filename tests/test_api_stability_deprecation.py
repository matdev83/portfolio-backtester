"""
Tests for API stability deprecation decorators.
"""

import warnings
import pytest
from unittest.mock import patch
from src.portfolio_backtester.api_stability.protection import deprecated, deprecated_signature


class TestDeprecatedDecorator:
    """Test the @deprecated decorator functionality."""
    
    def test_deprecated_basic_warning(self):
        """Test that deprecated decorator issues basic warning."""
        @deprecated(reason="This method is obsolete")
        def old_method():
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_method()
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Call to deprecated method 'TestDeprecatedDecorator.test_deprecated_basic_warning.<locals>.old_method'" in str(w[0].message)
            assert "Reason: This method is obsolete" in str(w[0].message)
            assert result == "result"
    
    def test_deprecated_full_warning_message(self):
        """Test deprecated decorator with all parameters."""
        @deprecated(
            reason="Better implementation available",
            version="2.0",
            removal_version="3.0",
            migration_guide="Use new_method() instead"
        )
        def old_method():
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_method()
            
            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert "Better implementation available" in warning_msg
            assert "Deprecated since version 2.0" in warning_msg
            assert "Will be removed in version 3.0" in warning_msg
            assert "Migration guide: Use new_method() instead" in warning_msg
    
    def test_deprecated_minimal_warning(self):
        """Test deprecated decorator with minimal parameters."""
        @deprecated()
        def old_method():
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_method()
            
            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert "Call to deprecated method" in warning_msg
            assert "Please update your code to use the recommended alternative" in warning_msg
    
    def test_deprecated_preserves_function_metadata(self):
        """Test that deprecated decorator preserves function metadata."""
        @deprecated(
            reason="Test reason",
            version="1.0",
            removal_version="2.0",
            migration_guide="Test guide"
        )
        def test_function():
            """Test docstring."""
            return "test"
        
        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."
        assert test_function._deprecated is True
        assert test_function._deprecated_reason == "Test reason"
        assert test_function._deprecated_version == "1.0"
        assert test_function._deprecated_removal_version == "2.0"
        assert test_function._deprecated_migration_guide == "Test guide"
    
    def test_deprecated_with_parameters(self):
        """Test deprecated decorator works with function parameters."""
        @deprecated(reason="Parameter handling changed")
        def old_method(param1: int, param2: str = "default"):
            return f"{param2}: {param1}"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_method(42, "test")
            
            assert len(w) == 1
            assert "Parameter handling changed" in str(w[0].message)
            assert result == "test: 42"
    
    @patch('src.portfolio_backtester.api_stability.protection.logger')
    def test_deprecated_logs_warning(self, mock_logger):
        """Test that deprecated decorator logs warnings."""
        @deprecated(reason="Test logging")
        def old_method():
            return "result"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old_method()
            
        mock_logger.warning.assert_called_once()
        assert "Deprecated method called" in mock_logger.warning.call_args[0][0]


class TestDeprecatedSignatureDecorator:
    """Test the @deprecated_signature decorator functionality."""
    
    def test_deprecated_signature_basic(self):
        """Test basic deprecated_signature functionality."""
        @deprecated_signature(
            old_signature="method(data, format='json')",
            new_signature="method(data, output_format='json')",
            parameter_mapping={"format": "output_format"}
        )
        def test_method(data, output_format='json'):
            return f"{data} in {output_format}"
        
        # Test with new parameter name (no warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_method("test_data", output_format="xml")
            
            assert len(w) == 0
            assert result == "test_data in xml"
        
        # Test with old parameter name (should warn and map)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_method("test_data", format="xml")
            
            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert "signature has changed" in warning_msg
            assert "method(data, format='json')" in warning_msg
            assert "method(data, output_format='json')" in warning_msg
            assert "'format' -> 'output_format'" in warning_msg
            assert result == "test_data in xml"
    
    def test_deprecated_signature_multiple_params(self):
        """Test deprecated_signature with multiple parameter mappings."""
        @deprecated_signature(
            old_signature="method(data, format='json', verbose=True)",
            new_signature="method(data, output_format='json', show_details=True)",
            version="2.1",
            removal_version="3.0",
            parameter_mapping={"format": "output_format", "verbose": "show_details"}
        )
        def test_method(data, output_format='json', show_details=True):
            return f"{data} in {output_format}, details: {show_details}"
        
        # Test with multiple old parameter names
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_method("test", format="xml", verbose=False)
            
            assert len(w) == 1
            warning_msg = str(w[0].message)
            assert "'format' -> 'output_format'" in warning_msg
            assert "'verbose' -> 'show_details'" in warning_msg
            assert "Deprecated since version 2.1" in warning_msg
            assert "removed in version 3.0" in warning_msg
            assert result == "test in xml, details: False"
    
    def test_deprecated_signature_no_mapping_no_warning(self):
        """Test that no warning is issued when new parameters are used."""
        @deprecated_signature(
            old_signature="method(old_param)",
            new_signature="method(new_param)",
            parameter_mapping={"old_param": "new_param"}
        )
        def test_method(new_param):
            return new_param
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_method("test_value")
            
            assert len(w) == 0
            assert result == "test_value"
    
    def test_deprecated_signature_preserves_metadata(self):
        """Test that deprecated_signature preserves function metadata."""
        @deprecated_signature(
            old_signature="old_sig",
            new_signature="new_sig",
            version="1.0",
            removal_version="2.0",
            parameter_mapping={"old": "new"}
        )
        def test_function():
            """Test docstring."""
            return "test"
        
        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."
        assert test_function._deprecated_signature is True
        assert test_function._old_signature == "old_sig"
        assert test_function._new_signature == "new_sig"
        assert test_function._deprecated_version == "1.0"
        assert test_function._deprecated_removal_version == "2.0"
        assert test_function._parameter_mapping == {"old": "new"}
    
    def test_deprecated_signature_no_parameter_mapping(self):
        """Test deprecated_signature without parameter mapping."""
        @deprecated_signature(
            old_signature="method(old_way)",
            new_signature="method(new_way)"
        )
        def test_method(param):
            return param
        
        # Should not issue warnings when no parameter mapping is provided
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_method("test")
            
            assert len(w) == 0
            assert result == "test"


class TestDeprecationIntegration:
    """Test integration scenarios with deprecation decorators."""
    
    def test_deprecated_with_api_stable(self):
        """Test that deprecated can be combined with other decorators."""
        from src.portfolio_backtester.api_stability.protection import api_stable
        
        @deprecated(reason="Old implementation")
        @api_stable(version="1.0")
        def test_method(param: int) -> str:
            return str(param)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_method(42)
            
            assert len(w) == 1
            assert "Old implementation" in str(w[0].message)
            assert result == "42"
    
    def test_multiple_calls_multiple_warnings(self):
        """Test that each call to deprecated method issues a warning."""
        @deprecated(reason="Test multiple calls")
        def test_method():
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_method()
            test_method()
            test_method()
            
            assert len(w) == 3
            for warning in w:
                assert "Test multiple calls" in str(warning.message)


if __name__ == "__main__":
    pytest.main([__file__])