"""
Tests for the API stability protection decorator.
"""

import pytest
from typing import Dict, Any, Optional
from src.portfolio_backtester.api_stability import (
    validate_signature, 
    api_stable,
    ParameterViolationError,
    ReturnTypeViolationError
)


class TestValidateSignatureDecorator:
    """Test the validate_signature decorator functionality."""
    
    def test_basic_parameter_validation(self):
        """Test that basic parameter validation works."""
        
        @validate_signature(strict_params=True)
        def test_method(param1: int, param2: str = "default") -> str:
            return f"{param1}_{param2}"
        
        # Valid calls should work
        result = test_method(42, "test")
        assert result == "42_test"
        
        result = test_method(42)  # Using default
        assert result == "42_default"
        
        # Invalid type should raise error
        with pytest.raises(ParameterViolationError):
            test_method("not_an_int", "test")
    
    def test_return_type_validation(self):
        """Test that return type validation works when enabled."""
        
        @validate_signature(strict_params=False, strict_return=True)
        def test_method(value: int) -> str:
            if value > 0:
                return str(value)  # Correct return type
            else:
                return value  # Wrong return type
        
        # Valid return should work
        result = test_method(42)
        assert result == "42"
        
        # Invalid return type should raise error
        with pytest.raises(ReturnTypeViolationError):
            test_method(-1)
    
    def test_optional_parameters(self):
        """Test handling of Optional parameters."""
        
        @validate_signature(strict_params=True)
        def test_method(required: int, optional: Optional[str] = None) -> int:
            return required
        
        # Both should work
        assert test_method(42) == 42
        assert test_method(42, "test") == 42
        assert test_method(42, None) == 42
    
    def test_api_stable_alias(self):
        """Test that api_stable works as an alias."""
        
        @api_stable(version="1.0", strict_params=True)
        def test_method(param: int) -> int:
            return param * 2
        
        assert test_method(21) == 42
        
        # Check metadata is stored
        assert hasattr(test_method, '_api_stable_version')
        assert test_method._api_stable_version == "1.0"
    
    def test_no_type_hints(self):
        """Test that decorator works with methods that have no type hints."""
        
        @validate_signature(strict_params=True)
        def test_method(param1, param2="default"):
            return f"{param1}_{param2}"
        
        # Should work without type checking
        result = test_method("test", 42)
        assert result == "test_42"
    
    def test_complex_types(self):
        """Test handling of complex types like Dict, List."""
        
        @validate_signature(strict_params=True)
        def test_method(data: Dict[str, Any], count: int = 1) -> Dict[str, Any]:
            return {"data": data, "count": count}
        
        # Valid call
        result = test_method({"key": "value"}, 2)
        assert result == {"data": {"key": "value"}, "count": 2}
        
        # Invalid type for data parameter
        with pytest.raises(ParameterViolationError):
            test_method("not_a_dict", 2)
    
    def test_method_metadata_preservation(self):
        """Test that original method metadata is preserved."""
        
        @validate_signature()
        def test_method(param: int) -> int:
            """Test method docstring."""
            return param
        
        assert test_method.__name__ == "test_method"
        assert test_method.__doc__ == "Test method docstring."
        assert hasattr(test_method, '_api_stable_version')
    
    def test_permissive_mode(self):
        """Test that decorator is permissive when strict checking is disabled."""
        
        @validate_signature(strict_params=False, strict_return=False)
        def test_method(param: int) -> str:
            return param  # Wrong return type, but should be allowed
        
        # Should not raise errors
        result = test_method("not_an_int")
        assert result == "not_an_int"