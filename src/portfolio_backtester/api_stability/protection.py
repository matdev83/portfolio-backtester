"""
API stability protection decorators and validation utilities.
"""

import functools
import inspect
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from .exceptions import ParameterViolationError, ReturnTypeViolationError
from .registry import register_method

logger = logging.getLogger(__name__)


def validate_signature(
    strict_params: bool = True,
    strict_return: bool = False,
    version: str = "1.0"
) -> Callable:
    """
    Decorator that validates method signatures at runtime to prevent breaking changes.
    
    Args:
        strict_params: If True, validates parameter types and names strictly
        strict_return: If True, validates return type strictly  
        version: Version string for tracking signature changes
        
    Returns:
        Decorated function with signature validation
        
    Example:
        @validate_signature(strict_params=True, strict_return=False)
        def critical_method(param1: int, param2: str = "default") -> Dict[str, Any]:
            return {"result": param1}
    """
    def decorator(func: Callable) -> Callable:
        # Store original signature for validation
        original_sig = inspect.signature(func)
        original_params = original_sig.parameters
        
        # Get type hints if available
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError):
            type_hints = {}
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if strict_params:
                _validate_parameters(func, original_params, type_hints, args, kwargs)
                
            # Call original function
            result = func(*args, **kwargs)
            
            if strict_return and 'return' in type_hints:
                _validate_return_type(func, type_hints['return'], result)
                
            return result
            
        # Store metadata for introspection
        wrapper._api_stable_version = version  # type: ignore[attr-defined]
        wrapper._api_stable_strict_params = strict_params  # type: ignore[attr-defined]
        wrapper._api_stable_strict_return = strict_return  # type: ignore[attr-defined]
        wrapper._api_stable_original_sig = original_sig  # type: ignore[attr-defined]

        # Register the method in the central registry
        register_method(func, version, strict_params, strict_return)
        
        return wrapper
    return decorator


def api_stable(
    version: str = "1.0",
    strict_params: bool = True,
    strict_return: bool = False
) -> Callable:
    """
    Alias for validate_signature with more descriptive name.
    Marks a method as having a stable API that should not break backward compatibility.
    """
    return validate_signature(
        strict_params=strict_params,
        strict_return=strict_return,
        version=version
    )


def deprecated(
    reason: str = "",
    version: Optional[str] = None,
    removal_version: Optional[str] = None,
    migration_guide: Optional[str] = None,
    category: str = "DeprecationWarning"
) -> Callable:
    """
    Decorator to mark methods as deprecated with clear migration guidance.
    
    This decorator issues deprecation warnings when deprecated methods are called,
    providing clear guidance on how to migrate to newer alternatives.
    
    Args:
        reason: Explanation of why the method is deprecated
        version: Version in which the method was deprecated
        removal_version: Version in which the method will be removed
        migration_guide: Detailed instructions for migrating to new method
        category: Warning category (default: "DeprecationWarning")
        
    Returns:
        Decorated function that issues deprecation warnings
        
    Example:
        @deprecated(
            reason="Parameter order changed for better consistency",
            version="2.0",
            removal_version="3.0",
            migration_guide="Use new_method(param2, param1) instead of old_method(param1, param2)"
        )
        def old_method(param1: int, param2: str) -> str:
            return f"{param2}: {param1}"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build comprehensive warning message
            warning_parts = [f"Call to deprecated method '{func.__qualname__}'"]
            
            if reason:
                warning_parts.append(f"Reason: {reason}")
                
            if version:
                warning_parts.append(f"Deprecated since version {version}")
                
            if removal_version:
                warning_parts.append(f"Will be removed in version {removal_version}")
                
            if migration_guide:
                warning_parts.append(f"Migration guide: {migration_guide}")
            else:
                warning_parts.append("Please update your code to use the recommended alternative")
                
            warning_message = ". ".join(warning_parts) + "."
            
            # Issue the deprecation warning
            warnings.warn(
                warning_message,
                category=DeprecationWarning,
                stacklevel=2
            )
            
            # Log the deprecation for monitoring
            logger.warning(f"Deprecated method called: {func.__qualname__}")
            
            # Call the original function
            return func(*args, **kwargs)
            
        # Store metadata for introspection
        wrapper._deprecated = True  # type: ignore[attr-defined]
        wrapper._deprecated_reason = reason  # type: ignore[attr-defined]
        wrapper._deprecated_version = version  # type: ignore[attr-defined]
        wrapper._deprecated_removal_version = removal_version  # type: ignore[attr-defined]
        wrapper._deprecated_migration_guide = migration_guide  # type: ignore[attr-defined]
        wrapper._deprecated_category = category  # type: ignore[attr-defined]
        
        return wrapper
    return decorator


def deprecated_signature(
    old_signature: str,
    new_signature: str,
    version: Optional[str] = None,
    removal_version: Optional[str] = None,
    parameter_mapping: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Decorator specifically for methods with signature changes.
    
    This decorator is designed for methods that need parameter changes but want
    to maintain backward compatibility temporarily while warning users.
    
    Args:
        old_signature: Description of the old method signature
        new_signature: Description of the new method signature  
        version: Version in which the signature was deprecated
        removal_version: Version in which old signature support will be removed
        parameter_mapping: Dictionary mapping old parameter names to new ones
        
    Returns:
        Decorated function that warns about signature changes
        
    Example:
        @deprecated_signature(
            old_signature="method(data, format='json')",
            new_signature="method(data, output_format='json')", 
            version="2.1",
            removal_version="3.0",
            parameter_mapping={"format": "output_format"}
        )
        def method(data, output_format='json'):
            return process_data(data, output_format)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if old parameter names are being used
            if parameter_mapping:
                old_params_used = [old_param for old_param in parameter_mapping.keys() 
                                 if old_param in kwargs]
                
                if old_params_used:
                    # Build warning message for parameter name changes
                    param_warnings = []
                    for old_param in old_params_used:
                        new_param = parameter_mapping[old_param]
                        param_warnings.append(f"'{old_param}' -> '{new_param}'")
                    
                    warning_parts = [
                        f"Method '{func.__qualname__}' signature has changed",
                        f"Old signature: {old_signature}",
                        f"New signature: {new_signature}",
                        f"Parameter name changes: {', '.join(param_warnings)}"
                    ]
                    
                    if version:
                        warning_parts.append(f"Deprecated since version {version}")
                        
                    if removal_version:
                        warning_parts.append(f"Old parameter names will be removed in version {removal_version}")
                    
                    warning_message = ". ".join(warning_parts) + "."
                    
                    warnings.warn(
                        warning_message,
                        category=DeprecationWarning,
                        stacklevel=2
                    )
                    
                    # Map old parameter names to new ones
                    for old_param in old_params_used:
                        new_param = parameter_mapping[old_param]
                        kwargs[new_param] = kwargs.pop(old_param)
            
            return func(*args, **kwargs)
            
        # Store metadata for introspection
        wrapper._deprecated_signature = True  # type: ignore[attr-defined]
        wrapper._old_signature = old_signature  # type: ignore[attr-defined]
        wrapper._new_signature = new_signature  # type: ignore[attr-defined]
        wrapper._deprecated_version = version  # type: ignore[attr-defined]
        wrapper._deprecated_removal_version = removal_version  # type: ignore[attr-defined]
        wrapper._parameter_mapping = parameter_mapping  # type: ignore[attr-defined]
        
        return wrapper
    return decorator


def _validate_parameters(
    func: Callable,
    original_params: Union[Dict[str, inspect.Parameter], Any],
    type_hints: Dict[str, Type],
    args: tuple,
    kwargs: dict
) -> None:
    """
    Validates that function parameters match the original signature.
    """
    try:
        # Bind arguments to original signature
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate parameter types if type hints are available
        for param_name, value in bound_args.arguments.items():
            if param_name in type_hints and value is not None:
                expected_type = type_hints[param_name]
                if not _is_compatible_type(value, expected_type):
                    raise ParameterViolationError(
                        f"Parameter '{param_name}' in {func.__qualname__} expected type "
                        f"{expected_type}, got {type(value).__name__} with value: {value}"
                    )
                    
    except TypeError as e:
        raise ParameterViolationError(
            f"Invalid parameters for {func.__qualname__}: {str(e)}"
        )


def _validate_return_type(func: Callable, expected_type: Type, return_value: Any) -> None:
    """
    Validates that the return value matches the expected type.
    """
    if return_value is not None and not _is_compatible_type(return_value, expected_type):
        raise ReturnTypeViolationError(
            f"Return value of {func.__qualname__} expected type {expected_type}, "
            f"got {type(return_value).__name__}"
        )


def _is_compatible_type(value: Any, expected_type: Type) -> bool:
    """
    Checks if a value is compatible with the expected type.
    Handles basic types, Union types, and generic types.
    Special cases:
        • unittest.mock.Mock instances – always considered compatible.
        • types.SimpleNamespace accepted where ``argparse.Namespace`` is expected
          (helps unit-test stubs without requiring real ``argparse`` objects).
    """
    try:
        # Handle None values
        if value is None:
            return True
            
        # Handle Mock objects in tests - be permissive for testing
        if hasattr(value, '_mock_name') or str(type(value)).startswith("<class 'unittest.mock."):
            return True
        # Accept SimpleNamespace when argparse.Namespace is expected
        import types, argparse
        if expected_type is argparse.Namespace and isinstance(value, types.SimpleNamespace):
            return True
            
        # Handle basic type checking
        if hasattr(expected_type, '__origin__'):
            # Handle Union types (e.g., Optional[int] = Union[int, None])
            if expected_type.__origin__ is Union:
                return any(_is_compatible_type(value, arg) for arg in expected_type.__args__)
            # Handle generic types like List[str], Dict[str, Any]
            elif expected_type.__origin__ in (list, dict, tuple, set):
                return isinstance(value, expected_type.__origin__)
        else:
            # Simple type check
            return isinstance(value, expected_type)
            
    except (AttributeError, TypeError):
        # Fallback to basic isinstance check
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # If type checking fails, be permissive to avoid breaking existing code
            logger.warning(f"Could not validate type {expected_type} for value {value}")
            return True
            
    return True