"""
Concrete implementations of attribute accessor interfaces.
Provides type-safe implementations following the Dependency Inversion Principle.
"""

import logging
from typing import Any, Optional, Type

from .attribute_accessor_interface import (
    IAttributeAccessor,
    IModuleAttributeAccessor,
    IClassAttributeAccessor,
    IObjectFieldAccessor,
)


class DefaultAttributeAccessor(IAttributeAccessor):
    """
    Default implementation of attribute accessor.
    Direct wrapper around getattr() function.
    """

    def get_attribute(self, obj: Any, attr_name: str, default: Optional[Any] = None) -> Any:
        """Get an attribute from an object using getattr."""
        if default is not None:
            return getattr(obj, attr_name, default)
        return getattr(obj, attr_name)


class ModuleAttributeAccessor(IModuleAttributeAccessor):
    """
    Implementation for module-level attribute access.
    Specialized for getting constants and classes from modules.
    """

    def get_module_attribute(
        self, module: Any, attr_name: str, default: Optional[Any] = None
    ) -> Any:
        """Get an attribute from a module with enhanced error handling."""
        try:
            if default is not None:
                return getattr(module, attr_name, default)
            return getattr(module, attr_name)
        except AttributeError as e:
            # Enhanced error reporting for module attributes
            module_name = getattr(module, "__name__", str(module))
            raise AttributeError(f"Module '{module_name}' has no attribute '{attr_name}'") from e


class ClassAttributeAccessor(IClassAttributeAccessor):
    """
    Implementation for dynamic class loading from modules.
    Includes validation that the retrieved attribute is actually a class.
    """

    def get_class_from_module(self, module: Any, class_name: str) -> Type:
        """Get a class from a module with validation."""
        try:
            cls = getattr(module, class_name)

            # Validate that the retrieved attribute is actually a class
            if not isinstance(cls, type):
                module_name = getattr(module, "__name__", str(module))
                raise TypeError(
                    f"'{class_name}' in module '{module_name}' is not a class "
                    f"(got {type(cls).__name__})"
                )

            return cls

        except AttributeError as e:
            module_name = getattr(module, "__name__", str(module))
            raise AttributeError(f"Module '{module_name}' has no class '{class_name}'") from e


class ObjectFieldAccessor(IObjectFieldAccessor):
    """
    Implementation for accessing fields from objects.
    Optimized for analysis and data extraction operations.
    """

    def get_field_value(self, obj: Any, field_name: str, default: Optional[Any] = None) -> Any:
        """Get a field value from an object with enhanced error context."""
        try:
            if default is not None:
                return getattr(obj, field_name, default)
            return getattr(obj, field_name)
        except AttributeError as e:
            # Enhanced error reporting for field access
            obj_type = type(obj).__name__
            raise AttributeError(f"Object of type '{obj_type}' has no field '{field_name}'") from e


# Specialized implementations for specific use cases


class LoggingLevelAccessor(IModuleAttributeAccessor):
    """
    Specialized accessor for logging level constants.
    Provides additional validation for logging level values.
    """

    _VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def get_module_attribute(
        self, module: Any, attr_name: str, default: Optional[Any] = None
    ) -> Any:
        """Get logging level with validation."""
        # Validate that we're accessing known logging levels
        if hasattr(logging, attr_name) and attr_name in self._VALID_LEVELS:
            return getattr(module, attr_name, default or logging.INFO)

        # For unknown attributes, use default behavior
        if default is not None:
            return getattr(module, attr_name, default)
        return getattr(module, attr_name)


# Factory functions for specialized implementations


def create_logging_level_accessor() -> IModuleAttributeAccessor:
    """Create specialized logging level accessor."""
    return LoggingLevelAccessor()


def create_safe_field_accessor() -> IObjectFieldAccessor:
    """
    Create a field accessor that always returns a safe default.
    Useful for analysis where missing fields should not cause errors.
    """

    class SafeFieldAccessor(ObjectFieldAccessor):
        def get_field_value(self, obj: Any, field_name: str, default: Optional[Any] = None) -> Any:
            # Always provide a safe default for analysis operations
            safe_default = default if default is not None else ""
            return getattr(obj, field_name, safe_default)

    return SafeFieldAccessor()
