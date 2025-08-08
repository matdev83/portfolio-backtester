"""
Interface for abstracting attribute access operations.
Implements Dependency Inversion Principle for getattr functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T")


class IAttributeAccessor(ABC):
    """
    Interface for abstracting attribute access operations.

    This interface abstracts the use of getattr() to enable better testability,
    type safety, and adherence to the Dependency Inversion Principle.
    """

    @abstractmethod
    def get_attribute(self, obj: Any, attr_name: str, default: Optional[Any] = None) -> Any:
        """
        Get an attribute from an object.

        Args:
            obj: Object to get attribute from
            attr_name: Name of the attribute to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            The attribute value or default if not found

        Raises:
            AttributeError: If attribute doesn't exist and no default provided
        """
        pass


class IModuleAttributeAccessor(ABC):
    """
    Specialized interface for module-level attribute access.
    Used for getting constants and classes from imported modules.
    """

    @abstractmethod
    def get_module_attribute(
        self, module: Any, attr_name: str, default: Optional[Any] = None
    ) -> Any:
        """
        Get an attribute from a module.

        Args:
            module: Module to get attribute from
            attr_name: Name of the attribute to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            The module attribute value or default if not found
        """
        pass


class IClassAttributeAccessor(ABC):
    """
    Specialized interface for dynamic class loading from modules.
    """

    @abstractmethod
    def get_class_from_module(self, module: Any, class_name: str) -> Type:
        """
        Get a class from a module by name.

        Args:
            module: Module to get class from
            class_name: Name of the class to retrieve

        Returns:
            The class object

        Raises:
            AttributeError: If class doesn't exist in module
            TypeError: If retrieved attribute is not a class
        """
        pass


class IObjectFieldAccessor(ABC):
    """
    Specialized interface for accessing fields from objects.
    Used for analysis and data extraction operations.
    """

    @abstractmethod
    def get_field_value(self, obj: Any, field_name: str, default: Optional[Any] = None) -> Any:
        """
        Get a field value from an object.

        Args:
            obj: Object to get field from
            field_name: Name of the field to retrieve
            default: Default value if field doesn't exist

        Returns:
            The field value or default if not found
        """
        pass


# Factory functions for creating attribute accessors


def create_attribute_accessor() -> IAttributeAccessor:
    """Create default attribute accessor implementation."""
    from .attribute_accessor_implementations import DefaultAttributeAccessor

    return DefaultAttributeAccessor()


def create_module_attribute_accessor() -> IModuleAttributeAccessor:
    """Create module attribute accessor implementation."""
    from .attribute_accessor_implementations import ModuleAttributeAccessor

    return ModuleAttributeAccessor()


def create_class_attribute_accessor() -> IClassAttributeAccessor:
    """Create class attribute accessor implementation."""
    from .attribute_accessor_implementations import ClassAttributeAccessor

    return ClassAttributeAccessor()


def create_object_field_accessor() -> IObjectFieldAccessor:
    """Create object field accessor implementation."""
    from .attribute_accessor_implementations import ObjectFieldAccessor

    return ObjectFieldAccessor()
