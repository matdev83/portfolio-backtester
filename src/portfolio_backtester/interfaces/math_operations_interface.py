"""
Math Operations Interface

This module provides abstract interfaces for mathematical operations,
implementing the Dependency Inversion Principle to remove direct dependencies
on built-in mathematical functions.

Key interfaces:
- IMathOperations: Core interface for mathematical operations
- IMathOperationsFactory: Factory interface for creating math operations providers

This eliminates direct dependencies on built-in functions like max, min, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union


class IMathOperations(ABC):
    """
    Abstract interface for mathematical operations.

    This interface defines the contract for mathematical operations,
    enabling dependency inversion for components that use mathematical functions.
    """

    @abstractmethod
    def max_value(
        self, *args: Union[int, float], default: Optional[Union[int, float]] = None
    ) -> Union[int, float]:
        """
        Return the maximum value from arguments.

        Args:
            *args: Variable number of numeric arguments
            default: Default value if no arguments provided

        Returns:
            Maximum value from arguments or default if no args

        Raises:
            ValueError: If no arguments and no default provided
        """
        pass

    @abstractmethod
    def min_value(
        self, *args: Union[int, float], default: Optional[Union[int, float]] = None
    ) -> Union[int, float]:
        """
        Return the minimum value from arguments.

        Args:
            *args: Variable number of numeric arguments
            default: Default value if no arguments provided

        Returns:
            Minimum value from arguments or default if no args

        Raises:
            ValueError: If no arguments and no default provided
        """
        pass

    @abstractmethod
    def max_from_iterable(
        self,
        iterable: List[Union[int, float]],
        default: Optional[Union[int, float]] = None,
    ) -> Union[int, float]:
        """
        Return the maximum value from an iterable.

        Args:
            iterable: List of numeric values
            default: Default value if iterable is empty

        Returns:
            Maximum value from iterable or default if empty

        Raises:
            ValueError: If iterable is empty and no default provided
        """
        pass

    @abstractmethod
    def min_from_iterable(
        self,
        iterable: List[Union[int, float]],
        default: Optional[Union[int, float]] = None,
    ) -> Union[int, float]:
        """
        Return the minimum value from an iterable.

        Args:
            iterable: List of numeric values
            default: Default value if iterable is empty

        Returns:
            Minimum value from iterable or default if empty

        Raises:
            ValueError: If iterable is empty and no default provided
        """
        pass

    @abstractmethod
    def abs_value(self, value: Union[int, float]) -> Union[int, float]:
        """
        Return the absolute value of a number.

        Args:
            value: Numeric value

        Returns:
            Absolute value of the input
        """
        pass


class IMathOperationsFactory(ABC):
    """
    Abstract factory interface for creating math operations providers.

    This factory enables creation of appropriate math operations implementations.
    """

    @abstractmethod
    def create_math_operations(self) -> IMathOperations:
        """
        Create a math operations provider instance.

        Returns:
            Configured math operations instance
        """
        pass


# Concrete implementations
class StandardMathOperations(IMathOperations):
    """
    Standard implementation of mathematical operations using built-in functions.

    This implementation provides standard mathematical operations while abstracting
    the direct dependency on built-in functions.
    """

    def max_value(
        self, *args: Union[int, float], default: Optional[Union[int, float]] = None
    ) -> Union[int, float]:
        """Return the maximum value from arguments."""
        if not args:
            if default is not None:
                return default
            raise ValueError("max_value() expected at least 1 argument, got 0")
        return max(args)

    def min_value(
        self, *args: Union[int, float], default: Optional[Union[int, float]] = None
    ) -> Union[int, float]:
        """Return the minimum value from arguments."""
        if not args:
            if default is not None:
                return default
            raise ValueError("min_value() expected at least 1 argument, got 0")
        return min(args)

    def max_from_iterable(
        self,
        iterable: List[Union[int, float]],
        default: Optional[Union[int, float]] = None,
    ) -> Union[int, float]:
        """Return the maximum value from an iterable."""
        if not iterable:
            if default is not None:
                return default
            raise ValueError("max() arg is an empty sequence")
        return max(iterable)

    def min_from_iterable(
        self,
        iterable: List[Union[int, float]],
        default: Optional[Union[int, float]] = None,
    ) -> Union[int, float]:
        """Return the minimum value from an iterable."""
        if not iterable:
            if default is not None:
                return default
            raise ValueError("min() arg is an empty sequence")
        return min(iterable)

    def abs_value(self, value: Union[int, float]) -> Union[int, float]:
        """Return the absolute value of a number."""
        return abs(value)


class MathOperationsFactory(IMathOperationsFactory):
    """
    Concrete factory for creating math operations providers.

    This factory creates StandardMathOperations instances that provide
    mathematical operations functionality.
    """

    def create_math_operations(self) -> IMathOperations:
        """Create a math operations provider instance."""
        return StandardMathOperations()


# Factory function for easy creation
def create_math_operations_factory() -> IMathOperationsFactory:
    """
    Create a math operations factory instance.

    Returns:
        Configured math operations factory
    """
    return MathOperationsFactory()


def create_math_operations() -> IMathOperations:
    """
    Create a math operations provider instance using the default factory.

    Returns:
        Configured math operations instance
    """
    factory = create_math_operations_factory()
    return factory.create_math_operations()
