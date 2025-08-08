"""
Interface for timeout management dependencies implementing Dependency Inversion Principle.

This module provides abstractions for timeout management functionality,
enabling dependency inversion for backtester components.
"""

from abc import ABC, abstractmethod
from typing import Optional, cast


class ITimeoutManager(ABC):
    """
    Abstract interface for timeout management.

    This interface defines the contract that all timeout manager implementations
    must follow, enabling dependency inversion for backtester components.
    """

    @abstractmethod
    def check_timeout(self) -> bool:
        """
        Check if the timeout has been exceeded.

        Returns:
            True if timeout has been exceeded, False otherwise
        """
        pass

    @abstractmethod
    def reset(self, new_start_time: Optional[float] = None) -> None:
        """
        Reset the timeout timer.

        Args:
            new_start_time: Optional new start time (float), uses current time if None
        """
        pass


class ITimeoutManagerFactory(ABC):
    """
    Abstract factory interface for creating timeout manager instances.
    """

    @abstractmethod
    def create_timeout_manager(
        self, timeout_seconds: Optional[float], start_time: Optional[float] = None
    ) -> ITimeoutManager:
        """
        Create a timeout manager instance.

        Args:
            timeout_seconds: Timeout duration in seconds, None for no timeout
            start_time: Optional start time (float), uses current time if None

        Returns:
            Timeout manager instance implementing ITimeoutManager
        """
        pass


class ConcreteTimeoutManager(ITimeoutManager):
    """
    Concrete implementation of timeout manager.

    This implementation wraps the existing TimeoutManager class to provide
    the interface abstraction while maintaining all existing functionality.
    """

    def __init__(self, timeout_seconds: Optional[float], start_time: Optional[float] = None):
        """
        Initialize the concrete timeout manager.

        Args:
            timeout_seconds: Timeout duration in seconds, None for no timeout
            start_time: Optional start time (float), uses current time if None
        """
        # Import here to avoid circular dependencies
        from ..utils.timeout import TimeoutManager

        self._timeout_manager = TimeoutManager(timeout_seconds, start_time)

    def check_timeout(self) -> bool:
        """
        Check if the timeout has been exceeded.

        Returns:
            True if timeout has been exceeded, False otherwise
        """
        return cast(bool, self._timeout_manager.check_timeout())

    def reset(self, new_start_time: Optional[float] = None) -> None:
        """
        Reset the timeout timer.

        Args:
            new_start_time: Optional new start time (float), uses current time if None
        """
        self._timeout_manager.reset(new_start_time)


class ConcreteTimeoutManagerFactory(ITimeoutManagerFactory):
    """
    Concrete implementation of timeout manager factory.

    This factory creates timeout manager instances without exposing
    concrete implementation details.
    """

    def create_timeout_manager(
        self, timeout_seconds: Optional[float], start_time: Optional[float] = None
    ) -> ITimeoutManager:
        """
        Create a timeout manager instance.

        Args:
            timeout_seconds: Timeout duration in seconds, None for no timeout
            start_time: Optional start time (float), uses current time if None

        Returns:
            Timeout manager instance implementing ITimeoutManager
        """
        return ConcreteTimeoutManager(timeout_seconds, start_time)


# Factory instance for dependency injection
def create_timeout_manager_factory() -> ITimeoutManagerFactory:
    """
    Create a timeout manager factory instance.

    Returns:
        Timeout manager factory implementing ITimeoutManagerFactory
    """
    return ConcreteTimeoutManagerFactory()


def create_timeout_manager(
    timeout_seconds: Optional[float], start_time: Optional[float] = None
) -> ITimeoutManager:
    """
    Create a timeout manager instance using the factory.

    Args:
        timeout_seconds: Timeout duration in seconds, None for no timeout
        start_time: Optional start time (float), uses current time if None

    Returns:
        Timeout manager instance implementing ITimeoutManager
    """
    factory = create_timeout_manager_factory()
    return factory.create_timeout_manager(timeout_seconds, start_time)
