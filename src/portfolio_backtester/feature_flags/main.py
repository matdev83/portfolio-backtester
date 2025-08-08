"""
Main FeatureFlags facade.

This module provides the main FeatureFlags class that acts as a facade
coordinating all feature flag components while maintaining backward compatibility.
"""

import logging
from typing import Dict, Generator, cast, Callable
from contextlib import contextmanager

from .flag_registry import FlagRegistry
from .flag_context_managers import FlagContextManagers
from ..interfaces.attribute_accessor_interface import IAttributeAccessor, create_attribute_accessor

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Feature flag system facade.

    This class provides the main interface to the feature flag system,
    coordinating all components while maintaining backward compatibility
    with the original monolithic implementation.
    """

    # Class-level attribute accessor for DIP compliance
    _attribute_accessor: IAttributeAccessor = create_attribute_accessor()

    # Delegate flag access to FlagRegistry
    @classmethod
    def get_flag(cls, flag_name: str, default: bool = False) -> bool:
        """
        Generic method to get the current value of a feature flag.

        This method delegates to FlagRegistry for typed flag access
        with proper default handling.

        Args:
            flag_name (str): The snake_case name of the flag.
            default (bool): The default value if flag is unknown.

        Returns:
            bool: The current value of the flag.
        """
        if FlagRegistry.is_valid_flag(flag_name):
            method = cast(
                Callable[[], bool], cls._attribute_accessor.get_attribute(FlagRegistry, flag_name)
            )
            return method()

        # For unknown flags, fall back to basic flag store behavior
        from .flag_store import FlagStore

        return FlagStore.get_flag(flag_name, default)

    # Architecture flags
    @classmethod
    def use_new_optimization_architecture(cls) -> bool:
        """Check if new optimization architecture should be used."""
        return FlagRegistry.use_new_optimization_architecture()

    @classmethod
    def use_new_backtesting_architecture(cls) -> bool:
        """Check if new backtesting architecture should be used."""
        return FlagRegistry.use_new_backtesting_architecture()

    @classmethod
    def use_new_backtester(cls) -> bool:
        """Check if new pure backtester should be used."""
        return FlagRegistry.use_new_backtester()

    @classmethod
    def use_optimization_orchestrator(cls) -> bool:
        """Check if optimization orchestrator should be used."""
        return FlagRegistry.use_optimization_orchestrator()

    # Parameter generator flags
    @classmethod
    def enable_optuna_generator(cls) -> bool:
        """Check if Optuna parameter generator is enabled."""
        return FlagRegistry.enable_optuna_generator()

    @classmethod
    def enable_genetic_generator(cls) -> bool:
        """Check if genetic parameter generator is enabled."""
        return FlagRegistry.enable_genetic_generator()

    # Compatibility flags
    @classmethod
    def enable_backward_compatibility(cls) -> bool:
        """Check if backward compatibility layer is enabled."""
        return FlagRegistry.enable_backward_compatibility()

    @classmethod
    def enable_deprecation_warnings(cls) -> bool:
        """Check if deprecation warnings should be shown."""
        return FlagRegistry.enable_deprecation_warnings()

    # Context managers - delegate to FlagContextManagers
    @classmethod
    @contextmanager
    def disable_optuna(cls) -> Generator[None, None, None]:
        """Context manager to disable Optuna parameter generator."""
        with FlagContextManagers.disable_optuna():
            yield

    @classmethod
    @contextmanager
    def disable_genetic(cls) -> Generator[None, None, None]:
        """Context manager to disable genetic parameter generator."""
        with FlagContextManagers.disable_genetic():
            yield

    @classmethod
    @contextmanager
    def disable_backward_compatibility(cls) -> Generator[None, None, None]:
        """Context manager to temporarily disable the legacy compatibility layer."""
        with FlagContextManagers.disable_backward_compatibility():
            yield

    @classmethod
    @contextmanager
    def disable_all_optimizers(cls) -> Generator[None, None, None]:
        """Context manager to disable all optimization components."""
        with FlagContextManagers.disable_all_optimizers():
            yield

    @classmethod
    @contextmanager
    def enable_new_architecture(cls) -> Generator[None, None, None]:
        """Context manager to enable new architecture components."""
        with FlagContextManagers.enable_new_architecture():
            yield

    # Utility methods
    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """Get current state of all feature flags."""
        return FlagRegistry.get_all_flags()

    @classmethod
    def log_current_flags(cls) -> None:
        """Log current state of all feature flags for debugging."""
        flags = cls.get_all_flags()
        logger.info("Current feature flag state:")
        for flag_name, flag_value in flags.items():
            logger.info(f"  {flag_name}: {flag_value}")


# Add backward compatibility for thread-local storage access
from .flag_store import FlagStore  # noqa: E402

FeatureFlags._local = FlagStore._local  # type: ignore[attr-defined]


# Convenience functions for common flag checks
def is_new_architecture_enabled() -> bool:
    """Check if any new architecture components are enabled."""
    return (
        FeatureFlags.use_new_optimization_architecture()
        or FeatureFlags.use_new_backtester()
        or FeatureFlags.use_optimization_orchestrator()
    )


def should_show_migration_warnings() -> bool:
    """Check if migration-related warnings should be shown."""
    return (
        FeatureFlags.enable_deprecation_warnings() and FeatureFlags.enable_backward_compatibility()
    )
