"""
Context managers for feature flag testing scenarios.

This module provides context managers for temporarily modifying
feature flags during testing and component isolation scenarios.
"""

import logging
from typing import Dict, Optional, Generator
from contextlib import contextmanager

from .flag_store import FlagStore

logger = logging.getLogger(__name__)


class FlagContextManagers:
    """
    Provides context managers for temporarily modifying feature flags.

    This class handles all context managers used for testing component
    isolation and architecture migration scenarios.
    """

    @classmethod
    @contextmanager
    def disable_optuna(cls) -> Generator[None, None, None]:
        """
        Context manager to disable Optuna parameter generator.

        This is used for testing component isolation to prove that
        other components work without Optuna dependencies.

        Yields:
            None
        """
        flag_name = "enable_optuna_generator"
        original_value = FlagStore.get_override(flag_name)

        try:
            FlagStore.set_override(flag_name, False)
            logger.debug("Optuna parameter generator disabled via context manager")
            yield
        finally:
            if original_value is not None:
                FlagStore.set_override(flag_name, original_value)
            else:
                FlagStore.clear_override(flag_name)
            logger.debug("Optuna parameter generator context manager restored")

    @classmethod
    @contextmanager
    def disable_genetic(cls) -> Generator[None, None, None]:
        """
        Context manager to disable genetic parameter generator.

        This is used for testing component isolation to prove that
        other components work without PyGAD dependencies.

        Yields:
            None
        """
        flag_name = "enable_genetic_generator"
        original_value = FlagStore.get_override(flag_name)

        try:
            FlagStore.set_override(flag_name, False)
            logger.debug("Genetic parameter generator disabled via context manager")
            yield
        finally:
            if original_value is not None:
                FlagStore.set_override(flag_name, original_value)
            else:
                FlagStore.clear_override(flag_name)
            logger.debug("Genetic parameter generator context manager restored")

    @classmethod
    @contextmanager
    def disable_backward_compatibility(cls) -> Generator[None, None, None]:
        """
        Context manager to temporarily disable the legacy compatibility layer.
        """
        flag_name = "enable_backward_compatibility"
        original_value = FlagStore.get_override(flag_name)
        try:
            FlagStore.set_override(flag_name, False)
            logger.debug("Backward compatibility disabled via context manager")
            yield
        finally:
            if original_value is not None:
                FlagStore.set_override(flag_name, original_value)
            else:
                FlagStore.clear_override(flag_name)
            logger.debug("Backward compatibility context manager restored")

    @classmethod
    @contextmanager
    def disable_all_optimizers(cls) -> Generator[None, None, None]:
        """
        Context manager to disable all optimization components.

        This is used for testing that the pure backtester works
        in complete isolation from optimization components.

        Yields:
            None
        """
        flags_to_disable = [
            "enable_optuna_generator",
            "enable_genetic_generator",
            "use_optimization_orchestrator",
            "use_new_optimization_architecture",
        ]

        original_values: Dict[str, Optional[bool]] = {}

        try:
            for flag_name in flags_to_disable:
                original_values[flag_name] = FlagStore.get_override(flag_name)
                FlagStore.set_override(flag_name, False)

            logger.debug("All optimization components disabled via context manager")
            yield

        finally:
            for flag_name in flags_to_disable:
                if original_values[flag_name] is not None:
                    FlagStore.set_override(flag_name, bool(original_values[flag_name]))
                else:
                    FlagStore.clear_override(flag_name)

            logger.debug("All optimization components context manager restored")

    @classmethod
    @contextmanager
    def enable_new_architecture(cls) -> Generator[None, None, None]:
        """
        Context manager to enable new architecture components.

        This is used for testing the new architecture in isolation
        or for gradual rollout scenarios.

        Yields:
            None
        """
        flags_to_enable = [
            "use_new_optimization_architecture",
            "use_new_backtester",
            "use_optimization_orchestrator",
        ]

        original_values: Dict[str, Optional[bool]] = {}

        try:
            for flag_name in flags_to_enable:
                original_values[flag_name] = FlagStore.get_override(flag_name)
                FlagStore.set_override(flag_name, True)

            logger.debug("New architecture components enabled via context manager")
            yield

        finally:
            for flag_name in flags_to_enable:
                if original_values[flag_name] is not None:
                    FlagStore.set_override(flag_name, bool(original_values[flag_name]))
                else:
                    FlagStore.clear_override(flag_name)

            logger.debug("New architecture components context manager restored")

    @classmethod
    @contextmanager
    def override_flags(cls, flag_overrides: Dict[str, bool]) -> Generator[None, None, None]:
        """
        Generic context manager to override multiple flags.

        Args:
            flag_overrides: Dictionary of flag names and their temporary values

        Yields:
            None
        """
        original_values: Dict[str, Optional[bool]] = {}

        try:
            for flag_name, value in flag_overrides.items():
                original_values[flag_name] = FlagStore.get_override(flag_name)
                FlagStore.set_override(flag_name, value)

            logger.debug(f"Flags overridden via context manager: {flag_overrides}")
            yield

        finally:
            for flag_name in flag_overrides:
                if original_values[flag_name] is not None:
                    FlagStore.set_override(flag_name, bool(original_values[flag_name]))
                else:
                    FlagStore.clear_override(flag_name)

            logger.debug("Flag override context manager restored")
