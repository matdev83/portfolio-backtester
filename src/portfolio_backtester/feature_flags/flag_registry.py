"""
Flag registry defining all available feature flags.

This module defines all feature flags available in the system,
their default values, and provides typed access methods.
"""

from typing import Dict
from .flag_store import FlagStore


class FlagRegistry:
    """
    Registry of all available feature flags and their configurations.

    This class centralizes the definition of all feature flags,
    their default values, and provides typed access methods.
    """

    # Define all flags and their defaults
    _FLAG_DEFINITIONS = {
        # Architecture flags - mostly default to True for new architecture
        "use_new_optimization_architecture": True,
        "use_new_backtesting_architecture": True,
        "use_new_backtester": False,  # Gradual rollout
        "use_optimization_orchestrator": False,  # Gradual rollout
        # Parameter generator availability
        "enable_optuna_generator": True,
        "enable_genetic_generator": True,
        # Compatibility and warnings
        "enable_backward_compatibility": True,
        "enable_deprecation_warnings": True,
    }

    @classmethod
    def get_flag_default(cls, flag_name: str) -> bool:
        """Get the default value for a flag."""
        if flag_name not in cls._FLAG_DEFINITIONS:
            raise ValueError(f"Unknown flag: {flag_name}")
        return cls._FLAG_DEFINITIONS[flag_name]

    @classmethod
    def get_all_flag_names(cls) -> list[str]:
        """Get list of all available flag names."""
        return list(cls._FLAG_DEFINITIONS.keys())

    @classmethod
    def is_valid_flag(cls, flag_name: str) -> bool:
        """Check if a flag name is valid."""
        return flag_name in cls._FLAG_DEFINITIONS

    # Architecture flags
    @classmethod
    def use_new_optimization_architecture(cls) -> bool:
        """Check if new optimization architecture should be used."""
        return FlagStore.get_flag(
            "use_new_optimization_architecture",
            cls.get_flag_default("use_new_optimization_architecture"),
        )

    @classmethod
    def use_new_backtesting_architecture(cls) -> bool:
        """Check if new backtesting architecture should be used."""
        return FlagStore.get_flag(
            "use_new_backtesting_architecture",
            cls.get_flag_default("use_new_backtesting_architecture"),
        )

    @classmethod
    def use_new_backtester(cls) -> bool:
        """Check if new pure backtester should be used."""
        return FlagStore.get_flag("use_new_backtester", cls.get_flag_default("use_new_backtester"))

    @classmethod
    def use_optimization_orchestrator(cls) -> bool:
        """Check if optimization orchestrator should be used."""
        return FlagStore.get_flag(
            "use_optimization_orchestrator",
            cls.get_flag_default("use_optimization_orchestrator"),
        )

    # Parameter generator flags
    @classmethod
    def enable_optuna_generator(cls) -> bool:
        """Check if Optuna parameter generator is enabled."""
        return FlagStore.get_flag(
            "enable_optuna_generator", cls.get_flag_default("enable_optuna_generator")
        )

    @classmethod
    def enable_genetic_generator(cls) -> bool:
        """Check if genetic parameter generator is enabled."""
        return FlagStore.get_flag(
            "enable_genetic_generator", cls.get_flag_default("enable_genetic_generator")
        )

    # Compatibility flags
    @classmethod
    def enable_backward_compatibility(cls) -> bool:
        """Check if backward compatibility layer is enabled."""
        return FlagStore.get_flag(
            "enable_backward_compatibility",
            cls.get_flag_default("enable_backward_compatibility"),
        )

    @classmethod
    def enable_deprecation_warnings(cls) -> bool:
        """Check if deprecation warnings should be shown."""
        return FlagStore.get_flag(
            "enable_deprecation_warnings",
            cls.get_flag_default("enable_deprecation_warnings"),
        )

    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """
        Get current state of all feature flags.

        Returns:
            Dict[str, bool]: Dictionary of all feature flags and their current values
        """
        return {
            "use_new_optimization_architecture": cls.use_new_optimization_architecture(),
            "use_new_backtesting_architecture": cls.use_new_backtesting_architecture(),
            "use_new_backtester": cls.use_new_backtester(),
            "use_optimization_orchestrator": cls.use_optimization_orchestrator(),
            "enable_optuna_generator": cls.enable_optuna_generator(),
            "enable_genetic_generator": cls.enable_genetic_generator(),
            "enable_backward_compatibility": cls.enable_backward_compatibility(),
            "enable_deprecation_warnings": cls.enable_deprecation_warnings(),
        }
