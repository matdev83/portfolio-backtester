"""
Feature flag system package.

This package provides a modular feature flag system with separated concerns:
- FlagStore: Low-level flag storage and retrieval
- FlagRegistry: Definition of all flags and their defaults
- FlagContextManagers: Context managers for testing scenarios
- FeatureFlags: Main facade coordinating all components
"""

from .flag_store import FlagStore
from .flag_registry import FlagRegistry
from .flag_context_managers import FlagContextManagers
from .main import FeatureFlags, is_new_architecture_enabled, should_show_migration_warnings

__all__ = [
    "FlagStore",
    "FlagRegistry",
    "FlagContextManagers",
    "FeatureFlags",
    "is_new_architecture_enabled",
    "should_show_migration_warnings",
]
