"""
Feature flag system for architecture component control.

This module provides feature flags to enable/disable architecture components
for testing, debugging, and gradual rollout of new features.

The FeatureFlags class has been refactored into a modular system with separated concerns:
- FlagStore: Low-level flag storage and retrieval
- FlagRegistry: Definition of all flags and their defaults
- FlagContextManagers: Context managers for testing scenarios
- FeatureFlags: Main facade coordinating all components

This file provides backward compatibility by re-exporting the new modular implementation.
"""

# Re-export everything from the new modular system for backward compatibility
from .feature_flags.flag_store import FlagStore
from .feature_flags.flag_registry import FlagRegistry
from .feature_flags.flag_context_managers import FlagContextManagers
from .feature_flags.main import (
    FeatureFlags,
    is_new_architecture_enabled,
    should_show_migration_warnings,
)

__all__ = [
    "FlagStore",
    "FlagRegistry",
    "FlagContextManagers",
    "FeatureFlags",
    "is_new_architecture_enabled",
    "should_show_migration_warnings",
]
