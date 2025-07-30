"""
Feature flag system for gradual migration to new architecture.

This module provides feature flags to enable/disable architecture components
during the migration from the legacy monolithic architecture to the new
modular architecture with separated concerns.
"""

import os
import logging
from typing import ContextManager, Optional, Dict, Any
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Feature flag system for controlling architecture component availability.
    
    This class provides methods to check feature flags and context managers
    for testing component isolation during the architecture migration.
    """
    
    # Thread-local storage for context manager overrides
    _local = threading.local()
    
    @classmethod
    def _get_env_flag(cls, env_var: str, default: str = "false") -> bool:
        """Get boolean value from environment variable."""
        return os.environ.get(env_var, default).lower() in ("true", "1", "yes", "on")
    
    @classmethod
    def _get_override(cls, flag_name: str) -> Optional[bool]:
        """Get override value from thread-local storage."""
        if not hasattr(cls._local, 'overrides'):
            return None
        return cls._local.overrides.get(flag_name)
    
    @classmethod
    def _set_override(cls, flag_name: str, value: bool) -> None:
        """Set override value in thread-local storage."""
        if not hasattr(cls._local, 'overrides'):
            cls._local.overrides = {}
        cls._local.overrides[flag_name] = value
    
    @classmethod
    def _clear_override(cls, flag_name: str) -> None:
        """Clear override value from thread-local storage."""
        if hasattr(cls._local, 'overrides') and flag_name in cls._local.overrides:
            del cls._local.overrides[flag_name]
    
    @classmethod
    def use_new_optimization_architecture(cls) -> bool:
        """
        Check if new optimization architecture should be used.
        
        Returns:
            bool: True if new optimization architecture is enabled
        """
        override = cls._get_override('use_new_optimization_architecture')
        if override is not None:
            return override
        return cls._get_env_flag("USE_NEW_OPTIMIZATION_ARCHITECTURE")
    
    @classmethod
    def use_new_backtester(cls) -> bool:
        """
        Check if new pure backtester should be used.
        
        Returns:
            bool: True if new backtester is enabled
        """
        override = cls._get_override('use_new_backtester')
        if override is not None:
            return override
        return cls._get_env_flag("USE_NEW_BACKTESTER")
    
    @classmethod
    def use_optimization_orchestrator(cls) -> bool:
        """
        Check if optimization orchestrator should be used.
        
        Returns:
            bool: True if optimization orchestrator is enabled
        """
        override = cls._get_override('use_optimization_orchestrator')
        if override is not None:
            return override
        return cls._get_env_flag("USE_OPTIMIZATION_ORCHESTRATOR")
    
    @classmethod
    def enable_optuna_generator(cls) -> bool:
        """
        Check if Optuna parameter generator is enabled.
        
        Returns:
            bool: True if Optuna generator is enabled
        """
        override = cls._get_override('enable_optuna_generator')
        if override is not None:
            return override
        return cls._get_env_flag("ENABLE_OPTUNA_GENERATOR", "true")
    
    @classmethod
    def enable_genetic_generator(cls) -> bool:
        """
        Check if genetic parameter generator is enabled.
        
        Returns:
            bool: True if genetic generator is enabled
        """
        override = cls._get_override('enable_genetic_generator')
        if override is not None:
            return override
        return cls._get_env_flag("ENABLE_GENETIC_GENERATOR", "true")
    

    
    @classmethod
    def enable_backward_compatibility(cls) -> bool:
        """
        Check if backward compatibility layer is enabled.
        Returns:
            bool: True if legacy compatibility is enabled (default True)
        """
        override = cls._get_override('enable_backward_compatibility')
        if override is not None:
            return override
        return cls._get_env_flag("ENABLE_BACKWARD_COMPATIBILITY", "true")

    @classmethod
    def enable_deprecation_warnings(cls) -> bool:
        """
        Check if deprecation warnings should be shown.
        
        Returns:
            bool: True if deprecation warnings are enabled
        """
        override = cls._get_override('enable_deprecation_warnings')
        if override is not None:
            return override
        return cls._get_env_flag("ENABLE_DEPRECATION_WARNINGS", "true")
    
    @classmethod
    @contextmanager
    def disable_optuna(cls) -> ContextManager[None]:
        """
        Context manager to disable Optuna parameter generator.
        
        This is used for testing component isolation to prove that
        other components work without Optuna dependencies.
        
        Yields:
            None
        """
        flag_name = 'enable_optuna_generator'
        original_value = cls._get_override(flag_name)
        
        try:
            cls._set_override(flag_name, False)
            logger.debug("Optuna parameter generator disabled via context manager")
            yield
        finally:
            if original_value is not None:
                cls._set_override(flag_name, original_value)
            else:
                cls._clear_override(flag_name)
            logger.debug("Optuna parameter generator context manager restored")
    
    @classmethod
    @contextmanager
    def disable_genetic(cls) -> ContextManager[None]:
        """
        Context manager to disable genetic parameter generator.
        
        This is used for testing component isolation to prove that
        other components work without PyGAD dependencies.
        
        Yields:
            None
        """
        flag_name = 'enable_genetic_generator'
        original_value = cls._get_override(flag_name)
        
        try:
            cls._set_override(flag_name, False)
            logger.debug("Genetic parameter generator disabled via context manager")
            yield
        finally:
            if original_value is not None:
                cls._set_override(flag_name, original_value)
            else:
                cls._clear_override(flag_name)
            logger.debug("Genetic parameter generator context manager restored")

    @classmethod
    @contextmanager
    def disable_backward_compatibility(cls) -> ContextManager[None]:
        """
        Context manager to temporarily disable the legacy compatibility layer.
        """
        flag_name = 'enable_backward_compatibility'
        original_value = cls._get_override(flag_name)
        try:
            cls._set_override(flag_name, False)
            logger.debug("Backward compatibility disabled via context manager")
            yield
        finally:
            if original_value is not None:
                cls._set_override(flag_name, original_value)
            else:
                cls._clear_override(flag_name)
            logger.debug("Backward compatibility context manager restored")
    
    @classmethod
    @contextmanager
    def disable_all_optimizers(cls) -> ContextManager[None]:
        """
        Context manager to disable all optimization components.
        
        This is used for testing that the pure backtester works
        in complete isolation from optimization components.
        
        Yields:
            None
        """
        flags_to_disable = [
            'enable_optuna_generator',
            'enable_genetic_generator',
            'use_optimization_orchestrator',
            'use_new_optimization_architecture'
        ]
        
        original_values = {}
        
        try:
            for flag_name in flags_to_disable:
                original_values[flag_name] = cls._get_override(flag_name)
                cls._set_override(flag_name, False)
            
            logger.debug("All optimization components disabled via context manager")
            yield
            
        finally:
            for flag_name in flags_to_disable:
                if original_values[flag_name] is not None:
                    cls._set_override(flag_name, original_values[flag_name])
                else:
                    cls._clear_override(flag_name)
            
            logger.debug("All optimization components context manager restored")
    
    @classmethod
    @contextmanager
    def enable_new_architecture(cls) -> ContextManager[None]:
        """
        Context manager to enable new architecture components.
        
        This is used for testing the new architecture in isolation
        or for gradual rollout scenarios.
        
        Yields:
            None
        """
        flags_to_enable = [
            'use_new_optimization_architecture',
            'use_new_backtester',
            'use_optimization_orchestrator'
        ]
        
        original_values = {}
        
        try:
            for flag_name in flags_to_enable:
                original_values[flag_name] = cls._get_override(flag_name)
                cls._set_override(flag_name, True)
            
            logger.debug("New architecture components enabled via context manager")
            yield
            
        finally:
            for flag_name in flags_to_enable:
                if original_values[flag_name] is not None:
                    cls._set_override(flag_name, original_values[flag_name])
                else:
                    cls._clear_override(flag_name)
            
            logger.debug("New architecture components context manager restored")
    

    
    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """
        Get current state of all feature flags.
        
        Returns:
            Dict[str, bool]: Dictionary of all feature flags and their current values
        """
        return {
            'use_new_optimization_architecture': cls.use_new_optimization_architecture(),
            'use_new_backtester': cls.use_new_backtester(),
            'use_optimization_orchestrator': cls.use_optimization_orchestrator(),
            'enable_optuna_generator': cls.enable_optuna_generator(),
            'enable_genetic_generator': cls.enable_genetic_generator(),
            'enable_backward_compatibility': cls.enable_backward_compatibility(),
            'enable_deprecation_warnings': cls.enable_deprecation_warnings(),
        }
    
    @classmethod
    def log_current_flags(cls) -> None:
        """Log current state of all feature flags for debugging."""
        flags = cls.get_all_flags()
        logger.info("Current feature flag state:")
        for flag_name, flag_value in flags.items():
            logger.info(f"  {flag_name}: {flag_value}")


# Convenience functions for common flag checks
def is_new_architecture_enabled() -> bool:
    """Check if any new architecture components are enabled."""
    return (FeatureFlags.use_new_optimization_architecture() or 
            FeatureFlags.use_new_backtester() or 
            FeatureFlags.use_optimization_orchestrator())


def is_legacy_mode() -> bool:
    """Check if running in legacy mode (no new architecture components)."""
    return not is_new_architecture_enabled()


def should_show_migration_warnings() -> bool:
    """Check if migration-related warnings should be shown."""
    return (FeatureFlags.enable_deprecation_warnings() and
            FeatureFlags.enable_backward_compatibility())