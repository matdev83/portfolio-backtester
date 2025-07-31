"""
Backward Compatibility Layer for Timing System

This module provides utilities to migrate legacy strategy configurations
to the new timing framework while maintaining full backward compatibility.
"""

import logging
from typing import Dict, Any, List, Optional, Set
import warnings

logger = logging.getLogger(__name__)

# Legacy configuration parameter mapping
LEGACY_CONFIG_MAPPING = {
    # Direct parameter mappings
    'rebalance_frequency': 'timing_config.rebalance_frequency',
    
    # Method-based mappings (handled in migration logic)
    'supports_daily_signals': 'timing_config.mode',  # True -> signal_based, False -> time_based
    
    # Strategy-specific legacy patterns
    'daily_signals': 'timing_config.mode',  # Custom daily signal strategies
    'signal_based': 'timing_config.mode',   # Explicit signal-based configuration
}

# Known strategies that use daily signals (legacy pattern detection)
KNOWN_DAILY_SIGNAL_STRATEGIES = {
    'uvxy_rsi',
    'intraday_momentum',
    'daily_reversal',
    'volatility_breakout',
}

# Default timing configurations for different strategy types
DEFAULT_TIMING_CONFIGS = {
    'time_based': {
        'mode': 'time_based',
        'rebalance_frequency': 'M',
        'rebalance_offset': 0,
    },
    'signal_based': {
        'mode': 'signal_based',
        'scan_frequency': 'D',
        'min_holding_period': 1,
        'max_holding_period': None,
    }
}

class TimingConfigValidator:
    """Validates timing configuration parameters and provides helpful error messages."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate timing configuration and return list of error messages.
        
        Args:
            config: Timing configuration dictionary
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        mode = config.get('mode', 'time_based')
        if mode not in ['time_based', 'signal_based']:
            errors.append(
                f"Invalid timing mode '{mode}'. Must be 'time_based' or 'signal_based'. "
                f"Use 'time_based' for traditional monthly/quarterly rebalancing, "
                f"or 'signal_based' for custom entry/exit timing."
            )
            return errors  # Don't validate further if mode is invalid
        
        if mode == 'time_based':
            errors.extend(TimingConfigValidator._validate_time_based_config(config))
        elif mode == 'signal_based':
            errors.extend(TimingConfigValidator._validate_signal_based_config(config))
        
        return errors
    
    @staticmethod
    def _validate_time_based_config(config: Dict[str, Any]) -> List[str]:
        """Validate time-based timing configuration."""
        errors = []
        
        frequency = config.get('rebalance_frequency', 'M')
        valid_frequencies = ['D', 'W', 'M', 'ME', 'Q', 'A', 'Y']
        if frequency not in valid_frequencies:
            errors.append(
                f"Invalid rebalance_frequency '{frequency}'. Must be one of {valid_frequencies}. "
                f"Common values: 'M' (monthly), 'Q' (quarterly), 'D' (daily), 'W' (weekly)."
            )
        
        offset = config.get('rebalance_offset', 0)
        if not isinstance(offset, int) or abs(offset) > 30:
            errors.append(
                f"rebalance_offset must be an integer between -30 and 30 days, got {offset}. "
                f"Use positive values to rebalance after period end, negative for before."
            )
        
        return errors
    
    @staticmethod
    def _validate_signal_based_config(config: Dict[str, Any]) -> List[str]:
        """Validate signal-based timing configuration."""
        errors = []
        
        scan_freq = config.get('scan_frequency', 'D')
        valid_scan_frequencies = ['D', 'W', 'M']
        if scan_freq not in valid_scan_frequencies:
            errors.append(
                f"Invalid scan_frequency '{scan_freq}'. Must be one of {valid_scan_frequencies}. "
                f"Use 'D' for daily signal scanning (most common), 'W' for weekly, 'M' for monthly."
            )
        
        max_holding = config.get('max_holding_period')
        if max_holding is not None:
            if not isinstance(max_holding, int) or max_holding < 1:
                errors.append(
                    f"max_holding_period must be a positive integer (days), got {max_holding}. "
                    f"This forces position exit after specified days. Use None for no limit."
                )
        
        min_holding = config.get('min_holding_period', 1)
        if not isinstance(min_holding, int) or min_holding < 1:
            errors.append(
                f"min_holding_period must be a positive integer (days), got {min_holding}. "
                f"This prevents rapid position changes. Minimum value is 1."
            )
        
        if max_holding is not None and min_holding > max_holding:
            errors.append(
                f"min_holding_period ({min_holding}) cannot exceed max_holding_period ({max_holding}). "
                f"Adjust these values so min_holding_period <= max_holding_period."
            )
        
        return errors

def migrate_legacy_config_with_strategy(strategy_config: Dict[str, Any], strategy_instance=None) -> Dict[str, Any]:
    """
    Migrate legacy strategy configuration to new timing format with strategy instance.
    
    This function ensures backward compatibility by detecting legacy timing
    patterns and converting them to the new timing_config format.
    
    Args:
        strategy_config: Original strategy configuration dictionary
        strategy_instance: Strategy instance for method override detection
        
    Returns:
        Updated strategy configuration with timing_config
    """
    # Make a copy to avoid modifying the original
    config = strategy_config.copy()
    
    # If timing_config already exists, validate it and return
    if 'timing_config' in config:
        timing_config = config['timing_config']
        validation_errors = TimingConfigValidator.validate_config(timing_config)
        if validation_errors:
            error_msg = "Invalid timing_config:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
        return config
    
    # Detect timing mode from legacy patterns with strategy instance
    timing_mode = _detect_timing_mode(config, strategy_instance)
    
    # Create timing configuration based on detected mode
    if timing_mode == 'signal_based':
        timing_config = _create_signal_based_timing_config(config, strategy_instance)
    else:
        timing_config = _create_time_based_timing_config(config)
    
    # Add timing_config to the strategy configuration
    config['timing_config'] = timing_config

    # Remove legacy timing-related keys to avoid duplication
    legacy_keys = [
        'rebalance_frequency', 'rebalance_offset',
        'scan_frequency', 'min_holding_period', 'max_holding_period',
        'daily_signals', 'signal_based'
    ]
    for k in legacy_keys:
        if k in config:
            del config[k]
    
    # Log migration for debugging
    logger.debug(f"Migrated legacy config to timing mode: {timing_mode}")
    
    return config

def migrate_legacy_config(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy strategy configuration to new timing format.
    
    This function ensures backward compatibility by detecting legacy timing
    patterns and converting them to the new timing_config format.
    
    Args:
        strategy_config: Original strategy configuration dictionary
        
    Returns:
        Updated strategy configuration with timing_config
    """
    # Make a copy to avoid modifying the original
    config = strategy_config.copy()
    
    # If timing_config already exists, validate it and return
    if 'timing_config' in config:
        timing_config = config['timing_config']
        validation_errors = TimingConfigValidator.validate_config(timing_config)
        if validation_errors:
            error_msg = "Invalid timing_config:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
        return config
    
    # Detect timing mode from legacy patterns
    timing_mode = _detect_timing_mode(config)
    
    # Create timing configuration based on detected mode
    if timing_mode == 'signal_based':
        timing_config = _create_signal_based_timing_config(config, None)
    else:
        timing_config = _create_time_based_timing_config(config)
    
    # Add timing_config to the strategy configuration
    config['timing_config'] = timing_config

    # Remove legacy timing-related keys to avoid duplication (same list as above)
    legacy_keys = [
        'rebalance_frequency', 'rebalance_offset',
        'scan_frequency', 'min_holding_period', 'max_holding_period',
        'daily_signals', 'signal_based'
    ]
    for k in legacy_keys:
        if k in config:
            del config[k]
    
    # Log migration for debugging
    logger.debug(f"Migrated legacy config to timing mode: {timing_mode}")
    
    return config

def _detect_timing_mode(config: Dict[str, Any], strategy_instance=None) -> str:
    """
    Detect whether a strategy should use time-based or signal-based timing
    based on legacy configuration patterns.
    """
    strategy_name = config.get('strategy', '').lower()
    
    # Check for explicit signal-based indicators
    if config.get('daily_signals', False):
        return 'signal_based'
    
    if config.get('signal_based', False):
        return 'signal_based'
    
    # Check for known daily signal strategies
    if strategy_name in KNOWN_DAILY_SIGNAL_STRATEGIES:
        return 'signal_based'
    
    # Check for daily rebalance frequency (often indicates signal-based)
    rebalance_freq = config.get('rebalance_frequency', 'M')
    if rebalance_freq == 'D':
        return 'signal_based'
    
    # Check strategy class name patterns (if available)
    strategy_class = config.get('strategy_class')
    if strategy_class:
        class_name = strategy_class.__name__.lower() if hasattr(strategy_class, '__name__') else str(strategy_class).lower()
        if any(pattern in class_name for pattern in ['daily', 'signal', 'intraday', 'rsi', 'breakout']):
            return 'signal_based'
    
    # Check strategy instance class name patterns (if available)
    if strategy_instance is not None:
        class_name = strategy_instance.__class__.__name__.lower()
        # Add guard to prevent infinite loop
        try:
            if any(pattern in class_name for pattern in ['daily', 'signal', 'intraday', 'rsi', 'breakout', 'uvxy']):
                return 'signal_based'
        except Exception:
            pass  # Guard against infinite loop
    
    # Check if strategy instance has overridden supports_daily_signals method
    if strategy_instance is not None:
        # Import here to avoid circular imports
        try:
            from ..strategies.base.base_strategy import BaseStrategy
            
            # Check if the method is overridden in the subclass
            base_method = BaseStrategy.__dict__.get('supports_daily_signals')
            current_method = strategy_instance.__class__.__dict__.get('supports_daily_signals')
            
            if current_method is not None and current_method != base_method:
                # Method is overridden in subclass, assume it's a legacy daily strategy
                return 'signal_based'
        except ImportError:
            pass
    
    # Default to time-based for backward compatibility
    return 'time_based'

def _create_time_based_timing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create time-based timing configuration from legacy config."""
    timing_config = DEFAULT_TIMING_CONFIGS['time_based'].copy()
    
    # Map legacy rebalance_frequency (handle None values)
    if 'rebalance_frequency' in config and config['rebalance_frequency'] is not None:
        timing_config['rebalance_frequency'] = config['rebalance_frequency']
    
    # Map legacy rebalance_offset if present (handle None values)
    if 'rebalance_offset' in config and config['rebalance_offset'] is not None:
        timing_config['rebalance_offset'] = config['rebalance_offset']
    
    return timing_config

def _create_signal_based_timing_config(config: Dict[str, Any], strategy_instance=None) -> Dict[str, Any]:
    """Create signal-based timing configuration from legacy config."""
    timing_config = DEFAULT_TIMING_CONFIGS['signal_based'].copy()
    
    # Map legacy scan frequency (handle None values)
    if 'scan_frequency' in config and config['scan_frequency'] is not None:
        timing_config['scan_frequency'] = config['scan_frequency']
    elif 'rebalance_frequency' in config and config['rebalance_frequency'] == 'D':
        timing_config['scan_frequency'] = 'D'
    
    # Map legacy holding period constraints (handle None values)
    if 'min_holding_period' in config and config['min_holding_period'] is not None:
        timing_config['min_holding_period'] = config['min_holding_period']
    
    if 'max_holding_period' in config and config['max_holding_period'] is not None:
        timing_config['max_holding_period'] = config['max_holding_period']
    
    # Strategy-specific defaults
    strategy_name = config.get('strategy', '').lower()
    is_uvxy_strategy = strategy_name == 'uvxy_rsi'
    
    # Also check strategy instance class name for UVXY detection
    if not is_uvxy_strategy and strategy_instance is not None:
        class_name = strategy_instance.__class__.__name__.lower()
        is_uvxy_strategy = 'uvxy' in class_name
    
    if is_uvxy_strategy:
        # UVXY strategy typically has 1-day holding period
        timing_config['min_holding_period'] = 1
        timing_config['max_holding_period'] = 1
    
    return timing_config

def validate_legacy_behavior(old_config: Dict[str, Any], new_config: Dict[str, Any], strategy_instance=None) -> bool:
    """
    Validate that migrated configuration will produce equivalent behavior
    to the legacy configuration.
    
    Args:
        old_config: Original legacy configuration
        new_config: Migrated configuration with timing_config
        strategy_instance: Strategy instance for method override detection
        
    Returns:
        True if behavior should be equivalent, False otherwise
    """
    # If old config already had timing_config, it's already in new format
    if 'timing_config' in old_config:
        return True
    
    # Check that timing mode detection is consistent
    expected_mode = _detect_timing_mode(old_config, strategy_instance)
    actual_mode = new_config.get('timing_config', {}).get('mode')
    
    if expected_mode != actual_mode:
        logger.warning(f"Timing mode mismatch: expected {expected_mode}, got {actual_mode}")
        return False
    
    # Check frequency mapping for time-based strategies
    if expected_mode == 'time_based':
        old_freq = old_config.get('rebalance_frequency')
        new_freq = new_config.get('timing_config', {}).get('rebalance_frequency', 'M')
        
        # Handle None values - if old frequency is None, default should be used
        if old_freq is None:
            old_freq = 'M'  # Default frequency
        
        if old_freq != new_freq:
            logger.warning(f"Rebalance frequency mismatch: expected {old_freq}, got {new_freq}")
            return False
    
    return True

def get_migration_warnings(config: Dict[str, Any]) -> List[str]:
    """
    Get list of warnings about potential migration issues.
    
    Args:
        config: Strategy configuration
        
    Returns:
        List of warning messages
    """
    warnings_list = []
    
    # Check for deprecated parameters
    deprecated_params = {
        'daily_signals': 'Use timing_config.mode = "signal_based" instead',
        'signal_based': 'Use timing_config.mode = "signal_based" instead',
        'rebalance_offset': 'Move to timing_config.rebalance_offset',
    }
    
    for param, suggestion in deprecated_params.items():
        if param in config:
            warnings_list.append(f"Parameter '{param}' is deprecated. {suggestion}")
    
    # Check for potential timing mode conflicts
    if config.get('rebalance_frequency') == 'D' and not config.get('timing_config', {}).get('mode') == 'signal_based':
        warnings_list.append(
            "Daily rebalance_frequency detected but timing mode is not signal_based. "
            "Consider using timing_config.mode = 'signal_based' for daily strategies."
        )
    
    return warnings_list

def ensure_backward_compatibility_with_strategy(strategy_config: Dict[str, Any], strategy_instance=None) -> Dict[str, Any]:
    """
    Main entry point for ensuring backward compatibility with strategy instance.
    
    This function:
    1. Migrates legacy configuration to new format
    2. Validates the migration
    3. Issues warnings for deprecated patterns
    4. Ensures all existing configurations continue to work
    5. Uses strategy instance for method override detection
    
    Args:
        strategy_config: Strategy configuration dictionary
        strategy_instance: Strategy instance for method override detection
        
    Returns:
        Updated configuration with timing_config
        
    Raises:
        ValueError: If configuration is invalid or migration fails
    """
    try:
        # Get migration warnings first (before modification)
        migration_warnings = get_migration_warnings(strategy_config)
        for warning in migration_warnings:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
        
        # Migrate configuration with strategy instance
        migrated_config = migrate_legacy_config_with_strategy(strategy_config, strategy_instance)
        
        # Validate that migration preserves behavior
        if not validate_legacy_behavior(strategy_config, migrated_config, strategy_instance):
            raise ValueError("Migration would change strategy behavior")
        
        # Final validation of timing configuration
        timing_config = migrated_config.get('timing_config', {})
        validation_errors = TimingConfigValidator.validate_config(timing_config)
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
        
        return migrated_config
        
    except Exception as e:
        logger.error(f"Failed to migrate legacy configuration: {e}")
        # Provide helpful error message with fallback suggestion
        raise ValueError(
            f"Configuration migration failed: {e}\n"
            f"To fix this, add a 'timing_config' section to your strategy configuration:\n"
            f"  timing_config:\n"
            f"    mode: 'time_based'  # or 'signal_based'\n"
            f"    rebalance_frequency: 'M'  # for time_based mode\n"
            f"    # scan_frequency: 'D'  # for signal_based mode"
        ) from e

def ensure_backward_compatibility(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for ensuring backward compatibility.
    
    This function:
    1. Migrates legacy configuration to new format
    2. Validates the migration
    3. Issues warnings for deprecated patterns
    4. Ensures all existing configurations continue to work
    
    Args:
        strategy_config: Strategy configuration dictionary
        
    Returns:
        Updated configuration with timing_config
        
    Raises:
        ValueError: If configuration is invalid or migration fails
    """
    try:
        # Get migration warnings first (before modification)
        migration_warnings = get_migration_warnings(strategy_config)
        for warning in migration_warnings:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
        
        # Migrate configuration
        migrated_config = migrate_legacy_config(strategy_config)
        
        # Validate that migration preserves behavior
        if not validate_legacy_behavior(strategy_config, migrated_config):
            raise ValueError("Migration would change strategy behavior")
        
        # Final validation of timing configuration
        timing_config = migrated_config.get('timing_config', {})
        validation_errors = TimingConfigValidator.validate_config(timing_config)
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
        
        return migrated_config
        
    except Exception as e:
        logger.error(f"Failed to migrate legacy configuration: {e}")
        # Provide helpful error message with fallback suggestion
        raise ValueError(
            f"Configuration migration failed: {e}\n"
            f"To fix this, add a 'timing_config' section to your strategy configuration:\n"
            f"  timing_config:\n"
            f"    mode: 'time_based'  # or 'signal_based'\n"
            f"    rebalance_frequency: 'M'  # for time_based mode\n"
            f"    # scan_frequency: 'D'  # for signal_based mode"
        ) from e

# Utility functions for testing and validation

def get_legacy_config_examples() -> Dict[str, Dict[str, Any]]:
    """
    Get examples of legacy configurations for testing.
    
    Returns:
        Dictionary of example configurations
    """
    return {
        'monthly_momentum': {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {'lookback_months': 6}
        },
        'quarterly_value': {
            'strategy': 'value',
            'rebalance_frequency': 'Q',
            'rebalance_offset': 5,
            'strategy_params': {'book_to_market': True}
        },
        'daily_uvxy': {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'strategy_params': {'rsi_period': 2, 'rsi_threshold': 30}
        },
        'signal_based_breakout': {
            'strategy': 'volatility_breakout',
            'daily_signals': True,
            'scan_frequency': 'D',
            'min_holding_period': 2,
            'max_holding_period': 10
        }
    }

def check_migration_compatibility() -> bool:
    """
    Test that all example legacy configurations migrate correctly.
    
    Returns:
        True if all migrations are successful, False otherwise
    """
    examples = get_legacy_config_examples()
    
    for name, config in examples.items():
        try:
            migrated = migrate_legacy_config(config)
            if not validate_legacy_behavior(config, migrated):
                logger.error(f"Migration test failed for {name}: behavior mismatch")
                return False
            logger.info(f"Migration test passed for {name}")
        except Exception as e:
            logger.error(f"Migration test failed for {name}: {e}")
            return False
    
    return True