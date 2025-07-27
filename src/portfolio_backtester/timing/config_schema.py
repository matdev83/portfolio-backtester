"""
YAML configuration schema validation for timing framework.
Provides comprehensive validation with helpful error messages.
"""

import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error with context."""
    field: str
    value: Any
    message: str
    suggestion: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"


class TimingConfigSchema:
    """YAML configuration schema validator for timing framework."""
    
    # Define valid values for each configuration option
    VALID_MODES = ['time_based', 'signal_based', 'custom']
    VALID_TIME_FREQUENCIES = ['D', 'W', 'M', 'ME', 'Q', 'QE', 'A', 'Y', 'YE']
    VALID_SCAN_FREQUENCIES = ['D', 'W', 'M']
    
    # Configuration schema definition
    SCHEMA = {
        'timing_config': {
            'type': 'object',
            'required': ['mode'],
            'properties': {
                'mode': {
                    'type': 'string',
                    'enum': VALID_MODES,
                    'description': 'Timing mode: time_based for scheduled rebalancing, signal_based for market-driven timing, custom for user-defined timing'
                },
                'rebalance_frequency': {
                    'type': 'string',
                    'enum': VALID_TIME_FREQUENCIES,
                    'description': 'Rebalancing frequency for time_based mode (D=Daily, W=Weekly, M=Monthly, Q=Quarterly, A=Annual)'
                },
                'rebalance_offset': {
                    'type': 'integer',
                    'minimum': -30,
                    'maximum': 30,
                    'description': 'Days offset from standard rebalancing date (-30 to +30)'
                },
                'scan_frequency': {
                    'type': 'string',
                    'enum': VALID_SCAN_FREQUENCIES,
                    'description': 'Signal scanning frequency for signal_based mode (D=Daily, W=Weekly, M=Monthly)'
                },
                'min_holding_period': {
                    'type': 'integer',
                    'minimum': 1,
                    'description': 'Minimum days to hold a position before allowing exit'
                },
                'max_holding_period': {
                    'type': ['integer', 'null'],
                    'minimum': 1,
                    'description': 'Maximum days to hold a position (null for unlimited)'
                },
                'custom_controller_class': {
                    'type': 'string',
                    'description': 'Fully qualified class name for custom timing controller'
                },
                'custom_controller_params': {
                    'type': 'object',
                    'description': 'Parameters to pass to custom timing controller constructor'
                },
                'enable_logging': {
                    'type': 'boolean',
                    'default': False,
                    'description': 'Enable detailed logging of timing decisions'
                },
                'log_level': {
                    'type': 'string',
                    'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    'default': 'INFO',
                    'description': 'Logging level for timing operations'
                }
            }
        }
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate timing configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if timing_config exists
        if 'timing_config' not in config:
            errors.append(ValidationError(
                field='timing_config',
                value=None,
                message='Missing timing_config section',
                suggestion='Add timing_config section with at least mode specified'
            ))
            return errors
        
        timing_config = config['timing_config']
        
        # Validate mode (required)
        errors.extend(cls._validate_mode(timing_config))
        
        # Mode-specific validation
        mode = timing_config.get('mode', 'time_based')
        if mode == 'time_based':
            errors.extend(cls._validate_time_based_config(timing_config))
        elif mode == 'signal_based':
            errors.extend(cls._validate_signal_based_config(timing_config))
        elif mode == 'custom':
            errors.extend(cls._validate_custom_config(timing_config))
        
        # Validate common optional fields
        errors.extend(cls._validate_logging_config(timing_config))
        
        return errors
    
    @classmethod
    def _validate_mode(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate timing mode."""
        errors = []
        
        if 'mode' not in config:
            errors.append(ValidationError(
                field='mode',
                value=None,
                message='Missing required field: mode',
                suggestion=f'Add mode field with one of: {", ".join(cls.VALID_MODES)}'
            ))
        else:
            mode = config['mode']
            if mode not in cls.VALID_MODES:
                errors.append(ValidationError(
                    field='mode',
                    value=mode,
                    message=f'Invalid timing mode: {mode}',
                    suggestion=f'Use one of: {", ".join(cls.VALID_MODES)}'
                ))
        
        return errors
    
    @classmethod
    def _validate_time_based_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate time-based timing configuration."""
        errors = []
        
        # Validate rebalance_frequency
        frequency = config.get('rebalance_frequency', 'M')
        if frequency not in cls.VALID_TIME_FREQUENCIES:
            errors.append(ValidationError(
                field='rebalance_frequency',
                value=frequency,
                message=f'Invalid rebalance frequency: {frequency}',
                suggestion=f'Use one of: {", ".join(cls.VALID_TIME_FREQUENCIES)}'
            ))
        
        # Validate rebalance_offset
        offset = config.get('rebalance_offset', 0)
        if not isinstance(offset, int):
            errors.append(ValidationError(
                field='rebalance_offset',
                value=offset,
                message=f'rebalance_offset must be an integer, got {type(offset).__name__}',
                suggestion='Use integer value between -30 and 30'
            ))
        elif abs(offset) > 30:
            errors.append(ValidationError(
                field='rebalance_offset',
                value=offset,
                message=f'rebalance_offset must be between -30 and 30, got {offset}',
                suggestion='Use value between -30 and 30 days'
            ))
        
        # Warn about signal-based fields in time-based mode
        signal_fields = ['scan_frequency', 'min_holding_period', 'max_holding_period']
        for field in signal_fields:
            if field in config:
                errors.append(ValidationError(
                    field=field,
                    value=config[field],
                    message=f'Field {field} is not used in time_based mode',
                    suggestion=f'Remove {field} or change mode to signal_based',
                    severity='warning'
                ))
        
        return errors
    
    @classmethod
    def _validate_signal_based_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate signal-based timing configuration."""
        errors = []
        
        # Validate scan_frequency
        scan_freq = config.get('scan_frequency', 'D')
        if scan_freq not in cls.VALID_SCAN_FREQUENCIES:
            errors.append(ValidationError(
                field='scan_frequency',
                value=scan_freq,
                message=f'Invalid scan frequency: {scan_freq}',
                suggestion=f'Use one of: {", ".join(cls.VALID_SCAN_FREQUENCIES)}'
            ))
        
        # Validate holding periods
        min_holding = config.get('min_holding_period', 1)
        max_holding = config.get('max_holding_period')
        
        if not isinstance(min_holding, int) or min_holding < 1:
            errors.append(ValidationError(
                field='min_holding_period',
                value=min_holding,
                message=f'min_holding_period must be a positive integer, got {min_holding}',
                suggestion='Use positive integer (e.g., 1 for minimum 1 day holding)'
            ))
        
        if max_holding is not None:
            if not isinstance(max_holding, int) or max_holding < 1:
                errors.append(ValidationError(
                    field='max_holding_period',
                    value=max_holding,
                    message=f'max_holding_period must be a positive integer or null, got {max_holding}',
                    suggestion='Use positive integer or null for unlimited holding'
                ))
            elif isinstance(min_holding, int) and min_holding > max_holding:
                errors.append(ValidationError(
                    field='max_holding_period',
                    value=max_holding,
                    message=f'max_holding_period ({max_holding}) cannot exceed min_holding_period ({min_holding})',
                    suggestion=f'Set max_holding_period to at least {min_holding} or null'
                ))
        
        # Warn about time-based fields in signal-based mode
        time_fields = ['rebalance_frequency', 'rebalance_offset']
        for field in time_fields:
            if field in config:
                errors.append(ValidationError(
                    field=field,
                    value=config[field],
                    message=f'Field {field} is not used in signal_based mode',
                    suggestion=f'Remove {field} or change mode to time_based',
                    severity='warning'
                ))
        
        return errors
    
    @classmethod
    def _validate_custom_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate custom timing configuration."""
        errors = []
        
        # Require custom_controller_class
        if 'custom_controller_class' not in config:
            errors.append(ValidationError(
                field='custom_controller_class',
                value=None,
                message='custom_controller_class is required for custom mode',
                suggestion='Specify fully qualified class name (e.g., "mymodule.MyTimingController")'
            ))
        else:
            controller_class = config['custom_controller_class']
            # Allow short names for built-in controllers registered in CustomTimingRegistry
            from .custom_timing_registry import CustomTimingRegistry  # local import to avoid cycle

            if not isinstance(controller_class, str):
                errors.append(ValidationError(
                    field='custom_controller_class',
                    value=controller_class,
                    message='custom_controller_class must be a string',
                    suggestion='Provide controller class name or fully qualified path'
                ))
            else:
                has_dot = '.' in controller_class
                is_registered = CustomTimingRegistry.get(controller_class) is not None
                if not has_dot and not is_registered:
                    errors.append(ValidationError(
                        field='custom_controller_class',
                        value=controller_class,
                        message='custom_controller_class must be a fully qualified class name or a built-in registered controller',
                        suggestion='Use format "module.submodule.ClassName" or one of: ' + ", ".join(CustomTimingRegistry.list_registered().keys())
                    ))
        
        # Validate custom_controller_params if present
        if 'custom_controller_params' in config:
            params = config['custom_controller_params']
            if not isinstance(params, dict):
                errors.append(ValidationError(
                    field='custom_controller_params',
                    value=params,
                    message='custom_controller_params must be a dictionary',
                    suggestion='Use key-value pairs for controller parameters'
                ))
        
        return errors
    
    @classmethod
    def _validate_logging_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate logging configuration."""
        errors = []
        
        # Validate enable_logging
        if 'enable_logging' in config:
            enable_logging = config['enable_logging']
            if not isinstance(enable_logging, bool):
                errors.append(ValidationError(
                    field='enable_logging',
                    value=enable_logging,
                    message='enable_logging must be a boolean',
                    suggestion='Use true or false'
                ))
        
        # Validate log_level
        if 'log_level' in config:
            log_level = config['log_level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            if log_level not in valid_levels:
                errors.append(ValidationError(
                    field='log_level',
                    value=log_level,
                    message=f'Invalid log level: {log_level}',
                    suggestion=f'Use one of: {", ".join(valid_levels)}'
                ))
        
        return errors
    
    @classmethod
    def get_default_config(cls, mode: str = 'time_based') -> Dict[str, Any]:
        """
        Get default configuration for a timing mode.
        
        Args:
            mode: Timing mode ('time_based', 'signal_based', 'custom')
            
        Returns:
            Default configuration dictionary
        """
        base_config = {
            'enable_logging': False,
            'log_level': 'INFO'
        }
        
        if mode == 'time_based':
            return {
                'mode': 'time_based',
                'rebalance_frequency': 'M',
                'rebalance_offset': 0,
                **base_config
            }
        elif mode == 'signal_based':
            return {
                'mode': 'signal_based',
                'scan_frequency': 'D',
                'min_holding_period': 1,
                'max_holding_period': None,
                **base_config
            }
        elif mode == 'custom':
            return {
                'mode': 'custom',
                'custom_controller_class': 'your.module.CustomTimingController',
                'custom_controller_params': {},
                **base_config
            }
        else:
            raise ValueError(f"Unknown timing mode: {mode}")
    
    @classmethod
    def validate_yaml_file(cls, file_path: Union[str, Path]) -> List[ValidationError]:
        """
        Validate a YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            List of validation errors
        """
        errors = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            errors.append(ValidationError(
                field='file',
                value=str(file_path),
                message=f'Configuration file not found: {file_path}',
                suggestion='Check file path and ensure file exists'
            ))
            return errors
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(ValidationError(
                field='yaml',
                value=str(e),
                message=f'Invalid YAML syntax: {e}',
                suggestion='Check YAML syntax and fix formatting errors'
            ))
            return errors
        except Exception as e:
            errors.append(ValidationError(
                field='file',
                value=str(e),
                message=f'Error reading file: {e}',
                suggestion='Check file permissions and format'
            ))
            return errors
        
        if not isinstance(config, dict):
            errors.append(ValidationError(
                field='config',
                value=type(config).__name__,
                message='Configuration must be a dictionary/object',
                suggestion='Ensure YAML file contains key-value pairs'
            ))
            return errors
        
        # Validate the loaded configuration
        errors.extend(cls.validate_config(config))
        
        return errors
    
    @classmethod
    def format_validation_report(cls, errors: List[ValidationError]) -> str:
        """
        Format validation errors into a human-readable report.
        
        Args:
            errors: List of validation errors
            
        Returns:
            Formatted error report
        """
        if not errors:
            return "✓ Configuration is valid"
        
        report = ["Configuration Validation Report", "=" * 35, ""]
        
        # Group errors by severity
        error_count = sum(1 for e in errors if e.severity == 'error')
        warning_count = sum(1 for e in errors if e.severity == 'warning')
        
        report.append(f"Found {error_count} error(s) and {warning_count} warning(s)")
        report.append("")
        
        # Format each error
        for i, error in enumerate(errors, 1):
            severity_symbol = "✗" if error.severity == 'error' else "⚠" if error.severity == 'warning' else "ℹ"
            report.append(f"{i}. {severity_symbol} {error.field}: {error.message}")
            
            if error.value is not None:
                report.append(f"   Current value: {error.value}")
            
            if error.suggestion:
                report.append(f"   Suggestion: {error.suggestion}")
            
            report.append("")
        
        return "\n".join(report)


def validate_timing_config(config: Dict[str, Any], raise_on_error: bool = True) -> List[ValidationError]:
    """
    Convenience function to validate timing configuration.
    
    Args:
        config: Configuration dictionary
        raise_on_error: Whether to raise exception on validation errors
        
    Returns:
        List of validation errors
        
    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    errors = TimingConfigSchema.validate_config(config)
    
    if raise_on_error and any(e.severity == 'error' for e in errors):
        error_report = TimingConfigSchema.format_validation_report(errors)
        raise ValueError(f"Configuration validation failed:\n{error_report}")
    
    return errors


def validate_timing_config_file(file_path: Union[str, Path], raise_on_error: bool = True) -> List[ValidationError]:
    """
    Convenience function to validate timing configuration file.
    
    Args:
        file_path: Path to YAML configuration file
        raise_on_error: Whether to raise exception on validation errors
        
    Returns:
        List of validation errors
        
    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    errors = TimingConfigSchema.validate_yaml_file(file_path)
    
    if raise_on_error and any(e.severity == 'error' for e in errors):
        error_report = TimingConfigSchema.format_validation_report(errors)
        raise ValueError(f"Configuration file validation failed:\n{error_report}")
    
    return errors