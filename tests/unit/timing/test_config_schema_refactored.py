"""
Tests for refactored TimingConfigSchema with SOLID principles.
Ensures backward compatibility while testing new architecture.
"""

import pytest
import tempfile
import os
from portfolio_backtester.timing.config_schema_refactored import (
    TimingConfigSchema,
    validate_timing_config,
)
from portfolio_backtester.timing.validation import ValidationError


class TestTimingConfigSchemaRefactored:
    """Test refactored TimingConfigSchema class."""
    
    def test_valid_time_based_config(self):
        """Test validation of valid time-based configuration."""
        config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M',
                'rebalance_offset': 0
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 0
    
    def test_valid_signal_based_config(self):
        """Test validation of valid signal-based configuration."""
        config = {
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'D',
                'min_holding_period': 1,
                'max_holding_period': 5
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 0
    
    def test_valid_custom_config(self):
        """Test validation of valid custom configuration."""
        config = {
            'timing_config': {
                'mode': 'custom',
                'custom_controller_class': 'my.module.CustomController',
                'custom_controller_params': {'param1': 'value1'}
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 0
    
    def test_missing_timing_config(self):
        """Test error when timing_config section is missing."""
        config = {}
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == 'timing_config'
        assert 'Missing timing_config section' in errors[0].message
    
    def test_invalid_mode(self):
        """Test error for invalid timing mode."""
        config = {
            'timing_config': {
                'mode': 'invalid_mode'
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) > 0
        mode_errors = [e for e in errors if e.field == 'mode']
        assert len(mode_errors) == 1
        assert 'Invalid timing mode' in mode_errors[0].message
    
    def test_get_default_config_time_based(self):
        """Test getting default time-based configuration."""
        config = TimingConfigSchema.get_default_config('time_based')
        
        assert config['mode'] == 'time_based'
        assert config['rebalance_frequency'] == 'M'
        assert config['rebalance_offset'] == 0
        assert config['enable_logging'] is False
        assert config['log_level'] == 'INFO'
    
    def test_get_default_config_signal_based(self):
        """Test getting default signal-based configuration."""
        config = TimingConfigSchema.get_default_config('signal_based')
        
        assert config['mode'] == 'signal_based'
        assert config['scan_frequency'] == 'D'
        assert config['min_holding_period'] == 1
        assert config['max_holding_period'] is None
    
    def test_get_default_config_custom(self):
        """Test getting default custom configuration."""
        config = TimingConfigSchema.get_default_config('custom')
        
        assert config['mode'] == 'custom'
        assert 'custom_controller_class' in config
        assert 'custom_controller_params' in config
    
    def test_format_validation_report(self):
        """Test formatting validation reports."""
        errors = [
            ValidationError(
                field='mode',
                value='invalid',
                message='Invalid mode',
                suggestion='Use valid mode'
            )
        ]
        
        report = TimingConfigSchema.format_validation_report(errors)
        assert 'Configuration Validation Report' in report
        assert 'Invalid mode' in report
        assert 'Use valid mode' in report
    
    def test_yaml_file_validation(self):
        """Test YAML file validation."""
        # Create temporary YAML file
        yaml_content = """
timing_config:
  mode: time_based
  rebalance_frequency: M
  rebalance_offset: 0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name
        
        try:
            errors = TimingConfigSchema.validate_yaml_file(temp_file)
            assert len(errors) == 0
        finally:
            os.unlink(temp_file)
    
    def test_convenience_functions(self):
        """Test convenience functions maintain compatibility."""
        config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M'
            }
        }
        
        # Test without raising errors
        errors = validate_timing_config(config, raise_on_error=False)
        assert len(errors) == 0
        
        # Test with invalid config and error raising
        invalid_config = {
            'timing_config': {
                'mode': 'invalid_mode'
            }
        }
        
        with pytest.raises(ValueError):
            validate_timing_config(invalid_config, raise_on_error=True)