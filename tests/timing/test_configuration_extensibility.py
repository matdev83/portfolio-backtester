"""
Tests for configuration and extensibility features (Task 10).
Tests YAML schema validation, custom timing controllers, and enhanced logging.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock, patch
from src.portfolio_backtester.timing.config_schema import (
    TimingConfigSchema, ValidationError, validate_timing_config, validate_timing_config_file
)
from src.portfolio_backtester.timing.custom_timing_registry import (
    CustomTimingRegistry, TimingControllerFactory, register_timing_controller
)
from src.portfolio_backtester.timing.timing_logger import (
    TimingLogger, get_timing_logger, configure_timing_logging
)
from src.portfolio_backtester.timing.timing_controller import TimingController


class TestYAMLConfigurationSchema:
    """Test YAML configuration schema validation."""
    
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
        """Test validation when timing_config is missing."""
        config = {'strategy_params': {}}
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == 'timing_config'
        assert 'Missing timing_config section' in errors[0].message
    
    def test_invalid_mode(self):
        """Test validation of invalid timing mode."""
        config = {
            'timing_config': {
                'mode': 'invalid_mode'
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == 'mode'
        assert 'Invalid timing mode' in errors[0].message
        assert 'time_based, signal_based, custom' in errors[0].suggestion
    
    def test_invalid_rebalance_frequency(self):
        """Test validation of invalid rebalance frequency."""
        config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'X'
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == 'rebalance_frequency'
        assert 'Invalid rebalance frequency' in errors[0].message
    
    def test_invalid_holding_periods(self):
        """Test validation of invalid holding periods."""
        config = {
            'timing_config': {
                'mode': 'signal_based',
                'min_holding_period': 5,
                'max_holding_period': 2
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert 'cannot exceed' in errors[0].message
    
    def test_warning_for_wrong_mode_fields(self):
        """Test warnings for fields used in wrong mode."""
        config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M',
                'scan_frequency': 'D'  # Wrong mode field
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        warnings = [e for e in errors if e.severity == 'warning']
        assert len(warnings) == 1
        assert warnings[0].field == 'scan_frequency'
        assert 'not used in time_based mode' in warnings[0].message
    
    def test_missing_custom_controller_class(self):
        """Test validation when custom controller class is missing."""
        config = {
            'timing_config': {
                'mode': 'custom'
            }
        }
        
        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == 'custom_controller_class'
        assert 'required for custom mode' in errors[0].message
    
    def test_default_config_generation(self):
        """Test default configuration generation."""
        time_based_default = TimingConfigSchema.get_default_config('time_based')
        assert time_based_default['mode'] == 'time_based'
        assert time_based_default['rebalance_frequency'] == 'M'
        
        signal_based_default = TimingConfigSchema.get_default_config('signal_based')
        assert signal_based_default['mode'] == 'signal_based'
        assert signal_based_default['scan_frequency'] == 'D'
        
        custom_default = TimingConfigSchema.get_default_config('custom')
        assert custom_default['mode'] == 'custom'
        assert 'custom_controller_class' in custom_default
    
    def test_yaml_file_validation(self):
        """Test YAML file validation."""
        # Create temporary YAML file
        config_data = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            errors = TimingConfigSchema.validate_yaml_file(temp_file)
            assert len(errors) == 0
        finally:
            os.unlink(temp_file)
    
    def test_yaml_file_not_found(self):
        """Test validation of non-existent YAML file."""
        errors = TimingConfigSchema.validate_yaml_file('nonexistent.yaml')
        assert len(errors) == 1
        assert errors[0].field == 'file'
        assert 'not found' in errors[0].message
    
    def test_invalid_yaml_syntax(self):
        """Test validation of invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: syntax: [')
            temp_file = f.name
        
        try:
            errors = TimingConfigSchema.validate_yaml_file(temp_file)
            assert len(errors) == 1
            assert errors[0].field == 'yaml'
            assert 'Invalid YAML syntax' in errors[0].message
        finally:
            os.unlink(temp_file)
    
    def test_validation_report_formatting(self):
        """Test validation report formatting."""
        errors = [
            ValidationError('field1', 'value1', 'Error message 1', 'Suggestion 1', 'error'),
            ValidationError('field2', 'value2', 'Warning message 2', 'Suggestion 2', 'warning')
        ]
        
        report = TimingConfigSchema.format_validation_report(errors)
        
        assert 'Configuration Validation Report' in report
        assert 'Found 1 error(s) and 1 warning(s)' in report
        assert 'Error message 1' in report
        assert 'Warning message 2' in report
        assert 'Suggestion 1' in report
    
    def test_convenience_validation_functions(self):
        """Test convenience validation functions."""
        valid_config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M'
            }
        }
        
        # Should not raise exception
        errors = validate_timing_config(valid_config, raise_on_error=False)
        assert len(errors) == 0
        
        # Should not raise exception
        validate_timing_config(valid_config, raise_on_error=True)
        
        # Should raise exception for invalid config
        invalid_config = {
            'timing_config': {
                'mode': 'invalid'
            }
        }
        
        with pytest.raises(ValueError, match='Configuration validation failed'):
            validate_timing_config(invalid_config, raise_on_error=True)


class TestCustomTimingRegistry:
    """Test custom timing controller registry and factory."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear registry before each test
        CustomTimingRegistry.clear()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clear registry after each test
        CustomTimingRegistry.clear()
    
    def test_register_timing_controller(self):
        """Test registering a custom timing controller."""
        class TestController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True
            
            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []
        
        CustomTimingRegistry.register('test_controller', TestController)
        
        retrieved = CustomTimingRegistry.get('test_controller')
        assert retrieved == TestController
        
        registered = CustomTimingRegistry.list_registered()
        assert 'test_controller' in registered
    
    def test_register_with_aliases(self):
        """Test registering controller with aliases."""
        class TestController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True
            
            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []
        
        CustomTimingRegistry.register('test_controller', TestController, aliases=['tc', 'test'])
        
        # Test direct access
        assert CustomTimingRegistry.get('test_controller') == TestController
        
        # Test alias access
        assert CustomTimingRegistry.get('tc') == TestController
        assert CustomTimingRegistry.get('test') == TestController
    
    def test_register_invalid_controller(self):
        """Test registering invalid controller class."""
        class InvalidController:
            pass
        
        with pytest.raises(ValueError, match='must inherit from TimingController'):
            CustomTimingRegistry.register('invalid', InvalidController)
    
    def test_unregister_controller(self):
        """Test unregistering a controller."""
        class TestController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True
            
            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []
        
        CustomTimingRegistry.register('test_controller', TestController, aliases=['tc'])
        
        # Verify registration
        assert CustomTimingRegistry.get('test_controller') == TestController
        assert CustomTimingRegistry.get('tc') == TestController
        
        # Unregister
        result = CustomTimingRegistry.unregister('test_controller')
        assert result == True
        
        # Verify removal
        assert CustomTimingRegistry.get('test_controller') is None
        assert CustomTimingRegistry.get('tc') is None
    
    def test_decorator_registration(self):
        """Test decorator-based registration."""
        @register_timing_controller('decorated_controller', aliases=['dc'])
        class DecoratedController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True
            
            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []
        
        # Test registration worked
        assert CustomTimingRegistry.get('decorated_controller') == DecoratedController
        assert CustomTimingRegistry.get('dc') == DecoratedController
    
    def test_timing_controller_factory(self):
        """Test timing controller factory."""
        # Test time-based creation
        time_config = {'mode': 'time_based', 'rebalance_frequency': 'M'}
        controller = TimingControllerFactory.create_controller(time_config)
        assert controller.__class__.__name__ == 'TimeBasedTiming'
        
        # Test signal-based creation
        signal_config = {'mode': 'signal_based', 'scan_frequency': 'D'}
        controller = TimingControllerFactory.create_controller(signal_config)
        assert controller.__class__.__name__ == 'SignalBasedTiming'
    
    def test_custom_controller_creation(self):
        """Test custom controller creation through factory."""
        class CustomController(TimingController):
            def __init__(self, config, custom_param=None):
                super().__init__(config)
                self.custom_param = custom_param
            
            def should_generate_signal(self, current_date, strategy):
                return True
            
            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []
        
        CustomTimingRegistry.register('custom_test', CustomController)
        
        config = {
            'mode': 'custom',
            'custom_controller_class': 'custom_test',
            'custom_controller_params': {'custom_param': 'test_value'}
        }
        
        controller = TimingControllerFactory.create_controller(config)
        assert isinstance(controller, CustomController)
        assert controller.custom_param == 'test_value'
    
    def test_factory_invalid_mode(self):
        """Test factory with invalid mode."""
        config = {'mode': 'invalid_mode'}
        
        with pytest.raises(ValueError, match='Unknown timing mode'):
            TimingControllerFactory.create_controller(config)
    
    def test_factory_missing_custom_class(self):
        """Test factory with missing custom controller class."""
        config = {'mode': 'custom'}
        
        with pytest.raises(ValueError, match='custom_controller_class is required'):
            TimingControllerFactory.create_controller(config)
    
    def test_factory_nonexistent_custom_class(self):
        """Test factory with non-existent custom controller class."""
        config = {
            'mode': 'custom',
            'custom_controller_class': 'nonexistent.Controller'
        }
        
        with pytest.raises(ValueError, match='Cannot find timing controller class'):
            TimingControllerFactory.create_controller(config)


class TestBuiltinCustomControllers:
    """Test built-in custom timing controllers."""
    
    def test_adaptive_timing_controller(self):
        """Test adaptive timing controller."""
        config = {
            'mode': 'custom',
            'custom_controller_class': 'adaptive_timing',
            'custom_controller_params': {
                'volatility_threshold': 0.03,
                'base_frequency': 'M'
            }
        }
        
        controller = TimingControllerFactory.create_controller(config)
        assert controller.__class__.__name__ == 'AdaptiveTimingController'
        assert controller.volatility_threshold == 0.03
        assert controller.base_frequency == 'M'
    
    def test_momentum_timing_controller(self):
        """Test momentum timing controller."""
        config = {
            'mode': 'custom',
            'custom_controller_class': 'momentum_timing',
            'custom_controller_params': {
                'momentum_period': 30
            }
        }
        
        controller = TimingControllerFactory.create_controller(config)
        assert controller.__class__.__name__ == 'MomentumTimingController'
        assert controller.momentum_period == 30
    
    def test_adaptive_controller_signal_generation(self):
        """Test adaptive controller signal generation."""
        config = {
            'base_frequency': 'M',
            'high_vol_frequency': 'W',
            'low_vol_frequency': 'Q'
        }
        
        from src.portfolio_backtester.timing.custom_timing_registry import AdaptiveTimingController
        controller = AdaptiveTimingController(config, volatility_threshold=0.02)
        
        # Test with mock strategy
        mock_strategy = Mock()
        test_date = pd.Timestamp('2023-01-02')  # Monday
        
        # Test without volatility method (should use base frequency)
        result = controller.should_generate_signal(test_date, mock_strategy)
        assert isinstance(result, bool)
    
    def test_momentum_controller_signal_generation(self):
        """Test momentum controller signal generation."""
        config = {}
        
        from src.portfolio_backtester.timing.custom_timing_registry import MomentumTimingController
        controller = MomentumTimingController(config, momentum_period=20)
        
        # Test signal generation
        mock_strategy = Mock()
        test_date = pd.Timestamp('2023-01-02')  # Monday
        
        result = controller.should_generate_signal(test_date, mock_strategy)
        assert result == True  # Should be True for Monday


class TestTimingLogger:
    """Test enhanced timing logger."""
    
    def setup_method(self):
        """Set up test environment."""
        self.logger = TimingLogger('test', enable_detailed_logging=True, log_level='DEBUG')
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        assert self.logger.enable_detailed_logging == True
        assert len(self.logger.log_entries) == 0
    
    def test_signal_generation_logging(self):
        """Test signal generation logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_signal_generation(
            strategy_name, current_date, True, 'RSI below threshold',
            rsi_value=25.0
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'signal_generation'
        assert entry.strategy_name == strategy_name
        assert entry.current_date == current_date
        assert 'YES' in entry.message
        assert entry.data['should_generate'] == True
        assert entry.data['rsi_value'] == 25.0
    
    def test_position_update_logging(self):
        """Test position update logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_position_update(
            strategy_name, current_date, 'AAPL', 'entry', 0.25, 150.0
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'position_update'
        assert 'AAPL' in entry.message
        assert 'entry' in entry.message
        assert entry.data['asset'] == 'AAPL'
        assert entry.data['weight'] == 0.25
        assert entry.data['price'] == 150.0
    
    def test_rebalance_event_logging(self):
        """Test rebalance event logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_rebalance_event(
            strategy_name, current_date, 5, 1.0
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'rebalance'
        assert '5 positions' in entry.message
        assert entry.data['num_positions'] == 5
        assert entry.data['total_weight'] == 1.0
    
    def test_error_logging(self):
        """Test error logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_error(
            strategy_name, current_date, 'ValidationError', 'Invalid configuration'
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'error'
        assert entry.level == 'ERROR'
        assert 'ValidationError' in entry.message
    
    def test_log_filtering(self):
        """Test log entry filtering."""
        # Add multiple entries
        dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
        strategies = ['Strategy1', 'Strategy2']
        
        for i, (date, strategy) in enumerate(zip(dates, strategies)):
            self.logger.log_signal_generation(strategy, date, True, f'Reason {i}')
            self.logger.log_position_update(strategy, date, 'AAPL', 'entry', 0.1, 100.0)
        
        # Test strategy filtering
        strategy1_entries = self.logger.get_log_entries(strategy_name='Strategy1')
        assert len(strategy1_entries) == 2
        assert all(e.strategy_name == 'Strategy1' for e in strategy1_entries)
        
        # Test event type filtering
        signal_entries = self.logger.get_log_entries(event_type='signal_generation')
        assert len(signal_entries) == 2
        assert all(e.event_type == 'signal_generation' for e in signal_entries)
        
        # Test date filtering
        date1_entries = self.logger.get_log_entries(start_date=dates[0], end_date=dates[0])
        assert len(date1_entries) == 2
        assert all(e.current_date == dates[0] for e in date1_entries)
        
        # Test last N filtering
        last_2_entries = self.logger.get_log_entries(last_n=2)
        assert len(last_2_entries) == 2
    
    def test_log_summary(self):
        """Test log summary generation."""
        # Add some entries
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Test')
        self.logger.log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.1, 100.0)
        self.logger.log_error(strategy_name, current_date, 'TestError', 'Test error')
        
        summary = self.logger.get_log_summary()
        
        assert summary['total_entries'] == 3
        assert summary['event_counts']['signal_generation'] == 1
        assert summary['event_counts']['position_update'] == 1
        assert summary['event_counts']['error'] == 1
        assert summary['level_counts']['INFO'] == 2
        assert summary['level_counts']['ERROR'] == 1
        assert 'TestStrategy' in summary['strategies']
    
    def test_log_export(self):
        """Test log export functionality."""
        # Add some entries
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Test')
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            self.logger.export_logs(json_file, format='json')
            assert os.path.exists(json_file)
            
            # Verify content
            import json
            with open(json_file, 'r') as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]['event_type'] == 'signal_generation'
        finally:
            os.unlink(json_file)
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_file = f.name
        
        try:
            self.logger.export_logs(csv_file, format='csv')
            assert os.path.exists(csv_file)
        finally:
            os.unlink(csv_file)
    
    def test_global_logger_functions(self):
        """Test global logger convenience functions."""
        from src.portfolio_backtester.timing.timing_logger import (
            get_timing_logger, log_signal_generation, configure_timing_logging
        )
        
        # Configure global logging
        configure_timing_logging(enable_detailed_logging=True, log_level='DEBUG')
        
        # Test convenience functions
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        log_signal_generation(strategy_name, current_date, True, 'Test signal')
        
        # Get global logger and check entries
        global_logger = get_timing_logger()
        entries = global_logger.get_log_entries()
        assert len(entries) >= 1
        
        # Find our entry
        our_entries = [e for e in entries if e.strategy_name == strategy_name]
        assert len(our_entries) == 1
        assert our_entries[0].event_type == 'signal_generation'


class TestIntegrationWithExistingFramework:
    """Test integration of new features with existing timing framework."""
    
    def test_enhanced_backward_compatibility(self):
        """Test enhanced backward compatibility with validation."""
        from src.portfolio_backtester.timing.backward_compatibility import ensure_backward_compatibility
        
        # Test valid legacy config
        legacy_config = {
            'strategy_params': {'lookback_period': 252},
            'rebalance_frequency': 'M'
        }
        
        migrated = ensure_backward_compatibility(legacy_config)
        assert 'timing_config' in migrated
        assert migrated['timing_config']['mode'] == 'time_based'
        assert migrated['timing_config']['rebalance_frequency'] == 'M'
        assert 'rebalance_frequency' not in migrated  # Should be removed
    
    def test_configuration_examples_validation(self):
        """Test that configuration examples are valid."""
        # Load and validate examples
        examples_file = 'config/timing_examples.yaml'
        
        if os.path.exists(examples_file):
            with open(examples_file, 'r') as f:
                examples = yaml.safe_load(f)
            
            # Test a few key examples
            test_examples = [
                'basic_monthly_strategy',
                'daily_signal_strategy',
                'adaptive_strategy'
            ]
            
            for example_name in test_examples:
                if example_name in examples:
                    config = examples[example_name]
                    if 'timing_config' in config:
                        errors = validate_timing_config(config, raise_on_error=False)
                        error_count = sum(1 for e in errors if e.severity == 'error')
                        assert error_count == 0, f"Example {example_name} has validation errors: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])