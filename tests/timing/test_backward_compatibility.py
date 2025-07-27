"""
Tests for backward compatibility layer in timing system.

These tests ensure that:
1. Legacy configurations are properly migrated
2. All existing strategy configurations remain valid
3. Migrated configurations produce identical behavior
4. Helpful error messages are provided for invalid configurations
"""

import pytest
import warnings
from unittest.mock import Mock, patch

from src.portfolio_backtester.timing.backward_compatibility import (
    migrate_legacy_config,
    ensure_backward_compatibility,
    validate_legacy_behavior,
    get_migration_warnings,
    TimingConfigValidator,
    get_legacy_config_examples,
    check_migration_compatibility,
    LEGACY_CONFIG_MAPPING,
    KNOWN_DAILY_SIGNAL_STRATEGIES,
    DEFAULT_TIMING_CONFIGS
)


class TestTimingConfigValidator:
    """Test timing configuration validation."""
    
    def test_validate_time_based_config_valid(self):
        """Test validation of valid time-based configuration."""
        config = {
            'mode': 'time_based',
            'rebalance_frequency': 'M',
            'rebalance_offset': 0
        }
        errors = TimingConfigValidator.validate_config(config)
        assert errors == []
    
    def test_validate_time_based_config_invalid_frequency(self):
        """Test validation catches invalid rebalance frequency."""
        config = {
            'mode': 'time_based',
            'rebalance_frequency': 'INVALID',
        }
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        assert 'Invalid rebalance_frequency' in errors[0]
        assert 'Common values' in errors[0]  # Helpful suggestion
    
    def test_validate_time_based_config_invalid_offset(self):
        """Test validation catches invalid rebalance offset."""
        config = {
            'mode': 'time_based',
            'rebalance_frequency': 'M',
            'rebalance_offset': 50  # Too large
        }
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        assert 'rebalance_offset must be an integer between -30 and 30' in errors[0]
    
    def test_validate_signal_based_config_valid(self):
        """Test validation of valid signal-based configuration."""
        config = {
            'mode': 'signal_based',
            'scan_frequency': 'D',
            'min_holding_period': 1,
            'max_holding_period': 10
        }
        errors = TimingConfigValidator.validate_config(config)
        assert errors == []
    
    def test_validate_signal_based_config_invalid_scan_frequency(self):
        """Test validation catches invalid scan frequency."""
        config = {
            'mode': 'signal_based',
            'scan_frequency': 'INVALID',
        }
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        assert 'Invalid scan_frequency' in errors[0]
        assert 'Use \'D\' for daily' in errors[0]  # Helpful suggestion
    
    def test_validate_signal_based_config_invalid_holding_periods(self):
        """Test validation catches invalid holding periods."""
        config = {
            'mode': 'signal_based',
            'scan_frequency': 'D',
            'min_holding_period': 5,
            'max_holding_period': 2  # Less than min
        }
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        assert 'min_holding_period (5) cannot exceed max_holding_period (2)' in errors[0]
    
    def test_validate_invalid_mode(self):
        """Test validation catches invalid timing mode."""
        config = {
            'mode': 'invalid_mode',
        }
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        assert 'Invalid timing mode' in errors[0]
        assert 'time_based' in errors[0] and 'signal_based' in errors[0]


class TestLegacyConfigMigration:
    """Test legacy configuration migration."""
    
    def test_migrate_monthly_momentum_strategy(self):
        """Test migration of typical monthly momentum strategy."""
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {'lookback_months': 6}
        }
        
        migrated = migrate_legacy_config(legacy_config)
        
        assert 'timing_config' in migrated
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'time_based'
        assert timing_config['rebalance_frequency'] == 'M'
        assert timing_config['rebalance_offset'] == 0
    
    def test_migrate_quarterly_strategy_with_offset(self):
        """Test migration of quarterly strategy with rebalance offset."""
        legacy_config = {
            'strategy': 'value',
            'rebalance_frequency': 'Q',
            'rebalance_offset': 5,
            'strategy_params': {'book_to_market': True}
        }
        
        migrated = migrate_legacy_config(legacy_config)
        
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'time_based'
        assert timing_config['rebalance_frequency'] == 'Q'
        assert timing_config['rebalance_offset'] == 5
    
    def test_migrate_daily_uvxy_strategy(self):
        """Test migration of UVXY strategy (known daily signal strategy)."""
        legacy_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'strategy_params': {'rsi_period': 2, 'rsi_threshold': 30}
        }
        
        migrated = migrate_legacy_config(legacy_config)
        
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['scan_frequency'] == 'D'
        assert timing_config['min_holding_period'] == 1
        assert timing_config['max_holding_period'] == 1  # UVXY-specific default
    
    def test_migrate_explicit_daily_signals(self):
        """Test migration of strategy with explicit daily_signals flag."""
        legacy_config = {
            'strategy': 'custom_strategy',
            'daily_signals': True,
            'scan_frequency': 'D',
            'min_holding_period': 2,
            'max_holding_period': 10
        }
        
        migrated = migrate_legacy_config(legacy_config)
        
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['scan_frequency'] == 'D'
        assert timing_config['min_holding_period'] == 2
        assert timing_config['max_holding_period'] == 10
    
    def test_migrate_signal_based_flag(self):
        """Test migration of strategy with explicit signal_based flag."""
        legacy_config = {
            'strategy': 'breakout',
            'signal_based': True,
            'scan_frequency': 'W'
        }
        
        migrated = migrate_legacy_config(legacy_config)
        
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['scan_frequency'] == 'W'
    
    def test_migrate_already_has_timing_config(self):
        """Test that strategies with timing_config are not modified."""
        config_with_timing = {
            'strategy': 'momentum',
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'Q'
            },
            'strategy_params': {'lookback_months': 6}
        }
        
        migrated = migrate_legacy_config(config_with_timing)
        
        # Should be unchanged
        assert migrated == config_with_timing
    
    def test_migrate_invalid_timing_config_raises_error(self):
        """Test that invalid existing timing_config raises helpful error."""
        config_with_invalid_timing = {
            'strategy': 'momentum',
            'timing_config': {
                'mode': 'invalid_mode',
                'rebalance_frequency': 'INVALID'
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            migrate_legacy_config(config_with_invalid_timing)
        
        assert 'Invalid timing_config' in str(exc_info.value)
        assert 'Invalid timing mode' in str(exc_info.value)


class TestLegacyBehaviorValidation:
    """Test validation that legacy behavior is preserved."""
    
    def test_validate_time_based_migration(self):
        """Test validation of time-based strategy migration."""
        old_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M'
        }
        
        new_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M',
                'rebalance_offset': 0
            }
        }
        
        assert validate_legacy_behavior(old_config, new_config) is True
    
    def test_validate_signal_based_migration(self):
        """Test validation of signal-based strategy migration."""
        old_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D'
        }
        
        new_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'D',
                'min_holding_period': 1,
                'max_holding_period': 1
            }
        }
        
        assert validate_legacy_behavior(old_config, new_config) is True
    
    def test_validate_behavior_mismatch_mode(self):
        """Test validation catches timing mode mismatch."""
        old_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M'
        }
        
        new_config = {
            'strategy': 'momentum',
            'timing_config': {
                'mode': 'signal_based',  # Wrong mode
                'scan_frequency': 'D'
            }
        }
        
        assert validate_legacy_behavior(old_config, new_config) is False
    
    def test_validate_behavior_mismatch_frequency(self):
        """Test validation catches frequency mismatch."""
        old_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M'
        }
        
        new_config = {
            'strategy': 'momentum',
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'Q'  # Wrong frequency
            }
        }
        
        assert validate_legacy_behavior(old_config, new_config) is False


class TestMigrationWarnings:
    """Test migration warning system."""
    
    def test_get_warnings_deprecated_parameters(self):
        """Test warnings for deprecated parameters."""
        config = {
            'strategy': 'test',
            'daily_signals': True,
            'signal_based': True,
            'rebalance_offset': 5
        }
        
        warnings_list = get_migration_warnings(config)
        
        assert len(warnings_list) == 3
        assert any('daily_signals' in warning for warning in warnings_list)
        assert any('signal_based' in warning for warning in warnings_list)
        assert any('rebalance_offset' in warning for warning in warnings_list)
    
    def test_get_warnings_daily_frequency_conflict(self):
        """Test warning for daily frequency without signal-based mode."""
        config = {
            'strategy': 'test',
            'rebalance_frequency': 'D',
            'timing_config': {
                'mode': 'time_based'  # Conflict
            }
        }
        
        warnings_list = get_migration_warnings(config)
        
        assert len(warnings_list) == 1
        assert 'Daily rebalance_frequency detected' in warnings_list[0]
        assert 'signal_based' in warnings_list[0]


class TestBackwardCompatibilityIntegration:
    """Test the main backward compatibility entry point."""
    
    def test_ensure_backward_compatibility_success(self):
        """Test successful backward compatibility migration."""
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {'lookback_months': 6}
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = ensure_backward_compatibility(legacy_config)
            
            # Should have timing_config
            assert 'timing_config' in result
            timing_config = result['timing_config']
            assert timing_config['mode'] == 'time_based'
            assert timing_config['rebalance_frequency'] == 'M'
            
            # Should not have warnings for this simple case
            assert len(w) == 0
    
    def test_ensure_backward_compatibility_with_warnings(self):
        """Test backward compatibility with deprecation warnings."""
        legacy_config = {
            'strategy': 'test',
            'daily_signals': True,  # Deprecated parameter
            'rebalance_frequency': 'D'
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = ensure_backward_compatibility(legacy_config)
            
            # Should have timing_config
            assert 'timing_config' in result
            assert result['timing_config']['mode'] == 'signal_based'
            
            # Should have deprecation warnings (may be multiple)
            assert len(w) >= 1
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            
            # Check that at least one warning mentions daily_signals
            daily_signals_warnings = [w for w in deprecation_warnings if 'daily_signals' in str(w.message)]
            assert len(daily_signals_warnings) >= 1
    
    def test_ensure_backward_compatibility_validation_failure(self):
        """Test backward compatibility with validation failure."""
        invalid_config = {
            'strategy': 'test',
            'timing_config': {
                'mode': 'invalid_mode'
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            ensure_backward_compatibility(invalid_config)
        
        # The error message should contain migration failure info since validation happens during migration
        assert 'Configuration migration failed' in str(exc_info.value)
        assert 'Invalid timing mode' in str(exc_info.value)
    
    def test_ensure_backward_compatibility_migration_failure(self):
        """Test backward compatibility with migration failure."""
        # Mock the migration to fail
        with patch('src.portfolio_backtester.timing.backward_compatibility.migrate_legacy_config') as mock_migrate:
            mock_migrate.side_effect = Exception("Migration failed")
            
            config = {'strategy': 'test'}
            
            with pytest.raises(ValueError) as exc_info:
                ensure_backward_compatibility(config)
            
            assert 'Configuration migration failed' in str(exc_info.value)
            assert 'timing_config' in str(exc_info.value)  # Helpful suggestion


class TestLegacyConfigExamples:
    """Test the provided legacy configuration examples."""
    
    def test_all_examples_migrate_successfully(self):
        """Test that all provided examples migrate successfully."""
        examples = get_legacy_config_examples()
        
        for name, config in examples.items():
            try:
                migrated = migrate_legacy_config(config)
                assert 'timing_config' in migrated
                assert validate_legacy_behavior(config, migrated)
            except Exception as e:
                pytest.fail(f"Example '{name}' failed to migrate: {e}")
    
    def test_migration_compatibility_test(self):
        """Test the built-in migration compatibility test."""
        # This should pass if all examples are valid
        result = check_migration_compatibility()
        assert result is True


class TestStrategySpecificMigration:
    """Test strategy-specific migration logic."""
    
    def test_uvxy_strategy_gets_correct_defaults(self):
        """Test that UVXY strategy gets appropriate defaults."""
        config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D'
        }
        
        migrated = migrate_legacy_config(config)
        timing_config = migrated['timing_config']
        
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['min_holding_period'] == 1
        assert timing_config['max_holding_period'] == 1
    
    def test_known_daily_strategies_detected(self):
        """Test that known daily signal strategies are detected."""
        for strategy_name in KNOWN_DAILY_SIGNAL_STRATEGIES:
            config = {
                'strategy': strategy_name,
                'rebalance_frequency': 'M'  # Even with monthly, should be signal-based
            }
            
            migrated = migrate_legacy_config(config)
            assert migrated['timing_config']['mode'] == 'signal_based'


class TestConfigurationConstants:
    """Test configuration constants and mappings."""
    
    def test_legacy_config_mapping_completeness(self):
        """Test that legacy config mapping covers expected parameters."""
        expected_mappings = [
            'rebalance_frequency',
            'supports_daily_signals',
            'daily_signals',
            'signal_based'
        ]
        
        for param in expected_mappings:
            assert param in LEGACY_CONFIG_MAPPING
    
    def test_default_timing_configs_valid(self):
        """Test that default timing configurations are valid."""
        for mode, config in DEFAULT_TIMING_CONFIGS.items():
            errors = TimingConfigValidator.validate_config(config)
            assert errors == [], f"Default config for {mode} is invalid: {errors}"
    
    def test_known_daily_strategies_not_empty(self):
        """Test that we have known daily signal strategies defined."""
        assert len(KNOWN_DAILY_SIGNAL_STRATEGIES) > 0
        assert 'uvxy_rsi' in KNOWN_DAILY_SIGNAL_STRATEGIES


class TestErrorMessages:
    """Test that error messages are helpful and actionable."""
    
    def test_validation_error_messages_helpful(self):
        """Test that validation error messages provide helpful guidance."""
        config = {
            'mode': 'time_based',
            'rebalance_frequency': 'INVALID'
        }
        
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        
        error = errors[0]
        # Should contain the invalid value
        assert 'INVALID' in error
        # Should suggest valid alternatives
        assert 'Common values' in error
        assert "'M'" in error  # Monthly example
        assert "'Q'" in error  # Quarterly example
    
    def test_migration_failure_error_helpful(self):
        """Test that migration failure provides helpful error message."""
        # This is tested in the integration test above
        # The error message should include timing_config example
        pass


if __name__ == '__main__':
    pytest.main([__file__])