"""
Tests for TimingConfigValidator.
"""

import pytest
from src.portfolio_backtester.timing.config_validator import TimingConfigValidator


class TestTimingConfigValidator:
    """Test cases for TimingConfigValidator."""
    
    def test_validate_config_valid_time_based(self):
        """Test validation of valid time-based config."""
        config = {
            'mode': 'time_based',
            'rebalance_frequency': 'M',
            'rebalance_offset': 0
        }
        
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_valid_signal_based(self):
        """Test validation of valid signal-based config."""
        config = {
            'mode': 'signal_based',
            'scan_frequency': 'D',
            'min_holding_period': 1,
            'max_holding_period': 30
        }
        
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_invalid_mode(self):
        """Test validation with invalid mode."""
        config = {'mode': 'invalid_mode'}
        
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 1
        assert "Invalid timing mode 'invalid_mode'" in errors[0]
    
    def test_validate_config_missing_mode_defaults_to_time_based(self):
        """Test validation with missing mode defaults to time_based."""
        config = {}
        
        errors = TimingConfigValidator.validate_config(config)
        # Should validate as time_based with defaults
        assert len(errors) == 0
    
    def test_validate_time_based_config_valid(self):
        """Test time-based config validation with valid parameters."""
        config = {
            'rebalance_frequency': 'Q',
            'rebalance_offset': 5
        }
        
        errors = TimingConfigValidator.validate_time_based_config(config)
        assert len(errors) == 0
    
    def test_validate_time_based_config_invalid_frequency(self):
        """Test time-based config validation with invalid frequency."""
        config = {'rebalance_frequency': 'X'}
        
        errors = TimingConfigValidator.validate_time_based_config(config)
        assert len(errors) == 1
        assert "Invalid rebalance_frequency 'X'" in errors[0]
    
    def test_validate_time_based_config_invalid_offset_type(self):
        """Test time-based config validation with invalid offset type."""
        config = {'rebalance_offset': 'invalid'}
        
        errors = TimingConfigValidator.validate_time_based_config(config)
        assert len(errors) == 1
        assert "rebalance_offset must be an integer" in errors[0]
    
    def test_validate_time_based_config_offset_too_large(self):
        """Test time-based config validation with offset too large."""
        config = {'rebalance_offset': 50}
        
        errors = TimingConfigValidator.validate_time_based_config(config)
        assert len(errors) == 1
        assert "rebalance_offset must be an integer between -30 and 30" in errors[0]
    
    def test_validate_time_based_config_negative_offset(self):
        """Test time-based config validation with valid negative offset."""
        config = {'rebalance_offset': -10}
        
        errors = TimingConfigValidator.validate_time_based_config(config)
        assert len(errors) == 0
    
    def test_validate_signal_based_config_valid(self):
        """Test signal-based config validation with valid parameters."""
        config = {
            'scan_frequency': 'W',
            'min_holding_period': 5,
            'max_holding_period': 20
        }
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 0
    
    def test_validate_signal_based_config_invalid_scan_frequency(self):
        """Test signal-based config validation with invalid scan frequency."""
        config = {'scan_frequency': 'X'}
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 1
        assert "Invalid scan_frequency 'X'" in errors[0]
    
    def test_validate_signal_based_config_invalid_max_holding_type(self):
        """Test signal-based config validation with invalid max holding type."""
        config = {'max_holding_period': 'invalid'}
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 1
        assert "max_holding_period must be a positive integer" in errors[0]
    
    def test_validate_signal_based_config_negative_max_holding(self):
        """Test signal-based config validation with negative max holding."""
        config = {'max_holding_period': -5}
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 1  # Only the max_holding error, not the comparison error
        assert "max_holding_period must be a positive integer" in errors[0]
    
    def test_validate_signal_based_config_invalid_min_holding_type(self):
        """Test signal-based config validation with invalid min holding type."""
        config = {'min_holding_period': 'invalid'}
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 1
        assert "min_holding_period must be a positive integer" in errors[0]
    
    def test_validate_signal_based_config_zero_min_holding(self):
        """Test signal-based config validation with zero min holding."""
        config = {'min_holding_period': 0}
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 1
        assert "min_holding_period must be a positive integer" in errors[0]
    
    def test_validate_signal_based_config_min_greater_than_max(self):
        """Test signal-based config validation with min > max holding."""
        config = {
            'min_holding_period': 10,
            'max_holding_period': 5
        }
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 1
        assert "min_holding_period (10) cannot exceed max_holding_period (5)" in errors[0]
    
    def test_validate_signal_based_config_none_max_holding(self):
        """Test signal-based config validation with None max holding."""
        config = {
            'min_holding_period': 5,
            'max_holding_period': None
        }
        
        errors = TimingConfigValidator.validate_signal_based_config(config)
        assert len(errors) == 0  # None is valid for max_holding_period
    
    def test_get_default_config_time_based(self):
        """Test getting default time-based config."""
        config = TimingConfigValidator.get_default_config('time_based')
        
        expected = {
            'mode': 'time_based',
            'rebalance_frequency': 'M',
            'rebalance_offset': 0
        }
        assert config == expected
    
    def test_get_default_config_signal_based(self):
        """Test getting default signal-based config."""
        config = TimingConfigValidator.get_default_config('signal_based')
        
        expected = {
            'mode': 'signal_based',
            'scan_frequency': 'D',
            'min_holding_period': 1,
            'max_holding_period': None
        }
        assert config == expected
    
    def test_get_default_config_invalid_mode(self):
        """Test getting default config with invalid mode."""
        with pytest.raises(ValueError, match="Unknown timing mode: invalid"):
            TimingConfigValidator.get_default_config('invalid')
    
    def test_migrate_legacy_config_already_new_format(self):
        """Test migration when config is already in new format."""
        config = {
            'strategy': 'test',
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'Q'
            }
        }
        
        result = TimingConfigValidator.migrate_legacy_config(config)
        assert result == config  # Should be unchanged
    
    def test_migrate_legacy_config_with_rebalance_frequency(self):
        """Test migration of legacy rebalance_frequency."""
        config = {
            'strategy': 'test',
            'rebalance_frequency': 'Q'
        }
        
        result = TimingConfigValidator.migrate_legacy_config(config)
        
        assert 'timing_config' in result
        assert result['timing_config']['mode'] == 'time_based'
        assert result['timing_config']['rebalance_frequency'] == 'Q'
    
    def test_migrate_legacy_config_minimal(self):
        """Test migration of minimal legacy config."""
        config = {'strategy': 'test'}
        
        result = TimingConfigValidator.migrate_legacy_config(config)
        
        assert 'timing_config' in result
        assert result['timing_config']['mode'] == 'time_based'
    
    def test_validate_all_valid_frequencies(self):
        """Test validation of all valid frequencies."""
        valid_frequencies = [
            # Daily and weekly
            'D', 'B', 'W', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
            # Monthly
            'M', 'ME', 'BM', 'BMS', 'MS',
            # Quarterly  
            'Q', 'QE', 'QS', 'BQ', 'BQS', '2Q',
            # Semi-annual
            '6M', '6ME', '6MS',
            # Annual
            'A', 'AS', 'Y', 'YE', 'YS', 'BA', 'BAS', 'BY', 'BYS', '2A',
            # Hourly (for high-frequency strategies)
            'H', '2H', '3H', '4H', '6H', '8H', '12H'
        ]
        
        for freq in valid_frequencies:
            config = {'rebalance_frequency': freq}
            errors = TimingConfigValidator.validate_time_based_config(config)
            assert len(errors) == 0, f"Frequency {freq} should be valid"
    
    def test_validate_all_valid_scan_frequencies(self):
        """Test validation of all valid scan frequencies."""
        valid_scan_frequencies = ['D', 'W', 'M']
        
        for freq in valid_scan_frequencies:
            config = {'scan_frequency': freq}
            errors = TimingConfigValidator.validate_signal_based_config(config)
            assert len(errors) == 0, f"Scan frequency {freq} should be valid"
    
    def test_multiple_errors_reported(self):
        """Test that multiple validation errors are reported."""
        config = {
            'mode': 'signal_based',
            'scan_frequency': 'X',
            'min_holding_period': 0,
            'max_holding_period': -5
        }
        
        errors = TimingConfigValidator.validate_config(config)
        assert len(errors) == 3  # scan_frequency, min_holding_period, max_holding_period errors