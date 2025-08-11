"""
Tests for YAML configuration schema validation.
Split from test_configuration_extensibility.py for better organization.
"""

import pytest
import tempfile
import os
import yaml
from portfolio_backtester.timing.config_schema import (
    TimingConfigSchema,
    ValidationError,
    validate_timing_config,
)


class TestYAMLConfigurationSchema:
    """Test YAML configuration schema validation."""

    def test_valid_time_based_config(self):
        """Test validation of valid time-based configuration."""
        config = {
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "M",
                "rebalance_offset": 0,
            }
        }

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 0

    def test_valid_signal_based_config(self):
        """Test validation of valid signal-based configuration."""
        config = {
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 1,
                "max_holding_period": 5,
            }
        }

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 0

    def test_valid_custom_config(self):
        """Test validation of valid custom configuration."""
        config = {
            "timing_config": {
                "mode": "custom",
                "custom_controller_class": "my.module.CustomController",
                "custom_controller_params": {"param1": "value1"},
            }
        }

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 0

    def test_missing_timing_config(self):
        """Test validation when timing_config is missing."""
        config = {"strategy_params": {}}

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == "timing_config"
        assert "Missing timing_config section" in errors[0].message

    def test_invalid_mode(self):
        """Test validation of invalid timing mode."""
        config = {"timing_config": {"mode": "invalid_mode"}}

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == "mode"
        assert "Invalid timing mode" in errors[0].message
        assert "time_based, signal_based, custom" in errors[0].suggestion

    def test_invalid_rebalance_frequency(self):
        """Test validation of invalid rebalance frequency."""
        config = {"timing_config": {"mode": "time_based", "rebalance_frequency": "X"}}

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == "rebalance_frequency"
        assert "Invalid rebalance frequency" in errors[0].message

    def test_invalid_holding_periods(self):
        """Test validation of invalid holding periods."""
        config = {
            "timing_config": {
                "mode": "signal_based",
                "min_holding_period": 5,
                "max_holding_period": 2,
            }
        }

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert "cannot exceed" in errors[0].message

    def test_warning_for_wrong_mode_fields(self):
        """Test warnings for fields used in wrong mode."""
        config = {
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "M",
                "scan_frequency": "D",  # Wrong mode field
            }
        }

        errors = TimingConfigSchema.validate_config(config)
        warnings = [e for e in errors if e.severity == "warning"]
        assert len(warnings) == 1
        assert warnings[0].field == "scan_frequency"
        assert "not used in time_based mode" in warnings[0].message

    def test_missing_custom_controller_class(self):
        """Test validation when custom controller class is missing."""
        config = {"timing_config": {"mode": "custom"}}

        errors = TimingConfigSchema.validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == "custom_controller_class"
        assert "required for custom mode" in errors[0].message

    def test_default_config_generation(self):
        """Test default configuration generation."""
        time_based_default = TimingConfigSchema.get_default_config("time_based")
        assert time_based_default["mode"] == "time_based"
        assert time_based_default["rebalance_frequency"] == "M"

        signal_based_default = TimingConfigSchema.get_default_config("signal_based")
        assert signal_based_default["mode"] == "signal_based"
        assert signal_based_default["scan_frequency"] == "D"

        custom_default = TimingConfigSchema.get_default_config("custom")
        assert custom_default["mode"] == "custom"
        assert "custom_controller_class" in custom_default

    def test_yaml_file_validation(self):
        """Test YAML file validation."""
        # Create temporary YAML file
        config_data = {"timing_config": {"mode": "time_based", "rebalance_frequency": "M"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            errors = TimingConfigSchema.validate_yaml_file(temp_file)
            assert len(errors) == 0
        finally:
            os.unlink(temp_file)

    def test_yaml_file_not_found(self):
        """Test validation of non-existent YAML file."""
        errors = TimingConfigSchema.validate_yaml_file("nonexistent.yaml")
        assert len(errors) == 1
        assert errors[0].field == "file"
        assert "not found" in errors[0].message

    def test_invalid_yaml_syntax(self):
        """Test validation of invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            temp_file = f.name

        try:
            errors = TimingConfigSchema.validate_yaml_file(temp_file)
            assert len(errors) == 1
            assert errors[0].field == "yaml"
            assert "Invalid YAML syntax" in errors[0].message
        finally:
            os.unlink(temp_file)

    def test_validation_report_formatting(self):
        """Test validation report formatting."""
        errors = [
            ValidationError("field1", "value1", "Error message 1", "Suggestion 1", "error"),
            ValidationError("field2", "value2", "Warning message 2", "Suggestion 2", "warning"),
        ]

        report = TimingConfigSchema.format_validation_report(errors)

        assert "Configuration Validation Report" in report
        assert "Found 1 error(s) and 1 warning(s)" in report
        assert "Error message 1" in report
        assert "Warning message 2" in report
        assert "Suggestion 1" in report

    def test_convenience_validation_functions(self):
        """Test convenience validation functions."""
        valid_config = {"timing_config": {"mode": "time_based", "rebalance_frequency": "M"}}

        # Should not raise exception
        errors = validate_timing_config(valid_config, raise_on_error=False)
        assert len(errors) == 0

        # Should not raise exception
        validate_timing_config(valid_config, raise_on_error=True)

        # Should raise exception for invalid config
        invalid_config = {"timing_config": {"mode": "invalid"}}

        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_timing_config(invalid_config, raise_on_error=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
