"""
Property-based tests for timing validation.

This module uses Hypothesis to test invariants and properties of the timing validation
functions in the timing/validation module.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.timing.config_validator import TimingConfigValidator
from portfolio_backtester.timing.validation.mode_validators import TimeBasedValidator, SignalBasedValidator

from tests.strategies import frequencies


# Define valid frequencies from TimingConfigValidator
VALID_FREQUENCIES = [
    # Daily and weekly
    "D", "B", "W", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN",
    # Monthly
    "M", "ME", "BM", "BMS", "MS",
    # Quarterly
    "Q", "QE", "QS", "BQ", "BQS", "2Q",
    # Semi-annual
    "6M", "6ME", "6MS",
    # Annual
    "A", "AS", "Y", "YE", "YS", "BA", "BAS", "BY", "BYS", "2A",
    # Hourly
    "H", "2H", "3H", "4H", "6H", "8H", "12H",
]

# TimeBasedValidator has a more restricted set of valid frequencies
TIME_BASED_VALID_FREQUENCIES = ["D", "W", "M", "ME", "Q", "QE", "A", "Y", "YE"]


@st.composite
def time_based_configs(draw):
    """Generate valid and invalid time-based timing configurations."""
    # Decide whether to generate valid or invalid config
    generate_valid = draw(st.booleans())
    
    # Base config
    config = {
        "mode": "time_based",
    }
    
    # Add frequency
    if generate_valid:
        # Valid frequency (must be from the restricted TimeBasedValidator list)
        config["rebalance_frequency"] = draw(
            st.sampled_from(TIME_BASED_VALID_FREQUENCIES)
        )
    else:
        # Invalid frequency (with 50% probability)
        if draw(st.booleans()):
            config["rebalance_frequency"] = draw(
                st.sampled_from([
                    "X", "ZZ", "MM", "invalid", "2X", "monthly", 
                    # Include some valid pandas frequencies that are not in TimeBasedValidator's list
                    "B", "BM", "MS", "QS", "6M"
                ])
            )
    
    # Add offset
    if generate_valid:
        # Valid offset
        config["rebalance_offset"] = draw(st.integers(min_value=-30, max_value=30))
    else:
        # Invalid offset (with 50% probability if we haven't already made the config invalid)
        if "rebalance_frequency" in config and config["rebalance_frequency"] in TIME_BASED_VALID_FREQUENCIES:
            if draw(st.booleans()):
                config["rebalance_offset"] = draw(
                    st.one_of(
                        st.integers(min_value=-100, max_value=-31),
                        st.integers(min_value=31, max_value=100),
                        st.text()
                    )
                )
    
    return config, generate_valid


@given(time_based_configs())
@settings(deadline=None)
def test_time_based_validator_properties(data):
    """Test properties of TimeBasedValidator."""
    config, is_valid = data
    
    # Validate the config
    validator = TimeBasedValidator()
    errors = validator.validate(config)
    
    # Check validation results
    if is_valid:
        # Valid configs should have no errors
        # Note: The validator might be stricter than our test, so we check for specific errors
        frequency_errors = [e for e in errors if "rebalance_frequency" in e.field]
        offset_errors = [e for e in errors if "rebalance_offset" in e.field]
        
        # If there are errors, they shouldn't be about frequency or offset for valid configs
        if "rebalance_frequency" in config and config["rebalance_frequency"] in TIME_BASED_VALID_FREQUENCIES:
            assert not frequency_errors, f"Unexpected frequency errors for valid config: {frequency_errors}"
        
        if "rebalance_offset" in config and isinstance(config["rebalance_offset"], int) and -30 <= config["rebalance_offset"] <= 30:
            assert not offset_errors, f"Unexpected offset errors for valid config: {offset_errors}"
    else:
        # For invalid configs with known issues, check for specific errors
        if "rebalance_frequency" in config and config["rebalance_frequency"] not in TIME_BASED_VALID_FREQUENCIES:
            frequency_errors = [e for e in errors if "rebalance_frequency" in e.field]
            assert frequency_errors, f"Expected frequency errors for invalid frequency {config['rebalance_frequency']}"
        
        if "rebalance_offset" in config and (not isinstance(config["rebalance_offset"], int) or config["rebalance_offset"] < -30 or config["rebalance_offset"] > 30):
            offset_errors = [e for e in errors if "rebalance_offset" in e.field]
            assert offset_errors, f"Expected offset errors for invalid offset {config['rebalance_offset']}"


@st.composite
def signal_based_configs(draw):
    """Generate valid and invalid signal-based timing configurations."""
    # Decide whether to generate valid or invalid config
    generate_valid = draw(st.booleans())
    
    # Base config
    config = {
        "mode": "signal_based",
    }
    
    # For valid configs, always add required fields
    if generate_valid:
        config["signal_source"] = draw(
            st.sampled_from([
                "momentum_strategy", "volatility_indicator", "rsi_signal", "macd_signal"
            ])
        )
        config["trigger_condition"] = draw(
            st.sampled_from([
                "position_change", "threshold_breach", "crossover", "reversal"
            ])
        )
        # Add optional fields that make the config more realistic
        if draw(st.booleans()):
            config["min_time_between_signals"] = draw(
                st.sampled_from(["1D", "2D", "3D", "1W", "2W"])
            )
    else:
        # For invalid configs, add invalid scan_frequency or holding periods
        if draw(st.booleans()):
            config["scan_frequency"] = draw(
                st.sampled_from(["X", "ZZ", "invalid", "hourly", "yearly"])
            )
        
        if draw(st.booleans()):
            config["min_holding_period"] = draw(
                st.one_of(
                    st.integers(max_value=0),  # Invalid: must be >= 1
                    st.floats(),
                    st.text()
                )
            )
        
        if draw(st.booleans()):
            config["max_holding_period"] = draw(
                st.one_of(
                    st.integers(max_value=0),  # Invalid: must be >= 1
                    st.floats(),
                    st.text()
                )
            )
    
    return config, generate_valid


@given(signal_based_configs())
@settings(deadline=None)
def test_signal_based_validator_properties(data):
    """Test properties of SignalBasedValidator."""
    config, is_valid = data
    
    # Validate the config
    validator = SignalBasedValidator()
    errors = validator.validate(config)
    
    # For valid configs, check that required fields are present and no errors related to them
    if is_valid:
        assert "signal_source" in config, "Valid signal-based config must have signal_source"
        assert "trigger_condition" in config, "Valid signal-based config must have trigger_condition"
        assert isinstance(config["signal_source"], str) and config["signal_source"], "signal_source must be a non-empty string"
        assert isinstance(config["trigger_condition"], str) and config["trigger_condition"], "trigger_condition must be a non-empty string"
        
        # Check that there are no errors about scan_frequency if using default
        if "scan_frequency" not in config or config["scan_frequency"] in SignalBasedValidator.VALID_SCAN_FREQUENCIES:
            scan_freq_errors = [e for e in errors if "scan_frequency" in e.field]
            assert not scan_freq_errors, f"Unexpected scan_frequency errors: {scan_freq_errors}"
    else:
        # For invalid configs, check for specific errors
        if "scan_frequency" in config and config["scan_frequency"] not in SignalBasedValidator.VALID_SCAN_FREQUENCIES:
            scan_freq_errors = [e for e in errors if "scan_frequency" in e.field]
            assert scan_freq_errors, f"Expected scan_frequency errors for invalid frequency {config['scan_frequency']}"
        
        if "min_holding_period" in config and (not isinstance(config["min_holding_period"], int) or config["min_holding_period"] < 1):
            min_hold_errors = [e for e in errors if "min_holding_period" in e.field]
            assert min_hold_errors, f"Expected min_holding_period errors for invalid value {config['min_holding_period']}"
        
        if "max_holding_period" in config and (not isinstance(config["max_holding_period"], int) or config["max_holding_period"] < 1):
            max_hold_errors = [e for e in errors if "max_holding_period" in e.field]
            assert max_hold_errors, f"Expected max_holding_period errors for invalid value {config['max_holding_period']}"


@st.composite
def valid_frequency_configs(draw):
    """Generate valid timing configurations with various frequencies."""
    # Choose a frequency
    freq = draw(st.sampled_from(TIME_BASED_VALID_FREQUENCIES))
    
    # Create a config with the frequency
    config = {
        "mode": "time_based",
        "rebalance_frequency": freq,
        "rebalance_offset": draw(st.integers(min_value=-30, max_value=30)),
    }
    
    return config, freq


@given(valid_frequency_configs())
@settings(deadline=None)
def test_frequency_validation_properties(data):
    """Test that valid pandas frequencies are accepted by the validator."""
    config, freq = data
    
    # Validate using TimingConfigValidator
    errors = TimingConfigValidator.validate_config(config)
    
    # All frequencies from TIME_BASED_VALID_FREQUENCIES should be valid
    frequency_errors = [e for e in errors if "frequency" in e.lower()]
    assert not frequency_errors, f"Expected no frequency errors for valid frequency {freq}, got: {frequency_errors}"