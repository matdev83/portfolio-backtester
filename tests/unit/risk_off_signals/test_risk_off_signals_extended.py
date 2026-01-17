import pytest
import pandas as pd
from src.portfolio_backtester.risk_off_signals.implementations import (
    NoRiskOffSignalGenerator,
    DummyRiskOffSignalGenerator
)

class TestRiskOffSignalGenerators:
    @pytest.fixture
    def dummy_data(self):
        return pd.DataFrame()

    def test_no_risk_off_generator(self, dummy_data):
        gen = NoRiskOffSignalGenerator()
        
        # Always returns False
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2020-01-01")) is False
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2008-10-01")) is False
        
        # Validation always passes
        is_valid, msg = gen.validate_configuration({})
        assert is_valid
        assert msg == ""

    def test_dummy_generator_defaults(self, dummy_data):
        # Default behavior: uses hardcoded crisis windows
        gen = DummyRiskOffSignalGenerator()
        
        # 2008 Crisis (in default window)
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2008-10-01")) is True
        
        # COVID (in default window)
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2020-03-15")) is True
        
        # Normal time
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2015-01-01")) is False

    def test_dummy_generator_custom_windows(self, dummy_data):
        config = {
            "risk_off_windows": [
                ("2021-01-01", "2021-01-31"),
                ("2022-01-01", "2022-01-31")
            ],
            "default_risk_state": "on" # Means risk-on (signal=False) when not in window
        }
        gen = DummyRiskOffSignalGenerator(config)
        
        # In custom window
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2021-01-15")) is True
        
        # Outside custom window
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2021-02-01")) is False

    def test_dummy_generator_default_off(self, dummy_data):
        # Inverse logic: default state is 'off' (signal=True), windows are exceptions?
        # Re-reading code: "Return default state when not in risk-off windows... return bool(self._default_risk_state == 'off')"
        # So if default_risk_state is 'off', it returns True (Risk Off) outside windows.
        
        config = {
            "risk_off_windows": [], # No explicit windows
            "default_risk_state": "off" # Always Risk Off
        }
        gen = DummyRiskOffSignalGenerator(config)
        
        assert gen.generate_risk_off_signal(dummy_data, dummy_data, dummy_data, pd.Timestamp("2021-01-01")) is True

    def test_dummy_generator_validation(self):
        gen = DummyRiskOffSignalGenerator()
        
        # Valid config
        valid_config = {
            "risk_off_windows": [("2021-01-01", "2021-01-31")],
            "default_risk_state": "on"
        }
        is_valid, msg = gen.validate_configuration(valid_config)
        assert is_valid
        
        # Invalid default state
        invalid_state = {"default_risk_state": "invalid"}
        is_valid, msg = gen.validate_configuration(invalid_state)
        assert not is_valid
        assert "Invalid default_risk_state" in msg
        
        # Invalid window format (not tuple)
        invalid_window = {"risk_off_windows": ["2021-01-01"]}
        is_valid, msg = gen.validate_configuration(invalid_window)
        assert not is_valid
        assert "must be a (start_date, end_date) tuple" in msg
        
        # Invalid dates (start > end)
        inverted_window = {"risk_off_windows": [("2021-01-31", "2021-01-01")]}
        is_valid, msg = gen.validate_configuration(inverted_window)
        assert not is_valid
        assert "start_date must be before end_date" in msg
