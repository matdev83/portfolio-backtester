"""Tests for risk-off signal generator system."""

import pytest
import pandas as pd

from portfolio_backtester.risk_off_signals import (
    IRiskOffSignalGenerator,
    NoRiskOffSignalGenerator,
    DummyRiskOffSignalGenerator,
    RiskOffSignalProviderFactory,
)


class TestNoRiskOffSignalGenerator:
    """Test the default no-risk-off signal generator."""

    def test_initialization(self):
        """Test that generator initializes properly."""
        generator = NoRiskOffSignalGenerator()
        assert generator is not None
        assert isinstance(generator, IRiskOffSignalGenerator)

    def test_never_signals_risk_off(self):
        """Test that generator never signals risk-off conditions."""
        generator = NoRiskOffSignalGenerator()

        # Create dummy data
        all_data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=10))
        benchmark_data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=10))
        non_universe_data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=10))
        current_date = pd.Timestamp("2020-01-05")

        signal = generator.generate_risk_off_signal(
            all_data, benchmark_data, non_universe_data, current_date
        )

        # Should always return False (risk-on)
        assert signal is False

    def test_configuration(self):
        """Test configuration methods."""
        generator = NoRiskOffSignalGenerator({"test": "value"})

        config = generator.get_configuration()
        assert isinstance(config, dict)
        assert config.get("test") == "value"

        is_valid, error = generator.validate_configuration({"any": "config"})
        assert is_valid is True
        assert error == ""

    def test_data_requirements(self):
        """Test data requirement methods."""
        generator = NoRiskOffSignalGenerator()

        assert generator.get_required_data_columns() == []
        assert generator.get_minimum_data_periods() == 0
        assert not generator.supports_non_universe_data()

        description = generator.get_signal_description()
        assert "Never signals risk-off" in description


class TestDummyRiskOffSignalGenerator:
    """Test the dummy risk-off signal generator."""

    def test_initialization_default(self):
        """Test default initialization."""
        generator = DummyRiskOffSignalGenerator()
        assert generator is not None
        assert isinstance(generator, IRiskOffSignalGenerator)

    def test_default_windows(self):
        """Test that default windows are set up correctly."""
        generator = DummyRiskOffSignalGenerator()

        # Test dates within default windows (2008 financial crisis)
        current_date = pd.Timestamp("2008-10-01")
        signal = generator.generate_risk_off_signal(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_date
        )
        assert signal is True  # Should signal risk-off

        # Test date outside windows
        current_date = pd.Timestamp("2010-01-01")
        signal = generator.generate_risk_off_signal(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_date
        )
        assert signal is False  # Should be risk-on by default

    def test_custom_windows(self):
        """Test custom risk-off windows."""
        config = {
            "risk_off_windows": [
                ("2023-01-01", "2023-01-31"),
                ("2023-06-01", "2023-06-30"),
            ],
            "default_risk_state": "on",
        }
        generator = DummyRiskOffSignalGenerator(config)

        # Test date in custom window
        signal = generator.generate_risk_off_signal(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2023-01-15")
        )
        assert signal is True

        # Test date outside window
        signal = generator.generate_risk_off_signal(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2023-02-15")
        )
        assert signal is False

    def test_default_risk_state_off(self):
        """Test default risk state set to off."""
        config = {"default_risk_state": "off"}
        generator = DummyRiskOffSignalGenerator(config)

        # Date outside any window with default off should return True (risk-off)
        signal = generator.generate_risk_off_signal(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2010-01-01")
        )
        assert signal is True

    def test_configuration_validation(self):
        """Test configuration validation."""
        generator = DummyRiskOffSignalGenerator()

        # Valid config
        valid_config = {
            "default_risk_state": "on",
            "risk_off_windows": [("2023-01-01", "2023-01-31")],
        }
        is_valid, error = generator.validate_configuration(valid_config)
        assert is_valid is True
        assert error == ""

        # Invalid default state
        invalid_config = {"default_risk_state": "invalid"}
        is_valid, error = generator.validate_configuration(invalid_config)
        assert is_valid is False
        assert "Invalid default_risk_state" in error

        # Invalid window format
        invalid_config = {"risk_off_windows": ["not_a_tuple"]}  # type: ignore[dict-item]
        is_valid, error = generator.validate_configuration(invalid_config)
        assert is_valid is False
        assert "must be a (start_date, end_date) tuple" in error


class TestRiskOffSignalProviderFactory:
    """Test the provider factory."""

    def test_get_default_provider(self):
        """Test getting default provider."""
        strategy_config: dict[str, object] = {}
        provider = RiskOffSignalProviderFactory.get_default_provider(strategy_config)

        assert provider is not None
        generator = provider.get_risk_off_signal_generator()
        assert isinstance(generator, NoRiskOffSignalGenerator)

    def test_config_provider_with_dummy(self):
        """Test config provider with dummy generator."""
        strategy_config = {
            "risk_off_signal_config": {
                "type": "DummyRiskOffSignalGenerator",
                "default_risk_state": "on",
            }
        }
        provider = RiskOffSignalProviderFactory.create_config_provider(strategy_config)

        generator = provider.get_risk_off_signal_generator()
        assert isinstance(generator, DummyRiskOffSignalGenerator)
        assert provider.supports_risk_off_signals() is True

    def test_fixed_provider(self):
        """Test fixed provider."""
        provider = RiskOffSignalProviderFactory.create_fixed_provider("NoRiskOffSignalGenerator")

        generator = provider.get_risk_off_signal_generator()
        assert isinstance(generator, NoRiskOffSignalGenerator)
        assert provider.supports_risk_off_signals() is False

    def test_invalid_generator_type(self):
        """Test invalid generator type raises error."""
        strategy_config = {"risk_off_signal_config": {"type": "InvalidGenerator"}}
        provider = RiskOffSignalProviderFactory.create_config_provider(strategy_config)

        with pytest.raises(ValueError, match="Unknown risk-off signal generator type"):
            provider.get_risk_off_signal_generator()


if __name__ == "__main__":
    pytest.main([__file__])
