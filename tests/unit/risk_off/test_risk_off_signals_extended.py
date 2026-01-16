import pytest
import pandas as pd
from src.portfolio_backtester.risk_off_signals.implementations import (
    NoRiskOffSignalGenerator,
    DummyRiskOffSignalGenerator,
)
from src.portfolio_backtester.risk_off_signals.provider import (
    RiskOffSignalProviderFactory,
    ConfigBasedRiskOffSignalProvider,
    FixedRiskOffSignalProvider,
)

class TestRiskOffSignalGenerators:
    def test_no_risk_off_signal(self):
        generator = NoRiskOffSignalGenerator()
        assert generator.generate_risk_off_signal(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2020-01-01")) is False
        assert generator.get_minimum_data_periods() == 0

    def test_dummy_risk_off_signal_defaults(self):
        generator = DummyRiskOffSignalGenerator()
        # Default windows: 2008-09-01 to 2009-03-31, 2020-02-15 to 2020-04-30
        
        # In window
        assert generator.generate_risk_off_signal(None, None, None, pd.Timestamp("2008-10-01")) is True
        assert generator.generate_risk_off_signal(None, None, None, pd.Timestamp("2020-03-01")) is True
        
        # Out of window
        assert generator.generate_risk_off_signal(None, None, None, pd.Timestamp("2019-01-01")) is False

    def test_dummy_risk_off_custom_windows(self):
        config = {
            "risk_off_windows": [
                ("2021-01-01", "2021-01-31")
            ],
            "default_risk_state": "on" # Means Risk ON (signal=False)
        }
        generator = DummyRiskOffSignalGenerator(config)
        
        assert generator.generate_risk_off_signal(None, None, None, pd.Timestamp("2021-01-15")) is True
        assert generator.generate_risk_off_signal(None, None, None, pd.Timestamp("2021-02-01")) is False

    def test_dummy_risk_off_validation(self):
        generator = DummyRiskOffSignalGenerator()
        
        valid_config = {
            "risk_off_windows": [("2021-01-01", "2021-01-31")],
            "default_risk_state": "on"
        }
        is_valid, msg = generator.validate_configuration(valid_config)
        assert is_valid
        
        invalid_config = {
            "risk_off_windows": "not a list"
        }
        is_valid, msg = generator.validate_configuration(invalid_config)
        assert not is_valid

        invalid_dates = {
            "risk_off_windows": [("2021-02-01", "2021-01-01")] # Start > End
        }
        is_valid, msg = generator.validate_configuration(invalid_dates)
        assert not is_valid


class TestRiskOffSignalProvider:
    def test_factory_config_based(self):
        strategy_config = {
            "risk_off_signal_config": {
                "type": "DummyRiskOffSignalGenerator"
            }
        }
        provider = RiskOffSignalProviderFactory.create_provider(strategy_config, "config")
        assert isinstance(provider, ConfigBasedRiskOffSignalProvider)
        assert provider.supports_risk_off_signals()
        
        generator = provider.get_risk_off_signal_generator()
        assert isinstance(generator, DummyRiskOffSignalGenerator)

    def test_factory_fixed(self):
        provider = RiskOffSignalProviderFactory.create_provider({}, "fixed")
        # Defaults to NoRiskOffSignalGenerator
        assert isinstance(provider, FixedRiskOffSignalProvider)
        assert not provider.supports_risk_off_signals()
        
        generator = provider.get_risk_off_signal_generator()
        assert isinstance(generator, NoRiskOffSignalGenerator)

    def test_unknown_generator_type(self):
        strategy_config = {
            "risk_off_signal_config": {
                "type": "UnknownGenerator"
            }
        }
        provider = ConfigBasedRiskOffSignalProvider(strategy_config)
        with pytest.raises(ValueError, match="Unknown risk-off signal generator type"):
            provider.get_risk_off_signal_generator()

    def test_default_provider(self):
        provider = RiskOffSignalProviderFactory.get_default_provider({})
        assert not provider.supports_risk_off_signals()
        assert isinstance(provider.get_risk_off_signal_generator(), NoRiskOffSignalGenerator)
