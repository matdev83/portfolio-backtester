import pandas as pd

from src.portfolio_backtester.risk_off_signals.implementations import (
    DummyRiskOffSignalGenerator,
)


def _dummy_risk_off_config() -> dict:
    """Mirrors legacy DummyRoRoSignal windows under risk-off semantics.

    Legacy RoRo returned True for risk-on in:
    2006-01-01..2009-12-31, 2020-01-01..2020-04-01, 2022-01-01..2022-11-05.
    Risk-off signal True is the complement; use point windows for sample dates.
    """
    return {
        "risk_off_windows": [
            ("2000-01-01", "2000-01-01"),
            ("2015-01-01", "2015-01-01"),
            ("2021-06-01", "2021-06-01"),
        ],
        "default_risk_state": "on",
    }


class TestRoRoSignals:
    def test_dummy_risk_off_signal_matches_legacy_dummy_windows(self):
        signal = DummyRiskOffSignalGenerator(_dummy_risk_off_config())
        empty = pd.DataFrame()

        # Legacy risk-on dates -> no risk-off
        assert (
            signal.generate_risk_off_signal(empty, empty, empty, pd.Timestamp("2007-06-01"))
            is False
        )
        assert (
            signal.generate_risk_off_signal(empty, empty, empty, pd.Timestamp("2020-02-15"))
            is False
        )
        assert (
            signal.generate_risk_off_signal(empty, empty, empty, pd.Timestamp("2022-05-01"))
            is False
        )

        # Legacy risk-off sample dates -> risk-off
        assert (
            signal.generate_risk_off_signal(empty, empty, empty, pd.Timestamp("2000-01-01")) is True
        )
        assert (
            signal.generate_risk_off_signal(empty, empty, empty, pd.Timestamp("2015-01-01")) is True
        )
        assert (
            signal.generate_risk_off_signal(empty, empty, empty, pd.Timestamp("2021-06-01")) is True
        )

    def test_required_data_columns(self):
        signal = DummyRiskOffSignalGenerator(_dummy_risk_off_config())
        assert signal.get_required_data_columns() == []
