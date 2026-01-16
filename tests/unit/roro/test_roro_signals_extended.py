import pytest
import pandas as pd
from src.portfolio_backtester.roro_signals.roro_signals import DummyRoRoSignal

class TestRoRoSignals:
    def test_dummy_roro_signal(self):
        signal = DummyRoRoSignal()
        
        # Risk ON windows:
        # 2006-01-01 to 2009-12-31
        # 2020-01-01 to 2020-04-01
        # 2022-01-01 to 2022-11-05
        
        # Test Risk ON (True)
        assert signal.generate_signal(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2007-06-01")) is True
        assert signal.generate_signal(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2020-02-15")) is True
        assert signal.generate_signal(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2022-05-01")) is True
        
        # Test Risk OFF (False)
        assert signal.generate_signal(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2000-01-01")) is False
        assert signal.generate_signal(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2015-01-01")) is False
        assert signal.generate_signal(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2021-06-01")) is False

    def test_required_features(self):
        signal = DummyRoRoSignal()
        assert signal.get_required_features() == set()
