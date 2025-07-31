import pandas as pd
import pytest
from src.portfolio_backtester.strategies.signal.intramonth_seasonal_strategy import IntramonthSeasonalStrategy

@pytest.mark.parametrize("direction,entry_day,hold_days,expected_weights", [
    ("long", 1, 2, 0.5),
])
def test_generate_signals_basic(direction, entry_day, hold_days, expected_weights):
    dates = pd.bdate_range(start="2023-01-01", end="2023-01-10")
    data = pd.DataFrame({"AAPL": 100, "GOOG": 200}, index=dates)
    config = {
        "strategy_params": {
            "direction": direction,
            "entry_day": entry_day,
            "hold_days": hold_days,
        }
    }
    strategy = IntramonthSeasonalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(dates[0], entry_day)
    signals = strategy.generate_signals(data, pd.DataFrame(), pd.DataFrame(), entry_date)
    assert all(signals.loc[entry_date] == expected_weights)


def test_generate_signals_short_entry():
    dates = pd.bdate_range(start="2023-01-01", end="2023-01-31")
    data = pd.DataFrame({"AAPL": 100, "GOOG": 200}, index=dates)
    config = {
        "strategy_params": {
            "direction": "short",
            "entry_day": -1,
            "hold_days": 2,
        }
    }
    strategy = IntramonthSeasonalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(dates[0], -1)
    signals = strategy.generate_signals(data, pd.DataFrame(), pd.DataFrame(), entry_date)
    assert all(signals.loc[entry_date] == -0.5)

def test_generate_signals_handles_missing_data():
    dates = pd.bdate_range(start="2023-01-01", end="2023-01-10")
    data = pd.DataFrame({"AAPL": 100, "GOOG": None}, index=dates)
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 2,
        }
    }
    strategy = IntramonthSeasonalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(dates[0], 1)
    signals = strategy.generate_signals(data, pd.DataFrame(), pd.DataFrame(), entry_date)
    # Only AAPL should have a weight
    assert signals.loc[entry_date, "AAPL"] == 1.0
    assert signals.loc[entry_date, "GOOG"] == 0.0

def test_generate_signals_allowed_months():
    dates = pd.bdate_range(start="2023-01-01", end="2023-03-31")
    data = pd.DataFrame({"AAPL": 100, "GOOG": 200}, index=dates)
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 2,
            "trade_month_1": False,
            "trade_month_2": True,
            "trade_month_3": False,
        }
    }
    strategy = IntramonthSeasonalStrategy(config)
    # Should only enter in February
    jan_entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 1)
    feb_entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-02-01"), 1)
    jan_signals = strategy.generate_signals(data, pd.DataFrame(), pd.DataFrame(), jan_entry)
    feb_signals = strategy.generate_signals(data, pd.DataFrame(), pd.DataFrame(), feb_entry)
    assert all(jan_signals.loc[jan_entry] == 0.0)
    assert all(feb_signals.loc[feb_entry] == 0.5)
