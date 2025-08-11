import pandas as pd
import pytest
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import SeasonalSignalStrategy

@pytest.fixture
def sample_historical_data():
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", end="2023-03-31", freq="B"))
    data = {"AAPL": 100, "GOOG": 200}
    df = pd.DataFrame(index=dates, columns=list(data.keys()))
    for col in df.columns:
        df[col] = data[col]
    return df

def test_month_filter_with_bool_params(sample_historical_data):
    """Strategy should handle boolean flags for allowed_months."""
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 3,
            "trade_month_1": True,
            "trade_month_2": False,
            "trade_month_3": True,
        }
    }
    strategy = SeasonalSignalStrategy(config)

    feb_first = pd.Timestamp("2023-02-01")
    signals_feb = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), feb_first
    )
    assert (signals_feb.loc[feb_first] == 0).all()

    jan_second = pd.Timestamp("2023-01-02")
    signals_jan = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), jan_second
    )
    assert (signals_jan.loc[jan_second] != 0).any()

    mar_first = pd.Timestamp("2023-03-01")
    signals_mar = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), mar_first
    )
    assert (signals_mar.loc[mar_first] != 0).any()

def test_get_entry_date_for_month_positive():
    strategy = SeasonalSignalStrategy({})
    date = pd.Timestamp("2023-01-15")
    entry_day = 5
    expected_date = pd.Timestamp("2023-01-06")
    assert strategy.get_entry_date_for_month(date, entry_day) == expected_date

def test_get_entry_date_for_month_negative():
    strategy = SeasonalSignalStrategy({})
    date = pd.Timestamp("2023-01-15")
    entry_day = -3
    expected_date = pd.Timestamp("2023-01-27")
    assert strategy.get_entry_date_for_month(date, entry_day) == expected_date

def test_long_strategy_entry_and_exit(sample_historical_data):
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 5,
            "hold_days": 3,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 5)
    exit_date = entry_date + pd.tseries.offsets.BDay(3)

    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), entry_date
    )
    assert signals.loc[entry_date, "AAPL"] == 0.5
    assert signals.loc[entry_date, "GOOG"] == 0.5

    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), entry_date + pd.tseries.offsets.BDay(1)
    )
    assert signals.loc[entry_date + pd.tseries.offsets.BDay(1), "AAPL"] == 0.5
    assert signals.loc[entry_date + pd.tseries.offsets.BDay(1), "GOOG"] == 0.5

    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), exit_date
    )
    assert signals.loc[exit_date, "AAPL"] == 0
    assert signals.loc[exit_date, "GOOG"] == 0
