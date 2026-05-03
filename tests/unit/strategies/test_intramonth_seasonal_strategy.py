import logging

import pandas as pd
import pytest

from portfolio_backtester.risk_management.atr_service import calculate_atr_fast
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)


@pytest.fixture
def sample_historical_data():
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", end="2023-03-31", freq="B"))
    data = {"AAPL": 100, "GOOG": 200}
    df = pd.DataFrame(index=dates, columns=list(data.keys()))
    for col in df.columns:
        df[col] = data[col]
    return df


def test_generate_signals_accepts_tz_aware_current_date(sample_historical_data):
    strategy = SeasonalSignalStrategy({"strategy_params": {"entry_day": 1, "hold_days": 5}})
    aware = pd.Timestamp("2023-01-03").tz_localize("US/Eastern")
    out = strategy.generate_signals(sample_historical_data, pd.DataFrame(), pd.DataFrame(), aware)
    assert out.index[0] == aware
    assert not out.isna().any().any()


def test_generate_signal_matrix_handles_tz_aware_rebalance_index() -> None:
    """Regression: naive bdate_range entry/window vs tz-aware scan index must not raise."""
    tz = "America/New_York"
    naive_idx = pd.bdate_range("2024-11-01", periods=40)
    aware_idx = naive_idx.tz_localize(tz)
    strategy = SeasonalSignalStrategy({"strategy_params": {"entry_day": 2, "hold_days": 4}})
    mat = strategy.generate_signal_matrix(
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        aware_idx,
        ["SPY"],
    )
    assert mat is not None
    assert len(mat) == len(aware_idx)
    assert mat.index.equals(aware_idx)


def test_tunable_parameters_include_calendar_filters_and_direction():
    tunables = SeasonalSignalStrategy.tunable_parameters()
    assert "month_local_seasonal_windows" in tunables
    assert "stop_loss_atr_multiple" in tunables
    assert "take_profit_atr_multiple" in tunables
    assert "simple_high_low_stop_loss" in tunables
    assert "simple_high_low_take_profit" in tunables
    assert "entry_day" in tunables
    assert "hold_days" in tunables
    assert tunables["hold_days"]["min"] == 5
    assert "direction" in tunables
    assert tunables["direction"]["type"] == "categorical"
    assert set(tunables["direction"]["values"]) == {"long", "short"}
    for m in range(1, 13):
        key = f"trade_month_{m}"
        assert key in tunables
        assert tunables[key]["type"] == "categorical"


def test_get_entry_date_for_month_positive():
    strategy = SeasonalSignalStrategy({})
    date = pd.Timestamp("2023-01-15")
    entry_day = 5
    expected_date = pd.Timestamp("2023-01-06")  # 5th business day of Jan 2023
    assert strategy.get_entry_date_for_month(date, entry_day) == expected_date


def test_get_entry_date_for_month_negative():
    strategy = SeasonalSignalStrategy({})
    date = pd.Timestamp("2023-01-15")
    entry_day = -3
    expected_date = pd.Timestamp("2023-01-27")  # 3rd to last business day of Jan 2023
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

    # On entry date
    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), entry_date
    )
    assert signals.loc[entry_date, "AAPL"] == 0.5
    assert signals.loc[entry_date, "GOOG"] == 0.5

    # Day after entry
    signals = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        entry_date + pd.tseries.offsets.BDay(1),
    )
    assert signals.loc[entry_date + pd.tseries.offsets.BDay(1), "AAPL"] == 0.5
    assert signals.loc[entry_date + pd.tseries.offsets.BDay(1), "GOOG"] == 0.5

    # On exit date
    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), exit_date
    )
    assert signals.loc[exit_date, "AAPL"] == 0
    assert signals.loc[exit_date, "GOOG"] == 0


def test_month_local_anchor_does_not_extend_prior_month_hold(sample_historical_data):
    """Legacy month-local: February uses February's entry anchor; early Feb is flat."""
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": -1,
            "hold_days": 10,
            "month_local_seasonal_windows": True,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    jan_last = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-15"), -1)
    assert jan_last.month == 1

    sig_jan31 = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Timestamp("2023-01-31"),
    )
    assert sig_jan31.loc["2023-01-31", "AAPL"] == pytest.approx(0.5)

    sig_feb1 = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Timestamp("2023-02-01"),
    )
    assert (sig_feb1 == 0.0).all().all()


def test_trade_month_filter_disables_december(sample_historical_data):
    config = {
        "strategy_params": {
            "entry_day": 1,
            "hold_days": 5,
            "trade_month_12": False,
            "month_local_seasonal_windows": True,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    dec_date = pd.Timestamp("2023-12-04")
    sig = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), dec_date
    )
    assert (sig == 0.0).all().all()


def test_cross_month_default_holds_january_window_into_february(sample_historical_data):
    """Default: Jan month-end entry + long hold can stay long on early Feb sessions."""
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": -1,
            "hold_days": 10,
            "month_local_seasonal_windows": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    sig_feb1 = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Timestamp("2023-02-01"),
    )
    assert sig_feb1.loc["2023-02-01", "AAPL"] == pytest.approx(0.5)


def test_cross_month_trade_month_gates_entry_month_not_tail_month(sample_historical_data):
    """Tail in February still counts as January trade if January is disabled."""
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": -1,
            "hold_days": 15,
            "trade_month_1": False,
            "trade_month_2": True,
            "month_local_seasonal_windows": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    sig_feb5 = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Timestamp("2023-02-06"),
    )
    assert (sig_feb5 == 0.0).all().all()


def test_per_month_hold_extends_january_into_february(sample_historical_data):
    """January uses a longer mapped hold; February default is shorter — tail still active."""
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 3,
            "hold_days_by_month": {1: 12},
            "month_local_seasonal_windows": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    jan_entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 1)
    window_end = jan_entry + pd.tseries.offsets.BDay(12 - 1)
    sig_mid = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Timestamp("2023-02-03"),
    )
    assert sig_mid.loc["2023-02-03", "AAPL"] == pytest.approx(0.5)
    sig_after = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        window_end + pd.tseries.offsets.BDay(1),
    )
    assert sig_after.loc[sig_after.index[0], "AAPL"] == pytest.approx(0.0)


def test_per_month_entry_day_for_february(sample_historical_data):
    """February uses mapped entry_day while January falls back to strategy default."""
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 5,
            "entry_day_by_month": {"feb": 10},
            "month_local_seasonal_windows": True,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    jan_entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 1)
    assert jan_entry.month == 1
    feb_entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-02-01"), 10)
    assert feb_entry.month == 2

    sig_jan = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), jan_entry
    )
    assert sig_jan.loc[jan_entry, "AAPL"] == pytest.approx(0.5)

    sig_before_feb = strategy.generate_signals(
        sample_historical_data,
        pd.DataFrame(),
        pd.DataFrame(),
        feb_entry - pd.tseries.offsets.BDay(1),
    )
    assert (sig_before_feb == 0.0).all().all()

    sig_feb = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), feb_entry
    )
    assert sig_feb.loc[feb_entry, "AAPL"] == pytest.approx(0.5)


def test_invalid_month_key_in_entry_day_map_raises():
    with pytest.raises(ValueError, match="entry_day_by_month"):
        SeasonalSignalStrategy(
            {"strategy_params": {"entry_day_by_month": {"invalid": 1}, "entry_day": 1}}
        )


def test_short_strategy_entry_and_exit(sample_historical_data):
    config = {
        "strategy_params": {
            "direction": "short",
            "entry_day": -1,
            "hold_days": 2,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(pd.Timestamp("2023-02-01"), -1)
    exit_date = entry_date + pd.tseries.offsets.BDay(2)

    # On entry date
    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), entry_date
    )
    assert signals.loc[entry_date, "AAPL"] == -0.5
    assert signals.loc[entry_date, "GOOG"] == -0.5

    # On exit date
    signals = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), exit_date
    )
    assert signals.loc[exit_date, "AAPL"] == 0
    assert signals.loc[exit_date, "GOOG"] == 0


def _ohlc_multiindex(
    dates: pd.DatetimeIndex, rows: dict[str, dict[str, list[float]]]
) -> pd.DataFrame:
    """rows[ticker][field] = list aligned with dates for Open, High, Low, Close."""
    pieces = []
    tickers = sorted(rows.keys())
    for t in tickers:
        for field in ("Open", "High", "Low", "Close"):
            series = pd.Series(rows[t][field], index=dates, name=(t, field))
            pieces.append(series)
    wide = pd.concat(pieces, axis=1)
    wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=["Ticker", "Field"])
    return wide


def test_long_simple_stop_loss_renormalizes_and_locks_cycle():
    dates = pd.to_datetime(pd.bdate_range("2023-01-02", periods=18))
    n = len(dates)
    day_low_trigger = 98.0  # prior bar low used when SL fires on next session
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": [100.0, 100.0, 102.0, 103.0, 98.0, 99.0] + [100.0] * (n - 6),
                "High": [101.0, 101.0, 105.0, 104.0, 104.0, 104.0] + [104.0] * (n - 6),
                "Low": [99.0, 99.0, day_low_trigger, 97.5, 97.5, 97.5] + [97.5] * (n - 6),
                "Close": [100.0, 100.0, 102.0, 97.0, 98.0, 99.0] + [100.0] * (n - 6),
            },
            "GOOG": {
                "Open": [200.0] * n,
                "High": [201.0] * n,
                "Low": [199.0] * n,
                "Close": [200.0] * n,
            },
        },
    )

    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 10,
            "month_local_seasonal_windows": True,
            "simple_high_low_stop_loss": True,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 1)
    assert entry == dates[0]

    # SL triggers on dates[3]: close vs prior bar low (dates[2] low == 98).
    d_before_sl = dates[2]
    sig_before = strategy.generate_signals(df, pd.DataFrame(), pd.DataFrame(), d_before_sl)
    assert sig_before.loc[d_before_sl, "AAPL"] == pytest.approx(0.5)
    assert sig_before.loc[d_before_sl, "GOOG"] == pytest.approx(0.5)

    d_sl = dates[3]
    sig_sl = strategy.generate_signals(df, pd.DataFrame(), pd.DataFrame(), d_sl)
    assert sig_sl.loc[d_sl, "AAPL"] == pytest.approx(0.0)
    assert sig_sl.loc[d_sl, "GOOG"] == pytest.approx(1.0)

    d_after = dates[4]
    sig_after = strategy.generate_signals(df.loc[:d_after], pd.DataFrame(), pd.DataFrame(), d_after)
    assert sig_after.loc[d_after, "AAPL"] == pytest.approx(0.0)
    assert sig_after.loc[d_after, "GOOG"] == pytest.approx(1.0)


def test_long_simple_take_profit_exits_one_ticker():
    dates = pd.to_datetime(pd.bdate_range("2023-01-02", periods=5))
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": [100.0, 100.0, 100.0, 100.0, 111.0],
                "High": [101.0, 101.0, 101.0, 110.0, 112.0],
                "Low": [99.0, 99.0, 99.0, 99.0, 110.0],
                "Close": [100.0, 100.0, 100.0, 105.0, 111.0],
            },
            "GOOG": {
                "Open": [200.0] * 5,
                "High": [201.0] * 5,
                "Low": [199.0] * 5,
                "Close": [200.0] * 5,
            },
        },
    )

    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 10,
            "month_local_seasonal_windows": True,
            "simple_high_low_stop_loss": False,
            "simple_high_low_take_profit": True,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    d_tp = dates[4]
    sig = strategy.generate_signals(df.loc[:d_tp], pd.DataFrame(), pd.DataFrame(), d_tp)
    assert sig.loc[d_tp, "AAPL"] == pytest.approx(0.0)
    assert sig.loc[d_tp, "GOOG"] == pytest.approx(1.0)


def test_short_simple_stop_loss():
    dates = pd.to_datetime(pd.bdate_range("2023-01-02", periods=5))
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": [100.0, 100.0, 100.0, 100.0, 106.0],
                "High": [101.0, 101.0, 101.0, 105.0, 107.0],
                "Low": [99.0, 99.0, 99.0, 99.0, 104.0],
                "Close": [100.0, 100.0, 100.0, 102.0, 106.0],
            },
            "GOOG": {
                "Open": [200.0] * 5,
                "High": [201.0] * 5,
                "Low": [199.0] * 5,
                "Close": [200.0] * 5,
            },
        },
    )

    config = {
        "strategy_params": {
            "direction": "short",
            "entry_day": 1,
            "hold_days": 10,
            "month_local_seasonal_windows": True,
            "simple_high_low_stop_loss": True,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    d_sl = dates[4]
    sig = strategy.generate_signals(df.loc[:d_sl], pd.DataFrame(), pd.DataFrame(), d_sl)
    assert sig.loc[d_sl, "AAPL"] == pytest.approx(0.0)
    assert sig.loc[d_sl, "GOOG"] == pytest.approx(-1.0)


def test_flat_panel_sl_tp_enabled_logs_once_and_falls_back(sample_historical_data, caplog):
    caplog.set_level(logging.WARNING)
    config = {
        "strategy_params": {
            "entry_day": 5,
            "hold_days": 3,
            "simple_high_low_stop_loss": True,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 5)
    sig = strategy.generate_signals(
        sample_historical_data, pd.DataFrame(), pd.DataFrame(), entry_date
    )
    assert sig.loc[entry_date, "AAPL"] == pytest.approx(0.5)
    assert "High/Low/Close MultiIndex" in caplog.text or "exit logic disabled" in caplog.text


def test_multiindex_sl_tp_disabled_matches_flat_equal_weight():
    dates = pd.to_datetime(pd.date_range(start="2023-01-03", end="2023-01-31", freq="B"))
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": [100.0] * len(dates),
                "High": [101.0] * len(dates),
                "Low": [99.0] * len(dates),
                "Close": [100.0] * len(dates),
            },
            "GOOG": {
                "Open": [200.0] * len(dates),
                "High": [201.0] * len(dates),
                "Low": [199.0] * len(dates),
                "Close": [200.0] * len(dates),
            },
        },
    )
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 5,
            "hold_days": 3,
            "simple_high_low_stop_loss": False,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    entry_date = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 5)
    sig = strategy.generate_signals(df, pd.DataFrame(), pd.DataFrame(), entry_date)
    assert sig.loc[entry_date, "AAPL"] == pytest.approx(0.5)
    assert sig.loc[entry_date, "GOOG"] == pytest.approx(0.5)


def test_negative_stop_loss_atr_multiple_raises():
    with pytest.raises(ValueError, match="stop_loss_atr_multiple"):
        SeasonalSignalStrategy({"strategy_params": {"stop_loss_atr_multiple": -1.0}})


def test_negative_take_profit_atr_multiple_raises():
    with pytest.raises(ValueError, match="take_profit_atr_multiple"):
        SeasonalSignalStrategy({"strategy_params": {"take_profit_atr_multiple": -0.01}})


def test_long_atr_stop_loss_exits_when_close_below_level():
    """Long ATR SL: exit when close <= entry_close - sl_mult * ATR(21) at entry."""
    dates = pd.to_datetime(pd.bdate_range("2022-10-03", periods=70))
    n = len(dates)
    base_a = 100.0
    # Mild volatility so ATR at entry is positive and stable
    closes_a = [base_a + (i % 5) * 0.4 for i in range(n)]
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": closes_a,
                "High": [c + 0.5 for c in closes_a],
                "Low": [c - 0.5 for c in closes_a],
                "Close": closes_a,
            },
            "GOOG": {
                "Open": [200.0] * n,
                "High": [201.0] * n,
                "Low": [199.0] * n,
                "Close": [200.0] * n,
            },
        },
    )
    strat_plain = SeasonalSignalStrategy(
        {"strategy_params": {"entry_day": 1, "hold_days": 15, "month_local_seasonal_windows": True}}
    )
    entry = strat_plain.get_entry_date_for_month(pd.Timestamp("2023-01-15"), 1)
    assert entry in df.index
    atr_entry = float(calculate_atr_fast(df, entry, 21)["AAPL"])
    assert atr_entry > 0 and atr_entry == atr_entry
    entry_px = float(df.xs("Close", level="Field", axis=1).loc[entry, "AAPL"])
    sl_mult = 1.5
    sl_price = entry_px - sl_mult * atr_entry

    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 15,
            "month_local_seasonal_windows": True,
            "stop_loss_atr_multiple": sl_mult,
            "take_profit_atr_multiple": 0.0,
            "simple_high_low_stop_loss": False,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)

    ei = dates.get_loc(entry)
    hit_ix = ei + 3
    assert hit_ix < n
    d_hit = dates[hit_ix]
    close_panel = df.xs("Close", level="Field", axis=1).copy()
    close_panel.loc[d_hit, "AAPL"] = sl_price - 1.0
    df2 = df.copy()
    for fld in ("Open", "High", "Low", "Close"):
        df2.loc[d_hit, ("AAPL", fld)] = close_panel.loc[d_hit, "AAPL"]

    sig = strategy.generate_signals(df2, pd.DataFrame(), pd.DataFrame(), d_hit)
    assert sig.loc[d_hit, "AAPL"] == pytest.approx(0.0)
    assert sig.loc[d_hit, "GOOG"] == pytest.approx(1.0)


def test_long_atr_take_profit_exits_when_close_above_level():
    dates = pd.to_datetime(pd.bdate_range("2022-10-03", periods=70))
    n = len(dates)
    closes_a = [100.0 + (i % 7) * 0.3 for i in range(n)]
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": closes_a,
                "High": [c + 0.6 for c in closes_a],
                "Low": [c - 0.6 for c in closes_a],
                "Close": closes_a,
            },
            "GOOG": {
                "Open": [200.0] * n,
                "High": [201.0] * n,
                "Low": [199.0] * n,
                "Close": [200.0] * n,
            },
        },
    )
    strat_plain = SeasonalSignalStrategy(
        {"strategy_params": {"entry_day": 1, "hold_days": 15, "month_local_seasonal_windows": True}}
    )
    entry = strat_plain.get_entry_date_for_month(pd.Timestamp("2023-01-15"), 1)
    atr_entry = float(calculate_atr_fast(df, entry, 21)["AAPL"])
    entry_px = float(df.xs("Close", level="Field", axis=1).loc[entry, "AAPL"])
    tp_mult = 3.2
    tp_price = entry_px + tp_mult * atr_entry

    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 15,
            "month_local_seasonal_windows": True,
            "stop_loss_atr_multiple": 0.0,
            "take_profit_atr_multiple": tp_mult,
            "simple_high_low_stop_loss": False,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)

    ei = dates.get_loc(entry)
    d_hit = dates[ei + 4]
    close_panel = df.xs("Close", level="Field", axis=1).copy()
    close_panel.loc[d_hit, "AAPL"] = tp_price + 0.5
    df2 = df.copy()
    for fld in ("Open", "High", "Low", "Close"):
        df2.loc[d_hit, ("AAPL", fld)] = close_panel.loc[d_hit, "AAPL"]

    sig = strategy.generate_signals(df2, pd.DataFrame(), pd.DataFrame(), d_hit)
    assert sig.loc[d_hit, "AAPL"] == pytest.approx(0.0)
    assert sig.loc[d_hit, "GOOG"] == pytest.approx(1.0)


def test_short_atr_stop_loss_exits_when_close_above_level():
    dates = pd.to_datetime(pd.bdate_range("2022-10-03", periods=70))
    n = len(dates)
    closes_a = [100.0 + (i % 5) * 0.35 for i in range(n)]
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": closes_a,
                "High": [c + 0.55 for c in closes_a],
                "Low": [c - 0.55 for c in closes_a],
                "Close": closes_a,
            },
            "GOOG": {
                "Open": [200.0] * n,
                "High": [201.0] * n,
                "Low": [199.0] * n,
                "Close": [200.0] * n,
            },
        },
    )
    strat_plain = SeasonalSignalStrategy(
        {"strategy_params": {"entry_day": 1, "hold_days": 15, "month_local_seasonal_windows": True}}
    )
    entry = strat_plain.get_entry_date_for_month(pd.Timestamp("2023-01-15"), 1)
    atr_entry = float(calculate_atr_fast(df, entry, 21)["AAPL"])
    entry_px = float(df.xs("Close", level="Field", axis=1).loc[entry, "AAPL"])
    sl_mult = 2.0
    sl_price = entry_px + sl_mult * atr_entry

    config = {
        "strategy_params": {
            "direction": "short",
            "entry_day": 1,
            "hold_days": 15,
            "month_local_seasonal_windows": True,
            "stop_loss_atr_multiple": sl_mult,
            "take_profit_atr_multiple": 0.0,
            "simple_high_low_stop_loss": False,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)

    ei = dates.get_loc(entry)
    d_hit = dates[ei + 3]
    close_panel = df.xs("Close", level="Field", axis=1).copy()
    close_panel.loc[d_hit, "AAPL"] = sl_price + 1.0
    df2 = df.copy()
    for fld in ("Open", "High", "Low", "Close"):
        df2.loc[d_hit, ("AAPL", fld)] = close_panel.loc[d_hit, "AAPL"]

    sig = strategy.generate_signals(df2, pd.DataFrame(), pd.DataFrame(), d_hit)
    assert sig.loc[d_hit, "AAPL"] == pytest.approx(0.0)
    assert sig.loc[d_hit, "GOOG"] == pytest.approx(-1.0)


def test_atr_multiples_zero_behaves_like_no_exit_rules():
    dates = pd.to_datetime(pd.bdate_range("2023-01-02", periods=12))
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": [100.0] * 12,
                "High": [101.0] * 12,
                "Low": [99.0] * 12,
                "Close": [100.0] * 12,
            },
            "GOOG": {
                "Open": [200.0] * 12,
                "High": [201.0] * 12,
                "Low": [199.0] * 12,
                "Close": [200.0] * 12,
            },
        },
    )
    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 10,
            "month_local_seasonal_windows": True,
            "stop_loss_atr_multiple": 0.0,
            "take_profit_atr_multiple": 0.0,
            "simple_high_low_stop_loss": False,
            "simple_high_low_take_profit": False,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    entry = strategy.get_entry_date_for_month(pd.Timestamp("2023-01-01"), 1)
    d_window = entry + pd.tseries.offsets.BDay(2)
    sig = strategy.generate_signals(df, pd.DataFrame(), pd.DataFrame(), d_window)
    assert sig.loc[d_window, "AAPL"] == pytest.approx(0.5)


def test_same_day_simple_sl_checked_before_atr_tp_long():
    """When both could fire, simple SL (first in order) locks before ATR TP."""
    dates = pd.to_datetime(pd.bdate_range("2022-10-03", periods=70))
    n = len(dates)
    closes_a = [100.0 + (i % 7) * 0.25 for i in range(n)]
    df = _ohlc_multiindex(
        dates,
        {
            "AAPL": {
                "Open": closes_a,
                "High": [c + 0.5 for c in closes_a],
                "Low": [c - 0.5 for c in closes_a],
                "Close": closes_a,
            },
            "GOOG": {
                "Open": [200.0] * n,
                "High": [201.0] * n,
                "Low": [199.0] * n,
                "Close": [200.0] * n,
            },
        },
    )
    strat_plain = SeasonalSignalStrategy(
        {"strategy_params": {"entry_day": 1, "hold_days": 15, "month_local_seasonal_windows": True}}
    )
    entry = strat_plain.get_entry_date_for_month(pd.Timestamp("2023-01-15"), 1)
    atr_entry = float(calculate_atr_fast(df, entry, 21)["AAPL"])
    entry_px = float(df.xs("Close", level="Field", axis=1).loc[entry, "AAPL"])
    tp_mult = 3.0
    tp_price = entry_px + tp_mult * atr_entry

    ei = dates.get_loc(entry)
    if isinstance(ei, slice):
        ei = ei.start
    ei = int(ei)
    hit_ix = min(ei + 5, n - 1)
    d_hit = dates[hit_ix]
    prev_day = dates[hit_ix - 1]
    df2 = df.copy()
    prev_low = tp_price + 10.0
    c_hit = tp_price + 3.0
    assert c_hit > tp_price
    assert c_hit < prev_low
    df2.loc[prev_day, ("AAPL", "Low")] = prev_low
    df2.loc[prev_day, ("AAPL", "High")] = prev_low + 2.0
    df2.loc[prev_day, ("AAPL", "Close")] = prev_low + 0.5
    df2.loc[prev_day, ("AAPL", "Open")] = prev_low + 0.5
    for fld in ("Open", "High", "Low", "Close"):
        df2.loc[d_hit, ("AAPL", fld)] = c_hit

    config = {
        "strategy_params": {
            "direction": "long",
            "entry_day": 1,
            "hold_days": 15,
            "month_local_seasonal_windows": True,
            "simple_high_low_stop_loss": True,
            "simple_high_low_take_profit": False,
            "stop_loss_atr_multiple": 0.0,
            "take_profit_atr_multiple": tp_mult,
        }
    }
    strategy = SeasonalSignalStrategy(config)
    sig = strategy.generate_signals(df2, pd.DataFrame(), pd.DataFrame(), d_hit)
    assert sig.loc[d_hit, "AAPL"] == pytest.approx(0.0)
    assert sig.loc[d_hit, "GOOG"] == pytest.approx(1.0)
