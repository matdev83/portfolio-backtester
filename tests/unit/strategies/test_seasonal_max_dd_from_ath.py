import pandas as pd
import pytest

from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)


def test_max_dd_from_ath_pct_negative_raises() -> None:
    with pytest.raises(ValueError, match="max_dd_from_ath_pct"):
        SeasonalSignalStrategy({"strategy_params": {"max_dd_from_ath_pct": -0.01}})


def test_tunable_parameters_include_max_dd_from_ath() -> None:
    t = SeasonalSignalStrategy.tunable_parameters()
    assert "max_dd_from_ath_pct" in t
    meta = t["max_dd_from_ath_pct"]
    assert meta["type"] == "float"
    assert meta["default"] == 0.0
    assert meta["min"] == 0.0
    assert meta["max"] == 15.0


def test_dd_below_threshold_allows_long_signals() -> None:
    dates = pd.bdate_range("2023-01-01", periods=25)
    close = pd.Series(dtype=float)
    close.loc[dates[0]] = 8000.0
    for i in range(1, len(dates)):
        close.loc[dates[i]] = 7900.0
    hist = pd.DataFrame({"SPY": close}).reindex(dates).ffill()
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 80,
                "month_local_seasonal_windows": True,
                "direction": "long",
                "max_dd_from_ath_pct": 10.0,
            }
        }
    )
    assert strat._resolve_active_entry_date(pd.Timestamp(dates[15])) is not None
    d = dates[15]
    sig = strat.generate_signals(hist, pd.DataFrame(), None, current_date=d)
    assert sig.loc[d, "SPY"] == pytest.approx(1.0)


def test_dd_above_threshold_blocks_long_signals() -> None:
    dates = pd.bdate_range("2023-01-01", periods=25)
    vals = [8000.0] + [7900.0] * 5 + [6000.0] * (len(dates) - 6)
    hist = pd.DataFrame({"SPY": vals}, index=dates)
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 80,
                "month_local_seasonal_windows": True,
                "direction": "long",
                "max_dd_from_ath_pct": 10.0,
            }
        }
    )
    d = dates[15]
    sig = strat.generate_signals(hist, pd.DataFrame(), None, current_date=d)
    assert sig.loc[d, "SPY"] == pytest.approx(0.0)


def test_max_dd_filter_skipped_for_short_direction() -> None:
    dates = pd.bdate_range("2023-01-01", periods=25)
    vals = [8000.0] + [5000.0] * (len(dates) - 1)
    hist = pd.DataFrame({"SPY": vals}, index=dates)
    d = dates[10]
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 80,
                "month_local_seasonal_windows": True,
                "direction": "short",
                "max_dd_from_ath_pct": 10.0,
            }
        }
    )
    sig = strat.generate_signals(hist, pd.DataFrame(), None, current_date=d)
    assert sig.loc[d, "SPY"] == pytest.approx(-1.0)


def test_two_tickers_partial_block_rescales_long_weights() -> None:
    dates = pd.bdate_range("2023-01-01", periods=30)
    ok = pd.Series([100.0] * len(dates), index=dates)
    bad = pd.Series([100.0] * 10 + [50.0] * (len(dates) - 10), index=dates)
    hist = pd.DataFrame({"OK": ok, "BAD": bad})
    d = dates[20]
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 80,
                "month_local_seasonal_windows": True,
                "direction": "long",
                "max_dd_from_ath_pct": 35.0,
            }
        }
    )
    sig = strat.generate_signals(hist, pd.DataFrame(), None, current_date=d)
    assert sig.loc[d, "OK"] == pytest.approx(1.0)
    assert sig.loc[d, "BAD"] == pytest.approx(0.0)


def test_generate_signal_matrix_parity_max_dd_long() -> None:
    idx = pd.bdate_range("2023-01-02", periods=35)
    low_dd = pd.Series([8000.0] + [7900.0] * (len(idx) - 1), index=idx)
    high_dd = pd.Series([8000.0] + [7900.0] * 5 + [6000.0] * (len(idx) - 6), index=idx)
    asset_df = pd.DataFrame({"A": low_dd, "B": high_dd})
    benchmark_df = asset_df[["A"]].copy()
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 80,
                "month_local_seasonal_windows": True,
                "direction": "long",
                "max_dd_from_ath_pct": 10.0,
            }
        }
    )
    rds = pd.DatetimeIndex(idx[10:30:5])
    mat = strat.generate_signal_matrix(
        asset_df,
        benchmark_df,
        None,
        rds,
        ["A", "B"],
    )
    assert mat is not None
    for rd in rds:
        h = asset_df.loc[:rd]
        b = benchmark_df.loc[:rd]
        row_gs = strat.generate_signals(h, b, None, current_date=pd.Timestamp(rd)).iloc[0]
        pd.testing.assert_series_equal(
            mat.loc[rd].reindex(["A", "B"]).fillna(0.0),
            row_gs.reindex(["A", "B"]).fillna(0.0),
            rtol=0.0,
            atol=1e-12,
        )
