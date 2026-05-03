import pandas as pd
import pytest

from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)

RORO = "MDMP:RORO.CARLOS"


@pytest.fixture
def sample_hist_two_tickers():
    dates = pd.bdate_range("2023-01-01", "2023-01-31")
    return pd.DataFrame({"AAPL": 100.0, "MSFT": 200.0}, index=dates)


def test_get_non_universe_empty_when_overlay_off() -> None:
    s = SeasonalSignalStrategy({"strategy_params": {}})
    assert s.get_non_universe_data_requirements() == []


def test_get_non_universe_returns_default_symbol_when_on() -> None:
    s = SeasonalSignalStrategy({"strategy_params": {"use_carlos_roro": True}})
    assert s.get_non_universe_data_requirements() == [RORO]


def test_get_non_universe_returns_custom_symbol() -> None:
    sym = "MDMP:RORO.CUSTOM"
    s = SeasonalSignalStrategy(
        {"strategy_params": {"use_carlos_roro": True, "carlos_roro_symbol": sym}}
    )
    assert s.get_non_universe_data_requirements() == [sym]


def test_generate_signals_risk_off_flattens_despite_seasonal_window(
    sample_hist_two_tickers,
) -> None:
    entry_day = 1
    d0 = pd.Timestamp("2023-01-02")
    assert d0 in sample_hist_two_tickers.index
    nu = pd.DataFrame({RORO: 1.0}, index=sample_hist_two_tickers.index)
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": entry_day,
                "hold_days": 5,
                "direction": "long",
                "month_local_seasonal_windows": True,
                "use_carlos_roro": True,
            }
        }
    )
    sig = strat.generate_signals(sample_hist_two_tickers, pd.DataFrame(), nu, current_date=d0)
    assert (sig == 0.0).all().all()


def test_generate_signals_risk_on_allows_entry(sample_hist_two_tickers) -> None:
    entry_day = 1
    d0 = pd.Timestamp("2023-01-02")
    nu = pd.DataFrame({RORO: 0.0}, index=sample_hist_two_tickers.index)
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": entry_day,
                "hold_days": 5,
                "direction": "long",
                "month_local_seasonal_windows": True,
                "use_carlos_roro": True,
            }
        }
    )
    sig = strat.generate_signals(sample_hist_two_tickers, pd.DataFrame(), nu, current_date=d0)
    assert sig.loc[d0, "AAPL"] == pytest.approx(0.5)
    assert sig.loc[d0, "MSFT"] == pytest.approx(0.5)


def test_generate_signal_matrix_matches_signals_on_overlay(sample_hist_two_tickers) -> None:
    nu = pd.DataFrame({RORO: 0.0}, index=sample_hist_two_tickers.index)
    nu.loc[pd.Timestamp("2023-01-04"), RORO] = 1.0
    idx = sample_hist_two_tickers.index
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 5,
                "direction": "long",
                "month_local_seasonal_windows": True,
                "use_carlos_roro": True,
            }
        }
    )
    mat = strat.generate_signal_matrix(
        pd.DataFrame(),
        pd.DataFrame(),
        nu,
        idx,
        ["AAPL", "MSFT"],
    )
    assert mat is not None
    d_block = pd.Timestamp("2023-01-04")
    assert mat.loc[d_block].abs().sum() == pytest.approx(0.0)
    d_other = pd.Timestamp("2023-01-03")
    row = mat.loc[d_other]
    assert row["AAPL"] == pytest.approx(0.5)
    assert row["MSFT"] == pytest.approx(0.5)
    gs = strat.generate_signals(sample_hist_two_tickers, pd.DataFrame(), nu, current_date=d_block)
    assert (gs.loc[d_block].values == mat.loc[d_block].values).all()


def test_overlay_uses_multiindex_close_when_present(sample_hist_two_tickers) -> None:
    tuples = [(RORO, "Close")]
    nu = pd.DataFrame(
        0.0,
        index=sample_hist_two_tickers.index,
        columns=pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"]),
    )
    nu.loc[pd.Timestamp("2023-01-03"), (RORO, "Close")] = 1.0
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 5,
                "direction": "long",
                "month_local_seasonal_windows": True,
                "use_carlos_roro": True,
            }
        }
    )
    mat = strat.generate_signal_matrix(
        pd.DataFrame(),
        pd.DataFrame(),
        nu,
        sample_hist_two_tickers.index,
        ["AAPL"],
    )
    assert mat is not None
    assert mat.loc[pd.Timestamp("2023-01-03"), "AAPL"] == pytest.approx(0.0)
    assert mat.loc[pd.Timestamp("2023-01-02"), "AAPL"] == pytest.approx(1.0)
