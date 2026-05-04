"""Parity checks for full-scan ``generate_target_weights`` vs expanding ``generate_signals``."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from portfolio_backtester.strategies._core.target_generation import (
    StrategyContext,
    default_benchmark_ticker,
)

from portfolio_backtester.strategies.builtins.portfolio.fixed_weight_portfolio_strategy import (
    FixedWeightPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.signal.ema_crossover_signal_strategy import (
    EmaCrossoverSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.ema_roro_signal_strategy import (
    EmaRoroSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.uvxy_rsi_signal_strategy import (
    UvxyRsiSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)
from portfolio_backtester.strategies.user.signal.hello_world_signal_strategy import (
    HelloWorldSignalStrategy,
)


def _slice_expanding(asset_df: pd.DataFrame, rd: pd.Timestamp) -> pd.DataFrame:
    end = cast(int, asset_df.index.searchsorted(rd, side="right"))
    return asset_df.iloc[:end]


def _assert_matrix_parity(
    strategy,
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    non_uni: pd.DataFrame,
    universe_tickers: list[str],
    rebalance_dates: pd.DatetimeIndex,
    *,
    use_sparse_nan: bool = False,
) -> None:
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_df,
        benchmark_data=benchmark_df,
        non_universe_data=non_uni,
        rebalance_dates=rebalance_dates,
        universe_tickers=universe_tickers,
        benchmark_ticker=default_benchmark_ticker(benchmark_df, universe_tickers),
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=use_sparse_nan,
    )
    mat = strategy.generate_target_weights(ctx)
    assert mat is not None
    for rd in rebalance_dates:
        hist = _slice_expanding(asset_df, pd.Timestamp(rd))
        bh = _slice_expanding(benchmark_df, pd.Timestamp(rd))
        nu = _slice_expanding(non_uni, pd.Timestamp(rd)) if len(non_uni) else pd.DataFrame()
        row = strategy.generate_signals(
            hist,
            bh,
            non_universe_historical_data=nu if len(nu.columns) else None,
            current_date=pd.Timestamp(rd),
        ).iloc[0]
        got = mat.loc[pd.Timestamp(rd)]
        pd.testing.assert_series_equal(
            got.reindex(universe_tickers).fillna(0.0),
            row.reindex(universe_tickers).fillna(0.0),
            rtol=0.0,
            atol=1e-12,
        )


def test_ema_crossover_signal_matrix_matches_generate_signals() -> None:
    idx = pd.date_range("2020-01-02", periods=80, freq="B")
    universe_tickers = ["AAA", "BBB"]
    asset_df = pd.DataFrame(
        {
            "AAA": pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float),
            "BBB": pd.Series(range(200, 200 + len(idx)), index=idx, dtype=float) * 0.5,
        }
    )
    benchmark_df = asset_df[["AAA"]].copy()
    strat = EmaCrossoverSignalStrategy(
        {"strategy_params": {"fast_ema_days": 5, "slow_ema_days": 12, "leverage": 1.0}}
    )
    rds = pd.DatetimeIndex(idx[25:70:3])
    _assert_matrix_parity(
        strat,
        asset_df,
        benchmark_df,
        pd.DataFrame(),
        universe_tickers,
        rds,
    )


def test_ema_roro_signal_matrix_matches_generate_signals() -> None:
    idx = pd.date_range("2020-01-02", periods=80, freq="B")
    universe_tickers = ["AAA", "BBB"]
    asset_df = pd.DataFrame(
        {
            "AAA": pd.Series(range(300, 300 + len(idx)), index=idx, dtype=float),
            "BBB": pd.Series(range(50, 50 + len(idx)), index=idx, dtype=float),
        }
    )
    benchmark_df = asset_df[["AAA"]].copy()
    strat = EmaRoroSignalStrategy(
        {
            "strategy_params": {
                "fast_ema_days": 6,
                "slow_ema_days": 15,
                "leverage": 1.0,
                "risk_off_leverage_multiplier": 0.25,
            }
        }
    )
    rds = pd.DatetimeIndex(idx[30:75:2])
    _assert_matrix_parity(
        strat,
        asset_df,
        benchmark_df,
        pd.DataFrame(),
        universe_tickers,
        rds,
    )


def test_uvxy_rsi_signal_matrix_matches_generate_signals() -> None:
    idx = pd.date_range("2020-03-02", periods=60, freq="B")
    universe_tickers = ["UVXY", "TQQQ"]
    asset_df = pd.DataFrame(
        {"UVXY": 10.0, "TQQQ": 100.0},
        index=idx,
        dtype=float,
    )
    benchmark_df = asset_df[["UVXY"]].copy()
    spy = pd.Series(
        [float(i % 7) * 2.5 + 300.0 for i in range(len(idx))],
        index=idx,
        dtype=float,
    )
    non_uni = pd.DataFrame({"SPY": spy})
    strat = UvxyRsiSignalStrategy({"strategy_params": {"rsi_period": 2, "rsi_threshold": 88.0}})
    rds = pd.DatetimeIndex(idx[10:55:4])
    _assert_matrix_parity(
        strat,
        asset_df,
        benchmark_df,
        non_uni,
        universe_tickers,
        rds,
    )


def test_uvxy_rsi_without_spy_is_zeros() -> None:
    idx = pd.date_range("2020-01-02", periods=20, freq="B")
    universe_tickers = ["UVXY"]
    asset_df = pd.DataFrame({"UVXY": 1.0}, index=idx)
    strat = UvxyRsiSignalStrategy({})
    mat = strat.generate_signal_matrix(
        asset_df,
        asset_df,
        pd.DataFrame(),
        pd.DatetimeIndex(idx[5:15]),
        universe_tickers,
    )
    assert mat is not None
    assert (mat.fillna(0.0).to_numpy() == 0.0).all()


def test_fixed_weight_signal_matrix_matches_generate_signals() -> None:
    idx = pd.date_range("2021-01-04", periods=40, freq="B")
    universe_tickers = ["X", "Y", "Z"]
    asset_df = pd.DataFrame({t: float(i + 1) for i, t in enumerate(universe_tickers)}, index=idx)
    benchmark_df = asset_df[["X"]].copy()
    strat = FixedWeightPortfolioStrategy({})
    rds = pd.DatetimeIndex(idx[5:30:5])
    _assert_matrix_parity(
        strat,
        asset_df,
        benchmark_df,
        pd.DataFrame(),
        universe_tickers,
        rds,
    )


def test_hello_world_signal_matrix_matches_generate_signals() -> None:
    idx = pd.date_range("2022-05-02", periods=35, freq="B")
    universe_tickers = ["A", "B"]
    asset_df = pd.DataFrame(
        {
            "A": pd.Series(range(len(idx)), index=idx, dtype=float),
            "B": pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float),
        }
    )
    benchmark_df = asset_df[["A"]].copy()
    strat = HelloWorldSignalStrategy({"strategy_params": {"leverage": 1.25}})
    rds = pd.DatetimeIndex(idx[8:30:3])
    _assert_matrix_parity(
        strat,
        asset_df,
        benchmark_df,
        pd.DataFrame(),
        universe_tickers,
        rds,
    )


def test_seasonal_signal_matrix_sparse_inactive_rows_are_explicit_flat() -> None:
    idx = pd.date_range("2023-01-02", periods=15, freq="B")
    universe_tickers = ["AAPL", "GOOG"]
    asset_df = pd.DataFrame({"AAPL": 100.0, "GOOG": 200.0}, index=idx)
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 2,
                "trade_month_1": False,
                "month_local_seasonal_windows": True,
            }
        }
    )

    mat = strat.generate_signal_matrix(
        asset_df,
        asset_df[["AAPL"]],
        pd.DataFrame(),
        pd.DatetimeIndex(idx[:5]),
        universe_tickers,
        use_sparse_nan_for_inactive_rows=True,
    )

    assert mat is not None
    assert not mat.isna().any().any()
    assert (mat == 0.0).all().all()


def test_seasonal_per_month_maps_signal_matrix_matches_generate_signals() -> None:
    idx = pd.date_range("2023-01-02", periods=45, freq="B")
    universe_tickers = ["AAA", "BBB"]
    asset_df = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=idx)
    benchmark_df = asset_df[["AAA"]].copy()
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 5,
                "hold_days_by_month": {1: 12, "February": 5},
                "entry_day_by_month": {"feb": 10},
                "month_local_seasonal_windows": False,
            }
        }
    )
    rds = pd.DatetimeIndex(idx[8:38:2])
    _assert_matrix_parity(
        strat,
        asset_df,
        benchmark_df,
        pd.DataFrame(),
        universe_tickers,
        rds,
    )


@pytest.mark.parametrize("use_sparse", [False, True])
def test_ema_crossover_sparse_nan_flag_builds_dataframe(use_sparse: bool) -> None:
    idx = pd.date_range("2020-01-02", periods=40, freq="B")
    universe_tickers = ["AAA"]
    asset_df = pd.DataFrame({"AAA": 1.0}, index=idx)
    strat = EmaCrossoverSignalStrategy({})
    mat = strat.generate_signal_matrix(
        asset_df,
        asset_df,
        pd.DataFrame(),
        pd.DatetimeIndex(idx[10:15]),
        universe_tickers,
        use_sparse_nan_for_inactive_rows=use_sparse,
    )
    assert mat is not None
    assert mat.shape == (5, 1)
