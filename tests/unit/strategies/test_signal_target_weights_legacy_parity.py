"""Full-scan ``generate_target_weights`` parity vs sequential legacy ``generate_signals`` rows."""

from __future__ import annotations

from unittest.mock import patch

from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.signal.donchian_asri_signal_strategy import (
    AsriThresholdSignalStrategy,
    DonchianAsriSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.ema_roro_signal_strategy import (
    EmaRoroSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy import (
    MmmQsSwingNasdaqSignalStrategy,
)
from portfolio_backtester.strategies.signal.momentum_strategy import MomentumStrategy
from portfolio_backtester.strategies.user.signal.momentum_signal_strategy import (
    MomentumSignalStrategy,
)


def _legacy_scan_rows(
    strategy: Any,
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    non_universe: pd.DataFrame | None,
    rebalance_dates: pd.DatetimeIndex,
    universe_tickers: list[str],
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for d in rebalance_dates:
        nu_view = pd.DataFrame()
        if non_universe is not None and not non_universe.empty:
            nu_view = non_universe.loc[:d]
        sig = strategy.generate_signals(
            asset.loc[:d],
            benchmark.loc[:d],
            nu_view,
            current_date=d,
        )
        if d not in sig.index:
            rows.append(pd.Series(0.0, index=universe_tickers, dtype=float))
        else:
            rows.append(sig.loc[d].reindex(universe_tickers).fillna(0.0))
    return pd.DataFrame(rows, index=rebalance_dates, columns=universe_tickers, dtype=float)


def _ema_like_close_panel(idx: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    base = np.linspace(100.0, 130.0, len(idx))
    data = {t: base + 0.5 * i for i, t in enumerate(tickers)}
    return pd.DataFrame(data, index=idx, dtype=float)


@pytest.mark.parametrize(
    "cls,cfg",
    [
        (
            EmaRoroSignalStrategy,
            {
                "strategy_params": {
                    "fast_ema_days": 5,
                    "slow_ema_days": 12,
                    "leverage": 1.0,
                    "risk_off_leverage_multiplier": 0.4,
                }
            },
        ),
        (MomentumSignalStrategy, {"strategy_params": {}}),
        (MomentumStrategy, {"strategy_params": {}}),
    ],
)
def test_generate_target_weights_exists(cls: type, cfg: dict) -> None:
    strat = cls(cfg)
    assert callable(getattr(strat, "generate_target_weights"))


@pytest.mark.parametrize(
    "cls,cfg",
    [
        (
            DonchianAsriSignalStrategy,
            {"strategy_params": {"use_asri_filter": False, "use_asri_sizing": False}},
        ),
        (
            AsriThresholdSignalStrategy,
            {"strategy_params": {"asri_threshold_quantile": 0.5}},
        ),
        (MmmQsSwingNasdaqSignalStrategy, {"strategy_params": {}}),
    ],
)
def test_generate_target_weights_callable_signal_strategies(cls: type, cfg: dict) -> None:
    strat = cls(cfg)
    assert callable(getattr(strat, "generate_target_weights"))


def test_ema_roro_target_weights_matches_legacy_scan() -> None:
    idx = pd.date_range("2024-03-01", periods=40, freq="B")
    universe = ["AAA", "BBB"]
    asset = _ema_like_close_panel(idx, universe)
    benchmark = asset[["AAA"]].copy()
    cfg = {
        "strategy_params": {
            "fast_ema_days": 8,
            "slow_ema_days": 21,
            "leverage": 1.0,
            "risk_off_leverage_multiplier": 0.35,
        }
    }
    strat_legacy = EmaRoroSignalStrategy(cfg)
    strat_tw = EmaRoroSignalStrategy(cfg)
    rebalance_dates = pd.DatetimeIndex(idx[15:])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset,
        benchmark_data=benchmark,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rebalance_dates,
        universe_tickers=universe,
        benchmark_ticker="AAA",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    legacy = _legacy_scan_rows(strat_legacy, asset, benchmark, None, rebalance_dates, universe)
    tw = strat_tw.generate_target_weights(ctx).fillna(0.0)
    pd.testing.assert_frame_equal(tw, legacy, rtol=0.0, atol=1e-12)


def test_mmm_qs_swing_target_weights_matches_legacy_scan_patched() -> None:
    idx = pd.date_range("2024-06-03", periods=55, freq="D")
    close = [100.0 + i * 0.05 for i in range(len(idx))]
    low = list(close)
    high = [c + 2.0 for c in close]
    low[-1] = close[-1] - 0.05
    high[-1] = close[-1] + 5.0
    columns = pd.MultiIndex.from_product(
        [["QQQ"], ["Open", "High", "Low", "Close"]],
        names=["Ticker", "Field"],
    )
    ohlc = pd.DataFrame(
        np.column_stack([close, high, low, close]),
        index=idx,
        columns=columns,
        dtype=float,
    )
    bench = pd.DataFrame(index=idx)
    strat_legacy = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    strat_tw = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    rebalance_dates = pd.DatetimeIndex(idx[25:])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=ohlc,
        benchmark_data=bench,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rebalance_dates,
        universe_tickers=["QQQ"],
        benchmark_ticker="QQQ",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    adx_const = pd.Series(35.0, index=idx)
    with (
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy._compute_adx_series",
            return_value=adx_const,
        ),
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy.calculate_atr_fast",
            return_value=pd.Series({"QQQ": 2.0}),
        ),
    ):
        legacy = _legacy_scan_rows(strat_legacy, ohlc, bench, None, rebalance_dates, ["QQQ"])
        tw = strat_tw.generate_target_weights(ctx).fillna(0.0)
    pd.testing.assert_frame_equal(tw, legacy, rtol=0.0, atol=1e-12)


def test_donchian_asri_target_weights_matches_legacy_scan_no_asri() -> None:
    idx = pd.date_range("2026-01-01", periods=45, freq="D")
    tickers = ["SPY", "IWM"]
    frames = []
    for t in tickers:
        c = pd.Series([100.0] * 22 + [108.0] * 13 + [94.0] * 10, index=idx, dtype=float)
        frames.append(c.rename((t, "Close")))
    asset = pd.concat(frames, axis=1)
    for t in tickers:
        asset[(t, "Open")] = asset[(t, "Close")]
        asset[(t, "High")] = asset[(t, "Close")] + 0.5
        asset[(t, "Low")] = asset[(t, "Close")] - 0.5
    asset = asset.sort_index(axis=1)
    benchmark = asset[[("SPY", "Close")]].copy()
    strat_legacy = DonchianAsriSignalStrategy(
        {"strategy_params": {"use_asri_filter": False, "use_asri_sizing": False}}
    )
    strat_tw = DonchianAsriSignalStrategy(
        {"strategy_params": {"use_asri_filter": False, "use_asri_sizing": False}}
    )
    rebalance_dates = pd.DatetimeIndex(idx[25:])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset,
        benchmark_data=benchmark,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rebalance_dates,
        universe_tickers=tickers,
        benchmark_ticker="SPY",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    legacy = _legacy_scan_rows(strat_legacy, asset, benchmark, None, rebalance_dates, tickers)
    tw = strat_tw.generate_target_weights(ctx).fillna(0.0)
    pd.testing.assert_frame_equal(tw, legacy, rtol=0.0, atol=1e-12)


def test_asri_threshold_target_weights_matches_legacy_scan() -> None:
    idx = pd.date_range("2026-02-01", periods=30, freq="D")
    universe = ["SPY", "QQQ"]
    asset_frames = []
    for t in universe:
        c = pd.Series(np.linspace(400.0, 410.0, len(idx)), index=idx, dtype=float)
        asset_frames.append(c.rename((t, "Close")))
    asset = pd.concat(asset_frames, axis=1)
    for t in universe:
        asset[(t, "Open")] = asset[(t, "Close")]
        asset[(t, "High")] = asset[(t, "Close")] + 0.2
        asset[(t, "Low")] = asset[(t, "Close")] - 0.2
    asset = asset.sort_index(axis=1)
    benchmark = asset[[(universe[0], "Close")]].copy()
    asri = pd.Series(np.linspace(50.0, 80.0, len(idx)), index=idx, dtype=float)
    nu = pd.DataFrame({"MDMP:ASRI": asri})
    strat_legacy = AsriThresholdSignalStrategy(
        {"strategy_params": {"asri_threshold_quantile": 0.5}}
    )
    strat_tw = AsriThresholdSignalStrategy({"strategy_params": {"asri_threshold_quantile": 0.5}})
    rebalance_dates = pd.DatetimeIndex(idx[10:])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset,
        benchmark_data=benchmark,
        non_universe_data=nu,
        rebalance_dates=rebalance_dates,
        universe_tickers=universe,
        benchmark_ticker=universe[0],
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    legacy = _legacy_scan_rows(strat_legacy, asset, benchmark, nu, rebalance_dates, universe)
    tw = strat_tw.generate_target_weights(ctx).fillna(0.0)
    pd.testing.assert_frame_equal(tw, legacy, rtol=0.0, atol=1e-12)


def test_momentum_stubs_return_aligned_zeros_or_nan_template() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    universe = ["AAPL", "GOOGL"]
    asset = pd.DataFrame(1.0, index=idx, columns=universe)
    bench = asset[["AAPL"]].copy()
    for sparse in (False, True):
        ctx = StrategyContext.from_standard_inputs(
            asset_data=asset,
            benchmark_data=bench,
            non_universe_data=pd.DataFrame(),
            rebalance_dates=idx,
            universe_tickers=universe,
            benchmark_ticker="AAPL",
            wfo_start_date=None,
            wfo_end_date=None,
            use_sparse_nan_for_inactive_rows=sparse,
        )
        ms = MomentumSignalStrategy({"strategy_params": {}})
        tw_ms = ms.generate_target_weights(ctx)
        assert list(tw_ms.columns) == universe
        assert tw_ms.shape == (len(idx), len(universe))
        if sparse:
            assert tw_ms.isna().all().all()
        else:
            assert (tw_ms == 0.0).all().all()

        mo = MomentumStrategy({"strategy_params": {}})
        tw_mo = mo.generate_target_weights(ctx)
        pd.testing.assert_frame_equal(tw_ms, tw_mo)
