from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, cast

import inspect
import numpy as np
import pandas as pd
import pytest
from portfolio_backtester.backtester_logic.strategy_logic import (
    _expanding_iloc_ends,
    generate_signals,
)
from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.portfolio.calmar_momentum_portfolio_strategy import (
    CalmarMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy import (
    DualMomentumLaggedPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.filtered_lagged_momentum_portfolio_strategy import (
    FilteredLaggedMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.low_volatility_factor_portfolio_strategy import (
    LowVolatilityFactorPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.momentum_dvol_sizer_portfolio_strategy import (
    MomentumDvolSizerPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.momentum_unfiltered_atr_portfolio_strategy import (
    MomentumUnfilteredAtrPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.sharpe_momentum_portfolio_strategy import (
    SharpeMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.simple_momentum_portfolio_strategy import (
    SimpleMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.sortino_momentum_portfolio_strategy import (
    SortinoMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.vams_momentum_portfolio_strategy import (
    VamsMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.vams_no_downside_portfolio_strategy import (
    VamsNoDownsidePortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.volatility_targeted_fixed_weight_portfolio_strategy import (
    VolatilityTargetedFixedWeightPortfolioStrategy,
)
from portfolio_backtester.strategies.portfolio.beta_filtered_momentum_portfolio_strategy import (
    MomentumBetaFilteredPortfolioStrategy,
)

MOMENTUM_PORTFOLIO_CONCRETES: list[tuple[Type[Any], dict[str, Any]]] = [
    (
        SimpleMomentumPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        DualMomentumLaggedPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 6,
                "lag_months": 1,
                "max_holdings": 2,
                "use_200sma_filter": False,
                "min_absolute_momentum": 0.0,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        CalmarMomentumPortfolioStrategy,
        {
            "strategy_params": {
                "rolling_window": 3,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        SortinoMomentumPortfolioStrategy,
        {
            "strategy_params": {
                "rolling_window": 3,
                "target_return": 0.0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        SharpeMomentumPortfolioStrategy,
        {
            "strategy_params": {
                "rolling_window": 3,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": 0,
            },
        },
    ),
    (
        VolatilityTargetedFixedWeightPortfolioStrategy,
        {
            "strategy_params": {
                "target_vol_annual": 1.0,
                "vol_lookback_days": 21,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        VamsMomentumPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        VamsNoDownsidePortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        MomentumUnfilteredAtrPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        MomentumDvolSizerPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "vol_lookback_days": 21,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        LowVolatilityFactorPortfolioStrategy,
        {
            "strategy_params": {
                "vol_lookback_days": 21,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        FilteredLaggedMomentumPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
    (
        MomentumBetaFilteredPortfolioStrategy,
        {
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "beta_lookback_days": 15,
                "num_high_beta_to_exclude": 1,
                "rsi_length": 3,
                "rsi_overbought": 95,
                "short_max_holding_days": 90,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": True,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        },
    ),
]


def _deterministic_daily_ohlc() -> tuple[pd.DataFrame, list[str], str]:
    rebalance_dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=6, freq="ME"))
    daily_start_date = rebalance_dates.min() - pd.DateOffset(months=14)
    daily_end_date = rebalance_dates.max()
    daily_dates = pd.date_range(start=daily_start_date, end=daily_end_date, freq="B")

    tickers = ["StockA", "StockB", "StockC"]
    frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        base = np.linspace(80.0 + i * 5.0, 150.0 + i * 10.0, len(daily_dates))
        close_px = base
        df = pd.DataFrame(
            {
                "Open": close_px * 0.999,
                "High": close_px * 1.001,
                "Low": close_px * 0.998,
                "Close": close_px,
                "Volume": np.full(len(daily_dates), 10000),
            },
            index=daily_dates,
        )
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=["Ticker", "Field"])
        frames.append(df)
    universe_daily = pd.concat(frames, axis=1)

    bench_close = np.linspace(95.0, 115.0, len(daily_dates))
    bench_df = pd.DataFrame(
        {
            "Open": bench_close * 0.999,
            "High": bench_close * 1.001,
            "Low": bench_close * 0.998,
            "Close": bench_close,
            "Volume": np.full(len(daily_dates), 50000),
        },
        index=daily_dates,
    )
    bench_df.columns = pd.MultiIndex.from_product(
        [["SPY"], bench_df.columns], names=["Ticker", "Field"]
    )
    panel = pd.concat([universe_daily, bench_df], axis=1)
    return panel, tickers, "SPY"


@pytest.mark.parametrize("mom_cls,mom_cfg", MOMENTUM_PORTFOLIO_CONCRETES)
def test_concrete_momentum_portfolio_has_generate_target_weights(
    mom_cls: Type[Any], mom_cfg: dict[str, Any]
) -> None:
    cfg = dict(mom_cfg)
    cfg.setdefault("strategy_params", {})
    strat = mom_cls(cfg)
    fn = getattr(strat, "generate_target_weights", None)
    assert callable(fn)


class _ForbiddenLegacyOutsideTw(MomentumUnfilteredAtrPortfolioStrategy):
    def generate_signals(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not getattr(self, "_generate_target_weights_scan_active", False):
            raise AssertionError("legacy StrategyLogic per-date generate_signals invoked")
        return super().generate_signals(*args, **kwargs)


def test_strategy_logic_portfolio_uses_generate_target_weights_path() -> None:
    panel, universe_tickers, benchmark_ticker = _deterministic_daily_ohlc()
    canonical = CanonicalScenarioConfig.from_dict(
        {
            "name": "tw_portfolio_smoke",
            "strategy": "MomentumUnfilteredAtrPortfolioStrategy",
            "benchmark_ticker": benchmark_ticker,
            "timing_config": {"mode": "time_based", "rebalance_frequency": "ME"},
            "strategy_params": {
                "lookback_months": 3,
                "skip_months": 0,
                "top_decile_fraction": 0.5,
                "num_holdings": 2,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "sma_filter_window": None,
            },
        }
    )
    strat = _ForbiddenLegacyOutsideTw(canonical.to_dict())

    out = generate_signals(
        strat,
        canonical,
        panel,
        universe_tickers,
        benchmark_ticker,
        lambda: False,
        global_config=cast(Optional[Dict[str, Any]], {"feature_flags": {}}),
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] > 0


def _legacy_expected_weights(
    strat_factory: Callable[[], Any],
    *,
    asset_panel: pd.DataFrame,
    benchmark_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_tickers: list[str],
    use_sparse_nan: bool,
    wfo_start: pd.Timestamp | None,
    wfo_end: pd.Timestamp | None,
) -> pd.DataFrame:
    strat = strat_factory()
    cols = list(universe_tickers)
    idx = pd.DatetimeIndex(rebalance_dates)
    fill_value = float("nan") if use_sparse_nan else 0.0
    expected = pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
    expanding_ends, date_masks = _expanding_iloc_ends(asset_panel.index, idx)

    sig = inspect.signature(strat.generate_signals)
    has_nu_param = "non_universe_historical_data" in sig.parameters

    for i, d in enumerate(idx):
        if expanding_ends is not None:
            end = int(expanding_ends[i])
            ahist = asset_panel.iloc[:end]
            bhist = benchmark_panel.iloc[:end]
            nu_hist = pd.DataFrame()
        else:
            assert date_masks is not None
            mask = date_masks[d]
            ahist = asset_panel.loc[mask]
            bhist = benchmark_panel.loc[mask]
            nu_hist = pd.DataFrame()

        kw: dict[str, Any] = {}
        if has_nu_param:
            kw["non_universe_historical_data"] = nu_hist

        row_df = strat.generate_signals(
            all_historical_data=ahist,
            benchmark_historical_data=bhist,
            current_date=d,
            start_date=wfo_start,
            end_date=wfo_end,
            **kw,
        )
        if row_df is None or row_df.empty:
            continue
        row_series = row_df.iloc[0].reindex(cols)
        expected.loc[d, :] = row_series.astype(float)
    return expected


@pytest.mark.parametrize(
    "parity_cls,parity_cfg",
    [
        (SimpleMomentumPortfolioStrategy, MOMENTUM_PORTFOLIO_CONCRETES[0][1]),
        (CalmarMomentumPortfolioStrategy, MOMENTUM_PORTFOLIO_CONCRETES[2][1]),
        (VamsMomentumPortfolioStrategy, MOMENTUM_PORTFOLIO_CONCRETES[6][1]),
        (MomentumBetaFilteredPortfolioStrategy, MOMENTUM_PORTFOLIO_CONCRETES[-1][1]),
    ],
)
def test_generate_target_weights_matches_legacy_generate_signals_semantics(
    parity_cls: Type[Any], parity_cfg: dict[str, Any]
) -> None:
    panel, universe_tickers, benchmark_ticker = _deterministic_daily_ohlc()
    rebalance_dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=6, freq="ME"))

    panel_mi = cast(pd.MultiIndex, panel.columns)
    asset_hist_cols = pd.MultiIndex.from_product(
        [universe_tickers, ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Field"],
    ).intersection(panel_mi)
    asset_full = panel[asset_hist_cols]

    bench_hist_cols = pd.MultiIndex.from_product(
        [[benchmark_ticker], ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Field"],
    ).intersection(panel_mi)
    benchmark_full = panel[bench_hist_cols]

    merged_cfg = dict(parity_cfg)

    def strat_factory() -> Any:
        return parity_cls(merged_cfg)

    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_full,
        benchmark_data=benchmark_full,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rebalance_dates,
        universe_tickers=list(universe_tickers),
        benchmark_ticker=benchmark_ticker,
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )

    tw = strat_factory().generate_target_weights(ctx)
    expected = _legacy_expected_weights(
        strat_factory,
        asset_panel=asset_full,
        benchmark_panel=benchmark_full,
        rebalance_dates=rebalance_dates,
        universe_tickers=list(universe_tickers),
        use_sparse_nan=False,
        wfo_start=None,
        wfo_end=None,
    )

    pd.testing.assert_frame_equal(
        tw.fillna(0.0),
        expected.fillna(0.0),
        rtol=0.0,
        atol=1e-9,
    )
