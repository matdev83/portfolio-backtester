import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest
from portfolio_backtester.backtester_logic.strategy_logic import _expanding_iloc_ends
from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.portfolio.drift_regime_conditional_factor_portfolio_strategy import (
    DriftRegimeConditionalFactorPortfolioStrategy,
)


def _legacy_expected_weights(
    strategy: DriftRegimeConditionalFactorPortfolioStrategy,
    *,
    asset_panel: pd.DataFrame,
    benchmark_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_tickers: list[str],
    use_sparse_nan: bool,
    wfo_start: pd.Timestamp | None,
    wfo_end: pd.Timestamp | None,
) -> pd.DataFrame:
    cols = list(universe_tickers)
    idx = pd.DatetimeIndex(rebalance_dates)
    fill_value = float("nan") if use_sparse_nan else 0.0
    expected = pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
    expanding_ends, date_masks = _expanding_iloc_ends(asset_panel.index, idx)
    sig = inspect.signature(strategy.generate_signals)
    has_nu_param = "non_universe_historical_data" in sig.parameters
    for i, d in enumerate(idx):
        if expanding_ends is not None:
            end = int(expanding_ends[i])
            ahist = asset_panel.iloc[:end]
            bhist = benchmark_panel.iloc[:end]
        else:
            assert date_masks is not None
            mask = date_masks[d]
            ahist = asset_panel.loc[mask]
            bhist = benchmark_panel.loc[mask]
        kw: dict[str, Any] = {}
        if has_nu_param:
            kw["non_universe_historical_data"] = None
        row_df = strategy.generate_signals(
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


@pytest.fixture
def drift_test_data():
    """Generate test data with a clear drift regime for some stocks."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    tickers = ["TrendStock", "MeanStock", "LoserStock"]

    data_frames = []
    for ticker in tickers:
        if ticker == "TrendStock":
            # Consistent uptrend (positive drift)
            # Drift for TrendStock should be > 0.6
            returns = np.random.normal(0.002, 0.01, size=len(dates))
        elif ticker == "MeanStock":
            # Sideways
            returns = np.random.normal(0, 0.01, size=len(dates))
        else:
            # Downtrend
            returns = np.random.normal(-0.002, 0.01, size=len(dates))

        prices = (1 + returns).cumprod() * 100

        df = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": 1000,
            },
            index=dates,
        )

        df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=["Ticker", "Field"])
        data_frames.append(df)

    all_data = pd.concat(data_frames, axis=1)

    # Mock benchmark
    bench_returns = np.random.normal(0.0005, 0.01, size=len(dates))
    bench_prices = (1 + bench_returns).cumprod() * 100
    benchmark_df = pd.DataFrame(
        {
            "Open": bench_prices * 0.99,
            "High": bench_prices * 1.01,
            "Low": bench_prices * 0.98,
            "Close": bench_prices,
            "Volume": 10000,
        },
        index=dates,
    )
    benchmark_df.columns = pd.MultiIndex.from_product(
        [["SPY"], benchmark_df.columns], names=["Ticker", "Field"]
    )

    return {"dates": dates, "all_data": all_data, "benchmark_data": benchmark_df}


class TestDriftRegimeConditionalFactorPortfolioStrategy:

    def test_initialization(self):
        config = {"strategy_params": {"drift_window": 63}}
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)
        params = strategy.strategy_params.get("strategy_params", strategy.strategy_params)
        assert params["drift_window"] == 63
        assert params["drift_threshold"] == 0.6

    def test_generate_signals_basic(self, drift_test_data):
        config = {
            "strategy_params": {
                "drift_window": 63,
                "drift_threshold": 0.5,  # Lower threshold to ensure we get signals
                "num_holdings": 2,
                "reversal_window": 21,
                "value_window": 126,
                "min_history_days": 0,
            }
        }
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)

        current_date = drift_test_data["dates"][-1]
        signals = strategy.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date,
        )

        assert isinstance(signals, pd.DataFrame)
        assert signals.index[0] == current_date
        assert not signals.empty

    def test_insufficient_data(self, drift_test_data):
        config = {
            "strategy_params": {"drift_window": 300, "value_window": 300, "min_history_days": 0}
        }
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)

        # Only 300 days available, so 300-day windows might fail or be borderline
        # Let's use an even earlier date
        current_date = drift_test_data["dates"][50]
        signals = strategy.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date,
        )

        assert (signals == 0).all().all()

    def test_drift_threshold_filtering(self, drift_test_data):
        # Set threshold very high to ensure NO stocks pass (unless extremely lucky)
        config = {
            "strategy_params": {"drift_threshold": 0.99, "drift_window": 63, "min_history_days": 0}
        }
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)

        current_date = drift_test_data["dates"][-1]
        signals = strategy.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date,
        )

        # Should be zero as no stock is likely to have 99% positive days
        assert (signals == 0).all().all()

    def test_long_short_signals(self, drift_test_data):
        # 1. Test Long/Short (Default)
        config_ls = {
            "strategy_params": {
                "drift_threshold": 0.4,  # Lower to ensure enough candidates
                "num_holdings": 1,
                "trade_longs": True,
                "trade_shorts": True,
                "min_history_days": 0,
            }
        }
        strategy_ls = DriftRegimeConditionalFactorPortfolioStrategy(config_ls)
        current_date = drift_test_data["dates"][-1]
        signals_ls = strategy_ls.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date,
        )

        # Should have both positive and negative weights (if enough candidates)
        assert (signals_ls > 0).any().any()
        assert (signals_ls < 0).any().any()
        # Sum should be near 0 (dollar neutral style with equal holdings)
        # Note: if only one side has candidates, sum will follow that side.
        # But in our test data, we should have multiple.

        # 2. Test Long Only
        config_lo = {
            "strategy_params": {
                "drift_threshold": 0.4,
                "num_holdings": 1,
                "trade_longs": True,
                "trade_shorts": False,
                "min_history_days": 0,
            }
        }
        strategy_lo = DriftRegimeConditionalFactorPortfolioStrategy(config_lo)
        signals_lo = strategy_lo.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date,
        )
        assert (signals_lo >= 0).all().all()
        assert (signals_lo > 0).any().any()

        # 3. Test Short Only
        config_so = {
            "strategy_params": {
                "drift_threshold": 0.4,
                "num_holdings": 1,
                "trade_longs": False,
                "trade_shorts": True,
                "min_history_days": 0,
            }
        }
        strategy_so = DriftRegimeConditionalFactorPortfolioStrategy(config_so)
        signals_so = strategy_so.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date,
        )
        assert (signals_so <= 0).all().all()
        assert (signals_so < 0).any().any()


def test_drift_strategy_exposes_generate_target_weights() -> None:
    cfg = {"strategy_params": {"min_history_days": 0}}
    strat = DriftRegimeConditionalFactorPortfolioStrategy(cfg)
    assert callable(getattr(strat, "generate_target_weights", None))


def test_drift_generate_target_weights_matches_legacy_generate_signals(
    drift_test_data: dict[str, Any],
) -> None:
    config = {
        "strategy_params": {
            "drift_window": 63,
            "drift_threshold": 0.5,
            "num_holdings": 2,
            "reversal_window": 21,
            "min_history_days": 0,
        }
    }
    dates = drift_test_data["dates"]
    rebalance_dates = pd.to_datetime([dates[150], dates[200], dates[250], dates[-1]])
    tickers = ["TrendStock", "MeanStock", "LoserStock"]
    asset_full = drift_test_data["all_data"]
    benchmark_full = drift_test_data["benchmark_data"]
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_full,
        benchmark_data=benchmark_full,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rebalance_dates,
        universe_tickers=tickers,
        benchmark_ticker="SPY",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    tw_strat = DriftRegimeConditionalFactorPortfolioStrategy(dict(config))
    tw = tw_strat.generate_target_weights(ctx)
    leg_strat = DriftRegimeConditionalFactorPortfolioStrategy(dict(config))
    expected = _legacy_expected_weights(
        leg_strat,
        asset_panel=asset_full,
        benchmark_panel=benchmark_full,
        rebalance_dates=rebalance_dates,
        universe_tickers=tickers,
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
