from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio_backtester.backtester_logic.strategy_logic import _expanding_iloc_ends
from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.portfolio.autogluon_sortino_ml_portfolio_strategy import (
    AutogluonSortinoMlPortfolioStrategy,
)


def _legacy_expected_weights_autogluon(
    strategy: AutogluonSortinoMlPortfolioStrategy,
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
def momentum_test_data():
    dates = pd.date_range(start="2023-01-02", periods=260, freq="B")
    tickers = ["StockA", "StockB", "StockC", "StockD"]
    data_frames = []

    rng = np.random.default_rng(seed=123)
    for ticker in tickers:
        base_price = rng.normal(100, 5) + np.linspace(0, 10, len(dates))
        noise = rng.normal(0, 0.5, size=len(dates))
        close_prices = base_price + noise
        open_prices = close_prices - rng.uniform(0, 0.5, size=len(dates))
        high_prices = close_prices + rng.uniform(0, 0.5, size=len(dates))
        low_prices = close_prices - rng.uniform(0, 0.5, size=len(dates))
        volume = rng.integers(1000, 5000, size=len(dates))

        df = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volume,
            },
            index=dates,
        )
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=["Ticker", "Field"])
        data_frames.append(df)

    daily_ohlc_data = pd.concat(data_frames, axis=1)

    benchmark_close = 100 + np.cumsum(rng.normal(0.02, 0.2, size=len(dates)))
    benchmark_df = pd.DataFrame(
        {
            "Open": benchmark_close - rng.uniform(0, 0.2, size=len(dates)),
            "High": benchmark_close + rng.uniform(0, 0.2, size=len(dates)),
            "Low": benchmark_close - rng.uniform(0, 0.2, size=len(dates)),
            "Close": benchmark_close,
            "Volume": rng.integers(10000, 50000, size=len(dates)),
        },
        index=dates,
    )
    benchmark_df.columns = pd.MultiIndex.from_product(
        [["SPY"], benchmark_df.columns], names=["Ticker", "Field"]
    )

    return {
        "daily_ohlc_data": daily_ohlc_data,
        "benchmark_ohlc_data": benchmark_df,
    }


class DummyPredictor:
    def __init__(self, value: float = 0.1) -> None:
        self.value = value

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series([self.value] * len(data), index=data.index)


def _make_strategy_config() -> dict:
    return {
        "strategy_params": {
            "rebalance_frequency": "ME",
            "feature_windows": [5, 10],
            "correlation_window": 3,
            "label_lookback_days": 10,
            "label_horizons_days": [5, 10, 15],
            "label_horizon_weights": [1.0, 1.0, 1.0],
            "training_lookback_days": 60,
            "min_training_dates": 2,
            "min_label_observations": 5,
            "target_return": 0.0,
            "exposure_penalty": 0.0,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
            "trade_longs": True,
            "trade_shorts": False,
        }
    }


def test_generate_signals_long_only(momentum_test_data):
    strategy = AutogluonSortinoMlPortfolioStrategy(_make_strategy_config())
    current_date = momentum_test_data["daily_ohlc_data"].index[-1]
    historical_assets = momentum_test_data["daily_ohlc_data"].loc[
        momentum_test_data["daily_ohlc_data"].index <= current_date
    ]
    historical_benchmark = momentum_test_data["benchmark_ohlc_data"].loc[
        momentum_test_data["benchmark_ohlc_data"].index <= current_date
    ]

    with patch.object(
        AutogluonSortinoMlPortfolioStrategy,
        "_get_or_train_predictor",
        return_value=DummyPredictor(0.1),
    ):
        weights_df = strategy.generate_signals(
            all_historical_data=historical_assets,
            benchmark_historical_data=historical_benchmark,
            current_date=current_date,
        )

    assert isinstance(weights_df, pd.DataFrame)
    assert not weights_df.empty
    weights = weights_df.iloc[0]
    assert (weights >= 0.0).all()
    assert weights.sum() == pytest.approx(1.0)


def test_vol_target_scales_weights_to_max_gross():
    dates = pd.date_range(start="2023-01-02", periods=40, freq="B")
    pattern = np.tile([0.001, 0.0015], 20)
    returns_df = pd.DataFrame(
        {
            "StockA": pattern,
            "StockB": pattern,
            "StockC": pattern,
        },
        index=dates,
    )

    config = _make_strategy_config()
    config["strategy_params"].update(
        {
            "vol_target_enabled": True,
            "target_vol_annual": 0.10,
            "vol_lookback_days": 20,
            "vol_max_gross_exposure": 1.0,
        }
    )
    strategy = AutogluonSortinoMlPortfolioStrategy(config)

    weights = pd.Series([0.2, 0.2, 0.2], index=returns_df.columns)
    scaled = strategy._post_process_weights(weights, returns_df, dates[-1])

    assert 0.0 < scaled.sum() <= 1.0 + 1e-6
    assert (scaled >= 0.0).all()


def test_feature_frame_excludes_pairwise_corr(momentum_test_data):
    strategy = AutogluonSortinoMlPortfolioStrategy(_make_strategy_config())
    current_date = momentum_test_data["daily_ohlc_data"].index[-1]
    close_df = momentum_test_data["daily_ohlc_data"].xs("Close", level="Field", axis=1)
    benchmark_close = (
        momentum_test_data["benchmark_ohlc_data"].xs("Close", level="Field", axis=1).iloc[:, 0]
    )
    returns_df = close_df.pct_change(fill_method=None).dropna(how="all")

    feature_frame = strategy._build_feature_frame(
        dates=[current_date],
        close_df=close_df,
        benchmark_close=benchmark_close,
        returns_df=returns_df,
    )

    assert not feature_frame.empty
    corr_columns = [col for col in feature_frame.columns if col.startswith("corr_")]
    assert corr_columns == []


def test_returns_zero_weights_with_insufficient_training(momentum_test_data):
    config = _make_strategy_config()
    config["strategy_params"]["min_training_dates"] = 10
    strategy = AutogluonSortinoMlPortfolioStrategy(config)

    current_date = momentum_test_data["daily_ohlc_data"].index[10]
    historical_assets = momentum_test_data["daily_ohlc_data"].loc[
        momentum_test_data["daily_ohlc_data"].index <= current_date
    ]
    historical_benchmark = momentum_test_data["benchmark_ohlc_data"].loc[
        momentum_test_data["benchmark_ohlc_data"].index <= current_date
    ]

    weights_df = strategy.generate_signals(
        all_historical_data=historical_assets,
        benchmark_historical_data=historical_benchmark,
        current_date=current_date,
    )

    assert (weights_df == 0.0).all().all()


def test_forward_labels_skip_incomplete_windows():
    dates = pd.date_range(start="2024-01-02", periods=6, freq="B")
    returns_df = pd.DataFrame({"StockA": np.linspace(0.001, 0.006, len(dates))}, index=dates)
    strategy = AutogluonSortinoMlPortfolioStrategy(_make_strategy_config())
    training_dates = [dates[0], dates[-2]]

    with patch.object(
        AutogluonSortinoMlPortfolioStrategy,
        "_optimize_sortino_weights",
        return_value=pd.Series([0.5], index=returns_df.columns),
    ):
        labels = strategy._build_label_frame(
            training_dates=training_dates,
            returns_df=returns_df,
            min_observations=2,
            label_horizons_days=[2],
            label_horizon_weights=[1.0],
            target_return=0.0,
            exposure_penalty=0.0,
        )

    assert not labels.empty
    assert set(labels["date"]) == {dates[0]}


def test_forward_labels_use_future_returns_only():
    dates = pd.date_range(start="2024-01-02", periods=8, freq="B")
    returns_df = pd.DataFrame(
        {
            "StockA": np.linspace(0.001, 0.008, len(dates)),
            "StockB": np.linspace(0.002, 0.009, len(dates)),
        },
        index=dates,
    )
    strategy = AutogluonSortinoMlPortfolioStrategy(_make_strategy_config())
    training_date = dates[2]
    captured_starts = []

    def _capture_window(window, target_return, exposure_penalty):
        captured_starts.append(window.index.min())
        return pd.Series([0.5, 0.5], index=returns_df.columns)

    with patch.object(
        AutogluonSortinoMlPortfolioStrategy,
        "_optimize_sortino_weights",
        side_effect=_capture_window,
    ):
        labels = strategy._build_label_frame(
            training_dates=[training_date],
            returns_df=returns_df,
            min_observations=2,
            label_horizons_days=[3],
            label_horizon_weights=[1.0],
            target_return=0.0,
            exposure_penalty=0.0,
        )

    assert not labels.empty
    assert captured_starts
    assert captured_starts[0] > training_date


def test_autogluon_strategy_exposes_generate_target_weights(
    momentum_test_data: dict[str, pd.DataFrame],
) -> None:
    strategy = AutogluonSortinoMlPortfolioStrategy(_make_strategy_config())
    assert callable(getattr(strategy, "generate_target_weights", None))


def test_autogluon_generate_target_weights_matches_legacy_generate_signals(
    momentum_test_data: dict[str, pd.DataFrame],
) -> None:
    daily = momentum_test_data["daily_ohlc_data"]
    bench = momentum_test_data["benchmark_ohlc_data"]
    ix = pd.DatetimeIndex(daily.index)
    rebalance_dates = ix[[-55, -40, -25, -1]]

    universe_tickers = ["StockA", "StockB", "StockC", "StockD"]
    ctx = StrategyContext.from_standard_inputs(
        asset_data=daily,
        benchmark_data=bench,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rebalance_dates,
        universe_tickers=universe_tickers,
        benchmark_ticker="SPY",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    cfg = _make_strategy_config()
    with patch.object(
        AutogluonSortinoMlPortfolioStrategy,
        "_get_or_train_predictor",
        return_value=DummyPredictor(0.1),
    ):
        tw_strat = AutogluonSortinoMlPortfolioStrategy(dict(cfg))
        tw = tw_strat.generate_target_weights(ctx)
        leg_strat = AutogluonSortinoMlPortfolioStrategy(dict(cfg))
        expected = _legacy_expected_weights_autogluon(
            leg_strat,
            asset_panel=daily,
            benchmark_panel=bench,
            rebalance_dates=rebalance_dates,
            universe_tickers=universe_tickers,
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
