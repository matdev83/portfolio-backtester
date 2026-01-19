import pandas as pd
import pytest
import numpy as np

from unittest.mock import patch

from portfolio_backtester.strategies.builtins.portfolio.autogluon_sortino_ml_portfolio_strategy import (
    AutogluonSortinoMlPortfolioStrategy,
)


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
            "training_lookback_days": 60,
            "min_training_dates": 2,
            "min_label_observations": 5,
            "target_return": 0.0,
            "exposure_penalty": 0.01,
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
    assert weights.sum() <= 1.0


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

    assert scaled.sum() == pytest.approx(1.0)
    assert (scaled >= 0.0).all()


def test_feature_frame_includes_pairwise_corr(momentum_test_data):
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
    expected_column = "corr_3d_StockA_StockB"
    assert expected_column in feature_frame.columns


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
