"""Integration tests for meta strategies with real strategy classes."""

import pytest
import pandas as pd
import numpy as np

from portfolio_backtester.strategies.builtins.meta.simple_meta_strategy import SimpleMetaStrategy
from portfolio_backtester.backtester_logic.strategy_logic import generate_signals
from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns


class TestMetaStrategyIntegration:
    """Integration tests for meta strategies."""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        assets = ["AAPL", "MSFT", "GOOGL"]
        columns = pd.MultiIndex.from_product(
            [assets, ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Field"]
        )
        np.random.seed(42)
        data = np.random.randn(len(dates), len(columns)) * 0.02 + 1.0
        data = np.cumprod(data, axis=0) * 100
        df = pd.DataFrame(data, index=dates, columns=columns)
        for asset in assets:
            df[(asset, "High")] = df[[(asset, "Open"), (asset, "Close")]].max(axis=1) * (
                1 + np.random.rand(len(dates)) * 0.01
            )
            df[(asset, "Low")] = df[[(asset, "Open"), (asset, "Close")]].min(axis=1) * (
                1 - np.random.rand(len(dates)) * 0.01
            )
            df[(asset, "Volume")] = np.random.randint(1_000_000, 10_000_000, len(dates))
        return df

    @pytest.fixture
    def benchmark_data(self):
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        columns = pd.MultiIndex.from_product(
            [["SPY"], ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Field"]
        )
        np.random.seed(43)
        data = np.random.randn(len(dates), len(columns)) * 0.015 + 1.0
        data = np.cumprod(data, axis=0) * 100
        df = pd.DataFrame(data, index=dates, columns=columns)
        df[("SPY", "High")] = df[("SPY", "Open")].combine(df[("SPY", "Close")], max) * (
            1 + np.random.rand(len(dates)) * 0.005
        )
        df[("SPY", "Low")] = df[("SPY", "Open")].combine(df[("SPY", "Close")], min) * (
            1 - np.random.rand(len(dates)) * 0.005
        )
        df[("SPY", "Volume")] = np.random.randint(50_000_000, 200_000_000, len(dates))
        return df

    def test_simple_meta_strategy_signal_generation(self, sample_data, benchmark_data):
        config = {
            "initial_capital": 1_000_000,
            "allocations": [
                {
                    "strategy_id": "momentum",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "num_holdings": 2,
                        "price_column_asset": "Close",
                        "price_column_benchmark": "Close",
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                    },
                    "weight": 0.7,
                },
                {
                    "strategy_id": "seasonal",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 5,
                        "hold_days": 5,
                        "price_column_asset": "Close",
                        "trade_longs": True,
                        "trade_shorts": False,
                        "timing_config": {"mode": "signal_based"},
                    },
                    "weight": 0.3,
                },
            ],
        }
        meta = SimpleMetaStrategy(config)
        current_date = pd.Timestamp("2023-06-15")
        historical = sample_data[sample_data.index <= current_date]
        bench_hist = benchmark_data[benchmark_data.index <= current_date]
        signals = meta.generate_signals(
            all_historical_data=historical,
            benchmark_historical_data=bench_hist,
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date,
        )
        assert isinstance(signals, pd.DataFrame)
        assert current_date in signals.index
        assert len(signals.columns) > 0
        assert all(abs(val) <= 2.0 for val in signals.loc[current_date] if not pd.isna(val))

    def test_capital_allocation_calculation(self):
        config = {
            "initial_capital": 1_000_000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 6,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                    },
                    "weight": 0.6,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {"entry_day": 5, "timing_config": {"mode": "signal_based"}},
                    "weight": 0.4,
                },
            ],
        }
        meta = SimpleMetaStrategy(config)
        alloc = meta.calculate_sub_strategy_capital()
        assert alloc["strategy1"] == 600_000
        assert alloc["strategy2"] == 400_000
        meta.update_available_capital({"strategy1": 0.10, "strategy2": -0.05})
        expected = 1_000_000 + (600_000 * 0.10) - (400_000 * 0.05)
        assert meta.available_capital == expected
        new_alloc = meta.calculate_sub_strategy_capital()
        assert new_alloc["strategy1"] == expected * 0.6
        assert new_alloc["strategy2"] == expected * 0.4

    def test_get_universe_combination(self):
        config = {
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "universe_config": ["AAPL", "MSFT"],
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                    },
                    "weight": 0.5,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "universe_config": ["GOOGL", "AMZN"],
                        "timing_config": {"mode": "signal_based"},
                    },
                    "weight": 0.5,
                },
            ]
        }
        meta = SimpleMetaStrategy(config)
        universe = meta.get_universe({"universe": ["DEFAULT1", "DEFAULT2"]})
        assert isinstance(universe, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in universe)
        assert all(isinstance(t, str) and isinstance(w, (int, float)) for t, w in universe)

    def test_end_to_end_trade_aggregation(self, sample_data, benchmark_data):
        meta_cfg = {
            "initial_capital": 1_000_000,
            "allocations": [
                {
                    "strategy_id": "momentum",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "num_holdings": 2,
                        "price_column_asset": "Close",
                        "price_column_benchmark": "Close",
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                    },
                    "weight": 0.7,
                },
                {
                    "strategy_id": "seasonal",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 5,
                        "hold_days": 5,
                        "price_column_asset": "Close",
                        "trade_longs": True,
                        "trade_shorts": False,
                        "timing_config": {"mode": "signal_based"},
                    },
                    "weight": 0.3,
                },
            ],
        }
        meta = SimpleMetaStrategy(meta_cfg)
        assert len(meta.get_aggregated_trades()) == 0
        assert meta.get_trade_aggregator().initial_capital == 1_000_000
        scenario_config = {
            "name": "test_meta_strategy",
            "strategy": "SimpleMetaStrategy",
            "strategy_params": meta_cfg,
            "timing_config": {"rebalance_frequency": "M"},
        }
        universe_tickers = ["AAPL", "MSFT", "GOOGL"]
        benchmark_ticker = "SPY"
        signals = generate_signals(
            strategy=meta,
            scenario_config=scenario_config,
            price_data_daily_ohlc=sample_data,
            universe_tickers=universe_tickers,
            benchmark_ticker=benchmark_ticker,
            has_timed_out=lambda: False,
        )
        assert not signals.empty
        assert len(signals.columns) == len(universe_tickers)
        aggregated_trades = meta.get_aggregated_trades()
        assert len(aggregated_trades) > 0
        rets_daily = sample_data.xs("Close", level="Field", axis=1).pct_change().fillna(0.0)
        global_config = {"benchmark": benchmark_ticker, "portfolio_value": 1_000_000.0}
        portfolio_returns, trade_tracker = calculate_portfolio_returns(
            sized_signals=signals,
            scenario_config=scenario_config,
            price_data_daily_ohlc=sample_data,
            rets_daily=rets_daily,
            universe_tickers=universe_tickers,
            global_config=global_config,
            track_trades=True,
            strategy=meta,
        )
        assert len(portfolio_returns) == len(sample_data)
        assert not portfolio_returns.isna().all()
        performance = meta.get_comprehensive_performance_metrics()
        assert performance["total_trades"] == len(aggregated_trades)
        assert performance["initial_capital"] == 1_000_000
        assert "total_return" in performance
        assert "sharpe_ratio" in performance
