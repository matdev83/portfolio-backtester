"""Invariant: optimization BacktestRunner path matches StrategyBacktester returns series."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.interfaces.cache_manager_interface import create_cache_manager
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from tests.fixtures.market_data import MarketDataFixture


@pytest.fixture()
def engine_path_invariant_setup() -> tuple:
    raw_scenario = {
        "name": "engine_path_invariant",
        "strategy": "SimpleMomentumPortfolioStrategy",
        "universe_config": {"type": "fixed", "tickers": ["AAPL", "MSFT", "GOOGL"]},
        "strategy_params": {"lookback_months": 12, "num_holdings": 2},
        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
    }
    global_config = {
        "rebalance_frequency": "M",
        "benchmark": "SPY",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
    }
    canonical = ScenarioNormalizer().normalize(scenario=raw_scenario, global_config=global_config)
    fixture = MarketDataFixture.create_basic_data(
        tickers=("AAPL", "MSFT", "GOOGL", "SPY"),
        start_date="2020-01-01",
        end_date="2023-12-31",
    )
    closes = fixture.xs("Close", axis=1, level="Field")
    assert isinstance(closes, pd.DataFrame)
    monthly_closes = closes.resample("BME").last()
    daily_closes = closes
    rets_full = closes.pct_change(fill_method=None).fillna(0)
    assert isinstance(rets_full, pd.DataFrame)
    shared_cache = create_cache_manager()
    manager = StrategyManager()
    runner = BacktestRunner(global_config, shared_cache, manager)
    tester = StrategyBacktester(global_config, MagicMock(), data_cache=shared_cache)
    return canonical, monthly_closes, daily_closes, rets_full, runner, tester


def test_runner_run_scenario_matches_strategy_backtester_returns(
    engine_path_invariant_setup: tuple,
) -> None:
    canonical, monthly_closes, daily_closes, rets_full, runner, tester = engine_path_invariant_setup
    timeout_checker = runner.timeout_checker

    res_runner = runner.run_scenario(
        canonical,
        price_data_monthly_closes=monthly_closes,
        price_data_daily_ohlc=daily_closes,
        rets_daily=rets_full,
        verbose=False,
    )
    res_engine = tester.backtest_strategy(
        canonical,
        monthly_closes,
        daily_closes,
        rets_full,
        track_trades=False,
        timeout_checker=timeout_checker,
    )

    assert res_runner is not None
    pd.testing.assert_series_equal(res_runner, res_engine.returns, check_freq=False)
