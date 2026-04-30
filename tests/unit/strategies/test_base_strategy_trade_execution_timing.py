"""BaseStrategy resolution tests for generic ``trade_execution_timing`` (TDD)."""

from __future__ import annotations

from unittest.mock import patch


from portfolio_backtester.canonical_config import CanonicalScenarioConfig, freeze_config
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
from tests.unit.strategies.test_base_strategy_extended import ConcreteBaseStrategyStub


def _minimal_timing_signal_based() -> dict:
    return {"mode": "signal_based", "scan_frequency": "D", "min_holding_period": 1}


def _minimal_timing_time_based() -> dict:
    return {"mode": "time_based", "rebalance_frequency": "ME", "rebalance_offset": 0}


def test_default_trade_execution_timing_is_bar_close_dict_config() -> None:
    with patch.object(BaseStrategy, "_initialize_providers"):
        with patch.object(BaseStrategy, "_initialize_timing_controller"):
            strat = ConcreteBaseStrategyStub(
                {
                    "strategy_params": {},
                    "timing_config": _minimal_timing_signal_based(),
                }
            )
    assert strat.get_trade_execution_timing() == "bar_close"


def test_timing_config_trade_execution_timing_next_bar_open_portfolio_time_based_dict() -> None:
    timing = dict(_minimal_timing_time_based())
    timing["trade_execution_timing"] = "next_bar_open"
    with patch.object(BaseStrategy, "_initialize_providers"):
        with patch.object(BaseStrategy, "_initialize_timing_controller"):
            strat = ConcreteBaseStrategyStub(
                {
                    "strategy_params": {},
                    "timing_config": timing,
                }
            )
    assert strat.get_trade_execution_timing() == "next_bar_open"


def test_timing_config_trade_execution_timing_next_bar_open_dict_config() -> None:
    timing = dict(_minimal_timing_signal_based())
    timing["trade_execution_timing"] = "next_bar_open"
    with patch.object(BaseStrategy, "_initialize_providers"):
        with patch.object(BaseStrategy, "_initialize_timing_controller"):
            strat = ConcreteBaseStrategyStub(
                {
                    "strategy_params": {},
                    "timing_config": timing,
                }
            )
    assert strat.get_trade_execution_timing() == "next_bar_open"


def test_timing_config_trade_execution_timing_via_canonical_config() -> None:
    timing = dict(_minimal_timing_signal_based())
    timing["trade_execution_timing"] = "next_bar_open"
    canonical = CanonicalScenarioConfig(
        name="t",
        strategy="ConcreteBaseStrategyStub",
        start_date=None,
        end_date=None,
        benchmark_ticker=None,
        timing_config=freeze_config(timing),
        universe_definition=freeze_config({"type": "fixed", "tickers": ["AAA"]}),
        position_sizer=None,
        optimization_metric=None,
        wfo_config=freeze_config({}),
        optimizer_config=freeze_config({}),
        strategy_params=freeze_config({"trade_longs": True, "trade_shorts": True}),
        optimize=None,
        extras=freeze_config({}),
    )
    with patch.object(BaseStrategy, "_initialize_providers"):
        with patch.object(BaseStrategy, "_initialize_timing_controller"):
            strat = ConcreteBaseStrategyStub(canonical)
    assert strat.get_trade_execution_timing() == "next_bar_open"


def test_subclass_may_override_get_trade_execution_timing() -> None:
    class OverrideTimingStrategy(ConcreteBaseStrategyStub):
        def get_trade_execution_timing(self) -> str:
            return "next_bar_open"

    with patch.object(BaseStrategy, "_initialize_providers"):
        with patch.object(BaseStrategy, "_initialize_timing_controller"):
            strat = OverrideTimingStrategy(
                {
                    "strategy_params": {},
                    "timing_config": _minimal_timing_signal_based(),
                }
            )
    assert strat.get_trade_execution_timing() == "next_bar_open"
