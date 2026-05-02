"""Regression guardrail: richer OptimizationData payloads must not perturb evaluation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.interfaces import create_cache_manager
from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.optimization.market_data_panel import MarketDataPanel
from portfolio_backtester.optimization.results import OptimizationData
from portfolio_backtester.optimization.window_bounds import build_window_bounds
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer


def test_evaluate_parameters_objective_stable_with_optional_typing() -> None:
    gc = {
        "benchmark": "SPY",
        "data_source": {"type": "memory", "data": {}},
    }

    scenario_raw = {
        "name": "typed_payload_guardrail",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"open_long_prob": 0.55, "seed": 404, "symbol": "SPY"},
        "start_date": "2024-01-03",
        "end_date": "2024-04-29",
        "extras": {},
    }

    canonical = ScenarioNormalizer().normalize(scenario=scenario_raw, global_config=dict(gc))

    start_val = canonical.start_date or scenario_raw["start_date"]
    end_val = canonical.end_date or scenario_raw["end_date"]
    assert start_val is not None and end_val is not None
    start_ts = pd.Timestamp(str(start_val))
    end_ts = pd.Timestamp(str(end_val))
    rng = pd.DatetimeIndex(pd.date_range(start_ts, end_ts, freq="B", tz=None).normalize())

    spy = np.linspace(100.0, 110.0, len(rng), dtype=float)
    qqq = np.linspace(260.0, 270.0, len(rng), dtype=float)

    closes = pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=rng)
    monthly = closes.resample("ME").last()
    daily = closes.copy()
    rets_full = closes.pct_change(fill_method=None).fillna(0.0)

    window = WFOWindow(
        train_start=rng[5],
        train_end=rng[40],
        test_start=rng[41],
        test_end=rng[-5],
    )

    base = OptimizationData(monthly=monthly, daily=daily, returns=rets_full, windows=[window])

    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets_full)
    enriched = OptimizationData(
        monthly=monthly,
        daily=daily,
        returns=rets_full,
        windows=[window],
        market_data=panel,
        daily_np=panel.daily_np,
        returns_np=panel.returns_np,
        daily_index_np=panel.row_index_naive_datetime64(),
        tickers_list=list(panel.tickers),
        window_bounds=[build_window_bounds(panel.daily_index_naive, window)],
    )

    targets_raw = canonical.extras.get("optimization_targets") or []
    targets = cast(list[dict[str, Any]], targets_raw)
    if targets:
        metrics = [str(t["name"]) for t in targets]
    else:
        metrics = [str(canonical.optimization_metric or "Calmar")]

    eval_global = dict(gc)
    backtester = StrategyBacktester(
        global_config=eval_global, data_source=None, data_cache=create_cache_manager()
    )

    evaluator = BacktestEvaluator(
        metrics_to_optimize=metrics,
        is_multi_objective=len(metrics) > 1,
        n_jobs=1,
        enable_parallel_optimization=False,
    )

    params = {"open_long_prob": 0.31}
    baseline = evaluator.evaluate_parameters(params, canonical, base, backtester)
    typed = evaluator.evaluate_parameters(params, canonical, enriched, backtester)

    assert typed.objective_value == pytest.approx(baseline.objective_value, rel=0.0, abs=1e-12)
