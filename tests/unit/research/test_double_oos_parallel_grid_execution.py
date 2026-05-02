"""Parallel grid evaluation safety for ``DoubleOOSWFOProtocol``."""

from __future__ import annotations

import argparse
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.optimization.results import OptimizationResult
from portfolio_backtester.research.artifacts import ResearchArtifactWriter
from portfolio_backtester.research.double_oos_wfo import DoubleOOSWFOProtocol
from portfolio_backtester.research.protocol_config import parse_double_oos_wfo_protocol

from tests.unit.research.test_protocol_config import _minimal_primary_inner


class _ConcurrencyMeter:
    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._depth = 0
        self.max_concurrent = 0

    def __enter__(self) -> None:
        with self._lock:
            self._depth += 1
            if self._depth > self.max_concurrent:
                self.max_concurrent = self._depth

    def __exit__(self, *_exc: object) -> None:
        with self._lock:
            self._depth -= 1


def _scenario_two_arch_grid() -> CanonicalScenarioConfig:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 18],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    inner["execution"] = {"max_parallel_grid_workers": 4, "fail_fast": True}
    raw = {
        "name": "par_scen",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"x": 1},
        "extras": {"research_protocol": inner},
    }
    return CanonicalScenarioConfig.from_dict(raw)


def _minimal_panels(n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    monthly = pd.DataFrame()
    return monthly, daily, rets_full


def test_parallel_without_factory_serializes_grid_and_warns(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    scenario = _scenario_two_arch_grid()
    cfg = parse_double_oos_wfo_protocol({"research_protocol": scenario.extras["research_protocol"]})
    monthly, daily, rets_full = _minimal_panels()

    meter = _ConcurrencyMeter()
    idx = daily.index[:2]

    class _Orc:
        def run_optimization(self, *_args: object, **_kwargs: object) -> OptimizationResult:
            with meter:
                s = pd.Series([0.01, 0.02], index=idx, name="portfolio")
                return OptimizationResult(
                    best_parameters={"a": 1},
                    best_value=1.0,
                    n_evaluations=1,
                    optimization_history=[],
                    stitched_returns=s,
                )

    proto = DoubleOOSWFOProtocol(
        _Orc(), MagicMock(), artifact_writer=ResearchArtifactWriter(tmp_path)
    )

    caplog.set_level(logging.WARNING, logger="portfolio_backtester.research.double_oos_wfo")

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_unseen_results",
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report",
            ),
        ):
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=monthly,
                daily_data=daily,
                rets_full=rets_full,
                args=argparse.Namespace(
                    protocol="double_oos_wfo",
                    force_new_research_run=False,
                    research_skip_unseen=True,
                ),
                global_config={"benchmark": "SPY"},
            )

    assert meter.max_concurrent == 1
    msgs = "\n".join(r.message for r in caplog.records)
    assert "max_parallel_grid_workers" in msgs
    assert "optimization_orchestrator_factory" in msgs


def test_parallel_with_factory_uses_distinct_orchestrator_instances(tmp_path: Any) -> None:
    scenario = _scenario_two_arch_grid()
    cfg = parse_double_oos_wfo_protocol({"research_protocol": scenario.extras["research_protocol"]})
    monthly, daily, rets_full = _minimal_panels()

    created: list[Any] = []
    idx = daily.index[:2]

    def factory() -> MagicMock:
        orch = MagicMock()
        orch._slot_id = object()
        orch.run_optimization.return_value = OptimizationResult(
            best_parameters={"a": 1},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01, 0.02], index=idx, name="portfolio"),
        )
        created.append(orch._slot_id)
        return orch

    opt_default = MagicMock()
    proto = DoubleOOSWFOProtocol(
        opt_default,
        MagicMock(),
        artifact_writer=ResearchArtifactWriter(tmp_path),
        optimization_orchestrator_factory=factory,
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_unseen_results",
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report",
            ),
        ):
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=monthly,
                daily_data=daily,
                rets_full=rets_full,
                args=argparse.Namespace(
                    protocol="double_oos_wfo",
                    force_new_research_run=False,
                    research_skip_unseen=True,
                ),
                global_config={"benchmark": "SPY"},
            )

    assert len(created) == 2
    assert len(set(created)) == 2
    opt_default.run_optimization.assert_not_called()
