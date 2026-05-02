"""Tests for ``DoubleOOSWFOProtocol`` grid, selection, unseen, and artifact ordering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from frozendict import frozendict

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.optimization.results import OptimizationResult
from portfolio_backtester.research.artifacts import ResearchArtifactWriter
from portfolio_backtester.research.registry import ResearchRegistryError
from portfolio_backtester.research.protocol_config import (
    FinalUnseenMode,
    ROBUST_COMPOSITE_METRIC_NAME,
    ResearchProtocolConfigError,
    parse_double_oos_wfo_protocol,
)
from portfolio_backtester.research.double_oos_wfo import (
    DoubleOOSWFOProtocol,
    expand_wfo_architecture_grid,
    slice_panel_by_dates,
    training_scenario_dict_for_architecture,
    unseen_scenario_dict_fixed_params,
    unseen_scenario_dict_reoptimize,
)
from portfolio_backtester.research.results import SelectedProtocol
from portfolio_backtester.research.constraints import ResearchConstraintError

from tests.unit.research.test_protocol_config import _minimal_primary_inner


def _scenario() -> CanonicalScenarioConfig:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 18],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    raw = {
        "name": "p_scen",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"x": 1},
        "extras": {"research_protocol": inner},
    }
    return CanonicalScenarioConfig.from_dict(raw)


@pytest.fixture
def protocol_config_dict() -> dict:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 18],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    return {"research_protocol": inner}


def _idx(n: int = 10) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _wide_idx() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", "2024-06-01")


@pytest.fixture()
def iso_writer(tmp_path: Path) -> ResearchArtifactWriter:
    return ResearchArtifactWriter(tmp_path)


def test_grid_calls_run_optimization_per_architecture(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()

    dates = _idx(30)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    monthly = pd.DataFrame()

    s1 = pd.Series([0.01, 0.02], index=dates[:2], name="portfolio")
    s2 = pd.Series([-0.01, 0.03], index=dates[:2], name="portfolio")

    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"a": 1},
            best_value=1.0,
            n_evaluations=3,
            optimization_history=[],
            stitched_returns=s1,
        ),
        OptimizationResult(
            best_parameters={"a": 2},
            best_value=1.0,
            n_evaluations=5,
            optimization_history=[],
            stitched_returns=s2,
        ),
    ]
    bt = MagicMock()

    call_order: list[str] = []

    def wg(*_a: object, **_k: object) -> None:
        call_order.append("grid")

    def ws(*_a: object, **_k: object) -> None:
        call_order.append("selected")

    def wl(*_a: object, **_k: object) -> None:
        call_order.append("lock")

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch("portfolio_backtester.research.double_oos_wfo.calculate_metrics") as mock_cm,
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_grid_results", side_effect=wg
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_selected_protocols",
                side_effect=ws,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file", side_effect=wl),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            mock_cm.return_value = pd.Series({"Sharpe": 1.0, "Calmar": 0.5})

            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=monthly,
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert opt.run_optimization.call_count == 2
    assert len(r.grid_results) == 2
    assert r.grid_results[0].n_evaluations == 3
    assert r.grid_results[1].best_parameters == {"a": 2}
    assert call_order == ["grid", "selected", "lock"]


def test_tie_break_prefers_first_grid_architecture(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()

    dates = _idx(8)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    identical = pd.Series([0.01, 0.01], index=dates[:2])
    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"tag": "first"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=identical,
        ),
        OptimizationResult(
            best_parameters={"tag": "second"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=identical.copy(),
        ),
    ]
    bt = MagicMock()

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    ranks = [(sp.rank, sp.selected_parameters["tag"]) for sp in r.selected_protocols]
    assert ranks[0][1] == "first"
    assert [sp.selected_parameters["tag"] for sp in r.selected_protocols] == ["first", "second"]


def test_robust_composite_selection_ranking_by_metric_weights(
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 18],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    del inner["selection"]["metric"]
    inner["scoring"] = {"type": "composite_rank", "weights": {"Calmar": 1.0}}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.selection.metric == ROBUST_COMPOSITE_METRIC_NAME
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _idx(30)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    s_low = pd.Series([0.01, 0.02], index=dates[:2])
    s_high = pd.Series([0.01, 0.03], index=dates[:2])
    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"tag": "low"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=s_low,
        ),
        OptimizationResult(
            best_parameters={"tag": "high"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=s_high,
        ),
    ]
    bt = MagicMock()
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                side_effect=[
                    pd.Series(
                        {
                            "Calmar": 0.5,
                            "Sortino": 1.0,
                            "Total Return": 0.1,
                            "Max Drawdown": -0.2,
                        }
                    ),
                    pd.Series(
                        {
                            "Calmar": 1.5,
                            "Sortino": 1.0,
                            "Total Return": 0.1,
                            "Max Drawdown": -0.2,
                        }
                    ),
                ],
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )
    assert r.grid_results[0].score < r.grid_results[1].score
    assert r.selected_protocols[0].selected_parameters["tag"] == "high"
    assert r.selected_protocols[0].score == r.grid_results[1].score


def test_omitted_metric_defaults_robust_composite_same_as_explicit() -> None:
    inner_a = _minimal_primary_inner()
    del inner_a["selection"]["metric"]
    inner_b = _minimal_primary_inner()
    inner_b["selection"]["metric"] = ROBUST_COMPOSITE_METRIC_NAME
    cfg_a = parse_double_oos_wfo_protocol({"research_protocol": inner_a})
    cfg_b = parse_double_oos_wfo_protocol({"research_protocol": inner_b})
    assert cfg_a.selection.metric == cfg_b.selection.metric
    assert cfg_a.composite_scoring == cfg_b.composite_scoring


def test_training_and_unseen_scenario_dicts_do_not_mutate_canonical_baseline() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    scenario = _scenario()
    baseline = scenario.to_dict()
    arch = expand_wfo_architecture_grid(grid=cfg.wfo_window_grid)[0]
    training_scenario_dict_for_architecture(scenario, cfg.global_train_period, arch)
    sel = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={},
        score=1.0,
        robust_score=None,
        selected_parameters={"z": 99},
    )
    unseen_scenario_dict_reoptimize(scenario, cfg.unseen_test_period, sel)
    unseen_scenario_dict_fixed_params(scenario, cfg.unseen_test_period, sel)
    assert scenario.to_dict() == baseline


def test_run_raises_when_architecture_grid_exceeds_max_grid_cells(
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 18, 24],
        "test_window_months": [6, 12],
        "wfo_step_months": [3, 6],
        "walk_forward_type": ["rolling", "expanding"],
    }
    inner["execution"] = {"max_grid_cells": 2}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert (
        len(expand_wfo_architecture_grid(grid=cfg.wfo_window_grid)) > cfg.execution.max_grid_cells
    )

    scenario = _scenario()
    opt = MagicMock()
    bt = MagicMock()

    proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
    with pytest.raises(ResearchProtocolConfigError, match="max_grid_cells"):
        proto.run(
            scenario_config=scenario,
            protocol_config=cfg,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame(),
            rets_full=pd.DataFrame(),
            args=argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            ),
            global_config={},
        )


def test_grid_and_unseen_phases_slice_panels_by_respective_periods(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    captured: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    real_slice = slice_panel_by_dates

    def spy(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        captured.append((start, end))
        return real_slice(df, start, end)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={},
        best_value=1.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.01], index=dates[:1]),
    )
    bt = MagicMock()
    bt.run_backtest_mode.return_value = {"returns": pd.Series([0.0], index=dates[:1])}

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.slice_panel_by_dates",
                side_effect=spy,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    gt = cfg.global_train_period
    ut = cfg.unseen_test_period
    assert len(captured) == 6
    assert all(s == gt.start and e == gt.end for s, e in captured[:3])
    assert all(s == ut.start and e == ut.end for s, e in captured[3:])


def test_artifact_then_unseen_order_reoptimize(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["final_unseen_mode"] = FinalUnseenMode.REOPTIMIZE_WITH_LOCKED_ARCHITECTURE.value
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
            "extras": {"research_protocol": inner},
        }
    )

    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    unseen_ret = OptimizationResult(
        best_parameters={},
        best_value=0.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.0], index=dates[:1]),
    )

    call_order: list[str] = []

    opt = MagicMock()
    _step = {"i": 0}

    def _opt_side_effect(*args, **kwargs):
        _step["i"] += 1
        if _step["i"] <= 2:
            call_order.append(f"opt_grid_{_step['i']}")
            return OptimizationResult(
                best_parameters={"p": 1},
                best_value=1.0,
                n_evaluations=1,
                optimization_history=[],
                stitched_returns=pd.Series([0.01], index=dates[:1]),
            )
        call_order.append("opt_unseen")
        return unseen_ret

    opt.run_optimization.side_effect = _opt_side_effect

    bt = MagicMock()

    def _wg(*a, **k):
        call_order.append("grid_write")

    def _ws(*a, **k):
        call_order.append("sel_write")

    def _wl(*a, **k):
        call_order.append("lock_write")

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 2.0}),
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_grid_results", side_effect=_wg
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_selected_protocols",
                side_effect=_ws,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file", side_effect=_wl),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert call_order.count("opt_unseen") == 1
    ix_grid = call_order.index("grid_write")
    ix_sel = call_order.index("sel_write")
    ix_lock = call_order.index("lock_write")
    ix_unseen = call_order.index("opt_unseen")
    assert ix_grid < ix_sel < ix_lock < ix_unseen


def test_artifact_then_unseen_order_fixed_selected_params(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    call_order: list[str] = []
    step = {"i": 0}

    def _opt_side_effect(*_a: object, **_k: object):
        step["i"] += 1
        call_order.append(f"opt_grid_{step['i']}")
        return OptimizationResult(
            best_parameters={"p": 1},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        )

    opt = MagicMock()
    opt.run_optimization.side_effect = _opt_side_effect
    bt = MagicMock()

    def _bt(*_a: object, **_k: object):
        call_order.append("bt_unseen")
        return {"returns": pd.Series([0.0], index=dates[:1])}

    bt.run_backtest_mode.side_effect = _bt

    def _wg(*_a: object, **_k: object):
        call_order.append("grid_write")

    def _ws(*_a: object, **_k: object):
        call_order.append("sel_write")

    def _wl(*_a: object, **_k: object):
        call_order.append("lock_write")

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 2.0}),
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_grid_results", side_effect=_wg
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_selected_protocols",
                side_effect=_ws,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file", side_effect=_wl),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert opt.run_optimization.call_count == 2
    assert call_order.count("bt_unseen") == 1
    ix_lock = call_order.index("lock_write")
    ix_bt = call_order.index("bt_unseen")
    assert ix_lock < ix_bt


def test_fixed_selected_params_calls_backtest_not_second_optimizer(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()

    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={"theta": 0.42},
        best_value=1.0,
        n_evaluations=2,
        optimization_history=[],
        stitched_returns=pd.Series([0.02], index=dates[:1]),
    )
    bt = MagicMock()
    bt.run_backtest_mode.return_value = {"returns": pd.Series([0.0, 0.0], index=dates[:2])}

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 3.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert opt.run_optimization.call_count == 2
    bt.run_backtest_mode.assert_called_once()


def test_unseen_result_includes_trade_history_from_backtest_payload(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()

    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    th = pd.DataFrame(
        {
            "date": pd.to_datetime([dates[0]]),
            "ticker": ["SPY"],
            "position": [1.0],
        }
    )

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={"theta": 0.42},
        best_value=1.0,
        n_evaluations=2,
        optimization_history=[],
        stitched_returns=pd.Series([0.02], index=dates[:1]),
    )
    bt = MagicMock()
    bt.run_backtest_mode.return_value = {
        "returns": pd.Series([0.0, 0.0], index=dates[:2]),
        "trade_history": th,
    }

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 3.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert r.unseen_result is not None
    assert r.unseen_result.trade_history is not None
    pd.testing.assert_frame_equal(r.unseen_result.trade_history, th)


def test_skip_unseen_returns_none_after_lock(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["final_unseen_mode"] = FinalUnseenMode.REOPTIMIZE_WITH_LOCKED_ARCHITECTURE.value
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": frozendict({}),
            "extras": {"research_protocol": inner},
        }
    )

    dates = _idx(12)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={},
        best_value=0.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.01], index=dates[:1]),
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, MagicMock(), artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results") as wu,
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=pd.DataFrame(),
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert r.unseen_result is None
    wu.assert_not_called()
    assert opt.run_optimization.call_count == 2


def test_reporting_calls_generate_report(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["reporting"] = {"enabled": True}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
            "extras": {"research_protocol": inner},
        }
    )

    dates = _idx(10)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={},
        best_value=0.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.01], index=dates[:1]),
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, MagicMock(), artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 4.0}),
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"
            ) as gr,
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=pd.DataFrame(),
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    gr.assert_called_once()
    c_args = gr.call_args[0]
    assert c_args[2].reporting.enabled is True


def test_force_new_research_run_disables_refuse_overwrite_kwarg(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()
    dates = _idx(8)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={},
        best_value=0.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.01], index=dates[:1]),
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, MagicMock(), artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file") as wl,
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=True,
                research_skip_unseen=True,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=pd.DataFrame(),
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    wl.assert_called_once()
    assert wl.call_args.kwargs.get("refuse_overwrite") is False


def test_normalized_cell_scenario_has_wfo_fields_and_dates(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()
    dates = _idx(12)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt = MagicMock()
    seen: list[CanonicalScenarioConfig] = []

    def _capture(canonical_cell, *_a, **_k):
        seen.append(canonical_cell)
        return OptimizationResult(
            best_parameters={},
            best_value=0.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        )

    opt.run_optimization.side_effect = _capture

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, MagicMock(), artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert len(seen) >= 2
    w = seen[0]
    gt = cfg.global_train_period
    assert w.start_date == pd.Timestamp(gt.start).strftime("%Y-%m-%d")
    assert w.end_date == pd.Timestamp(gt.end).strftime("%Y-%m-%d")
    assert w.wfo_config.get("wfo_mode") == "reoptimize"
    assert w.wfo_config.get("train_window_months") in (12,)


def test_selection_skips_higher_score_when_constraint_fails(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["constraints"] = [{"metric": "Max Drawdown", "min_value": -0.25}]
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = _scenario()

    dates = _idx(8)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"tag": "first"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01, 0.01], index=dates[:2]),
        ),
        OptimizationResult(
            best_parameters={"tag": "second"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01, 0.01], index=dates[:2]),
        ),
    ]
    bt = MagicMock()

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                side_effect=[
                    pd.Series({"Sharpe": 3.0, "Max Drawdown": -0.5, "Calmar": 1.0}),
                    pd.Series({"Sharpe": 1.0, "Max Drawdown": -0.1, "Calmar": 0.5}),
                ],
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    assert r.grid_results[0].constraint_passed is False
    assert r.grid_results[1].constraint_passed is True
    assert r.selected_protocols[0].selected_parameters["tag"] == "second"


def test_all_rows_failing_constraints_raises(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["constraints"] = [{"metric": "Sharpe", "min_value": 10.0}]
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = _scenario()
    dates = _idx(6)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={"tag": "only"},
        best_value=1.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.01], index=dates[:1]),
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, MagicMock(), artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            with pytest.raises(ResearchConstraintError, match="no walk-forward"):
                proto.run(
                    scenario_config=scenario,
                    protocol_config=cfg,
                    monthly_data=pd.DataFrame(),
                    daily_data=daily,
                    rets_full=rets_full,
                    args=args,
                    global_config={"benchmark": "SPY"},
                )


def test_slice_panel_by_dates_tz_aware_index_naive_bounds() -> None:
    """Protocol periods are naive timestamps; market panels may be tz-aware (e.g. US/Eastern)."""
    idx = pd.DatetimeIndex(
        pd.date_range("2018-01-02", periods=5, freq="B", tz="America/New_York"),
    )
    df = pd.DataFrame({"x": range(len(idx))}, index=idx)
    out = slice_panel_by_dates(df, pd.Timestamp("2018-01-02"), pd.Timestamp("2018-01-04"))
    assert len(out) == 3
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None


def test_robust_selection_prefers_plateau_over_isolated_peak_sharpe(
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 13, 14, 15],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 1, "metric": "Sharpe"}
    inner["robust_selection"] = {
        "enabled": True,
        "weights": {"cell": 0.5, "neighbor_median": 0.3, "neighbor_min": 0.2},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _idx(30)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    sharpe_series = [
        pd.Series({"Sharpe": 100.0}),
        pd.Series({"Sharpe": 40.0}),
        pd.Series({"Sharpe": 85.0}),
        pd.Series({"Sharpe": 85.0}),
    ]
    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"tag": "spike"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "a"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "plateau_lo"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "plateau_hi"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
    ]
    bt = MagicMock()
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                side_effect=sharpe_series,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )
    assert r.selected_protocols[0].selected_parameters["tag"] == "plateau_hi"
    assert r.grid_results[0].score > r.grid_results[-1].score


def test_robust_selection_does_not_see_constraint_failed_neighbors_for_smoothing(
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 13],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 1, "metric": "Sharpe"}
    inner["constraints"] = [{"metric": "Max Drawdown", "min_value": -0.05}]
    inner["robust_selection"] = {"enabled": True}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _idx(30)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"tag": "fail"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "solo"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
    ]
    side = [
        pd.Series({"Sharpe": 1000.0, "Max Drawdown": -0.5}),
        pd.Series({"Sharpe": 10.0, "Max Drawdown": -0.01}),
    ]
    bt = MagicMock()
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                side_effect=side,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )
    solo = r.grid_results[1]
    assert r.grid_results[0].constraint_passed is False
    assert solo.constraint_passed is True
    assert solo.robust_score == pytest.approx(solo.score)
    assert r.selected_protocols[0].selected_parameters["tag"] == "solo"


def test_robust_selection_disabled_ranks_by_raw_score(
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 13, 14, 15],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 1, "metric": "Sharpe"}
    inner["robust_selection"] = {"enabled": False}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _idx(30)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    sharpe_series = [
        pd.Series({"Sharpe": 100.0}),
        pd.Series({"Sharpe": 40.0}),
        pd.Series({"Sharpe": 85.0}),
        pd.Series({"Sharpe": 85.0}),
    ]
    opt = MagicMock()
    opt.run_optimization.side_effect = [
        OptimizationResult(
            best_parameters={"tag": "spike"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "a"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "b"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
        OptimizationResult(
            best_parameters={"tag": "c"},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        ),
    ]
    bt = MagicMock()
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                side_effect=sharpe_series,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )
    assert r.selected_protocols[0].selected_parameters["tag"] == "spike"


def test_duplicate_completed_unseen_raises_on_second_full_run_without_force(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={"theta": 0.42},
        best_value=1.0,
        n_evaluations=2,
        optimization_history=[],
        stitched_returns=pd.Series([0.02], index=dates[:1]),
    )
    bt = MagicMock()
    bt.run_backtest_mode.return_value = {"returns": pd.Series([0.0, 0.0], index=dates[:2])}

    args = argparse.Namespace(
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 3.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )
            with pytest.raises(ResearchRegistryError, match="already has unseen_completed"):
                proto.run(
                    scenario_config=scenario,
                    protocol_config=cfg,
                    monthly_data=pd.DataFrame(),
                    daily_data=daily,
                    rets_full=rets_full,
                    args=args,
                    global_config={"benchmark": "SPY"},
                )


def test_duplicate_completed_unseen_allowed_with_force_second_run(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    cfg = parse_double_oos_wfo_protocol(protocol_config_dict)
    scenario = _scenario()
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={"theta": 0.42},
        best_value=1.0,
        n_evaluations=2,
        optimization_history=[],
        stitched_returns=pd.Series([0.02], index=dates[:1]),
    )
    bt = MagicMock()
    bt.run_backtest_mode.return_value = {"returns": pd.Series([0.0, 0.0], index=dates[:2])}

    args1 = argparse.Namespace(
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
    )
    args2 = argparse.Namespace(
        protocol="double_oos_wfo",
        force_new_research_run=True,
        research_skip_unseen=False,
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 3.0}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            r1 = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args1,
                global_config={"benchmark": "SPY"},
            )
            r2 = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args2,
                global_config={"benchmark": "SPY"},
            )
    assert r1.artifact_dir != r2.artifact_dir
    root = iso_writer.scenario_protocol_root(scenario.name)
    reg_text = (root / "registry.yaml").read_text(encoding="utf-8")
    assert reg_text.count("unseen_completed: true") == 2


def test_cost_sensitivity_runs_after_lock_and_baseline_unseen_uses_locked_winner(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["final_unseen_mode"] = FinalUnseenMode.FIXED_SELECTED_PARAMS.value
    inner["cost_sensitivity"] = {
        "enabled": True,
        "slippage_bps_grid": [0.0, 1.0],
        "commission_multiplier_grid": [1.0],
        "run_on": "unseen",
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    call_order: list[str] = []

    def _opt_grid(*_a: object, **_k: object):
        call_order.append("opt_grid")
        return OptimizationResult(
            best_parameters={"p": 1},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        )

    opt = MagicMock()
    opt.run_optimization.side_effect = _opt_grid
    bt = MagicMock()

    def _bt(*_a: object, **_k: object):
        call_order.append("bt_unseen")
        return {"returns": pd.Series([0.0], index=dates[:1])}

    bt.run_backtest_mode.side_effect = _bt

    def _wl(*_a: object, **_k: object):
        call_order.append("lock_write")

    captured_arch: list[tuple[int, int, int, str]] = []

    real_fixed = unseen_scenario_dict_fixed_params

    def _wrap_fixed(sc, period, sel):
        captured_arch.append(
            (
                sel.architecture.train_window_months,
                sel.architecture.test_window_months,
                sel.architecture.wfo_step_months,
                sel.architecture.walk_forward_type,
            )
        )
        return real_fixed(sc, period, sel)

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 2.0, "Total Return": 0.05}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file", side_effect=_wl),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch(
                "portfolio_backtester.research.double_oos_wfo.unseen_scenario_dict_fixed_params",
                _wrap_fixed,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={
                    "benchmark": "SPY",
                    "slippage_bps": 0.0,
                    "commission_per_share": 0.005,
                },
            )

    run_dir = Path(r.artifact_dir)
    assert call_order.index("lock_write") < call_order.index("bt_unseen")
    assert call_order.count("bt_unseen") == 3
    winner = (12, 6, 3, "rolling")
    assert all(a == winner for a in captured_arch)
    assert (run_dir / "cost_sensitivity.csv").is_file()
    assert (run_dir / "cost_sensitivity_summary.yaml").is_file()


def test_bootstrap_runs_after_cost_sensitivity_writes_artifacts(
    protocol_config_dict: dict, iso_writer: ResearchArtifactWriter
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["final_unseen_mode"] = FinalUnseenMode.FIXED_SELECTED_PARAMS.value
    inner["cost_sensitivity"] = {
        "enabled": True,
        "slippage_bps_grid": [0.0],
        "commission_multiplier_grid": [1.0],
        "run_on": "unseen",
    }
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 15,
        "random_seed": 2,
        "random_wfo_architecture": {"enabled": True},
        "block_shuffled_returns": {"enabled": True, "block_size_days": 5},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)
    call_order: list[str] = []

    def _opt_grid(*_a: object, **_k: object):
        return OptimizationResult(
            best_parameters={"p": 1},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        )

    opt = MagicMock()
    opt.run_optimization.side_effect = _opt_grid
    bt = MagicMock()

    def _bt(*_a: object, **_k: object):
        return {"returns": pd.Series([0.002] * 25, index=dates[:25])}

    bt.run_backtest_mode.side_effect = _bt

    from portfolio_backtester.research.bootstrap import write_bootstrap_artifacts as real_bs
    from portfolio_backtester.research.cost_sensitivity import (
        write_cost_sensitivity_artifacts as real_cs,
    )

    def _wrap_cs(*a: Any, **k: Any) -> Any:
        call_order.append("cost_sensitivity_write")
        return real_cs(*a, **k)

    def _wrap_bs(*a: Any, **k: Any) -> Any:
        call_order.append("bootstrap_write")
        return real_bs(*a, **k)

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.5, "Total Return": 0.05}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_cost_sensitivity_artifacts",
                side_effect=_wrap_cs,
            ),
            patch(
                "portfolio_backtester.research.double_oos_wfo.write_bootstrap_artifacts",
                side_effect=_wrap_bs,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY", "slippage_bps": 0.0},
            )

    run_dir = Path(r.artifact_dir)
    assert call_order.index("cost_sensitivity_write") < call_order.index("bootstrap_write")
    assert (run_dir / "bootstrap_significance.csv").is_file()
    assert (run_dir / "bootstrap_summary.yaml").is_file()


def test_run_research_bootstrap_called_with_param_space_when_rsp_enabled(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["final_unseen_mode"] = FinalUnseenMode.FIXED_SELECTED_PARAMS.value
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 8,
        "random_seed": 3,
        "random_wfo_architecture": {"enabled": False},
        "block_shuffled_returns": {"enabled": False, "block_size_days": 5},
        "random_strategy_parameters": {"enabled": True, "sample_size": 6},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"strategy_params_space": {"lag": [1, 2, 3]}},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _wide_idx()
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    def _opt_grid(*_a: object, **_k: object):
        return OptimizationResult(
            best_parameters={"p": 1},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[],
            stitched_returns=pd.Series([0.01], index=dates[:1]),
        )

    opt = MagicMock()
    opt.run_optimization.side_effect = _opt_grid
    bt = MagicMock()

    def _bt(*_a: object, **_k: object):
        return {"returns": pd.Series([0.002] * 25, index=dates[:25])}

    bt.run_backtest_mode.side_effect = _bt
    mock_bootstrap = MagicMock(return_value=None)

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
        with (
            patch(
                "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
                return_value=pd.Series({"Sharpe": 1.5, "Total Return": 0.05}),
            ),
            patch("portfolio_backtester.research.double_oos_wfo.write_grid_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_selected_protocols"),
            patch("portfolio_backtester.research.double_oos_wfo.write_lock_file"),
            patch("portfolio_backtester.research.double_oos_wfo.write_unseen_results"),
            patch("portfolio_backtester.research.double_oos_wfo.write_cost_sensitivity_artifacts"),
            patch(
                "portfolio_backtester.research.double_oos_wfo.run_research_bootstrap",
                mock_bootstrap,
            ),
            patch("portfolio_backtester.research.double_oos_wfo.generate_research_markdown_report"),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=False,
            )
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY", "slippage_bps": 0.0},
            )

    mock_bootstrap.assert_called_once()
    call_kw = mock_bootstrap.call_args.kwargs
    assert call_kw.get("param_space") == {"lag": [1, 2, 3]}
    assert call_kw.get("run_with_params_fn") is not None


def test_run_writes_heatmap_pngs_when_reporting_heatmaps_enabled(
    protocol_config_dict: dict,
    iso_writer: ResearchArtifactWriter,
) -> None:
    inner = protocol_config_dict["research_protocol"].copy()
    inner["reporting"] = {"enabled": True, "generate_heatmaps": True, "heatmap_metrics": ["score"]}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
            "extras": {"research_protocol": inner},
        }
    )
    dates = _idx(10)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)

    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={},
        best_value=0.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=pd.Series([0.01], index=dates[:1]),
    )

    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        proto = DoubleOOSWFOProtocol(opt, MagicMock(), artifact_writer=iso_writer)
        with patch(
            "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
            return_value=pd.Series({"Sharpe": 1.0}),
        ):
            args = argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=False,
                research_skip_unseen=True,
            )
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=pd.DataFrame(),
                rets_full=rets_full,
                args=args,
                global_config={"benchmark": "SPY"},
            )

    run_dir = Path(r.artifact_dir)
    assert (run_dir / "wfo_heatmap_score_step_3_rolling.png").is_file()
