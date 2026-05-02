"""Phase 2 research execution: parallelism, resume checkpoints, cross-validation."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.optimization.results import OptimizationResult
from portfolio_backtester.research.artifacts import (
    ResearchArtifactWriter,
    protocol_config_to_plain,
)
from portfolio_backtester.research.double_oos_wfo import (
    DoubleOOSWFOProtocol,
    expand_wfo_architecture_grid,
)
from portfolio_backtester.research.execution_manifest import (
    RESUME_ARCHITECTURE_ORDER_MISMATCH_MESSAGE,
    RESUME_PROTOCOL_CONFIG_HASH_MISMATCH_MESSAGE,
    RESUME_SCENARIO_HASH_MISMATCH_MESSAGE,
    load_execution_manifest_or_raise,
    split_global_train_blocked_folds,
    write_execution_manifest,
)
from portfolio_backtester.research.registry import compute_registry_hashes
from portfolio_backtester.research.protocol_config import (
    CrossValidationConfig,
    ResearchProtocolConfigError,
    ResumePartialRunConfig,
    parse_double_oos_wfo_protocol,
)
from portfolio_backtester.research.checkpoint_io import (
    ARCH_CHECKPOINT_SUBDIR_NAME,
    write_grid_cell_checkpoint,
)

from tests.unit.research.test_protocol_config import _minimal_primary_inner


def _phase_inner_two_cell() -> dict:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [12, 18],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    return inner


def _scenario_from_inner(inner: dict) -> CanonicalScenarioConfig:
    return CanonicalScenarioConfig.from_dict(
        {
            "name": "p_scen",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        }
    )


def _idx(n: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_execution_phase2_defaults() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    assert cfg.execution.max_parallel_grid_workers == 1
    assert cfg.execution.resume_partial == ResumePartialRunConfig()
    assert cfg.cross_validation == CrossValidationConfig()


def test_execution_parallel_and_resume_parse() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {
        "max_parallel_grid_workers": 4,
        "resume_partial": {"enabled": True, "run_directory": "/tmp/partial_run"},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.execution.max_parallel_grid_workers == 4
    assert cfg.execution.resume_partial.enabled is True
    assert cfg.execution.resume_partial.run_directory == "/tmp/partial_run"


def test_execution_resume_requires_directory_when_enabled() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {"resume_partial": {"enabled": True}}
    with pytest.raises(ResearchProtocolConfigError, match="run_directory"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})

    inner["execution"] = {"resume_partial": {"enabled": False, "run_directory": ""}}
    parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_cross_validation_parse() -> None:
    inner = _minimal_primary_inner()
    inner["cross_validation"] = {
        "enabled": True,
        "n_folds": 3,
        "strategy": "blocked_global_train_period",
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    assert cfg.cross_validation.enabled is True
    assert cfg.cross_validation.n_folds == 3


def test_cross_validation_n_folds_invalid() -> None:
    inner = _minimal_primary_inner()
    inner["cross_validation"] = {"enabled": True, "n_folds": 1}
    with pytest.raises(ResearchProtocolConfigError, match="n_folds"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})

    inner["cross_validation"] = {"enabled": True, "n_folds": 3, "strategy": "invalid"}
    with pytest.raises(ResearchProtocolConfigError, match="strategy"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_protocol_config_to_plain_includes_phase2_execution() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {
        "max_parallel_grid_workers": 2,
        "resume_partial": {"enabled": True, "run_directory": "/x"},
    }
    inner["cross_validation"] = {"enabled": True, "n_folds": 4}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    plain = protocol_config_to_plain(cfg)
    assert plain["cross_validation"]["enabled"] is True
    assert plain["cross_validation"]["n_folds"] == 4
    assert plain["execution"]["max_parallel_grid_workers"] == 2
    assert "resume_partial" not in plain["execution"]


def test_split_blocked_folds_three_partitions() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    folds = split_global_train_blocked_folds(cfg.global_train_period, 3)
    assert len(folds) == 3
    assert folds[0].start <= folds[0].end < folds[1].start <= folds[1].end < folds[2].start


def test_load_manifest_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ResearchProtocolConfigError, match="manifest"):
        load_execution_manifest_or_raise(tmp_path)


def test_resume_partial_with_cross_validation_raises() -> None:
    inner = _minimal_primary_inner()
    inner["execution"] = {
        "resume_partial": {"enabled": True, "run_directory": "/tmp/never_used"},
    }
    inner["cross_validation"] = {"enabled": True, "n_folds": 2}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = _scenario_from_inner(inner)
    opt = MagicMock()
    bt = MagicMock()
    dates = _idx(20)
    with pytest.raises(ResearchProtocolConfigError, match="incompatible"):
        DoubleOOSWFOProtocol(opt, bt, artifact_writer=ResearchArtifactWriter(Path("."))).run(
            scenario_config=scenario,
            protocol_config=cfg,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame({"SPY": 1.0}, index=dates),
            rets_full=pd.DataFrame({"SPY": 0.0}, index=dates),
            args=argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=True,
                research_skip_unseen=True,
            ),
            global_config={"benchmark": "SPY"},
        )


def test_resume_protocol_config_hash_mismatch_raises(tmp_path: Path) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [24],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    run_root = tmp_path / "resume_run"
    run_root.mkdir(parents=True, exist_ok=True)
    inner["execution"] = {"resume_partial": {"enabled": True, "run_directory": str(run_root)}}
    scenario = _scenario_from_inner(inner)
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    arch_list = expand_wfo_architecture_grid(grid=cfg.wfo_window_grid)
    sh, _, _ = compute_registry_hashes(scenario, cfg)
    write_execution_manifest(
        run_root,
        scenario_hash=sh,
        protocol_config_hash="wrong_pch",
        architectures=arch_list,
    )
    opt = MagicMock()
    bt = MagicMock()
    with pytest.raises(
        ResearchProtocolConfigError, match=RESUME_PROTOCOL_CONFIG_HASH_MISMATCH_MESSAGE
    ):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=ResearchArtifactWriter(tmp_path))
        proto.run(
            scenario_config=scenario,
            protocol_config=cfg,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame({"SPY": 1.0}, index=_idx()),
            rets_full=pd.DataFrame({"SPY": 0.0}, index=_idx()),
            args=argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=True,
                research_skip_unseen=True,
            ),
            global_config={"benchmark": "SPY"},
        )


def test_resume_architecture_order_mismatch_raises(tmp_path: Path) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [24],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    run_root = tmp_path / "resume_run"
    run_root.mkdir(parents=True, exist_ok=True)
    inner["execution"] = {"resume_partial": {"enabled": True, "run_directory": str(run_root)}}
    scenario = _scenario_from_inner(inner)
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    sh, pch, _ = compute_registry_hashes(scenario, cfg)
    other_inner = inner.copy()
    other_inner["wfo_window_grid"] = {
        "train_window_months": [18, 24],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    other_cfg = parse_double_oos_wfo_protocol({"research_protocol": other_inner})
    other_arch = expand_wfo_architecture_grid(grid=other_cfg.wfo_window_grid)
    write_execution_manifest(
        run_root, scenario_hash=sh, protocol_config_hash=pch, architectures=other_arch
    )
    opt = MagicMock()
    bt = MagicMock()
    with pytest.raises(
        ResearchProtocolConfigError, match=RESUME_ARCHITECTURE_ORDER_MISMATCH_MESSAGE
    ):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=ResearchArtifactWriter(tmp_path))
        proto.run(
            scenario_config=scenario,
            protocol_config=cfg,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame({"SPY": 1.0}, index=_idx()),
            rets_full=pd.DataFrame({"SPY": 0.0}, index=_idx()),
            args=argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=True,
                research_skip_unseen=True,
            ),
            global_config={"benchmark": "SPY"},
        )


def test_resume_scenario_hash_mismatch_raises(tmp_path: Path) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [24],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 2, "metric": "Sharpe"}
    run_root = tmp_path / "resume_run"
    run_root.mkdir(parents=True, exist_ok=True)
    inner["execution"] = {"resume_partial": {"enabled": True, "run_directory": str(run_root)}}
    scenario = _scenario_from_inner(inner)
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    arch_list = expand_wfo_architecture_grid(grid=cfg.wfo_window_grid)
    _, pch, _ = compute_registry_hashes(scenario, cfg)
    write_execution_manifest(
        run_root, scenario_hash="wrong_hash", protocol_config_hash=pch, architectures=arch_list
    )
    opt = MagicMock()
    bt = MagicMock()
    with pytest.raises(ResearchProtocolConfigError, match=RESUME_SCENARIO_HASH_MISMATCH_MESSAGE):
        proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=ResearchArtifactWriter(tmp_path))
        proto.run(
            scenario_config=scenario,
            protocol_config=cfg,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame({"SPY": 1.0}, index=_idx()),
            rets_full=pd.DataFrame({"SPY": 0.0}, index=_idx()),
            args=argparse.Namespace(
                protocol="double_oos_wfo",
                force_new_research_run=True,
                research_skip_unseen=True,
            ),
            global_config={"benchmark": "SPY"},
        )


def test_parallel_grid_respects_worker_cap(tmp_path: Path) -> None:
    inner = _phase_inner_two_cell()
    inner["execution"] = {"max_parallel_grid_workers": 2}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = _scenario_from_inner(inner)
    dates = _idx(40)
    rets_full = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    concurrency_lock = threading.Lock()
    concurrent = {"n": 0, "peak": 0}

    def track_and_return(delay: float, ret: OptimizationResult) -> OptimizationResult:
        with concurrency_lock:
            concurrent["n"] += 1
            concurrent["peak"] = max(concurrent["peak"], concurrent["n"])
        threading.Event().wait(delay)
        with concurrency_lock:
            concurrent["n"] -= 1
        return ret

    s = pd.Series([0.01, 0.02], index=dates[:2])
    tmpl = OptimizationResult(
        best_parameters={"a": 1},
        best_value=1.0,
        n_evaluations=1,
        optimization_history=[],
        stitched_returns=s,
    )

    opt = MagicMock()

    def side_effect(*_a: object, **_k: object) -> OptimizationResult:
        return track_and_return(0.05, tmpl)

    opt.run_optimization.side_effect = side_effect
    bt = MagicMock()
    iso_writer = ResearchArtifactWriter(tmp_path)
    proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        with patch(
            "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
            return_value=pd.Series({"Sharpe": 1.2}),
        ):
            proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets_full,
                args=argparse.Namespace(
                    protocol="double_oos_wfo",
                    force_new_research_run=True,
                    research_skip_unseen=True,
                ),
                global_config={"benchmark": "SPY"},
            )
    assert opt.run_optimization.call_count == 2
    assert concurrent["peak"] <= 2


def test_resume_second_cell_only_runs_missing(tmp_path: Path) -> None:
    inner_run = _phase_inner_two_cell()
    cfg_first = parse_double_oos_wfo_protocol({"research_protocol": inner_run})
    scenario_first = _scenario_from_inner(inner_run)
    arch_list = expand_wfo_architecture_grid(grid=cfg_first.wfo_window_grid)
    assert len(arch_list) == 2
    sh, pch, _ = compute_registry_hashes(scenario_first, cfg_first)
    run_root = tmp_path / "partial"
    run_root.mkdir(parents=True, exist_ok=True)
    write_execution_manifest(
        run_root, scenario_hash=sh, protocol_config_hash=pch, architectures=arch_list
    )
    chk_dir = run_root / ARCH_CHECKPOINT_SUBDIR_NAME
    chk_dir.mkdir(parents=True, exist_ok=True)
    first_arch = arch_list[0]
    write_grid_cell_checkpoint(
        chk_dir,
        architecture=first_arch,
        checkpoint_body={
            "metrics": {"Sharpe": 9.9},
            "score": 9.9,
            "robust_score": None,
            "best_parameters": {"a": 1},
            "n_evaluations": 42,
            "constraint_passed": True,
            "constraint_failures": (),
        },
    )

    dates = _idx(20)
    rets = pd.DataFrame({"SPY": 0.0}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    inner_resume = inner_run.copy()
    inner_resume["execution"] = {
        "resume_partial": {"enabled": True, "run_directory": str(run_root)},
    }
    cfg_resume = parse_double_oos_wfo_protocol({"research_protocol": inner_resume})
    scenario_resume = scenario_first

    s2 = pd.Series([0.02], index=dates[:1])
    opt = MagicMock()
    opt.run_optimization.return_value = OptimizationResult(
        best_parameters={"b": 2},
        best_value=1.0,
        n_evaluations=99,
        optimization_history=[],
        stitched_returns=s2,
    )
    bt = MagicMock()
    writer = ResearchArtifactWriter(tmp_path)
    proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=writer)
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        with patch(
            "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
            return_value=pd.Series({"Sharpe": 1.0}),
        ):
            r = proto.run(
                scenario_config=scenario_resume,
                protocol_config=cfg_resume,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets,
                args=argparse.Namespace(
                    protocol="double_oos_wfo",
                    force_new_research_run=True,
                    research_skip_unseen=True,
                ),
                global_config={"benchmark": "SPY"},
            )
    assert opt.run_optimization.call_count == 1
    assert len(r.grid_results) == 2
    assert r.grid_results[0].metrics["Sharpe"] == pytest.approx(9.9)
    assert r.grid_results[1].metrics["Sharpe"] == pytest.approx(1.0)


def test_cross_validation_averages_sharpe_two_folds(tmp_path: Path) -> None:
    inner = _minimal_primary_inner()
    inner["wfo_window_grid"] = {
        "train_window_months": [24],
        "test_window_months": [6],
        "wfo_step_months": [3],
        "walk_forward_type": ["rolling"],
    }
    inner["selection"] = {"top_n": 1, "metric": "Sharpe"}
    inner["cross_validation"] = {
        "enabled": True,
        "n_folds": 2,
        "strategy": "blocked_global_train_period",
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    scenario = _scenario_from_inner(inner)
    dates = pd.bdate_range("2020-01-01", "2025-06-01")
    rets = pd.DataFrame({"SPY": 0.001}, index=dates)
    daily = pd.DataFrame({"SPY": 100.0}, index=dates)

    opt_counts = {"n": 0}

    def opt_side_eff(*args: object, **_k: object) -> OptimizationResult:
        opt_counts["n"] += 1
        ix = slice((opt_counts["n"] - 1) * 10, opt_counts["n"] * 10)
        sub_ix = rets.index[ix]
        vals = [0.01 + 0.001 * j for j in range(len(sub_ix))]
        return OptimizationResult(
            best_parameters={"fold": opt_counts["n"]},
            best_value=1.0,
            n_evaluations=10,
            optimization_history=[],
            stitched_returns=pd.Series(vals, index=sub_ix),
        )

    opt = MagicMock()
    opt.run_optimization.side_effect = opt_side_eff
    bt = MagicMock()
    metrics_seq = iter(
        (
            pd.Series({"Sharpe": 4.0}),
            pd.Series({"Sharpe": 2.0}),
        )
    )

    iso_writer = ResearchArtifactWriter(tmp_path)
    proto = DoubleOOSWFOProtocol(opt, bt, artifact_writer=iso_writer)
    with patch.object(DoubleOOSWFOProtocol, "_effective_benchmark_ticker", return_value="SPY"):
        with patch(
            "portfolio_backtester.research.double_oos_wfo.calculate_metrics",
            side_effect=lambda *_a, **_k: next(metrics_seq),
        ):
            r = proto.run(
                scenario_config=scenario,
                protocol_config=cfg,
                monthly_data=pd.DataFrame(),
                daily_data=daily,
                rets_full=rets,
                args=argparse.Namespace(
                    protocol="double_oos_wfo",
                    force_new_research_run=True,
                    research_skip_unseen=True,
                ),
                global_config={"benchmark": "SPY"},
            )
    assert opt.run_optimization.call_count == 2
    assert len(r.grid_results) == 1
    assert r.grid_results[0].metrics["Sharpe"] == pytest.approx(3.0)
    assert r.cross_validation_summary is not None
    assert r.cross_validation_summary["n_folds"] == 2
    assert (r.artifact_dir / "cross_validation_summary.yaml").is_file()
