"""Tests for research artifact writers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.research.artifacts import (
    ResearchArtifactExistsError,
    ResearchArtifactWriter,
    protocol_config_to_plain,
    write_grid_results,
    write_lock_file,
    write_selected_protocols,
    write_unseen_results,
)
from portfolio_backtester.research.protocol_config import (
    RESEARCH_PROTOCOL_ARTIFACT_VERSION,
    ArchitectureLockConfig,
    CostSensitivityConfig,
    CostSensitivityRunOn,
    DateRangeConfig,
    DoubleOOSWFOProtocolConfig,
    FinalUnseenMode,
    ReportingConfig,
    SelectionConfig,
    WFOGridConfig,
    default_bootstrap_config,
)
from portfolio_backtester.research.scoring import RobustSelectionConfig
from portfolio_backtester.research.results import (
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)


def _minimal_protocol_config() -> DoubleOOSWFOProtocolConfig:
    return DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2021-12-31"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2023-06-30"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=3, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(enabled=True),
        constraints=(),
        robust_selection=RobustSelectionConfig(
            enabled=False,
            cell_weight=0.5,
            neighbor_median_weight=0.3,
            neighbor_min_weight=0.2,
        ),
        cost_sensitivity=CostSensitivityConfig(
            enabled=False,
            slippage_bps_grid=(),
            commission_multiplier_grid=(1.0,),
            run_on=CostSensitivityRunOn.UNSEEN,
        ),
        bootstrap=default_bootstrap_config(),
    )


def test_protocol_config_to_plain_includes_robust_selection() -> None:
    proto = _minimal_protocol_config()
    plain = protocol_config_to_plain(proto)
    assert "robust_selection" in plain
    assert plain["robust_selection"]["enabled"] is False
    assert plain["reporting"]["generate_heatmaps"] is False
    assert plain["reporting"]["generate_html"] is False
    assert plain["reporting"]["heatmap_metrics"] == ["score", "robust_score"]
    assert plain["cost_sensitivity"]["enabled"] is False
    assert plain["cost_sensitivity"]["run_on"] == "unseen"
    assert plain["bootstrap"]["enabled"] is False
    assert plain["bootstrap"]["n_samples"] == 200
    assert plain["bootstrap"]["random_seed"] == 42
    rsp = plain["bootstrap"]["random_strategy_parameters"]
    assert rsp["enabled"] is False
    assert rsp["sample_size"] == 100
    bpos = plain["bootstrap"]["block_shuffled_positions"]
    assert bpos["enabled"] is False
    assert bpos["block_size_days"] == 20
    assert plain["execution"]["max_grid_cells"] == 100
    assert plain["execution"]["fail_fast"] is True


def test_scenario_protocol_root_matches_run_parent(tmp_path: Path) -> None:
    w = ResearchArtifactWriter(tmp_path)
    root = w.scenario_protocol_root("My Scenario!/v1")
    run_dir = w.create_run_directory("My Scenario!/v1", run_id="rid42")
    assert run_dir.parent == root


def test_writer_creates_expected_run_directory(tmp_path: Path) -> None:
    w = ResearchArtifactWriter(tmp_path)
    run_dir = w.create_run_directory("My Scenario!/v1", run_id="rid42")
    assert run_dir == tmp_path / "My_Scenario_v1" / "research_protocol" / "rid42"
    assert run_dir.is_dir()


def test_writer_timestamped_run_id_unique(tmp_path: Path) -> None:
    w = ResearchArtifactWriter(tmp_path)
    a = w.create_run_directory("s")
    b = w.create_run_directory("s")
    assert a != b
    assert a.parent == b.parent


def test_write_grid_results_columns(tmp_path: Path) -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    rows = [
        WFOArchitectureResult(
            architecture=arch,
            metrics={"Calmar": 1.1, "Sharpe": 0.5},
            score=1.1,
            robust_score=0.9,
            best_parameters={"p": 2},
            n_evaluations=100,
        ),
    ]
    write_grid_results(tmp_path, rows)
    path = tmp_path / "wfo_architecture_grid.csv"
    assert path.is_file()
    df = pd.read_csv(path, keep_default_na=False)
    expected_cols = {
        "train_window_months",
        "test_window_months",
        "wfo_step_months",
        "walk_forward_type",
        "score",
        "robust_score",
        "n_evaluations",
        "constraint_passed",
        "constraint_failures",
        "best_parameters_json",
        "Calmar",
        "Sharpe",
    }
    assert expected_cols.issubset(set(df.columns))
    assert str(df.iloc[0]["constraint_passed"]).lower() == "true"
    assert df.iloc[0]["constraint_failures"] == ""
    assert df.iloc[0]["best_parameters_json"] == json.dumps({"p": 2}, sort_keys=True)


def test_write_selected_protocols_yaml(tmp_path: Path) -> None:
    arch = WFOArchitecture(12, 3, 1, "expanding")
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 2.0},
        score=2.0,
        robust_score=None,
        selected_parameters={"a": 1},
    )
    write_selected_protocols(tmp_path, [sp])
    loaded = yaml.safe_load((tmp_path / "selected_protocols.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, list) and len(loaded) == 1
    row = loaded[0]
    assert row["rank"] == 1
    assert row["score"] == 2.0
    assert row["robust_score"] is None
    assert row["architecture"] == arch.to_dict()
    assert row["selected_parameters"] == {"a": 1}
    assert row["metrics"] == {"Calmar": 2.0}
    assert row["constraint_passed"] is True
    assert row["constraint_failures"] == []


def test_write_lock_file_refuse_overwrite(tmp_path: Path) -> None:
    scenario = CanonicalScenarioConfig(name="scen", strategy="st")
    proto = _minimal_protocol_config()
    arch = WFOArchitecture(24, 6, 3, "rolling")
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"x": 1},
    )
    global_cfg = {"optimizer": {"trials": 10}}
    write_lock_file(tmp_path, scenario, global_cfg, proto, sp, refuse_overwrite=True)
    lock_path = tmp_path / "protocol_lock.yaml"
    assert lock_path.is_file()
    with pytest.raises(ResearchArtifactExistsError):
        write_lock_file(tmp_path, scenario, global_cfg, proto, sp, refuse_overwrite=True)


def test_protocol_lock_yaml_includes_protocol_version_and_final_unseen_mode(
    tmp_path: Path,
) -> None:
    scenario = CanonicalScenarioConfig(name="scen", strategy="st")
    proto = _minimal_protocol_config()
    arch = WFOArchitecture(24, 6, 3, "rolling")
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={},
        score=1.0,
        robust_score=None,
        selected_parameters={"x": 1},
    )
    write_lock_file(tmp_path, scenario, {}, proto, sp, refuse_overwrite=True)
    data = yaml.safe_load((tmp_path / "protocol_lock.yaml").read_text(encoding="utf-8"))
    assert data["protocol_version"] == RESEARCH_PROTOCOL_ARTIFACT_VERSION
    assert data["final_unseen_mode"] == proto.final_unseen_mode.value


def test_write_lock_file_allow_overwrite(tmp_path: Path) -> None:
    scenario = CanonicalScenarioConfig(name="scen", strategy="st")
    proto = _minimal_protocol_config()
    arch = WFOArchitecture(24, 6, 3, "rolling")
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={},
        score=1.0,
        robust_score=None,
        selected_parameters={"x": 1},
    )
    write_lock_file(tmp_path, scenario, {}, proto, sp, refuse_overwrite=True)
    sp2 = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={},
        score=2.0,
        robust_score=None,
        selected_parameters={"x": 2},
    )
    write_lock_file(tmp_path, scenario, {}, proto, sp2, refuse_overwrite=False)
    data = yaml.safe_load((tmp_path / "protocol_lock.yaml").read_text(encoding="utf-8"))
    assert data["selected_parameters"] == {"x": 2}


def test_write_unseen_results(tmp_path: Path) -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={},
        score=1.0,
        robust_score=None,
        selected_parameters={},
    )
    idx = pd.date_range("2023-01-01", periods=2, freq="D")
    rets = pd.Series([0.01, -0.005], index=idx)
    uv = UnseenValidationResult(
        selected_protocol=sp,
        metrics={"Calmar": 0.5, "Sharpe": 0.1},
        returns=rets,
        mode="fixed_selected_params",
    )
    write_unseen_results(tmp_path, uv)
    df = pd.read_csv(tmp_path / "unseen_test_returns.csv", index_col=0, parse_dates=True)
    assert len(df) == 2
    m = yaml.safe_load((tmp_path / "unseen_test_metrics.yaml").read_text(encoding="utf-8"))
    assert m["mode"] == "fixed_selected_params"
    assert m["metrics"]["Calmar"] == 0.5
    assert m["metrics"]["Sharpe"] == 0.1
