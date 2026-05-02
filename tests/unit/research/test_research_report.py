"""Tests for research markdown reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_backtester.research.protocol_config import (
    RESEARCH_PROTOCOL_ARTIFACT_VERSION,
    ArchitectureLockConfig,
    BlockShuffledPositionsBootstrapConfig,
    BlockShuffledReturnsBootstrapConfig,
    BootstrapConfig,
    CostSensitivityConfig,
    CostSensitivityRunOn,
    DateRangeConfig,
    DoubleOOSWFOProtocolConfig,
    FinalUnseenMode,
    MetricConstraint,
    RandomStrategyParametersBootstrapConfig,
    RandomWfoArchitectureBootstrapConfig,
    ReportingConfig,
    SelectionConfig,
    WFOGridConfig,
    default_bootstrap_config,
)
from portfolio_backtester.research.report import generate_research_markdown_report
from portfolio_backtester.research.scoring import RobustSelectionConfig
from portfolio_backtester.research.results import (
    ResearchProtocolResult,
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)


def _protocol_config() -> DoubleOOSWFOProtocolConfig:
    return DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2021-11-30"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-05-15"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=2, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=False,
            heatmap_metrics=("score", "robust_score"),
        ),
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


def test_generate_research_markdown_report_without_unseen(tmp_path: Path) -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_alpha",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    cfg = _protocol_config()
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "Robust selection" in text
    assert str(RESEARCH_PROTOCOL_ARTIFACT_VERSION) in text
    assert "scenario_alpha" in text
    assert "2019-01-02" in text or "2019-01-02T00:00:00" in text
    assert "2023-05-15" in text or "2023-05-15T00:00:00" in text
    assert "train_window_months" in text or "24" in text
    assert "unseen" in text.lower()
    assert "not run" in text.lower() or "none" in text.lower()
    assert "WFO heatmaps" in text
    assert "disabled" in text.lower()


def test_report_lists_heatmap_files_when_present(tmp_path: Path) -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_hm",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    (tmp_path / "wfo_heatmap_score_step_3_rolling.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2021-11-30"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-05-15"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=2, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=True,
            heatmap_metrics=("score",),
        ),
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
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "WFO heatmaps" in text
    assert "wfo_heatmap_score_step_3_rolling.png" in text


def test_report_includes_metric_constraints_section_when_configured(tmp_path: Path) -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0, "Turnover": 100.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
        constraint_passed=False,
        constraint_failures=("Turnover: 100.0 above maximum 60",),
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
        constraint_passed=True,
        constraint_failures=(),
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_gamma",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2021-11-30"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-05-15"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=2, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=False,
            heatmap_metrics=("score", "robust_score"),
        ),
        constraints=(MetricConstraint(display_key="Turnover", min_value=None, max_value=60.0),),
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
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "Metric constraints" in text
    assert "Turnover" in text


def test_report_includes_cost_sensitivity_section_when_enabled(tmp_path: Path) -> None:
    import yaml

    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_cs",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    summary = {
        "survival_metric": "Calmar",
        "chosen_metrics": ["Calmar", "Total Return"],
        "breakeven_slippage_bps_at_multiplier_1": 5.0,
        "baseline_slippage_bps": 2.5,
    }
    (tmp_path / "cost_sensitivity_summary.yaml").write_text(
        yaml.safe_dump(summary), encoding="utf-8"
    )
    (tmp_path / "cost_sensitivity.csv").write_text(
        "slippage_bps,commission_multiplier,survives,Calmar\n0.0,1.0,True,0.5\n",
        encoding="utf-8",
    )
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2021-11-30"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-05-15"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=2, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=False,
            heatmap_metrics=("score", "robust_score"),
        ),
        constraints=(),
        robust_selection=RobustSelectionConfig(
            enabled=False,
            cell_weight=0.5,
            neighbor_median_weight=0.3,
            neighbor_min_weight=0.2,
        ),
        cost_sensitivity=CostSensitivityConfig(
            enabled=True,
            slippage_bps_grid=(0.0, 5.0),
            commission_multiplier_grid=(1.0,),
            run_on=CostSensitivityRunOn.UNSEEN,
        ),
        bootstrap=default_bootstrap_config(),
    )
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "## Cost sensitivity" in text
    assert "cost_sensitivity.csv" in text
    assert "Calmar" in text


def test_generate_research_markdown_report_with_unseen(tmp_path: Path) -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    idx = pd.date_range("2023-01-01", periods=2, freq="D")
    uv = UnseenValidationResult(
        selected_protocol=sp,
        metrics={"Calmar": 0.7, "Sharpe": 0.3},
        returns=pd.Series([0.0, 0.01], index=idx),
        mode="fixed_selected_params",
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_beta",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=uv,
        artifact_dir=tmp_path,
    )
    cfg = _protocol_config()
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "scenario_beta" in text
    assert str(RESEARCH_PROTOCOL_ARTIFACT_VERSION) in text
    assert "0.7" in text
    assert "Sharpe" in text


def test_report_includes_bootstrap_section_when_enabled(tmp_path: Path) -> None:
    import yaml

    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_bs",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    summary = {
        "n_samples": 200,
        "random_seed": 42,
        "random_wfo_architecture": {"enabled": True, "p_value": 0.12},
        "block_shuffled_returns": {"enabled": True, "p_value": 0.08, "block_size_days": 20},
        "block_shuffled_positions": {"enabled": True, "p_value": 0.03, "block_size_days": 18},
    }
    (tmp_path / "bootstrap_summary.yaml").write_text(yaml.safe_dump(summary), encoding="utf-8")
    (tmp_path / "bootstrap_significance.csv").write_text(
        "test,p_value,n_samples\nrandom_wfo_architecture,0.12,200\n", encoding="utf-8"
    )
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2021-11-30"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-05-15"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=2, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=False,
            heatmap_metrics=("score", "robust_score"),
        ),
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
        bootstrap=BootstrapConfig(
            enabled=True,
            n_samples=200,
            random_seed=42,
            random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=True),
            block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
                enabled=True,
                block_size_days=20,
            ),
            block_shuffled_positions=BlockShuffledPositionsBootstrapConfig(
                enabled=False,
                block_size_days=20,
            ),
            random_strategy_parameters=RandomStrategyParametersBootstrapConfig(
                enabled=False,
                sample_size=100,
            ),
        ),
    )
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "## Bootstrap significance" in text
    assert "bootstrap_significance.csv" in text
    assert "bootstrap_summary.yaml" in text
    assert "0.12" in text
    assert "0.08" in text
    assert "block_shuffled_positions" in text
    assert "0.03" in text


def test_report_bootstrap_section_mentions_random_strategy_parameters_p_value(
    tmp_path: Path,
) -> None:
    import yaml

    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_rsp",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    summary = {
        "n_samples": 200,
        "random_seed": 42,
        "random_wfo_architecture": {"enabled": False, "p_value": None},
        "block_shuffled_returns": {"enabled": False, "p_value": None, "block_size_days": 20},
        "block_shuffled_positions": {"enabled": False, "p_value": None, "block_size_days": 20},
        "random_strategy_parameters": {"enabled": True, "p_value": 0.31, "sample_size": 80},
    }
    (tmp_path / "bootstrap_summary.yaml").write_text(yaml.safe_dump(summary), encoding="utf-8")
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2021-11-30"),
        ),
        unseen_test_period=DateRangeConfig(
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-05-15"),
        ),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(24,),
            test_window_months=(6,),
            wfo_step_months=(3,),
            walk_forward_type=("rolling",),
        ),
        selection=SelectionConfig(top_n=2, metric="Calmar"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=False,
            heatmap_metrics=("score", "robust_score"),
        ),
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
        bootstrap=BootstrapConfig(
            enabled=True,
            n_samples=200,
            random_seed=42,
            random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=False),
            block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
                enabled=False,
                block_size_days=20,
            ),
            block_shuffled_positions=BlockShuffledPositionsBootstrapConfig(
                enabled=False,
                block_size_days=20,
            ),
            random_strategy_parameters=RandomStrategyParametersBootstrapConfig(
                enabled=True,
                sample_size=80,
            ),
        ),
    )
    generate_research_markdown_report(tmp_path, rpr, cfg)
    text = (tmp_path / "research_validation_report.md").read_text(encoding="utf-8")
    assert "random_strategy_parameters" in text
    assert "0.31" in text
