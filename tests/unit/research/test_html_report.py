"""Tests for optional HTML research validation reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_backtester.research.protocol_config import (
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
from portfolio_backtester.research.html_report import generate_research_html_report
from portfolio_backtester.research.results import (
    ResearchProtocolResult,
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)
from portfolio_backtester.research.scoring import RobustSelectionConfig


def _base_protocol_config(
    *,
    bootstrap: BootstrapConfig | None = None,
    **reporting_kw: Any,
) -> DoubleOOSWFOProtocolConfig:
    rep_fields: dict[str, Any] = {
        "enabled": True,
        "generate_heatmaps": False,
        "heatmap_metrics": ("score", "robust_score"),
    }
    rep_fields.update(reporting_kw)
    rep = ReportingConfig(**rep_fields)
    bs = bootstrap if bootstrap is not None else default_bootstrap_config()
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
        reporting=rep,
        constraints=(),
        robust_selection=RobustSelectionConfig(
            enabled=True,
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
        bootstrap=bs,
    )


def test_generate_research_html_report_writes_expected_sections(tmp_path: Path) -> None:
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
        metrics={"Calmar": 0.7},
        returns=pd.Series([0.0, 0.01], index=idx),
        mode="fixed_selected_params",
    )
    rpr = ResearchProtocolResult(
        scenario_name="scenario_html",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=uv,
        artifact_dir=tmp_path,
    )
    cfg = _base_protocol_config()
    generate_research_html_report(tmp_path, rpr, cfg)
    path = tmp_path / "research_validation_report.html"
    assert path.is_file()
    html = path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Research validation report" in html
    assert "double_oos_wfo" in html
    assert "scenario_html" in html
    assert '<nav aria-label="Report sections"' in html
    assert "Global train" in html
    assert "Unseen test" in html
    assert "Robust selection" in html
    assert "enabled" in html.lower()
    assert "Top selected architectures" in html
    assert "<table" in html
    assert "Unseen validation" in html
    assert "0.7" in html
    assert "WFO heatmaps" in html


def test_html_escapes_scenario_name(tmp_path: Path) -> None:
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
        scenario_name="foo<script>",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    cfg = _base_protocol_config()
    generate_research_html_report(tmp_path, rpr, cfg)
    html = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "foo&lt;script&gt;" in html or "&lt;script&gt;" in html


def test_html_lists_heatmap_pngs_when_present(tmp_path: Path) -> None:
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
    (tmp_path / "wfo_heatmap_score_step_3_rolling.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    rpr = ResearchProtocolResult(
        scenario_name="hm_scen",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    cfg = _base_protocol_config(
        generate_heatmaps=True,
        heatmap_metrics=("score",),
    )
    generate_research_html_report(tmp_path, rpr, cfg)
    html = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert "wfo_heatmap_score_step_3_rolling.png" in html


def test_html_includes_constraints_cost_bootstrap_when_applicable(tmp_path: Path) -> None:
    import yaml

    arch = WFOArchitecture(24, 6, 3, "rolling")
    war_fail = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0, "Turnover": 100.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=5,
        constraint_passed=False,
        constraint_failures=("Turnover too high",),
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
        scenario_name="full_sections",
        grid_results=[war_fail],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    (tmp_path / "cost_sensitivity_summary.yaml").write_text(
        yaml.safe_dump(
            {"survival_metric": "Calmar", "breakeven_slippage_bps_at_multiplier_1": 3.0}
        ),
        encoding="utf-8",
    )
    (tmp_path / "bootstrap_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "n_samples": 10,
                "random_seed": 1,
                "random_wfo_architecture": {"enabled": True, "p_value": 0.2},
            }
        ),
        encoding="utf-8",
    )
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(pd.Timestamp("2019-01-02"), pd.Timestamp("2021-11-30")),
        unseen_test_period=DateRangeConfig(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-05-15")),
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
        reporting=ReportingConfig(enabled=True, generate_heatmaps=False),
        constraints=(MetricConstraint(display_key="Turnover", min_value=None, max_value=60.0),),
        robust_selection=RobustSelectionConfig(
            enabled=False,
            cell_weight=0.5,
            neighbor_median_weight=0.3,
            neighbor_min_weight=0.2,
        ),
        cost_sensitivity=CostSensitivityConfig(
            enabled=True,
            slippage_bps_grid=(0.0,),
            commission_multiplier_grid=(1.0,),
            run_on=CostSensitivityRunOn.UNSEEN,
        ),
        bootstrap=BootstrapConfig(
            enabled=True,
            n_samples=10,
            random_seed=1,
            random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=True),
            block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
                enabled=False, block_size_days=20
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
    generate_research_html_report(tmp_path, rpr, cfg)
    html = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert "Metric constraints" in html
    assert "Cost sensitivity" in html
    assert "Bootstrap significance" in html


def test_html_includes_cross_validation_when_yaml_present(tmp_path: Path) -> None:
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
        scenario_name="cv_scen",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    blob = {
        "enabled": True,
        "strategy": "blocked_global_train_period",
        "n_folds": 2,
        "fold_periods": [{"start": "2019-01-01T00:00:00", "end": "2020-01-01T00:00:00"}],
    }
    (tmp_path / "cross_validation_summary.yaml").write_text(yaml.safe_dump(blob), encoding="utf-8")
    cfg = _base_protocol_config()
    generate_research_html_report(tmp_path, rpr, cfg)
    html_text = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert 'id="cross-validation"' in html_text


def test_html_generates_and_embeds_bootstrap_viz_pngs_when_configured(tmp_path: Path) -> None:
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
        scenario_name="boot_plots",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    pd.DataFrame({"value": [0.08 * float(i) for i in range(1, 10)]}).to_csv(
        tmp_path / "bootstrap_distribution_random_wfo_architecture.csv",
        index=False,
    )
    cfg = _base_protocol_config(
        generate_bootstrap_distribution_plots=True,
        html_embed_figures=True,
        bootstrap=BootstrapConfig(
            enabled=True,
            n_samples=10,
            random_seed=5,
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
                enabled=False,
                sample_size=100,
            ),
        ),
    )
    generate_research_html_report(tmp_path, rpr, cfg)
    png_hist = tmp_path / "bootstrap_viz_random_wfo_architecture_histogram.png"
    assert png_hist.is_file()
    html_text = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert png_hist.name in html_text
    assert "<img" in html_text


def test_html_lists_bootstrap_viz_pngs_when_embed_disabled(tmp_path: Path) -> None:
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
        scenario_name="boot_plots_links",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    pd.DataFrame({"value": [0.08 * float(i) for i in range(1, 10)]}).to_csv(
        tmp_path / "bootstrap_distribution_block_shuffled_returns.csv",
        index=False,
    )
    cfg = _base_protocol_config(
        generate_bootstrap_distribution_plots=True,
        html_embed_figures=False,
        bootstrap=BootstrapConfig(
            enabled=True,
            n_samples=10,
            random_seed=5,
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
                enabled=False,
                sample_size=100,
            ),
        ),
    )
    generate_research_html_report(tmp_path, rpr, cfg)
    png_hist = tmp_path / "bootstrap_viz_block_shuffled_returns_histogram.png"
    png_qq = tmp_path / "bootstrap_viz_block_shuffled_returns_qq.png"
    assert png_hist.is_file()
    assert png_qq.is_file()
    html_text = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert f'href="{png_hist.name}"' in html_text
    assert f'href="{png_qq.name}"' in html_text
    assert "<figure>" not in html_text


def test_html_links_preseeded_bootstrap_viz_pngs_when_generation_off_embed_off(
    tmp_path: Path,
) -> None:
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
        scenario_name="boot_preseed",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    (tmp_path / "bootstrap_viz_custom_slug_histogram.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "bootstrap_viz_custom_slug_qq.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cfg = _base_protocol_config(
        generate_bootstrap_distribution_plots=False,
        html_embed_figures=False,
        bootstrap=BootstrapConfig(
            enabled=True,
            n_samples=10,
            random_seed=5,
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
                enabled=False,
                sample_size=100,
            ),
        ),
    )
    generate_research_html_report(tmp_path, rpr, cfg)
    html_text = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert 'id="bootstrap"' in html_text
    assert 'href="bootstrap_viz_custom_slug_histogram.png"' in html_text
    assert 'href="bootstrap_viz_custom_slug_qq.png"' in html_text
    assert "<img " not in html_text


def test_html_links_cost_sensitivity_png_when_embed_disabled(tmp_path: Path) -> None:
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
        scenario_name="cost_fig_link",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    rows = [
        {
            "slippage_bps": 0.0,
            "commission_multiplier": 1.0,
            "Calmar": 1.1,
        },
        {
            "slippage_bps": 5.0,
            "commission_multiplier": 1.0,
            "Calmar": 0.9,
        },
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "cost_sensitivity.csv", index=False)
    (tmp_path / "cost_sensitivity_summary.yaml").write_text(
        yaml.safe_dump({"survival_metric": "Calmar"}),
        encoding="utf-8",
    )
    cfg = DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(pd.Timestamp("2019-01-02"), pd.Timestamp("2021-11-30")),
        unseen_test_period=DateRangeConfig(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-05-15")),
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
            generate_html=True,
            html_embed_figures=False,
            generate_cost_sensitivity_figure=True,
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
    generate_research_html_report(tmp_path, rpr, cfg)
    png = tmp_path / "cost_sensitivity_survival_curve.png"
    assert png.is_file()
    html_text = (tmp_path / "research_validation_report.html").read_text(encoding="utf-8")
    assert f'href="{png.name}"' in html_text
    assert "Cost sensitivity figure:" in html_text
    assert '<img alt="Cost sensitivity survival curve"' not in html_text
