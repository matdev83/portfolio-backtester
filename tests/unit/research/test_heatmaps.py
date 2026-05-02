"""Tests for research WFO heatmap artifact writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_backtester.research.heatmaps import write_wfo_heatmaps
from portfolio_backtester.research.protocol_config import (
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
from portfolio_backtester.research.results import WFOArchitecture, WFOArchitectureResult
from portfolio_backtester.research.scoring import RobustSelectionConfig


def _proto(
    *,
    heatmaps: bool,
    metrics: tuple[str, ...] = ("score", "robust_score"),
) -> DoubleOOSWFOProtocolConfig:
    return DoubleOOSWFOProtocolConfig(
        enabled=True,
        global_train_period=DateRangeConfig(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
        unseen_test_period=DateRangeConfig(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
        wfo_window_grid=WFOGridConfig(
            train_window_months=(12, 24),
            test_window_months=(6, 12),
            wfo_step_months=(3, 6),
            walk_forward_type=("rolling", "expanding"),
        ),
        selection=SelectionConfig(top_n=2, metric="Sharpe"),
        composite_scoring=None,
        final_unseen_mode=FinalUnseenMode.FIXED_SELECTED_PARAMS,
        lock=ArchitectureLockConfig(enabled=True, refuse_overwrite=True),
        reporting=ReportingConfig(
            enabled=True,
            generate_heatmaps=heatmaps,
            heatmap_metrics=metrics,
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


def test_write_heatmaps_skips_when_generate_disabled(tmp_path: Path) -> None:
    arch = WFOArchitecture(12, 6, 3, "rolling")
    rows = [
        WFOArchitectureResult(
            architecture=arch,
            metrics={"Sharpe": 1.0},
            score=0.5,
            robust_score=None,
            best_parameters={},
            n_evaluations=1,
        ),
    ]
    cfg = _proto(heatmaps=False)
    out = write_wfo_heatmaps(tmp_path, rows, cfg)
    assert out == ()
    assert list(tmp_path.glob("*.png")) == []


def test_write_heatmaps_creates_stable_names_per_subgroup(tmp_path: Path) -> None:
    rows = [
        WFOArchitectureResult(
            architecture=WFOArchitecture(12, 6, 3, "rolling"),
            metrics={"Calmar": 2.0, "Sharpe": 1.0},
            score=0.1,
            robust_score=0.2,
            best_parameters={},
            n_evaluations=1,
        ),
        WFOArchitectureResult(
            architecture=WFOArchitecture(24, 12, 3, "rolling"),
            metrics={"Calmar": 3.0, "Sharpe": 1.5},
            score=0.3,
            robust_score=0.4,
            best_parameters={},
            n_evaluations=1,
        ),
        WFOArchitectureResult(
            architecture=WFOArchitecture(12, 6, 6, "expanding"),
            metrics={"Calmar": 1.0, "Sharpe": 0.5},
            score=0.05,
            robust_score=0.06,
            best_parameters={},
            n_evaluations=1,
        ),
    ]
    cfg = _proto(
        heatmaps=True,
        metrics=("score", "calmar"),
    )
    written = write_wfo_heatmaps(tmp_path, rows, cfg)
    names = sorted(p.name for p in written)
    assert "wfo_heatmap_score_step_3_rolling.png" in names
    assert "wfo_heatmap_calmar_step_3_rolling.png" in names
    assert "wfo_heatmap_score_step_6_expanding.png" in names
    assert "wfo_heatmap_calmar_step_6_expanding.png" in names
    for p in written:
        assert p.is_file()
        assert p.stat().st_size > 0


def test_write_heatmaps_missing_grid_cells(tmp_path: Path) -> None:
    rows = [
        WFOArchitectureResult(
            architecture=WFOArchitecture(12, 6, 3, "rolling"),
            metrics={"Sharpe": 1.0},
            score=1.0,
            robust_score=None,
            best_parameters={},
            n_evaluations=1,
        ),
    ]
    cfg = _proto(heatmaps=True, metrics=("score",))
    write_wfo_heatmaps(tmp_path, rows, cfg)
    path = tmp_path / "wfo_heatmap_score_step_3_rolling.png"
    assert path.is_file()


def test_max_drawdown_metric_slug_in_filename(tmp_path: Path) -> None:
    rows = [
        WFOArchitectureResult(
            architecture=WFOArchitecture(12, 6, 3, "rolling"),
            metrics={"Max Drawdown": -0.2},
            score=1.0,
            robust_score=None,
            best_parameters={},
            n_evaluations=1,
        ),
    ]
    cfg = _proto(heatmaps=True, metrics=("Max Drawdown",))
    write_wfo_heatmaps(tmp_path, rows, cfg)
    assert (tmp_path / "wfo_heatmap_max_drawdown_step_3_rolling.png").is_file()
