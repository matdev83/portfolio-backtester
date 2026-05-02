"""Contract tests for research result dataclasses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_backtester.research.results import (
    ResearchProtocolResult,
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)


def test_wfo_architecture_frozen_hashable_ordering() -> None:
    a = WFOArchitecture(
        train_window_months=24,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="rolling",
    )
    b = WFOArchitecture(
        train_window_months=24,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="rolling",
    )
    c = WFOArchitecture(
        train_window_months=36,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="rolling",
    )
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a < c


def test_wfo_architecture_dict_round_trip() -> None:
    a = WFOArchitecture(
        train_window_months=24,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="expanding",
    )
    d = a.to_dict()
    assert WFOArchitecture.from_dict(d) == a
    assert set(d.keys()) == {
        "train_window_months",
        "test_window_months",
        "wfo_step_months",
        "walk_forward_type",
    }


def test_dataclass_fields_present() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    stitched = pd.Series([0.01, -0.02, 0.0], index=idx)
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.2, "Max Drawdown": -0.2},
        score=1.2,
        robust_score=None,
        best_parameters={"x": 1},
        n_evaluations=10,
        stitched_returns=stitched,
    )
    assert war.architecture == arch
    assert war.metrics["Calmar"] == 1.2
    assert war.score == 1.2
    assert war.robust_score is None
    assert war.best_parameters == {"x": 1}
    assert war.n_evaluations == 10
    assert war.stitched_returns is not None and len(war.stitched_returns) == 3
    assert war.constraint_passed is True
    assert war.constraint_failures == ()

    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.2},
        score=1.2,
        robust_score=0.5,
        selected_parameters={"x": 1},
    )
    assert sp.constraint_passed is True
    assert sp.constraint_failures == ()

    ridx = pd.date_range("2023-01-01", periods=3, freq="D")
    rets = pd.Series([0.0, 0.01, 0.02], index=ridx)
    uv = UnseenValidationResult(
        selected_protocol=sp,
        metrics={"Calmar": 0.8},
        returns=rets,
        mode="fixed_selected_params",
        trade_history=None,
    )
    assert uv.selected_protocol is sp
    assert uv.trade_history is None

    artifact = Path("artifacts/run1")
    rpr = ResearchProtocolResult(
        scenario_name="s1",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=None,
        artifact_dir=artifact,
    )
    assert rpr.scenario_name == "s1"
    assert rpr.grid_results == [war]
    assert rpr.unseen_result is None

    uv2 = UnseenValidationResult(
        selected_protocol=sp,
        metrics={"Sharpe": 1.1},
        returns=rets,
        mode="reoptimize_with_locked_architecture",
    )
    rpr2 = ResearchProtocolResult(
        scenario_name="s2",
        grid_results=[war],
        selected_protocols=(sp,),
        unseen_result=uv2,
        artifact_dir=artifact,
    )
    assert rpr2.unseen_result is uv2


def test_wfo_architecture_from_dict_bad_walk_forward_raises() -> None:
    with pytest.raises(ValueError):
        WFOArchitecture.from_dict(
            {
                "train_window_months": 1,
                "test_window_months": 1,
                "wfo_step_months": 1,
                "walk_forward_type": "nope",
            }
        )


def test_backward_compat_from_dict_aliases_optional() -> None:
    legacy = {
        "train_months": 12,
        "test_months": 6,
        "step_months": 3,
        "walk_forward_type": "rolling",
    }
    arch = WFOArchitecture.from_dict(legacy)
    assert arch.train_window_months == 12
    assert arch.test_window_months == 6
