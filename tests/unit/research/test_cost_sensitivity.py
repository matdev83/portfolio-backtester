"""Unit tests for research cost sensitivity helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from portfolio_backtester.research.cost_sensitivity import (
    breakeven_slippage_bps_at_multiplier,
    build_cost_sensitivity_summary,
    effective_global_config_for_cost_cell,
    expand_cost_sensitivity_grid,
    row_survives,
    survival_metric_for_selection,
    write_cost_sensitivity_artifacts,
)
from portfolio_backtester.research.protocol_config import (
    ROBUST_COMPOSITE_METRIC_NAME,
    CostSensitivityConfig,
    CostSensitivityRunOn,
)


def test_expand_grid_order_and_disabled() -> None:
    off = CostSensitivityConfig(
        enabled=False,
        slippage_bps_grid=(),
        commission_multiplier_grid=(1.0,),
        run_on=CostSensitivityRunOn.UNSEEN,
    )
    assert expand_cost_sensitivity_grid(off) == ()
    on = CostSensitivityConfig(
        enabled=True,
        slippage_bps_grid=(0.0, 5.0),
        commission_multiplier_grid=(1.0, 2.0),
        run_on=CostSensitivityRunOn.UNSEEN,
    )
    assert expand_cost_sensitivity_grid(on) == (
        (0.0, 1.0),
        (0.0, 2.0),
        (5.0, 1.0),
        (5.0, 2.0),
    )


def test_effective_global_sets_slippage_and_scales_commissions_when_present() -> None:
    base = {
        "benchmark": "SPY",
        "slippage_bps": 2.5,
        "commission_per_share": 0.01,
        "commission_min_per_order": 2.0,
        "commission_max_percent_of_trade": 0.004,
    }
    out = effective_global_config_for_cost_cell(
        base, slippage_bps=10.0, commission_multiplier=3.0
    )
    assert out["slippage_bps"] == pytest.approx(10.0)
    assert out["commission_per_share"] == pytest.approx(0.03)
    assert out["commission_min_per_order"] == pytest.approx(6.0)
    assert out["commission_max_percent_of_trade"] == pytest.approx(0.012)
    assert out["benchmark"] == "SPY"


def test_effective_global_omits_commission_scaling_when_keys_absent() -> None:
    base = {"benchmark": "SPY", "slippage_bps": 1.0}
    out = effective_global_config_for_cost_cell(base, slippage_bps=0.0, commission_multiplier=2.0)
    assert out == {"benchmark": "SPY", "slippage_bps": 0.0}
    assert "commission_per_share" not in out


def test_survival_metric_robust_composite_uses_total_return() -> None:
    assert survival_metric_for_selection(ROBUST_COMPOSITE_METRIC_NAME) == "Total Return"
    assert survival_metric_for_selection("Sharpe") == "Sharpe"


def test_row_survives_gt_zero() -> None:
    assert row_survives({"Sharpe": 0.1, "Total Return": -0.05}, "Sharpe") is True
    assert row_survives({"Sharpe": -0.1}, "Sharpe") is False
    assert row_survives({}, "Sharpe") is False


def test_breakeven_slippage_max_surviving_at_mult_one() -> None:
    rows = [
        {"slippage_bps": 0.0, "commission_multiplier": 1.0, "survives": True},
        {"slippage_bps": 5.0, "commission_multiplier": 1.0, "survives": True},
        {"slippage_bps": 25.0, "commission_multiplier": 1.0, "survives": False},
        {"slippage_bps": 25.0, "commission_multiplier": 2.0, "survives": True},
    ]
    be = breakeven_slippage_bps_at_multiplier(
        rows, commission_multiplier=1.0, survival_metric="Sharpe"
    )
    assert be == pytest.approx(5.0)


def test_breakeven_none_when_no_survivors() -> None:
    rows = [
        {"slippage_bps": 0.0, "commission_multiplier": 1.0, "survives": False},
    ]
    assert (
        breakeven_slippage_bps_at_multiplier(
            rows, commission_multiplier=1.0, survival_metric="x"
        )
        is None
    )


def test_write_artifacts_and_summary_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "slippage_bps": 0.0,
            "commission_multiplier": 1.0,
            "survives": True,
            "Total Return": 0.1,
        },
        {
            "slippage_bps": 5.0,
            "commission_multiplier": 1.0,
            "survives": True,
            "Total Return": 0.02,
        },
    ]
    cfg = CostSensitivityConfig(
        enabled=True,
        slippage_bps_grid=(0.0, 5.0),
        commission_multiplier_grid=(1.0,),
        run_on=CostSensitivityRunOn.UNSEEN,
    )
    summary = build_cost_sensitivity_summary(
        rows=rows,
        cost_config=cfg,
        baseline_global={"slippage_bps": 2.5},
        selection_metric="Sharpe",
    )
    write_cost_sensitivity_artifacts(tmp_path, rows, summary)
    assert (tmp_path / "cost_sensitivity.csv").is_file()
    assert (tmp_path / "cost_sensitivity_summary.yaml").is_file()
    loaded = yaml.safe_load((tmp_path / "cost_sensitivity_summary.yaml").read_text(encoding="utf-8"))
    assert loaded["run_on"] == "unseen"
    assert loaded["survival_metric"] == "Sharpe"
    assert loaded["breakeven_slippage_bps_at_multiplier_1"] == pytest.approx(5.0)
