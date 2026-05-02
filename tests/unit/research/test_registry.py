"""Tests for research run registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_backtester.research.registry import (
    ResearchRegistryError,
    ResearchRunRegistry,
    unseen_period_plain,
)
from portfolio_backtester.research.hashing import stable_hash
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
from portfolio_backtester.research.scoring import RobustSelectionConfig


def _proto() -> DoubleOOSWFOProtocolConfig:
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


def test_empty_registry_path(tmp_path: Path) -> None:
    reg = ResearchRunRegistry(tmp_path / "registry.yaml")
    assert not reg.path.is_file()
    assert reg.load_runs() == []


def test_record_lock_and_unseen_completed(tmp_path: Path) -> None:
    reg = ResearchRunRegistry(tmp_path / "registry.yaml")
    up = unseen_period_plain(_proto())
    uph = stable_hash(up)
    reg.record_lock(
        run_id="r1",
        scenario_hash="sh",
        protocol_config_hash="pch",
        unseen_period_hash=uph,
        lock_path="r1/protocol_lock.yaml",
        created_at="2024-01-01T00:00:00+00:00",
    )
    reg.mark_unseen_completed("r1")
    runs = reg.load_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r1"
    assert runs[0]["scenario_hash"] == "sh"
    assert runs[0]["protocol_config_hash"] == "pch"
    assert runs[0]["unseen_period_hash"] == uph
    assert runs[0]["lock_path"] == "r1/protocol_lock.yaml"
    assert runs[0]["unseen_completed"] is True
    assert runs[0]["created_at"] == "2024-01-01T00:00:00+00:00"


def test_find_completed_duplicate(tmp_path: Path) -> None:
    reg = ResearchRunRegistry(tmp_path / "registry.yaml")
    p = _proto()
    uph = stable_hash(unseen_period_plain(p))
    reg.record_lock(
        run_id="old",
        scenario_hash="a",
        protocol_config_hash="b",
        unseen_period_hash=uph,
        lock_path="old/protocol_lock.yaml",
        created_at="t0",
    )
    reg.mark_unseen_completed("old")
    hit = reg.find_completed_duplicate("a", "b", uph)
    assert hit is not None
    assert hit["run_id"] == "old"


def test_raises_on_duplicate_when_not_forced(tmp_path: Path) -> None:
    reg = ResearchRunRegistry(tmp_path / "registry.yaml")
    p = _proto()
    uph = stable_hash(unseen_period_plain(p))
    reg.record_lock(
        run_id="old",
        scenario_hash="a",
        protocol_config_hash="b",
        unseen_period_hash=uph,
        lock_path="old/protocol_lock.yaml",
        created_at="t0",
    )
    reg.mark_unseen_completed("old")
    with pytest.raises(ResearchRegistryError, match="old"):
        reg.assert_no_completed_duplicate("a", "b", uph, force_new_research_run=False)


def test_force_allows_despite_duplicate(tmp_path: Path) -> None:
    reg = ResearchRunRegistry(tmp_path / "registry.yaml")
    p = _proto()
    uph = stable_hash(unseen_period_plain(p))
    reg.record_lock(
        run_id="old",
        scenario_hash="a",
        protocol_config_hash="b",
        unseen_period_hash=uph,
        lock_path="old/protocol_lock.yaml",
        created_at="t0",
    )
    reg.mark_unseen_completed("old")
    reg.assert_no_completed_duplicate("a", "b", uph, force_new_research_run=True)


def test_skip_unseen_does_not_mark_completed(tmp_path: Path) -> None:
    reg = ResearchRunRegistry(tmp_path / "registry.yaml")
    p = _proto()
    uph = stable_hash(unseen_period_plain(p))
    reg.record_lock(
        run_id="r2",
        scenario_hash="a",
        protocol_config_hash="b",
        unseen_period_hash=uph,
        lock_path="r2/protocol_lock.yaml",
        created_at="t1",
    )
    runs = reg.load_runs()
    assert runs[0]["unseen_completed"] is False
    assert reg.find_completed_duplicate("a", "b", uph) is None
