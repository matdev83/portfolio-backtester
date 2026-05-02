"""Tests for walk-forward architecture grid expansion."""

from __future__ import annotations

from portfolio_backtester.research.double_oos_wfo import expand_wfo_architecture_grid
from portfolio_backtester.research.protocol_config import WFOGridConfig


def test_cartesian_order_train_test_step_type_from_grid_config() -> None:
    grid = WFOGridConfig(
        train_window_months=(12, 24),
        test_window_months=(6,),
        wfo_step_months=(3,),
        walk_forward_type=("rolling", "expanding"),
    )
    archs = expand_wfo_architecture_grid(grid=grid)
    keys = [
        (a.train_window_months, a.test_window_months, a.wfo_step_months, a.walk_forward_type)
        for a in archs
    ]
    assert keys == [
        (12, 6, 3, "rolling"),
        (12, 6, 3, "expanding"),
        (24, 6, 3, "rolling"),
        (24, 6, 3, "expanding"),
    ]


def test_dedup_preserves_first_occurrence_order() -> None:
    grid = WFOGridConfig(
        train_window_months=(24, 24),
        test_window_months=(6, 6),
        wfo_step_months=(3,),
        walk_forward_type=("rolling",),
    )
    archs = expand_wfo_architecture_grid(grid=grid)
    assert len(archs) == 1
    assert archs[0].train_window_months == 24


def test_expand_named_sequences_equivalent() -> None:
    grid = WFOGridConfig(
        train_window_months=(12,),
        test_window_months=(6,),
        wfo_step_months=(3,),
        walk_forward_type=("rolling",),
    )
    a = expand_wfo_architecture_grid(grid=grid)
    b = expand_wfo_architecture_grid(
        train_window_months=[12],
        test_window_months=[6],
        wfo_step_months=[3],
        walk_forward_type=["rolling"],
    )
    assert a == b
