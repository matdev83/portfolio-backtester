"""Unit tests for monthly optimization run summary helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from portfolio_backtester.reporting.monthly_opt_run_summary import (
    build_summary_row,
    find_latest_run_dir,
    is_timestamp_report_dir,
    parse_optimal_entry_day,
    scenario_name_to_calendar_month,
)


def test_is_timestamp_report_dir() -> None:
    assert is_timestamp_report_dir("20260502_224527") is True
    assert is_timestamp_report_dir("not_a_stamp") is False


def test_scenario_name_to_calendar_month() -> None:
    assert scenario_name_to_calendar_month("seasonal_optimize_march_entry_sortino_spy") == "March"


def test_find_latest_run_dir_picks_newer_mtime(tmp_path: Path) -> None:
    scenario = "seasonal_optimize_january_entry_sortino_spy"
    old = tmp_path / f"{scenario}_aaa111" / "20260101_000000"
    new = tmp_path / f"{scenario}_bbb222" / "20260102_000000"
    old.mkdir(parents=True)
    time.sleep(0.02)
    new.mkdir(parents=True)
    (old / "performance_metrics.csv").write_text("x", encoding="utf-8")
    (new / "performance_metrics.csv").write_text("y", encoding="utf-8")
    # Windows can assign identical mtimes in quick succession; compare timestamp dirs.
    old_ts = old.stat().st_mtime
    os.utime(new, (old_ts + 5.0, old_ts + 5.0))
    assert find_latest_run_dir(tmp_path, scenario) == new


def test_parse_optimal_entry_day(tmp_path: Path) -> None:
    p = tmp_path / "optimal_params_x_Optimized.txt"
    p.write_text("entry_day: 7\n", encoding="utf-8")
    assert parse_optimal_entry_day(p) == 7


def test_build_summary_row_reads_csv_and_params(tmp_path: Path) -> None:
    scenario = "seasonal_optimize_january_entry_sortino_spy"
    run = tmp_path / "run1"
    run.mkdir()
    csv_text = (
        ",SPY,seasonal_optimize_january_entry_sortino_spy_Optimized\n"
        "Total Return,1.0,0.5\n"
        "Ann. Return,0.1,0.05\n"
        "Sharpe,0.5,0.25\n"
        "Sortino,0.8,0.4\n"
        "Calmar,0.2,0.1\n"
        "Max Drawdown,-0.3,-0.15\n"
    )
    (run / "performance_metrics.csv").write_text(csv_text, encoding="utf-8")
    (run / f"optimal_params_{scenario}_Optimized.txt").write_text(
        "entry_day: 3\n", encoding="utf-8"
    )
    row = build_summary_row(scenario_name=scenario, run_dir=run)
    assert row["best_entry_day"] == 3
    assert row["strategy_Sortino"] == pytest.approx(0.4)
    assert row["SPY_Sortino"] == pytest.approx(0.8)
