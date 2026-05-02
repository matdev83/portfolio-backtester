"""Helpers to summarize latest optimization report folders for monthly seasonal SPY runs."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_TIMESTAMP_DIR = re.compile(r"^\d{8}_\d{6}$")

# Scenario `name` field from YAML (calendar order for default summary table).
DEFAULT_MONTHLY_SCENARIO_NAMES: tuple[str, ...] = tuple(
    f"seasonal_optimize_{m}_entry_sortino_spy"
    for m in (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
)

_METRIC_KEYS: tuple[str, ...] = (
    "Total Return",
    "Ann. Return",
    "Sharpe",
    "Sortino",
    "Calmar",
    "Max Drawdown",
)


def is_timestamp_report_dir(name: str) -> bool:
    """Return True if ``name`` matches ``YYYYMMDD_HHMMSS`` report run folders."""
    return bool(_TIMESTAMP_DIR.match(name))


def find_latest_run_dir(reports_dir: Path, scenario_name: str) -> Optional[Path]:
    """Locate the newest timestamp child under any ``{scenario_name}_*`` report root.

    Args:
        reports_dir: Typically ``data/reports``.
        scenario_name: Canonical scenario ``name`` (e.g. ``seasonal_optimize_january_entry_sortino_spy``).

    Returns:
        Path to the run directory containing ``performance_metrics.csv``, or ``None``.
    """
    if not reports_dir.is_dir():
        logger.warning("Reports directory does not exist: %s", reports_dir)
        return None

    best: Optional[tuple[float, Path]] = None
    prefix = scenario_name + "_"
    for root in reports_dir.iterdir():
        if not root.is_dir():
            continue
        if root.name != scenario_name and not root.name.startswith(prefix):
            continue
        for child in root.iterdir():
            if child.is_dir() and is_timestamp_report_dir(child.name):
                mtime = float(child.stat().st_mtime)
                if best is None or mtime > best[0]:
                    best = (mtime, child)
    return best[1] if best else None


def find_optimal_params_file(run_dir: Path) -> Optional[Path]:
    """Return the single ``optimal_params_*_Optimized.txt`` under ``run_dir``, if any."""
    matches = sorted(run_dir.glob("optimal_params_*_Optimized.txt"))
    if not matches:
        return None
    if len(matches) > 1 and logger.isEnabledFor(logging.WARNING):
        logger.warning("Multiple optimal_params files in %s; using %s", run_dir, matches[0])
    return matches[0]


def parse_optimal_entry_day(optimal_params_path: Path) -> Optional[int]:
    """Parse ``entry_day`` from an optimal-params text file."""
    text = optimal_params_path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("entry_day:"):
            return int(line.split(":", 1)[1].strip(), 10)
    return None


def _read_metrics_csv(
    csv_path: Path,
    scenario_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load strategy and SPY metric columns from ``performance_metrics.csv``."""
    strat_col = f"{scenario_name}_Optimized"
    strat: dict[str, float] = {}
    bench: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        return strat, bench
    header = rows[0]
    try:
        i_spy = header.index("SPY")
        i_strat = header.index(strat_col)
    except ValueError as exc:
        raise ValueError(f"Missing expected columns in {csv_path}: {exc}") from exc

    for row in rows[1:]:
        if not row or not row[0]:
            continue
        key = row[0].strip()
        if key not in _METRIC_KEYS:
            continue
        try:
            bench[key] = float(row[i_spy])
            strat[key] = float(row[i_strat])
        except (ValueError, IndexError):
            continue
    return strat, bench


def scenario_name_to_calendar_month(scenario_name: str) -> str:
    """Map ``seasonal_optimize_<month>_entry_sortino_spy`` to a title-case month label."""
    prefix = "seasonal_optimize_"
    suffix = "_entry_sortino_spy"
    if not scenario_name.startswith(prefix) or not scenario_name.endswith(suffix):
        return scenario_name
    slug = scenario_name[len(prefix) : -len(suffix)]
    return slug[:1].upper() + slug[1:]


def build_summary_row(
    *,
    scenario_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Assemble one summary row for a scenario's latest run directory."""
    month = scenario_name_to_calendar_month(scenario_name)
    csv_path = run_dir / "performance_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing performance_metrics.csv under {run_dir}")

    strat_metrics, spy_metrics = _read_metrics_csv(csv_path, scenario_name)
    opt_path = find_optimal_params_file(run_dir)
    entry_day: Optional[int] = parse_optimal_entry_day(opt_path) if opt_path else None

    row: dict[str, Any] = {
        "month": month,
        "scenario_name": scenario_name,
        "best_entry_day": entry_day,
        "run_dir": str(run_dir.resolve()),
    }
    for k in _METRIC_KEYS:
        row[f"strategy_{k.replace(' ', '_').replace('.', '')}"] = strat_metrics.get(k)
        row[f"SPY_{k.replace(' ', '_').replace('.', '')}"] = spy_metrics.get(k)
    return row


def collect_monthly_summary(
    reports_dir: Path,
    scenario_names: tuple[str, ...] = DEFAULT_MONTHLY_SCENARIO_NAMES,
) -> list[dict[str, Any]]:
    """Build summary rows for each scenario, using the latest report run per scenario."""
    rows: list[dict[str, Any]] = []
    for name in scenario_names:
        latest = find_latest_run_dir(reports_dir, name)
        if latest is None:
            rows.append(
                {
                    "month": scenario_name_to_calendar_month(name),
                    "scenario_name": name,
                    "best_entry_day": None,
                    "run_dir": "",
                    "_error": "no_report_run_found",
                }
            )
            continue
        try:
            rows.append(build_summary_row(scenario_name=name, run_dir=latest))
        except (OSError, ValueError, FileNotFoundError) as exc:
            rows.append(
                {
                    "month": scenario_name_to_calendar_month(name),
                    "scenario_name": name,
                    "best_entry_day": None,
                    "run_dir": str(latest.resolve()),
                    "_error": str(exc),
                }
            )
    return rows


def rows_to_markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render a compact markdown table for stdout (subset of columns)."""
    cols = [
        "month",
        "best_entry_day",
        "strategy_Sortino",
        "strategy_Sharpe",
        "strategy_Total_Return",
        "strategy_Ann_Return",
        "strategy_Max_Drawdown",
        "SPY_Sortino",
        "SPY_Total_Return",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
