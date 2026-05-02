#!/usr/bin/env python3
"""Summarize latest monthly seasonal SPY optimization runs into one CSV + markdown table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from portfolio_backtester.reporting.monthly_opt_run_summary import (  # noqa: E402
    DEFAULT_MONTHLY_SCENARIO_NAMES,
    collect_monthly_summary,
    rows_to_markdown_table,
)


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate best entry_day and headline metrics from latest monthly runs."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "reports",
        help="Root containing <scenario>_<hash>/<timestamp>/ artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "reports" / "seasonal_month_spylong_sortino_summary.csv",
        help="Path for the aggregated CSV.",
    )
    parser.add_argument(
        "--no-print-md",
        action="store_true",
        help="Skip printing the markdown table to stdout.",
    )
    args = parser.parse_args()

    rows_any: list[dict[str, object]] = list(
        collect_monthly_summary(args.reports_dir, DEFAULT_MONTHLY_SCENARIO_NAMES)
    )
    _write_csv(rows_any, args.output)
    print(f"Wrote {args.output.resolve()}", file=sys.stderr)
    if not args.no_print_md:
        print(rows_to_markdown_table(rows_any))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
