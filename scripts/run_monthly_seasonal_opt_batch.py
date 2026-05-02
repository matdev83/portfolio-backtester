"""Run twelve monthly SeasonalSignalStrategy SPY Sortino optimizations (Jan–Dec) sequentially."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PY = _REPO / ".venv" / "Scripts" / "python.exe"
_SCENARIO_DIR = _REPO / "config" / "scenarios" / "builtins" / "signal" / "seasonal_signal_strategy"
_MONTHS = (
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-month",
        default="january",
        metavar="MONTH",
        help="Start from this English month name (e.g. february) to resume a long batch.",
    )
    args = parser.parse_args()
    start = args.from_month.strip().lower()
    if start not in _MONTHS:
        raise SystemExit(f"Unknown month {start!r}; choose one of: {', '.join(_MONTHS)}")
    months = _MONTHS[_MONTHS.index(start) :]

    for month in months:
        yaml_path = _SCENARIO_DIR / f"optimize_{month}_entry_sortino_spy.yaml"
        rel = yaml_path.relative_to(_REPO)
        print(f"==== RUN {month} ({rel}) ====", flush=True)
        cmd = [
            str(_PY),
            "-m",
            "src.portfolio_backtester.backtester",
            "--mode",
            "optimize",
            "--scenario-filename",
            str(rel).replace("\\", "/"),
            "--mdmp-cache-only",
            "--optuna-trials",
            "40",
            "--early-stop-patience",
            "0",
            # Single worker avoids nested parallel pools during long batch runs on Windows.
            "--n-jobs",
            "1",
            # WARNING hides per-trial INFO; output then looks "stuck" after Optuna's first line.
            "--log-level",
            "INFO",
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        subprocess.run(cmd, cwd=str(_REPO), check=True, env=env)
    print("ALL_MONTHLY_OPT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
