"""Profile a small optimization run using DummySignalStrategy.

This script runs the optimizer under cProfile and writes:
- Raw stats to: profile_dummy.prof
- Human-readable summary to: profile_dummy.txt

It uses a 20-combination grid defined in
`config/scenarios/builtins/signal/dummy_signal_strategy/default.yaml`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


SCENARIO_FILE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "scenarios"
    / "builtins"
    / "signal"
    / "dummy_signal_strategy"
    / "default.yaml"
)

PROF_FILE = Path("profile_dummy.prof")
SUMMARY_FILE = Path("profile_dummy.txt")


def run_profile() -> None:
    cmd = [
        str(Path(".venv") / "Scripts" / "python.exe"),
        "-m",
        "cProfile",
        "-o",
        str(PROF_FILE),
        "-m",
        "portfolio_backtester.backtester",
        "--mode",
        "optimize",
        "--scenario-filename",
        str(SCENARIO_FILE),
        "--optuna-trials",
        "20",
        "--n-jobs",
        "1",
        "--log-level",
        "INFO",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize_profile() -> None:
    # Use pstats to produce a human-readable summary sorted by cumtime
    code = (
        "import pstats; s=pstats.Stats('" + str(PROF_FILE) + "'); "
        "s.strip_dirs().sort_stats('cumtime').print_stats(100)"
    )
    cmd = [str(Path(".venv") / "Scripts" / "python.exe"), "-c", code]
    print("Summarizing profile to", SUMMARY_FILE)
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def main() -> None:
    if not SCENARIO_FILE.exists():
        raise SystemExit(f"Scenario file not found: {SCENARIO_FILE}")
    run_profile()
    summarize_profile()
    print("\nDone. Files written:")
    print(" -", PROF_FILE.resolve())
    print(" -", SUMMARY_FILE.resolve())


if __name__ == "__main__":
    main()
