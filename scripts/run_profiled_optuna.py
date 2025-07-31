"""Run an Optuna optimisation under cProfile and save stats.

Usage (example):
    python run_profiled_optuna.py test_optuna_minimal --trials 10 --n_jobs 4 \
        --outfile profile_optuna_minimal.prof
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Optuna optimisation run")
    parser.add_argument("scenario", help="Scenario name (must exist in BACKTEST_SCENARIOS)")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials to request")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--outfile", default="profile_optuna.prof", help="Output .prof file path")
    args = parser.parse_args()

    # Construct command: python -m cProfile -o outfile -m src.portfolio_backtester.backtester ...
    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        args.outfile,
        "-m",
        "src.portfolio_backtester.backtester",
        "--mode",
        "optimize",
        "--scenario-name",
        args.scenario,
        "--optuna-trials",
        str(args.trials),
        "--n-jobs",
        str(args.n_jobs),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nProfile saved to {Path(args.outfile).resolve()}")


if __name__ == "__main__":
    main()
