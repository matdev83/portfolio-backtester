import argparse
from pathlib import Path
import sys

from portfolio_backtester.backtester import main as backtester_main


def run_with_scenario(scenario_file: Path) -> None:
    # Invoke the backtester CLI as a module by adjusting sys.argv minimally
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            sys.argv[0],
            "--mode",
            "backtest",
            "--scenario-filename",
            str(scenario_file),
            "--log-level",
            "INFO",
        ]
        backtester_main()
    finally:
        sys.argv = argv_backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick stop-loss smoke test using a scenario YAML file."
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("config/scenarios/examples/diagnostic/default.yaml"),
        help="Path to scenario YAML to run (default: examples/diagnostic/default.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.scenario.exists():
        print(f"Scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)
    run_with_scenario(args.scenario)
