import sys
import tempfile
from pathlib import Path

from line_profiler import LineProfiler

from portfolio_backtester.backtester import main as backtester_main
from portfolio_backtester.backtester_logic.portfolio_logic import (
    calculate_portfolio_returns,
)
from portfolio_backtester.backtester_logic.strategy_logic import (
    generate_signals,
    size_positions,
)


def profile_hotspots():
    """
    Profiles the core backtesting functions to identify performance hotspots.
    """
    # Add project root to path for module resolution
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = project_root / "src"
    if src_path.is_dir():
        sys.path.insert(0, str(src_path))
    else:
        # Fallback for different execution contexts
        sys.path.insert(0, str(project_root))

    # Functions to profile
    functions_to_profile = [
        generate_signals,
        size_positions,
        calculate_portfolio_returns,
    ]

    profiler = LineProfiler()
    for func in functions_to_profile:
        profiler.add_function(func)

    # Prepare arguments for the backtester
    scenario_path = (
        project_root
        / "config"
        / "scenarios"
        / "builtins"
        / "signal"
        / "dummy_signal_strategy"
        / "default.yaml"
    )
    sys.argv = [
        "backtester",
        "--mode",
        "optimize",
        "--scenario-filename",
        str(scenario_path),
        "--optuna-trials",
        "1",
        "--n-jobs",
        "-1",
    ]

    print("Running profiler on hotspots...")
    profiler.runcall(backtester_main)
    print("Profiling complete.")

    # Save results
    temp_dir = tempfile.gettempdir()
    results_file = Path(temp_dir) / "hotspot_profile_results_final.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        profiler.print_stats(f)

    print(f"Profiling results saved to: {results_file}")


if __name__ == "__main__":
    profile_hotspots()
