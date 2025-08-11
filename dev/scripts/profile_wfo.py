import cProfile
import sys
import tempfile
from pathlib import Path

from line_profiler import LineProfiler

# Add project root to path for module resolution
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if src_path.is_dir():
    sys.path.insert(0, str(src_path))
else:
    # Fallback for different execution contexts
    sys.path.insert(0, str(project_root))

from portfolio_backtester.backtester import main as backtester_main  # noqa: E402
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester  # noqa: E402
from portfolio_backtester.optimization.evaluator import BacktestEvaluator  # noqa: E402


def profile_wfo():
    """
    Profiles the Walk-Forward Optimization process to identify performance bottlenecks,
    specifically focusing on the window evaluation logic.
    """
    # Prepare arguments for the backtester
    original_argv = sys.argv
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
        "1",  # One trial is enough to profile the WFO loop
        "--n-jobs",
        "-1",
    ]

    # Set up profilers
    line_profiler = LineProfiler()
    line_profiler.add_function(BacktestEvaluator.evaluate_parameters)
    line_profiler.add_function(StrategyBacktester.backtest_strategy)

    c_profiler = cProfile.Profile()

    print("Running WFO profiler...")
    try:
        c_profiler.enable()
        line_profiler.runcall(backtester_main)
    finally:
        c_profiler.disable()
        sys.argv = original_argv
        print("Profiling complete.")

        # Save cProfile results
        temp_dir = tempfile.gettempdir()
        cprofile_file = Path(temp_dir) / "wfo_cprofile_results.pstats"
        c_profiler.dump_stats(cprofile_file)
        print(f"cProfile results saved to: {cprofile_file}")

        # Save line_profiler results
        lprofile_file = Path(temp_dir) / "wfo_lprofile_results.txt"
        with open(lprofile_file, "w", encoding="utf-8") as f:
            line_profiler.print_stats(f)
        print(f"line_profiler results saved to: {lprofile_file}")


if __name__ == "__main__":
    profile_wfo()
