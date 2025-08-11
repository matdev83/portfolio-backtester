import cProfile
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path for module resolution
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if src_path.is_dir():
    sys.path.insert(0, str(src_path))
else:
    # Fallback for different execution contexts
    sys.path.insert(0, str(project_root))

from line_profiler import LineProfiler  # noqa: E402

from portfolio_backtester.backtester import main as backtester_main  # noqa: E402
from portfolio_backtester.backtesting.strategy_backtester import (  # noqa: E402
    StrategyBacktester,
)


def profile_optimizer():
    """
    A general-purpose profiler for the optimization process, focusing on the
    core `backtest_strategy` method.
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
        "5",  # A few trials to get a good sample
        "--n-jobs",
        "-1",
    ]

    # Set up profilers
    line_profiler = LineProfiler()
    line_profiler.add_function(StrategyBacktester.backtest_strategy)
    c_profiler = cProfile.Profile()

    print("Running general optimizer profiler...")
    try:
        c_profiler.enable()
        # Use runcall to handle the main function execution
        line_profiler.runcall(backtester_main)
    except SystemExit:
        # The backtester main function can call sys.exit, which we catch here
        print("Backtester main function exited as expected.")
    finally:
        c_profiler.disable()
        sys.argv = original_argv
        print("Profiling complete.")

        # Create a unique filename for the results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = tempfile.gettempdir()

        # Save cProfile results
        cprofile_file = Path(temp_dir) / f"optimizer_cprofile_{timestamp}.pstats"
        c_profiler.dump_stats(cprofile_file)
        print(f"cProfile results saved to: {cprofile_file}")

        # Save line_profiler results
        lprofile_file = Path(temp_dir) / f"optimizer_lprofile_{timestamp}.txt"
        with open(lprofile_file, "w", encoding="utf-8") as f:
            line_profiler.print_stats(f)
        print(f"line_profiler results saved to: {lprofile_file}")


if __name__ == "__main__":
    profile_optimizer()
