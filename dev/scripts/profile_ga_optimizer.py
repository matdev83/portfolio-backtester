import cProfile
import sys
import tempfile
import time
from pathlib import Path

# Ensure src on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if src_path.is_dir():
    sys.path.insert(0, str(src_path))
else:
    sys.path.insert(0, str(project_root))

from line_profiler import LineProfiler  # noqa: E402
from portfolio_backtester.backtester import main as backtester_main  # noqa: E402
from portfolio_backtester.backtesting.strategy_backtester import (  # noqa: E402
    StrategyBacktester,
)
import argparse  # noqa: E402


def profile_ga_optimizer():
    """Run a GA optimization and capture cProfile and line_profiler outputs."""
    original_argv = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", type=str, default="builtins/signal/dummy_signal_strategy/default.yaml"
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--ga-population-size", type=int, default=50)
    parser.add_argument("--ga-max-generations", type=int, default=5)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.1)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8)
    parser.add_argument("--joblib-batch-size", type=str, default=None)
    parser.add_argument("--joblib-pre-dispatch", type=str, default=None)
    parser.add_argument("--use-persistent-cache", action="store_true", help="Enable persistent cache")
    args, _ = parser.parse_known_args()

    # Resolve scenario path
    scenario_path = project_root / "config" / "scenarios" / Path(args.scenario)

    # GA knobs are read in optimization_orchestrator via parsed args attributes
    sys.argv = [
        "backtester",
        "--mode",
        "optimize",
        "--scenario-filename",
        str(scenario_path),
        "--optimizer",
        "genetic",
        "--n-jobs",
        str(args["n_jobs"]) if isinstance(args, dict) else str(args.n_jobs),
        "--ga-population-size",
        str(args.ga_population_size),
        "--ga-max-generations",
        str(args.ga_max_generations),
        "--ga-mutation-rate",
        str(args.ga_mutation_rate),
        "--ga-crossover-rate",
        str(args.ga_crossover_rate),
    ]

    if args.joblib_batch_size is not None:
        sys.argv += ["--joblib-batch-size", str(args.joblib_batch_size)]
    if args.joblib_pre_dispatch is not None:
        sys.argv += ["--joblib-pre-dispatch", str(args.joblib_pre_dispatch)]
    if args.use_persistent_cache:
        sys.argv += ["--use-persistent-cache"]

    # Optionally allow overriding GA settings via env or edit here
    # Defaults applied in orchestrator: population_size=50, max_generations=10

    line_profiler = LineProfiler()
    line_profiler.add_function(StrategyBacktester.backtest_strategy)
    c_profiler = cProfile.Profile()

    print("Running GA optimizer profiler...")
    try:
        c_profiler.enable()
        # Pass args directly to backtester_main to avoid SystemExit
        line_profiler.runcall(backtester_main)
    except SystemExit as e:
        print(f"Backtester main function exited with code {e.code} (expected).")
    except Exception as e:
        print(f"Error during profiling: {e}")
    finally:
        c_profiler.disable()
        sys.argv = original_argv

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = tempfile.gettempdir()

        cprofile_file = Path(temp_dir) / f"ga_optimizer_cprofile_{timestamp}.pstats"
        c_profiler.dump_stats(cprofile_file)
        print(f"cProfile results saved to: {cprofile_file}")

        lprofile_file = Path(temp_dir) / f"ga_optimizer_lprofile_{timestamp}.txt"
        with open(lprofile_file, "w", encoding="utf-8") as f:
            line_profiler.print_stats(f)
        print(f"line_profiler results saved to: {lprofile_file}")


if __name__ == "__main__":
    profile_ga_optimizer()
