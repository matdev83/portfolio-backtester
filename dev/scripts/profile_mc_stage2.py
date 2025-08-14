import argparse
import cProfile
import os
import pstats
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Stage 2 Monte Carlo stress testing")
    parser.add_argument(
        "--scenario-name",
        default="momentum_unfiltered_atr_portfolio",
        help="Scenario name to optimize (must exist in loaded configs)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["optuna", "genetic"],
        default="optuna",
        help="Optimizer to use for the run",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=1,
        help="Number of Optuna trials (only used when optimizer=optuna)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Workers for evaluation (-1 to use all cores)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--profile-out",
        default=str(Path("prof") / "mc_stage2.prof"),
        help="Path to write cProfile stats file",
    )
    args = parser.parse_args()

    # Ensure profile output directory exists
    out_path = Path(args.profile_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build argv for backtester CLI
    cli_argv = [
        "backtester",
        "--mode",
        "optimize",
        "--scenario-name",
        args.scenario_name,
        "--optimizer",
        args.optimizer,
        "--n-jobs",
        str(args.n_jobs),
        "--log-level",
        args.log_level,
    ]
    if args.optimizer == "optuna":
        cli_argv += ["--optuna-trials", str(args.optuna_trials)]

    # Run the backtester under cProfile
    prof = cProfile.Profile()
    try:
        import portfolio_backtester.backtester as backtester_main

        # Temporarily override sys.argv for the CLI entrypoint
        old_argv = sys.argv
        sys.argv = cli_argv
        try:
            prof.enable()
            backtester_main.main()
        finally:
            prof.disable()
            sys.argv = old_argv
    finally:
        prof.dump_stats(str(out_path))

    # Print top hotspots focused on Stage 2 MC modules
    print(f"\nProfile saved to: {out_path}")
    stats = pstats.Stats(str(out_path))
    stats.strip_dirs()
    print("\nTop cumulative time (global):")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)

    print("\nTop cumulative time (Stage 2 MC focused):")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(
        r"monte_carlo_stage2|asset_replacement|synthetic|run_scenario|strategy_backtester"
    )


if __name__ == "__main__":
    main()


