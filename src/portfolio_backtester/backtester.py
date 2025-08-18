import argparse
import logging
import sys
import warnings
import os
import time
from pathlib import Path
from typing import List, Optional
import portfolio_backtester.config_loader as config_loader

from portfolio_backtester.core import Backtester
from portfolio_backtester.config_loader import (
    ConfigurationError,
    load_scenario_from_file,
)
from portfolio_backtester.scenario_validator import (
    validate_scenario_semantics,
    YamlValidator,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portfolio backtester.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["backtest", "optimize"],
        help="Mode to run the backtester in.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        help="Name of the scenario to run/optimize from BACKTEST_SCENARIOS.",
    )
    parser.add_argument(
        "--scenario-filename",
        type=str,
        help="Path to a single scenario file to run/optimize.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Name of the Optuna study to use for optimization or to load best parameters from.",
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        help="Optuna storage URL. If not provided, SQLite will be used.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Set a random seed for reproducibility.",
    )
    parser.add_argument(
        "--optimize-min-positions",
        type=int,
        default=10,
        help="Minimum number of positions to consider during optimization of num_holdings.",
    )
    parser.add_argument(
        "--optimize-max-positions",
        type=int,
        default=30,
        help="Maximum number of positions to consider during optimization of num_holdings.",
    )
    parser.add_argument(
        "--top-n-params",
        type=int,
        default=3,
        help="Number of top performing parameter values to keep per grid.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel worker processes to use (-1 ⇒ all cores).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop optimisation after N successive ~zero-return evaluations.",
    )
    parser.add_argument(
        "--early-stop-zero-trials",
        type=int,
        default=20,
        help="Stop optimization early after N consecutive trials with zero values. Default: 20.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=200,
        help="Maximum trials per optimization.",
    )
    parser.add_argument(
        "--optuna-timeout-sec",
        type=int,
        default=None,
        help="Time budget per optimization (seconds).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="optuna",
        choices=["optuna", "genetic"],
        help="Optimizer to use ('optuna' or 'genetic'). Default: optuna.",
    )
    # GA-specific knobs
    parser.add_argument(
        "--ga-population-size",
        type=int,
        default=50,
        help="Genetic algorithm population size (GA only).",
    )
    parser.add_argument(
        "--ga-max-generations",
        type=int,
        default=10,
        help="Genetic algorithm maximum generations (GA only).",
    )
    parser.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=0.1,
        help="Genetic algorithm mutation rate (GA only).",
    )
    parser.add_argument(
        "--ga-crossover-rate",
        type=float,
        default=0.8,
        help="Genetic algorithm crossover rate (GA only).",
    )
    # Population diversity settings
    parser.add_argument(
        "--ga-similarity-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for population diversity (0.0-1.0, GA only).",
    )
    parser.add_argument(
        "--ga-min-diversity-ratio",
        type=float,
        default=0.7,
        help="Minimum ratio of unique individuals in population (0.0-1.0, GA only).",
    )
    parser.add_argument(
        "--ga-enforce-diversity",
        action="store_true",
        default=True,
        help="Actively enforce population diversity to avoid similar individuals (GA only).",
    )
    # Joblib tuning knobs for population evaluation
    parser.add_argument(
        "--joblib-batch-size",
        type=str,
        default="auto",
        help="Batch size for joblib. Use 'auto' or an integer. (Population optimizers only)",
    )
    parser.add_argument(
        "--joblib-pre-dispatch",
        type=str,
        default="3*n_jobs",
        help="Pre-dispatch setting for joblib (e.g., '3*n_jobs'). (Population optimizers only)",
    )
    parser.add_argument(
        "--enable-adaptive-batch-sizing",
        action="store_true",
        default=True,
        help="Dynamically adjust batch sizes based on parameter space and population diversity",
    )
    parser.add_argument(
        "--enable-hybrid-parallelism",
        action="store_true",
        default=True,
        help="Use hybrid parallelism with process-level parallelism for population and thread-level for windows",
    )
    parser.add_argument(
        "--enable-incremental-evaluation",
        action="store_true",
        default=True,
        help="Enable incremental window evaluation to avoid redundant calculations when parameters change minimally",
    )
    parser.add_argument(
        "--enable-gpu-acceleration",
        action="store_true",
        default=True,
        help="Enable GPU acceleration for fitness evaluation when CuPy is available (auto-fallback to CPU)",
    )
    # Deduplication settings
    parser.add_argument(
        "--use-persistent-cache",
        action="store_true",
        help="Use persistent deduplication cache across processes and runs. Default: False.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store persistent deduplication cache. Default: system temp directory.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Filename for persistent deduplication cache. Default: <optimizer_type>_dedup_cache.pkl.",
    )
    parser.add_argument(
        "--pruning-enabled",
        action="store_true",
        help="Enable trial pruning with MedianPruner (Optuna only). Default: False.",
    )
    parser.add_argument(
        "--pruning-n-startup-trials",
        type=int,
        default=5,
        help="MedianPruner: Number of trials to complete before pruning begins. Default: 5.",
    )
    parser.add_argument(
        "--pruning-n-warmup-steps",
        type=int,
        default=0,
        help="MedianPruner: Number of intermediate steps (walk-forward windows) to observe before pruning a trial. Default: 0.",
    )
    parser.add_argument(
        "--pruning-interval-steps",
        type=int,
        default=1,
        help="MedianPruner: Report intermediate value and check for pruning every X walk-forward windows. Default: 1.",
    )
    parser.add_argument(
        "--fresh-study",
        action="store_true",
        help="Force a new Optuna study by deleting any existing study with the same name. Default: False.",
    )
    parser.add_argument(
        "--mc-simulations",
        type=int,
        default=1000,
        help="Number of simulations for Monte Carlo analysis.",
    )
    parser.add_argument(
        "--mc-years",
        type=int,
        default=10,
        help="Number of years to project in Monte Carlo analysis.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show plots interactively (blocks execution). Default: off, only saves plots.",
    )
    parser.add_argument(
        "--enable-stage2-mc",
        action="store_true",
        help="Enable Stage 2 Monte Carlo robustness visualization (disabled by default).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Global timeout in seconds for the entire run.",
    )
    parser.add_argument(
        "--test-fast-optimize",
        action="store_true",
        default=False,
        help="Enable test-only fast optimization path (skip heavy reporting, deterministic evaluators).",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv[1:]

    parser = _create_parser()
    parsed_args = parser.parse_args(args)

    # Configure logging at the earliest point
    log_level = getattr(logging, parsed_args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )

    # Re-get the logger now that the configuration is set
    logger = logging.getLogger(__name__)

    # Set up file logging for debug output
    file_handler = logging.FileHandler("optimizer_debug.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    try:
        # Load global configuration and scenarios (call through module for test patching)
        config_loader.load_config()
        # Read from module attributes so test patches on config_loader are respected
        GLOBAL_CONFIG_RELOADED = config_loader.GLOBAL_CONFIG
        BACKTEST_SCENARIOS_RELOADED = config_loader.BACKTEST_SCENARIOS

        # Apply performance tweaks from config
        performance_config = GLOBAL_CONFIG_RELOADED.get("performance", {})
        if performance_config.get("enable_numba_fastmath", False):
            os.environ["NUMBA_NUM_THREADS"] = str(performance_config.get("NUMBA_NUM_THREADS", 1))
            os.environ["OMP_NUM_THREADS"] = str(performance_config.get("OMP_NUM_THREADS", 1))
            os.environ["MKL_NUM_THREADS"] = str(performance_config.get("MKL_NUM_THREADS", 1))
            os.environ["OPENBLAS_NUM_THREADS"] = str(
                performance_config.get("OPENBLAS_NUM_THREADS", 1)
            )
            logger.info("Applied performance.thread_caps environment limits.")

        logger.info("Configuration loaded successfully")

    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        print(f"\n❌ Unexpected configuration error: {e}", file=sys.stderr)
        print("Please check your configuration files for syntax errors.", file=sys.stderr)
        sys.exit(1)

    if (
        parsed_args.mode == "optimize"
        and parsed_args.scenario_name is None
        and parsed_args.scenario_filename is None
    ):
        parser.error("--scenario-name or --scenario-filename is required for 'optimize' mode.")

    if parsed_args.scenario_filename:
        scenario_path = Path(parsed_args.scenario_filename)
        if not scenario_path.exists():
            parser.error(f"Scenario file not found: {parsed_args.scenario_filename}")

        try:
            scenario_from_file = load_scenario_from_file(scenario_path)
            selected_scenarios = [scenario_from_file]
        except ConfigurationError as e:
            parser.error(f"Error loading scenario file: {e}")

    elif parsed_args.scenario_name is not None:
        scenario_from_config = next(
            (s for s in BACKTEST_SCENARIOS_RELOADED if s.get("name") == parsed_args.scenario_name),
            None,
        )

        if scenario_from_config:
            selected_scenarios = [scenario_from_config]
        else:
            parser.error(
                f"Scenario '{parsed_args.scenario_name}' not found in any of the loaded configuration files."
            )
    else:
        selected_scenarios = BACKTEST_SCENARIOS_RELOADED

    if parsed_args.mode == "optimize":
        # The original code had config_loader_module.load_globals_only() here,
        # but config_loader_module is no longer imported.
        # Assuming the intent was to load globals if they were loaded globally.
        # Since load_config is now global, this line is effectively a no-op.
        # For now, I'm removing it as it's not directly related to the new_code.
        pass  # config_loader_module.load_globals_only()
        for scen in selected_scenarios:
            errors = validate_scenario_semantics(
                scen,
                GLOBAL_CONFIG_RELOADED.get("optimizer", {}).get("parameter_defaults", {}),
            )
            if errors:
                formatted = YamlValidator().format_errors(errors)
                formatted += "\nActionable Suggestions:\n- Verify param names match strategy.tunable_parameters()\n- Check types (int/float) and ranges\n- Add missing required params\n- See docs for more."
                print(formatted, file=sys.stderr)
                raise ConfigurationError("Optimization config invalid")

    # Start timing the backtester run
    start_time = time.time()

    backtester = Backtester(
        GLOBAL_CONFIG_RELOADED,
        selected_scenarios,
        parsed_args,
    )
    try:
        backtester.run()
        # Log execution time
        execution_time = time.time() - start_time

        # We don't duplicate the period and days/second info here as it's already logged in backtester_facade.py
        logger.info(
            f"Total backtester execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        )
    except Exception as e:
        logger.error(f"An unhandled exception occurred during backtester run: {e}", exc_info=True)
        # Log execution time even on failure
        execution_time = time.time() - start_time
        logger.info(
            f"Failed backtester execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        )
        # Assuming the intent was to exit with a non-zero status if an error occurs.
        sys.exit(1)
    # Remove unconditional sys.exit(0) to allow tests to call main() without exiting


if __name__ == "__main__":
    main()
