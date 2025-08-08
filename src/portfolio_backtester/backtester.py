import argparse
import logging
import sys
import warnings
from pathlib import Path
import os

from .core import Backtester
from .utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from .utils import register_signal_handler as register_central_signal_handler
from .config_loader import ConfigurationError
from . import config_loader as config_loader_module
from .scenario_validator import validate_scenario_semantics, YamlValidator
from .interfaces.attribute_accessor_interface import create_module_attribute_accessor

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    try:
        # Assuming the script is run from the root of the project
        from .strategy_config_validator import validate_strategy_configs

        project_root = Path(__file__).parent.parent.parent
        strategies_directory = project_root / "src" / "portfolio_backtester" / "strategies"
        scenarios_directory = project_root / "config" / "scenarios"

        is_valid, validation_errors = validate_strategy_configs(
            str(strategies_directory), str(scenarios_directory)
        )
        if not is_valid:
            for error in validation_errors:
                logger.error(error)
            logger.error(
                "Strategy validation failed! Please ensure all strategies have corresponding default.yaml config files."
            )
            sys.exit(1)
        else:
            logger.info(
                "Strategy validation passed - all strategies have corresponding default.yaml config files."
            )
    except FileNotFoundError as e:
        logger.error(f"Missing YAML configuration for strategy: {e}")
        sys.exit(1)

    # Safely configure UTF-8 on Windows terminals where supported
    if sys.platform == "win32":
        try:
            # Only TextIO has reconfigure in recent Python; guard for type-checker and runtime
            out = sys.stdout
            err = sys.stderr
            if hasattr(out, "reconfigure"):
                out.reconfigure(encoding="utf-8")
            if hasattr(err, "reconfigure"):
                err.reconfigure(encoding="utf-8")
        except Exception:
            # Non-fatal if the environment doesn't support reconfigure
            pass

    register_central_signal_handler()

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
        "--scenario-filename", type=str, help="Path to a single scenario file to run/optimize."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Name of the Optuna study to use for optimization or to load best parameters from.",
    )
    parser.add_argument(
        "--storage-url", type=str, help="Optuna storage URL. If not provided, SQLite will be used."
    )
    parser.add_argument(
        "--random-seed", type=int, default=None, help="Set a random seed for reproducibility."
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
        "--n-jobs", type=int, default=8, help="Parallel worker processes to use (-1 ⇒ all cores)."
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
        "--optuna-trials", type=int, default=200, help="Maximum trials per optimization."
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
        "--timeout", type=int, default=None, help="Global timeout in seconds for the entire run."
    )
    args = parser.parse_args()

    # Use DIP interface instead of direct getattr call
    module_accessor = create_module_attribute_accessor()
    log_level = module_accessor.get_module_attribute(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Set up file logging for debug output
    file_handler = logging.FileHandler("optimizer_debug.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    try:
        config_loader_module.load_config()
        GLOBAL_CONFIG_RELOADED = config_loader_module.GLOBAL_CONFIG
        BACKTEST_SCENARIOS_RELOADED = config_loader_module.BACKTEST_SCENARIOS

        logger.info("Configuration loaded successfully")

        # Optional thread caps to avoid oversubscription (config: performance.thread_caps)
        try:
            perf_cfg = (
                GLOBAL_CONFIG_RELOADED.get("performance", {})
                if isinstance(GLOBAL_CONFIG_RELOADED, dict)
                else {}
            )
            thread_caps = perf_cfg.get("thread_caps", {}) if isinstance(perf_cfg, dict) else {}
            if thread_caps.get("enabled", False):
                omp = thread_caps.get("OMP_NUM_THREADS")
                numba_threads = thread_caps.get("NUMBA_NUM_THREADS")
                mkl = thread_caps.get("MKL_NUM_THREADS")
                openblas = thread_caps.get("OPENBLAS_NUM_THREADS")

                if isinstance(omp, int) and omp > 0:
                    os.environ["OMP_NUM_THREADS"] = str(omp)
                if isinstance(numba_threads, int) and numba_threads > 0:
                    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
                if isinstance(mkl, int) and mkl > 0:
                    os.environ["MKL_NUM_THREADS"] = str(mkl)
                if isinstance(openblas, int) and openblas > 0:
                    os.environ["OPENBLAS_NUM_THREADS"] = str(openblas)

                logger.info("Applied performance.thread_caps environment limits.")
        except Exception as _e:
            logger.warning(f"Could not apply performance.thread_caps: {_e}")

    except ConfigurationError as e:
        logger.error("Configuration validation failed!")
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        print("\nTo validate your configuration files, run:", file=sys.stderr)
        print("  python -m src.portfolio_backtester.config_loader --validate", file=sys.stderr)
        print("  python -m src.portfolio_backtester.yaml_lint --config-check", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        print(f"\n❌ Unexpected configuration error: {e}", file=sys.stderr)
        print("Please check your configuration files for syntax errors.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "optimize" and args.scenario_name is None and args.scenario_filename is None:
        parser.error("--scenario-name or --scenario-filename is required for 'optimize' mode.")

    if args.scenario_filename:
        scenario_path = Path(args.scenario_filename)
        if not scenario_path.exists():
            parser.error(f"Scenario file not found: {args.scenario_filename}")

        try:
            scenario_from_file = config_loader_module.load_scenario_from_file(scenario_path)
            selected_scenarios = [scenario_from_file]
        except ConfigurationError as e:
            parser.error(f"Error loading scenario file: {e}")

    elif args.scenario_name is not None:
        scenario_from_config = next(
            (s for s in BACKTEST_SCENARIOS_RELOADED if s.get("name") == args.scenario_name), None
        )

        if scenario_from_config:
            selected_scenarios = [scenario_from_config]
        else:
            parser.error(
                f"Scenario '{args.scenario_name}' not found in any of the loaded configuration files."
            )
    else:
        selected_scenarios = BACKTEST_SCENARIOS_RELOADED

    if args.mode == "optimize":
        config_loader_module.load_globals_only()
        for scen in selected_scenarios:
            errors = validate_scenario_semantics(
                scen, config_loader_module.OPTIMIZER_PARAMETER_DEFAULTS
            )
            if errors:
                formatted = YamlValidator().format_errors(errors)
                formatted += "\nActionable Suggestions:\n- Verify param names match strategy.tunable_parameters()\n- Check types (int/float) and ranges\n- Add missing required params\n- See docs for more."
                print(formatted, file=sys.stderr)
                raise ConfigurationError("Optimization config invalid")

    backtester = Backtester(
        GLOBAL_CONFIG_RELOADED, selected_scenarios, args, random_state=args.random_seed
    )
    try:
        backtester.run()
    except Exception as e:
        logger.error(f"An unhandled exception occurred during backtester run: {e}", exc_info=True)
        if CENTRAL_INTERRUPTED_FLAG:
            logger.info("The error occurred after an interruption signal was received.")
            sys.exit(130)
        sys.exit(1)
    finally:
        if CENTRAL_INTERRUPTED_FLAG:
            logger.info("Backtester run finished or was terminated due to user interruption.")
            sys.exit(130)
        else:
            logger.info("Backtester run completed.")


if __name__ == "__main__":
    main()
