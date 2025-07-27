import argparse
import logging
import sys
import warnings

from .core import Backtester
from .utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from .utils import register_signal_handler as register_central_signal_handler
from .config_loader import ConfigurationError
import src.portfolio_backtester.config_loader as config_loader_module

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
        sys.stderr.reconfigure(encoding='utf-8')  # type: ignore
        
    register_central_signal_handler()

    parser = argparse.ArgumentParser(description="Run portfolio backtester.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--mode", type=str, required=True, choices=["backtest", "optimize"], help="Mode to run the backtester in.")
    parser.add_argument("--scenario-name", type=str, help="Name of the scenario to run/optimize from BACKTEST_SCENARIOS. Required for all modes.")
    parser.add_argument("--study-name", type=str, help="Name of the Optuna study to use for optimization or to load best parameters from.")
    parser.add_argument("--storage-url", type=str, help="Optuna storage URL. If not provided, SQLite will be used.")
    parser.add_argument("--random-seed", type=int, default=None, help="Set a random seed for reproducibility.")
    parser.add_argument("--optimize-min-positions", type=int, default=10, help="Minimum number of positions to consider during optimization of num_holdings.")
    parser.add_argument("--optimize-max-positions", type=int, default=30, help="Maximum number of positions to consider during optimization of num_holdings.")
    parser.add_argument("--top-n-params", type=int, default=3,
                        help="Number of top performing parameter values to keep per grid.")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Parallel worker processes to use (-1 ⇒ all cores).")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Stop optimisation after N successive ~zero-return evaluations.")
    parser.add_argument("--optuna-trials", type=int, default=200,
                        help="Maximum trials per optimization.")
    parser.add_argument("--optuna-timeout-sec", type=int, default=None,
                        help="Time budget per optimization (seconds).")
    parser.add_argument("--optimizer", type=str, default="optuna", choices=["optuna", "genetic"],
                        help="Optimizer to use ('optuna' or 'genetic'). Default: optuna.")
    parser.add_argument("--pruning-enabled", action="store_true", help="Enable trial pruning with MedianPruner (Optuna only). Default: False.")
    parser.add_argument("--pruning-n-startup-trials", type=int, default=5, help="MedianPruner: Number of trials to complete before pruning begins. Default: 5.")
    parser.add_argument("--pruning-n-warmup-steps", type=int, default=0, help="MedianPruner: Number of intermediate steps (walk-forward windows) to observe before pruning a trial. Default: 0.")
    parser.add_argument("--pruning-interval-steps", type=int, default=1, help="MedianPruner: Report intermediate value and check for pruning every X walk-forward windows. Default: 1.")
    parser.add_argument("--mc-simulations", type=int, default=1000, help="Number of simulations for Monte Carlo analysis.")
    parser.add_argument("--mc-years", type=int, default=10, help="Number of years to project in Monte Carlo analysis.")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively (blocks execution). Default: off, only saves plots.")
    parser.add_argument("--timeout", type=int, default=None, help="Global timeout in seconds for the entire run.")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        config_loader_module.load_config()
        GLOBAL_CONFIG_RELOADED = config_loader_module.GLOBAL_CONFIG
        BACKTEST_SCENARIOS_RELOADED = config_loader_module.BACKTEST_SCENARIOS
        
        logger.info("Configuration loaded successfully")
        
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

    if args.mode == "optimize" and args.scenario_name is None:
        parser.error("--scenario-name is required for 'optimize' mode.")

    if args.scenario_name is not None:
        scenario_name = args.scenario_name.strip("'\"")
        selected_scenarios = [s for s in BACKTEST_SCENARIOS_RELOADED if s["name"] == scenario_name]
        if not selected_scenarios:
            parser.error(f"Scenario '{scenario_name}' not found in BACKTEST_SCENARIOS_RELOADED.")
    else:
        selected_scenarios = BACKTEST_SCENARIOS_RELOADED

    backtester = Backtester(GLOBAL_CONFIG_RELOADED, selected_scenarios, args, random_state=args.random_seed)
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