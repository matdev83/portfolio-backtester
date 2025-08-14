import logging
import sys
from pathlib import Path
import os
from typing import Any

# Import our YAML validator
from .yaml_validator import validate_yaml_file, YamlValidator
from .scenario_validator import validate_scenario_semantics
from .config_cache import get_cache


def get_ga_optimizer_parameter_defaults():
    """Returns default parameters for GA specific settings."""
    return {
        "ga_num_generations": {
            "default": 100,
            "type": "int",
            "low": 10,
            "high": 1000,
            "help": "Number of generations for GA.",
        },
        "ga_num_parents_mating": {
            "default": 10,
            "type": "int",
            "low": 2,
            "high": 50,
            "help": "Number of parents to mate in GA.",
        },
        "ga_sol_per_pop": {
            "default": 50,
            "type": "int",
            "low": 10,
            "high": 200,
            "help": "Solutions per population in GA.",
        },
        "ga_parent_selection_type": {
            "default": "sss",
            "type": "categorical",
            "values": ["sss", "rws", "sus", "rank", "random", "tournament"],
            "help": "Parent selection type for GA.",
        },
        "ga_crossover_type": {
            "default": "single_point",
            "type": "categorical",
            "values": ["single_point", "two_points", "uniform", "scattered"],
            "help": "Crossover type for GA.",
        },
        "ga_advanced_crossover_type": {
            "default": None,
            "type": "categorical",
            "values": [
                None,
                "simulated_binary",
                "multi_point",
                "uniform_variant",
                "arithmetic",
            ],
            "help": "Advanced crossover operator for GA.",
        },
        "ga_sbx_distribution_index": {
            "default": 20.0,
            "type": "float",
            "low": 1.0,
            "high": 100.0,
            "help": "Distribution index for Simulated Binary Crossover.",
        },
        "ga_num_crossover_points": {
            "default": 3,
            "type": "int",
            "low": 2,
            "high": 10,
            "help": "Number of crossover points for Multi-point crossover.",
        },
        "ga_uniform_crossover_bias": {
            "default": 0.5,
            "type": "float",
            "low": 0.1,
            "high": 0.9,
            "help": "Bias parameter for Uniform crossover variant.",
        },
        "ga_mutation_type": {
            "default": "random",
            "type": "categorical",
            "values": ["random", "swap", "inversion", "scramble", "adaptive"],
            "help": "Mutation type for GA.",
        },
        "ga_mutation_percent_genes": {
            "default": "default",
            "type": "str",
            "help": "Percentage of genes to mutate (e.g., 'default', 10 for 10%).",
        },
    }


# Set up logger
logger = logging.getLogger(__name__)

# Global configuration variables
GLOBAL_CONFIG = {}
OPTIMIZER_PARAMETER_DEFAULTS: dict[str, Any] = {}
BACKTEST_SCENARIOS: list[dict[str, Any]] = []

# Configuration file paths
PARAMETERS_FILE = Path(__file__).parent.parent.parent / "config" / "parameters.yaml"
SCENARIOS_DIR = Path(__file__).parent.parent.parent / "config" / "scenarios"


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    pass


def is_example_file(scenario_file: str, root_path: str) -> bool:
    """
    Determine if a YAML scenario file should be skipped as an example file.

    Args:
        scenario_file: The filename of the YAML file
        root_path: The directory path containing the file

    Returns:
        True if the file should be skipped (is an example file), False otherwise
    """
    scenario_path_str = str(Path(root_path) / scenario_file).lower()
    return "example" in scenario_file.lower() or "example" in scenario_path_str


def merge_optimizer_config(scenario_data, optimizer_type):
    if "optimizers" in scenario_data:
        optimizer_config = scenario_data["optimizers"].get(optimizer_type)
        if optimizer_config:
            for key, value in optimizer_config.items():
                if key not in scenario_data:
                    scenario_data[key] = value
                elif isinstance(scenario_data[key], dict) and isinstance(value, dict):
                    scenario_data[key].update(value)
                else:
                    scenario_data[key] = value
        del scenario_data["optimizers"]
    return scenario_data


def load_scenario_from_file(file_path: Path) -> dict:
    """
    Loads a single scenario from a specific YAML file.

    Args:
        file_path: The path to the scenario file.

    Returns:
        The loaded scenario data.

    Raises:
        ConfigurationError: If the file is invalid or cannot be loaded.
    """
    logger.info(f"Loading scenario from file: {file_path}")
    is_valid, scenario_data, error_message = validate_yaml_file(file_path)

    if not is_valid:
        logger.error(f"Scenario file validation failed: {file_path}")
        print(error_message, file=sys.stderr)
        raise ConfigurationError(f"Invalid scenario file: {file_path}. See error details above.")

    if scenario_data is None:
        raise ConfigurationError(f"Scenario file is empty: {file_path}")

    # ------------------------------------------------------------------
    # Semantic validation (optimizer settings, strategy parameters, etc.)
    # ------------------------------------------------------------------
    semantic_errors = validate_scenario_semantics(
        scenario_data,
        optimizer_parameter_defaults=OPTIMIZER_PARAMETER_DEFAULTS,
        file_path=file_path,
    )
    if semantic_errors:
        non_blocking = []
        blocking = []
        for err in semantic_errors:
            msg = getattr(err, "message", str(err))
            if "position_sizer" in msg and "Invalid position_sizer" in msg:
                non_blocking.append(err)
            else:
                blocking.append(err)
        if blocking:
            formatted = YamlValidator().format_errors(blocking)
            print(formatted, file=sys.stderr)
            raise ConfigurationError(
                f"Semantic validation failed for scenario file: {file_path}. See error details above."
            )
        else:
            for err in non_blocking:
                logger.warning(getattr(err, "message", str(err)))

    # Flatten optimizer-specific config (if present) into top-level keys for easier consumption
    if "optimizers" in scenario_data:
        # Prefer optuna if both present – matches current default optimiser
        preferred = (
            "optuna"
            if "optuna" in scenario_data["optimizers"]
            else next(iter(scenario_data["optimizers"].keys()))
        )
        scenario_data = merge_optimizer_config(scenario_data, preferred)

    # Normalise path separators in scenario name if present
    if "name" in scenario_data and isinstance(scenario_data["name"], str):
        scenario_data["name"] = scenario_data["name"].replace("\\", "/")

    return scenario_data


def load_config():
    """
    Loads configurations from YAML files with comprehensive error handling.

    Raises:
        ConfigurationError: If configuration files are invalid or corrupted
    """
    global GLOBAL_CONFIG, OPTIMIZER_PARAMETER_DEFAULTS, BACKTEST_SCENARIOS

    try:
        # Load and validate parameters file
        import sys as _sys

        _this_mod = _sys.modules.get(__name__)
        parameters_file_path = getattr(_this_mod, "PARAMETERS_FILE", PARAMETERS_FILE)
        logger.info(f"Loading parameters from: {parameters_file_path}")
        # Validate the currently configured parameters file path
        is_valid, params_data, error_message = validate_yaml_file(parameters_file_path)

        if not is_valid:
            logger.error(f"Parameters file validation failed: {parameters_file_path}")
            print(error_message, file=sys.stderr)
            # Keep message compatible with tests expecting 'validation failed'
            raise ConfigurationError("parameters.yaml validation failed. See error details above.")

        if params_data is None:
            raise ConfigurationError(f"Parameters file is empty: {parameters_file_path}")

        # Extract configuration sections with validation
        GLOBAL_CONFIG = params_data.get("GLOBAL_CONFIG", {})
        if not GLOBAL_CONFIG:
            raise ConfigurationError(
                "'GLOBAL_CONFIG' section is missing or empty in parameters.yaml."
            )

        # Add Monte Carlo and WFO robustness configurations to global config
        GLOBAL_CONFIG["monte_carlo_config"] = params_data.get("monte_carlo_config", {})
        GLOBAL_CONFIG["wfo_robustness_config"] = params_data.get("wfo_robustness_config", {})

        # Load optimizer parameter defaults
        OPTIMIZER_PARAMETER_DEFAULTS = params_data.get("OPTIMIZER_PARAMETER_DEFAULTS", {})
        if not OPTIMIZER_PARAMETER_DEFAULTS:
            raise ConfigurationError(
                "'OPTIMIZER_PARAMETER_DEFAULTS' section is missing or empty in parameters.yaml."
            )

        # Update with GA defaults, allowing GA defaults to be overridden by parameters.yaml if specified there
        try:
            ga_defaults = get_ga_optimizer_parameter_defaults()
            for key, value in ga_defaults.items():
                if key not in OPTIMIZER_PARAMETER_DEFAULTS:  # Add if not present
                    OPTIMIZER_PARAMETER_DEFAULTS[key] = value
                elif isinstance(OPTIMIZER_PARAMETER_DEFAULTS[key], dict) and isinstance(
                    value, dict
                ):  # Merge if both are dicts
                    OPTIMIZER_PARAMETER_DEFAULTS[key].update(value)
                # If key exists in OPTIMIZER_PARAMETER_DEFAULTS but is not a dict, it means it was overridden in parameters.yaml
                # and we keep that overridden value.
        except Exception as e:
            logger.warning(f"Failed to load GA defaults: {e}")

        # Load and validate scenario files from subdirectories
        logger.info(f"Loading scenarios from: {SCENARIOS_DIR}")
        BACKTEST_SCENARIOS = []
        if SCENARIOS_DIR.exists() and SCENARIOS_DIR.is_dir():
            # Walk through all subdirectories to find YAML files
            for root, _, files in os.walk(SCENARIOS_DIR):
                for scenario_file in files:
                    if scenario_file.endswith(".yaml"):
                        # Skip files with 'example' in the filename or path
                        if is_example_file(scenario_file, root):
                            logger.info(f"Skipping example file: {scenario_file}")
                            continue

                        from pathlib import Path as _Path

                        scenario_path = _Path(root) / scenario_file
                        is_valid, scenario_data, error_message = validate_yaml_file(scenario_path)
                        if not is_valid:
                            logger.error(f"Scenario file validation failed: {scenario_path}")
                            print(error_message, file=sys.stderr)
                            raise ConfigurationError(
                                f"Invalid scenario file: {scenario_file}. See error details above."
                            )
                        if scenario_data is None:
                            raise ConfigurationError(f"Scenario file is empty: {scenario_path}")

                        # Semantic validation
                        semantic_errors = validate_scenario_semantics(
                            scenario_data,
                            optimizer_parameter_defaults=OPTIMIZER_PARAMETER_DEFAULTS,
                            file_path=scenario_path,
                        )
                        if semantic_errors:
                            non_blocking = []
                            blocking = []
                            for err in semantic_errors:
                                msg = getattr(err, "message", str(err))
                                if "position_sizer" in msg and "Invalid position_sizer" in msg:
                                    non_blocking.append(err)
                                else:
                                    blocking.append(err)
                            if blocking:
                                formatted = YamlValidator().format_errors(blocking)
                                print(formatted, file=sys.stderr)
                                raise ConfigurationError(
                                    f"Semantic validation failed for scenario file: {scenario_path}. See error details above."
                                )
                            else:
                                for err in non_blocking:
                                    logger.warning(getattr(err, "message", str(err)))

                        # Flatten optimizer section if present
                        if "optimizers" in scenario_data:
                            preferred = (
                                "optuna"
                                if "optuna" in scenario_data["optimizers"]
                                else next(iter(scenario_data["optimizers"].keys()))
                            )
                            scenario_data = merge_optimizer_config(scenario_data, preferred)

                        if "name" in scenario_data and isinstance(scenario_data["name"], str):
                            scenario_data["name"] = scenario_data["name"].replace("\\", "/")

                        BACKTEST_SCENARIOS.append(scenario_data)

        if not BACKTEST_SCENARIOS:
            logger.warning(f"No scenarios found in {SCENARIOS_DIR}.")

        # After parameters and scenarios are verified, validate strategy configs cross-references
        from pathlib import Path
        from .strategy_config_validator import validate_strategy_configs_comprehensive

        project_root = Path(__file__).parent.parent.parent
        strategies_directory = project_root / "src" / "portfolio_backtester" / "strategies"
        scenarios_directory = project_root / "config" / "scenarios"

        is_valid, validation_errors = validate_strategy_configs_comprehensive(
            str(strategies_directory), str(scenarios_directory)
        )
        if not is_valid:
            error_message = "\n".join(validation_errors)
            logger.error("Strategy configuration validation failed!")
            print(error_message, file=sys.stderr)
            raise ConfigurationError(
                "Strategy configuration validation failed. See error details above."
            )

        logger.info(f"Successfully loaded configuration: {len(BACKTEST_SCENARIOS)} scenarios found")

        # Return the loaded configuration
        return GLOBAL_CONFIG, BACKTEST_SCENARIOS

    except ConfigurationError:
        # Re-raise configuration errors as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        import traceback

        error_msg = f"Unexpected error loading configuration: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ConfigurationError(error_msg) from e


def validate_config_files(force_refresh: bool = False) -> bool:
    """
    Validate configuration files without loading them.

    Args:
        force_refresh: If True, ignores cache and forces fresh validation

    Returns:
        True if all files are valid, False otherwise
    """
    cache = get_cache()
    cache_key = "yaml_validation_full"

    # Check if we can use cached results (unless force refresh is requested)
    if not force_refresh:
        cached_result = cache.get_cached_result(cache_key)
        if cached_result is not None:
            print("Using cached validation results...")
            all_valid: bool = cached_result.get("all_valid", False)

            # Display cached results
            if all_valid:
                print("[OK] All configuration files are valid (cached)")
            else:
                print("[ERROR] Configuration validation failed (cached):")
                for error in cached_result.get("errors", []):
                    print(f"  - {error}")
            return all_valid
        else:
            # Log why cache was invalidated
            is_valid, reason = cache.is_cache_valid(cache_key)
            if not is_valid and reason:
                print(f"Running fresh validation: {reason}")
            else:
                print("Running fresh validation: no cached data")
    else:
        print("Running fresh validation: force refresh requested")

    all_valid = True
    validation_errors = []

    # First, run comprehensive strategy configuration validation
    from .strategy_config_validator import validate_strategy_configs_comprehensive

    project_root = Path(__file__).parent.parent.parent
    strategies_directory = project_root / "src" / "portfolio_backtester" / "strategies"
    scenarios_directory = project_root / "config" / "scenarios"

    print("Running comprehensive strategy configuration validation...")
    is_valid, strategy_validation_errors = validate_strategy_configs_comprehensive(
        str(strategies_directory), str(scenarios_directory), force_refresh
    )
    if is_valid:
        print("[OK] Strategy configuration cross-references are valid")
    else:
        print("[ERROR] Strategy configuration validation failed:")
        for error in strategy_validation_errors:
            print(f"  - {error}")
            validation_errors.append(f"Strategy: {error}")
        all_valid = False

    validator = YamlValidator()

    # Validate parameters file
    print(f"Validating {PARAMETERS_FILE}...")
    is_valid, _, errors = validate_yaml_file(PARAMETERS_FILE)
    if is_valid:
        print("[OK] parameters.yaml is valid")
    else:
        print("[ERROR] parameters.yaml has errors:")
        if isinstance(errors, str):
            print(errors)
            validation_errors.append(f"Parameters: {errors}")
        else:
            formatted_errors = validator.format_errors(errors)
            print(formatted_errors)
            validation_errors.append(f"Parameters: {formatted_errors}")
        all_valid = False

    # Validate scenario files in subdirectories
    print(f"\nValidating scenarios in {SCENARIOS_DIR}...")
    if SCENARIOS_DIR.exists() and SCENARIOS_DIR.is_dir():
        # Walk through all subdirectories to find YAML files
        for root, _, files in os.walk(SCENARIOS_DIR):
            for scenario_file in files:
                if scenario_file.endswith(".yaml"):
                    # Skip files with 'example' in the filename or path
                    if is_example_file(scenario_file, root):
                        print(f"Skipping example file: {scenario_file}")
                        continue

                    scenario_path = Path(root) / scenario_file
                    print(f"Validating {scenario_path}...")
                    is_valid, _, errors = validate_yaml_file(scenario_path)
                    if is_valid:
                        print(f"[OK] {scenario_file} is valid")
                    else:
                        print(f"[ERROR] {scenario_file} has errors:")
                        if isinstance(errors, str):
                            print(errors)
                            validation_errors.append(f"Scenario {scenario_file}: {errors}")
                        else:
                            formatted_errors = validator.format_errors(errors)
                            print(formatted_errors)
                            validation_errors.append(
                                f"Scenario {scenario_file}: {formatted_errors}"
                            )
                        all_valid = False

    # Cache the validation results
    result = {"all_valid": all_valid, "errors": validation_errors}
    cache.store_result(result, cache_key)

    if all_valid:
        print("\nValidation completed successfully")
    else:
        print(f"\nValidation completed with {len(validation_errors)} total issues")

    return all_valid


def safe_load_config() -> bool:
    """
    Safely load configuration with error handling.

    Returns:
        True if configuration loaded successfully, False otherwise
    """
    try:
        load_config()
        return True
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return False


# Load configuration when the module is imported
try:
    load_config()
except ConfigurationError as e:
    logger.error(f"Failed to load configuration on import: {e}")
    # Don't raise here to allow the module to be imported for validation purposes
except Exception as e:
    logger.error(f"Unexpected error loading configuration on import: {e}")

if __name__ == "__main__":
    # CLI for testing and validation
    import argparse

    parser = argparse.ArgumentParser(description="Configuration loader and validator")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration files without loading",
    )
    parser.add_argument("--show-config", action="store_true", help="Show loaded configuration")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh validation, ignoring cache",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear validation cache")
    parser.add_argument("--cache-info", action="store_true", help="Show cache information")

    args = parser.parse_args()

    if args.clear_cache:
        from .config_cache import get_cache

        cache = get_cache()
        cache.clear_cache()
        print("Configuration validation cache cleared.")
        sys.exit(0)

    elif args.cache_info:
        from .config_cache import get_cache

        cache = get_cache()
        info = cache.get_cache_info()
        print("\nCache Information:")
        print(f"Cache file: {info['cache_file']}")
        print(f"Cache exists: {info['cache_exists']}")
        print(f"Cached keys: {info['cached_keys']}")
        print("\nMonitored directories:")
        for name, path in info["monitored_directories"].items():
            print(f"  {name}: {path}")

        # Show file counts for each cached key
        for key in info["cached_keys"]:
            timestamp_key = f"{key}_timestamp"
            counts_key = f"{key}_file_counts"
            if timestamp_key in info:
                import datetime

                dt = datetime.datetime.fromtimestamp(info[timestamp_key])
                print(f"\n{key} cache:")
                print(f"  Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                if counts_key in info:
                    print(f"  File counts: {info[counts_key]}")
        sys.exit(0)

    elif args.validate:
        print("Validating configuration files...")
        force_refresh = args.force_refresh
        if validate_config_files(force_refresh):
            print("\n[OK] All configuration files are valid!")
            sys.exit(0)
        else:
            print("\n[ERROR] Configuration validation failed!")
            sys.exit(1)

    elif args.show_config:
        if safe_load_config():
            print("GLOBAL_CONFIG:", GLOBAL_CONFIG)
            print("\nOPTIMIZER_PARAMETER_DEFAULTS:", OPTIMIZER_PARAMETER_DEFAULTS)
            print(
                "\nBACKTEST_SCENARIOS (first scenario):",
                BACKTEST_SCENARIOS[0] if BACKTEST_SCENARIOS else "None",
            )
        else:
            print("Failed to load configuration!")
            sys.exit(1)

    else:
        print(
            "Configuration loader and validator\n"
            "\nOptions:"
            "\n  --validate       Validate configuration files"
            "\n  --force-refresh  Force validation refresh (ignore cache)"
            "\n  --show-config    Display loaded configuration"
            "\n  --cache-info     Show validation cache information"
            "\n  --clear-cache    Clear validation cache"
            "\n  --help          Show detailed help"
        )
