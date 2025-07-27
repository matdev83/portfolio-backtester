import logging
import sys
from pathlib import Path

import logging
import sys
from pathlib import Path

# Import our YAML validator
from .optimization.genetic_optimizer import get_ga_optimizer_parameter_defaults
from .yaml_validator import validate_yaml_file

# Set up logger
logger = logging.getLogger(__name__)

# Global configuration variables
GLOBAL_CONFIG = {}
OPTIMIZER_PARAMETER_DEFAULTS = {}
BACKTEST_SCENARIOS = []

# Configuration file paths
PARAMETERS_FILE = Path(__file__).parent.parent.parent / "config" / "parameters.yaml"
SCENARIOS_FILE = Path(__file__).parent.parent.parent / "config" / "scenarios.yaml"


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


def load_config():
    """
    Loads configurations from YAML files with comprehensive error handling.
    
    Raises:
        ConfigurationError: If configuration files are invalid or corrupted
    """
    global GLOBAL_CONFIG, OPTIMIZER_PARAMETER_DEFAULTS, BACKTEST_SCENARIOS

    try:
        # Load and validate parameters file
        logger.info(f"Loading parameters from: {PARAMETERS_FILE}")
        is_valid, params_data, error_message = validate_yaml_file(PARAMETERS_FILE)
        
        if not is_valid:
            logger.error(f"Parameters file validation failed: {PARAMETERS_FILE}")
            print(error_message, file=sys.stderr)
            raise ConfigurationError("Invalid parameters.yaml file. See error details above.")
        
        if params_data is None:
            raise ConfigurationError(f"Parameters file is empty: {PARAMETERS_FILE}")
        
        # Extract configuration sections with validation
        GLOBAL_CONFIG = params_data.get("GLOBAL_CONFIG", {})
        if not GLOBAL_CONFIG:
            logger.warning("GLOBAL_CONFIG section is missing or empty in parameters.yaml")
        
        # Add Monte Carlo and WFO robustness configurations to global config
        GLOBAL_CONFIG["monte_carlo_config"] = params_data.get("monte_carlo_config", {})
        GLOBAL_CONFIG["wfo_robustness_config"] = params_data.get("wfo_robustness_config", {})
        
        # Load optimizer parameter defaults
        OPTIMIZER_PARAMETER_DEFAULTS = params_data.get("OPTIMIZER_PARAMETER_DEFAULTS", {})
        if not OPTIMIZER_PARAMETER_DEFAULTS:
            logger.warning("OPTIMIZER_PARAMETER_DEFAULTS section is missing or empty in parameters.yaml")
        
        # Update with GA defaults, allowing GA defaults to be overridden by parameters.yaml if specified there
        try:
            ga_defaults = get_ga_optimizer_parameter_defaults()
            for key, value in ga_defaults.items():
                if key not in OPTIMIZER_PARAMETER_DEFAULTS: # Add if not present
                    OPTIMIZER_PARAMETER_DEFAULTS[key] = value
                elif isinstance(OPTIMIZER_PARAMETER_DEFAULTS[key], dict) and isinstance(value, dict): # Merge if both are dicts
                    OPTIMIZER_PARAMETER_DEFAULTS[key].update(value)
                # If key exists in OPTIMIZER_PARAMETER_DEFAULTS but is not a dict, it means it was overridden in parameters.yaml
                # and we keep that overridden value.
        except Exception as e:
            logger.warning(f"Failed to load GA defaults: {e}")

        # Load and validate scenarios file
        logger.info(f"Loading scenarios from: {SCENARIOS_FILE}")
        is_valid, scenarios_data, error_message = validate_yaml_file(SCENARIOS_FILE)
        
        if not is_valid:
            logger.error(f"Scenarios file validation failed: {SCENARIOS_FILE}")
            print(error_message, file=sys.stderr)
            raise ConfigurationError("Invalid scenarios.yaml file. See error details above.")
        
        if scenarios_data is None:
            raise ConfigurationError(f"Scenarios file is empty: {SCENARIOS_FILE}")
        
        # Extract scenarios with validation
        BACKTEST_SCENARIOS = scenarios_data.get("BACKTEST_SCENARIOS", [])
        if not BACKTEST_SCENARIOS:
            logger.warning("BACKTEST_SCENARIOS section is missing or empty in scenarios.yaml")
        
        logger.info(f"Successfully loaded configuration: {len(BACKTEST_SCENARIOS)} scenarios found")
        
        # Return the loaded configuration
        return GLOBAL_CONFIG, BACKTEST_SCENARIOS
        
    except ConfigurationError:
        # Re-raise configuration errors as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error loading configuration: {str(e)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


def validate_config_files() -> bool:
    """
    Validate configuration files without loading them.
    
    Returns:
        True if all files are valid, False otherwise
    """
    all_valid = True
    
    # Validate parameters file
    print(f"Validating {PARAMETERS_FILE}...")
    is_valid, _, errors = validator.validate_file(PARAMETERS_FILE)
    if is_valid:
        print("[OK] parameters.yaml is valid")
    else:
        print("[ERROR] parameters.yaml has errors:")
        print(validator.format_errors(errors))
        all_valid = False
    
    # Validate scenarios file
    print(f"\nValidating {SCENARIOS_FILE}...")
    is_valid, _, errors = validator.validate_file(SCENARIOS_FILE)
    if is_valid:
        print("[OK] scenarios.yaml is valid")
    else:
        print("[ERROR] scenarios.yaml has errors:")
        print(validator.format_errors(errors))
        all_valid = False
    
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

    # populate_default_optimizations() # We will add this later

# Load configuration when the module is imported
try:
    load_config()
except ConfigurationError as e:
    logger.error(f"Failed to load configuration on import: {e}")
    # Don't raise here to allow the module to be imported for validation purposes
except Exception as e:
    logger.error(f"Unexpected error loading configuration on import: {e}")

if __name__ == '__main__':
    # CLI for testing and validation
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration loader and validator')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate configuration files without loading')
    parser.add_argument('--show-config', action='store_true',
                       help='Show loaded configuration')
    
    args = parser.parse_args()
    
    if args.validate:
        print("Validating configuration files...")
        if validate_config_files():
            print("\n[OK] All configuration files are valid!")
            sys.exit(0)
        else:
            print("\n[ERROR] Configuration validation failed!")
            sys.exit(1)
    
    elif args.show_config:
        if safe_load_config():
            print("GLOBAL_CONFIG:", GLOBAL_CONFIG)
            print("\nOPTIMIZER_PARAMETER_DEFAULTS:", OPTIMIZER_PARAMETER_DEFAULTS)
            print("\nBACKTEST_SCENARIOS (first scenario):", BACKTEST_SCENARIOS[0] if BACKTEST_SCENARIOS else "None")
        else:
            print("Failed to load configuration!")
            sys.exit(1)
    
    else:
        print("Use --validate to check configuration files or --show-config to display loaded configuration")
        print("Run with --help for more options")
