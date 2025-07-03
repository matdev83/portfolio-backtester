import yaml
import os
from pathlib import Path

# Define paths to the configuration files
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
PARAMETERS_FILE = CONFIG_DIR / "parameters.yaml"
SCENARIOS_FILE = CONFIG_DIR / "scenarios.yaml"

# Initialize config variables
GLOBAL_CONFIG = {}
OPTIMIZER_PARAMETER_DEFAULTS = {}
BACKTEST_SCENARIOS = []

# Import GA defaults
from portfolio_backtester.optimization.genetic_optimizer import get_ga_optimizer_parameter_defaults

def load_config():
    """Loads configurations from YAML files."""
    global GLOBAL_CONFIG, OPTIMIZER_PARAMETER_DEFAULTS, BACKTEST_SCENARIOS

    if not PARAMETERS_FILE.exists():
        raise FileNotFoundError(f"Parameters file not found: {PARAMETERS_FILE}")
    if not SCENARIOS_FILE.exists():
        raise FileNotFoundError(f"Scenarios file not found: {SCENARIOS_FILE}")

    with open(PARAMETERS_FILE, 'r') as f:
        params_data = yaml.safe_load(f)
        GLOBAL_CONFIG = params_data.get("GLOBAL_CONFIG", {})
        # Start with Optuna defaults from parameters.yaml
        OPTIMIZER_PARAMETER_DEFAULTS = params_data.get("OPTIMIZER_PARAMETER_DEFAULTS", {})
        # Update with GA defaults, allowing GA defaults to be overridden by parameters.yaml if specified there
        ga_defaults = get_ga_optimizer_parameter_defaults()
        for key, value in ga_defaults.items():
            if key not in OPTIMIZER_PARAMETER_DEFAULTS: # Add if not present
                OPTIMIZER_PARAMETER_DEFAULTS[key] = value
            elif isinstance(OPTIMIZER_PARAMETER_DEFAULTS[key], dict) and isinstance(value, dict): # Merge if both are dicts
                OPTIMIZER_PARAMETER_DEFAULTS[key].update(value)
            # If key exists in OPTIMIZER_PARAMETER_DEFAULTS but is not a dict, it means it was overridden in parameters.yaml
            # and we keep that overridden value.

    with open(SCENARIOS_FILE, 'r') as f:
        scenarios_data = yaml.safe_load(f)
        BACKTEST_SCENARIOS = scenarios_data.get("BACKTEST_SCENARIOS", [])

    # populate_default_optimizations() # We will add this later

# Load configuration when the module is imported
load_config()

if __name__ == '__main__':
    # For testing the loader
    print("GLOBAL_CONFIG:", GLOBAL_CONFIG)
    print("\nOPTIMIZER_PARAMETER_DEFAULTS:", OPTIMIZER_PARAMETER_DEFAULTS)
    print("\nBACKTEST_SCENARIOS (first scenario):", BACKTEST_SCENARIOS[0] if BACKTEST_SCENARIOS else "None")
