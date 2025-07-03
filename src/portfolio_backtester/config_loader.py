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
        OPTIMIZER_PARAMETER_DEFAULTS = params_data.get("OPTIMIZER_PARAMETER_DEFAULTS", {})

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
