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

# --- Helper functions for populating default optimizations ---
# These will be adapted from the original config.py

def _get_strategy_tunable_params(strategy_name: str) -> set[str]:
    """Resolves strategy and returns its tunable parameters."""
    # This function needs access to .utils._resolve_strategy
    # We will need to adjust this import based on the new file structure
    from .utils import _resolve_strategy
    strat_cls = _resolve_strategy(strategy_name)
    if strat_cls:
        return set(strat_cls.tunable_parameters())
    return set()

def _get_sizer_tunable_param(sizer_name: str | None, sizer_param_map: dict) -> str | None:
    """Returns the tunable parameter name for a given sizer, if applicable."""
    if sizer_name:
        return sizer_param_map.get(sizer_name)
    return None

def populate_default_optimizations():
    """Ensure each scenario has an optimize section covering all tunable
    parameters of its strategy and dynamic position sizer.
    Min/max/step values for these parameters are sourced from
    OPTIMIZER_PARAMETER_DEFAULTS at runtime by the optimizer.
    """
    sizer_param_map = {
        "rolling_sharpe": "sizer_sharpe_window",
        "rolling_sortino": "sizer_sortino_window",
        "rolling_beta": "sizer_beta_window",
        "rolling_benchmark_corr": "sizer_corr_window",
        "rolling_downside_volatility": "sizer_dvol_window",
    }

    for scenario_config in BACKTEST_SCENARIOS:
        # Ensure "optimize" list exists
        if "optimize" not in scenario_config:
            scenario_config["optimize"] = []

        optimized_parameters_in_scenario = {opt_spec["parameter"] for opt_spec in scenario_config["optimize"]}

        # Get tunable parameters from the strategy
        strategy_params_to_add = _get_strategy_tunable_params(scenario_config["strategy"])

        # Get tunable parameter from the sizer, if any
        sizer_param_to_add = _get_sizer_tunable_param(scenario_config.get("position_sizer"), sizer_param_map)

        # Combine all potential parameters to be added
        all_potential_params = strategy_params_to_add
        if sizer_param_to_add:
            all_potential_params.add(sizer_param_to_add)

        # Add missing parameters to the scenario's "optimize" list
        for param_name in all_potential_params:
            if param_name not in optimized_parameters_in_scenario:
                # Ensure the parameter exists in OPTIMIZER_PARAMETER_DEFAULTS before adding
                if param_name in OPTIMIZER_PARAMETER_DEFAULTS:
                    scenario_config["optimize"].append({"parameter": param_name})
                # If not in defaults, it might be an error or a parameter that doesn't need min/max/step from defaults.
                # For now, only add if it's a known optimizable parameter from the defaults.
                # This behavior matches the original config.py's implicit reliance on defaults being present.

# Load configuration when the module is imported
load_config()
populate_default_optimizations()

if __name__ == '__main__':
    # For testing the loader
    print("GLOBAL_CONFIG:", GLOBAL_CONFIG)
    print("\nOPTIMIZER_PARAMETER_DEFAULTS:", OPTIMIZER_PARAMETER_DEFAULTS)
    print("\nBACKTEST_SCENARIOS (first scenario):", BACKTEST_SCENARIOS[0] if BACKTEST_SCENARIOS else "None")
    # Print optimize section of the first scenario to check population
    if BACKTEST_SCENARIOS and "optimize" in BACKTEST_SCENARIOS[0]:
        print("\nFirst scenario 'optimize' section after population:")
        for opt in BACKTEST_SCENARIOS[0]["optimize"]:
            print(opt)
