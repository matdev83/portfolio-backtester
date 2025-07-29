from typing import Set
from .utils import _resolve_strategy
from .api_stability import api_stable

def _get_strategy_tunable_params(strategy_name: str) -> Set[str]:
    """Resolves strategy and returns its tunable parameters."""
    strat_cls = _resolve_strategy(strategy_name)
    if strat_cls:
        return set(strat_cls.tunable_parameters())
    return set()

def _get_sizer_tunable_param(sizer_name: str | None, sizer_param_map: dict) -> str | None:
    """Returns the tunable parameter name for a given sizer, if applicable."""
    if sizer_name:
        return sizer_param_map.get(sizer_name)
    return None

@api_stable(version="1.0", strict_params=True, strict_return=False)
def populate_default_optimizations(scenarios: list, optimizer_parameter_defaults: dict):
    """Ensure each scenario has an optimize section covering all tunable
    parameters of its strategy and dynamic position sizer.
    Min/max/step values for these parameters are sourced from
    OPTIMIZER_PARAMETER_DEFAULTS at runtime by the optimizer.
    Also populate strategy_params with default values for optimization parameters.
    """
    sizer_param_map = {
        "rolling_sharpe": "sizer_sharpe_window",
        "rolling_sortino": "sizer_sortino_window",
        "rolling_beta": "sizer_beta_window",
        "rolling_benchmark_corr": "sizer_corr_window",
        "rolling_downside_volatility": "sizer_dvol_window",
    }

    for scenario_config in scenarios:
        # Ensure "optimize" list exists
        if "optimize" not in scenario_config:
            scenario_config["optimize"] = []
        
        # Ensure "strategy_params" exists
        if "strategy_params" not in scenario_config:
            scenario_config["strategy_params"] = {}

        optimized_parameters_in_scenario = {opt_spec["parameter"] for opt_spec in scenario_config["optimize"]}
        
        # Validate existing optimization parameters against strategy tunable parameters
        strategy_name = scenario_config.get("strategy", "unknown")
        strategy_tunable_params = _get_strategy_tunable_params(strategy_name)
        sizer_param = _get_sizer_tunable_param(scenario_config.get("position_sizer"), sizer_param_map)
        
        # Add sizer parameter to valid parameters if applicable
        valid_params = strategy_tunable_params.copy()
        if sizer_param:
            valid_params.add(sizer_param)
        
        # Add special scenario config keys that are always valid
        valid_params.add("position_sizer")
        
        # Check for invalid parameters in existing optimization specs
        invalid_params = []
        for param_name in optimized_parameters_in_scenario:
            if param_name not in valid_params and param_name not in optimizer_parameter_defaults:
                invalid_params.append(param_name)
            elif param_name not in valid_params:
                # Parameter exists in defaults but not tunable by strategy
                print(f"Warning: Parameter '{param_name}' in scenario '{scenario_config.get('name', 'unnamed')}' "
                      f"is not tunable by strategy '{strategy_name}'. It will be skipped during optimization.")
        
        if invalid_params:
            print(f"Warning: Scenario '{scenario_config.get('name', 'unnamed')}' contains invalid parameters "
                  f"not defined in OPTIMIZER_PARAMETER_DEFAULTS: {invalid_params}")
        

        # Only add missing parameters if the scenario has no optimize list defined originally
        # This preserves scenarios that intentionally want to optimize only a subset of parameters
        original_optimize_list = scenario_config.get("optimize", [])
        
        # Check if this scenario originally had an optimize list defined
        # If it did, don't add any missing parameters - respect the explicit choice
        if len(original_optimize_list) > 0:
            # Scenario has explicit optimization parameters - don't add any missing ones
            # Just populate strategy_params with defaults for existing optimization parameters
            for param_name in optimized_parameters_in_scenario:
                if param_name not in scenario_config["strategy_params"] and param_name in optimizer_parameter_defaults:
                    param_config = optimizer_parameter_defaults[param_name]
                    default_value = _get_default_value_for_parameter(param_config)
                    scenario_config["strategy_params"][param_name] = default_value
        else:
            # Scenario has no optimize list - add all tunable parameters
            # Get tunable parameters from the strategy
            strategy_params_to_add = _get_strategy_tunable_params(scenario_config["strategy"])

            # Get tunable parameter from the sizer, if any
            sizer_param_to_add = _get_sizer_tunable_param(scenario_config.get("position_sizer"), sizer_param_map)

            # Combine all potential parameters to be added
            all_potential_params = strategy_params_to_add
            if sizer_param_to_add:
                all_potential_params.add(sizer_param_to_add)

            # Add missing parameters to the scenario's "optimize" list and populate strategy_params with defaults
            for param_name in all_potential_params:
                if param_name not in optimized_parameters_in_scenario:
                    # Ensure the parameter exists in OPTIMIZER_PARAMETER_DEFAULTS before adding
                    if param_name in optimizer_parameter_defaults:
                        scenario_config["optimize"].append({"parameter": param_name})
                
                # Also populate strategy_params with default values if not already present
                if param_name not in scenario_config["strategy_params"] and param_name in optimizer_parameter_defaults:
                    param_config = optimizer_parameter_defaults[param_name]
                    default_value = _get_default_value_for_parameter(param_config)
                    scenario_config["strategy_params"][param_name] = default_value


def _get_default_value_for_parameter(param_config: dict):
    """Get a reasonable default value for a parameter based on its configuration."""
    param_type = param_config.get("type")
    
    if param_type == "int":
        # Use the low value as default for int parameters
        return param_config.get("low", 1)
    elif param_type == "float":
        # Use the low value as default for float parameters
        return param_config.get("low", 0.0)
    elif param_type == "categorical":
        # Use the first value as default for categorical parameters
        values = param_config.get("values", [])
        return values[0] if values else None
    else:
        # Fallback to low value or None
        return param_config.get("low")
