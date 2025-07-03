from typing import Set
from .utils import _resolve_strategy

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

def populate_default_optimizations(scenarios: list, optimizer_parameter_defaults: dict):
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

    for scenario_config in scenarios:
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
                if param_name in optimizer_parameter_defaults:
                    scenario_config["optimize"].append({"parameter": param_name})
