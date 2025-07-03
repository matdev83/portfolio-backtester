from typing import Dict, Any, Tuple
import optuna
import pandas as pd
import numpy as np

from ..utils import _run_scenario_static
from ..reporting.performance_metrics import calculate_metrics
from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from ..constants import ZERO_RET_EPS


def build_objective(
    g_cfg: Dict,
    base_scen_cfg: Dict,
    train_data_monthly,
    train_data_daily,
    train_rets_daily,
    bench_series_daily,
    features_slice,
    # metric: str = "Calmar", # Metric is now part of base_scen_cfg
):
    """
    Factory to build a customized Optuna objective function.
    """
    optimizer_config = OPTIMIZER_PARAMETER_DEFAULTS

    # Extract optimization specifications from base_scen_cfg
    optimization_specs = base_scen_cfg.get("optimize", [])
    # Get the optimization metric from the scenario configuration
    optimization_targets_config = base_scen_cfg.get("optimization_targets")
    single_metric_to_optimize = base_scen_cfg.get("optimization_metric")
    constraints_config = base_scen_cfg.get("optimization_constraints", [])

    is_multi_objective = bool(optimization_targets_config)

    if is_multi_objective:
        metrics_to_optimize = [target["name"] for target in optimization_targets_config]
        metric_directions = [
            target.get("direction", "maximize").lower()
            for target in optimization_targets_config
        ]
    elif single_metric_to_optimize:
        metrics_to_optimize = [single_metric_to_optimize]
        metric_directions = ["maximize"]
    else:
        # Default to Calmar if no optimization metric is specified at all
        metrics_to_optimize = ["Calmar"]
        metric_directions = ["maximize"]
        # Ensure this default is reflected in study creation if it reaches that far
        # For single objective, Optuna defaults to 'maximize' if direction isn't specified,
        # but our old code implicitly maximized.

    def objective(
        trial: optuna.trial.Trial,
    ) -> Any:  # Return type can be float or Tuple[float, ...]
        # 1 ─ suggest parameters ----------------------------------------
        # p holds parameters that will go into strategy_params
        p = base_scen_cfg["strategy_params"].copy()
        # scen_cfg_overrides holds parameters that will directly set/override keys in scen_cfg
        scen_cfg_overrides = {}

        # Define keys that, if optimized, should directly modify the scenario config
        # instead of just strategy_params.
        # Example: "position_sizer" is a key in scen_cfg, not scen_cfg["strategy_params"].
        # If we optimize "position_sizer", the chosen value should go to scen_cfg["position_sizer"].
        SPECIAL_SCEN_CFG_KEYS = ["position_sizer"] # Add other keys like "signal_generator_class" if needed

        for opt_spec in optimization_specs:
            param_name = opt_spec["parameter"]
            default_param_config = optimizer_config.get(param_name, {})

            # Type: opt_spec takes precedence over default_param_config
            param_type = opt_spec.get("type", default_param_config.get("type"))

            if not param_type:
                print(
                    f"Warning: Parameter '{param_name}' has no type defined in scenario specification's 'optimize' section or in OPTIMIZER_PARAMETER_DEFAULTS. Skipping parameter."
                )
                continue

            suggested_value = None

            if param_type == "int":
                low = int(opt_spec.get("min_value", default_param_config.get("low", 0))) # Default low to 0 if not found
                high = int(opt_spec.get("max_value", default_param_config.get("high", 10))) # Default high to 10 if not found
                step = int(opt_spec.get("step", default_param_config.get("step", 1)))
                # Ensure low < high, Optuna requires low <= high for suggest_int if step=1, and low < high if step > 1.
                # More robustly, Optuna expects low <= high for integers.
                # If high < low due to bad config, Optuna will error. Let it.
                suggested_value = trial.suggest_int(param_name, low, high, step=step)
            elif param_type == "float":
                low = float(opt_spec.get("min_value", default_param_config.get("low", 0.0)))
                high = float(opt_spec.get("max_value", default_param_config.get("high", 1.0)))
                step = opt_spec.get("step", default_param_config.get("step"))  # Can be None
                log = bool(opt_spec.get("log", default_param_config.get("log", False)))
                # Optuna requires low <= high for suggest_float.
                # If high < low due to bad config, Optuna will error. Let it.
                suggested_value = trial.suggest_float(param_name, low, high, step=step, log=log)
            elif param_type == "categorical":
                choices = opt_spec.get("values", default_param_config.get("values"))
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(
                        f"Categorical parameter '{param_name}' has no choices defined or choices are invalid. Ensure 'values' is a non-empty list in scenario or defaults."
                    )
                suggested_value = trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(f"Unsupported parameter type '{param_type}' for parameter '{param_name}'. Supported types are 'int', 'float', 'categorical'.")

            if suggested_value is None: # Should not happen if param_type is valid and choices are present for categorical
                print(f"Warning: No value suggested for parameter '{param_name}'. This may indicate a configuration issue.")
                continue

            if param_name in SPECIAL_SCEN_CFG_KEYS:
                scen_cfg_overrides[param_name] = suggested_value
            else:
                p[param_name] = suggested_value


        # 2 ─ evaluate --------------------------------------------------
        scen_cfg = base_scen_cfg.copy()
        scen_cfg["strategy_params"] = p
        # Apply direct scenario config overrides
        for key, value in scen_cfg_overrides.items():
            scen_cfg[key] = value

        rets = _run_scenario_static(
            g_cfg,
            scen_cfg,
            train_data_monthly,
            train_data_daily,
            train_rets_daily,
            bench_series_daily,
            features_slice,
        )

        # Mark trials that effectively produce no returns for early stopping
        if isinstance(rets, pd.Series):
            zero_ret = rets.abs().max() < ZERO_RET_EPS
        else:
            try:
                zero_ret = rets.abs().max() < ZERO_RET_EPS
            except Exception:
                zero_ret = False
        trial.set_user_attr("zero_returns", bool(zero_ret))

        bench_rets_daily = bench_series_daily.pct_change(fill_method=None).fillna(0)

        all_calculated_metrics = calculate_metrics(
            rets, bench_rets_daily, g_cfg["benchmark"]
        )

        # ----- constraint handling -----
        for constraint in constraints_config:
            metric_name = constraint.get("metric") or constraint.get("name")
            metric_val = all_calculated_metrics.get(metric_name)
            violated = False
            if metric_val is None or pd.isna(metric_val) or not np.isfinite(metric_val):
                violated = True
            if (
                not violated
                and "min_value" in constraint
                and metric_val < constraint["min_value"]
            ):
                violated = True
            if (
                not violated
                and "max_value" in constraint
                and metric_val > constraint["max_value"]
            ):
                violated = True
            if violated:
                penalty = [
                    float("-inf") if d == "maximize" else float("inf")
                    for d in metric_directions
                ]
                return penalty[0] if len(penalty) == 1 else tuple(penalty)

        results = []
        for metric_name in metrics_to_optimize:
            val = all_calculated_metrics.get(metric_name)
            if pd.isna(val) or not np.isfinite(val):
                # For multi-objective, Optuna prefers numerical values or NaN.
                # For single-objective maximization, -inf was used.
                # Let's return NaN for invalid metrics in multi-objective,
                # and -inf for single-objective if it's maximizing (or +inf if minimizing).
                # However, the direction isn't explicitly passed here for single objective fallback.
                # The original code assumed maximization for single objective.
                if not is_multi_objective:  # Single objective case
                    # Assuming maximization as per original logic for single objective.
                    # If minimization is ever supported for single obj via this path, this needs adjustment.
                    val = float("-inf")
                else:
                    val = float(
                        "nan"
                    )  # Optuna handles NaNs appropriately in multi-objective.
            results.append(val)

        if is_multi_objective:
            return tuple(results)
        else:
            return results[0]

    return objective
