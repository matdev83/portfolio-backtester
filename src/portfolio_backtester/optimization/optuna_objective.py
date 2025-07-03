from typing import Dict, Any, Tuple
import optuna
import pandas as pd
import numpy as np

from ..utils import _run_scenario_static
from ..reporting.performance_metrics import calculate_metrics
from ..config import OPTIMIZER_PARAMETER_DEFAULTS


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
        p = base_scen_cfg["strategy_params"].copy()

        for opt_spec in optimization_specs:
            param_name = opt_spec["parameter"]
            if param_name not in optimizer_config:
                # This should ideally not happen if optimizer_config is comprehensive
                print(
                    f"Warning: Parameter '{param_name}' requested for optimization but not found in optimizer_config.json"
                )
                continue

            param_type = optimizer_config[param_name]["type"]
            low = float(opt_spec.get("min_value", optimizer_config[param_name]["low"]))
            high = float(
                opt_spec.get("max_value", optimizer_config[param_name]["high"])
            )
            step = float(opt_spec.get("step", optimizer_config[param_name].get("step")))

            if param_type == "int":
                p[param_name] = trial.suggest_int(
                    param_name, int(low), int(high), step=int(step)
                )
            elif param_type == "float":
                log = opt_spec.get(
                    "log", optimizer_config[param_name].get("log", False)
                )
                p[param_name] = trial.suggest_float(
                    param_name, low, high, step=step, log=log
                )

        # 2 ─ evaluate --------------------------------------------------
        scen_cfg = base_scen_cfg.copy()
        scen_cfg["strategy_params"] = p

        rets = _run_scenario_static(
            g_cfg,
            scen_cfg,
            train_data_monthly,
            train_data_daily,
            train_rets_daily,
            bench_series_daily,
            features_slice,
        )

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
