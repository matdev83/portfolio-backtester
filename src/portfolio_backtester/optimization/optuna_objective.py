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
    metric: str = "Calmar",
):
    """
    Factory to build a customized Optuna objective function.
    """
    optimizer_config = OPTIMIZER_PARAMETER_DEFAULTS

    # Extract optimization specifications from base_scen_cfg
    optimization_specs = base_scen_cfg.get("optimize", [])

    def objective(trial: optuna.trial.Trial) -> float:
        # 1 ─ suggest parameters ----------------------------------------
        p = base_scen_cfg["strategy_params"].copy()

        for opt_spec in optimization_specs:
            param_name = opt_spec["parameter"]
            if param_name not in optimizer_config:
                # This should ideally not happen if optimizer_config is comprehensive
                print(f"Warning: Parameter '{param_name}' requested for optimization but not found in optimizer_config.json")
                continue

            param_type = optimizer_config[param_name]["type"]
            low = float(opt_spec.get("min_value", optimizer_config[param_name]["low"]))
            high = float(opt_spec.get("max_value", optimizer_config[param_name]["high"]))
            step = float(opt_spec.get("step", optimizer_config[param_name].get("step")))

            if param_type == "int":
                p[param_name] = trial.suggest_int(param_name, int(low), int(high), step=int(step))
            elif param_type == "float":
                log = opt_spec.get("log", optimizer_config[param_name].get("log", False))
                p[param_name] = trial.suggest_float(param_name, low, high, step=step, log=log)

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

        val = calculate_metrics(
            rets, bench_rets_daily, g_cfg["benchmark"])[metric]

        # Penalise invalid metrics with negative infinity (since we maximise)
        if pd.isna(val) or not np.isfinite(val):
            return float("-inf")

        # Study direction is "maximize" – return the metric directly
        return val

    return objective
