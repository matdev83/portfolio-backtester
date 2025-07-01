import optuna
from typing import Dict, Any, Tuple
from ..utils import _run_scenario_static
from ..reporting.performance_metrics import calculate_metrics

def build_objective(g_cfg: Dict, base_scen_cfg: Dict,
                    train_data, train_rets, bench_series, features_slice,
                    metric: str = "Calmar"):
    """
    Factory to build a customized Optuna objective function.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        # 1 ─ suggest parameters ----------------------------------------
        p = base_scen_cfg["strategy_params"].copy()

        if "max_lookback" in p:
            p["max_lookback"] = trial.suggest_int("max_lookback", 20, 252, step=10)

        if "calmar_lookback" in p:
            p["calmar_lookback"] = trial.suggest_int("calmar_lookback", 20, 252, step=10)

        if "leverage" in p:
            p["leverage"] = trial.suggest_float("leverage", 0.5, 2.0, step=0.1)

        if "smoothing_lambda" in p:
            p["smoothing_lambda"] = trial.suggest_float("smoothing_lambda", 0.0, 0.9)

        # 2 ─ evaluate --------------------------------------------------
        scen_cfg = base_scen_cfg.copy()
        scen_cfg["strategy_params"] = p

        rets = _run_scenario_static(
            g_cfg, scen_cfg,
            train_data, train_rets,
            train_data[g_cfg["benchmark"]],
            features_slice)

        val = calculate_metrics(
            rets, bench_series, g_cfg["benchmark"])[metric]

        # 3 ─ Optuna expects *lower* is better by default
        return -val

    return objective
