from typing import Dict, Any
import optuna
import pandas as pd
import numpy as np

from .. import strategies
from ..reporting.performance_metrics import calculate_metrics
from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from ..constants import ZERO_RET_EPS
from ..portfolio.position_sizer import get_position_sizer
from ..portfolio.rebalancing import rebalance
from ..api_stability import api_stable


def _resolve_strategy(name: str):
    class_name = "".join(w.capitalize() for w in name.split('_')) + "Strategy"
    if name == "momentum_unfiltered_atr":
        class_name = "MomentumUnfilteredAtrStrategy"
    elif name == "vams_momentum":
        class_name = "VAMSMomentumStrategy"
    elif name == "vams_no_downside":
        class_name = "VAMSNoDownsideStrategy"
    return getattr(strategies, class_name, None)

def _run_scenario_static(
    global_cfg,
    scenario_cfg,
    price_monthly,
    price_daily,
    rets_daily,
    benchmark_daily,
    features_slice,
):
    """Lightweight version of Backtester.run_scenario() suitable for Optuna."""
    strat_cls = _resolve_strategy(scenario_cfg["strategy"])
    if not strat_cls:
        raise ValueError(f"Could not resolve strategy: {scenario_cfg['strategy']}")
    strategy = strat_cls(scenario_cfg["strategy_params"])

    universe_cols = [c for c in price_monthly.columns if c != global_cfg["benchmark"]]
    signals = strategy.generate_signals(
        price_monthly[universe_cols],
        features_slice,
        price_monthly[global_cfg["benchmark"]],
    )

    sizer_name = scenario_cfg.get("position_sizer", "equal_weight")
    sizer_func = get_position_sizer(sizer_name)

    sizer_params = scenario_cfg.get("strategy_params", {}).copy()
    sizer_param_mapping = {
        "sizer_sharpe_window": "window",
        "sizer_sortino_window": "window",
        "sizer_beta_window": "window",
        "sizer_corr_window": "window",
        "sizer_dvol_window": "window",
        "sizer_target_return": "target_return",
    }
    for old_key, new_key in sizer_param_mapping.items():
        if old_key in sizer_params:
            sizer_params[new_key] = sizer_params.pop(old_key)

    sized = sizer_func(
        signals,
        price_monthly[universe_cols],
        price_monthly[global_cfg["benchmark"]],
        **sizer_params,
    )
    weights_monthly = rebalance(sized, scenario_cfg["rebalance_frequency"])

    weights_daily = (
        weights_monthly.reindex(price_daily.index, method="ffill").fillna(0.0)
    )

    gross = (weights_daily.shift(1).fillna(0.0) * rets_daily).sum(axis=1)
    turn = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1)
    tc = turn * (scenario_cfg["transaction_costs_bps"] / 10_000)

    return (gross - tc).reindex(price_daily.index).fillna(0)


@api_stable(version="1.0", strict_params=True, strict_return=False)
def build_objective(
    g_cfg: Dict,
    base_scen_cfg: Dict,
    train_data_monthly,
    train_data_daily,
    train_rets_daily,
    bench_series_daily,
    features_slice,
):
    """
    Factory to build a customized Optuna objective function.
    """
    optimizer_config = OPTIMIZER_PARAMETER_DEFAULTS
    optimization_specs = base_scen_cfg.get("optimize", [])
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
        metrics_to_optimize = ["Calmar"]
        metric_directions = ["maximize"]

    def objective(
        trial: optuna.trial.Trial,
    ) -> Any:
        p = base_scen_cfg["strategy_params"].copy()
        scen_cfg_overrides = {}
        SPECIAL_SCEN_CFG_KEYS = ["position_sizer"]

        strategy_name = base_scen_cfg.get("strategy")
        strategy_tunable_params = set()
        if strategy_name:
            strat_cls = _resolve_strategy(strategy_name)
            if strat_cls:
                strategy_tunable_params = strat_cls.tunable_parameters()
            else:
                print(f"Warning: Could not resolve strategy '{strategy_name}' for parameter filtering. Optimizing all parameters.")

        sizer_param_map = {
            "rolling_sharpe": "sizer_sharpe_window",
            "rolling_sortino": "sizer_sortino_window",
            "rolling_beta": "sizer_beta_window",
            "rolling_benchmark_corr": "sizer_corr_window",
            "rolling_downside_volatility": "sizer_dvol_window",
        }
        
        position_sizer = base_scen_cfg.get("position_sizer")
        if position_sizer and position_sizer in sizer_param_map:
            strategy_tunable_params.add(sizer_param_map[position_sizer])

        skipped_params = []
        for opt_spec in optimization_specs:
            param_name = opt_spec["parameter"]
            
            if (strategy_tunable_params and 
                param_name not in strategy_tunable_params and 
                param_name not in SPECIAL_SCEN_CFG_KEYS):
                skipped_params.append(param_name)
                continue
            
            default_param_config = optimizer_config.get(param_name, {})
            param_type = opt_spec.get("type", default_param_config.get("type"))

            if not param_type:
                print(f"Warning: Parameter '{param_name}' has no type defined in scenario specification's 'optimize' section or in OPTIMIZER_PARAMETER_DEFAULTS. Skipping parameter.")
                continue

            suggested_value = None
            if param_type == "int":
                low = int(opt_spec.get("min_value", default_param_config.get("low", 0)))
                high = int(opt_spec.get("max_value", default_param_config.get("high", 10)))
                step = int(opt_spec.get("step", default_param_config.get("step", 1)))
                suggested_value = trial.suggest_int(param_name, low, high, step=step)
            elif param_type == "float":
                low = float(opt_spec.get("min_value", default_param_config.get("low", 0.0)))
                high = float(opt_spec.get("max_value", default_param_config.get("high", 1.0)))
                step = opt_spec.get("step", default_param_config.get("step"))
                log = bool(opt_spec.get("log", default_param_config.get("log", False)))
                suggested_value = trial.suggest_float(param_name, low, high, step=step, log=log)
            elif param_type == "categorical":
                choices = opt_spec.get("values", default_param_config.get("values"))
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"Categorical parameter '{param_name}' has no choices defined or choices are invalid.")
                suggested_value = trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(f"Unsupported parameter type '{param_type}' for parameter '{param_name}'.")

            if suggested_value is None:
                print(f"Warning: No value suggested for parameter '{param_name}'.")
                continue

            if param_name in SPECIAL_SCEN_CFG_KEYS:
                scen_cfg_overrides[param_name] = suggested_value
            else:
                p[param_name] = suggested_value

        if skipped_params:
            trial.set_user_attr("skipped_parameters", skipped_params)
        
        scen_cfg = base_scen_cfg.copy()
        scen_cfg["strategy_params"] = p
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

        if isinstance(rets, pd.Series):
            zero_ret = rets.abs().max() < ZERO_RET_EPS
        else:
            zero_ret = False
        trial.set_user_attr("zero_returns", bool(zero_ret))

        bench_rets_daily = bench_series_daily.pct_change(fill_method=None).fillna(0)
        all_calculated_metrics = calculate_metrics(
            rets, bench_rets_daily, g_cfg["benchmark"]
        )

        for constraint in constraints_config:
            metric_name = constraint.get("metric")
            min_value = constraint.get("min_value")
            max_value = constraint.get("max_value")
            operator = constraint.get("operator", "").upper()
            value = constraint.get("value")

            # Support both old format (operator/value) and new format (min_value/max_value)
            if min_value is not None or max_value is not None:
                # New format: min_value/max_value
                if not metric_name:
                    print(f"Warning: Skipping invalid constraint: {constraint}")
                    continue
                
                metric_val = all_calculated_metrics.get(metric_name)
                violated = False

                if metric_val is None or pd.isna(metric_val) or not np.isfinite(metric_val):
                    violated = True
                else:
                    if min_value is not None and metric_val < min_value:
                        violated = True
                    if max_value is not None and metric_val > max_value:
                        violated = True
                
                if violated:
                    penalty = [
                        float("-inf") if d == "maximize" else float("inf")
                        for d in metric_directions
                    ]
                    return penalty[0] if len(penalty) == 1 else tuple(penalty)
            
            elif operator and value is not None:
                # Old format: operator/value
                if not metric_name:
                    print(f"Warning: Skipping invalid constraint: {constraint}")
                    continue

                metric_val = all_calculated_metrics.get(metric_name)
                violated = False

                if metric_val is None or pd.isna(metric_val) or not np.isfinite(metric_val):
                    violated = True
                else:
                    if operator == "LT" and metric_val >= value:
                        violated = True
                    elif operator == "LE" and metric_val > value:
                        violated = True
                    elif operator == "GT" and metric_val <= value:
                        violated = True
                    elif operator == "GE" and metric_val < value:
                        violated = True
                    elif operator == "EQ" and metric_val != value:
                        violated = True
                    elif operator not in ["LT", "LE", "GT", "GE", "EQ"]:
                        print(f"Warning: Unsupported operator '{operator}' in constraint: {constraint}")
                        continue
                
                if violated:
                    penalty = [
                        float("-inf") if d == "maximize" else float("inf")
                        for d in metric_directions
                    ]
                    return penalty[0] if len(penalty) == 1 else tuple(penalty)
            else:
                print(f"Warning: Skipping invalid constraint: {constraint}")
                continue

        results = []
        for i, metric_name in enumerate(metrics_to_optimize):
            val = all_calculated_metrics.get(metric_name)
            if pd.isna(val) or not np.isfinite(val):
                if is_multi_objective:
                    # For multi-objective, preserve NaN/inf as NaN to let Optuna handle it
                    val = float('nan')
                else:
                    # For single objective, use penalty values
                    direction = metric_directions[i] if i < len(metric_directions) else "maximize"
                    if direction == "maximize":
                        val = float("-inf")  # Worst possible value for maximization
                    else:
                        val = float("inf")   # Worst possible value for minimization
            results.append(val)

        if is_multi_objective:
            return tuple(results)
        else:
            return results[0]

    return objective
