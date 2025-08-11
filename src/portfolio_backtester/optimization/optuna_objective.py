from typing import Dict, Any
import optuna
import pandas as pd
import numpy as np

from .. import strategies
from ..reporting.performance_metrics import calculate_metrics
from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from ..constants import ZERO_RET_EPS

# Removed direct position sizer import - now using strategy's position sizer provider
from ..portfolio.rebalancing import rebalance
from ..api_stability import api_stable
from ..interfaces.attribute_accessor_interface import create_class_attribute_accessor


def _resolve_strategy(name: str):
    class_name = "".join(w.capitalize() for w in name.split("_")) + "Strategy"
    if name == "momentum_unfiltered_atr":
        class_name = "MomentumUnfilteredAtrStrategy"
    elif name == "vams_momentum":
        class_name = "VAMSMomentumPortfolioStrategy"
    elif name == "vams_no_downside":
        class_name = "VAMSNoDownsideStrategy"

    # Use DIP interface instead of direct getattr call
    class_accessor = create_class_attribute_accessor()
    try:
        return class_accessor.get_class_from_module(strategies, class_name)
    except AttributeError:
        # Return None if class doesn't exist (preserving original behavior)
        return None


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

    # Use strategy's position sizer provider instead of direct config access
    position_sizer_provider = strategy.get_position_sizer_provider()
    sizer = position_sizer_provider.get_position_sizer()
    sizer_config = position_sizer_provider.get_position_sizer_config()

    # Extract position sizer parameters from provider config
    sizer_params = {k: v for k, v in sizer_config.items() if k != "position_sizer"}

    sizer_kwargs = {
        "signals": signals,
        "prices": price_monthly[universe_cols],
        "benchmark": price_monthly[global_cfg["benchmark"]],
        **sizer_params,
    }

    sized = sizer.calculate_weights(**sizer_kwargs)
    weights_monthly = rebalance(sized, scenario_cfg["rebalance_frequency"])

    weights_daily = weights_monthly.reindex(price_daily.index, method="ffill").fillna(0.0)

    gross = (weights_daily.shift(1).fillna(0.0) * rets_daily).sum(axis=1)
    turn = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1)

    from ..trading import get_transaction_cost_model

    tx_cost_model = get_transaction_cost_model(global_cfg)
    tc, _ = tx_cost_model.calculate(
        turnover=turn,
        weights_daily=weights_daily,
        price_data=price_daily,
        portfolio_value=global_cfg.get("portfolio_value", 100000.0),
    )

    portfolio_returns = (gross - tc).reindex(price_daily.index).fillna(0)

    # Calculate excess returns vs benchmark for better optimization
    if benchmark_daily is not None and len(benchmark_daily) > 0:
        # Align benchmark with portfolio returns
        aligned_benchmark = benchmark_daily.reindex(portfolio_returns.index).fillna(0)
        excess_returns = portfolio_returns - aligned_benchmark

        # Store both portfolio and excess returns for flexible optimization
        portfolio_returns = (
            portfolio_returns.to_frame("portfolio_returns")
            if hasattr(portfolio_returns, "to_frame")
            else portfolio_returns
        )
        if hasattr(portfolio_returns, "assign"):
            portfolio_returns = portfolio_returns.assign(
                excess_returns=excess_returns, benchmark_returns=aligned_benchmark
            )

    return portfolio_returns


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

    if is_multi_objective and optimization_targets_config:
        metrics_to_optimize = [target["name"] for target in optimization_targets_config]
        metric_directions = [
            target.get("direction", "maximize").lower() for target in optimization_targets_config
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
                print(
                    f"Warning: Could not resolve strategy '{strategy_name}' for parameter filtering. Optimizing all parameters."
                )

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

            if (
                strategy_tunable_params
                and param_name not in strategy_tunable_params
                and param_name not in SPECIAL_SCEN_CFG_KEYS
            ):
                skipped_params.append(param_name)
                continue

            default_param_config = optimizer_config.get(param_name, {})
            param_type = opt_spec.get("type", default_param_config.get("type"))

            if not param_type:
                print(
                    f"Warning: Parameter '{param_name}' has no type defined in scenario specification's 'optimize' section or in OPTIMIZER_PARAMETER_DEFAULTS. Skipping parameter."
                )
                continue

            suggested_value: Any = (
                None  # Initialize to satisfy mypy, though Optuna suggest methods always return a value
            )
            if param_type == "int":
                low_int: int = int(opt_spec.get("min_value", default_param_config.get("low", 0)))
                high_int: int = int(opt_spec.get("max_value", default_param_config.get("high", 10)))
                step_int: int = int(opt_spec.get("step", default_param_config.get("step", 1)))
                suggested_value = trial.suggest_int(param_name, low_int, high_int, step=step_int)
            elif param_type == "float":
                low_float: float = float(
                    opt_spec.get("min_value", default_param_config.get("low", 0.0))
                )
                high_float: float = float(
                    opt_spec.get("max_value", default_param_config.get("high", 1.0))
                )
                step_float = opt_spec.get("step", default_param_config.get("step"))
                log_float = bool(opt_spec.get("log", default_param_config.get("log", False)))
                suggested_value = trial.suggest_float(
                    param_name, low_float, high_float, step=step_float, log=log_float
                )
            elif param_type == "categorical":
                choices = opt_spec.get("values", default_param_config.get("values"))
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(
                        f"Categorical parameter '{param_name}' has no choices defined or choices are invalid."
                    )
                suggested_value = trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(
                    f"Unsupported parameter type '{param_type}' for parameter '{param_name}'."
                )

            # Optuna suggest methods always return a value, so this check is theoretically unreachable.
            # Kept for defensive programming in case Optuna API changes or unexpected behavior occurs.
            if suggested_value is None:
                # This case should ideally never be hit.
                # Consider raising an error or logging a critical warning if it does.
                # For now, to satisfy linters and make logic explicit, we handle it.
                # However, 'continue' here would skip the parameter, which might not be desired.
                # A better approach might be to fail the trial or use a default.
                # For now, we'll assume Optuna works as documented and this is dead code.
                # To make mypy/linters happy and avoid "unreachable" if we keep the print:
                pass  # Or handle more robustly if this state is ever possible.

            if param_name in SPECIAL_SCEN_CFG_KEYS:
                scen_cfg_overrides[param_name] = suggested_value
            else:
                p[f"{strategy_name}.{param_name}"] = suggested_value

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
        all_calculated_metrics = calculate_metrics(rets, bench_rets_daily, g_cfg["benchmark"])

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
                        print(
                            f"Warning: Unsupported operator '{operator}' in constraint: {constraint}"
                        )
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
                    val = float("nan")
                else:
                    # For single objective, use penalty values
                    direction = metric_directions[i] if i < len(metric_directions) else "maximize"
                    if direction == "maximize":
                        val = float("-inf")  # Worst possible value for maximization
                    else:
                        val = float("inf")  # Worst possible value for minimization
            results.append(val)

        if is_multi_objective:
            return tuple(results)
        else:
            return results[0]

    return objective
