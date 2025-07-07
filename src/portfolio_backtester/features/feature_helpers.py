from typing import Any, Set
import numpy as np
from .base import Feature
from .calmar_ratio import CalmarRatio
from .vams import VAMS
from .momentum import Momentum
from .benchmark_sma import BenchmarkSMA
from .sharpe_ratio import SharpeRatio
from .sortino_ratio import SortinoRatio
from .dp_vams import DPVAMS
from .atr import ATRFeature

def _get_opt_values_for_param(
    param_name: str,
    scen_optimize_specs: list,
    static_params: dict,
    default_val: Any,
    param_type_override: str | None = None
) -> list:
    from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS
    
    opt_spec = next((s for s in scen_optimize_specs if s["parameter"] == param_name), None)

    if opt_spec:
        min_v = opt_spec.get("min_value")
        max_v = opt_spec.get("max_value")
        step = opt_spec.get("step")
        values_categorical = opt_spec.get("values")

        default_param_config = OPTIMIZER_PARAMETER_DEFAULTS.get(param_name, {})
        param_type = param_type_override or opt_spec.get("type", default_param_config.get("type"))

        if param_type is None:
            if isinstance(default_val, bool): param_type = "categorical"
            elif isinstance(default_val, int): param_type = "int"
            elif isinstance(default_val, float): param_type = "float"
            else: param_type = "categorical"

        if min_v is None and max_v is None and values_categorical is None:
            if default_param_config:
                min_v = default_param_config.get("low")
                max_v = default_param_config.get("high")
                step = default_param_config.get("step")
                values_categorical = default_param_config.get("values")
            
            static_value = static_params.get(param_name, default_val)
            if (min_v is None and max_v is None and values_categorical is None) or static_value is None:
                if static_value is None and default_param_config:
                    min_v = default_param_config.get("low")
                    max_v = default_param_config.get("high")
                    step = default_param_config.get("step")
                    values_categorical = default_param_config.get("values")
                
                if min_v is None and max_v is None and values_categorical is None:
                    return [default_val]

        if param_type == "categorical":
            return values_categorical if values_categorical is not None else [static_params.get(param_name, default_val)]

        if min_v is None: min_v = default_param_config.get("low", default_val)
        if max_v is None: max_v = default_param_config.get("high", default_val)
        if step is None: step = default_param_config.get("step", 1 if param_type == "int" else 0.1)

        if min_v is None: min_v = default_val
        if max_v is None: max_v = default_val
        if step is None: step = 1 if param_type == "int" else 0.1

        if param_type == "int":
            return list(range(int(min_v), int(max_v) + 1, int(step)))
        elif param_type == "float":
            return list(np.arange(float(min_v), float(max_v) + float(step), float(step)))
        else:
            return [static_params.get(param_name, default_val)]
    else:
        return [static_params.get(param_name, default_val)]

def get_required_features_from_scenarios(strategy_configs: list, strategy_registry: dict) -> Set[Feature]:
    from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS
    
    required_features: Set[Feature] = set()
    for scen in strategy_configs:
        strategy_name = scen.get("strategy")
        strategy_class = strategy_registry.get(strategy_name)

        if strategy_class:
            required_features.update(strategy_class.get_required_features(scen))

        scen_optimize_specs = scen.get("optimize", [])
        static_params = scen.get("strategy_params", scen)

        if strategy_name == "filtered_lagged_momentum":
            is_flm_mom_params_opt = any(s["parameter"] in [
                "momentum_lookback_standard", "momentum_skip_standard",
                "momentum_lookback_predictive", "momentum_skip_predictive"
            ] for s in scen_optimize_specs)

            if is_flm_mom_params_opt:
                look_std_vals = _get_opt_values_for_param("momentum_lookback_standard", scen_optimize_specs, static_params, 11, "int")
                skip_std_vals = _get_opt_values_for_param("momentum_skip_standard", scen_optimize_specs, static_params, 1, "int")
                look_pred_vals = _get_opt_values_for_param("momentum_lookback_predictive", scen_optimize_specs, static_params, 11, "int")
                skip_pred_vals = _get_opt_values_for_param("momentum_skip_predictive", scen_optimize_specs, static_params, 0, "int")

                for l_std in look_std_vals:
                    for s_std in skip_std_vals:
                        required_features.add(Momentum(lookback_months=int(l_std), skip_months=int(s_std), name_suffix="std"))
                for l_pred in look_pred_vals:
                    for s_pred in skip_pred_vals:
                        required_features.add(Momentum(lookback_months=int(l_pred), skip_months=int(s_pred), name_suffix="pred"))

        for opt_spec in scen_optimize_specs:
            param_name = opt_spec["parameter"]

            if strategy_name == "filtered_lagged_momentum" and param_name in [
                "momentum_lookback_standard", "momentum_skip_standard",
                "momentum_lookback_predictive", "momentum_skip_predictive"
            ]:
                continue

            default_param_config = OPTIMIZER_PARAMETER_DEFAULTS.get(param_name, {})
            param_type = opt_spec.get("type", default_param_config.get("type"))

            if param_name == "sma_filter_window":
                static_val = static_params.get(param_name)
                default_val = static_val if static_val is not None else 20
                sma_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "int")
                for window in sma_values:
                    if window is not None:
                        required_features.add(BenchmarkSMA(sma_filter_window=int(window)))

            elif param_name == "lookback_months":
                default_val = static_params.get(param_name, 6)
                lookback_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "int")

                for lookback in lookback_values:
                    if lookback is not None:
                        required_features.add(Momentum(lookback_months=int(lookback)))
                        required_features.add(VAMS(lookback_months=int(lookback)))

                        alpha_default = 0.5
                        alpha_values = _get_opt_values_for_param("alpha", scen_optimize_specs, static_params, alpha_default, "float")
                        for alpha_val in alpha_values:
                            if alpha_val is not None:
                                required_features.add(DPVAMS(lookback_months=int(lookback), alpha=float(alpha_val)))

            elif param_name == "alpha":
                default_val = static_params.get(param_name, 0.5)
                alpha_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "float")

                lookback_default = static_params.get("lookback_months",6)
                lookback_values_for_alpha_opt = _get_opt_values_for_param("lookback_months", scen_optimize_specs, static_params, lookback_default, "int")

                for lookback in lookback_values_for_alpha_opt:
                    for alpha_val in alpha_values:
                        if lookback is not None and alpha_val is not None:
                            required_features.add(DPVAMS(lookback_months=int(lookback), alpha=float(alpha_val)))

            elif param_name == "rolling_window":
                default_val = static_params.get(param_name, 6)
                rw_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "int")
                if strategy_class:
                    if hasattr(strategy_class, 'signal_generator_class'):
                        gen_class_name = strategy_class.signal_generator_class.__name__
                        for rw in rw_values:
                            if rw is not None:
                                if gen_class_name == "SharpeSignalGenerator":
                                    required_features.add(SharpeRatio(rolling_window=int(rw)))
                                elif gen_class_name == "SortinoSignalGenerator":
                                    target_ret = static_params.get("target_return", 0.0)
                                    required_features.add(SortinoRatio(rolling_window=int(rw), target_return=float(target_ret)))
                                elif gen_class_name == "CalmarSignalGenerator":
                                    required_features.add(CalmarRatio(rolling_window=int(rw)))
    return required_features