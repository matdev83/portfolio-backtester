from abc import ABC, abstractmethod
from typing import Set

import numpy as np
import pandas as pd


class Feature(ABC):
    """Abstract base class for a feature required by a strategy."""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame | pd.Series:
        """Computes the feature."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the feature."""
        pass

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.params == other.params

    def __hash__(self):
        return hash((self.__class__, frozenset(self.params.items())))


class CalmarRatio(Feature):
    """Computes the Calmar ratio."""

    def __init__(self, rolling_window: int):
        super().__init__(rolling_window=rolling_window)
        self.rolling_window = rolling_window

    @property
    def name(self) -> str:
        return f"calmar_{self.rolling_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        cal_factor = 12  # Annualization factor for monthly data
        rolling_mean = rets.rolling(self.rolling_window).mean() * cal_factor

        def max_drawdown(series):
            series = series.dropna()
            if series.empty:
                return 0.0
            cumulative_returns = (1 + series).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            peak = peak.replace(0, 1e-9)
            drawdown = (cumulative_returns / peak) - 1
            drawdown = drawdown.replace([np.inf, -np.inf], [0, 0]).fillna(0)
            min_drawdown = abs(drawdown.min())
            return min_drawdown

        rolling_max_dd = rets.rolling(self.rolling_window).apply(max_drawdown, raw=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            calmar_ratio = rolling_mean / rolling_max_dd
        calmar_ratio.replace([np.inf, -np.inf], [10.0, -10.0], inplace=True)
        calmar_ratio = calmar_ratio.clip(-10.0, 10.0)
        return calmar_ratio


class VAMS(Feature):
    """Computes Volatility Adjusted Momentum Scores (VAMS)."""

    def __init__(self, lookback_months: int):
        super().__init__(lookback_months=lookback_months)
        self.lookback_months = lookback_months

    @property
    def name(self) -> str:
        return f"vams_{self.lookback_months}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        momentum = (1 + rets).rolling(self.lookback_months).apply(np.prod, raw=True) - 1
        total_vol = rets.rolling(self.lookback_months).std()
        denominator = total_vol.replace(0, np.nan)
        vams = momentum / denominator
        return vams


class Momentum(Feature):
    """Computes momentum for each asset: P(t-skip_months)/P(t-skip_months-lookback_months) - 1."""

    def __init__(self, lookback_months: int, skip_months: int = 0, name_suffix: str = ""):
        super().__init__(lookback_months=lookback_months, skip_months=skip_months, name_suffix=name_suffix)
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.name_suffix = name_suffix # e.g. "std" or "pred"

    @property
    def name(self) -> str:
        if self.skip_months == 0 and not self.name_suffix:
            # Backward compatibility for simple momentum feature name
            return f"momentum_{self.lookback_months}m"

        base_name = f"momentum_{self.lookback_months}m_skip{self.skip_months}m"
        if self.name_suffix: # Add underscore if suffix exists and suffix is not empty
            return f"{base_name}_{self.name_suffix}"
        return base_name

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        # Assuming data is monthly prices for feature computation.
        monthly_rets = data.pct_change().fillna(0) # Use default pct_change, then fillna

        # Rolling product of (1+monthly_return) to get P(t)/P(t-L) - 1
        momentum_lookback_period = (1 + monthly_rets).rolling(
            window=self.lookback_months,
            min_periods=int(self.lookback_months * 0.9) # Ensure most of the window has data
        ).apply(np.prod, raw=True) - 1

        # Shift the result by skip_months.
        # So, value at row 't' becomes P(t-skip_months)/P(t-skip_months-lookback_months) - 1
        final_momentum = momentum_lookback_period.shift(self.skip_months)

        return final_momentum


class BenchmarkSMA(Feature):
    """Computes the Simple Moving Average (SMA) for the benchmark."""

    def __init__(self, sma_filter_window: int):
        super().__init__(sma_filter_window=sma_filter_window)
        self.sma_filter_window = sma_filter_window

    @property
    def name(self) -> str:
        return f"benchmark_sma_{self.sma_filter_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.Series:
        if benchmark_data is None:
            raise ValueError("Benchmark data is required for BenchmarkSMA feature.")
        return (benchmark_data > benchmark_data.rolling(self.sma_filter_window).mean()).astype(int)


class SharpeRatio(Feature):
    """Computes the Sharpe ratio."""

    def __init__(self, rolling_window: int):
        super().__init__(rolling_window=rolling_window)
        self.rolling_window = rolling_window

    @property
    def name(self) -> str:
        return f"sharpe_{self.rolling_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        cal_factor = np.sqrt(12)  # Annualization factor for monthly data
        rolling_mean = rets.rolling(self.rolling_window).mean()
        rolling_std = rets.rolling(self.rolling_window).std()
        sharpe_ratio = (rolling_mean * cal_factor) / (rolling_std * cal_factor).replace(0, np.nan)
        return sharpe_ratio.fillna(0)


class SortinoRatio(Feature):
    """Computes the Sortino ratio."""

    def __init__(self, rolling_window: int, target_return: float = 0.0):
        super().__init__(rolling_window=rolling_window, target_return=target_return)
        self.rolling_window = rolling_window
        self.target_return = target_return

    @property
    def name(self) -> str:
        return f"sortino_{self.rolling_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        cal_factor = np.sqrt(12)  # Annualization factor for monthly data
        rolling_mean = rets.rolling(self.rolling_window).mean()

        def downside_deviation(series):
            downside_returns = series[series < self.target_return]
            if len(downside_returns) == 0:
                return 1e-9
            return np.sqrt(np.mean((downside_returns - self.target_return) ** 2))

        rolling_downside_dev = rets.rolling(self.rolling_window).apply(downside_deviation, raw=False)
        excess_return = rolling_mean - self.target_return
        stable_downside_dev = np.maximum(rolling_downside_dev, 1e-6)
        sortino_ratio = (excess_return * cal_factor) / (stable_downside_dev * cal_factor)
        sortino_ratio = sortino_ratio.clip(-10.0, 10.0)
        return pd.DataFrame(sortino_ratio, index=rets.index, columns=rets.columns).fillna(0)


class DPVAMS(Feature):
    """Computes Downside Penalized Volatility Adjusted Momentum Scores (dp-VAMS)."""

    def __init__(self, lookback_months: int, alpha: float):
        super().__init__(lookback_months=lookback_months, alpha=alpha)
        self.lookback_months = lookback_months
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"dp_vams_{self.lookback_months}m_{self.alpha:.2f}a"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        momentum = (1 + rets).rolling(self.lookback_months).apply(np.prod, raw=True) - 1
        
        negative_rets = rets[rets < 0].fillna(0)
        downside_dev = negative_rets.rolling(self.lookback_months).std().fillna(0)
        total_vol = rets.rolling(self.lookback_months).std().fillna(0)
        
        denominator = self.alpha * downside_dev + (1 - self.alpha) * total_vol
        denominator = denominator.replace(0, np.nan)
        
        dp_vams = momentum / denominator
        return dp_vams.fillna(0)


from .config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from typing import Any # For type hinting

# Helper function to extract a list of values for an optimizable parameter
def _get_opt_values_for_param(
    param_name: str,
    scen_optimize_specs: list,
    static_params: dict,
    default_val: Any,
    param_type_override: str | None = None
) -> list:
    opt_spec = next((s for s in scen_optimize_specs if s["parameter"] == param_name), None)

    if opt_spec:
        min_v = opt_spec.get("min_value")
        max_v = opt_spec.get("max_value")
        step = opt_spec.get("step")
        values_categorical = opt_spec.get("values")

        # Determine type: opt_spec -> default_param_config -> type of default_val -> param_type_override
        default_param_config = OPTIMIZER_PARAMETER_DEFAULTS.get(param_name, {})
        param_type = param_type_override or opt_spec.get("type", default_param_config.get("type"))

        if param_type is None: # Infer from default_val if still None
            if isinstance(default_val, bool): param_type = "categorical" # Treat bools as categorical [True, False]
            elif isinstance(default_val, int): param_type = "int"
            elif isinstance(default_val, float): param_type = "float"
            else: param_type = "categorical" # Fallback for other types

        if param_type == "categorical":
            return values_categorical if values_categorical is not None else [static_params.get(param_name, default_val)]

        # For numerical types, ensure min_v, max_v are present
        if min_v is None: min_v = default_param_config.get("low", default_val)
        if max_v is None: max_v = default_param_config.get("high", default_val)
        if step is None: step = default_param_config.get("step", 1 if param_type == "int" else 0.1)


        if param_type == "int":
            return list(range(int(min_v), int(max_v) + int(step), int(step)))
        elif param_type == "float":
            # Use np.arange for float steps, ensuring endpoint is handled correctly
            # Add a small epsilon to max_v to include it if max_v is a multiple of step
            return [round(v, 4) for v in np.arange(float(min_v), float(max_v) + float(step) * 0.5, float(step))]
        else: # Should not happen if type is inferred or specified
            return [static_params.get(param_name, default_val)]

    else: # Parameter not being optimized, return its static or default value
        return [static_params.get(param_name, default_val)]

def get_required_features_from_scenarios(strategy_configs: list, strategy_registry: dict) -> Set[Feature]:
    """
    Gathers all unique feature requirements from a list of strategy configurations.
    It considers both static strategy parameters and optimized parameters.
    """
    required_features: Set[Feature] = set()
    for scen in strategy_configs:
        strategy_name = scen.get("strategy")
        strategy_class = strategy_registry.get(strategy_name)

        # Static features from strategy_params
        if strategy_class:
            required_features.update(strategy_class.get_required_features(scen))

        scen_optimize_specs = scen.get("optimize", [])
        static_params = scen.get("strategy_params", scen) # Fallback to scen itself

        # Handle specific parameter combinations for FilteredLaggedMomentumStrategy if optimized
        if strategy_name == "filtered_lagged_momentum":
            # Check if any of its core momentum parameters are being optimized
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

        # General loop for other optimizable parameters
        for opt_spec in scen_optimize_specs:
            param_name = opt_spec["parameter"]

            # Skip if handled by the filtered_lagged_momentum block above
            if strategy_name == "filtered_lagged_momentum" and param_name in [
                "momentum_lookback_standard", "momentum_skip_standard",
                "momentum_lookback_predictive", "momentum_skip_predictive"
            ]:
                continue

            default_param_config = OPTIMIZER_PARAMETER_DEFAULTS.get(param_name, {})
            param_type = opt_spec.get("type", default_param_config.get("type"))

            if param_name == "sma_filter_window":
                default_val = static_params.get(param_name, 20) # Example default
                sma_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "int")
                for window in sma_values:
                    required_features.add(BenchmarkSMA(sma_filter_window=int(window)))

            elif param_name == "lookback_months": # For old MomentumStrategy, VAMS etc.
                default_val = static_params.get(param_name, 6)
                lookback_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "int")

                for lookback in lookback_values:
                    # Default Momentum (skip=0, no_suffix) for compatibility with MomentumSignalGenerator
                    required_features.add(Momentum(lookback_months=int(lookback)))
                    required_features.add(VAMS(lookback_months=int(lookback)))

                    # DPVAMS: Needs alpha. Alpha can be static or optimized itself.
                    alpha_default = 0.5
                    alpha_values = _get_opt_values_for_param("alpha", scen_optimize_specs, static_params, alpha_default, "float")
                    for alpha_val in alpha_values:
                        required_features.add(DPVAMS(lookback_months=int(lookback), alpha=float(alpha_val)))

            elif param_name == "alpha": # For DPVAMS, when alpha is optimized directly
                # This is coupled with lookback_months. If lookback_months is also optimized,
                # this block will correctly iterate alphas for each static/optimized lookback.
                default_val = static_params.get(param_name, 0.5)
                alpha_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "float")

                # Get lookback_months values (optimized or static)
                lookback_default = static_params.get("lookback_months",6) # Default if not optimizing lookback
                lookback_values_for_alpha_opt = _get_opt_values_for_param("lookback_months", scen_optimize_specs, static_params, lookback_default, "int")

                for lookback in lookback_values_for_alpha_opt:
                    for alpha_val in alpha_values:
                        required_features.add(DPVAMS(lookback_months=int(lookback), alpha=float(alpha_val)))

            # Other optimizable params like 'rolling_window' for Sharpe, Sortino, Calmar
            elif param_name == "rolling_window":
                default_val = static_params.get(param_name, 6)
                rw_values = _get_opt_values_for_param(param_name, [opt_spec], static_params, default_val, param_type or "int")
                # Need to know which strategy this rolling_window is for to create the right feature
                if strategy_class: # Check if strategy_class is resolved
                    if hasattr(strategy_class, 'signal_generator_class'):
                        gen_class_name = strategy_class.signal_generator_class.__name__
                        for rw in rw_values:
                            if gen_class_name == "SharpeSignalGenerator":
                                required_features.add(SharpeRatio(rolling_window=int(rw)))
                            elif gen_class_name == "SortinoSignalGenerator":
                                target_ret = static_params.get("target_return", 0.0) # Get static target_return
                                required_features.add(SortinoRatio(rolling_window=int(rw), target_return=float(target_ret)))
                            elif gen_class_name == "CalmarSignalGenerator":
                                required_features.add(CalmarRatio(rolling_window=int(rw)))
    return required_features
