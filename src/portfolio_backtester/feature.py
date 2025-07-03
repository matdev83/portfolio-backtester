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
    """Computes momentum for each asset."""

    def __init__(self, lookback_months: int):
        super().__init__(lookback_months=lookback_months)
        self.lookback_months = lookback_months

    @property
    def name(self) -> str:
        return f"momentum_{self.lookback_months}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change(fill_method=None).fillna(0)
        momentum = (1 + rets).rolling(self.lookback_months).apply(np.prod, raw=True) - 1
        return momentum


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


def get_required_features_from_scenarios(strategy_configs: list, strategy_registry: dict) -> Set[Feature]:
    """
    Gathers all unique feature requirements from a list of strategy configurations.
    It considers both static strategy parameters and optimized parameters.
    """
    required_features = set()
    for scen in strategy_configs:
        strategy_name = scen.get("strategy")
        strategy_class = strategy_registry.get(strategy_name)
        if strategy_class:
            required_features.update(strategy_class.get_required_features(scen))

        # Also check for features needed for optimization
        if "optimize" in scen:
            for opt_spec in scen["optimize"]: # Renamed opt_param to opt_spec for clarity
                param_name = opt_spec["parameter"]
                default_param_config = OPTIMIZER_PARAMETER_DEFAULTS.get(param_name, {})

                # Determine type: opt_spec takes precedence over default_param_config
                param_type = opt_spec.get("type", default_param_config.get("type"))

                if param_name == "sma_filter_window":
                    if param_type == "categorical":
                        values = opt_spec.get("values", default_param_config.get("values", []))
                        for window in values:
                            if isinstance(window, int):
                                required_features.add(BenchmarkSMA(sma_filter_window=window))
                    elif param_type == "int":
                        min_val = int(opt_spec.get("min_value", default_param_config.get("low", 2)))
                        max_val = int(opt_spec.get("max_value", default_param_config.get("high", 24)))
                        step = int(opt_spec.get("step", default_param_config.get("step", 1)))
                        for window in range(min_val, max_val + 1, step):
                            required_features.add(BenchmarkSMA(sma_filter_window=window))

                elif param_name == "lookback_months":
                    lookback_values_to_add = set()
                    if param_type == "categorical":
                        values = opt_spec.get("values", default_param_config.get("values", []))
                        for val in values:
                            if isinstance(val, int):
                                lookback_values_to_add.add(val)
                    elif param_type == "int":
                        min_val = int(opt_spec.get("min_value", default_param_config.get("low", 1)))
                        max_val = int(opt_spec.get("max_value", default_param_config.get("high", 12)))
                        step = int(opt_spec.get("step", default_param_config.get("step", 1)))
                        for lookback in range(min_val, max_val + 1, step):
                            lookback_values_to_add.add(lookback)

                    for lookback in lookback_values_to_add:
                        required_features.add(Momentum(lookback_months=lookback))
                        required_features.add(VAMS(lookback_months=lookback))
                        # For DPVAMS, alpha is also needed. If alpha is optimized, this gets more complex.
                        # Assuming a default or common alpha if not explicitly handled.
                        # This part might need further refinement if alpha optimization is deeply tied here.
                        # For now, using a common placeholder or relying on strategy_class.get_required_features
                        # to handle DPVAMS based on static/optimized alpha.
                        # Let's assume alpha is handled by the strategy's get_required_features or is static for this feature collection.
                        # If 'alpha' is also in opt_spec, we might need to create combinations.
                        # For simplicity, we'll stick to the original DPVAMS addition or assume strategy handles it.
                        # A fixed common alpha for DPVAMS feature precomputation:
                        fixed_alpha_for_dpvams_precomp = 0.5 # Or fetch from strategy_params if available
                        alpha_param_config = OPTIMIZER_PARAMETER_DEFAULTS.get("alpha", {})
                        alpha_values = scen.get("strategy_params",{}).get("alpha") # static alpha

                        if isinstance(alpha_values, (int,float)): # static alpha
                             required_features.add(DPVAMS(lookback_months=lookback, alpha=alpha_values))
                        else: # alpha is optimized or not set, use placeholder or default
                            # Check if alpha is optimized
                            alpha_is_optimized = any(p["parameter"] == "alpha" for p in scen.get("optimize", []))
                            if alpha_is_optimized:
                                # If alpha is optimized, we'd ideally iterate its possible values too.
                                # This creates a cartesian product of features, which can be large.
                                # For now, let's add DPVAMS with a common alpha or skip if too complex here.
                                # Sticking to simpler: add with a default alpha if alpha is optimized.
                                # The strategy will ultimately pick the right one at runtime.
                                alpha_opt_spec = next((s for s in scen["optimize"] if s["parameter"] == "alpha"), None)
                                default_alpha_config = OPTIMIZER_PARAMETER_DEFAULTS.get("alpha", {})
                                alpha_type = alpha_opt_spec.get("type", default_alpha_config.get("type")) if alpha_opt_spec else default_alpha_config.get("type")

                                if alpha_type == "categorical":
                                    alpha_choices = alpha_opt_spec.get("values", default_alpha_config.get("values", [fixed_alpha_for_dpvams_precomp])) if alpha_opt_spec else default_alpha_config.get("values", [fixed_alpha_for_dpvams_precomp])
                                    for a_val in alpha_choices: required_features.add(DPVAMS(lookback_months=lookback, alpha=float(a_val)))
                                elif alpha_type == "float": # iterate through range
                                    a_min = float(alpha_opt_spec.get("min_value", default_alpha_config.get("low", 0.1))) if alpha_opt_spec else float(default_alpha_config.get("low", 0.1))
                                    a_max = float(alpha_opt_spec.get("max_value", default_alpha_config.get("high", 0.9))) if alpha_opt_spec else float(default_alpha_config.get("high", 0.9))
                                    a_step = float(alpha_opt_spec.get("step", default_alpha_config.get("step", 0.1))) if alpha_opt_spec else float(default_alpha_config.get("step", 0.1))
                                    current_a = a_min
                                    while current_a <= a_max:
                                        required_features.add(DPVAMS(lookback_months=lookback, alpha=round(current_a,4))) # round to avoid float issues
                                        current_a += a_step
                                else: # Default if alpha type is unknown or not optimized
                                     required_features.add(DPVAMS(lookback_months=lookback, alpha=fixed_alpha_for_dpvams_precomp))
                            else: # Alpha not optimized, use static from strategy_params or default
                                static_alpha = scen.get("strategy_params",{}).get("alpha", fixed_alpha_for_dpvams_precomp)
                                required_features.add(DPVAMS(lookback_months=lookback, alpha=float(static_alpha)))


    return required_features
