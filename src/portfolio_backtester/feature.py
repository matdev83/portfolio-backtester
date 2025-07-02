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


def get_required_features_from_scenarios(strategy_configs: list, strategy_registry: dict) -> Set[Feature]:
    """
    Gathers all unique feature requirements from a list of strategy configurations.
    """
    required_features = set()
    for scen in strategy_configs:
        strategy_name = scen.get("strategy")
        strategy_class = strategy_registry.get(strategy_name)
        if strategy_class:
            required_features.update(strategy_class.get_required_features(scen))

        # Also check for features needed for optimization
        if "optimize" in scen:
            for opt_param in scen["optimize"]:
                if opt_param["parameter"] == "sma_filter_window":
                    min_val = opt_param.get("min_value", 2)
                    max_val = opt_param.get("max_value", 24)
                    step = opt_param.get("step", 1)
                    for window in range(min_val, max_val + 1, step):
                        required_features.add(BenchmarkSMA(sma_filter_window=window))
                elif opt_param["parameter"] == "lookback_months":
                    min_val = opt_param.get("min_value", 1)
                    max_val = opt_param.get("max_value", 12)
                    step = opt_param.get("step", 1)
                    for lookback in range(min_val, max_val + 1, step):
                        required_features.add(Momentum(lookback_months=lookback))
                        required_features.add(VAMS(lookback_months=lookback))
                        required_features.add(DPVAMS(lookback_months=lookback, alpha=0.5)) # Alpha is a placeholder, as it's also optimized

    return required_features
