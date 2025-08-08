import logging
from typing import Dict, Any, Type

import numpy as np
import pandas as pd

from .base_sizer import BasePositionSizer
from ..numba_optimized import (
    rolling_sharpe_batch,
    rolling_sortino_batch,
    rolling_beta_batch,
    rolling_correlation_batch,
    rolling_downside_volatility_fast,
)


logger = logging.getLogger(__name__)


def _normalize_weights(weights: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
    """Normalize weights to sum to 1, applying leverage."""
    weight_sums = weights.abs().sum(axis=1)
    # Avoid division by zero
    normalized = weights.div(weight_sums, axis=0).fillna(0.0)
    # Apply leverage
    if leverage != 1.0:
        normalized *= leverage
    return normalized


class EqualWeightSizer(BasePositionSizer):
    """Apply equal weighting to the signals."""

    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        leverage = kwargs.get("leverage", 1.0)
        weights = signals.abs()
        return _normalize_weights(weights, leverage)


class RollingSharpeSizer(BasePositionSizer):
    """Size positions based on their rolling Sharpe ratio."""

    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        window = kwargs.get("window", 252)
        rets = prices.pct_change(fill_method=None).fillna(0)

        # Handle empty data edge case
        if rets.empty:
            return signals.abs()

        # Use optimized batch calculation with proper ddof=1 handling
        returns_matrix: np.ndarray = rets.values.astype(np.float64)
        sharpe_matrix = rolling_sharpe_batch(returns_matrix, window, annualization_factor=1.0)
        sharpe = pd.DataFrame(sharpe_matrix, index=rets.index, columns=rets.columns)

        # Use absolute signals for sizing
        weights = signals.abs().mul(sharpe.abs())
        return _normalize_weights(weights)


class RollingSortinoSizer(BasePositionSizer):
    """Size positions based on their rolling Sortino ratio."""

    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        window = kwargs.get("window", 252)
        target_return = kwargs.get("target_return", 0.0)
        rets = prices.pct_change(fill_method=None).fillna(0)

        # Handle empty data edge case
        if rets.empty:
            return signals.abs()

        # Use optimized batch calculation with proper ddof=1 handling
        returns_matrix: np.ndarray = rets.values.astype(np.float64)
        sortino_matrix = rolling_sortino_batch(
            returns_matrix, window, target_return, annualization_factor=1.0
        )
        sortino = pd.DataFrame(sortino_matrix, index=rets.index, columns=rets.columns)

        weights = signals.abs().mul(sortino.abs())
        return _normalize_weights(weights)


class RollingBetaSizer(BasePositionSizer):
    """Size positions based on their rolling beta."""

    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        benchmark = kwargs.get("benchmark")
        window = kwargs.get("window", 252)
        rets = prices.pct_change(fill_method=None).fillna(0)
        bench_rets = (
            benchmark.pct_change(fill_method=None).fillna(0)
            if benchmark is not None
            else pd.Series()
        )

        # Handle empty data edge case
        if rets.empty or bench_rets.isna().all() or benchmark is None:
            return signals.abs()

        # Use optimized batch calculation
        returns_matrix = rets.values.astype(np.float64)
        benchmark_returns = bench_rets.values.astype(np.float64)
        beta_matrix = rolling_beta_batch(returns_matrix, benchmark_returns, window)
        beta = pd.DataFrame(beta_matrix, index=rets.index, columns=rets.columns)

        factor = 1 / beta.abs().replace(0, np.nan)
        weights = signals.abs().mul(factor)
        return _normalize_weights(weights)


class RollingBenchmarkCorrSizer(BasePositionSizer):
    """Size positions based on their rolling correlation with a benchmark."""

    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        benchmark = kwargs.get("benchmark")
        window = kwargs.get("window", 252)
        rets = prices.pct_change(fill_method=None).fillna(0)
        bench_rets = (
            benchmark.pct_change(fill_method=None).fillna(0)
            if benchmark is not None
            else pd.Series()
        )

        # Handle empty data edge case
        if rets.empty or bench_rets.isna().all() or benchmark is None:
            return signals.abs()

        # Use optimized batch calculation
        returns_matrix = rets.values.astype(np.float64)
        benchmark_returns = bench_rets.values.astype(np.float64)
        corr_matrix = rolling_correlation_batch(returns_matrix, benchmark_returns, window)
        corr = pd.DataFrame(corr_matrix, index=rets.index, columns=rets.columns)

        factor = 1 / (corr.abs() + 1e-9)
        weights = signals.abs().mul(factor)
        return _normalize_weights(weights)


class RollingDownsideVolatilitySizer(BasePositionSizer):
    """Size positions inversely proportional to downside volatility, scaled by a target volatility."""

    def calculate_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        benchmark = kwargs.get("benchmark")
        daily_prices_for_vol = kwargs.get("daily_prices_for_vol")
        window = kwargs.get("window", 252)
        target_volatility = kwargs.get("target_volatility", 1.0)
        max_leverage = kwargs.get("max_leverage", 2.0)

        # Handle None cases
        if benchmark is None or daily_prices_for_vol is None:
            return signals.abs()

        # Use optimized downside volatility calculation
        downside_vol_monthly = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            downside_vol_monthly[col] = rolling_downside_volatility_fast(
                prices[col].values.astype(np.float64), window
            )

        epsilon = 1e-9
        factor = target_volatility / np.maximum(downside_vol_monthly, epsilon)
        factor = pd.DataFrame(factor, index=signals.index, columns=signals.columns)
        factor = factor.clip(upper=max_leverage)

        sized_initial = signals.abs().mul(factor)

        daily_weights_from_sized_initial = (
            sized_initial.reindex(daily_prices_for_vol.index, method="ffill")
            if daily_prices_for_vol is not None
            else pd.DataFrame()
        )
        daily_weights_from_sized_initial = daily_weights_from_sized_initial.shift(1).fillna(0.0)

        daily_rets_for_vol = (
            daily_prices_for_vol.pct_change(fill_method=None).fillna(0)
            if daily_prices_for_vol is not None
            else pd.DataFrame()
        )
        daily_portfolio_returns_initial = (
            daily_weights_from_sized_initial * daily_rets_for_vol
        ).sum(axis=1)

        annualization_factor = np.sqrt(252)
        actual_portfolio_vol = (
            daily_portfolio_returns_initial.rolling(window=window * 21).std() * annualization_factor
        )

        scaling_factor = target_volatility / np.maximum(actual_portfolio_vol, epsilon)
        scaling_factor = scaling_factor.clip(upper=max_leverage)

        scaling_factor_monthly = (
            scaling_factor.reindex(signals.index, method="ffill")
            if scaling_factor is not None
            else pd.Series()
        )

        weights = sized_initial.mul(scaling_factor_monthly, axis=0)
        weights = weights.clip(upper=max_leverage)

        weights = weights.fillna(0)

        return weights


SIZER_REGISTRY: Dict[str, Type[BasePositionSizer]] = {
    "equal_weight": EqualWeightSizer,
    "rolling_sharpe": RollingSharpeSizer,
    "rolling_sortino": RollingSortinoSizer,
    "rolling_beta": RollingBetaSizer,
    "rolling_benchmark_corr": RollingBenchmarkCorrSizer,
    "rolling_downside_volatility": RollingDownsideVolatilitySizer,
}

SIZER_PARAM_MAPPING = {
    "sizer_sharpe_window": "window",
    "sizer_sortino_window": "window",
    "sizer_beta_window": "window",
    "sizer_corr_window": "window",
    "sizer_dvol_window": "window",
    "sizer_target_return": "target_return",
    "sizer_max_leverage": "max_leverage",
}


def get_position_sizer(name: str) -> BasePositionSizer:
    try:
        return SIZER_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown position sizer: {name}") from exc


# Internal function used by position sizer providers
def get_position_sizer_from_config(config: Dict[str, Any]) -> BasePositionSizer:
    """Get a position sizer object from a configuration dictionary."""
    sizer_name = config.get("position_sizer", "equal_weight")
    return get_position_sizer(sizer_name)
