import logging
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import Numba optimizations with fallback
try:
    from ..numba_optimized import (
        rolling_sharpe_fast,
        rolling_sortino_fast, rolling_beta_fast, rolling_correlation_fast,
        rolling_sharpe_batch, rolling_sortino_batch, rolling_beta_batch,
        rolling_correlation_batch, rolling_downside_volatility_fast
    )
    from .base_sizer import BasePositionSizer
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def _normalize_weights(weights: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
    """Normalize weights to sum to 1, applying leverage."""
    weight_sums = weights.abs().sum(axis=1)
    # Avoid division by zero
    normalized = weights.div(weight_sums, axis=0)
    # Apply leverage
    if leverage != 1.0:
        normalized *= leverage
    return normalized


class EqualWeightSizer(BasePositionSizer):
    """Apply equal weighting to the signals."""

    def calculate_weights(self, signals: pd.DataFrame, *_, **kwargs) -> pd.DataFrame:
        leverage = kwargs.get("leverage", 1.0)
        weights = signals.abs()
        return _normalize_weights(weights, leverage)


def rolling_sharpe_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    **_,
) -> pd.DataFrame:
    """Size positions based on their rolling Sharpe ratio.

    Note: This sizer returns only positive weights, as it operates on the absolute
    value of the signals. The direction of the trade is determined by the strategy.
    """
    rets = prices.pct_change(fill_method=None).fillna(0)
    
    if NUMBA_AVAILABLE and not rets.empty:
        returns_matrix = rets.values
        sharpe_matrix = rolling_sharpe_batch(returns_matrix, window, annualization_factor=1.0)
        sharpe = pd.DataFrame(sharpe_matrix, index=rets.index, columns=rets.columns)
    else:
        sharpe = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            if NUMBA_AVAILABLE and not rets[col].isna().all():
                sharpe_values = rolling_sharpe_fast(rets[col].values, window, 1.0)
                sharpe[col] = sharpe_values
            else:
                if NUMBA_AVAILABLE and not rets[col].isna().all():
                    mean_values = rolling_mean_fast(rets[col].values, window)
                    std_values = rolling_std_fast(rets[col].values, window)
                    sharpe[col] = pd.Series(mean_values, index=rets.index) / pd.Series(std_values, index=rets.index).replace(0, np.nan)
                else:
                    mean_rets = rets[col].rolling(window).mean()
                    std_rets = rets[col].rolling(window).std()
                    sharpe[col] = mean_rets / std_rets.replace(0, np.nan)
    
    # Use absolute signals for sizing
    weights = signals.abs().mul(sharpe)
    return _normalize_weights(weights)


def rolling_sortino_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    target_return: float = 0.0,
    **_,
) -> pd.DataFrame:
    """Size positions based on their rolling Sortino ratio.

    Note: This sizer returns only positive weights, as it operates on the absolute
    value of the signals. The direction of the trade is determined by the strategy.
    """
    rets = prices.pct_change(fill_method=None).fillna(0)
    
    if NUMBA_AVAILABLE and not rets.empty:
        returns_matrix = rets.values
        sortino_matrix = rolling_sortino_batch(returns_matrix, window, target_return, annualization_factor=1.0)
        sortino = pd.DataFrame(sortino_matrix, index=rets.index, columns=rets.columns)
    else:
        sortino = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            if NUMBA_AVAILABLE and not rets[col].isna().all():
                sortino_values = rolling_sortino_fast(rets[col].values, window, target_return, 1.0)
                sortino[col] = sortino_values
            else:
                def downside(series):
                    downside_returns = series[series < target_return]
                    if len(downside_returns) == 0:
                        return 1e-9
                    return np.sqrt(np.mean((downside_returns - target_return) ** 2))
                
                if NUMBA_AVAILABLE and not rets[col].isna().all():
                    sortino_values = rolling_sortino_fast(rets[col].values, window, target_return, 1.0)
                    sortino[col] = pd.Series(sortino_values, index=rets.index)
                else:
                    mean_rets = rets[col].rolling(window).mean()
                    downside_dev = rets[col].rolling(window).apply(downside, raw=False)
                    sortino[col] = (mean_rets - target_return) / downside_dev.replace(0, np.nan)
    
    weights = signals.abs().mul(sortino)
    return _normalize_weights(weights)


def rolling_beta_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    **_,
) -> pd.DataFrame:
    """Size positions based on their rolling beta.

    Note: This sizer returns only positive weights, as it operates on the absolute
    value of the signals. The direction of the trade is determined by the strategy.
    """
    rets = prices.pct_change(fill_method=None).fillna(0)
    bench_rets = benchmark.pct_change(fill_method=None).fillna(0)
    
    if NUMBA_AVAILABLE and not rets.empty and not bench_rets.isna().all():
        returns_matrix = rets.values
        benchmark_returns = bench_rets.values
        beta_matrix = rolling_beta_batch(returns_matrix, benchmark_returns, window)
        beta = pd.DataFrame(beta_matrix, index=rets.index, columns=rets.columns)
    else:
        beta = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            if NUMBA_AVAILABLE and not rets[col].isna().all():
                beta_values = rolling_beta_fast(rets[col].values, bench_rets.values, window)
                beta[col] = beta_values
            else:
                if NUMBA_AVAILABLE and not rets[col].isna().all():
                    beta_values = rolling_beta_fast(rets[col].values, bench_rets.values, window)
                    beta[col] = pd.Series(beta_values, index=rets.index)
                else:
                    cov = rets[col].rolling(window).cov(bench_rets)
                    var = bench_rets.rolling(window).var()
                    beta[col] = cov / var
    
    factor = 1 / beta.abs().replace(0, np.nan)
    weights = signals.abs().mul(factor)
    return _normalize_weights(weights)


def rolling_benchmark_corr_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    **_,
) -> pd.DataFrame:
    """Size positions based on their rolling correlation with a benchmark.

    Note: This sizer returns only positive weights, as it operates on the absolute
    value of the signals. The direction of the trade is determined by the strategy.
    """
    rets = prices.pct_change(fill_method=None).fillna(0)
    bench_rets = benchmark.pct_change(fill_method=None).fillna(0)
    
    if NUMBA_AVAILABLE and not rets.empty and not bench_rets.isna().all():
        returns_matrix = rets.values
        benchmark_returns = bench_rets.values
        corr_matrix = rolling_correlation_batch(returns_matrix, benchmark_returns, window)
        corr = pd.DataFrame(corr_matrix, index=rets.index, columns=rets.columns)
    else:
        corr = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            if NUMBA_AVAILABLE and not rets[col].isna().all():
                corr_values = rolling_correlation_fast(rets[col].values, bench_rets.values, window)
                corr[col] = corr_values
            else:
                if NUMBA_AVAILABLE and not rets[col].isna().all():
                    corr_values = rolling_correlation_fast(rets[col].values, bench_rets.values, window)
                    corr[col] = pd.Series(corr_values, index=rets.index)
                else:
                    corr[col] = rets[col].rolling(window).corr(bench_rets)
    
    factor = 1 / (corr.abs() + 1e-9)
    weights = signals.abs().mul(factor)
    return _normalize_weights(weights)


def rolling_downside_volatility_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    daily_prices_for_vol: pd.DataFrame,
    window: int,
    target_volatility: float = 1.0,
    max_leverage: float = 2.0,
    **_,
) -> pd.DataFrame:
    """Size positions inversely proportional to downside volatility, scaled by a target volatility.

    Only negative returns are used when computing volatility so that
    upside moves do not lead to smaller position sizes.

    Note: This sizer returns only positive weights, as it operates on the absolute
    value of the signals. The direction of the trade is determined by the strategy.
    """
    
    # Calculate downside volatility for each asset using monthly prices
    if NUMBA_AVAILABLE:
        downside_vol_monthly = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            downside_vol_monthly[col] = rolling_downside_volatility_fast(prices[col].values, window)
    else:
        rets_monthly = prices.pct_change(fill_method=None).fillna(0)
        downside_monthly = rets_monthly.clip(upper=0)
        downside_sq_sum_monthly = (downside_monthly ** 2).rolling(window).sum()
        downside_vol_monthly = (downside_sq_sum_monthly / window).pow(0.5)
        downside_vol_monthly = pd.DataFrame(downside_vol_monthly, index=signals.index, columns=signals.columns)

    # Calculate initial factor based on target volatility and asset downside volatility
    epsilon = 1e-9
    factor = target_volatility / np.maximum(downside_vol_monthly, epsilon)
    factor = pd.DataFrame(factor, index=signals.index, columns=signals.columns)
    factor = factor.clip(upper=max_leverage)

    # Apply factor to absolute signals to get initial sized positions
    sized_initial = signals.abs().mul(factor)

    # Now, perform volatility targeting using daily data
    daily_weights_from_sized_initial = sized_initial.reindex(daily_prices_for_vol.index, method="ffill")
    daily_weights_from_sized_initial = daily_weights_from_sized_initial.shift(1).fillna(0.0)

    daily_rets_for_vol = daily_prices_for_vol.pct_change(fill_method=None).fillna(0)
    daily_portfolio_returns_initial = (daily_weights_from_sized_initial * daily_rets_for_vol).sum(axis=1)

    annualization_factor = np.sqrt(252)
    actual_portfolio_vol = daily_portfolio_returns_initial.rolling(window=window*21).std() * annualization_factor

    scaling_factor = target_volatility / np.maximum(actual_portfolio_vol, epsilon)
    scaling_factor = scaling_factor.clip(upper=max_leverage)

    scaling_factor_monthly = scaling_factor.reindex(signals.index, method="ffill")
    
    weights = sized_initial.mul(scaling_factor_monthly, axis=0)
    weights = weights.clip(upper=max_leverage) # Weights are now always positive

    weights = weights.fillna(0)

    return weights


SIZER_REGISTRY: Dict[str, Callable] = {
    "equal_weight": equal_weight_sizer,
    "rolling_sharpe": rolling_sharpe_sizer,
    "rolling_sortino": rolling_sortino_sizer,
    "rolling_beta": rolling_beta_sizer,
    "rolling_benchmark_corr": rolling_benchmark_corr_sizer,
    "rolling_downside_volatility": rolling_downside_volatility_sizer,
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


def get_position_sizer(name: str) -> Callable:
    try:
        return SIZER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown position sizer: {name}") from exc


def get_position_sizer_from_config(config: Dict[str, Any]) -> Callable:
    """Get a position sizer function from a configuration dictionary.
    
    This function centralizes the logic for getting a position sizer from configuration,
    avoiding repetition of the same pattern across the codebase.
    
    Args:
        config: Configuration dictionary that may contain a "position_sizer" key
        
    Returns:
        Callable: The position sizer function
        
    Example:
        sizer_func = get_position_sizer_from_config(scenario_cfg)
    """
    sizer_name = config.get("position_sizer", "equal_weight")
    return get_position_sizer(sizer_name)
