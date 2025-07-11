import pandas as pd
import numpy as np
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# Import Numba optimizations with fallback
try:
    from ..numba_optimized import (
        rolling_mean_fast, rolling_std_fast, rolling_sharpe_fast,
        rolling_sortino_fast, rolling_beta_fast, rolling_correlation_fast
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def equal_weight_sizer(signals: pd.DataFrame, *_, **__) -> pd.DataFrame:
    """Apply equal weighting to the signals."""
    signal_sums = signals.abs().sum(axis=1)
    # Handle case where signal sum is zero to avoid NaN from division by zero
    result = signals.div(signal_sums, axis=0)
    # Keep NaN values when all signals are zero (as expected by tests)
    return result


def rolling_sharpe_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    
    # Use Numba optimization if available and data is suitable
    if NUMBA_AVAILABLE and not rets.empty:
        # Apply Numba optimization column by column
        sharpe_data = {}
        for col in rets.columns:
            if not rets[col].isna().all():
                sharpe_values = rolling_sharpe_fast(rets[col].values, window)
                sharpe_data[col] = sharpe_values
            else:
                sharpe_data[col] = np.full(len(rets), np.nan)
        
        sharpe = pd.DataFrame(sharpe_data, index=rets.index)
    else:
        # Fallback to pandas
        mean = rets.rolling(window).mean()
        std = rets.rolling(window).std()
        sharpe = mean / std.replace(0, np.nan)
    
    sized = signals.mul(sharpe)
    sized_sums = sized.abs().sum(axis=1)
    result = sized.div(sized_sums, axis=0)
    return result


def rolling_sortino_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    target_return: float = 0.0,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    
    # Use Numba optimization if available and data is suitable
    if NUMBA_AVAILABLE and not rets.empty:
        # Apply Numba optimization column by column
        sortino_data = {}
        for col in rets.columns:
            if not rets[col].isna().all():
                sortino_values = rolling_sortino_fast(rets[col].values, window, target_return)
                sortino_data[col] = sortino_values
            else:
                sortino_data[col] = np.full(len(rets), np.nan)
        
        sortino = pd.DataFrame(sortino_data, index=rets.index)
    else:
        # Fallback to pandas
        mean = rets.rolling(window).mean() - target_return

        def downside(series):
            d = series[series < target_return]
            if len(d) == 0:
                return np.nan
            return np.sqrt(np.mean((d - target_return) ** 2))

        downside_dev = rets.rolling(window).apply(downside, raw=False)
        sortino = mean / downside_dev.replace(0, np.nan)
    
    sized = signals.mul(sortino)
    sized_sums = sized.abs().sum(axis=1)
    result = sized.div(sized_sums, axis=0)
    return result


def rolling_beta_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    bench_rets = benchmark.pct_change(fill_method=None).fillna(0)
    
    # Use Numba optimization if available and data is suitable
    if NUMBA_AVAILABLE and not rets.empty and not bench_rets.isna().all():
        # Apply Numba optimization column by column
        beta_data = {}
        for col in rets.columns:
            if not rets[col].isna().all():
                beta_values = rolling_beta_fast(rets[col].values, bench_rets.values, window)
                beta_data[col] = beta_values
            else:
                beta_data[col] = np.full(len(rets), np.nan)
        
        beta = pd.DataFrame(beta_data, index=rets.index)
    else:
        # Fallback to pandas
        beta = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            cov = rets[col].rolling(window).cov(bench_rets)
            var = bench_rets.rolling(window).var()
            beta[col] = cov / var
    
    factor = 1 / beta.abs().replace(0, np.nan)
    sized = signals.mul(factor)
    sized_sums = sized.abs().sum(axis=1)
    result = sized.div(sized_sums, axis=0)
    return result


def rolling_benchmark_corr_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    bench_rets = benchmark.pct_change(fill_method=None).fillna(0)
    
    # Use Numba optimization if available and data is suitable
    if NUMBA_AVAILABLE and not rets.empty and not bench_rets.isna().all():
        # Apply Numba optimization column by column
        corr_data = {}
        for col in rets.columns:
            if not rets[col].isna().all():
                corr_values = rolling_correlation_fast(rets[col].values, bench_rets.values, window)
                corr_data[col] = corr_values
            else:
                corr_data[col] = np.full(len(rets), np.nan)
        
        corr = pd.DataFrame(corr_data, index=rets.index)
    else:
        # Fallback to pandas
        corr = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            corr[col] = rets[col].rolling(window).corr(bench_rets)
    
    factor = 1 / (corr.abs() + 1e-9)
    sized = signals.mul(factor)
    sized_sums = sized.abs().sum(axis=1)
    result = sized.div(sized_sums, axis=0)
    return result


def rolling_downside_volatility_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame, # This is monthly prices for signals
    benchmark: pd.Series,
    daily_prices_for_vol: pd.DataFrame, # New parameter for daily prices for vol calc
    window: int,
    target_volatility: float = 1.0,
    max_leverage: float = 2.0,
    **_,
) -> pd.DataFrame:
    """Size positions inversely proportional to downside volatility, scaled by a target volatility.

    Only negative returns are used when computing volatility so that
    upside moves do not lead to smaller position sizes."""
    
    # Calculate downside volatility for each asset using monthly prices
    if NUMBA_AVAILABLE and 'rolling_downside_volatility_fast' in globals():
        downside_vol_monthly = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            downside_vol_monthly[col] = rolling_downside_volatility_fast(prices[col].values, window)
    else:
        rets_monthly = prices.pct_change(fill_method=None).fillna(0)
        downside_monthly = rets_monthly.clip(upper=0)
        downside_sq_sum_monthly = (downside_monthly ** 2).rolling(window).sum()
        downside_vol_monthly = (downside_sq_sum_monthly / window).pow(0.5)
        downside_vol_monthly = pd.DataFrame(downside_vol_monthly, index=signals.index, columns=signals.columns)
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"downside_vol_monthly:\n{downside_vol_monthly}")

    # Calculate initial factor based on target volatility and asset downside volatility
    epsilon = 1e-9
    factor = target_volatility / np.maximum(downside_vol_monthly, epsilon)
    factor = pd.DataFrame(factor, index=signals.index, columns=signals.columns)
    factor = factor.clip(upper=max_leverage)
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"factor (after clip):\n{factor}")

    # Apply factor to signals to get initial sized positions
    sized_initial = signals.mul(factor)
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"sized_initial:\n{sized_initial}")

    # Now, perform volatility targeting using daily data
    # Expand monthly sized_initial to daily frequency (forward fill)
    daily_weights_from_sized_initial = sized_initial.reindex(daily_prices_for_vol.index, method="ffill")
    daily_weights_from_sized_initial = daily_weights_from_sized_initial.shift(1).fillna(0.0) # Shift to avoid look-ahead bias
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"daily_weights_from_sized_initial:\n{daily_weights_from_sized_initial}")

    # Calculate daily returns for the assets
    daily_rets_for_vol = daily_prices_for_vol.pct_change(fill_method=None).fillna(0)
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"daily_rets_for_vol:\n{daily_rets_for_vol}")

    # Calculate daily portfolio returns based on initial sized weights
    daily_portfolio_returns_initial = (daily_weights_from_sized_initial * daily_rets_for_vol).sum(axis=1)
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"daily_portfolio_returns_initial:\n{daily_portfolio_returns_initial}")

    # Calculate rolling actual portfolio volatility (annualized)
    # Assuming 252 trading days in a year for annualization
    annualization_factor = np.sqrt(252)
    actual_portfolio_vol = daily_portfolio_returns_initial.rolling(window=window*21).std() * annualization_factor # Approx monthly to daily window
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"actual_portfolio_vol:\n{actual_portfolio_vol}")

    # Calculate scaling factor to hit target volatility
    # Avoid division by zero or very small numbers for actual_portfolio_vol
    scaling_factor = target_volatility / np.maximum(actual_portfolio_vol, epsilon)
    scaling_factor = scaling_factor.clip(upper=max_leverage) # Cap the scaling factor as well
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"scaling_factor (after clip):\n{scaling_factor}")

    # Apply the scaling factor to the initial sized positions
    # Need to reindex scaling_factor to match signals index (monthly)
    scaling_factor_monthly = scaling_factor.reindex(signals.index, method="ffill")
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"scaling_factor_monthly:\n{scaling_factor_monthly}")
    
    # Final weights are initial sized signals multiplied by the overall scaling factor
    weights = sized_initial.mul(scaling_factor_monthly, axis=0)
    
    # Ensure weights are capped by max_leverage after scaling
    weights = weights.clip(upper=max_leverage, lower=-max_leverage) # Assuming long/short possible, clip both sides

    # Fill any remaining NaNs with 0 (e.g., for periods where no signals or vol could be calculated)
    weights = weights.fillna(0)
    if logger.isEnabledFor(logging.DEBUG):
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"final weights:\n{weights}")

    return weights


SIZER_REGISTRY: Dict[str, Callable] = {
    "equal_weight": equal_weight_sizer,
    "rolling_sharpe": rolling_sharpe_sizer,
    "rolling_sortino": rolling_sortino_sizer,
    "rolling_beta": rolling_beta_sizer,
    "rolling_benchmark_corr": rolling_benchmark_corr_sizer,
    "rolling_downside_volatility": rolling_downside_volatility_sizer,
}


def get_position_sizer(name: str) -> Callable:
    try:
        return SIZER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown position sizer: {name}") from exc
