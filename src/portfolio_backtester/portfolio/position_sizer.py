import pandas as pd
import numpy as np
from typing import Callable, Dict


def equal_weight_sizer(signals: pd.DataFrame, *_, **__) -> pd.DataFrame:
    """Apply equal weighting to the signals."""
    return signals.div(signals.abs().sum(axis=1), axis=0)


def rolling_sharpe_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    mean = rets.rolling(window).mean()
    std = rets.rolling(window).std()
    sharpe = mean / std.replace(0, np.nan)
    sized = signals.mul(sharpe)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_sortino_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window: int,
    target_return: float = 0.0,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    mean = rets.rolling(window).mean() - target_return

    def downside(series):
        d = series[series < target_return]
        if len(d) == 0:
            return np.nan
        return np.sqrt(np.mean((d - target_return) ** 2))

    downside_dev = rets.rolling(window).apply(downside, raw=False)
    sortino = mean / downside_dev.replace(0, np.nan)
    sized = signals.mul(sortino)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_beta_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    bench_rets = benchmark.pct_change(fill_method=None).fillna(0)
    beta = pd.DataFrame(index=rets.index, columns=rets.columns)
    for col in rets.columns:
        cov = rets[col].rolling(window).cov(bench_rets)
        var = bench_rets.rolling(window).var()
        beta[col] = cov / var
    factor = 1 / beta.abs().replace(0, np.nan)
    sized = signals.mul(factor)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_benchmark_corr_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    **_,
) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).fillna(0)
    bench_rets = benchmark.pct_change(fill_method=None).fillna(0)
    corr = pd.DataFrame(index=rets.index, columns=rets.columns)
    for col in rets.columns:
        corr[col] = rets[col].rolling(window).corr(bench_rets)
    factor = 1 / (corr.abs() + 1e-9)
    sized = signals.mul(factor)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_downside_volatility_sizer(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int,
    target_volatility: float = 1.0, # New parameter for target volatility
    **_,
) -> pd.DataFrame:
    """Size positions inversely proportional to downside volatility, scaled by a target volatility.

    Only negative returns are used when computing volatility so that
    upside moves do not lead to smaller position sizes."""
    rets = prices.pct_change(fill_method=None).fillna(0)
    downside = rets.clip(upper=0)
    # Calculate rolling sum of squared downside returns and count of non-zero downside returns
    downside_sq_sum = (downside ** 2).rolling(window).sum()
    # A window might have all zero returns, so downside_vol would be 0.
    # We need to handle division by zero or by very small numbers.
    # Let's use a minimum threshold for downside_vol.
    epsilon = 1e-9  # Small constant to prevent division by zero

    # Calculate downside volatility.
    # .mean() would count all days in window. We only want to average over days with actual downside returns,
    # but standard deviation calculation typically uses N or N-1 of the window.
    # For this sizer, the intent is usually vol of negative returns.
    # If all returns in window are >=0, downside_vol will be 0.
    downside_vol = (downside_sq_sum / window).pow(0.5) # More aligned with typical volatility calc.

    # Replace exact zeros with epsilon to prevent division by zero and extremely large factors.
    # Also, if downside_vol is NaN (e.g., not enough data points yet in window), factor becomes NaN.
    # We will fill NaN factors with 0 later, effectively giving no weight.
    factor = target_volatility / np.maximum(downside_vol, epsilon)
    
    # Handle cases where factor might be inf (if downside_vol was epsilon and target_volatility is large)
    # or NaN (if downside_vol was NaN due to insufficient window data initially).
    # Fill NaN/inf with 0 to avoid issues in portfolio construction. This means no position.
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)

    sized = signals.mul(factor)
    # Normalize weights. If all factors for a given day are zero (e.g. all assets had zero downside vol),
    # then sum will be zero, leading to NaN weights. Handle this by filling NaN with 0.
    # It's important that signals are already 0 for assets not to be included.
    weight_sum = sized.abs().sum(axis=1)
    weights = sized.div(weight_sum, axis=0).fillna(0)

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
