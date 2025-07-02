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
    downside_vol = (downside ** 2).rolling(window).mean().pow(0.5)
    
    # Scale the factor by target_volatility. If target_volatility is 1.0, no change.
    # If downside_vol is 0, replace with a small number to avoid division by zero.
    factor = target_volatility / downside_vol.replace(0, np.nan)
    
    sized = signals.mul(factor)
    return sized.div(sized.abs().sum(axis=1), axis=0)


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
