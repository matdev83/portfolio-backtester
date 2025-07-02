
import numpy as np
import pandas as pd


def equal_weight_sizer(signals: pd.DataFrame) -> pd.DataFrame:
    """Applies equal weighting to the signals."""
    sized_signals = signals.div(signals.abs().sum(axis=1), axis=0)
    return sized_signals


def rolling_sharpe_sizer(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Size positions proportionally to rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std().replace(0, np.nan)
    sharpe = (rolling_mean / rolling_std).fillna(0)
    sized = signals.mul(sharpe)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_sortino_sizer(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    window: int,
    target_return: float = 0.0,
) -> pd.DataFrame:
    """Size positions proportionally to rolling Sortino ratio."""

    def downside_dev(x: pd.Series) -> float:
        downside = x[x < target_return] - target_return
        if downside.empty:
            return 0.0
        return np.sqrt(np.mean(downside**2))

    rolling_mean = returns.rolling(window).mean()
    downside = returns.rolling(window).apply(downside_dev, raw=False)
    downside = downside.replace(0, np.nan)
    sortino = (rolling_mean / downside).fillna(0)
    sized = signals.mul(sortino)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_beta_sizer(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Size positions inversely proportional to rolling beta versus benchmark."""

    cov = returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    beta = cov.div(var, axis=0).fillna(0)
    factor = 1 / (1 + beta.abs())
    sized = signals.mul(factor)
    return sized.div(sized.abs().sum(axis=1), axis=0)


def rolling_corr_sizer(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Size positions inversely proportional to Spearman correlation with benchmark."""

    ranks = returns.rank(axis=0, pct=True)
    bench_rank = benchmark_returns.rank(pct=True)
    corr = ranks.rolling(window).corr(bench_rank)
    corr = corr.replace([np.inf, -np.inf], 0).fillna(0)
    factor = (1 - corr.abs()).clip(lower=0)
    sized = signals.mul(factor)
    return sized.div(sized.abs().sum(axis=1), axis=0)


POSITION_SIZER_REGISTRY = {
    "equal_weight": equal_weight_sizer,
    "rolling_sharpe": rolling_sharpe_sizer,
    "rolling_sortino": rolling_sortino_sizer,
    "rolling_beta": rolling_beta_sizer,
    "rolling_corr": rolling_corr_sizer,
}

__all__ = [
    "equal_weight_sizer",
    "rolling_sharpe_sizer",
    "rolling_sortino_sizer",
    "rolling_beta_sizer",
    "rolling_corr_sizer",
    "POSITION_SIZER_REGISTRY",
]

