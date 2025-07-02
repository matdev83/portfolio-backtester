"""Reusable signal generation helpers for strategies."""

from __future__ import annotations

import pandas as pd


# --------------------------------------------------------------------------- #
# Generic ranking-based signal generator                                     #
# --------------------------------------------------------------------------- #

def ranking_signal_generator(
    feature_name: str,
    *,
    dropna: bool = False,
    zero_if_any_nan: bool = False,
    immediate_derisk_days: int | None = None,
):
    """Factory returning a callable that generates signals.

    Parameters
    ----------
    feature_name:
        Name of the pre-computed feature in the ``features`` dictionary used
        for ranking assets.
    dropna:
        If ``True``, NaN values in the ranking series are dropped before
        computing weights.
    zero_if_any_nan:
        If ``True`` and any NaNs are present in the ranking series for a
        particular date, all weights are set to zero.
    immediate_derisk_days:
        Optional number of consecutive days below the benchmark SMA after which
        all weights are set to zero.  Requires ``sma_filter_window`` to be set in
        ``strategy_config``.
    """

    def _generator(strategy, prices: pd.DataFrame, features: dict, benchmark_data: pd.Series) -> pd.DataFrame:
        look_df: pd.DataFrame = features[feature_name]
        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0)

        # Setup immediate de-risk logic if requested
        if immediate_derisk_days is not None:
            sma_window = strategy.strategy_config.get("sma_filter_window")
            if sma_window:
                sma_feature_name = f"benchmark_sma_{sma_window}m"
                risk_on_series = features[sma_feature_name].reindex(prices.index, fill_value=1)
                under_sma_counter = 0
                derisk_flags = pd.Series(False, index=prices.index)
                for date in prices.index:
                    if risk_on_series.loc[date]:
                        under_sma_counter = 0
                    else:
                        under_sma_counter += 1
                        if under_sma_counter > immediate_derisk_days:
                            derisk_flags.loc[date] = True
            else:
                derisk_flags = pd.Series(False, index=prices.index)
        else:
            derisk_flags = pd.Series(False, index=prices.index)

        for date in prices.index:
            look = look_df.loc[date]

            if zero_if_any_nan and look.isna().any():
                w_new = pd.Series(0.0, index=prices.columns)
                weights.loc[date] = w_new
                w_prev = w_new
                continue

            if dropna:
                look = look.dropna()

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue

            cand = strategy._calculate_candidate_weights(look)
            w_new = strategy._apply_leverage_and_smoothing(cand, w_prev)

            if derisk_flags.loc[date]:
                w_new[:] = 0.0

            weights.loc[date] = w_new
            w_prev = w_new

        sma_window = strategy.strategy_config.get("sma_filter_window")
        if sma_window:
            sma_feature_name = f"benchmark_sma_{sma_window}m"
            risk_on = features[sma_feature_name].reindex(weights.index, fill_value=True)
            weights.loc[risk_on.index[~risk_on]] = 0.0

        return weights

    return _generator

__all__ = ["ranking_signal_generator"]
