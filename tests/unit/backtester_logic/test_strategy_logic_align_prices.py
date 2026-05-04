"""Contract for partial-universe price alignment used in ``strategy_logic`` signal paths."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _partial_universe_prices_like_strategy_logic(
    current_prices: pd.Series,
    universe_tickers: list[str],
) -> pd.Series:
    """Mirror ``generate_signals`` partial-universe branch (non-subset price index).

    See ``strategy_logic`` around the ``universe_set.issubset(prices_set)`` else-branch:
    build a Series on ``universe_tickers``, copy overlapping tickers, then ``ffill``.
    """

    universe_set = set(universe_tickers)
    prices_set = set(current_prices.index)
    if universe_set.issubset(prices_set):
        return current_prices.loc[universe_tickers].copy()

    universe_prices = pd.Series(index=universe_tickers, dtype=current_prices.dtype)
    for ticker in universe_set.intersection(prices_set):
        universe_prices[ticker] = current_prices[ticker]
    if universe_prices.isna().any():
        universe_prices = universe_prices.ffill()
    return universe_prices


def test_partial_universe_reindex_fill_matches_full_subset_path() -> None:
    """When all universe tickers exist in ``current``, behavior matches ``.loc`` order."""
    current = pd.Series({"A": 1.0, "B": 2.0})
    out = _partial_universe_prices_like_strategy_logic(current, ["B", "C", "A"])
    assert out.loc["B"] == 2.0
    assert out.loc["C"] == 2.0
    assert out.loc["A"] == 1.0


def test_partial_universe_leading_nan_unfilled() -> None:
    """Leading missing tickers have no prior value; ``ffill`` does not invent one."""
    current = pd.Series({"B": 2.0})
    out = _partial_universe_prices_like_strategy_logic(current, ["C", "B"])
    assert np.isnan(out.loc["C"])
    assert out.loc["B"] == 2.0
