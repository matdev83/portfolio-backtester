import datetime as dt

import pytest
import pandas as pd

from portfolio_backtester.universe_data.spy_holdings import (
    get_spy_holdings,
    get_top_weight_sp500_components,
)

pytestmark = pytest.mark.universe

dates_to_test = [
    # A weekend date to exercise fallback to previous business day
    ("2021-05-30", 30),  # Sunday
    ("2016-04-15", 25),  # daily data in SSGA bundle
    ("2017-02-28", 40),  # Month-end date with SSGA daily bundle
]


@pytest.mark.parametrize("date_str, n", dates_to_test)
def test_top_weight_components_length(date_str: str, n: int):
    comps = get_top_weight_sp500_components(date_str, n)
    assert isinstance(comps, list)
    # The universe should not be empty and should not exceed *n*
    assert 0 < len(comps) <= n

    # Each item must be a (ticker, weight) tuple
    assert all(isinstance(t, tuple) and len(t) == 2 for t in comps)
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in comps)

    # Ensure weights are sorted descending
    weights = [w for _, w in comps]
    assert weights == sorted(weights, reverse=True)

    # No NaN allowed in weights
    assert not any(pd.isna(w) for w in weights)


def test_no_lookahead_bias():
    """Ensure the helper never looks *forward* in time when exact=False."""
    target = dt.date(2021, 5, 30)  # Sunday, markets closed
    df = get_spy_holdings(target, exact=False)
    # Returned holdings date must not be after the target date
    returned_date = df["date"].iloc[0].date()
    assert returned_date <= target


def test_cache_consistency():
    date, n = "2021-05-28", 30
    first_call = get_top_weight_sp500_components(date, n)
    second_call = get_top_weight_sp500_components(date, n)
    assert first_call == second_call  # LRU cache should guarantee identical result
