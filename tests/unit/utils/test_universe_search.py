import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.universe_search import (
    compute_subset_returns,
    compute_symbol_contributions,
    normalize_candidates,
    sample_subsets,
)


def test_normalize_candidates_dedup_and_strip():
    result = normalize_candidates([" amex:spy ", "AMEX:SPY", "  ", "nasdaq:qqq"])
    assert result == ["AMEX:SPY", "NASDAQ:QQQ"]


def test_normalize_candidates_empty_raises():
    with pytest.raises(ValueError):
        normalize_candidates([" ", ""])


def test_sample_subsets_unique():
    rng = np.random.default_rng(123)
    subsets = sample_subsets(["A", "B", "C", "D"], subset_size=2, n_samples=3, rng=rng)
    assert len(subsets) == 3
    assert all(len(subset) == 2 for subset in subsets)
    assert len({tuple(subset) for subset in subsets}) == 3


def test_compute_symbol_contributions_shifted():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates)
    returns = pd.DataFrame({"A": [0.1, 0.0, 0.0], "B": [0.0, 0.2, 0.0]}, index=dates)
    contributions = compute_symbol_contributions(weights, returns)
    assert contributions.loc["A", "total_contribution"] == pytest.approx(0.0)
    assert contributions.loc["B", "total_contribution"] == pytest.approx(0.1)


def test_compute_subset_returns_normalize():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    weights = pd.DataFrame({"A": [0.4, 0.4, 0.4], "B": [0.1, 0.1, 0.1]}, index=dates)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.1, 0.1, 0.1]}, index=dates)
    subset_returns = compute_subset_returns(weights, returns, ["B"], normalize_weights=True)
    assert subset_returns.iloc[1] == pytest.approx(0.1)
