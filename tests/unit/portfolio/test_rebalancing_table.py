"""Deterministic table tests for :mod:`portfolio_backtester.portfolio.rebalancing`.

Complements Hypothesis coverage in ``test_rebalancing_properties.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.portfolio.rebalancing import rebalance


@pytest.mark.parametrize(
    "frequency",
    ["D", "W", "M", "Q", "Y"],
    ids=["daily", "weekly", "monthly", "quarterly", "yearly"],
)
def test_rebalance_preserves_weights_columns_for_each_frequency(frequency: str) -> None:
    """Resampled output keeps asset columns; row weights remain normalized when input rows sum to 1."""
    idx = pd.date_range("2023-01-03", periods=40, freq="B")
    signals = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    rebalanced = rebalance(signals, frequency)
    assert isinstance(rebalanced, pd.DataFrame)
    assert list(rebalanced.columns) == ["A", "B"]
    sums = rebalanced[["A", "B"]].sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-9
    if frequency != "D":
        assert len(rebalanced) <= len(signals)


def test_rebalance_empty_frame_roundtrip() -> None:
    """Empty weights frame with DatetimeIndex stays empty through rebalance."""
    empty = pd.DataFrame(columns=["X", "Y"], index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
    out = rebalance(empty, "M")
    assert out.empty
    assert list(out.columns) == ["X", "Y"]
