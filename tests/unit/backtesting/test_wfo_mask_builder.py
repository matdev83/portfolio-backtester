"""Unit tests for walk-forward test-period mask builder."""

import pandas as pd

from portfolio_backtester.backtesting.wfo_mask_builder import build_wfo_test_mask


def test_build_wfo_test_mask_empty_windows() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    mask = build_wfo_test_mask({}, idx)
    assert not mask.any()


def test_build_wfo_test_mask_single_window() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    overlay = {
        "windows": [
            {"test_start": "2020-01-03", "test_end": "2020-01-05"},
        ]
    }
    mask = build_wfo_test_mask(overlay, idx)
    assert mask.loc["2020-01-03"]
    assert mask.loc["2020-01-05"]
    assert not mask.loc["2020-01-02"]
    assert not mask.loc["2020-01-06"]
