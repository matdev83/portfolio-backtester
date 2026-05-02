"""Tests for expanding-window iloc slicing and close-panel row access."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.strategy_logic import _expanding_iloc_ends


def test_expanding_ends_matches_boolean_mask_counts() -> None:
    idx = pd.to_datetime(pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]))
    rebalance = pd.to_datetime(pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"]))
    ends, masks = _expanding_iloc_ends(idx, rebalance)
    assert ends is not None and masks is None
    for i, d in enumerate(rebalance):
        assert int((idx <= d).sum()) == int(ends[i])


def test_expanding_ends_legacy_when_non_monotonic() -> None:
    idx = pd.to_datetime(pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"]))
    rebalance = pd.to_datetime(pd.DatetimeIndex(["2020-01-02"]))
    ends, masks = _expanding_iloc_ends(idx, rebalance)
    assert ends is None and masks is not None
    assert int(masks[rebalance[0]].sum()) == int((idx <= rebalance[0]).sum())


def test_close_panel_row_matches_loc_xs_multiindex() -> None:
    dates = pd.to_datetime(pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"]))
    tickers = ["AAA", "BBB"]
    cols = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close"]],
        names=["Ticker", "Field"],
    )
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(dates), len(cols)))
    ohlc = pd.DataFrame(data, index=dates, columns=cols)

    close_panel = ohlc.xs("Close", level="Field", axis=1)
    for d in dates:
        d_ts = pd.Timestamp(d)
        pos = cast(Any, ohlc.index.searchsorted(d_ts, side="right"))
        end = int(pos)
        assert end > 0
        row_iloc = close_panel.iloc[end - 1]
        row_loc_xs_raw = ohlc.loc[d_ts].xs("Close", level="Field")
        row_loc_xs = cast(pd.Series, row_loc_xs_raw)
        pd.testing.assert_series_equal(row_iloc, row_loc_xs, check_names=False)
