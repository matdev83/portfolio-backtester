"""Sparse target-weight execution timing (decision date -> execution date)."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, cast

import pandas as pd

TradeExecutionTimingName = Literal["bar_close", "next_bar_open"]

TRADE_EXECUTION_TIMING_DEFAULT: TradeExecutionTimingName = "bar_close"

VALID_TRADE_EXECUTION_TIMINGS: frozenset[str] = frozenset({"bar_close", "next_bar_open"})


def map_sparse_target_weights_to_execution_dates(
    weights: pd.DataFrame,
    *,
    trade_execution_timing: str,
    calendar: pd.DatetimeIndex,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Map sparse event/decision rows to execution dates on ``calendar``.

    Args:
        weights: Sparse targets indexed by decision/event timestamps (rows), assets as columns.
        trade_execution_timing: ``bar_close`` or ``next_bar_open``.
        calendar: Strictly increasing trading-session index (e.g. business days).
        logger: Optional logger for overflow/drop warnings.

    Returns:
        Sparse ``DataFrame`` indexed by execution dates (subset of ``calendar``).

    Raises:
        ValueError: If ``trade_execution_timing`` is not a supported mode.
    """
    if trade_execution_timing not in VALID_TRADE_EXECUTION_TIMINGS:
        raise ValueError(
            f"Invalid trade_execution_timing {trade_execution_timing!r}; "
            f"must be one of {sorted(VALID_TRADE_EXECUTION_TIMINGS)}"
        )

    log = logger or logging.getLogger(__name__)
    if not isinstance(calendar, pd.DatetimeIndex):
        calendar = pd.DatetimeIndex(calendar)

    if weights.empty:
        return weights.copy()

    keep = ~weights.isna().all(axis=1)
    w = weights.loc[keep].copy()

    if w.empty:
        tz = getattr(weights.index, "tz", None)
        return pd.DataFrame(columns=list(weights.columns), index=pd.DatetimeIndex([], tz=tz))

    if trade_execution_timing == "bar_close":
        return w.sort_index()

    mapped_idx: list[pd.Timestamp] = []
    out_rows: list[pd.Series] = []

    for ts, row in w.iterrows():
        dec = pd.Timestamp(cast(Any, ts))
        pos = int(calendar.searchsorted(dec, side="right"))
        if pos >= len(calendar):
            log.warning(
                "trade_execution_timing=next_bar_open: dropping event at %s — no later "
                "session on supplied calendar",
                dec,
            )
            continue
        exec_ts = pd.Timestamp(calendar[pos])
        mapped_idx.append(exec_ts)
        out_rows.append(row)

    if not out_rows:
        tz = getattr(w.index, "tz", None)
        return pd.DataFrame(columns=list(w.columns), index=pd.DatetimeIndex([], tz=tz))

    out = pd.DataFrame(out_rows, index=pd.DatetimeIndex(mapped_idx), columns=w.columns)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()
