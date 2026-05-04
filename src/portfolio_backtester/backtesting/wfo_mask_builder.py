"""Walk-forward overlay test-period mask construction."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def build_wfo_test_mask(overlay_diagnostics: Dict[str, Any], index: pd.Index) -> pd.Series:
    """Return a boolean series True on dates that fall inside any WFO test window."""
    windows = overlay_diagnostics.get("windows", []) if overlay_diagnostics else []
    if not windows:
        return pd.Series(False, index=index)

    try:
        idx = pd.DatetimeIndex(index)
    except (TypeError, ValueError):
        logger.debug(
            "WFO test mask: index not convertible to DatetimeIndex; returning empty mask.",
            exc_info=True,
        )
        return pd.Series(False, index=index)

    if idx.tz is not None:
        idx = idx.tz_convert(None)

    mask_vals = pd.Series(False, index=index)
    for window in windows:
        test_start = window.get("test_start")
        test_end = window.get("test_end")
        if not test_start or not test_end:
            continue
        try:
            start_ts = pd.Timestamp(test_start)
            end_ts = pd.Timestamp(test_end)
        except (ValueError, TypeError):
            logger.debug(
                "WFO test mask: skipping window with invalid test_start/test_end.",
                exc_info=True,
            )
            continue
        mask_vals |= (idx >= start_ts) & (idx <= end_ts)

    return mask_vals
