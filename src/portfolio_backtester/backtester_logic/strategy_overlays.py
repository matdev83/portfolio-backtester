from __future__ import annotations

import logging
import math
from collections import deque
from datetime import tzinfo
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def apply_wfo_scaling_and_kill_switch(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    overlay_config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply WFO-style scaling and kill-switch logic to a weights DataFrame.

    Args:
        weights: Target weights indexed by date.
        asset_returns: Daily asset returns indexed by date (close-to-close).
        overlay_config: Overlay settings (train/test window sizes, risk caps).

    Returns:
        Tuple of (adjusted_weights, diagnostics).
    """
    if weights.empty or asset_returns.empty:
        return weights, {"windows": []}

    aligned_returns = asset_returns.reindex(weights.index)
    aligned_returns = aligned_returns.fillna(0.0)

    train_years = int(overlay_config.get("train_years", 5))
    test_years = int(overlay_config.get("test_years", 1))
    step_years = int(overlay_config.get("step_years", test_years))
    if step_years <= 0:
        step_years = test_years
    min_train_days = int(overlay_config.get("min_train_days", train_years * _TRADING_DAYS_PER_YEAR))

    window_start_raw = overlay_config.get("window_start_date")
    window_end_raw = overlay_config.get("window_end_date")
    window_start = pd.Timestamp(window_start_raw) if window_start_raw else None
    window_end = pd.Timestamp(window_end_raw) if window_end_raw else None

    target_vol = float(overlay_config.get("target_vol_annual", 0.12))
    target_max_dd = float(overlay_config.get("target_max_drawdown", 0.15))
    max_gross = float(overlay_config.get("max_gross_exposure", 2.0))
    min_gross = float(overlay_config.get("min_gross_exposure", 0.0))
    allow_leverage = bool(overlay_config.get("allow_leverage", True))

    kill_enabled = bool(overlay_config.get("kill_switch_enabled", True))
    kill_dd = float(overlay_config.get("kill_switch_drawdown", 0.30))
    kill_roll_days = int(overlay_config.get("kill_switch_rolling_days", 63))
    kill_roll_min = float(overlay_config.get("kill_switch_min_return", 0.0))

    windows = _generate_wfo_windows(
        weights.index,
        train_years,
        test_years,
        step_years=step_years,
        window_start=window_start,
        window_end=window_end,
    )
    diagnostics: Dict[str, Any] = {"windows": []}

    adjusted = weights.copy()
    base_returns = _compute_portfolio_returns(adjusted, aligned_returns)

    for window in windows:
        train_start, train_end, test_start, test_end = window
        train_mask = (base_returns.index >= train_start) & (base_returns.index <= train_end)
        test_mask = pd.Series(
            (adjusted.index >= test_start) & (adjusted.index <= test_end),
            index=adjusted.index,
        )

        train_returns = base_returns.loc[train_mask]
        if len(train_returns) < min_train_days:
            diagnostics["windows"].append(
                {
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "scale": 1.0,
                    "reason": "insufficient_train_data",
                }
            )
            continue

        scale = _compute_scale_factor(
            train_returns,
            target_vol=target_vol,
            target_max_drawdown=target_max_dd,
            allow_leverage=allow_leverage,
            max_gross=max_gross,
            min_gross=min_gross,
        )

        if scale != 1.0:
            adjusted.loc[test_mask] = adjusted.loc[test_mask] * scale

        kill_date: Optional[pd.Timestamp] = None
        if kill_enabled and kill_roll_days > 0:
            kill_date = _apply_kill_switch(
                adjusted,
                aligned_returns,
                test_mask,
                kill_dd_threshold=kill_dd,
                rolling_days=kill_roll_days,
                rolling_min_return=kill_roll_min,
            )

        diagnostics["windows"].append(
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "scale": scale,
                "kill_date": str(kill_date.date()) if kill_date is not None else None,
            }
        )

    return adjusted, diagnostics


def build_wfo_test_mask(index: pd.Index, overlay_config: Mapping[str, Any]) -> pd.Series:
    """Build a boolean mask for WFO test windows based on overlay configuration.

    Args:
        index: Index to align the mask to (typically returns index).
        overlay_config: Overlay settings with train/test window sizes.

    Returns:
        Boolean Series where True indicates dates in WFO test windows.
    """
    if not overlay_config or overlay_config.get("metrics_window") != "wfo_test":
        return pd.Series(False, index=index)

    train_years = int(overlay_config.get("train_years", 5))
    test_years = int(overlay_config.get("test_years", 1))
    step_years = int(overlay_config.get("step_years", test_years))
    if step_years <= 0:
        step_years = test_years

    window_start_raw = overlay_config.get("window_start_date")
    window_end_raw = overlay_config.get("window_end_date")
    window_start = pd.Timestamp(window_start_raw) if window_start_raw else None
    window_end = pd.Timestamp(window_end_raw) if window_end_raw else None

    windows = _generate_wfo_windows(
        pd.DatetimeIndex(index),
        train_years,
        test_years,
        step_years=step_years,
        window_start=window_start,
        window_end=window_end,
    )

    idx = pd.DatetimeIndex(index)
    idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx

    mask_vals = pd.Series(False, index=index)
    for train_start, train_end, test_start, test_end in windows:
        start_ts = pd.Timestamp(test_start)
        end_ts = pd.Timestamp(test_end)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(None)
        mask_vals |= (idx_cmp >= start_ts) & (idx_cmp <= end_ts)

    return mask_vals


def _generate_wfo_windows(
    index: pd.Index,
    train_years: int,
    test_years: int,
    *,
    step_years: int,
    window_start: Optional[pd.Timestamp] = None,
    window_end: Optional[pd.Timestamp] = None,
) -> list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex(index).sort_values()
    start_date = dates.min()
    end_date = dates.max()

    def _align_to_index_tz(ts: pd.Timestamp, tz: Optional[tzinfo]) -> pd.Timestamp:
        if tz is None:
            return ts.tz_convert(None) if ts.tzinfo is not None else ts
        if ts.tzinfo is None:
            return ts.tz_localize(tz)
        return ts.tz_convert(tz)

    if window_start is not None:
        start_date = pd.Timestamp(window_start)
    if window_end is not None:
        end_date = pd.Timestamp(window_end)

    tz = cast(Optional[tzinfo], dates.tz)
    start_date = _align_to_index_tz(pd.Timestamp(start_date), tz)
    end_date = _align_to_index_tz(pd.Timestamp(end_date), tz)

    windows: list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    if start_date is pd.NaT or end_date is pd.NaT:
        return windows

    test_start = start_date + pd.DateOffset(years=train_years)
    while test_start <= end_date:
        train_start = test_start - pd.DateOffset(years=train_years)
        train_end = test_start - pd.offsets.BDay(1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.offsets.BDay(1)

        train_start_aligned = _align_date(dates, train_start, direction="next")
        train_end_aligned = _align_date(dates, train_end, direction="prev")
        test_start_aligned = _align_date(dates, test_start, direction="next")
        test_end_aligned = _align_date(dates, min(test_end, end_date), direction="prev")

        if (
            train_start_aligned is None
            or train_end_aligned is None
            or test_start_aligned is None
            or test_end_aligned is None
        ):
            break

        if train_start_aligned > train_end_aligned or test_start_aligned > test_end_aligned:
            break

        windows.append(
            (train_start_aligned, train_end_aligned, test_start_aligned, test_end_aligned)
        )
        test_start = test_start + pd.DateOffset(years=step_years)

    return windows


def _align_date(
    dates: pd.DatetimeIndex,
    target: pd.Timestamp,
    *,
    direction: str,
) -> Optional[pd.Timestamp]:
    if direction == "next":
        matches = dates[dates >= target]
        return matches[0] if len(matches) else None
    if direction == "prev":
        matches = dates[dates <= target]
        return matches[-1] if len(matches) else None
    raise ValueError("direction must be 'next' or 'prev'")


def _compute_portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    aligned_weights = weights.reindex(returns.index).fillna(0.0)
    weights_shifted = aligned_weights.shift(1).fillna(0.0)
    weighted = weights_shifted * returns
    daily = weighted.sum(axis=1, skipna=True)
    return daily.astype(float).fillna(0.0)


def _compute_scale_factor(
    returns: pd.Series,
    *,
    target_vol: float,
    target_max_drawdown: float,
    allow_leverage: bool,
    max_gross: float,
    min_gross: float,
) -> float:
    cleaned = returns.dropna()
    if cleaned.empty:
        return 1.0

    vol = _annualized_vol(cleaned)
    max_dd = abs(_max_drawdown(cleaned))

    scale_vol = target_vol / vol if vol > 0 else 1.0
    scale_dd = target_max_drawdown / max_dd if max_dd > 0 else 1.0
    scale = min(scale_vol, scale_dd)

    if not allow_leverage:
        scale = min(scale, 1.0)

    if not math.isfinite(scale):
        return 1.0

    return float(max(min(scale, max_gross), min_gross))


def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    equity: pd.Series = (1.0 + returns).cumprod()
    running_max: pd.Series = equity.cummax()
    drawdown: pd.Series = equity / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _apply_kill_switch(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    test_mask: pd.Series,
    *,
    kill_dd_threshold: float,
    rolling_days: int,
    rolling_min_return: float,
) -> Optional[pd.Timestamp]:
    window_dates = weights.index[test_mask]
    if len(window_dates) == 0:
        return None

    rolling: deque[float] = deque(maxlen=rolling_days)
    equity = 1.0
    peak = 1.0

    prev_idx = weights.index.get_indexer([window_dates[0]])[0] - 1
    prev_weights = (
        weights.iloc[prev_idx] if prev_idx >= 0 else pd.Series(0.0, index=weights.columns)
    )

    for idx, date in enumerate(window_dates):
        daily_ret = float((prev_weights * returns.loc[date]).sum())
        equity *= 1.0 + daily_ret
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1.0

        rolling.append(daily_ret)
        rolling_perf: Optional[float] = None
        if len(rolling) == rolling_days:
            rolling_perf = float(math.prod([1.0 + r for r in rolling]) - 1.0)

        if drawdown <= -abs(kill_dd_threshold) or (
            rolling_perf is not None and rolling_perf <= rolling_min_return
        ):
            if idx + 1 < len(window_dates):
                weights.loc[window_dates[idx + 1 :], :] = 0.0
            logger.info(
                "Kill-switch triggered on %s (drawdown=%.2f, rolling=%s).",
                date.date(),
                drawdown,
                f"{rolling_perf:.4f}" if rolling_perf is not None else "n/a",
            )
            return pd.Timestamp(date)

        prev_weights = weights.loc[date]

    return None
