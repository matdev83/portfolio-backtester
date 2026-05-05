"""Prepared ndarray bundles for portfolio return simulation (internal / optimizer)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
import pandas as pd

from ..optimization.market_data_panel import MarketDataPanel
from ..timing.trade_execution_timing import TRADE_EXECUTION_TIMING_DEFAULT

logger = logging.getLogger(__name__)

EXECUTION_TIMING_BAR_CLOSE: Final[int] = 0
EXECUTION_TIMING_NEXT_BAR_OPEN: Final[int] = 1


@dataclass(frozen=True)
class PortfolioSimulationInput:
    """Aligned numeric inputs for the drifting-weights + cost simulation kernels."""

    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    weights_target: np.ndarray
    close_prices: np.ndarray
    close_price_mask: np.ndarray
    execution_prices: np.ndarray
    execution_price_mask: np.ndarray
    rebalance_mask: np.ndarray
    execution_timing: int
    ledger_decision_idx: np.ndarray


def market_panel_aligns_with_ohlc(
    panel: MarketDataPanel,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
) -> bool:
    """Return True if ``panel`` row index and ticker map cover ``price_index`` and ``valid_cols``."""

    if len(panel.daily_index_naive) != len(price_index):
        return False
    if not bool(panel.daily_index_naive.equals(price_index)):
        if not np.array_equal(
            panel.row_index_naive_datetime64(),
            np.asarray(price_index.values, dtype="datetime64[ns]"),
        ):
            return False
    for t in valid_cols:
        if t not in panel.ticker_to_column:
            return False
    return True


def _ohlc_field_level_axis1(mi: pd.MultiIndex) -> int:
    """Index of Field/OHLC-like level, else last level (bare OHLC tuples)."""

    names = mi.names or ()
    for i, nm in enumerate(names):
        if nm is not None and str(nm).lower() in ("field", "ohlc"):
            return int(i)
    return int(mi.nlevels - 1)


def extract_field_frame_from_ohlc(daily_ohlc: pd.DataFrame, field: str) -> pd.DataFrame:
    """Return a 2D frame for one OHLC field (e.g. Close, Open).

    For a flat column index, returns ``daily_ohlc`` unchanged. For ``MultiIndex`` columns,
    selects the level named Field/OHLC (case-insensitive) when present; otherwise uses the
    last level if it contains a value equal to ``field`` (case-insensitive).

    Raises:
        KeyError: If ``field`` is not present on the resolved level.
    """

    want = str(field).strip()
    cols = daily_ohlc.columns
    if not isinstance(cols, pd.MultiIndex):
        return daily_ohlc
    mi = cols
    li = _ohlc_field_level_axis1(mi)
    level_vals = mi.get_level_values(li)
    picked: str | None = None
    for v in level_vals.unique():
        if str(v).strip().lower() == want.lower():
            picked = str(v)
            break
    if picked is None:
        raise KeyError(f"OHLC field {field!r} not found on MultiIndex axis=1 level {li}")
    out = daily_ohlc.xs(picked, level=li, axis=1)
    if isinstance(out, pd.DataFrame):
        return out
    return pd.DataFrame(out)


def extract_open_frame_from_ohlc(daily_ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return an Open panel aligned with Close-style columns, or None if opens are unavailable."""

    if not isinstance(daily_ohlc.columns, pd.MultiIndex):
        return None
    try:
        open_maybe = extract_field_frame_from_ohlc(daily_ohlc, "Open")
    except KeyError:
        return None
    if isinstance(open_maybe, pd.DataFrame):
        return open_maybe
    return pd.DataFrame(open_maybe)


def build_close_and_mask_from_dataframe(
    close_prices_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy OHLC/close DataFrame path (float64, invalid prices zeroed)."""

    close_prices_use = close_prices_df.reindex(index=price_index, columns=valid_cols).astype(float)
    close_ok = close_prices_use.notna() & (close_prices_use > 0)
    close_arr = close_prices_use.fillna(0.0).to_numpy(copy=True)
    close_ok_arr = close_ok.to_numpy(copy=True)
    return close_arr, close_ok_arr


def build_close_and_mask_from_market_panel(
    panel: MarketDataPanel,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``MarketDataPanel`` closes/masks for ``valid_cols`` (float64)."""

    idxs = [panel.ticker_to_column[t] for t in valid_cols]
    raw = np.ascontiguousarray(panel.daily_close_np[:, idxs], dtype=np.float64)
    mask_arr = np.isfinite(raw) & (raw > 0.0)
    close_arr = np.where(mask_arr, raw, 0.0)
    return close_arr, mask_arr


def build_open_and_mask_from_dataframe(
    open_prices_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Open DataFrame path (float64); invalid opens zeroed."""

    op = open_prices_df.reindex(index=price_index, columns=valid_cols).astype(float)
    ok = op.notna() & (op > 0)
    arr = op.fillna(0.0).to_numpy(copy=True)
    return arr, ok.to_numpy(copy=True)


def build_open_and_mask_from_market_panel(
    panel: MarketDataPanel,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``MarketDataPanel`` opens."""

    assert panel.open_np is not None
    idxs = [panel.ticker_to_column[t] for t in valid_cols]
    raw = np.ascontiguousarray(panel.open_np[:, idxs], dtype=np.float64)
    mask_arr = np.isfinite(raw) & (raw > 0.0)
    open_arr = np.where(mask_arr, raw, 0.0)
    return open_arr, mask_arr


def sparse_execution_rebalance_event_mask(
    sparse_execution_targets: Optional[pd.DataFrame],
    calendar: pd.DatetimeIndex,
    valid_cols: list[str],
) -> np.ndarray:
    """Mark rebalance days where sparse execution rows carry explicit active targets.

    Each calendar-aligned sparse row (not all-NaN) sets ``rebalance_mask`` on that date,
    including consecutive dates that repeat the same target weights.
    """

    mask = np.zeros(len(calendar), dtype=np.bool_)
    if not valid_cols:
        return mask
    if sparse_execution_targets is None or sparse_execution_targets.empty:
        return mask

    active = sparse_execution_targets.loc[~sparse_execution_targets.isna().all(axis=1)]
    if active.empty:
        return mask

    cal_pos = {pd.Timestamp(ts): i for i, ts in enumerate(calendar)}

    for ts in sorted(active.index.unique()):
        row = active.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        ix = cal_pos.get(pd.Timestamp(ts))
        if ix is None:
            continue
        mask[ix] = True

    return mask


def propagate_rebalance_mask_for_invalid_next_bar_opens_with_decision_idx(
    rebalance_mask: np.ndarray,
    weights_target: np.ndarray,
    execution_price_mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend ``rebalance_mask`` for invalid next-bar opens and build per-asset ledger decision indices.

    ``ledger_decision_idx[i, j]`` is the audit **decision** session index for a fill on bar
    ``i`` for asset ``j`` under ``next_bar_open``: ``max(0, anchor[j] - 1)`` where ``anchor[j]``
    is the first bar in the current pending-retry chain where ``new_intent[j]`` was true. This
    matches ``execution_date_idx - 1`` on the first attempt but stays pinned to the original
    decision when invalid opens delay execution across multiple sessions.

    Invariant: this loop must stay aligned with :func:`propagate_rebalance_mask_for_invalid_next_bar_opens`
    (which delegates here) so mask propagation and audit anchors never drift.
    """

    rb = rebalance_mask.astype(np.bool_, copy=True)
    t = int(rb.shape[0])
    if weights_target.shape[0] != t or execution_price_mask.shape[0] != t:
        msg = "rebalance_mask, weights_target, and execution_price_mask must align on time"
        raise ValueError(msg)
    n_assets = int(weights_target.shape[1])
    carried = np.zeros(n_assets, dtype=np.bool_)
    anchor = np.full(n_assets, -1, dtype=np.int64)
    ledger_decision_idx = np.zeros((t, n_assets), dtype=np.int64)
    for i in range(t):
        cur = weights_target[i]
        if i > 0:
            prior = weights_target[i - 1]
        else:
            prior = np.zeros(n_assets, dtype=cur.dtype)
        if bool(carried.any()) and not bool(rb[i]):
            rb[i] = True
        effective = carried.copy()
        new_intent = np.zeros(n_assets, dtype=np.bool_)
        if bool(rb[i]):
            new_intent = (np.abs(cur) > eps) | (np.abs(cur - prior) > eps)
            effective |= new_intent
            for j in range(n_assets):
                if bool(new_intent[j]):
                    anchor[j] = np.int64(i)
        if bool(rb[i]):
            for j in range(n_assets):
                if bool(effective[j]):
                    aj = int(anchor[j])
                    if aj >= 0:
                        di = aj - 1
                        ledger_decision_idx[i, j] = np.int64(0) if di < 0 else np.int64(di)
                    else:
                        ledger_decision_idx[i, j] = np.int64(max(0, i - 1))
                else:
                    ledger_decision_idx[i, j] = np.int64(i)
        else:
            ledger_decision_idx[i, :] = np.int64(i)
        if not bool(effective.any()):
            continue
        blocked = effective & (~execution_price_mask[i])
        carried = blocked
        if bool(carried.any()) and i + 1 < t:
            rb[i + 1] = True
    return rb, ledger_decision_idx


def propagate_rebalance_mask_for_invalid_next_bar_opens(
    rebalance_mask: np.ndarray,
    weights_target: np.ndarray,
    execution_price_mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Carry ``rebalance_mask`` forward when ``next_bar_open`` cannot execute a pending leg.

    A leg is pending when the target weight is non-zero or changed from the prior bar
    (including explicit ``0.0`` exits). The canonical kernel leaves positions unchanged
    for assets without a valid execution price. If ``rebalance_mask`` is only true on the
    blocked session, no later bar retries the trade; this propagates rebalance intent to
    subsequent rows until opens exist or the calendar ends (weights remain dense/ffilled
    from upstream mapping).

    Blocked intents are tracked per asset and carried forward across chained invalid-open
    days so ``effective_pending`` survives after dense targets plateau (e.g. explicit
    exits that cannot execute remain pending until a valid execution row clears them).
    """

    rb, _ = propagate_rebalance_mask_for_invalid_next_bar_opens_with_decision_idx(
        rebalance_mask, weights_target, execution_price_mask, eps=eps
    )
    return rb


def ledger_decision_idx_bar_close(*, n_rows: int, n_assets: int) -> np.ndarray:
    """Per-bar decision index equals execution bar (``bar_close`` audit convention)."""

    col = np.arange(n_rows, dtype=np.int64).reshape(-1, 1)
    return np.broadcast_to(col, (n_rows, n_assets)).copy()


def ledger_decision_idx_next_bar_open_legacy(*, n_rows: int, n_assets: int) -> np.ndarray:
    """Kernel-equivalent ``execution_idx - 1`` grid without invalid-open propagation (tests only)."""

    out = np.zeros((n_rows, n_assets), dtype=np.int64)
    for i in range(n_rows):
        row_val = np.int64(0) if i == 0 else np.int64(i - 1)
        out[i, :] = row_val
    return out


def build_portfolio_simulation_input(
    *,
    weights_daily: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
    close_arr: np.ndarray,
    close_price_mask_arr: np.ndarray,
    open_arr: Optional[np.ndarray] = None,
    open_price_mask_arr: Optional[np.ndarray] = None,
    sparse_execution_targets: Optional[pd.DataFrame] = None,
    rebalance_mask_arr: Optional[np.ndarray] = None,
    trade_execution_timing: str = TRADE_EXECUTION_TIMING_DEFAULT,
) -> PortfolioSimulationInput:
    """Assemble aligned arrays for simulate_portfolio from dense weights plus mask provenance."""

    weights_target = (
        weights_daily.reindex(index=price_index, columns=valid_cols).fillna(0.0).to_numpy()
    )

    if rebalance_mask_arr is not None:
        rb = rebalance_mask_arr.astype(np.bool_, copy=False).reshape(-1)
    else:
        rb = sparse_execution_rebalance_event_mask(
            sparse_execution_targets, price_index, valid_cols
        )
    if len(rb) != len(price_index):
        raise ValueError("rebalance_mask length must equal len(price_index)")

    if trade_execution_timing == "next_bar_open":
        if open_arr is None or open_price_mask_arr is None:
            raise ValueError("next_bar_open requires open_arr and open_price_mask_arr")
        exec_prices = open_arr.astype(np.float64, copy=False)
        exec_mask = open_price_mask_arr.astype(np.bool_, copy=False)
        timing_const = EXECUTION_TIMING_NEXT_BAR_OPEN
        rb_before_prop = rb.copy()
        rb, ledger_decision_idx = (
            propagate_rebalance_mask_for_invalid_next_bar_opens_with_decision_idx(
                rb, weights_target, exec_mask
            )
        )
        added = rb & ~rb_before_prop
        if bool(np.any(added)):
            logger.warning(
                "Extended rebalance_mask on %d session(s): next_bar_open had invalid open(s) for "
                "one or more legs with pending target changes; retrying on subsequent bar(s).",
                int(np.sum(added)),
            )
    elif trade_execution_timing == "bar_close":
        exec_prices = close_arr.astype(np.float64, copy=False)
        exec_mask = close_price_mask_arr.astype(np.bool_, copy=False)
        timing_const = EXECUTION_TIMING_BAR_CLOSE
        t_rows, n_ast = weights_target.shape
        ledger_decision_idx = ledger_decision_idx_bar_close(n_rows=t_rows, n_assets=n_ast)
    else:
        raise ValueError(f"unsupported trade_execution_timing {trade_execution_timing!r}")

    ledger_decision_idx = np.ascontiguousarray(np.asarray(ledger_decision_idx, dtype=np.int64))

    return PortfolioSimulationInput(
        dates=price_index,
        tickers=tuple(valid_cols),
        weights_target=np.ascontiguousarray(weights_target.astype(np.float64, copy=False)),
        close_prices=np.ascontiguousarray(close_arr.astype(np.float64, copy=False)),
        close_price_mask=np.ascontiguousarray(close_price_mask_arr.astype(np.bool_, copy=False)),
        execution_prices=np.ascontiguousarray(exec_prices),
        execution_price_mask=np.ascontiguousarray(exec_mask),
        rebalance_mask=np.ascontiguousarray(rb),
        execution_timing=timing_const,
        ledger_decision_idx=ledger_decision_idx,
    )


def prepare_close_arrays_for_simulation(
    *,
    market_data_panel: Optional[MarketDataPanel],
    close_prices_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Pick close/mask arrays from ``MarketDataPanel`` when aligned, else from OHLC DataFrame."""

    if market_data_panel is not None and market_panel_aligns_with_ohlc(
        market_data_panel, price_index, valid_cols
    ):
        return build_close_and_mask_from_market_panel(market_data_panel, valid_cols)
    return build_close_and_mask_from_dataframe(close_prices_df, price_index, valid_cols)


def prepare_open_arrays_for_simulation(
    *,
    market_data_panel: Optional[MarketDataPanel],
    open_prices_df: Optional[pd.DataFrame],
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Pick aligned open arrays from the dense panel only when opens exist; otherwise use OHLC dataframe."""

    if market_data_panel is not None and market_panel_aligns_with_ohlc(
        market_data_panel, price_index, valid_cols
    ):
        if market_data_panel.open_np is not None:
            return build_open_and_mask_from_market_panel(market_data_panel, valid_cols)
    if open_prices_df is None:
        raise ValueError("Open panel required for next_bar_open when market panel omits opens")
    return build_open_and_mask_from_dataframe(open_prices_df, price_index, valid_cols)
