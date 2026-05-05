from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _multiindex_level_index(columns: pd.MultiIndex, name: str) -> Optional[int]:
    want = name.lower()
    for i, col_name in enumerate(columns.names):
        if col_name is not None and str(col_name).lower() == want:
            return i
    return None


def _universe_close_matrix_multiindex(
    asset_data: pd.DataFrame, universe_tickers: tuple[str, ...]
) -> np.ndarray:
    mi = asset_data.columns
    if not isinstance(mi, pd.MultiIndex):
        raise TypeError("expected MultiIndex columns")
    ti = _multiindex_level_index(mi, "Ticker")
    if ti is None:
        ti = _multiindex_level_index(mi, "symbol")
    if ti is None:
        ti = 0
    fi = _multiindex_level_index(mi, "Field")
    if fi is None:
        fi = _multiindex_level_index(mi, "OHLC")
    if fi is None:
        fi = 1 if mi.nlevels > 1 else 0
    tick_level = mi.get_level_values(ti)
    field_level = mi.get_level_values(fi)
    parts: list[pd.Series] = []
    for t in universe_tickers:
        mask = tick_level == t
        if not np.any(mask):
            parts.append(pd.Series(np.nan, index=asset_data.index))
            continue
        indices = np.flatnonzero(np.asarray(mask))
        picked: Optional[int] = None
        for ix in indices:
            if str(field_level[int(ix)]).lower() == "close":
                picked = int(ix)
                break
        if picked is None and len(indices) == 1:
            picked = int(indices[0])
        if picked is None:
            parts.append(pd.Series(np.nan, index=asset_data.index))
            continue
        col_key = mi[picked]
        parts.append(asset_data[col_key])
    stacked = pd.concat(parts, axis=1)
    return np.ascontiguousarray(stacked.to_numpy(dtype=np.float64, copy=True))


def default_benchmark_ticker(
    benchmark_data: pd.DataFrame,
    universe_tickers: Optional[Sequence[str]] = None,
) -> str:
    """Best-effort ticker key from benchmark columns, else first universe ticker."""
    if benchmark_data.shape[1] > 0:
        c0 = benchmark_data.columns[0]
        if isinstance(benchmark_data.columns, pd.MultiIndex) and isinstance(c0, tuple):
            return str(c0[0])
        return str(c0)
    if universe_tickers:
        return str(universe_tickers[0])
    return ""


@dataclass(frozen=True)
class StrategyContext:
    """Inputs for deterministic full-scan target weight authoring.

    Bundle of price panels and scheduling metadata passed to ``generate_target_weights``.
    """

    asset_data: pd.DataFrame
    benchmark_data: pd.DataFrame
    non_universe_data: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex
    universe_tickers: tuple[str, ...]
    benchmark_ticker: str
    wfo_start_date: Optional[pd.Timestamp]
    wfo_end_date: Optional[pd.Timestamp]
    use_sparse_nan_for_inactive_rows: bool
    full_price_panel: Optional[pd.DataFrame] = None

    @property
    def universe_close_np(self) -> np.ndarray:
        """Close (or sole price) matrix aligned to ``universe_tickers`` columns, shape (T, N).

        Float64; missing cells are NaN. Prefer this over repeated per-date pandas slices in
        hot ``generate_target_weights`` loops when full scans already materialized ``asset_data``.
        """
        cols = list(self.universe_tickers)
        if isinstance(self.asset_data.columns, pd.MultiIndex):
            return _universe_close_matrix_multiindex(self.asset_data, self.universe_tickers)
        sub = self.asset_data.reindex(columns=cols)
        numeric = sub.apply(pd.to_numeric, errors="coerce")
        return np.ascontiguousarray(numeric.to_numpy(dtype=np.float64, copy=True))

    @property
    def rebalance_session_mask_np(self) -> np.ndarray:
        """Boolean length ``len(asset_data.index)``: True on rows whose session date is in
        ``rebalance_dates``.
        """
        ix = self.asset_data.index
        mask = np.asarray(ix.isin(self.rebalance_dates), dtype=np.bool_)
        return np.ascontiguousarray(mask)

    @classmethod
    def from_standard_inputs(
        cls,
        *,
        asset_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        non_universe_data: Optional[pd.DataFrame],
        rebalance_dates: pd.DatetimeIndex,
        universe_tickers: List[str],
        benchmark_ticker: str,
        wfo_start_date: Optional[pd.Timestamp],
        wfo_end_date: Optional[pd.Timestamp],
        use_sparse_nan_for_inactive_rows: bool,
        full_price_panel: Optional[pd.DataFrame] = None,
    ) -> StrategyContext:
        nu = (
            non_universe_data
            if non_universe_data is not None and len(non_universe_data.columns) > 0
            else pd.DataFrame()
        )
        return cls(
            asset_data=asset_data,
            benchmark_data=benchmark_data,
            non_universe_data=nu,
            rebalance_dates=pd.DatetimeIndex(rebalance_dates),
            universe_tickers=tuple(str(t) for t in universe_tickers),
            benchmark_ticker=str(benchmark_ticker),
            wfo_start_date=wfo_start_date,
            wfo_end_date=wfo_end_date,
            use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
            full_price_panel=full_price_panel,
        )


__all__ = ["StrategyContext", "default_benchmark_ticker"]
