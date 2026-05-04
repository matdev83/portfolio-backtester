from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


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
