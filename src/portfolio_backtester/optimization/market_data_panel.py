"""Typed immutable view of aligned daily OHLC and returns numeric panels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def drop_tz_like_evaluator(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize a timestamp to naive wall time matching ``evaluate_window``."""
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


def evaluation_naive_datetimes_like_evaluator(values: pd.Index) -> pd.DatetimeIndex:
    """Produce a naive ``DatetimeIndex`` consistent with evaluator slicing."""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _dataframe_with_evaluation_index(df: pd.DataFrame) -> pd.DataFrame:
    new_ix = evaluation_naive_datetimes_like_evaluator(df.index)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.equals(new_ix):
        return df
    out = df.copy()
    out.index = new_ix
    return out


def _close_frame_from_daily(daily_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Extract the closes panel with column order preserved (or treat frame as closes)."""

    if isinstance(daily_ohlc.columns, pd.MultiIndex):
        names_tuple = tuple(daily_ohlc.columns.names or ())
        if "Field" not in names_tuple:
            raise ValueError("MultiIndex OHLC frames must label the Field level.")
        fields = set(daily_ohlc.columns.get_level_values("Field").unique())
        if "Close" not in fields:
            raise ValueError("MultiIndex daily OHLC must include Close under the Field level.")
        close_maybe = daily_ohlc.xs("Close", level="Field", axis=1)
        if isinstance(close_maybe, pd.DataFrame):
            return close_maybe
        return pd.DataFrame(close_maybe)
    return daily_ohlc


def _field_frame_or_none(daily_work: pd.DataFrame, field: str) -> Optional[pd.DataFrame]:
    cols = daily_work.columns
    if not isinstance(cols, pd.MultiIndex):
        return None
    names = tuple(cols.names or ())
    if "Field" not in names:
        return None
    fields = cols.get_level_values("Field").unique().tolist()
    if field not in fields:
        return None
    sub = daily_work.xs(field, level="Field", axis=1)
    if isinstance(sub, pd.DataFrame):
        return sub
    return pd.DataFrame(sub)


@dataclass(frozen=True)
class MarketDataPanel:
    """Dense float32-ready arrays aligned to a naive daily DatetimeIndex and ticker order."""

    daily_index_naive: pd.DatetimeIndex
    tickers: tuple[str, ...]
    ticker_to_column: Mapping[str, int]
    daily_close_np: np.ndarray
    returns_np: np.ndarray
    open_np: Optional[np.ndarray] = None
    high_np: Optional[np.ndarray] = None
    low_np: Optional[np.ndarray] = None

    @property
    def daily_np(self) -> np.ndarray:
        """Alias for close prices array (backward compat with orchestrator naming)."""

        return self.daily_close_np

    def row_index_naive_datetime64(self) -> np.ndarray:
        """``datetime64[ns]`` vector aligned to ``daily_index_naive``."""

        return np.asarray(self.daily_index_naive.values, dtype="datetime64[ns]")

    def to_close_dataframe(self) -> pd.DataFrame:
        """Reconstruct closes as a ``DataFrame`` (tests / adapters)."""

        return pd.DataFrame(
            np.asarray(self.daily_close_np, dtype=float),
            index=self.daily_index_naive.copy(),
            columns=list(self.tickers),
        )

    def to_returns_dataframe(self) -> pd.DataFrame:
        """Reconstruct aligned returns filled as used for trials."""

        return pd.DataFrame(
            np.asarray(self.returns_np, dtype=float),
            index=self.daily_index_naive.copy(),
            columns=list(self.tickers),
        )

    @classmethod
    def from_daily_ohlc_and_returns(
        cls, daily: pd.DataFrame, returns: pd.DataFrame
    ) -> MarketDataPanel:
        """Mirror orchestrator/array prep: OHLC-aware closes, tz-naive index, float32 contiguous."""

        daily_work = _dataframe_with_evaluation_index(daily)
        raw_close = _close_frame_from_daily(daily_work)
        daily_close_df = _dataframe_with_evaluation_index(raw_close)
        tickers_list = [str(x) for x in daily_close_df.columns]
        tickers = tuple(tickers_list)
        ticker_to_column = {t: i for i, t in enumerate(tickers)}

        daily_close_np = np.ascontiguousarray(daily_close_df.to_numpy(dtype=np.float32))

        rets_full_df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
        rets_naive_ix = evaluation_naive_datetimes_like_evaluator(rets_full_df.index)
        rets_work = rets_full_df.copy()
        rets_work.index = rets_naive_ix

        returns_df = (
            rets_work.reindex(daily_close_df.index).reindex(columns=tickers_list).fillna(0.0)
        )
        returns_np = np.ascontiguousarray(returns_df.to_numpy(dtype=np.float32))

        ix = pd.DatetimeIndex(daily_close_df.index)

        def _maybe_field(fr: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
            if fr is None:
                return None
            fr_aligned = fr.reindex(index=daily_close_df.index).reindex(columns=tickers_list)
            return np.ascontiguousarray(fr_aligned.to_numpy(dtype=np.float32))

        open_np = _maybe_field(_field_frame_or_none(daily_work, "Open"))
        high_np = _maybe_field(_field_frame_or_none(daily_work, "High"))
        low_np = _maybe_field(_field_frame_or_none(daily_work, "Low"))

        return cls(
            daily_index_naive=ix,
            tickers=tickers,
            ticker_to_column=ticker_to_column,
            daily_close_np=daily_close_np,
            returns_np=returns_np,
            open_np=open_np,
            high_np=high_np,
            low_np=low_np,
        )
