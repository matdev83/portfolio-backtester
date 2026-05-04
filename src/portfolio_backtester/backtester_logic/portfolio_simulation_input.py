"""Prepared ndarray bundles for portfolio return simulation (internal / optimizer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..optimization.market_data_panel import MarketDataPanel


@dataclass(frozen=True)
class PortfolioSimulationInput:
    """Aligned numeric inputs for the drifting-weights + cost simulation kernels."""

    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    close_prices: np.ndarray
    price_mask: np.ndarray
    weights_target: np.ndarray


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


def build_close_and_mask_from_dataframe(
    close_prices_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy OHLC/close DataFrame path (float64, invalid prices zeroed)."""

    close_prices_use = close_prices_df.reindex(index=price_index, columns=valid_cols).astype(float)
    price_mask = close_prices_use.notna() & (close_prices_use > 0)
    close_arr = close_prices_use.fillna(0.0).to_numpy(copy=True)
    price_mask_arr = price_mask.to_numpy(copy=True)
    return close_arr, price_mask_arr


def build_close_and_mask_from_market_panel(
    panel: MarketDataPanel,
    valid_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``MarketDataPanel`` closes/masks for ``valid_cols`` (float64)."""

    idxs = [panel.ticker_to_column[t] for t in valid_cols]
    raw = np.ascontiguousarray(panel.daily_close_np[:, idxs], dtype=np.float64)
    price_mask_arr = np.isfinite(raw) & (raw > 0.0)
    close_arr = np.where(price_mask_arr, raw, 0.0)
    return close_arr, price_mask_arr


def build_portfolio_simulation_input(
    *,
    weights_daily: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    valid_cols: list[str],
    close_arr: np.ndarray,
    price_mask_arr: np.ndarray,
) -> PortfolioSimulationInput:
    """Assemble a :class:`PortfolioSimulationInput` from prepared price and weight arrays."""

    weights_target = (
        weights_daily.reindex(index=price_index, columns=valid_cols).fillna(0.0).to_numpy()
    )
    return PortfolioSimulationInput(
        dates=price_index,
        tickers=tuple(valid_cols),
        close_prices=close_arr,
        price_mask=price_mask_arr,
        weights_target=weights_target,
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
