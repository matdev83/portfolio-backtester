"""
Numba-oriented trade-tracking helper utilities.

Delegates to :class:`~portfolio_backtester.trading.trade_tracker.TradeTracker`
and shared portfolio kernels for deterministic trade statistics aligned with core
portfolio simulation outputs.
"""

import logging
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..numba_kernels import trade_lifecycle_kernel, trade_tracking_kernel
from .trade_tracker import TradeTracker

logger = logging.getLogger(__name__)


def _allocation_mode_to_int(mode: str) -> int:
    if mode in ("reinvestment", "compound"):
        return 0
    return 1


def track_trades_canonical(
    weights_daily: pd.DataFrame,
    price_data_daily_ohlc: pd.DataFrame,
    transaction_costs: pd.Series,
    portfolio_value: float = 100000.0,
    allocation_mode: str = "reinvestment",
) -> Dict[str, Any]:
    """
    Trade statistics helper using shared Numba kernels on dense daily inputs.

    Args:
        weights_daily: Daily portfolio weights
        price_data_daily_ohlc: Daily price data (OHLC format)
        transaction_costs: Daily transaction costs
        portfolio_value: Total portfolio value
        allocation_mode: ``reinvestment`` / ``compound`` (kernel mode 0) or fixed-capital style (mode 1)

    Returns:
        Dictionary with comprehensive trade statistics
    """
    tracker = TradeTracker(initial_portfolio_value=portfolio_value, allocation_mode=allocation_mode)

    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        extracted_close = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
    else:
        extracted_close = price_data_daily_ohlc

    if isinstance(extracted_close, pd.Series):
        close_prices_df = extracted_close.to_frame()
    else:
        close_prices_df = extracted_close

    valid_cols = list(weights_daily.columns.intersection(close_prices_df.columns))
    common_index = weights_daily.index.intersection(close_prices_df.index)

    if len(common_index) == 0 or not valid_cols:
        logger.warning("No overlapping dates/columns between weights and price data.")
        return tracker.get_trade_statistics()

    aligned_weights = weights_daily.reindex(common_index, columns=valid_cols).fillna(0.0)
    aligned_prices = close_prices_df.reindex(common_index, columns=valid_cols).ffill()
    aligned_costs = transaction_costs.reindex(common_index).fillna(0.0)

    weights_arr = aligned_weights.astype(float).to_numpy()
    prices_arr = aligned_prices.astype(float).to_numpy()

    tc = aligned_costs.astype(float).to_numpy(dtype=np.float64)
    commissions_arr = np.broadcast_to(tc[:, np.newaxis], weights_arr.shape).astype(np.float64)

    price_mask = prices_arr > 0
    portfolio_values, _, positions = trade_tracking_kernel(
        initial_portfolio_value=portfolio_value,
        allocation_mode=_allocation_mode_to_int(allocation_mode),
        weights=weights_arr,
        prices=prices_arr,
        price_mask=price_mask,
        commissions=commissions_arr,
    )

    completed_trades_buffer = np.zeros(
        weights_arr.shape[0] * weights_arr.shape[1],
        dtype=[
            ("ticker_idx", "i8"),
            ("entry_date", "i8"),
            ("exit_date", "i8"),
            ("entry_price", "f8"),
            ("exit_price", "f8"),
            ("quantity", "f8"),
            ("pnl", "f8"),
            ("commission", "f8"),
        ],
    )
    open_positions_buffer = np.zeros(
        weights_arr.shape[1],
        dtype=[
            ("is_open", "?"),
            ("entry_date", "i8"),
            ("entry_price", "f8"),
            ("quantity", "f8"),
            ("total_commission", "f8"),
        ],
    )
    commissions_dollars = commissions_arr * float(portfolio_value)
    dates_i8 = common_index.to_numpy(dtype="datetime64[ns]").view(np.int64)
    trade_count = trade_lifecycle_kernel(
        positions=positions,
        prices=prices_arr,
        dates=dates_i8,
        commissions=commissions_dollars,
        initial_capital=portfolio_value,
        out_trades=completed_trades_buffer,
        out_open_pos=open_positions_buffer,
    )
    completed_trades_arr = completed_trades_buffer[:trade_count]

    pv_series = pd.Series(portfolio_values, index=common_index, dtype=float)
    positions_df = pd.DataFrame(positions, index=common_index, columns=valid_cols)
    prices_df = pd.DataFrame(prices_arr, index=common_index, columns=valid_cols)
    tracker.populate_from_kernel_results(
        portfolio_values=pv_series,
        positions=positions_df,
        completed_trades=completed_trades_arr,
        tickers=np.asarray(valid_cols, dtype=object),
        prices=prices_df,
    )

    return tracker.get_trade_statistics()


def track_trades_vectorized(
    weights_daily: pd.DataFrame,
    price_data_daily_ohlc: pd.DataFrame,
    transaction_costs: pd.Series,
    portfolio_value: float = 100000.0,
    allocation_mode: str = "reinvestment",
) -> Dict[str, Any]:
    warnings.warn(
        "track_trades_vectorized is deprecated; use track_trades_canonical.",
        DeprecationWarning,
        stacklevel=2,
    )
    return track_trades_canonical(
        weights_daily,
        price_data_daily_ohlc,
        transaction_costs,
        portfolio_value=portfolio_value,
        allocation_mode=allocation_mode,
    )
