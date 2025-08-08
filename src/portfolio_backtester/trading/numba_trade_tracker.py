"""
Numba-optimized trade tracking for high-performance backtesting.

This module provides vectorized implementations of trade tracking operations
that can process entire time series at once instead of day-by-day loops.

This is a backward compatibility wrapper that delegates to the new
VectorizedTradeTracker in the performance optimization module.
"""

import logging
import pandas as pd
from typing import Dict, Any

# Import the new implementation
from ..optimization.performance.vectorized_tracking import VectorizedTradeTracker

logger = logging.getLogger(__name__)


def track_trades_vectorized(
    weights_daily: pd.DataFrame,
    price_data_daily_ohlc: pd.DataFrame,
    transaction_costs: pd.Series,
    portfolio_value: float = 100000.0,
) -> Dict[str, Any]:
    """
    Vectorized trade tracking that processes entire time series at once.

    This function replaces the day-by-day loop in _track_trades with vectorized
    operations that are much faster for large datasets.

    Args:
        weights_daily: Daily portfolio weights
        price_data_daily_ohlc: Daily price data (OHLC format)
        transaction_costs: Daily transaction costs
        portfolio_value: Total portfolio value

    Returns:
        Dictionary with trade statistics
    """
    # Create the new vectorized trade tracker
    tracker = VectorizedTradeTracker(portfolio_value=portfolio_value)

    # Delegate to the new implementation
    return tracker.track_trades_optimized(weights_daily, price_data_daily_ohlc, transaction_costs)
