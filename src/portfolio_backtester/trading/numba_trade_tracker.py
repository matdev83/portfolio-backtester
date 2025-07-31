"""
Numba-optimized trade tracking for high-performance backtesting.

This module provides vectorized implementations of trade tracking operations
that can process entire time series at once instead of day-by-day loops.

This is a backward compatibility wrapper that delegates to the new 
VectorizedTradeTracker in the performance optimization module.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Import the new implementation
from ..optimization.performance.vectorized_tracking import VectorizedTradeTracker

logger = logging.getLogger(__name__)

# Check if Numba is available
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - falling back to pure Python implementations")


def track_trades_vectorized(
    weights_daily: pd.DataFrame,
    price_data_daily_ohlc: pd.DataFrame,
    transaction_costs: pd.Series,
    portfolio_value: float = 100000.0
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


def _get_empty_trade_stats() -> Dict[str, Any]:
    """Return empty trade statistics when no trades are available."""
    return {
        'total_trades': 0,
        'total_turnover': 0.0,
        'total_transaction_costs': 0.0,
        'max_margin_load': 0.0,
        'mean_margin_load': 0.0,
        'avg_trade_cost': 0.0,
        'all_num_trades': 0,
        'all_win_rate': 0.0,
        'all_avg_win': 0.0,
        'all_avg_loss': 0.0,
        'all_profit_factor': 0.0,
        'all_max_consecutive_wins': 0,
        'all_max_consecutive_losses': 0,
        'all_avg_trade_duration': 0.0,
        'all_total_pnl': 0.0,
        'all_total_pnl_net': 0.0,
        'all_avg_mfe': 0.0,
        'all_avg_mae': 0.0,
        'long_num_trades': 0,
        'long_win_rate': 0.0,
        'long_avg_win': 0.0,
        'long_avg_loss': 0.0,
        'long_profit_factor': 0.0,
        'long_max_consecutive_wins': 0,
        'long_max_consecutive_losses': 0,
        'long_avg_trade_duration': 0.0,
        'long_total_pnl': 0.0,
        'long_total_pnl_net': 0.0,
        'long_avg_mfe': 0.0,
        'long_avg_mae': 0.0,
        'short_num_trades': 0,
        'short_win_rate': 0.0,
        'short_avg_win': 0.0,
        'short_avg_loss': 0.0,
        'short_profit_factor': 0.0,
        'short_max_consecutive_wins': 0,
        'short_max_consecutive_losses': 0,
        'short_avg_trade_duration': 0.0,
        'short_total_pnl': 0.0,
        'short_total_pnl_net': 0.0,
        'short_avg_mfe': 0.0,
        'short_avg_mae': 0.0,
    }