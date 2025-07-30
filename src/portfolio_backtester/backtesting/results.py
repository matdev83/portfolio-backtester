"""
Structured result data classes for backtesting and optimization.

This module defines all result data structures used throughout the backtesting
and optimization process, ensuring type safety and clear data flow between components.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Tuple, Optional
import pandas as pd


@dataclass
class BacktestResult:
    """Complete backtest results with all performance data.
    
    This class contains all the data generated from a complete backtest run,
    including returns, metrics, trade history, and chart data.
    
    Attributes:
        returns: Time series of portfolio returns
        metrics: Dictionary of calculated performance metrics
        trade_history: DataFrame containing all trade records
        performance_stats: Additional performance statistics
        charts_data: Data for generating performance charts
        trade_stats: Dictionary of detailed trade statistics split by direction
    """
    returns: pd.Series
    metrics: Dict[str, float]
    trade_history: pd.DataFrame
    performance_stats: Dict[str, Any]
    charts_data: Dict[str, Any]
    trade_stats: Optional[Dict[str, Any]] = None


@dataclass 
class WindowResult:
    """Single walk-forward window evaluation result.
    
    Contains the results from evaluating a parameter set on a single
    walk-forward window, including the time boundaries and metrics.
    
    Attributes:
        window_returns: Portfolio returns for this window
        metrics: Performance metrics calculated for this window
        train_start: Start date of training period
        train_end: End date of training period  
        test_start: Start date of test period
        test_end: End date of test period
    """
    window_returns: pd.Series
    metrics: Dict[str, float]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


