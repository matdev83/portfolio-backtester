"""
Window Evaluation Engine for Daily Strategy Evaluation.

This module provides the WindowEvaluator class that evaluates strategies
across all evaluation dates within a single WFO window, supporting both
daily and traditional monthly evaluation modes.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from .position_tracker import PositionTracker, Trade
from .results import WindowResult
from ..optimization.wfo_window import WFOWindow

logger = logging.getLogger(__name__)


class WindowEvaluator:
    """Evaluates strategy performance across a single WFO window with daily evaluation support.
    
    This class handles the evaluation of a strategy across all evaluation dates
    within a single walk-forward window. It supports both daily evaluation for
    intramonth strategies and traditional monthly evaluation for backward compatibility.
    
    Attributes:
        data_cache: Optional cache for historical data to improve performance
    """
    
    def __init__(self, data_cache: Optional[Dict] = None):
        """Initialize the window evaluator.
        
        Args:
            data_cache: Optional dictionary for caching historical data
        """
        self.data_cache = data_cache or {}
        logger.debug("WindowEvaluator initialized")
    
    def evaluate_window(self, 
                       window: WFOWindow,
                       strategy,
                       daily_data: pd.DataFrame,
                       benchmark_data: pd.DataFrame,
                       universe_tickers: list,
                       benchmark_ticker: str) -> WindowResult:
        """Evaluate strategy across all evaluation dates in the window.
        
        This method evaluates a strategy across all required evaluation dates
        within the window, tracking positions and generating trade records.
        
        Args:
            window: WFOWindow object defining the evaluation window
            strategy: Strategy instance to evaluate
            daily_data: Daily price data
            benchmark_data: Benchmark price data
            universe_tickers: List of tickers in the trading universe
            benchmark_ticker: Benchmark ticker symbol
            
        Returns:
            WindowResult containing evaluation results
        """
        logger.debug(
            f"Evaluating window {window.test_start.date()} to {window.test_end.date()} "
            f"with frequency {window.evaluation_frequency}"
        )
        
        # Get evaluation dates for this window
        available_dates = daily_data.index
        eval_dates = window.get_evaluation_dates(available_dates)
        
        if len(eval_dates) == 0:
            logger.warning(
                f"No evaluation dates found for window {window.test_start.date()} to {window.test_end.date()}"
            )
            return self._create_empty_result(window)
        
        # Initialize position tracking
        position_tracker = PositionTracker()
        daily_returns = []
        
        # Evaluate strategy on each date
        for i, current_date in enumerate(eval_dates):
            try:
                # Get historical data up to current date
                historical_data = self._get_historical_data(daily_data, current_date, window.train_start)
                benchmark_historical = self._get_historical_data(benchmark_data, current_date, window.train_start)
                
                # Generate signals
                signals = strategy.generate_signals(
                    all_historical_data=historical_data,
                    benchmark_historical_data=benchmark_historical,
                    non_universe_historical_data=pd.DataFrame(),  # Empty for now
                    current_date=current_date,
                    start_date=window.train_start,
                    end_date=window.test_end
                )
                
                # Update positions
                current_prices = self._get_current_prices(daily_data, current_date)
                current_weights = position_tracker.update_positions(signals, current_date, current_prices)
                
                # Calculate daily return
                if i > 0:  # Skip first day (no previous positions)
                    daily_return = self._calculate_daily_return(
                        current_weights, 
                        daily_data, 
                        current_date, 
                        eval_dates[i-1],
                        universe_tickers
                    )
                    daily_returns.append(daily_return)
                
            except Exception as e:
                logger.error(f"Error evaluating strategy on {current_date.date()}: {e}")
                daily_returns.append(0.0)
        
        # Create result
        return WindowResult(
            window_returns=pd.Series(daily_returns, index=eval_dates[1:]) if daily_returns else pd.Series(dtype=float),
            metrics=self._calculate_window_metrics(daily_returns),
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            trades=position_tracker.get_completed_trades(),
            final_weights=position_tracker.get_current_weights()
        )
    
    def _get_historical_data(self, data: pd.DataFrame, current_date: pd.Timestamp, 
                           train_start: pd.Timestamp) -> pd.DataFrame:
        """Get historical data up to current date, starting from train_start.
        
        Args:
            data: Full dataset
            current_date: Current evaluation date
            train_start: Start of training period
            
        Returns:
            Historical data up to current date
        """
        cache_key = f"{train_start}_{current_date}"
        
        if cache_key not in self.data_cache:
            # Filter data from train_start to current_date
            mask = (data.index >= train_start) & (data.index <= current_date)
            self.data_cache[cache_key] = data.loc[mask].copy()
        
        return self.data_cache[cache_key]
    
    def _get_current_prices(self, price_data: pd.DataFrame, current_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Get current prices for the given date.
        
        Args:
            price_data: Price DataFrame
            current_date: Date to get prices for
            
        Returns:
            DataFrame with current prices or None if not available
        """
        if current_date not in price_data.index:
            return None
        
        return price_data.loc[[current_date]]
    
    def _calculate_daily_return(self, 
                               weights: pd.Series,
                               price_data: pd.DataFrame,
                               current_date: pd.Timestamp,
                               previous_date: pd.Timestamp,
                               universe_tickers: list) -> float:
        """Calculate daily portfolio return based on weights and price changes.
        
        Args:
            weights: Current position weights
            price_data: Price data
            current_date: Current date
            previous_date: Previous evaluation date
            universe_tickers: List of universe tickers
            
        Returns:
            Daily portfolio return
        """
        try:
            # Get price changes
            if isinstance(price_data.columns, pd.MultiIndex):
                # Multi-level columns (Ticker, Field)
                current_prices = {}
                previous_prices = {}
                
                for ticker in universe_tickers:
                    if (ticker, 'Close') in price_data.columns:
                        if current_date in price_data.index:
                            current_prices[ticker] = price_data.loc[current_date, (ticker, 'Close')]
                        if previous_date in price_data.index:
                            previous_prices[ticker] = price_data.loc[previous_date, (ticker, 'Close')]
            else:
                # Single-level columns
                current_prices = price_data.loc[current_date, universe_tickers].to_dict() if current_date in price_data.index else {}
                previous_prices = price_data.loc[previous_date, universe_tickers].to_dict() if previous_date in price_data.index else {}
            
            # Calculate returns for each asset
            daily_return = 0.0
            for ticker in universe_tickers:
                if ticker in weights and ticker in current_prices and ticker in previous_prices:
                    if previous_prices[ticker] != 0 and not pd.isna(previous_prices[ticker]) and not pd.isna(current_prices[ticker]):
                        asset_return = (current_prices[ticker] - previous_prices[ticker]) / previous_prices[ticker]
                        daily_return += weights[ticker] * asset_return
            
            return daily_return
            
        except Exception as e:
            logger.error(f"Error calculating daily return for {current_date.date()}: {e}")
            return 0.0
    
    def _calculate_window_metrics(self, daily_returns: list) -> Dict[str, float]:
        """Calculate basic metrics for the window.
        
        Args:
            daily_returns: List of daily returns
            
        Returns:
            Dictionary of calculated metrics
        """
        if not daily_returns:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'volatility': 0.0}
        
        returns_series = pd.Series(daily_returns)
        
        # Basic metrics
        total_return = (1 + returns_series).prod() - 1
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (returns_series.mean() * 252) / volatility if volatility > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'num_periods': len(daily_returns)
        }
    
    def _create_empty_result(self, window: WFOWindow) -> WindowResult:
        """Create empty result for windows with no evaluation dates.
        
        Args:
            window: WFO window
            
        Returns:
            Empty WindowResult
        """
        return WindowResult(
            window_returns=pd.Series(dtype=float),
            metrics={'total_return': 0.0, 'sharpe_ratio': 0.0, 'volatility': 0.0},
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            trades=[],
            final_weights={}
        )