"""
Window Evaluation Engine for Daily Strategy Evaluation.

This module provides the WindowEvaluator class that evaluates strategies
across all evaluation dates within a single WFO window, supporting both
daily and traditional monthly evaluation modes.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
import logging
from .position_tracker import PositionTracker
from .results import WindowResult
from ..optimization.wfo_window import WFOWindow
from ..interfaces.daily_risk_monitor_interface import (
    IDailyRiskMonitorFactory,
    DefaultDailyRiskMonitorFactory,
)

# Re-export monitors so tests can patch them via this module path
from ..risk_management.daily_stop_loss_monitor import DailyStopLossMonitor  # noqa: F401
from ..risk_management.daily_take_profit_monitor import DailyTakeProfitMonitor  # noqa: F401

logger = logging.getLogger(__name__)


class WindowEvaluator:
    """Evaluates strategy performance across a single WFO window with daily evaluation support.

    This class handles the evaluation of a strategy across all evaluation dates
    within a single walk-forward window. It supports both daily evaluation for
    intramonth strategies and traditional monthly evaluation for backward compatibility.

    Attributes:
        data_cache: Optional cache for historical data to improve performance
    """

    def __init__(
        self,
        data_cache: Optional[Dict] = None,
        risk_monitor_factory: Optional[IDailyRiskMonitorFactory] = None,
    ):
        """Initialize the window evaluator.

        Args:
            data_cache: Optional dictionary for caching historical data
            risk_monitor_factory: Factory for creating daily risk monitors (dependency injection)
        """
        self.data_cache = data_cache or {}
        self.risk_monitor_factory = risk_monitor_factory or DefaultDailyRiskMonitorFactory()
        logger.debug("WindowEvaluator initialized")

    def evaluate_window(
        self,
        window: WFOWindow,
        strategy,
        daily_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        universe_tickers: list,
        benchmark_ticker: str,
    ) -> WindowResult:
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
        available_dates = pd.DatetimeIndex(daily_data.index)
        eval_dates = window.get_evaluation_dates(available_dates)

        if len(eval_dates) == 0:
            logger.warning(
                f"No evaluation dates found for window {window.test_start.date()} to {window.test_end.date()}"
            )
            return self._create_empty_result(window)

        # Initialize position tracking and daily risk management monitoring
        position_tracker = PositionTracker()
        daily_stop_loss_monitor = DailyStopLossMonitor()
        daily_take_profit_monitor = DailyTakeProfitMonitor()
        daily_returns = []

        # Evaluate strategy on each date
        for i, current_date in enumerate(eval_dates):
            try:
                # Get historical data up to current date
                historical_data = self._get_historical_data(
                    daily_data, current_date, window.train_start
                )
                benchmark_historical = self._get_historical_data(
                    benchmark_data, current_date, window.train_start
                )

                # Generate signals
                signals = strategy.generate_signals(
                    all_historical_data=historical_data,
                    benchmark_historical_data=benchmark_historical,
                    non_universe_historical_data=pd.DataFrame(),  # Empty for now
                    current_date=current_date,
                    start_date=window.train_start,
                    end_date=window.test_end,
                )

                # Update positions from strategy signals
                current_prices = self._get_current_prices(daily_data, current_date)
                current_weights = position_tracker.update_positions(
                    signals, current_date, current_prices
                )

                # Get current prices as Series for risk management monitoring
                current_prices_series = self._get_current_prices_as_series(daily_data, current_date)

                # 🚨 CRITICAL: Daily stop loss check - independent of strategy rebalancing schedule
                # This runs every day regardless of when the strategy last rebalanced
                try:
                    stop_loss_signals = daily_stop_loss_monitor.check_positions_for_stop_loss(
                        current_date=current_date,
                        position_tracker=position_tracker,
                        current_prices=current_prices_series,
                        stop_loss_handler=strategy.get_stop_loss_handler(),
                        historical_data=historical_data,
                    )

                    # Apply stop loss liquidations if any positions triggered
                    if not stop_loss_signals.empty:
                        current_weights = position_tracker.update_positions(
                            stop_loss_signals, current_date, current_prices
                        )
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(f"Applied stop loss liquidations on {current_date.date()}")

                except Exception as e:
                    logger.error(f"Error in daily stop loss check on {current_date.date()}: {e}")
                    # Continue with evaluation even if stop loss check fails

                # 🚨 CRITICAL: Daily take profit check - independent of strategy rebalancing schedule
                # This runs every day regardless of when the strategy last rebalanced
                try:
                    take_profit_signals = daily_take_profit_monitor.check_positions_for_take_profit(
                        current_date=current_date,
                        position_tracker=position_tracker,
                        current_prices=current_prices_series,
                        take_profit_handler=strategy.get_take_profit_handler(),
                        historical_data=historical_data,
                    )

                    # Apply take profit liquidations if any positions triggered
                    if not take_profit_signals.empty:
                        current_weights = position_tracker.update_positions(
                            take_profit_signals, current_date, current_prices
                        )
                        if logger.isEnabledFor(logging.INFO):
                            logger.info(f"Applied take profit liquidations on {current_date.date()}")

                except Exception as e:
                    logger.error(f"Error in daily take profit check on {current_date.date()}: {e}")
                    # Continue with evaluation even if take profit check fails

                # Calculate daily return
                if i > 0:  # Skip first day (no previous positions)
                    daily_return = self._calculate_daily_return(
                        current_weights,
                        daily_data,
                        current_date,
                        eval_dates[i - 1],
                        universe_tickers,
                    )
                    daily_returns.append(daily_return)

            except Exception as e:
                logger.error(f"Error evaluating strategy on {current_date.date()}: {e}")
                daily_returns.append(0.0)

        # Create result
        return WindowResult(
            window_returns=(
                pd.Series(daily_returns, index=eval_dates[1:])
                if daily_returns
                else pd.Series(dtype=float)
            ),
            metrics=self._calculate_window_metrics(daily_returns),
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            trades=position_tracker.get_completed_trades(),
            final_weights=position_tracker.get_current_weights(),
        )

    def _get_historical_data(
        self, data: pd.DataFrame, current_date: pd.Timestamp, train_start: pd.Timestamp
    ) -> pd.DataFrame:
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

        result = self.data_cache[cache_key]
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    def _get_current_prices(
        self, price_data: pd.DataFrame, current_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
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

    def _get_current_prices_as_series(
        self, price_data: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        """Get current prices for the given date as a Series.

        Args:
            price_data: Price DataFrame
            current_date: Date to get prices for

        Returns:
            Series with current prices (empty if not available)
        """
        if current_date not in price_data.index:
            return pd.Series(dtype=float)

        current_prices = price_data.loc[current_date]

        # If price_data has MultiIndex columns, extract Close prices
        if isinstance(price_data.columns, pd.MultiIndex) and "Field" in price_data.columns.names:
            try:
                # Extract Close prices across all tickers
                close_prices = price_data.xs("Close", level="Field", axis=1)
                close_prices_at_date = close_prices.loc[current_date]
                # Ensure we return a Series
                if isinstance(close_prices_at_date, pd.Series):
                    return close_prices_at_date
                else:
                    return pd.Series(close_prices_at_date, index=close_prices.columns)
            except KeyError:
                # No Close prices available, return empty Series
                return pd.Series(dtype=float)

        # For simple column structure, return the prices directly
        if isinstance(current_prices, pd.Series):
            return current_prices
        else:
            # Convert to Series if needed
            return pd.Series(current_prices, index=price_data.columns)

    def _calculate_daily_return(
        self,
        weights: pd.Series,
        price_data: pd.DataFrame,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
        universe_tickers: list,
    ) -> float:
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
                    if (ticker, "Close") in price_data.columns:
                        if current_date in price_data.index:
                            current_prices[ticker] = price_data.loc[current_date, (ticker, "Close")]
                        if previous_date in price_data.index:
                            previous_prices[ticker] = price_data.loc[
                                previous_date, (ticker, "Close")
                            ]
            else:
                # Single-level columns
                current_prices = {
                    ticker: price_data.loc[current_date, ticker]
                    for ticker in universe_tickers
                    if ticker in price_data.columns and current_date in price_data.index
                }
                previous_prices = {
                    ticker: price_data.loc[previous_date, ticker]
                    for ticker in universe_tickers
                    if ticker in price_data.columns and previous_date in price_data.index
                }

            # Calculate returns for each asset
            daily_return = 0.0
            for ticker in universe_tickers:
                if ticker in weights and ticker in current_prices and ticker in previous_prices:
                    prev_price = previous_prices[ticker]
                    curr_price = current_prices[ticker]
                    if (
                        prev_price is not None
                        and curr_price is not None
                        and prev_price != 0
                        and not pd.isna(prev_price)
                        and not pd.isna(curr_price)
                    ):
                        # Ensure we're working with numeric types using polymorphic extractor
                        from ..pandas_utils import extract_numeric_scalar

                        prev_price_float = extract_numeric_scalar(prev_price)
                        curr_price_float = extract_numeric_scalar(curr_price)

                        if prev_price_float is not None and curr_price_float is not None:
                            try:
                                asset_return = (
                                    curr_price_float - prev_price_float
                                ) / prev_price_float
                                daily_return += weights[ticker] * asset_return
                            except (ValueError, TypeError, ZeroDivisionError):
                                # Skip this ticker if we can't calculate return
                                continue
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
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "volatility": 0.0,
            }

        returns_series = pd.Series(daily_returns)

        # Basic metrics
        total_return = (1 + returns_series).prod() - 1
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (returns_series.mean() * 252) / volatility if volatility > 0 else 0.0

        # Sortino ratio (uses downside volatility)
        downside = returns_series[returns_series < 0]
        downside_vol = downside.std() * np.sqrt(252)
        sortino_ratio = (returns_series.mean() * 252) / downside_vol if downside_vol > 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "volatility": volatility,
            "num_periods": len(daily_returns),
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
            metrics={"total_return": 0.0, "sharpe_ratio": 0.0, "volatility": 0.0},
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            trades=[],
            final_weights={},
        )
