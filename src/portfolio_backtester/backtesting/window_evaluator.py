"""
Window Evaluation Engine for Daily Strategy Evaluation.

This module provides the WindowEvaluator class that evaluates strategies
across all evaluation dates within a single WFO window, supporting both
daily and traditional monthly evaluation modes.
"""

import pandas as pd
from loguru import logger

from typing import TYPE_CHECKING, Dict, Optional, List, Tuple

from ..interfaces.daily_risk_monitor_interface import (
    DefaultDailyRiskMonitorFactory,
    IDailyRiskMonitorFactory,
)
from ..optimization.wfo_window import WFOWindow
from .position_tracker import PositionTracker
from .results import WindowResult

if TYPE_CHECKING:
    from .strategy_backtester import StrategyBacktester


# Re-export monitors so tests can patch them via this module path
from ..risk_management.daily_stop_loss_monitor import (  # noqa: F401
    DailyStopLossMonitor,
)
from ..risk_management.daily_take_profit_monitor import (  # noqa: F401
    DailyTakeProfitMonitor,
)

logger = logger.bind(name=__name__)


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
        backtester: "StrategyBacktester",
        data_cache: Optional[Dict] = None,
        risk_monitor_factory: Optional[IDailyRiskMonitorFactory] = None,
    ):
        """Initialize the window evaluator.

        Args:
            data_cache: Optional dictionary for caching historical data
            risk_monitor_factory: Factory for creating daily risk monitors (dependency injection)
        """
        self.backtester = backtester
        self.data_cache = data_cache or {}
        self.risk_monitor_factory = risk_monitor_factory or DefaultDailyRiskMonitorFactory()
        # Pre-computed data for window evaluation
        self._cached_universe_tickers: Optional[List[str]] = None
        self._cached_eval_dates: Optional[pd.DatetimeIndex] = None
        self._cached_daily_data_index: Optional[pd.Index] = None
        self._cached_daily_columns_is_multiindex: Optional[bool] = None
        logger.debug("WindowEvaluator initialized")

    def _prepare_window_evaluation(
        self,
        window: WFOWindow,
        daily_data: pd.DataFrame,
        universe_tickers: List[str],
    ) -> Tuple[pd.DatetimeIndex, bool, List[str]]:
        """Prepare invariant data for window evaluation.

        This method extracts and caches data that remains constant across multiple
        evaluations of the same window, reducing redundant work in hot paths.

        Args:
            window: The walk-forward window to evaluate
            daily_data: Daily price data
            universe_tickers: List of tickers in the universe

        Returns:
            Tuple containing:
                - Evaluation dates within the window
                - Whether daily_data.columns is a MultiIndex
                - List of universe tickers
        """
        # Check if we can reuse cached data
        if (
            self._cached_daily_data_index is not None
            and self._cached_universe_tickers is not None
            and self._cached_daily_columns_is_multiindex is not None
            and daily_data.index.equals(self._cached_daily_data_index)
            and universe_tickers == self._cached_universe_tickers
        ):
            # Reuse cached evaluation dates
            if self._cached_eval_dates is not None:
                eval_dates = self._cached_eval_dates
            else:
                eval_dates = window.get_evaluation_dates(pd.DatetimeIndex(daily_data.index))
                self._cached_eval_dates = eval_dates

            return (
                eval_dates,
                self._cached_daily_columns_is_multiindex,
                self._cached_universe_tickers,
            )

        # Compute and cache new data
        self._cached_daily_data_index = daily_data.index
        self._cached_universe_tickers = universe_tickers
        self._cached_daily_columns_is_multiindex = isinstance(daily_data.columns, pd.MultiIndex)
        eval_dates = window.get_evaluation_dates(pd.DatetimeIndex(daily_data.index))
        self._cached_eval_dates = eval_dates

        return (
            eval_dates,
            self._cached_daily_columns_is_multiindex,
            self._cached_universe_tickers,
        )

    def evaluate_window(
        self,
        window: WFOWindow,
        strategy,
        daily_data: pd.DataFrame,
        full_monthly_data: pd.DataFrame,
        full_rets_daily: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        universe_tickers: list,
        benchmark_ticker: str,
    ) -> WindowResult:
        """Evaluate strategy across a single walk-forward window by calling the main backtester."""
        logger.debug(
            f"Evaluating window {window.test_start.date()} to {window.test_end.date()} "
            f"by calling the centralized backtester."
        )

        # The strategy object passed here already has the correct parameters for the trial
        strategy_config = strategy.config

        # Initialize risk monitors if strategy has risk handlers
        stop_loss_monitor = None
        take_profit_monitor = None
        stop_loss_handler = None
        take_profit_handler = None

        # Check if strategy has stop loss handler
        if hasattr(strategy, "get_stop_loss_handler") and callable(strategy.get_stop_loss_handler):
            stop_loss_handler = strategy.get_stop_loss_handler()
            if stop_loss_handler:
                # Use the factory to create monitors (allows for easier testing)
                stop_loss_monitor = self.risk_monitor_factory.create_stop_loss_monitor()
                # logger.debug("Stop loss monitor initialized")

        # Check if strategy has take profit handler
        if hasattr(strategy, "get_take_profit_handler") and callable(
            strategy.get_take_profit_handler
        ):
            take_profit_handler = strategy.get_take_profit_handler()
            if take_profit_handler:
                # Use the factory to create monitors (allows for easier testing)
                take_profit_monitor = self.risk_monitor_factory.create_take_profit_monitor()
                # logger.debug("Take profit monitor initialized")

        # Set up position tracker for risk monitoring if needed
        position_tracker = None

        # Prepare window evaluation data once (hoisted out of inner loop)
        eval_dates, is_multiindex, universe = self._prepare_window_evaluation(
            window, daily_data, universe_tickers
        )

        if stop_loss_monitor or take_profit_monitor:
            position_tracker = PositionTracker()
            if len(eval_dates) > 0:
                logger.debug(f"Daily risk monitoring will run on {len(eval_dates)} dates")

        # Run the main backtester for the specific window
        backtest_result = self.backtester.backtest_strategy(
            strategy_config=strategy_config,
            monthly_data=full_monthly_data,
            daily_data=daily_data,  # This is already the full daily data
            rets_full=full_rets_daily,
            start_date=window.test_start,
            end_date=window.test_end,
        )

        # Perform daily risk monitoring if needed
        if position_tracker and (stop_loss_monitor or take_profit_monitor) and len(eval_dates) > 0:
            try:
                # Pre-extract test window dates to avoid repeated checks
                test_dates = eval_dates[
                    (eval_dates >= window.test_start) & (eval_dates <= window.test_end)
                ]

                # Process each evaluation date for risk monitoring
                for current_date in test_dates:
                    # Get current prices for risk evaluation
                    try:
                        current_prices = self._get_current_prices(
                            daily_data, current_date, is_multiindex
                        )
                    except Exception as e:
                        logger.error(f"Error getting current prices for {current_date.date()}: {e}")
                        continue

                    # Check for stop loss triggers
                    if stop_loss_monitor and stop_loss_handler:
                        try:
                            stop_loss_signals = stop_loss_monitor.check_positions_for_stop_loss(
                                current_date=current_date,
                                position_tracker=position_tracker,
                                current_prices=current_prices,
                                stop_loss_handler=stop_loss_handler,
                                historical_data=daily_data,
                            )

                            # Apply stop loss signals if any
                            if not stop_loss_signals.empty:
                                position_tracker.update_positions(stop_loss_signals, current_date)
                                logger.info(
                                    f"Applied stop loss liquidations on {current_date.date()}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error in stop loss monitoring on {current_date.date()}: {e}"
                            )

                    # Check for take profit triggers (after stop loss)
                    if take_profit_monitor and take_profit_handler:
                        try:
                            take_profit_signals = (
                                take_profit_monitor.check_positions_for_take_profit(
                                    current_date=current_date,
                                    position_tracker=position_tracker,
                                    current_prices=current_prices,
                                    take_profit_handler=take_profit_handler,
                                    historical_data=daily_data,
                                )
                            )

                            # Apply take profit signals if any
                            if not take_profit_signals.empty:
                                position_tracker.update_positions(take_profit_signals, current_date)
                                logger.info(
                                    f"Applied take profit liquidations on {current_date.date()}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error in take profit monitoring on {current_date.date()}: {e}"
                            )
            except Exception as e:
                logger.error(f"Error in daily risk monitoring: {e}")
                # Continue with evaluation even if risk monitoring fails

        # Adapt the BacktestResult to the WindowResult format
        return WindowResult(
            window_returns=backtest_result.returns,
            metrics=backtest_result.metrics,
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            trades=backtest_result.trade_history.to_dict("records"),
            final_weights=backtest_result.performance_stats.get("final_weights", {}),
        )

    def _get_current_prices(
        self,
        daily_data: pd.DataFrame,
        current_date: pd.Timestamp,
        is_multiindex: Optional[bool] = None,
    ) -> pd.Series:
        """Extract current prices for all assets on a specific date."""
        try:
            # Check if we have data for the current date
            if current_date not in daily_data.index:
                closest_date = daily_data.index[daily_data.index <= current_date][-1]
                logger.debug(
                    f"Using closest available date {closest_date.date()} for {current_date.date()}"
                )
                current_date = closest_date

            # Extract close prices for all assets
            if is_multiindex is None:
                is_multiindex = isinstance(daily_data.columns, pd.MultiIndex)

            if is_multiindex:
                # Handle MultiIndex columns (ticker, field)
                current_prices = {}
                for ticker in set(idx[0] for idx in daily_data.columns):
                    # Type-safe access to MultiIndex data
                    try:
                        # Try to get Close field first
                        current_prices[ticker] = daily_data.loc[current_date, (ticker, "Close")]
                    except (KeyError, ValueError):
                        try:
                            # Try lowercase 'close' if 'Close' doesn't exist
                            current_prices[ticker] = daily_data.loc[current_date, (ticker, "close")]
                        except (KeyError, ValueError):
                            # Skip this ticker if neither field exists
                            pass

                return pd.Series(current_prices)
            else:
                # Handle flat columns (assume it's price data directly)
                squeezed_data = daily_data.loc[current_date].squeeze()
                if isinstance(squeezed_data, pd.Series):
                    return squeezed_data
                else:
                    return pd.Series(squeezed_data)
        except Exception as e:
            logger.error(f"Error extracting current prices: {e}")
            # Return empty Series as fallback
            return pd.Series(dtype=float)
