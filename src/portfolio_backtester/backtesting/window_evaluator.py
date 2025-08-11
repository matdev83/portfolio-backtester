"""
Window Evaluation Engine for Daily Strategy Evaluation.

This module provides the WindowEvaluator class that evaluates strategies
across all evaluation dates within a single WFO window, supporting both
daily and traditional monthly evaluation modes.
"""

from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd
from loguru import logger

from ..interfaces.daily_risk_monitor_interface import (
    DefaultDailyRiskMonitorFactory,
    IDailyRiskMonitorFactory,
)
from ..optimization.wfo_window import WFOWindow
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
        logger.debug("WindowEvaluator initialized")

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

        # Run the main backtester for the specific window
        backtest_result = self.backtester.backtest_strategy(
            strategy_config=strategy_config,
            monthly_data=full_monthly_data,
            daily_data=daily_data,  # This is already the full daily data
            rets_full=full_rets_daily,
            start_date=window.test_start,
            end_date=window.test_end,
        )

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
