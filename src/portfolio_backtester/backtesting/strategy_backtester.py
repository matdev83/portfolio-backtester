"""
Pure backtesting engine that contains only backtesting logic without optimization methods.

This module implements the StrategyBacktester class which is responsible for executing
trading strategies and calculating performance metrics, completely separated from
optimization concerns.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from .results import BacktestResult, WindowResult
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
from ..strategies._core.registry import get_strategy_registry
from ..interfaces import create_cache_manager
from ..backtester_logic.strategy_logic import generate_signals, size_positions
from ..backtester_logic.portfolio_logic import calculate_portfolio_returns
from ..backtester_logic.data_manager import prepare_scenario_data
from ..reporting.performance_metrics import calculate_metrics
from ..optimization.wfo_window import WFOWindow

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """Pure backtesting engine that contains only backtesting logic.

    This class is responsible for executing trading strategies and calculating
    performance metrics. It has no optimization-related imports or dependencies
    and can run completely independently of any optimization framework.

    Attributes:
        global_config: Global configuration dictionary
        data_source: Data source for fetching price data
        data_cache: Cache for data preprocessing
    """

    def __init__(self, global_config: Dict[str, Any], data_source: Any):
        """Initialize the pure backtesting engine.

        Args:
            global_config: Global configuration dictionary
            data_source: Data source for fetching price data
        """
        self.global_config = global_config
        self.data_source = data_source

        # Initialize strategy registry (SOLID-compliant, supports aliases)
        self._registry = get_strategy_registry()
        self.strategy: Optional[BaseStrategy] = None

        # Initialize data cache using DIP
        self.data_cache = create_cache_manager()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("StrategyBacktester initialized without optimization dependencies")

    def backtest_strategy(
        self,
        strategy_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        track_trades: bool = True,
    ) -> BacktestResult:
        """Execute a complete backtest for a strategy configuration.

        This method runs a full backtest for the given strategy configuration
        and returns structured results with metrics, trades, and P&L data.

        Args:
            strategy_config: Strategy configuration including name and parameters
            monthly_data: Monthly price data
            daily_data: Daily OHLC price data
            rets_full: Daily returns data

        Returns:
            BacktestResult: Complete backtest results with all performance data
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Running backtest for strategy: {strategy_config.get('name', 'Unknown')}")

        # Filter data based on the window if start_date and end_date are provided
        if start_date and end_date:
            daily_data = daily_data.loc[start_date:end_date]
            monthly_data = monthly_data.loc[start_date:end_date]
            rets_full = rets_full.loc[start_date:end_date]

        # Get strategy instance
        strategy = self._get_strategy(
            strategy_config["strategy"],
            strategy_config["strategy_params"],
            strategy_config,
        )

        # Determine universe
        if "universe" in strategy_config:
            try:
                from ..interfaces.ticker_collector import TickerCollectorFactory

                collector = TickerCollectorFactory.create_collector(strategy_config["universe"])
                universe_tickers = collector.collect_tickers(strategy_config["universe"])
            except Exception as e:
                logger.error(f"Failed to collect universe tickers: {e}")
                universe_tickers = []
        elif "universe_config" in strategy_config:
            try:
                from ..interfaces.ticker_collector import TickerCollectorFactory

                collector = TickerCollectorFactory.create_collector(
                    strategy_config["universe_config"]
                )
                universe_tickers = collector.collect_tickers(strategy_config["universe_config"])
            except Exception as e:
                logger.error(f"Failed to collect universe config tickers: {e}")
                universe_tickers = []
        else:
            # Only use strategy.get_universe as last resort, not global config universe
            universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]

        # Filter out missing tickers
        missing_cols = [t for t in universe_tickers if t not in monthly_data.columns]
        if missing_cols:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Tickers {missing_cols} not found in price data; they will be skipped."
                )
            universe_tickers = [t for t in universe_tickers if t not in missing_cols]

        if not universe_tickers:
            logger.warning("No universe tickers remain after filtering. Returning empty results.")
            return self._create_empty_backtest_result()

        benchmark_ticker = self.global_config["benchmark"]

        # Prepare data
        monthly_closes, rets_daily = prepare_scenario_data(daily_data, self.data_cache)

        # Generate signals
        signals = generate_signals(
            strategy,
            strategy_config,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            lambda: False,  # No timeout for pure backtesting
        )

        # Size positions
        sized_signals = size_positions(
            signals,
            strategy_config,
            monthly_closes,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            strategy,
        )

        # Calculate portfolio returns (optionally with trade tracking)
        portfolio_returns, trade_tracker = calculate_portfolio_returns(
            sized_signals,
            strategy_config,
            daily_data,
            rets_daily,
            universe_tickers,
            self.global_config,
            track_trades=track_trades,
            strategy=strategy,
        )

        trade_stats = trade_tracker.get_trade_statistics() if trade_tracker else None

        if portfolio_returns is None or portfolio_returns.empty:
            logger.warning("No portfolio returns generated. Returning empty results.")
            return self._create_empty_backtest_result()

        # Calculate benchmark returns for metrics
        benchmark_data = (
            daily_data[benchmark_ticker] if benchmark_ticker in daily_data.columns else None
        )
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change(fill_method=None).fillna(0)
        else:
            benchmark_returns = pd.Series(0.0, index=portfolio_returns.index)

        # Calculate performance metrics with trade statistics
        metrics = calculate_metrics(
            portfolio_returns,
            benchmark_returns,
            benchmark_ticker,
            trade_stats=trade_stats,
        )

        # Create trade history from trade tracker when tracking is enabled
        if trade_tracker:
            completed_trades = trade_tracker.trade_lifecycle_manager.get_completed_trades()
            trade_history = pd.DataFrame([t.__dict__ for t in completed_trades])
        else:
            trade_history = self._create_trade_history(sized_signals, daily_data)

        # Create performance stats
        performance_stats = self._create_performance_stats(portfolio_returns, metrics)

        # Create charts data
        charts_data = self._create_charts_data(portfolio_returns, benchmark_returns)

        # Store trade statistics in the result
        trade_stats = trade_tracker.get_trade_statistics() if trade_tracker else None

        return BacktestResult(
            returns=portfolio_returns,
            metrics=metrics,
            trade_history=trade_history,
            performance_stats=performance_stats,
            charts_data=charts_data,
            trade_stats=trade_stats,
        )

    def evaluate_window(
        self,
        strategy_config: Dict[str, Any],
        window: WFOWindow,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
    ) -> WindowResult:
        """Evaluate a strategy configuration on a single walk-forward window.

        This method evaluates a parameter set on a single walk-forward window
        and returns structured results without any optimization-specific logic.

        Args:
            strategy_config: Strategy configuration including parameters
            window: A WFOWindow object defining the train/test periods.
            monthly_data: Monthly price data
            daily_data: Daily OHLC price data
            rets_full: Daily returns data

        Returns:
            WindowResult: Results for this specific window
        """
        train_start, train_end, test_start, test_end = (
            window.train_start,
            window.train_end,
            window.test_start,
            window.test_end,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Evaluating window: {train_start} to {test_end}")

        # Slice data for this window
        monthly_slice = monthly_data.loc[train_start:train_end]
        daily_slice = daily_data.loc[train_start:test_end]

        # Get cached window returns
        cached_window_returns = self.data_cache.get_window_returns_by_dates(
            daily_slice, train_start, test_end
        )

        # Run scenario for this window
        window_returns = self._run_scenario_for_window(
            strategy_config, monthly_slice, daily_slice, cached_window_returns
        )

        if window_returns is None or window_returns.empty:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No returns generated for window {train_start}-{test_end}")
            return self._create_empty_window_result(train_start, train_end, test_start, test_end)

        # Extract test period returns
        test_returns = window_returns.loc[test_start:test_end]
        if test_returns.empty:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Test returns empty for window {train_start}-{test_end}")
            return self._create_empty_window_result(train_start, train_end, test_start, test_end)

        # Calculate benchmark returns for this window
        benchmark_ticker = self.global_config["benchmark"]
        benchmark_data = daily_slice[benchmark_ticker].loc[test_start:test_end]
        benchmark_returns = benchmark_data.pct_change(fill_method=None).fillna(0)

        # Calculate metrics for this window
        metrics = calculate_metrics(test_returns, benchmark_returns, benchmark_ticker)

        return WindowResult(
            window_returns=test_returns,
            metrics=metrics,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    def _get_strategy(
        self, strategy_spec, params: Dict[str, Any], strategy_config: Dict[str, Any]
    ) -> BaseStrategy:
        """Get a strategy instance by name and parameters.

        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters

        Returns:
            BaseStrategy: Configured strategy instance

        Raises:
            ValueError: If strategy name is not supported
            TypeError: If strategy class doesn't return BaseStrategy instance
        """
        if isinstance(strategy_spec, dict):
            strategy_name = (
                strategy_spec.get("strategy")
                or strategy_spec.get("name")
                or strategy_spec.get("type")
            )
        else:
            strategy_name = strategy_spec

        if strategy_name is None:
            raise ValueError("Strategy name cannot be None")

        # Resolve strategy class via registry to support unit-test mocking and DIP
        strategy_class = self._registry.get_strategy_class(str(strategy_name))
        if strategy_class is None:
            # Fallback for test-defined strategies declared after registry discovery
            try:

                def _all_subclasses(cls: type) -> set[type]:
                    subs = set()
                    for sub in cls.__subclasses__():
                        subs.add(sub)
                        subs.update(_all_subclasses(sub))
                    return subs

                dynamic_map = {cls.__name__: cls for cls in _all_subclasses(BaseStrategy)}
                strategy_class = dynamic_map.get(str(strategy_name))
            except Exception:
                strategy_class = None

            if strategy_class is None:
                raise ValueError(f"Unsupported strategy: {strategy_name}")

        # Diagnostic: log resolved class details before instantiation
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(
                    "Resolved strategy_class: %s type=%s module=%s",
                    strategy_class,
                    type(strategy_class),
                    getattr(strategy_class, "__module__", None),
                )
            except Exception:
                pass

        instance = strategy_class(params)
        instance.config = strategy_config  # Attach the full config

        if not isinstance(instance, BaseStrategy):
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    logger.debug(
                        "Instance type=%s module=%s base_expected=%s base_module=%s",
                        type(instance),
                        getattr(type(instance), "__module__", None),
                        BaseStrategy,
                        getattr(BaseStrategy, "__module__", None),
                    )
                except Exception:
                    pass
            raise TypeError("Strategy factory did not return a BaseStrategy instance")

        self.strategy = instance
        return instance

    def _run_scenario_for_window(
        self,
        strategy_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_daily: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        """Run a scenario for a specific window (extracted from core.py run_scenario).

        Args:
            strategy_config: Strategy configuration
            monthly_data: Monthly price data for the window
            daily_data: Daily OHLC data for the window
            rets_daily: Daily returns data

        Returns:
            Portfolio returns series or None if failed
        """
        strategy = self._get_strategy(
            strategy_config["strategy"],
            strategy_config["strategy_params"],
            strategy_config,
        )

        if "universe" in strategy_config:
            try:
                from ..interfaces.ticker_collector import TickerCollectorFactory

                collector = TickerCollectorFactory.create_collector(strategy_config["universe"])
                universe_tickers = collector.collect_tickers(strategy_config["universe"])
            except Exception as e:
                logger.error(f"Failed to collect universe tickers: {e}")
                universe_tickers = []
        elif "universe_config" in strategy_config:
            try:
                from ..interfaces.ticker_collector import TickerCollectorFactory

                collector = TickerCollectorFactory.create_collector(
                    strategy_config["universe_config"]
                )
                universe_tickers = collector.collect_tickers(strategy_config["universe_config"])
            except Exception as e:
                logger.error(f"Failed to collect universe config tickers: {e}")
                universe_tickers = []
        else:
            universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]

        # Filter missing tickers using daily_data (more reliable for presence)
        from ..interfaces.ohlc_normalizer import OHLCNormalizerFactory

        normalizer = OHLCNormalizerFactory.create_normalizer(daily_data)
        available_tickers = normalizer.get_available_tickers(daily_data)

        missing_cols = [t for t in universe_tickers if t not in available_tickers]
        if missing_cols:
            universe_tickers = [t for t in universe_tickers if t not in missing_cols]

        if not universe_tickers:
            logger.warning(
                "No available universe tickers for this window after filtering missing data. Skipping window."
            )
            return None

        benchmark_ticker = self.global_config["benchmark"]

        # Prepare scenario data if not provided
        if rets_daily is None:
            monthly_closes, rets_daily = prepare_scenario_data(daily_data, self.data_cache)
        else:
            monthly_closes = monthly_data

        # Generate signals
        signals = generate_signals(
            strategy,
            strategy_config,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            lambda: False,  # No timeout
        )

        # Size positions
        sized_signals = size_positions(
            signals,
            strategy_config,
            monthly_closes,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            strategy,
        )

        # Calculate portfolio returns (no trade tracking for window evaluation)
        result = calculate_portfolio_returns(
            sized_signals,
            strategy_config,
            daily_data,
            rets_daily,
            universe_tickers,
            self.global_config,
            track_trades=False,
            strategy=strategy,
        )

        # Handle both old and new return formats
        if isinstance(result, tuple):
            portfolio_returns, _ = result
        else:
            portfolio_returns = result

        return portfolio_returns  # type: ignore[no-any-return]

    def _create_empty_backtest_result(self) -> BacktestResult:
        """Create an empty BacktestResult for error cases."""
        return BacktestResult(
            returns=pd.Series(dtype=float),
            metrics={},
            trade_history=pd.DataFrame(),
            performance_stats={},
            charts_data={},
        )

    def _create_empty_window_result(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> WindowResult:
        """Create an empty WindowResult for error cases."""
        return WindowResult(
            window_returns=pd.Series(dtype=float),
            metrics={},
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    def _create_trade_history(
        self, sized_signals: pd.DataFrame, daily_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Create trade history from sized signals (simplified implementation)."""
        # This is a simplified implementation - in a full implementation,
        # this would track actual trades, entry/exit prices, etc.
        trades = []

        for date in sized_signals.index:
            for ticker in sized_signals.columns:
                position = sized_signals.loc[date, ticker]
                try:
                    # Convert position to float for comparison
                    if pd.notna(position) and isinstance(position, (int, float, np.number)):
                        position_value = float(position)
                    else:
                        position_value = 0.0
                    if abs(position_value) > 1e-6:  # Non-zero position
                        trades.append(
                            {
                                "date": date,
                                "ticker": ticker,
                                "position": position_value,
                                "price": (
                                    daily_data.loc[date, ticker]
                                    if ticker in daily_data.columns
                                    else np.nan
                                ),
                            }
                        )
                except (TypeError, ValueError):
                    # Skip positions that can't be converted to float
                    continue

        return pd.DataFrame(trades)

    def _create_performance_stats(
        self, returns: pd.Series, metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create additional performance statistics."""
        # Calculate total return safely
        if returns.empty:
            total_return = 0.0
        else:
            try:
                prod_result = (1 + returns).prod()
                if pd.notna(prod_result) and isinstance(prod_result, (int, float, np.number)):
                    total_return = float(prod_result) - 1.0
                else:
                    total_return = 0.0
            except (TypeError, ValueError):
                total_return = 0.0

        return {
            "total_return": total_return,
            "annualized_return": returns.mean() * 252,
            "annualized_volatility": returns.std() * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(returns),
            "num_observations": len(returns),
            "start_date": returns.index.min() if not returns.empty else None,
            "end_date": returns.index.max() if not returns.empty else None,
        }

    def _create_charts_data(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Create data for performance charts."""
        if portfolio_returns.empty:
            return {}

        # Calculate cumulative returns
        portfolio_cumulative: pd.Series = (1 + portfolio_returns).cumprod()
        benchmark_cumulative: pd.Series = (1 + benchmark_returns).cumprod()

        return {
            "portfolio_cumulative": portfolio_cumulative,
            "benchmark_cumulative": benchmark_cumulative,
            "drawdown": self._calculate_drawdown_series(portfolio_returns),
            "rolling_sharpe": self._calculate_rolling_sharpe(portfolio_returns),
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if returns.empty:
            return 0.0

        cumulative: pd.Series = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    def _calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series from returns."""
        if returns.empty:
            return pd.Series(dtype=float)

        cumulative: pd.Series = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        if returns.empty or len(returns) < window:
            return pd.Series(dtype=float, index=returns.index)

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        # Annualize
        rolling_sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252))
        return rolling_sharpe.fillna(0)  # type: ignore[no-any-return]
