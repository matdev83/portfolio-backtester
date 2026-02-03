"""
Pure backtesting engine that contains only backtesting logic without optimization methods.

This module implements the StrategyBacktester class which is responsible for executing
trading strategies and calculating performance metrics, completely separated from
optimization concerns.
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, TYPE_CHECKING, Union, cast, Mapping

if TYPE_CHECKING:
    from ..canonical_config import CanonicalScenarioConfig

from .results import BacktestResult, WindowResult
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
from ..strategies._core.registry import get_strategy_registry
from ..interfaces import create_cache_manager
from ..backtester_logic.strategy_logic import generate_signals, size_positions
from ..backtester_logic.portfolio_logic import calculate_portfolio_returns
from ..backtester_logic.data_manager import prepare_scenario_data
from ..backtester_logic.strategy_overlays import apply_wfo_scaling_and_kill_switch
from ..reporting.performance_metrics import calculate_metrics
from ..optimization.wfo_window import WFOWindow

logger = logging.getLogger(__name__)


def _build_wfo_test_mask(overlay_diagnostics: Dict[str, Any], index: pd.Index) -> pd.Series:
    windows = overlay_diagnostics.get("windows", []) if overlay_diagnostics else []
    if not windows:
        return pd.Series(False, index=index)

    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)

    mask_vals = pd.Series(False, index=index)
    for window in windows:
        test_start = window.get("test_start")
        test_end = window.get("test_end")
        if not test_start or not test_end:
            continue
        try:
            start_ts = pd.Timestamp(test_start)
            end_ts = pd.Timestamp(test_end)
        except Exception:
            continue
        mask_vals |= (idx >= start_ts) & (idx <= end_ts)

    return mask_vals


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
        strategy_config: Union[Dict[str, Any], CanonicalScenarioConfig],
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
            strategy_config: Strategy configuration including name and parameters (raw dict or canonical object)
            monthly_data: Monthly price data
            daily_data: Daily OHLC price data
            rets_full: Daily returns data

        Returns:
            BacktestResult: Complete backtest results with all performance data
        """
        from ..canonical_config import CanonicalScenarioConfig
        from ..scenario_normalizer import ScenarioNormalizer

        # Ensure we are working with a canonical config
        if not isinstance(strategy_config, CanonicalScenarioConfig):
            logger.warning(
                "ACCIDENTAL BYPASS: Raw strategy_config dictionary passed to StrategyBacktester.backtest_strategy. "
                "All scenarios should be canonicalized at the boundary. "
                "Scenario: %s",
                strategy_config.get("name", "unnamed"),
            )
            normalizer = ScenarioNormalizer()
            canonical_config = normalizer.normalize(
                scenario=strategy_config, global_config=self.global_config
            )
        else:
            canonical_config = strategy_config

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Running backtest for strategy: {canonical_config.name}")

        # Filter data based on the window if start_date and end_date are provided
        if start_date and end_date:
            daily_data = daily_data.loc[start_date:end_date]
            monthly_data = monthly_data.loc[start_date:end_date]
            rets_full = rets_full.loc[start_date:end_date]

        # Get strategy instance
        strategy = self._get_strategy(
            canonical_config.strategy,
            canonical_config.strategy_params,
            canonical_config,
        )

        # Determine universe
        # Use strategy's universe provider
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

        benchmark_ticker = canonical_config.benchmark_ticker or self.global_config.get(
            "benchmark", "SPY"
        )

        # Prepare data
        monthly_closes, rets_daily = prepare_scenario_data(daily_data, self.data_cache)

        # Generate signals
        signals = generate_signals(
            strategy,
            canonical_config,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            lambda: False,  # No timeout for pure backtesting
        )

        # Size positions
        sized_signals = size_positions(
            signals,
            canonical_config,
            monthly_closes,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            strategy,
        )

        overlay_config = canonical_config.get("risk_overlay_config")
        overlay_diagnostics: Optional[Dict[str, Any]] = None
        if isinstance(overlay_config, Mapping) and overlay_config.get("type") == "drift_regime_wfo":
            asset_returns = rets_daily.reindex(columns=universe_tickers)
            sized_signals, overlay_diagnostics = apply_wfo_scaling_and_kill_switch(
                sized_signals, asset_returns, overlay_config
            )

        # Calculate portfolio returns (optionally with trade tracking)
        portfolio_returns, trade_tracker = calculate_portfolio_returns(
            sized_signals,
            canonical_config,
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

        metrics_returns = portfolio_returns
        benchmark_returns_for_metrics = benchmark_returns
        if overlay_diagnostics and isinstance(overlay_config, Mapping):
            if overlay_config.get("metrics_window") == "wfo_test":
                test_mask = _build_wfo_test_mask(overlay_diagnostics, portfolio_returns.index)
                if test_mask.any():
                    metrics_returns = portfolio_returns.loc[test_mask]
                    benchmark_returns_for_metrics = benchmark_returns.loc[test_mask]
                else:
                    logger.warning("WFO test window mask empty; using full-period returns.")

        # Calculate performance metrics with trade statistics
        metrics = calculate_metrics(
            metrics_returns,
            benchmark_returns_for_metrics,
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
        strategy_config: Union[Dict[str, Any], CanonicalScenarioConfig],
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
        from ..canonical_config import CanonicalScenarioConfig

        train_start, train_end, test_start, test_end = (
            window.train_start,
            window.train_end,
            window.test_start,
            window.test_end,
        )

        # Normalize timezone awareness across window boundaries and data indexes.
        # WFO windows are often generated from tz-naive monthly indexes, while
        # MDMP-sourced daily data can be tz-aware. Pandas slicing raises if they
        # don't match.
        def _drop_tz(ts: pd.Timestamp) -> pd.Timestamp:
            if isinstance(ts, pd.Timestamp) and ts.tz is not None:
                return ts.tz_localize(None)
            return ts

        train_start = _drop_tz(train_start)
        train_end = _drop_tz(train_end)
        test_start = _drop_tz(test_start)
        test_end = _drop_tz(test_end)

        if isinstance(monthly_data.index, pd.DatetimeIndex) and monthly_data.index.tz is not None:
            monthly_data = monthly_data.copy()
            monthly_index = cast(pd.DatetimeIndex, monthly_data.index)
            monthly_data.index = monthly_index.tz_localize(None)

        if isinstance(daily_data.index, pd.DatetimeIndex) and daily_data.index.tz is not None:
            daily_data = daily_data.copy()
            daily_index = cast(pd.DatetimeIndex, daily_data.index)
            daily_data.index = daily_index.tz_localize(None)

        if isinstance(rets_full.index, pd.DatetimeIndex) and rets_full.index.tz is not None:
            rets_full = rets_full.copy()
            rets_index = cast(pd.DatetimeIndex, rets_full.index)
            rets_full.index = rets_index.tz_localize(None)

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
        benchmark_ticker = (
            strategy_config.benchmark_ticker
            if isinstance(strategy_config, CanonicalScenarioConfig)
            else strategy_config.get("benchmark_ticker", self.global_config.get("benchmark", "SPY"))
        )
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
        self,
        strategy_spec,
        params: Mapping[str, Any],
        strategy_config: Union[Dict[str, Any], CanonicalScenarioConfig],
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

        # Prefer passing the canonical config if available to support new features
        # BaseStrategy.__init__ handles both dict and CanonicalScenarioConfig
        instance = strategy_class(strategy_config)
        instance.config = strategy_config  # Attach for backward compatibility

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
        strategy_config: Union[Dict[str, Any], CanonicalScenarioConfig],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_daily: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Series]:
        """Run a scenario for a specific window (extracted from core.py run_scenario).

        Args:
            strategy_config: Strategy configuration (raw dict or canonical object)
            monthly_data: Monthly price data for the window
            daily_data: Daily OHLC data for the window
            rets_daily: Daily returns data

        Returns:
            Portfolio returns series or None if failed
        """
        from ..canonical_config import CanonicalScenarioConfig
        from ..scenario_normalizer import ScenarioNormalizer

        # Ensure we are working with a canonical config
        if not isinstance(strategy_config, CanonicalScenarioConfig):
            logger.warning(
                "ACCIDENTAL BYPASS: Raw strategy_config dictionary passed to StrategyBacktester._run_scenario_for_window. "
                "All scenarios should be canonicalized at the boundary. "
                "Scenario: %s",
                strategy_config.get("name", "unnamed"),
            )
            normalizer = ScenarioNormalizer()
            canonical_config = normalizer.normalize(
                scenario=strategy_config, global_config=self.global_config
            )
        else:
            canonical_config = strategy_config

        strategy = self._get_strategy(
            canonical_config.strategy,
            canonical_config.strategy_params,
            canonical_config,
        )

        # Use strategy's universe provider
        universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]

        # Filter missing tickers using daily_data (more reliable for presence)
        from ..interfaces.ohlc_normalizer import OHLCNormalizerFactory

        normalizer_ohlc = OHLCNormalizerFactory.create_normalizer(daily_data)
        available_tickers = normalizer_ohlc.get_available_tickers(daily_data)

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
            canonical_config,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            lambda: False,  # No timeout
        )

        # Size positions
        sized_signals = size_positions(
            signals,
            canonical_config,
            monthly_closes,
            daily_data,
            universe_tickers,
            benchmark_ticker,
            strategy,
        )

        # Calculate portfolio returns (no trade tracking for window evaluation)
        result = calculate_portfolio_returns(
            sized_signals,
            canonical_config,
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
        """Create a history of trades from signal weights using vectorized operations."""
        if sized_signals.empty:
            return pd.DataFrame()

        # PERFORMANCE: Avoid nested loops over large universes (e.g. R2K)
        # Use stack() to get non-zero positions efficiently
        stacked_signals = cast(pd.Series, sized_signals.stack())
        
        # Robustness: ensure numeric values before taking abs()
        numeric_signals = pd.to_numeric(stacked_signals, errors="coerce")
        non_zero_mask = numeric_signals.notna() & (numeric_signals.abs() > 1e-6)
        non_zero_signals = stacked_signals.loc[non_zero_mask]

        if non_zero_signals.empty:
            return pd.DataFrame()

        # Extract close prices if MultiIndex (standard for MDMP)
        if isinstance(daily_data.columns, pd.MultiIndex):
            field_level = "Field" if "Field" in daily_data.columns.names else -1
            if "Close" in daily_data.columns.get_level_values(field_level):
                prices_df = daily_data.xs("Close", level=field_level, axis=1)
            else:
                # Fallback to the first available field for each ticker
                first_field = daily_data.columns.get_level_values(field_level)[0]
                prices_df = daily_data.xs(first_field, level=field_level, axis=1)
        else:
            prices_df = daily_data

        if isinstance(prices_df, pd.Series):
            prices_df = prices_df.to_frame()

        # Align prices to signals and stack
        # This ensures we have a price for every position date/ticker
        # Limit to relevant tickers to speed up reindex
        relevant_tickers = sized_signals.columns
        aligned_prices = prices_df.reindex(index=sized_signals.index, columns=relevant_tickers)
        stacked_prices = aligned_prices.stack()

        # Create the result DataFrame
        trade_history = pd.DataFrame(non_zero_signals, columns=["position"])
        trade_history.reset_index(inplace=True)
        # Signals stacked index names might be [None, None] or [DateName, TickerName]
        # We force standard names for the output
        trade_history.columns = ["date", "ticker", "position"]

        # Add prices via mapping from the stacked price series (fast lookup)
        trade_history["price"] = non_zero_signals.index.map(stacked_prices)

        return trade_history

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
