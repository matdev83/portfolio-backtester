"""Base class for meta strategies that allocate capital across multiple sub-strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .trade_record import TradeRecord, TradeSide
from .trade_aggregator import TradeAggregator
from .trade_interceptor import MetaStrategyTradeInterceptor
from .portfolio_value_tracker import PortfolioValueTracker
from .meta_reporting import MetaStrategyReporter


# Custom exception for naming convention validation
class NamingConventionError(ValueError):
    """Raised when a class name doesn't follow naming conventions."""

    def __init__(self, message):
        super().__init__(message)


logger = logging.getLogger(__name__)


@dataclass
class SubStrategyAllocation:
    """Configuration for a sub-strategy allocation."""

    strategy_id: str
    strategy_class: str
    strategy_params: Dict[str, Any]
    weight: float
    strategy_instance: Optional[BaseStrategy] = None


class BaseMetaStrategy(BaseStrategy, ABC):
    """
    Abstract base class for meta strategies that allocate capital across multiple sub-strategies.

    === FOR CODING AGENTS: AUTO-DISCOVERY REQUIREMENTS ===
    🚨 IMPORTANT: Strategies are AUTOMATICALLY DISCOVERED! 🚨
    DO NOT manually register strategies - the system finds them automatically.

    TO CREATE A META STRATEGY:
    1. Class name MUST end with "MetaStrategy"
       ✅ Good: class MyAwesomeMetaStrategy(BaseMetaStrategy)
       ❌ Bad:  class MyAwesome(BaseMetaStrategy)

    2. File name MUST end with "_meta_strategy.py"
       ✅ Good: my_awesome_meta_strategy.py
       ❌ Bad:  my_awesome.py

    3. File MUST be in builtins/ or user/ category:
       ✅ Good (built-ins): src/portfolio_backtester/strategies/builtins/meta/my_awesome_meta_strategy.py
       ✅ Good (user):      src/portfolio_backtester/strategies/user/meta/my_awesome_meta_strategy.py
       ❌ Bad:  any other directory

    4. Class must be CONCRETE (implement all abstract methods)
       ✅ Good: Implement all required methods from BaseMetaStrategy
       ❌ Bad:  Leave abstract methods unimplemented

    5. Import and inherit correctly:
       ✅ Good: from ..base.meta_strategy import BaseMetaStrategy
                class MyAwesomeMetaStrategy(BaseMetaStrategy):

    The system will AUTOMATICALLY find and register your strategy!
    NEVER call register_strategy() or hardcode class names!

    Meta strategies are timing-agnostic capital allocators that:
    - Track available capital (initial + cumulative P&L from sub-strategies)
    - Allocate capital to sub-strategies based on percentage weights
    - Aggregate weighted signals from sub-strategies
    - Handle compounding of returns over time
    - Support recursive meta-strategy nesting
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "BaseMetaStrategy":
            return

        # Allow test classes and classes in test modules to bypass naming convention
        if (
            cls.__name__.startswith(("Test", "Mock", "Incomplete", "Illegal"))
            and not cls.__name__.startswith("IncorrectlyNamed")
            or cls.__module__
            and ("test" in cls.__module__ or "tests" in cls.__module__)
        ):
            return

        if not cls.__name__.endswith("MetaStrategy"):
            raise NamingConventionError(
                f"Meta-strategy class names must end with MetaStrategy: {cls.__name__}"
            )

    def __init__(
        self, strategy_params: Dict[str, Any], global_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(strategy_params)

        # Store global config for commission calculations
        self.global_config = global_config or {}
        self.strategy_name = self.global_config.get("strategy")

        # Initialize capital tracking
        self.initial_capital = strategy_params.get("initial_capital", 1000000.0)
        self.available_capital = self.initial_capital
        self.cumulative_pnl = 0.0

        # Get allocation mode from strategy params (defaults to reinvestment for compounding)
        self.allocation_mode = strategy_params.get("allocation_mode", "reinvestment")

        # Initialize sub-strategy allocations
        self.allocations: List[SubStrategyAllocation] = []
        self._initialize_allocations()

        # Validate allocations
        self.validate_allocations()

        # Cache for sub-strategy instances
        self._sub_strategy_cache: Dict[str, BaseStrategy] = {}

        # Trade tracking infrastructure
        self._trade_aggregator = TradeAggregator(self.initial_capital, self.allocation_mode)
        self._sub_strategy_capital_tracking: Dict[str, float] = {}

        # Trade interceptors for sub-strategies
        self._trade_interceptors: Dict[str, MetaStrategyTradeInterceptor] = {}

        # Portfolio value tracking and reporting
        self._portfolio_tracker = PortfolioValueTracker(self.initial_capital)
        self._reporter = MetaStrategyReporter(self._trade_aggregator)

        # Initialize capital tracking for each sub-strategy
        for allocation in self.allocations:
            self._sub_strategy_capital_tracking[allocation.strategy_id] = (
                self.available_capital * allocation.weight
            )

        # Performance tracking for each sub-strategy (legacy - will be replaced by trade aggregator)
        self._sub_strategy_performance: Dict[str, float] = {}

    def _initialize_allocations(self) -> None:
        """Initialize sub-strategy allocations from configuration."""
        allocations_config = self.strategy_params.get("allocations")
        if allocations_config is None and self.strategy_name:
            allocations_config = self.strategy_params.get(f"{self.strategy_name}.allocations")

        if allocations_config is None:
            allocations_config = []

        for allocation_config in allocations_config:
            allocation = SubStrategyAllocation(
                strategy_id=allocation_config["strategy_id"],
                strategy_class=allocation_config["strategy_class"],
                strategy_params=allocation_config["strategy_params"],
                weight=allocation_config["weight"],
            )
            self.allocations.append(allocation)

    def validate_allocations(self) -> None:
        """Validate that allocation weights sum to 1.0 and are within valid ranges."""
        if not self.allocations:
            raise ValueError("Meta strategy must have at least one sub-strategy allocation")

        total_weight = sum(allocation.weight for allocation in self.allocations)
        if not np.isclose(total_weight, 1.0, rtol=1e-6):
            raise ValueError(f"Allocation weights must sum to 1.0, got {total_weight}")

        min_allocation = self.strategy_params.get("min_allocation", 0.0)
        for allocation in self.allocations:
            if allocation.weight < 0:
                raise ValueError(f"Allocation weight cannot be negative: {allocation.weight}")
            if allocation.weight < min_allocation:
                raise ValueError(
                    f"Allocation weight {allocation.weight} below minimum {min_allocation}"
                )

    def get_sub_strategies(self) -> Dict[str, Union[BaseStrategy, MetaStrategyTradeInterceptor]]:
        """Return instantiated sub-strategies with trade interceptors, using cache to avoid recreation."""
        strategies: Dict[str, Union[BaseStrategy, MetaStrategyTradeInterceptor]] = {}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"get_sub_strategies cache keys before: {list(self._sub_strategy_cache.keys())}, interceptors: {list(self._trade_interceptors.keys())}"
            )

        for allocation in self.allocations:
            if allocation.strategy_id not in self._sub_strategy_cache:
                # Dynamically instantiate the strategy
                strategy_instance = self._create_strategy_instance(
                    allocation.strategy_class, allocation.strategy_params
                )
                self._sub_strategy_cache[allocation.strategy_id] = strategy_instance
                allocation.strategy_instance = strategy_instance

                # Create trade interceptor for this strategy
                self._create_trade_interceptor(allocation.strategy_id, strategy_instance)

            # Return the interceptor (which wraps the strategy)
            if allocation.strategy_id in self._trade_interceptors:
                strategies[allocation.strategy_id] = self._trade_interceptors[
                    allocation.strategy_id
                ]
            else:
                strategies[allocation.strategy_id] = self._sub_strategy_cache[
                    allocation.strategy_id
                ]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"get_sub_strategies cache keys after: {list(self._sub_strategy_cache.keys())}, interceptors: {list(self._trade_interceptors.keys())}"
            )

        return strategies

    def _create_trade_interceptor(self, strategy_id: str, strategy_instance: BaseStrategy) -> None:
        """
        Create a trade interceptor for a sub-strategy.

        Args:
            strategy_id: ID of the sub-strategy
            strategy_instance: The strategy instance to wrap
        """
        allocated_capital = self._sub_strategy_capital_tracking.get(strategy_id, 0.0)

        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=strategy_instance,
            strategy_id=strategy_id,
            allocated_capital=allocated_capital,
            trade_callback=self._on_sub_strategy_trade,
            global_config=self.global_config,
        )

        self._trade_interceptors[strategy_id] = interceptor
        logger.debug(
            f"Created trade interceptor for {strategy_id} with ${allocated_capital:,.2f} capital"
        )

    def _on_sub_strategy_trade(self, trade: TradeRecord) -> None:
        """
        Callback function called when a sub-strategy generates a trade.

        Args:
            trade: TradeRecord from the sub-strategy
        """
        # Track the trade in our aggregator
        self._trade_aggregator.track_sub_strategy_trade(trade)

        # Update portfolio tracker
        self._portfolio_tracker.update_from_trade(trade)

        # Update available capital based on current portfolio value from aggregator
        # This ensures that the meta strategy's available capital reflects P&L from closed trades
        current_portfolio_value = self._trade_aggregator.calculate_portfolio_value(trade.date)
        self.available_capital = current_portfolio_value
        self.cumulative_pnl = current_portfolio_value - self.initial_capital

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tracked intercepted trade: {trade}")
            logger.debug(
                f"Updated available capital: ${self.available_capital:.2f} (P&L: ${self.cumulative_pnl:.2f})"
            )

    def _create_strategy_instance(
        self, strategy_class: str, strategy_params: Dict[str, Any]
    ) -> BaseStrategy:
        """Create a strategy instance from class name and parameters."""
        # Import here to avoid circular imports
        from ...strategy_factory import StrategyFactory

        # Debug logging for strategy creation
        logger.debug(
            f"Creating sub-strategy instance: {strategy_class} with params: {strategy_params}"
        )

        # Process strategy_params to ensure they're in the right format for the sub-strategy
        # This is important because the params might be prefixed with the strategy name
        processed_params = {}
        for key, value in strategy_params.items():
            if "." in key:
                # Remove prefix if present
                param_name = key.split(".", 1)[1]
                processed_params[param_name] = value
            else:
                # Keep non-prefixed params as is
                processed_params[key] = value

        try:
            # Use StrategyFactory to instantiate sub-strategy. Pass global_config only when provided
            if self.global_config:
                created = StrategyFactory.create_strategy(
                    strategy_class, processed_params, self.global_config
                )
            else:
                created = StrategyFactory.create_strategy(strategy_class, processed_params)
            # Do not enforce runtime type assertion here to allow patching in tests
            return created
        except ValueError as e:
            # Provide more helpful error message for strategy resolution
            logger.error(f"Failed to create sub-strategy {strategy_class}: {e}")

            # Check if this is a known alias
            from ...registry import get_strategy_registry

            registry = get_strategy_registry()

            # Try to find a matching strategy by name
            available_strategies = list(registry.get_all_strategies().keys())
            logger.debug(f"Available strategies: {available_strategies}")

            # Find closest match
            import difflib

            matches = difflib.get_close_matches(
                strategy_class, available_strategies, n=3, cutoff=0.6
            )

            if matches:
                suggestion = f"Did you mean: {', '.join(matches)}?"
                logger.error(f"Strategy '{strategy_class}' not found. {suggestion}")
                raise ValueError(f"Strategy '{strategy_class}' not found. {suggestion}") from e

            raise

    def calculate_sub_strategy_capital(self) -> Dict[str, float]:
        """Calculate capital allocation for each sub-strategy based on allocation mode."""
        capital_allocations = {}

        # Determine base capital based on allocation mode
        if self.allocation_mode in ["reinvestment", "compound"]:
            # Use current available capital for compounding
            base_capital = self.available_capital
        else:  # fixed_fractional or fixed_capital
            # Use initial capital (no compounding)
            base_capital = self.initial_capital

        for allocation in self.allocations:
            allocated_capital = base_capital * allocation.weight
            capital_allocations[allocation.strategy_id] = allocated_capital

            # Update tracking
            self._sub_strategy_capital_tracking[allocation.strategy_id] = allocated_capital

            # Update interceptor capital if it exists
            if allocation.strategy_id in self._trade_interceptors:
                self._trade_interceptors[allocation.strategy_id].update_allocated_capital(
                    allocated_capital
                )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Capital allocation (mode={self.allocation_mode}): base=${base_capital:.2f}, allocations={capital_allocations}"
            )

        return capital_allocations

    def track_sub_strategy_trade(
        self,
        strategy_id: str,
        date: pd.Timestamp,
        asset: str,
        signal_weight: float,
        price: float,
        transaction_cost_bps: float = 0.0,
    ) -> Optional[TradeRecord]:
        """
        Track a trade from a sub-strategy by converting signal weight to actual trade.

        Args:
            strategy_id: ID of the sub-strategy generating the trade
            date: Date of the trade
            asset: Asset being traded
            signal_weight: Signal weight from sub-strategy (e.g., 0.1 for 10% allocation)
            price: Price per share
            transaction_cost_bps: Transaction cost in basis points

        Returns:
            TradeRecord if a trade was executed, None if no trade
        """
        if abs(signal_weight) < 1e-8:
            return None  # No trade for zero weight

        # Get allocated capital for this sub-strategy
        allocated_capital = self._sub_strategy_capital_tracking.get(strategy_id, 0.0)
        if allocated_capital <= 0:
            logger.warning(f"No capital allocated to strategy {strategy_id}")
            return None

        # Calculate trade value and quantity
        trade_value = abs(signal_weight * allocated_capital)
        quantity = trade_value / price if price > 0 else 0

        if quantity < 1e-8:
            return None  # Trade too small

        # Determine trade side
        side = TradeSide.BUY if signal_weight > 0 else TradeSide.SELL

        trade = TradeRecord(
            date=date,
            asset=asset,
            quantity=quantity if side == TradeSide.BUY else -quantity,
            price=price,
            side=side,
            strategy_id=strategy_id,
            allocated_capital=allocated_capital,
            trade_value=trade_value,
        )

        # Track the trade
        self._trade_aggregator.track_sub_strategy_trade(trade)

        logger.debug(f"Tracked trade for {strategy_id}: {trade}")
        return trade

    def get_aggregated_trades(self) -> List[TradeRecord]:
        """Get all trades from all sub-strategies."""
        return self._trade_aggregator.get_aggregated_trades()

    def get_trade_aggregator(self) -> TradeAggregator:
        """Get the trade aggregator instance."""
        return self._trade_aggregator

    def get_current_portfolio_value(self) -> float:
        """Get current portfolio value based on actual trades."""
        return self._trade_aggregator.calculate_portfolio_value(pd.Timestamp.now())

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics based on actual trade execution."""
        return self._trade_aggregator.calculate_weighted_performance()

    def get_strategy_attribution(self) -> Dict[str, Dict[str, float]]:
        """Get performance attribution by sub-strategy."""
        return self._trade_aggregator.get_strategy_attribution()

    def calculate_returns(
        self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate returns based on aggregated trade history.

        Args:
            start_date: Start date for return calculation
            end_date: End date for return calculation

        Returns:
            Series of daily returns
        """
        portfolio_timeline = self._trade_aggregator.get_portfolio_timeline()

        if portfolio_timeline.empty:
            return pd.Series(dtype=float)

        # Filter by date range if provided
        if start_date is not None:
            portfolio_timeline = portfolio_timeline[portfolio_timeline.index >= start_date]
        if end_date is not None:
            portfolio_timeline = portfolio_timeline[portfolio_timeline.index <= end_date]

        return portfolio_timeline["returns"].dropna()

    def calculate_cumulative_returns(
        self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate cumulative returns based on aggregated trade history.

        Args:
            start_date: Start date for return calculation
            end_date: End date for return calculation

        Returns:
            Series of cumulative returns
        """
        portfolio_timeline = self._trade_aggregator.get_portfolio_timeline()

        if portfolio_timeline.empty:
            return pd.Series(dtype=float)

        # Filter by date range if provided
        if start_date is not None:
            portfolio_timeline = portfolio_timeline[portfolio_timeline.index >= start_date]
        if end_date is not None:
            portfolio_timeline = portfolio_timeline[portfolio_timeline.index <= end_date]

        return portfolio_timeline["cumulative_return"]

    def calculate_drawdown(
        self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate drawdown based on aggregated trade history.

        Args:
            start_date: Start date for drawdown calculation
            end_date: End date for drawdown calculation

        Returns:
            Series of drawdown values
        """
        portfolio_timeline = self._trade_aggregator.get_portfolio_timeline()

        if portfolio_timeline.empty:
            return pd.Series(dtype=float)

        # Filter by date range if provided
        if start_date is not None:
            portfolio_timeline = portfolio_timeline[portfolio_timeline.index >= start_date]
        if end_date is not None:
            portfolio_timeline = portfolio_timeline[portfolio_timeline.index <= end_date]

        # Calculate running maximum (peak)
        portfolio_values = portfolio_timeline["portfolio_value"]
        running_max = portfolio_values.expanding().max()

        # Calculate drawdown as percentage from peak
        drawdown = (portfolio_values - running_max) / running_max

        return drawdown

    def calculate_max_drawdown(
        self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Calculate maximum drawdown based on aggregated trade history.

        Args:
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Maximum drawdown as a negative percentage
        """
        drawdown_series = self.calculate_drawdown(start_date, end_date)

        if drawdown_series.empty:
            return 0.0

        return float(drawdown_series.min())

    def calculate_sharpe_ratio(
        self,
        risk_free_rate: float = 0.0,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> float:
        """
        Calculate Sharpe ratio based on aggregated trade history.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Sharpe ratio
        """
        returns = self.calculate_returns(start_date, end_date)

        if returns.empty or len(returns) < 2:
            return 0.0

        # Annualize returns (assuming daily returns)
        mean_return = returns.mean() * 252  # 252 trading days per year
        return_std = returns.std() * np.sqrt(252)

        if return_std == 0:
            return 0.0

        return float((mean_return - risk_free_rate) / return_std)

    def calculate_sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> float:
        """
        Calculate Sortino ratio based on aggregated trade history.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Sortino ratio
        """
        returns = self.calculate_returns(start_date, end_date)

        if returns.empty or len(returns) < 2:
            return 0.0

        # Annualize returns
        mean_return = returns.mean() * 252

        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float("inf") if mean_return > risk_free_rate else 0.0

        downside_std = negative_returns.std() * np.sqrt(252)

        if downside_std == 0:
            return 0.0

        return float((mean_return - risk_free_rate) / downside_std)

    def calculate_calmar_ratio(
        self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Calculate Calmar ratio based on aggregated trade history.

        Args:
            start_date: Start date for calculation
            end_date: End date for calculation

        Returns:
            Calmar ratio
        """
        returns = self.calculate_returns(start_date, end_date)
        max_drawdown = self.calculate_max_drawdown(start_date, end_date)

        if returns.empty or max_drawdown == 0:
            return 0.0

        # Annualize returns
        mean_return = returns.mean() * 252

        return mean_return / abs(max_drawdown)

    def get_comprehensive_performance_metrics(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Get comprehensive performance metrics based on aggregated trade history.

        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculation

        Returns:
            Dictionary containing comprehensive performance metrics
        """
        # Get basic metrics from trade aggregator
        basic_metrics = self._trade_aggregator.calculate_weighted_performance()

        # Calculate advanced metrics
        returns = self.calculate_returns(start_date, end_date)

        advanced_metrics = {
            "sharpe_ratio": self.calculate_sharpe_ratio(risk_free_rate, start_date, end_date),
            "sortino_ratio": self.calculate_sortino_ratio(risk_free_rate, start_date, end_date),
            "calmar_ratio": self.calculate_calmar_ratio(start_date, end_date),
            "max_drawdown": self.calculate_max_drawdown(start_date, end_date),
            "volatility": returns.std() * np.sqrt(252) if not returns.empty else 0.0,
            "annualized_return": returns.mean() * 252 if not returns.empty else 0.0,
            "total_trading_days": len(returns) if not returns.empty else 0,
        }

        # Combine metrics
        return {**basic_metrics, **advanced_metrics}

    def get_meta_strategy_reporter(self) -> MetaStrategyReporter:
        """Get the meta strategy reporter instance."""
        return self._reporter

    def get_portfolio_tracker(self) -> PortfolioValueTracker:
        """Get the portfolio value tracker instance."""
        return self._portfolio_tracker

    def generate_meta_strategy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive meta strategy report.

        Returns:
            Dictionary containing detailed meta strategy analysis
        """
        return self._reporter.generate_comprehensive_report()

    def export_trades_for_framework(self) -> pd.DataFrame:
        """
        Export trades in framework-compatible format.

        This method returns all sub-strategy trades as if they were executed
        by the meta strategy itself, which is the correct approach for
        performance reporting.

        Returns:
            DataFrame of trades formatted for framework consumption
        """
        return self._reporter.export_trades_for_framework()

    def get_framework_trade_summary(self) -> Dict[str, Any]:
        """
        Get trade summary in framework-expected format.

        Returns:
            Dictionary with trade statistics for framework reporting
        """
        return self._reporter.generate_trade_summary_for_framework()

    def calculate_portfolio_value_with_market_data(
        self, date: pd.Timestamp, market_data: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio value using current market prices.

        Args:
            date: Date to calculate value for
            market_data: Market data with current prices

        Returns:
            Portfolio value using market prices
        """
        return self._portfolio_tracker.calculate_portfolio_value(date, market_data)

    def update_available_capital(self, sub_strategy_returns: Dict[str, float]) -> None:
        """
        Update available capital based on sub-strategy returns.

        Args:
            sub_strategy_returns: Dictionary mapping strategy_id to return (as decimal, e.g., 0.05 for 5%)
        """
        total_pnl = 0.0

        for allocation in self.allocations:
            strategy_id = allocation.strategy_id
            if strategy_id in sub_strategy_returns:
                strategy_return = sub_strategy_returns[strategy_id]
                allocated_capital = self.available_capital * allocation.weight
                strategy_pnl = allocated_capital * strategy_return
                total_pnl += strategy_pnl

                # Track individual strategy performance
                self._sub_strategy_performance[strategy_id] = strategy_return

        # Update cumulative P&L and available capital
        self.cumulative_pnl += total_pnl
        self.available_capital = self.initial_capital + self.cumulative_pnl

        logger.debug(
            f"Updated available capital: {self.available_capital:.2f} (P&L: {self.cumulative_pnl:.2f})"
        )

    def aggregate_signals(
        self, sub_strategy_signals: Dict[str, pd.DataFrame], current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Aggregate signals from sub-strategies using capital-weighted combination.

        Args:
            sub_strategy_signals: Dictionary mapping strategy_id to signal DataFrame
            current_date: Current date for signal generation

        Returns:
            Aggregated signal DataFrame
        """
        if not sub_strategy_signals:
            return pd.DataFrame()

        # Get all unique assets from all sub-strategies
        all_assets: Set[str] = set()
        for signals in sub_strategy_signals.values():
            if not signals.empty:
                all_assets.update(signals.columns)

        if not all_assets:
            return pd.DataFrame()

        all_assets_list = sorted(list(all_assets))

        # Initialize aggregated signals
        aggregated_signals = pd.Series(0.0, index=all_assets_list)

        # Calculate capital allocations based on allocation mode
        self.calculate_sub_strategy_capital()

        # Determine total capital for weighting based on allocation mode
        if self.allocation_mode in ["reinvestment", "compound"]:
            pass
        else:  # fixed_fractional or fixed_capital
            pass

        # Aggregate signals weighted by capital allocation
        for allocation in self.allocations:
            strategy_id = allocation.strategy_id

            if strategy_id not in sub_strategy_signals:
                continue

            strategy_signals = sub_strategy_signals[strategy_id]
            if strategy_signals.empty or current_date not in strategy_signals.index:
                continue

            # Get signals for current date
            current_signals = strategy_signals.loc[current_date]
            if isinstance(current_signals, pd.DataFrame):
                current_signals = current_signals.iloc[0]  # Take first row if DataFrame

            # Weight by capital allocation (respects allocation mode)
            weight = allocation.weight
            for asset in current_signals.index:
                if asset in all_assets_list:
                    aggregated_signals[asset] += weight * current_signals[asset]

        # Return as DataFrame with single row for current_date
        return pd.DataFrame([aggregated_signals], index=[current_date], columns=all_assets_list)

    @abstractmethod
    def allocate_capital(self) -> Dict[str, float]:
        """
        Abstract method for determining capital allocation logic.

        Returns:
            Dictionary mapping strategy_id to allocation weight (must sum to 1.0)
        """
        pass

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate signals from all sub-strategies with automatic trade tracking.

        This method coordinates signal generation across all sub-strategies.
        Trade interceptors automatically capture and track trades from signal changes.
        """
        logger.debug(f"Meta strategy generating signals for date: {current_date}")

        # Get sub-strategies (with interceptors)
        sub_strategies = self.get_sub_strategies()
        logger.debug(f"Got {len(sub_strategies)} sub-strategies: {list(sub_strategies.keys())}")

        # Update capital allocations (this updates interceptor capital too)
        capital_allocations = self.calculate_sub_strategy_capital()
        logger.debug(f"Capital allocations: {capital_allocations}")

        # Generate signals from each sub-strategy
        # Trade interceptors will automatically capture trades
        sub_strategy_signals = {}

        for strategy_id, strategy in sub_strategies.items():
            try:
                logger.debug(f"Generating signals for sub-strategy {strategy_id}")
                signals = strategy.generate_signals(
                    all_historical_data=all_historical_data,
                    benchmark_historical_data=benchmark_historical_data,
                    non_universe_historical_data=non_universe_historical_data,
                    current_date=current_date,
                    start_date=start_date,
                    end_date=end_date,
                )

                if signals is None or signals.empty:
                    logger.warning(f"Sub-strategy {strategy_id} returned empty signals")
                    continue

                logger.debug(f"Sub-strategy {strategy_id} signals: {signals}")
                sub_strategy_signals[strategy_id] = signals

            except Exception as e:
                logger.error(
                    f"Error generating signals for sub-strategy {strategy_id}: {e}", exc_info=True
                )
                # Continue with other strategies
                continue

        # Check if we have any signals
        if not sub_strategy_signals:
            logger.warning("No sub-strategy signals generated")
            # Return empty DataFrame with proper index and columns
            return pd.DataFrame(index=[current_date])

        # Aggregate signals for framework compatibility
        aggregated_signals = self.aggregate_signals(sub_strategy_signals, current_date)
        logger.debug(f"Aggregated signals: {aggregated_signals}")

        # Enforce trade direction constraints - this will raise an exception if violated
        aggregated_signals = self._enforce_trade_direction_constraints(aggregated_signals)

        return aggregated_signals

    def get_trade_interceptors(self) -> Dict[str, MetaStrategyTradeInterceptor]:
        """Get all trade interceptors."""
        return self._trade_interceptors.copy()

    def reset_interceptor_state(self) -> None:
        """Reset state for all trade interceptors."""
        for interceptor in self._trade_interceptors.values():
            interceptor.reset_state()
        logger.debug("Reset state for all trade interceptors")

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods based on sub-strategies.

        Returns the maximum requirement among all sub-strategies.
        """
        sub_strategies = self.get_sub_strategies()

        max_periods = 12  # Default minimum
        for strategy in sub_strategies.values():
            strategy_periods = strategy.get_minimum_required_periods()
            max_periods = max(max_periods, strategy_periods)

        return max_periods

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Names of hyper-parameters this meta strategy understands."""
        return {
            "initial_capital": {
                "type": "float",
                "default": 1000000.0,
                "min": 10000.0,
                "max": 100000000.0,
                "step": 10000.0,
            },
            "min_allocation": {
                "type": "float",
                "default": 0.05,
                "min": 0.01,
                "max": 0.5,
                "step": 0.01,
            },
            "rebalance_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.01,
                "max": 0.2,
                "step": 0.01,
            },
        }

    def get_universe(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get the combined universe from all sub-strategies.

        Returns the union of all sub-strategy universes.
        """
        sub_strategies = self.get_sub_strategies()

        combined_universe = set()
        for strategy in sub_strategies.values():
            try:
                strategy_universe = strategy.get_universe(global_config)
                for ticker, weight in strategy_universe:
                    combined_universe.add(ticker)
            except Exception as e:
                logger.warning(f"Error getting universe from sub-strategy: {e}")
                continue

        # Return as list of (ticker, 1.0) tuples
        return [(ticker, 1.0) for ticker in sorted(combined_universe)]

    def get_non_universe_data_requirements(self) -> List[str]:
        """
        Get combined non-universe data requirements from all sub-strategies.
        """
        sub_strategies = self.get_sub_strategies()

        combined_requirements: Set[str] = set()
        for strategy in sub_strategies.values():
            try:
                requirements = strategy.get_non_universe_data_requirements()
                combined_requirements.update(requirements)
            except Exception as e:
                logger.warning(f"Error getting non-universe requirements from sub-strategy: {e}")
                continue

        return list(combined_requirements)
