"""
Vulture whitelist for intentional public API/extension points in strategies.

This module references symbols reported by vulture when scanning only the
`src/portfolio_backtester/strategies` subtree, but which are part of the
intended public API consumed by other packages (e.g., interfaces, backtester
logic) or by external users. Importing and referencing these names marks them
as used for vulture without affecting runtime behavior.

Usage:
    vulture src/portfolio_backtester/strategies tools/vulture_whitelist.py
"""

from __future__ import annotations

# Imports must be at top (ruff E402)
from typing import Any

# Base strategy API imports
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy

# Meta strategy API imports
from portfolio_backtester.strategies._core.base.base.meta_strategy import BaseMetaStrategy

# Infra helpers imports
from portfolio_backtester.strategies._core.base.base.meta_reporting import (
    MetaStrategyReporter,
)
from portfolio_backtester.strategies._core.base.base.portfolio_value_tracker import (
    PortfolioValueTracker,
)
from portfolio_backtester.strategies._core.base.base.trade_aggregator import (
    TradeAggregator,
)
from portfolio_backtester.strategies._core.base.base.trade_interceptor import (
    MetaStrategyTradeInterceptor,
)

# Registry/validator/factory imports
from portfolio_backtester.strategies._core.registry.registry.solid_strategy_registry import (
    AutoDiscoveryStrategyRegistry,
    FileSystemStrategyDiscoveryEngine,
    ConcreteStrategyValidator,
)
from portfolio_backtester.strategies._core.strategy_factory_impl import StrategyFactory
from portfolio_backtester.strategies._core.registry.registry.strategy_validator import (
    StrategyValidator,
)

# Built-in example meta strategy import
from portfolio_backtester.strategies.builtins.meta.simple_meta_strategy import (
    SimpleMetaStrategy,
)

# EMA signal generator import
from portfolio_backtester.strategies.signal.ema_signal_generator import (
    EmaCrossoverSignalGenerator,
)


def _use(*_args: Any) -> None:  # runtime no-op to mark names as used
    return None


# Base strategy API
_use(
    BaseStrategy.risk_off_signal_generator_class,
    BaseStrategy.stop_loss_handler_class,
    BaseStrategy.take_profit_handler_class,
    BaseStrategy.supports_daily_signals,
    BaseStrategy.get_position_sizer_provider,
    BaseStrategy.get_universe_method_with_date,
    BaseStrategy.get_synthetic_data_requirements,
    BaseStrategy._apply_signal_strategy_stop_loss,
    BaseStrategy._apply_signal_strategy_take_profit,
    BaseStrategy.is_long_short_strategy,
    BaseStrategy.is_long_only_strategy,
    BaseStrategy.is_short_only_strategy,
    BaseStrategy.run_logic,
)


# Meta strategy API
_use(
    BaseMetaStrategy.get_trade_aggregator,
    BaseMetaStrategy.get_current_portfolio_value,
    BaseMetaStrategy.get_performance_metrics,
    BaseMetaStrategy.calculate_cumulative_returns,
    BaseMetaStrategy.get_comprehensive_performance_metrics,
    BaseMetaStrategy.get_meta_strategy_reporter,
    BaseMetaStrategy.get_portfolio_tracker,
    BaseMetaStrategy.generate_meta_strategy_report,
    BaseMetaStrategy.get_framework_trade_summary,
    BaseMetaStrategy.calculate_portfolio_value_with_market_data,
    BaseMetaStrategy.update_available_capital,
    BaseMetaStrategy.allocate_capital,
    BaseMetaStrategy.get_trade_interceptors,
    BaseMetaStrategy.reset_interceptor_state,
)


# Infra helpers inside strategies
_use(
    MetaStrategyReporter,
    PortfolioValueTracker.get_cash_balance,
    PortfolioValueTracker.get_positions,
    PortfolioValueTracker.reset,
    PortfolioValueTracker.export_value_history,
    PortfolioValueTracker.get_summary_statistics,
    TradeAggregator.get_trades_by_asset,
    TradeAggregator.get_trades_by_date_range,
    TradeAggregator.get_current_positions,
    TradeAggregator.get_position_at_date,
    TradeAggregator.update_portfolio_values_with_market_data,
    TradeAggregator.get_current_cash_balance,
    TradeAggregator.get_current_capital,
    TradeAggregator.get_total_return,
    TradeAggregator.get_summary_statistics,
    MetaStrategyTradeInterceptor.get_strategy_info,
)


# Registry/validator/factory extension points
_use(
    AutoDiscoveryStrategyRegistry.get_strategy_count,
    FileSystemStrategyDiscoveryEngine.get_discovery_paths,
    ConcreteStrategyValidator.get_base_strategy_types,
    StrategyFactory.get_registered_strategies,
    StrategyFactory.clear_registry,
    StrategyValidator,
    StrategyValidator.validate_strategy_safe,
)


# Built-in example meta strategy required methods
_use(SimpleMetaStrategy.allocate_capital)


# EMA signal generator API (interface-compliant methods)
_use(
    EmaCrossoverSignalGenerator.generate_signals_for_range,
    EmaCrossoverSignalGenerator.generate_signal_for_date,
    EmaCrossoverSignalGenerator.is_in_position,
    EmaCrossoverSignalGenerator.get_configuration,
)
